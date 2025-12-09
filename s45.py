# FILE: eval/evaluate_semantic_planner_sota.py
# (v18.0 - CORRECTED Goal Image Generation & Control Loop)
# 
# CHANGELOG v18.0:
#   - [CRITICAL FIX] Goal image now rendered as END STATE (object at goal, robot retracted)
#     This matches training data where goal_image = all_images[-1] = final episode frame
#   - [FIX] Gripper convention: model outputs prob 1.0=CLOSED, 0.0=OPEN (matches training)
#   - [FIX] Removed unnecessary IK delta computation - model predicts absolute target poses
#   - [IMPROVEMENT] Added debug logging for first few steps

import logging
import sys
import time
import csv
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional

import cv2
import hydra
import mujoco
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# --- Project Imports ---
ROOT = Path(__file__).resolve().parents[0]  # Script is in project root
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from envs.panda_env import PandaEnv
from train.train_semantic_planner import SemanticPlannerLightningModule
from utils.ik_solver import IKSolver

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [SOTA-EVAL-v18] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("evaluation_log.txt", mode='w')
    ]
)
log = logging.getLogger("SOTA_Eval")

# ==============================================================================
# 1. STABILITY MODULE: TEMPORAL ENSEMBLING WITH CORRECT GRIPPER HANDLING
# ==============================================================================

class TemporalEnsembler:
    """
    Motion Smoothing with CORRECTED Gripper Logic.
    
    Training Convention:
    - gt_gripper_intent = 1.0 means CLOSED
    - gt_gripper_intent = 0.0 means OPEN
    
    Model outputs logits, sigmoid(logit) > 0.5 means CLOSED
    """
    
    def __init__(self, alpha_pose: float = 0.7, alpha_grip: float = 0.5):
        self.alpha_pose = alpha_pose
        self.alpha_grip = alpha_grip
        
        self.smoothed_pose: Optional[np.ndarray] = None
        self.smoothed_grip_prob: float = 0.0
        
        # Schmitt Trigger State
        self.gripper_is_closed = False
        
        # Thresholds for PROBABILITY (sigmoid output)
        # prob > 0.5 means model wants CLOSED
        self.CLOSE_THRESHOLD = 0.50   # Close when prob > 0.50
        self.OPEN_THRESHOLD = 0.50    # Open when prob < 0.50

    def reset(self):
        self.smoothed_pose = None
        self.smoothed_grip_prob = 0.0
        self.gripper_is_closed = False

    def update(self, target_pose: np.ndarray, grip_logit: float) -> Tuple[np.ndarray, float]:
        """
        Args:
            target_pose: [x, y, z, qx, qy, qz, qw] from model
            grip_logit: Raw logit from model
            
        Returns:
            smoothed_pose, grip_cmd (-1.0=close, +1.0=open)
        """
        # Convert logit to probability
        grip_prob = 1.0 / (1.0 + np.exp(-np.clip(grip_logit, -50, 50)))
        
        if self.smoothed_pose is None:
            self.smoothed_pose = target_pose.copy()
            self.smoothed_grip_prob = grip_prob
        else:
            # Position EMA
            pos_new = target_pose[:3]
            pos_old = self.smoothed_pose[:3]
            pos_smooth = (self.alpha_pose * pos_new) + ((1 - self.alpha_pose) * pos_old)
            
            # Quaternion NLERP with sign flip handling
            quat_new = target_pose[3:].copy()
            quat_old = self.smoothed_pose[3:]
            
            if np.dot(quat_new, quat_old) < 0.0:
                quat_new = -quat_new
                
            quat_smooth = (self.alpha_pose * quat_new) + ((1 - self.alpha_pose) * quat_old)
            quat_norm = np.linalg.norm(quat_smooth)
            if quat_norm > 1e-8:
                quat_smooth = quat_smooth / quat_norm
            
            self.smoothed_pose = np.concatenate([pos_smooth, quat_smooth])
            
            # Gripper EMA
            self.smoothed_grip_prob = (self.alpha_grip * grip_prob) + \
                                       ((1 - self.alpha_grip) * self.smoothed_grip_prob)

        # Decision: prob > 0.5 means CLOSE, prob < 0.5 means OPEN
        if self.smoothed_grip_prob > self.CLOSE_THRESHOLD:
            self.gripper_is_closed = True
        elif self.smoothed_grip_prob < self.OPEN_THRESHOLD:
            self.gripper_is_closed = False
        
        # Environment command: -1.0 = Close, +1.0 = Open
        grip_cmd = -1.0 if self.gripper_is_closed else 1.0
        
        return self.smoothed_pose.copy(), grip_cmd


# ==============================================================================
# 2. GOAL IMAGE GENERATION (MATCHING TRAINING DATA)
# ==============================================================================

@contextmanager
def render_goal_image_like_training(env: PandaEnv, goal_pos_world: np.ndarray):
    """
    Renders a goal image that matches training data convention:
    - Object is at the GOAL position
    - Robot is in RETRACTED/HOME position (task complete state)
    - Gripper is OPEN
    
    This simulates what the LAST FRAME of a successful episode looks like.
    """
    saved_qpos = env.data.qpos.copy()
    saved_qvel = env.data.qvel.copy()
    saved_ctrl = env.data.ctrl.copy()
    
    try:
        # 1. Robot in "retracted" home position (task done pose)
        # This matches the end of a successful pick-and-place
        home_qpos = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.8, 0.785])
        env.data.qpos[:7] = home_qpos
        env.data.qpos[7:9] = 0.04  # Open gripper
        
        # 2. Object at goal position
        obj_jnt_adr = env.model.jnt_qposadr[env.object_joint_id]
        current_quat = env.data.qpos[obj_jnt_adr+3 : obj_jnt_adr+7].copy()
        
        env.data.qpos[obj_jnt_adr : obj_jnt_adr+3] = goal_pos_world
        env.data.qpos[obj_jnt_adr+3 : obj_jnt_adr+7] = current_quat
        
        # 3. Zero velocities
        env.data.qvel[:] = 0.0
        
        # 4. Forward kinematics
        mujoco.mj_forward(env.model, env.data)
        yield
        
    finally:
        # Restore
        env.data.qpos[:] = saved_qpos
        env.data.qvel[:] = saved_qvel
        env.data.ctrl[:] = saved_ctrl
        mujoco.mj_forward(env.model, env.data)


# ==============================================================================
# 3. MAIN EVALUATOR
# ==============================================================================

class AWSPEvaluator:
    """
    Semantic Planner Evaluator v18.0 - Corrected Goal Image
    """
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- A. Load Model ---
        log.info(f"Loading Checkpoint: {cfg.checkpoint_path}")
        self.pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
            cfg.checkpoint_path,
            map_location=self.device
        )
        self.model = self.pl_module.model.eval().to(self.device)
        self.chunk_size = self.pl_module.cfg.model.get("chunk_size", 10)
        log.info(f"Model Loaded. Action Chunk Size: {self.chunk_size}")

        # --- B. Initialize Environment ---
        xml_path = cfg.env.get("xml_path", "envs/panda_pick_place.xml")
        self.env = PandaEnv(
            xml_path=xml_path,
            control_mode='delta',
            render_mode="rgb_array",
            action_scaling_factor=0.5
        )
        
        self.ik_solver = IKSolver(urdf_path="urdf/panda_mujoco_kinematics.urdf")
        
        # Controller timing
        SIM_SUBSTEPS = 20
        self.effective_dt = self.env.model.opt.timestep * SIM_SUBSTEPS
        self.max_dq = self.env.ACTION_SCALING_FACTOR / self.effective_dt
        
        log.info(f"Control Config: dt={self.effective_dt:.4f}s, max_dq={self.max_dq:.2f}")
        
        # --- C. Visual Transforms (MUST match training) ---
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # --- D. Components ---
        self.ensembler = TemporalEnsembler(alpha_pose=0.7, alpha_grip=0.5)
        self.prev_img_buffer = None

    def run(self):
        """Main Execution Loop"""
        out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        video_path = out_dir / "sota_eval.mp4"
        summary_csv_path = out_dir / "eval_summary.csv"
        telemetry_csv_path = out_dir / "eval_telemetry.csv"
        
        # Video setup
        dummy = self.env.reset()[0]
        h, w, _ = self.env.render().shape
        video_writer = cv2.VideoWriter(
            str(video_path), 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            30, (w, h)
        )
        
        # Telemetry CSV
        telemetry_file = open(telemetry_csv_path, 'w', newline='')
        telemetry_writer = csv.writer(telemetry_file)
        telemetry_writer.writerow([
            "episode_id", "step", "success_state", "dist_to_goal",
            "phase_pred", "latency_ms",
            "ee_x", "ee_y", "ee_z", 
            "target_x", "target_y", "target_z",
            "raw_grip_logit", "grip_prob", "smooth_grip_prob", "grip_cmd",
            "obj_x", "obj_y", "obj_z", "is_grasped"
        ])
        
        log.info(f"Starting SOTA Evaluation for {self.cfg.num_episodes} episodes...")
        
        summary_results = []
        total_success = 0
        
        try:
            for ep_idx in tqdm(range(self.cfg.num_episodes), desc="Evaluator"):
                seed = self.cfg.seed + ep_idx
                ep_result = self.run_episode(ep_idx, seed, video_writer, telemetry_writer)
                summary_results.append(ep_result)
                if ep_result['success']:
                    total_success += 1
                    
        finally:
            video_writer.release()
            telemetry_file.close()
            self.env.close()
            
            success_rate = (total_success / self.cfg.num_episodes) * 100
            avg_steps = np.mean([r['steps'] for r in summary_results]) if summary_results else 0
            
            log.info("=" * 60)
            log.info(f"FINAL EVALUATION REPORT")
            log.info(f"Episodes: {self.cfg.num_episodes}")
            log.info(f"Success Rate: {success_rate:.2f}%")
            log.info(f"Avg Steps: {avg_steps:.1f}")
            log.info("=" * 60)
            
            with open(summary_csv_path, 'w', newline='') as f:
                if summary_results:
                    w = csv.DictWriter(f, fieldnames=summary_results[0].keys())
                    w.writeheader()
                    w.writerows(summary_results)

    def run_episode(self, ep_idx: int, seed: int, 
                    video_writer, telemetry_writer) -> Dict[str, Any]:
        """Run a single evaluation episode"""
        
        # 1. Reset
        obs, _ = self.env.reset(seed=seed)
        self.ensembler.reset()
        self.ik_solver.reset_controller_state()
        
        # 2. Generate Goal Image (MATCHING TRAINING CONVENTION)
        # Training: goal_image = final frame of episode = object at goal, robot retracted
        with render_goal_image_like_training(self.env, obs['goal_pos_world']):
            goal_img_raw = self.env.render()
        
        goal_tensor = self.transform(Image.fromarray(goal_img_raw)).unsqueeze(0).to(self.device)
        
        # 3. Initialize History
        curr_img_pil = Image.fromarray(obs['image_primary'])
        self.prev_img_buffer = self.transform(curr_img_pil).unsqueeze(0).to(self.device)
        
        success = False
        final_error = 99.9
        grasp_achieved = False
        
        for step in range(self.cfg.max_steps):
            t0 = time.time()
            
            # --- A. Data Preparation ---
            curr_img_pil = Image.fromarray(obs['image_primary'])
            curr_tensor = self.transform(curr_img_pil).unsqueeze(0).to(self.device)
            
            proprio_tensor = torch.from_numpy(obs['proprio']).float().unsqueeze(0).to(self.device)
            
            batch = {
                'prev_image': self.prev_img_buffer,
                'curr_image': curr_tensor,
                'goal_image': goal_tensor,
                'curr_proprio': proprio_tensor
            }
            self.prev_img_buffer = curr_tensor.clone()
            
            # --- B. Inference ---
            with torch.no_grad():
                outputs = self.model(batch)
            
            inference_time_ms = (time.time() - t0) * 1000
            
            # Model outputs: pose_chunk (B, K, 7), gripper_chunk (B, K, 1)
            chunk_pose = outputs['pose_chunk'][0].cpu().numpy()
            chunk_grip = outputs['gripper_chunk'][0].cpu().numpy()
            
            # Use first prediction (receding horizon)
            target_pose_raw = chunk_pose[0]
            target_grip_logit = chunk_grip[0].item()
            
            phase_logits = outputs['phase_logits'][0].cpu().numpy()
            predicted_phase = np.argmax(phase_logits)
            
            # Debug logging for first step
            if step == 0 and ep_idx == 0:
                log.info(f"Step 0 Debug: target_pose={target_pose_raw[:3]}, grip_logit={target_grip_logit:.3f}")
            
            # --- C. Temporal Ensembling ---
            target_pose, grip_cmd = self.ensembler.update(target_pose_raw, target_grip_logit)
            
            # --- D. IK Control ---
            # Model outputs pose in [x,y,z, qx,qy,qz,qw] format
            # IK solver expects [x,y,z, w,x,y,z] (MuJoCo quaternion convention)
            pos = target_pose[:3]
            quat_xyzw = target_pose[3:]
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            
            target_ee_pose_mujoco = np.concatenate([pos, quat_wxyz])
            
            try:
                delta_joints = self.ik_solver.compute_delta_action(
                    target_ee_pose=target_ee_pose_mujoco,
                    model=self.env.model,
                    data=self.env.data,
                    ee_site_id=self.env.ee_site_id,
                    joint_qpos_indices=np.arange(7),
                    effective_dt=self.effective_dt,
                    max_dq=self.max_dq
                )
            except Exception as e:
                log.warning(f"IK failed at step {step}: {e}")
                delta_joints = np.zeros(7)

            # --- E. Step Environment ---
            full_action = np.concatenate([delta_joints, [grip_cmd]])
            obs, _, terminated, truncated, _ = self.env.step(full_action)
            
            # --- F. Metrics ---
            ee_pos_now = obs['ee_pose_world'][:3]
            obj_pos = obs['object_pos_world']
            goal_pos = obs['goal_pos_world']
            dist_to_goal = np.linalg.norm(obj_pos - goal_pos)
            is_grasped = obs['is_grasped'][0] > 0.5
            
            if is_grasped:
                grasp_achieved = True
            
            if dist_to_goal < 0.05:
                success = True
            
            grip_prob = 1.0 / (1.0 + np.exp(-np.clip(target_grip_logit, -50, 50)))
            
            # --- G. Telemetry ---
            telemetry_writer.writerow([
                ep_idx, step, int(success), f"{dist_to_goal:.4f}",
                predicted_phase, f"{inference_time_ms:.1f}",
                f"{ee_pos_now[0]:.3f}", f"{ee_pos_now[1]:.3f}", f"{ee_pos_now[2]:.3f}",
                f"{pos[0]:.3f}", f"{pos[1]:.3f}", f"{pos[2]:.3f}",
                f"{target_grip_logit:.3f}", f"{grip_prob:.3f}",
                f"{self.ensembler.smoothed_grip_prob:.3f}", f"{grip_cmd:.1f}",
                f"{obj_pos[0]:.3f}", f"{obj_pos[1]:.3f}", f"{obj_pos[2]:.3f}",
                int(is_grasped)
            ])
            
            # --- H. Video ---
            frame = self.env.render()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            color = (0, 255, 0) if success else (0, 0, 255)
            grip_color = (0, 255, 0) if grip_cmd < 0 else (255, 255, 0)
            
            cv2.putText(frame_bgr, f"Ep:{ep_idx} Ph:{predicted_phase} Err:{dist_to_goal:.3f}m", 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame_bgr, f"Grip: {'CLOSE' if grip_cmd < 0 else 'OPEN'} ({grip_prob:.2f})", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, grip_color, 1)
            cv2.putText(frame_bgr, f"Logit: {target_grip_logit:.2f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            video_writer.write(frame_bgr)
            
            if success or terminated or truncated:
                final_error = dist_to_goal
                break
        
        log.info(f"Episode {ep_idx}: {'SUCCESS' if success else 'FAIL'}, "
                 f"Error: {final_error:.3f}m, Grasp: {grasp_achieved}")
                
        return {
            "episode_id": ep_idx,
            "success": success,
            "steps": step + 1,
            "final_error": final_error,
            "grasp_achieved": grasp_achieved,
            "seed": seed
        }


@hydra.main(version_base=None, config_path="./configs", config_name="evaluate_semantic_planner_config")
def main(cfg: DictConfig):
    if "checkpoint_path" not in cfg:
        raise ValueError("Must provide 'checkpoint_path' in config")
    
    evaluator = AWSPEvaluator(cfg)
    evaluator.run()


if __name__ == "__main__":
    main()