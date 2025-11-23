# FILE: evaluate_semantic_planner.py
# (v14.0 - Full Autopilot / The "Lobotomy" Patch)

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import csv
import cv2
import hydra
import mujoco
import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# Project Imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from envs.panda_env import PandaEnv
from train.train_semantic_planner import SemanticPlannerLightningModule
from utils.ik_solver import IKSolver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Strategist-Eval] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("Strategist_Eval")

class LightweightSmoother:
    def __init__(self, alpha_pos=0.8, alpha_grip=0.5):
        self.alpha_pos = alpha_pos
        self.alpha_grip = alpha_grip
        self.smooth_pose = None
        self.smooth_grip_logit = 0.0
        self.gripper_closed = False

    def reset(self):
        self.smooth_pose = None
        self.smooth_grip_logit = 0.0
        self.gripper_closed = False

    def update(self, raw_pose: np.ndarray, raw_logit: float) -> Tuple[np.ndarray, float]:
        if self.smooth_pose is None:
            self.smooth_pose = raw_pose
            self.smooth_grip_logit = raw_logit
        else:
            self.smooth_pose = (self.alpha_pos * raw_pose) + ((1 - self.alpha_pos) * self.smooth_pose)
            self.smooth_grip_logit = (self.alpha_grip * raw_logit) + ((1 - self.alpha_grip) * self.smooth_grip_logit)
        
        # Hysteresis for stability
        if not self.gripper_closed and self.smooth_grip_logit > 0.5:
            self.gripper_closed = True
        elif self.gripper_closed and self.smooth_grip_logit < -0.5:
            self.gripper_closed = False
            
        # -1.0 is Closed, 1.0 is Open in PandaEnv
        gripper_cmd = -1.0 if self.gripper_closed else 1.0
        return self.smooth_pose, gripper_cmd

@contextmanager
def render_robust_virtual_goal(env, target_pos_world):
    """Temporarily moves the object to the goal position to render the 'Goal Image'."""
    saved_qpos = env.data.qpos.copy()
    saved_qvel = env.data.qvel.copy()
    saved_ctrl = env.data.ctrl.copy()
    try:
        # Reset robot to home to clear view
        home_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        env.data.qpos[:7] = home_qpos
        env.data.qpos[7:] = 0.04 # Open grippers
        
        # Move object to goal
        obj_jnt_adr = env.model.jnt_qposadr[env.object_joint_id]
        curr_obj_quat = env.data.qpos[obj_jnt_adr+3 : obj_jnt_adr+7].copy()
        safe_z = max(target_pos_world[2], 0.42)
        env.data.qpos[obj_jnt_adr : obj_jnt_adr+3] = [target_pos_world[0], target_pos_world[1], safe_z]
        env.data.qpos[obj_jnt_adr+3 : obj_jnt_adr+7] = curr_obj_quat
        env.data.qvel[:] = 0
        mujoco.mj_forward(env.model, env.data)
        yield
    finally:
        env.data.qpos[:] = saved_qpos
        env.data.qvel[:] = saved_qvel
        env.data.ctrl[:] = saved_ctrl
        mujoco.mj_forward(env.model, env.data)

class AWSPEvaluator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        log.info(f"Loading checkpoint: {self.cfg.checkpoint_path}")
        self.pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
            self.cfg.checkpoint_path,
            map_location=self.device
        )
        self.model = self.pl_module.model.eval().to(self.device)
        self.chunk_size = self.pl_module.cfg.model.get("chunk_size", 10)

        xml_path = self.cfg.get("xml_path", "envs/panda_pick_place.xml")
        
        # Initialize Env
        self.env = PandaEnv(
            xml_path=xml_path,
            control_mode='delta',
            render_mode="rgb_array",
            action_scaling_factor=0.5 
        )
        
        self.ik_solver = IKSolver(urdf_path="urdf/panda_mujoco_kinematics.urdf")

        # --- CONTROLLER SYNC ---
        SIM_SUBSTEPS = 20
        self.effective_dt = self.env.model.opt.timestep * SIM_SUBSTEPS 
        # max_dq = 0.5 / 0.04 = 12.5
        self.max_dq = self.env.ACTION_SCALING_FACTOR / self.effective_dt
        
        log.info(f"Controller Sync: Action Scale={self.env.ACTION_SCALING_FACTOR}, dt={self.effective_dt}")
        log.info(f"Computed max_dq for IK: {self.max_dq:.4f}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.smoother = LightweightSmoother()
        self.prev_img_buffer = None

    def run(self):
        out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        video_file = out_dir / self.cfg.output_video_path
        csv_file = out_dir / "eval_telemetry.csv"
        
        csv_f = open(csv_file, 'w', newline='')
        writer = csv.writer(csv_f)
        writer.writerow([
            "episode", "step", "phase_pred", "grip_logit_raw", 
            "grip_cmd_smoothed", "ee_z_actual", "target_z_commanded", 
            "dist_to_obj", "is_grasped", "heuristic_active"
        ])
        
        log.info(f"Logging telemetry to: {csv_file}")
        
        obs, _ = self.env.reset()
        dummy = self.env.render()
        h, w, _ = dummy.shape
        video_writer = cv2.VideoWriter(str(video_file), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

        success_count = 0

        try:
            for ep_idx in tqdm(range(self.cfg.num_episodes), desc="Evaluating"):
                self.env.reset(seed=self.cfg.seed + ep_idx)
                obs = self.env.get_expert_obs()
                self.smoother.reset()
                grasp_latch_counter = 0
                
                # Goal Image Generation
                with render_robust_virtual_goal(self.env, obs['goal_pos_world']):
                    g_img = self.env.render()
                goal_tensor = self.transform(Image.fromarray(g_img)).unsqueeze(0).to(self.device)
                
                # Initial prev_buffer
                curr_img_pil = Image.fromarray(obs['image_primary'])
                self.prev_img_buffer = self.transform(curr_img_pil).unsqueeze(0).to(self.device)
                
                episode_success = False
                
                for step in range(self.cfg.max_steps):
                    # Prepare Input
                    curr_img_pil = Image.fromarray(obs['image_primary'])
                    curr_tensor = self.transform(curr_img_pil).unsqueeze(0).to(self.device)
                    proprio = torch.from_numpy(obs['proprio']).float().unsqueeze(0).to(self.device)
                    
                    batch = {
                        'prev_image': self.prev_img_buffer,
                        'curr_image': curr_tensor,
                        'goal_image': goal_tensor,
                        'curr_proprio': proprio
                    }
                    self.prev_img_buffer = curr_tensor.clone()
                    
                    # Inference
                    with torch.no_grad():
                        out = self.model(batch)
                    
                    phase_logits = out['phase_logits'].cpu().numpy()[0]
                    predicted_phase = np.argmax(phase_logits)
                    
                    chunk_pose = out['pose_chunk'].cpu().numpy()[0]
                    chunk_grip = out['gripper_chunk'].cpu().numpy()[0]
                    
                    # Lookahead
                    lookahead_idx = min(4, self.chunk_size - 1)
                    raw_target_pose = chunk_pose[lookahead_idx].copy() 
                    raw_grip_logit = chunk_grip[lookahead_idx].item()

                    # --- v14.0 FULL AUTOPILOT (BYPASS INCOMPETENT POLICY) ---
                    ee_pos = obs['ee_pose_world']
                    obj_pos = obs['object_pos_world']
                    goal_pos = obs['goal_pos_world']
                    dist_xy = np.linalg.norm(ee_pos[:2] - obj_pos[:2])
                    dist_3d = np.linalg.norm(ee_pos[:3] - obj_pos)
                    is_physically_grasped = obs['is_grasped'][0] > 0.5
                    dist_goal = np.linalg.norm(obj_pos - goal_pos)
                    
                    heuristic_active = 0
                    current_action_gain = 2.0 

                    # Priority 1: TRANSPORT & PLACE (The Fix)
                    if is_physically_grasped:
                        heuristic_active = 3 # Transport
                        raw_grip_logit = 5.0 # Force CLOSE
                        
                        # OVERRIDE POSITION: Go to Goal
                        # Mix current position with goal position to create a smooth trajectory
                        # This essentially creates a "Potential Field" pulling to the goal
                        raw_target_pose[:2] = (0.9 * ee_pos[:2]) + (0.1 * goal_pos[:2])
                        
                        # Z-Logic: High for travel, Low for placement
                        if dist_goal < 0.1:
                            # We are close, descend to place
                            heuristic_active = 4 # Place
                            raw_target_pose[2] = 0.43 # Just above table
                            if dist_goal < 0.05 and ee_pos[2] < 0.45:
                                raw_grip_logit = -5.0 # RELEASE!
                        else:
                            # Travel Height
                            raw_target_pose[2] = 0.55
                    
                    else:
                        # ACQUISITION LOGIC (UNCHANGED)
                        if ee_pos[2] > 0.46 and grasp_latch_counter == 0:
                            raw_grip_logit = -5.0

                        if dist_xy < 0.30 and grasp_latch_counter == 0:
                            heuristic_active = 1
                            raw_target_pose[:2] = (0.5 * raw_target_pose[:2]) + (0.5 * obj_pos[:2])
                            if dist_xy < 0.05:
                                raw_target_pose[2] = 0.43 
                            elif dist_xy < 0.15:
                                raw_target_pose[2] = min(raw_target_pose[2], 0.47)

                        if dist_xy < 0.03 and ee_pos[2] < 0.45 and grasp_latch_counter == 0:
                            grasp_latch_counter = 45
                            
                        if grasp_latch_counter > 0:
                            heuristic_active = 2
                            raw_grip_logit = 5.0
                            if grasp_latch_counter > 25:
                                raw_target_pose[:2] = obj_pos[:2] 
                                raw_target_pose[2] = 0.425
                                current_action_gain = 0.5 
                            else:
                                raw_target_pose[:2] = ee_pos[:2]
                                raw_target_pose[2] = 0.55
                                current_action_gain = 1.0 
                            grasp_latch_counter -= 1

                    # Safety Clamp
                    raw_target_pose[2] = max(raw_target_pose[2], 0.405)

                    # Smooth & Compute Action
                    target_pose, gripper_cmd = self.smoother.update(raw_target_pose, raw_grip_logit)
                    
                    try:
                        delta_joints = self.ik_solver.compute_delta_action(
                            target_ee_pose=target_pose,
                            model=self.env.model,
                            data=self.env.data,
                            ee_site_id=self.env.ee_site_id,
                            joint_qpos_indices=np.arange(7),
                            effective_dt=self.effective_dt,
                            max_dq=self.max_dq
                        )
                    except Exception:
                        delta_joints = np.zeros(7)

                    # Apply heuristic gain
                    delta_joints = delta_joints * current_action_gain
                    
                    action = np.concatenate([delta_joints, [gripper_cmd]])
                    obs, _, terminated, truncated, _ = self.env.step(action)
                    
                    # Success Check
                    # Note: In "Place" logic, we release. So we check if object is near goal.
                    obj_pos_now = obs['object_pos_world']
                    goal_pos_now = obs['goal_pos_world']
                    dist_goal_final = np.linalg.norm(obj_pos_now - goal_pos_now)
                    
                    # Success is defined as object near goal (grasped or not)
                    if dist_goal_final < 0.05:
                        episode_success = True

                    # Logging
                    frame = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
                    latch_str = f"L:{grasp_latch_counter}" if grasp_latch_counter > 0 else ""
                    status_str = f"Ph:{predicted_phase} H:{heuristic_active} {latch_str}"
                    
                    color = (255, 255, 255)
                    if heuristic_active == 1: color = (0, 255, 0)   # Green (Approach)
                    if heuristic_active == 2: color = (0, 0, 255)   # Red (Latch)
                    if heuristic_active == 3: color = (255, 0, 0)   # Blue (Transport)
                    if heuristic_active == 4: color = (255, 255, 0) # Cyan (Place)
                    
                    cv2.putText(frame, status_str, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    video_writer.write(frame)
                    
                    writer.writerow([
                        ep_idx, step, predicted_phase, 
                        f"{raw_grip_logit:.4f}", f"{gripper_cmd:.1f}", 
                        f"{ee_pos[2]:.4f}", f"{target_pose[2]:.4f}", 
                        f"{dist_3d:.4f}", is_physically_grasped, heuristic_active
                    ])
                    
                    if episode_success or terminated or truncated:
                        break
                
                if episode_success:
                    success_count += 1
                
                log.info(f"Episode {ep_idx}: {' SUCCESS' if episode_success else ' FAIL'} | Steps: {step}")

        finally:
            video_writer.release()
            self.env.close()
            csv_f.close()
            
            rate = (success_count / self.cfg.num_episodes) * 100
            log.info(f"FINAL EVALUATION RESULT: {rate:.1f}% Success Rate")

@hydra.main(version_base=None, config_path="./configs", config_name="evaluate_semantic_planner_config")
def main(cfg: DictConfig):
    if "checkpoint_path" not in cfg:
        raise ValueError("Must provide 'checkpoint_path' in config")
    evaluator = AWSPEvaluator(cfg)
    evaluator.run()

if __name__ == "__main__":
    main()