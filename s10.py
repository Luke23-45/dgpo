# FILE: evaluate_semantic_planner.py
# (v9.7 - Corrected Horizon & Goal consistency)

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

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
        # Alpha 0.8 = Trust New Data 80% (Very low lag, high responsiveness)
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

        # PATCH: Lower threshold to 0.5 to encourage grasping
        if not self.gripper_closed and self.smooth_grip_logit > 0.5:
            self.gripper_closed = True
        elif self.gripper_closed and self.smooth_grip_logit < -0.5:
            self.gripper_closed = False
            
        gripper_cmd = -1.0 if self.gripper_closed else 1.0
        return self.smooth_pose, gripper_cmd

# ==============================================================================
# 2. ROBUST VIRTUAL GOAL
# ==============================================================================
@contextmanager
def render_robust_virtual_goal(env, target_pos_world):
    """
    Renders the goal state by manually teleporting the object.
    CRITICAL: Does NOT solve IK. Uses a fixed 'Home' pose for the arm.
    This ensures the 'Goal' image looks exactly like the training data's 
    'Done' state, avoiding OOD visual features for the Transformer.
    """
    # 1. Save State
    saved_qpos = env.data.qpos.copy()
    saved_qvel = env.data.qvel.copy()
    saved_ctrl = env.data.ctrl.copy()
    
    try:
        # 2. Set Robot to Standard Home/Retract Pose (Joint Space)
        # This must match your Expert's 'home_pose_7d' or standard retract.
        # [0, -45, 0, -135, 0, 90, 45] degrees approx
        home_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        env.data.qpos[:7] = home_qpos
        env.data.qpos[7:] = 0.04 # Open Gripper

        # 3. Teleport Object to Goal
        # Find object address
        obj_jnt_adr = env.model.jnt_qposadr[env.object_joint_id]
        
        # Keep object orientation, just move position
        curr_obj_quat = env.data.qpos[obj_jnt_adr+3 : obj_jnt_adr+7].copy()
        
        # Ensure Z is correct (on table surface)
        safe_z = max(target_pos_world[2], 0.42)
        
        env.data.qpos[obj_jnt_adr : obj_jnt_adr+3] = [target_pos_world[0], target_pos_world[1], safe_z]
        env.data.qpos[obj_jnt_adr+3 : obj_jnt_adr+7] = curr_obj_quat

        # 4. Stabilize
        env.data.qvel[:] = 0
        mujoco.mj_forward(env.model, env.data)
        
        yield

    finally:
        # 5. Restore State
        env.data.qpos[:] = saved_qpos
        env.data.qvel[:] = saved_qvel
        env.data.ctrl[:] = saved_ctrl
        mujoco.mj_forward(env.model, env.data)

# ==============================================================================
# 3. EVALUATOR
# ==============================================================================
class AWSPEvaluator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Checkpoint
        log.info(f"Loading checkpoint: {self.cfg.checkpoint_path}")
        self.pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
            self.cfg.checkpoint_path, map_location=self.device
        )
        self.model = self.pl_module.model.eval().to(self.device)
        self.chunk_size = self.pl_module.cfg.model.get("chunk_size", 10)

        # Setup Environment
        xml_path = self.cfg.get("xml_path", "envs/panda_pick_place.xml")
        self.env = PandaEnv(
            xml_path=xml_path, 
            control_mode='delta', # Must match training setup
            render_mode="rgb_array",
            action_scaling_factor=0.5 # Ensure this matches training env
        )
        self.ik_solver = IKSolver(urdf_path="urdf/panda_mujoco_kinematics.urdf")
        
        # Evaluation Config
        SIM_SUBSTEPS = 20
        self.effective_dt = self.env.model.opt.timestep * SIM_SUBSTEPS
        # Reduced max_dq to prevent instability, relying on accumulation
        self.max_dq = 8.0 

        # Transform (Must match SigLIP requirements/Training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            # SigLIP Mean/Std (0.5 is standard approximation for SigLIP/CLIP)
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.smoother = LightweightSmoother()
        self.prev_img_buffer = None

    def run(self):
        out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        video_file = out_dir / self.cfg.output_video_path
        
        # Init Video
        obs, _ = self.env.reset()
        dummy_frame = self.env.render()
        h, w, _ = dummy_frame.shape
        video_writer = cv2.VideoWriter(str(video_file), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        
        success_count = 0

        try:
            for ep_idx in tqdm(range(self.cfg.num_episodes), desc="Evaluating"):
                # 1. Reset
                self.env.reset(seed=self.cfg.seed + ep_idx)
                obs = self.env.get_expert_obs()
                self.smoother.reset()

                # 2. Render Goal (Using Robust Method)
                with render_robust_virtual_goal(self.env, obs['goal_pos_world']):
                    g_img_np = self.env.render()
                
                goal_tensor = self.transform(Image.fromarray(g_img_np)).unsqueeze(0).to(self.device)

                # 3. Init Context
                curr_img_pil = Image.fromarray(obs['image_primary'])
                # For first frame, prev = curr (zero velocity start)
                self.prev_img_buffer = self.transform(curr_img_pil).unsqueeze(0).to(self.device)

                episode_success = False
                
                for step in range(self.cfg.max_steps):
                    # Prepare Inputs
                    curr_img_pil = Image.fromarray(obs['image_primary'])
                    curr_tensor = self.transform(curr_img_pil).unsqueeze(0).to(self.device)
                    proprio = torch.from_numpy(obs['proprio']).float().unsqueeze(0).to(self.device)

                    batch = {
                        'prev_image': self.prev_img_buffer, 
                        'curr_image': curr_tensor,
                        'goal_image': goal_tensor, 
                        'curr_proprio': proprio
                    }
                    
                    # Store current for next step
                    self.prev_img_buffer = curr_tensor.clone()

                    # Model Forward
                    with torch.no_grad():
                        out = self.model(batch)

                    # --- STRATEGIST v9.0 LOGIC FIX ---
                    
                    # 1. Unpack Chunk
# Model Forward
                    with torch.no_grad():
                        out = self.model(batch)

                    # --- STRATEGIST v9.7 LOGIC FIX ---
                    
                    # 1. Unpack Chunk & Phase (Restored)
                    chunk_pose = out['pose_chunk'].cpu().numpy()[0]    # (K, 7)
                    chunk_grip = out['gripper_chunk'].cpu().numpy()[0] # (K, 1)
                    
                    # FIX: Define predicted_phase so the logger doesn't crash
                    phase_logits = out['phase_logits'].cpu().numpy()[0]
                    predicted_phase = np.argmax(phase_logits)

                    # 2. HYBRID HORIZON SELECTION (The "Hawk-Eye" Patch)
                    # XY & Rotation: Use t+1 (Index 1) for precise steering
                    target_xy = chunk_pose[1][:2]
                    target_quat = chunk_pose[1][3:]
                    
                    # Z-Axis: Use t+4 (Index 4) for aggressive descent intentions
                    target_z = chunk_pose[4][2]
                    
                    # Reassemble Target
                    raw_target_pose = np.concatenate([target_xy, [target_z], target_quat])
                    
                    # Gripper: Use t+3 to anticipate closing slightly early
                    raw_grip_logit = chunk_grip[3].item()

                    # 3. Smooth
                    target_pose, gripper_cmd = self.smoother.update(raw_target_pose, raw_grip_logit)

                    # 4. Compute Control (Delta)
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
                    except Exception as e:
                        delta_joints = np.zeros(7)

                    # 5. GAIN BOOST (The "Adrenaline" Patch)
                    # Multiplier to overcome simulation damping for small deltas
                    action_gain = 2.5
                    delta_joints = delta_joints * action_gain

                    # 6. Step Environment
                    action = np.concatenate([delta_joints, [gripper_cmd]])
                    obs, _, terminated, truncated, _ = self.env.step(action)

                    # 6. Check Success
                    # Standard check: Object lifted and near goal
                    obj_pos = obs['object_pos_world']
                    goal_pos = obs['goal_pos_world']
                    dist_goal = np.linalg.norm(obj_pos - goal_pos)
                    is_lifted = obj_pos[2] > 0.41
                    
                    if dist_goal < 0.05 and is_lifted and obs['is_grasped'][0] > 0.5:
                        episode_success = True
                    
                    # Render
                    frame = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
                    
                    # Overlay Telemetry
                    phase_str = f"Phase Pred: {predicted_phase}"
                    grip_str = f"Grip: {'CLOSE' if gripper_cmd < 0 else 'OPEN'}"
                    cv2.putText(frame, phase_str, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, grip_str, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    video_writer.write(frame)

                    if episode_success or terminated or truncated:
                        break

                if episode_success:
                    success_count += 1
                log.info(f"Episode {ep_idx}: {'✅ SUCCESS' if episode_success else '❌ FAIL'} | Steps: {step}")

        finally:
            video_writer.release()
            self.env.close()
            rate = (success_count / self.cfg.num_episodes) * 100
            log.info(f"FINAL EVALUATION RESULT: {rate:.1f}% Success Rate")

@hydra.main(version_base=None, config_path="./configs", config_name="evaluate_semantic_planner_config")
def main(cfg: DictConfig):
    # Ensure config structure for hydra
    if "checkpoint_path" not in cfg:
        raise ValueError("Must provide 'checkpoint_path' in config")
        
    evaluator = AWSPEvaluator(cfg)
    evaluator.run()

if __name__ == "__main__":
    main()