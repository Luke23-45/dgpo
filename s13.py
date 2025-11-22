# FILE: debug_semantic_planner.py
# (Diagnostic Tool for AWSP - Semantic Planner)

import logging
import os
import sys
import csv
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import hydra
import mujoco
import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# --- Project Imports ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from envs.panda_env import PandaEnv
from train.train_semantic_planner import SemanticPlannerLightningModule
from utils.ik_solver import IKSolver

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s", # Simplified format for readability
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("DEBUG")

# ==============================================================================
# 1. LOGIC COMPONENTS
# ==============================================================================

class DebugStateEstimator:
    def __init__(self):
        self.current_phase = 0
        self.prev_is_grasped = False
        # Thresholds (Meters)
        self.thresh_grasp_enter = 0.10
        self.thresh_grasp_exit  = 0.20
        self.thresh_goal_enter = 0.05

    def reset(self):
        self.current_phase = 0
        self.prev_is_grasped = False

    def update(self, obs: Dict[str, Any]) -> int:
        is_grasped = obs['is_grasped'][0] > 0.5
        ee_pos = obs['ee_pose_world'][:3]
        obj_pos = obs['object_pos_world']
        goal_pos = obs['goal_pos_world']
        
        dist_ee_obj = np.linalg.norm(ee_pos - obj_pos)
        dist_obj_goal = np.linalg.norm(obj_pos - goal_pos)
        
        # State Transition Logic
        if is_grasped:
            self.prev_is_grasped = True
            if dist_obj_goal < self.thresh_goal_enter:
                self.current_phase = 3 # Place
            else:
                self.current_phase = 2 # Transport
        else:
            if self.prev_is_grasped:
                if dist_obj_goal < self.thresh_goal_enter:
                    self.current_phase = 4 # Success
                else:
                    self.current_phase = 0 # Failure -> Restart
                self.prev_is_grasped = False
            else:
                if self.current_phase == 0:
                    if dist_ee_obj < self.thresh_grasp_enter:
                        self.current_phase = 1 # Attempt Grasp
                elif self.current_phase == 1:
                    if dist_ee_obj > self.thresh_grasp_exit:
                        self.current_phase = 0 # Lost it, go back to reach

        return self.current_phase, dist_ee_obj

class TrajectorySmoother:
    def __init__(self, alpha_pos=0.5, alpha_grip=0.2):
        self.alpha_pos = alpha_pos
        self.alpha_grip = alpha_grip
        self.reset()

    def reset(self):
        self.smooth_pose = None
        self.smooth_grip_logit = None
        self.gripper_command = 1.0
        self.sticky_timer = 0


    def update(self, raw_pose: np.ndarray, raw_logit: float) -> Tuple[np.ndarray, float]:
        # 1. EMA Position Smoothing
        if self.smooth_pose is None:
            self.smooth_pose = raw_pose.copy()
            self.smooth_grip_logit = raw_logit
        else:
            self.smooth_pose[:3] = self.alpha_pos * raw_pose[:3] + (1 - self.alpha_pos) * self.smooth_pose[:3]
            self.smooth_pose[3:] = self.alpha_pos * raw_pose[3:] + (1 - self.alpha_pos) * self.smooth_pose[3:]
            norm = np.linalg.norm(self.smooth_pose[3:])
            if norm > 1e-6: self.smooth_pose[3:] /= norm
            
            self.smooth_grip_logit = self.alpha_grip * raw_logit + (1 - self.alpha_grip) * self.smooth_grip_logit
        
        # 2. INTELLIGENT LATCHING (The Hot Potato Fix)
        if self.sticky_timer > 0:
            self.sticky_timer -= 1
            # We hold the previous command (Locked)
        else:
            # SWITCH LOGIC
            
            # If currently OPEN (1.0) and model says CLOSE (> 0.0)
            if self.gripper_command == 1.0 and raw_logit > 0.5:
                self.gripper_command = -1.0 # Close!
                self.sticky_timer = 45      # LOCK for 30 steps (~1.5 seconds) to guarantee grasp
                
            # If currently CLOSED (-1.0) and model says OPEN (< -2.0)
            # Note the strict threshold (-2.0) to prevent accidental drops
            elif self.gripper_command == -1.0 and raw_logit < -2.0:
                self.gripper_command = 1.0 # Open
                self.sticky_timer = 10
            
            # Default: maintain state if signal is weak/noisy
            
        return self.smooth_pose, self.gripper_command

# ==============================================================================
# 2. DEBUG RUNNER
# ==============================================================================

class DebugEvaluator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- Load Model ---
        log.info("--------------------------------------------------")
        log.info(f"LOADING MODEL: {self.cfg.checkpoint_path}")
        if not os.path.exists(self.cfg.checkpoint_path):
            log.error("!!! CHECKPOINT FILE NOT FOUND !!!")
            sys.exit(1)

        pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
            self.cfg.checkpoint_path, map_location=self.device
        )
        self.model = pl_module.model.eval().to(self.device)
        self.train_cfg = pl_module.cfg
        
        # --- Load Env ---
        xml_path = self.cfg.get("xml_path", "envs/panda_pick_place.xml")
        log.info(f"LOADING ENV: {xml_path}")
        self.env = PandaEnv(xml_path=xml_path, control_mode='delta')
        
        # --- Load IK ---
        ik_path = self.cfg.get("urdf_path", "urdf/panda_mujoco_kinematics.urdf")
        self.ik_solver = IKSolver(urdf_path=ik_path)

        # --- Calibration ---
        # Matching the hardcoded N_SUBSTEPS=20 in panda_env.py
        SIM_SUBSTEPS = 20 
        self.effective_dt = self.env.model.opt.timestep * SIM_SUBSTEPS
        self.max_dq = self.env.ACTION_SCALING_FACTOR / self.effective_dt
        
        # --- Utils ---
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor()
        ])
        self.state_estimator = DebugStateEstimator()
        self.smoother = TrajectorySmoother()
        
        np.set_printoptions(precision=3, suppress=True)

    @contextmanager
    def virtual_goal_context(self, goal_pos):
        saved_qpos = self.env.data.qpos.copy()
        saved_qvel = self.env.data.qvel.copy()
        try:
            obj_addr = self.env.model.jnt_qposadr[self.env.object_joint_id]
            curr_quat = self.env.data.qpos[obj_addr+3:obj_addr+7].copy()
            self.env.data.qpos[obj_addr:obj_addr+3] = goal_pos
            self.env.data.qpos[obj_addr+3:obj_addr+7] = curr_quat
            mujoco.mj_forward(self.env.model, self.env.data)
            yield
        finally:
            self.env.data.qpos[:] = saved_qpos
            self.env.data.qvel[:] = saved_qvel
            mujoco.mj_forward(self.env.model, self.env.data)

    def run_diagnostic(self):
        """Runs a SINGLE episode with verbose output."""
        video_path = "debug_run.mp4"
        
        # 1. Setup Video
        obs = self.env.reset(seed=self.cfg.seed)[0]
        h, w, _ = obs['image_primary'].shape
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        
        log.info("--------------------------------------------------")
        log.info(f"STARTING DIAGNOSTIC EPISODE (Seed {self.cfg.seed})")
        log.info("--------------------------------------------------")
        
        # 2. Reset Components
        self.state_estimator.reset()
        self.smoother.reset()
        # IMPORTANT: Get expert obs for Ground Truth keys
        obs = self.env.get_expert_obs()
        
        # 3. Dream Goal
        with self.virtual_goal_context(obs['goal_pos_world']):
            goal_img_np = self.env.render(camera_name="fixed_camera")
        goal_tensor = self.transform(Image.fromarray(goal_img_np)).unsqueeze(0).to(self.device)
        
        ep_steps = 250 # Short run
        
        for step in range(ep_steps):
            
            # --- LOGIC: Update Phase ---
            phase, dist_ee_obj = self.state_estimator.update(obs)
            
            # --- PERCEPTION ---
            img_tensor = self.transform(Image.fromarray(obs['image_primary'])).unsqueeze(0).to(self.device)
            proprio_tensor = torch.from_numpy(obs['proprio']).float().unsqueeze(0).to(self.device)
            
            batch = {
                'initial_image': img_tensor,
                'goal_image': goal_tensor,
                'task_phase': torch.tensor([phase], device=self.device),
                'current_proprio': proprio_tensor
            }
            
            # --- MODEL INFERENCE ---
            with torch.no_grad():
                out = self.model(batch)
                
            raw_pose = out['pose'].squeeze().cpu().numpy() # [x,y,z,qx,qy,qz,qw]
            raw_logit = out['gripper_logit'].item()
            
            # --- SMOOTHING ---
            target_pose, gripper_cmd = self.smoother.update(raw_pose, raw_logit)
            
            # --- DEBUG LOGGING (Every 10 steps or on Phase Change) ---
            if step % 5 == 0:
                self.print_diagnostics(step, phase, obs, raw_pose, raw_logit, gripper_cmd, dist_ee_obj)

            # --- CONTROL ---
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
                
                # PATCH: BOOST GAIN to fix "Large Drift"
                # We artificially double the speed to overcome MuJoCo damping
                delta_joints = delta_joints * 1.5
                delta_joints = np.clip(delta_joints, -1.0, 1.0)
                
            except Exception as e:
                log.error(f"IK CRASH: {e}")
                delta_joints = np.zeros(7)
                
            action = np.concatenate([delta_joints, [gripper_cmd]])
            
            # --- PHYSICS ---
            self.env.step(action)
            obs = self.env.get_expert_obs()
            
            # --- VISUALIZATION ---
            frame = self.env.render(camera_name="fixed_camera")
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Draw HUD
            status = f"Ph:{phase} | Grip:{gripper_cmd:.0f} | Logit:{raw_logit:.2f}"
            cv2.putText(bgr, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            
            # Draw Coordinates (Target vs Actual)
            cur_pos = obs['ee_pose_world'][:3]
            pos_txt = f"Tgt:[{target_pose[0]:.2f},{target_pose[1]:.2f},{target_pose[2]:.2f}]"
            act_txt = f"Act:[{cur_pos[0]:.2f},{cur_pos[1]:.2f},{cur_pos[2]:.2f}]"
            cv2.putText(bgr, pos_txt, (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            cv2.putText(bgr, act_txt, (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
            
            writer.write(bgr)
            
        writer.release()
        self.env.close()
        log.info(f"Diagnostic Video Saved: {video_path}")

    def print_diagnostics(self, step, phase, obs, raw_pose, raw_logit, cmd, dist_ee_obj):
        ee_pos = obs['ee_pose_world'][:3]
        obj_pos = obs['object_pos_world']
        
        print(f"\n[Step {step:03d}] PHASE: {phase} | Dist to Obj: {dist_ee_obj:.3f}m")
        print(f"  >> Robot Actual: {ee_pos}")
        print(f"  >> Model Target: {raw_pose[:3]}")
        print(f"  >> Grip Logit:   {raw_logit:.4f} (Command: {cmd})")
        
        # --- INTELLIGENT WARNINGS ---
        
        # 1. Stuck Phase?
        if phase == 0 and dist_ee_obj < 0.15:
             print("  *** WARNING: Robot is CLOSE (hovering?), but Phase 0 didn't switch to 1!")
             print("      Check StateEstimator thresholds.")

        # 2. Confused Gripper?
        if phase == 1: # Grasp Phase
            if raw_logit < 1.0:
                print("  *** WARNING: Phase is 1 (GRASP), but Model predicts OPEN (Logit < 1.0).")
                print("      The model is hesitating to close the hand.")

        # 3. Hallucination? (Target outside table)
        if raw_pose[2] < 0.30 or raw_pose[2] > 0.80:
             print("  *** WARNING: Model Target Z is erratic (Outside workspace).")

        # 4. Frozen Robot?
        dist_delta = np.linalg.norm(raw_pose[:3] - ee_pos)
        if dist_delta > 0.20:
             print(f"  *** WARNING: Large Drift! Robot is {dist_delta:.2f}m away from target.")
             print("      IK Solver might be scaling actions too small.")

# ==============================================================================
# 3. MAIN
# ==============================================================================

@hydra.main(version_base=None, config_path="./configs", config_name="evaluate_semantic_planner_config")
def main(cfg: DictConfig):
    debugger = DebugEvaluator(cfg)
    debugger.run_diagnostic()

if __name__ == "__main__":
    main()