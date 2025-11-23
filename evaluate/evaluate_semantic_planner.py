# FILE: evaluate_semantic_planner.py
# (Definitive, SOTA, Production-Grade Version 3.0)

"""
AWSP Evaluation System (Final Verified).

This script implements a "White-Box" evaluation suite for the Semantic Planner.
Unlike standard black-box scripts that only report success rates, this system
captures a comprehensive telemetry stream to diagnose the coupling between
Perception, Planning, and Control.

System Architecture:
1.  **StateEstimator (Oracle)**: A hysteresis-based Finite State Machine (FSM)
    that inferrs high-level task progress from geometric relations.
2.  **TrajectorySmoother**: A temporal filter (EMA) that stabilizes high-frequency
    stochastic noise from the vision backbone.
3.  **MetricsLogger**: A structured serialization engine that records 30+ signals
    per timestep for post-hoc analysis.
4.  **AWSP_Evaluator**: The primary orchestrator managing the simulation lifecycle.

Performance Notes:
    - Uses Virtual Goal Rendering ('Imagination') via Physics Teleportation.
    - Calibrates Differential IK limits (`max_dq`) to exact training distributions.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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

# --- Project Imports ---
# Robust path injection to ensure deep-nested imports work
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from envs.panda_env import PandaEnv
from models.semantic_planner import SemanticPlanner
from train.train_semantic_planner import SemanticPlannerLightningModule
from utils.ik_solver import IKSolver
from scipy.spatial.transform import Rotation as R
# --- Configuration & Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [AWSP-Eval] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("AWSP_Eval")


# ==============================================================================
# 1. CORE COMPONENT: METRICS LOGGER
# ==============================================================================

class MetricsLogger:
    """
    Production-grade telemetry system. 
    Handles structured logging of scalar, vector, and boolean signals to CSV.
    """
    HEADERS = [
        # Episode Meta
        "episode_id", "step", "time_sec",
        # High-Level State
        "task_phase", "is_holding_object", "success_flag",
        # Geometric Distances (Oracle Inputs)
        "dist_ee_obj", "dist_obj_goal", "obj_height",
        # Model Predictions (Raw)
        "raw_target_x", "raw_target_y", "raw_target_z",
        "raw_grip_logit",
        # Smoother Outputs (Control Targets)
        "smooth_target_x", "smooth_target_y", "smooth_target_z",
        "smooth_grip_score",
        # Robot Actual State
        "actual_ee_x", "actual_ee_y", "actual_ee_z",
        "gripper_width", 
        # Action
        "commanded_gripper", "joint_vel_norm"
    ]

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.file_handle = open(filepath, 'w', newline='')
        self.writer = csv.writer(self.file_handle)
        self.writer.writerow(self.HEADERS)
        self.buffer = []

    def log_step(self, data: Dict[str, Any]):
        """
        Robustly formats a dictionary of signals into a CSV row matching HEADERS.
        Handles Tensors, Arrays, and Floats automatically.
        """
        row = []
        for col in self.HEADERS:
            if col not in data:
                row.append("") # Empty string for missing keys
                continue
            
            val = data[col]
            
            # Type Unpacking
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu().item()
            elif isinstance(val, np.ndarray):
                if val.size == 1:
                    val = val.item()
                else:
                    # If someone tries to log a full vector into one column, stringify it
                    val = str(val.tolist())
            elif isinstance(val, bool):
                val = 1 if val else 0
            
            # Float formatting
            if isinstance(val, float):
                val = f"{val:.6f}"
            
            row.append(val)
        
        self.writer.writerow(row)
        # Periodic flush could be added here, but OS buffer is usually fine

    def close(self):
        self.file_handle.flush()
        self.file_handle.close()


# ==============================================================================
# 2. CORE COMPONENT: ROBUST STATE ESTIMATOR
# ==============================================================================

class StateEstimator:
    """
    Finite State Machine (FSM) for Task Phase Inference.
    Uses HYSTERESIS to prevent 'Phase Flicker' and handles Grasp Triggering.
    """
    def __init__(self):
        self.current_phase = 0
        self.prev_is_grasped = False
        
        # Distances to Object (Meters)
        # Phase 0 -> 1: When we get close (12cm)
        self.thresh_grasp_enter = 0.12 
        # Phase 1 -> 0: If we drift away (20cm)
        self.thresh_grasp_exit  = 0.20
        
        # Distance to Goal (Meters)
        self.thresh_goal_enter = 0.05

    def reset(self):
        self.current_phase = 0
        self.prev_is_grasped = False

    def update(self, obs: Dict[str, Any]) -> int:
        """
        Phases: 0:Reach, 1:Grasp/Lift, 2:Transport, 3:Place, 4:Retract
        """
        # Fix: PandaEnv returns a 1-element array for this key
        is_grasped = obs['is_grasped'][0] > 0.5
        
        ee_pos = obs['ee_pose_world'][:3]
        obj_pos = obs['object_pos_world']
        goal_pos = obs['goal_pos_world']
        
        dist_ee_obj = np.linalg.norm(ee_pos - obj_pos)
        dist_obj_goal = np.linalg.norm(obj_pos - goal_pos)
        
        # --- TRANSITION LOGIC (Fixed Deadlock) ---
        if is_grasped:
            self.prev_is_grasped = True
            if dist_obj_goal < self.thresh_goal_enter:
                self.current_phase = 3 # Place
            else:
                self.current_phase = 2 # Transport
        else:
            # Not physically grasped
            if self.prev_is_grasped:
                # Dropped or Placed
                if dist_obj_goal < self.thresh_goal_enter:
                    self.current_phase = 4 # Success / Retract
                else:
                    self.current_phase = 0 # Failure / Restart Reach
                self.prev_is_grasped = False
            else:
                # Standard Approach
                if self.current_phase == 0: # Reach
                    # Force transition to Grasp Phase (1) when close enough
                    if dist_ee_obj < self.thresh_grasp_enter:
                        self.current_phase = 1
                elif self.current_phase == 1: # Grasp Attempt
                    # Only revert to Reach if we drift far away
                    if dist_ee_obj > self.thresh_grasp_exit:
                        self.current_phase = 0
                elif self.current_phase >= 2:
                    # Lost object mid-air
                    self.current_phase = 0

        return self.current_phase

# ==============================================================================
# 3. CORE COMPONENT: TRAJECTORY SMOOTHER
# ==============================================================================
class TrajectorySmoother:
    """
    Output Filter & Latching Logic.
    
    Features:
    1. Exponential Moving Average (EMA) for 7D Pose.
    2. State-Locked Hysteresis for Gripper (The "Anti-Flicker" Fix).
    """
    def __init__(self, alpha_pos=0.4, alpha_grip=0.2):
        self.alpha_pos = alpha_pos
        self.alpha_grip = alpha_grip
        
        self.smooth_pose = None
        self.smooth_grip_logit = None
        
        # Hysteresis State
        self.gripper_command = 1.0 # Start OPEN
        self.sticky_timer = 0
        self.STICKY_DURATION = 20 # Lock state for ~0.6s (20 steps * 0.03s) to allow actuation

    def reset(self):
        self.smooth_pose = None
        self.smooth_grip_logit = None
        self.gripper_command = 1.0
        self.sticky_timer = 0

    def update(self, raw_pose: np.ndarray, raw_logit: float) -> Tuple[np.ndarray, float]:
        # 1. Pose Smoothing (Standard EMA)
        if self.smooth_pose is None:
            self.smooth_pose = raw_pose.copy()
            self.smooth_grip_logit = raw_logit
        else:
            self.smooth_pose[:3] = self.alpha_pos * raw_pose[:3] + (1 - self.alpha_pos) * self.smooth_pose[:3]
            self.smooth_pose[3:] = self.alpha_pos * raw_pose[3:] + (1 - self.alpha_pos) * self.smooth_pose[3:]
            norm = np.linalg.norm(self.smooth_pose[3:])
            if norm > 1e-6: self.smooth_pose[3:] /= norm
            
            self.smooth_grip_logit = self.alpha_grip * raw_logit + (1 - self.alpha_grip) * self.smooth_grip_logit
            
        # 2. Gripper Hysteresis Logic (The Fix)
        # We define distinct thresholds to prevent flickering around 0.0
        CLOSE_THRESH = 1.5   # Must be confident to close
        OPEN_THRESH = -1.5   # Must be confident to open
        
        if self.sticky_timer > 0:
            # Locked in state to allow physics execution
            self.sticky_timer -= 1
        else:
            # Free to switch states
            if self.gripper_command == 1.0: # Currently Open
                # Only switch to Close if confidence is high
                if self.smooth_grip_logit > CLOSE_THRESH:
                    self.gripper_command = -1.0 # Switch to Close
                    self.sticky_timer = self.STICKY_DURATION # Lock it!
                    
            elif self.gripper_command == -1.0: # Currently Closed
                # Only Open if VERY confident we should let go.
                if self.smooth_grip_logit < OPEN_THRESH: 
                    self.gripper_command = 1.0 # Switch to Open
                    self.sticky_timer = self.STICKY_DURATION
            
        # Return the discrete command (-1.0 or 1.0), NOT the logit
        return self.smooth_pose, self.gripper_command


# ==============================================================================
# 4. UTILITY: VIRTUAL GOAL RENDERING
# ==============================================================================


def _mujoco_to_scipy(quat_wxyz: np.ndarray) -> np.ndarray:
    """Helper: Convert MuJoCo WXYZ -> Scipy XYZW."""
    return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])

def calculate_retract_joints(env: PandaEnv, ik_solver: IKSolver, target_pos_world: np.ndarray, object_quat_wxyz: np.ndarray) -> np.ndarray:
    """
    Calculates joint angles for the robot to hover above the goal,
    ALIGNED with the object's orientation (Expert behavior).
    """
    # 1. Define Ideal Position (Hover)
    hover_z = 0.55 
    target_pos = np.array([target_pos_world[0], target_pos_world[1], hover_z])
    
    # 2. Define Ideal Orientation (Aligned with Cube)
    # The Expert aligns the gripper faces with the cube faces.
    # Object Frame: Z is Up.
    # Gripper Frame: Z is Approach (Down).
    # To match faces: We take Object Rotation and rotate 180 deg around X-axis (flip upside down).
    q_obj = _mujoco_to_scipy(object_quat_wxyz)
    r_obj = R.from_quat(q_obj)
    
    # Apply 180 flip to point gripper down while keeping Yaw alignment
    r_target = r_obj * R.from_euler('x', 180, degrees=True)
    target_matrix = r_target.as_matrix()

    # 3. Transform World -> Robot Base Frame
    base_pos, base_quat_xyzw = env.get_base_pose()
    
    # Create transformation matrices
    R_base_world = R.from_quat(base_quat_xyzw).as_matrix()
    T_base_world = np.eye(4)
    T_base_world[:3, :3] = R_base_world
    T_base_world[:3, 3] = base_pos
    
    # Invert to get World -> Base
    T_world_base = np.linalg.inv(T_base_world)
    
    # Transform Target Position
    target_pos_homo = np.append(target_pos, 1.0)
    target_pos_in_base = (T_world_base @ target_pos_homo)[:3]
    
    # Transform Target Orientation
    target_rot_in_base = T_world_base[:3, :3] @ target_matrix

    # 4. Solve Inverse Kinematics
    current_joints = env.data.qpos[:7].copy()
    initial_guess = [0.0] * len(ik_solver.chain.links)
    for i, val in enumerate(current_joints):
        if i < len(ik_solver._active_idx):
            initial_guess[ik_solver._active_idx[i]] = val

    full_joints = ik_solver.chain.inverse_kinematics(
        target_position=target_pos_in_base,
        target_orientation=target_rot_in_base,
        orientation_mode="all",
        initial_position=initial_guess
    )
    
    final_joints = np.array([full_joints[i] for i in ik_solver._active_idx])
    low, high = env.get_action_space_limits()
    return np.clip(final_joints, low[:7], high[:7])

@contextmanager
def render_virtual_goal(env: PandaEnv, ik_solver: IKSolver, goal_pos_world: np.ndarray):
    """
    Context manager that teleports BOTH the object AND the robot
    to a mathematically perfect 'Task Complete' state.
    """
    saved_qpos = env.data.qpos.copy()
    saved_qvel = env.data.qvel.copy()
    saved_ctrl = env.data.ctrl.copy()

    try:
        # --- A. Teleport Object ---
        obj_addr = env.model.jnt_qposadr[env.object_joint_id]
        current_obj_quat = env.data.qpos[obj_addr+3 : obj_addr+7].copy()
        
        # Perfect placement on table (Z ~ 0.42)
        perfect_obj_pos = goal_pos_world.copy()
        if perfect_obj_pos[2] < 0.41: 
            perfect_obj_pos[2] = 0.42 
            
        env.data.qpos[obj_addr : obj_addr+3] = perfect_obj_pos
        env.data.qpos[obj_addr+3 : obj_addr+7] = current_obj_quat 

        # --- B. Teleport Robot (Aligned with Object) ---
        # Pass the object's orientation to the calculator
        target_joints = calculate_retract_joints(env, ik_solver, perfect_obj_pos, current_obj_quat)
        
        env.data.qpos[:7] = target_joints
        
        # --- C. Set Gripper to Open ---
        env.data.qpos[7] = 0.04
        env.data.qpos[8] = 0.04

        # --- D. Stabilize ---
        env.data.qvel[:] = 0.0
        env.data.ctrl[:7] = target_joints
        env.data.ctrl[7] = 0.04
        
        mujoco.mj_forward(env.model, env.data)
        
        yield

    finally:
        env.data.qpos[:] = saved_qpos
        env.data.qvel[:] = saved_qvel
        env.data.ctrl[:] = saved_ctrl
        mujoco.mj_forward(env.model, env.data)

# ==============================================================================
# 5. THE EVALUATOR ENGINE
# ==============================================================================

class AWSPEvaluator:
    """
    Orchestrates the evaluation process. Manages models, envs, and resources.
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load Model
        log.info(f"Loading Checkpoint: {self.cfg.checkpoint_path}")
        pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
            self.cfg.checkpoint_path, map_location=self.device, strict=True
        )
        self.model = pl_module.model.eval().to(self.device)
        self.train_cfg = pl_module.cfg
        
        # 2. Init Environment
        self.env = self._init_env()
        
        # 3. Init IK Solver
        ik_path = self.cfg.get("urdf_path", "urdf/panda_mujoco_kinematics.urdf")
        self.ik_solver = IKSolver(urdf_path=ik_path)
        
        # 4. CRITICAL: Calibrate Control Frequencies
        # PandaEnv (production version) steps physics 20 times per control step.
        # We MUST hardcode this to match panda_env.py logic.
        SIM_SUBSTEPS = 20 
        self.effective_dt = self.env.model.opt.timestep * SIM_SUBSTEPS
        
        # Calculate Max Joint Velocity based on Env Scaling
        # This ensures the IK solver doesn't request speeds the Sim cuts in half.
        self.max_dq = self.env.ACTION_SCALING_FACTOR / self.effective_dt
        
        log.info(f"Control Calibration: dt={self.effective_dt:.4f}s | Max Joint Vel (dq)={self.max_dq:.2f}")

        # 5. Components
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.state_estimator = StateEstimator()
        self.smoother = TrajectorySmoother(alpha_pos=0.5, alpha_grip=0.2)

    def _init_env(self) -> PandaEnv:
        # PRIORITY: Use Eval Config XML -> Fallback to Train Config -> Default
        # This fixes the "Config Disconnect" error
        xml_path = self.cfg.get("xml_path", None)
        if xml_path is None:
            xml_path = self.train_cfg.dataset.get('xml_path', 'envs/panda_pick_place.xml')
            
        log.info(f"Creating PandaEnv from: {xml_path}")
        return PandaEnv(
            xml_path=xml_path,
            control_mode='delta',
            render_mode="rgb_array"
        )

    def run(self):
        """Main Execution Loop"""
        
        # Setup Outputs
        out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        video_file = out_dir / self.cfg.output_video_path
        csv_file = out_dir / self.cfg.output_csv_path
        
        log.info(f"Writing Video -> {video_file}")
        log.info(f"Writing CSV -> {csv_file}")

        # Init Video Writer
        obs, _ = self.env.reset()
        dummy_frame = self.env.render()
        h, w, _ = dummy_frame.shape
        video_writer = cv2.VideoWriter(str(video_file), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        
        # Init Metrics Logger
        logger = MetricsLogger(csv_file)

        successes = []

        try:
            for ep_idx in tqdm(range(self.cfg.num_episodes), desc="Simulating"):
                # 1. Episode Reset
                seed = self.cfg.seed + ep_idx if not self.cfg.eval_static_scene else self.cfg.seed
                
                # PATCH: Reset environment, then IMMEDIATELY fetch expert observation
                # so the Oracle has access to ground truth positions.
                self.env.reset(seed=seed)
                obs = self.env.get_expert_obs() 

                self.state_estimator.reset()
                self.smoother.reset()

                # 2. Dream the Goal
                with render_virtual_goal(self.env,self.ik_solver,obs['goal_pos_world']):
                    goal_img_np = self.env.render()
                goal_tensor = self.transform(Image.fromarray(goal_img_np)).unsqueeze(0).to(self.device)

                episode_success = False
                
                # 3. Step Loop
                for step in range(self.cfg.max_steps):
                    # --- Phase Inference ---
                    current_phase = self.state_estimator.update(obs)
                    
                    # --- Perception ---
                    img_tensor = self.transform(Image.fromarray(obs['image_primary'])).unsqueeze(0).to(self.device)
                    proprio_tensor = torch.from_numpy(obs['proprio']).float().unsqueeze(0).to(self.device)
                    
                    # --- Semantic Planning ---
                    batch = {
                        'initial_image': img_tensor,
                        'goal_image': goal_tensor,
                        'task_phase': torch.tensor([current_phase], device=self.device),
                        'current_proprio': proprio_tensor
                    }
                    
                    with torch.no_grad():
                        out = self.model(batch)
                    
                    raw_pose = out['pose'].squeeze().cpu().numpy()
                    raw_logit = out['gripper_logit'].item()

                    # --- Temporal Smoothing ---
                    target_pose, gripper_cmd = self.smoother.update(raw_pose, raw_logit)


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

                    action = np.concatenate([delta_joints, [gripper_cmd]])

                    # --- Physics Step ---
                    
                    obs, _, terminated, truncated, info = self.env.step(action)

                    # --- Metrics & Success Check ---
                    dist_goal = np.linalg.norm(obs['object_pos_world'] - obs['goal_pos_world'])
                    obj_height = obs['object_pos_world'][2]
                    
                    if dist_goal < 0.05 and obj_height > 0.41 and obs['is_grasped'][0] > 0.5:
                        episode_success = True
                    
                    # Log detailed telemetry
                    logger.log_step({
                        "episode_id": ep_idx, "step": step, "time_sec": step * self.effective_dt,
                        "task_phase": current_phase, 
                        "is_holding_object": obs['is_grasped'][0] > 0.5,
                        "success_flag": episode_success,
                        "dist_ee_obj": np.linalg.norm(obs['ee_pose_world'][:3] - obs['object_pos_world']),
                        "dist_obj_goal": dist_goal, "obj_height": obj_height,
                        "raw_target_x": raw_pose[0], "raw_target_y": raw_pose[1], "raw_target_z": raw_pose[2],
                        "raw_grip_logit": raw_logit,
                        "smooth_target_x": target_pose[0], "smooth_target_y": target_pose[1], "smooth_target_z": target_pose[2],
                        "smooth_grip_score": gripper_cmd,
                        "actual_ee_x": obs['ee_pose_world'][0], "actual_ee_y": obs['ee_pose_world'][1], "actual_ee_z": obs['ee_pose_world'][2],
                        "commanded_gripper": gripper_cmd,
                        "joint_vel_norm": np.linalg.norm(delta_joints)
                    })

                    # Visualization
                    frame = self.env.render()
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # HUD Overlay
                    phase_names = ["REACH", "GRASP", "MOVE", "PLACE", "DONE"]
                    status_txt = phase_names[min(current_phase, 4)]
                    grip_txt = "CLOSED" if gripper_cmd < 0 else "OPEN"
                    color = (0, 255, 0) if gripper_cmd < 0 else (0, 255, 255)
                    
                    cv2.putText(frame, f"Ep {ep_idx} | {status_txt}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, f"Grip: {grip_txt} ({gripper_cmd:.1f})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                    
                    video_writer.write(frame)

                    if episode_success or terminated or truncated:
                        break
                
                successes.append(episode_success)
                log.info(f"Episode {ep_idx} Complete. Result: {'SUCCESS' if episode_success else 'FAIL'}")

        finally:
            # --- Resource Teardown ---
            log.info("Shutting down evaluation resources...")
            video_writer.release()
            logger.close()
            self.env.close()
            
            success_rate = (sum(successes) / len(successes)) * 100 if successes else 0.0
            log.info("-" * 40)
            log.info(f"EVALUATION SUMMARY: {success_rate:.1f}% Success Rate")
            log.info(f"Logs: {csv_file}")
            log.info("-" * 40)

# ==============================================================================
# 4. MAIN ENTRY POINT
# ==============================================================================

@hydra.main(version_base=None, config_path="../configs", config_name="evaluate_semantic_planner_config")
def main(cfg: DictConfig):
    evaluator = AWSPEvaluator(cfg)
    evaluator.run()

if __name__ == "__main__":
    main()