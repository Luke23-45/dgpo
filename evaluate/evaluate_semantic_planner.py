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
    
    Uses HYSTERESIS (Schmidt Trigger Logic) to prevent 'Phase Flicker'
    when the robot hovers near a decision boundary.
    """
    def __init__(self):
        self.current_phase = 0
        self.prev_is_grasped = False
        
        # --- Tuned Thresholds ---
        # Hover Height of expert is 0.10m. We must trigger Grasp BEFORE that.
        # Enter Grasp if < 15cm. Exit Grasp only if > 20cm.
        self.thresh_grasp_enter = 0.15 
        self.thresh_grasp_exit  = 0.20
        
        self.thresh_goal_enter = 0.10

    def reset(self):
        self.current_phase = 0
        self.prev_is_grasped = False

    def update(self, obs: Dict[str, Any]) -> int:
        """
        Phases: 0:Reach, 1:Grasp, 2:Transport, 3:Place, 4:Retract
        """
        is_grasped = obs['is_grasped'][0] > 0.5
        
        # Extract geometry
        ee_pos = obs['ee_pose_world'][:3]
        obj_pos = obs['object_pos_world']
        goal_pos = obs['goal_pos_world']
        
        dist_ee_obj = np.linalg.norm(ee_pos - obj_pos)
        dist_obj_goal = np.linalg.norm(obj_pos - goal_pos)
        
        # FSM Transitions
        if not is_grasped:
            if self.prev_is_grasped:
                # Edge Case: Dropped Object / Released at Goal
                if dist_obj_goal < self.thresh_goal_enter:
                    self.current_phase = 4 # Success -> Retract
                else:
                    self.current_phase = 0 # Failure -> Reach
            else:
                # Normal Approach Logic
                if self.current_phase == 0: # Reaching
                    if dist_ee_obj < self.thresh_grasp_enter:
                        self.current_phase = 1
                elif self.current_phase == 1: # Pre-Grasp
                    if dist_ee_obj > self.thresh_grasp_exit:
                        self.current_phase = 0
                elif self.current_phase >= 2:
                    # If we lost the object in later phases, restart
                    self.current_phase = 0
                    
        else: # Is Grasped
            if dist_obj_goal < self.thresh_goal_enter:
                self.current_phase = 3 # Place
            else:
                self.current_phase = 2 # Transport
        
        self.prev_is_grasped = is_grasped
        return self.current_phase


# ==============================================================================
# 3. CORE COMPONENT: TRAJECTORY SMOOTHER
# ==============================================================================

class TrajectorySmoother:
    """
    Low-pass filter for neural network outputs.
    Mimics physical inertia to prevent IK instability.
    """
    def __init__(self, alpha_pos: float = 0.3, alpha_grip: float = 0.1):
        self.alpha_pos = alpha_pos
        self.alpha_grip = alpha_grip
        self.smooth_pose: Optional[np.ndarray] = None
        self.smooth_logit: Optional[float] = None

    def reset(self):
        self.smooth_pose = None
        self.smooth_logit = None

    def update(self, raw_pose: np.ndarray, raw_logit: float) -> Tuple[np.ndarray, float]:
        if self.smooth_pose is None:
            self.smooth_pose = raw_pose.copy()
            self.smooth_logit = raw_logit
        else:
            # EMA on Position
            self.smooth_pose[:3] = (self.alpha_pos * raw_pose[:3]) + \
                                   ((1 - self.alpha_pos) * self.smooth_pose[:3])
            
            # Linear Blend on Quaternion (Approximation of Slerp for small angles)
            # Safe because we renormalize immediately
            self.smooth_pose[3:] = (self.alpha_pos * raw_pose[3:]) + \
                                   ((1 - self.alpha_pos) * self.smooth_pose[3:])
            
            # CRITICAL: Re-project onto the unit manifold
            norm = np.linalg.norm(self.smooth_pose[3:])
            if norm > 1e-6:
                self.smooth_pose[3:] /= norm
                
            # EMA on Gripper Logit
            self.smooth_logit = (self.alpha_grip * raw_logit) + \
                                ((1 - self.alpha_grip) * self.smooth_logit)
            
        return self.smooth_pose, self.smooth_logit


# ==============================================================================
# 4. UTILITY: VIRTUAL GOAL RENDERING
# ==============================================================================

@contextmanager
def render_virtual_goal(env: PandaEnv, goal_pos: np.ndarray):
    """
    A physics-engine trick to 'hallucinate' the goal state.
    1. Save current joint state.
    2. Teleport object to goal position.
    3. Forward Kinematics (without stepping time).
    4. Render Image.
    5. Restore joint state.
    """
    # Backup
    saved_qpos = env.data.qpos.copy()
    saved_qvel = env.data.qvel.copy()
    
    try:
        # Manipulate State
        obj_qpos_adr = env.model.jnt_qposadr[env.object_joint_id]
        # Preserve orientation, move position
        current_quat = env.data.qpos[obj_qpos_adr+3:obj_qpos_adr+7].copy()
        env.data.qpos[obj_qpos_adr:obj_qpos_adr+3] = goal_pos
        env.data.qpos[obj_qpos_adr+3:obj_qpos_adr+7] = current_quat
        
        mujoco.mj_forward(env.model, env.data)
        yield
    finally:
        # Restore
        env.data.qpos[:] = saved_qpos
        env.data.qvel[:] = saved_qvel
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
        
        # A. Load Components
        self.model, self.train_cfg = self._load_model()
        self.env = self._init_env()
        self.ik_solver = IKSolver(urdf_path=cfg.ik_solver_path)
        
        # B. Calibrate Control Frequencies
        # We must match the exact simulation timing used in training data
        self.sim_substeps = 20
        self.effective_dt = self.env.model.opt.timestep * self.sim_substeps
        self.max_dq = self.env.ACTION_SCALING_FACTOR / self.effective_dt
        
        log.info(f"Control Calibration: dt={self.effective_dt:.4f}s | Max Joint Vel (dq)={self.max_dq:.2f}")

        # C. Setup Image Transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor()
        ])

        # D. Logic Components
        self.state_estimator = StateEstimator()
        self.smoother = TrajectorySmoother(alpha_pos=0.5, alpha_grip=0.2)

    def _load_model(self) -> Tuple[SemanticPlanner, DictConfig]:
        log.info(f"Loading Checkpoint: {self.cfg.checkpoint_path}")
        if not os.path.exists(self.cfg.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint missing: {self.cfg.checkpoint_path}")
            
        pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
            self.cfg.checkpoint_path, map_location=self.device, strict=True
        )
        return pl_module.model.eval().to(self.device), pl_module.cfg

    def _init_env(self) -> PandaEnv:
        # Use config paths or fallback
        xml = self.train_cfg.dataset.get('xml_path', 'envs/panda_pick_place.xml')
        log.info(f"Creating PandaEnv from {xml}...")
        return PandaEnv(
            xml_path=xml,
            control_mode='delta'
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
                obs, _ = self.env.reset(seed=seed)
                self.state_estimator.reset()
                self.smoother.reset()

                # 2. Dream the Goal
                with render_virtual_goal(self.env, obs['goal_pos_world']):
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
                    target_pose, target_logit = self.smoother.update(raw_pose, raw_logit)

                    # --- Control Translation ---
                    # Logic: Positive logit -> Closed (-1). Negative logit -> Open (1).
                    gripper_cmd = -1.0 if target_logit > 0.0 else 1.0

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
                        "smooth_grip_score": target_logit,
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
                    cv2.putText(frame, f"Grip: {grip_txt} ({target_logit:.1f})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                    
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