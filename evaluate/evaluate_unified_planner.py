# FILE: evaluate/evaluate_unified_planner.py
# (Production-Grade Evaluation Script for UnifiedDiffusionPlanner)

"""
Unified Planner Evaluation Script

This script evaluates a trained UnifiedDiffusionPlanner checkpoint by running it
in the PandaEnv with DDIM sampling and recording detailed metrics.

Features:
- DDIM sampling with Classifier-Free Guidance (CFG)
- Goal image rendering via object teleportation
- Delta action execution (compatible with PandaEnv delta mode)
- MP4 video recording with HUD overlay showing:
  - Episode ID, step count, time
  - Distance to goal (color-coded)
  - Gripper state and grasp indicator
  - Diffusion sampling info (DDIM steps, CFG scale)
  - Success/failure status
- Detailed CSV metrics logging
- Multi-episode evaluation with success rate calculation

Usage:
    python evaluate/evaluate_unified_planner.py \\
        --checkpoint /path/to/unified_planner.ckpt \\
        --config configs/eval_unified_planner_config.yaml \\
        --n_episodes 5 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import hydra
import mujoco
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import transforms

# Project Imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.panda_env import PandaEnv
from train.train_unified_planner import UnifiedPlannerLightningModule
from models.unified_diffusion_planner import UnifiedDiffusionPlanner
from utils.ik_solver import IKSolver
from utils.scripted_expert import ScriptedExpert, ObjectProfile
from scipy.spatial.transform import Rotation as R

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("UnifiedPlanner_Eval")


# ==============================================================================
# 1. UTILITY: GOAL IMAGE RENDERING
# ==============================================================================

def render_goal_image(env: PandaEnv, ik_solver: Optional[IKSolver], obs: Dict) -> np.ndarray:
    """
    Renders the goal image by determining the Robot's Hover Pose using Inverse Kinematics,
    matching the Expert Policy's Retract State.
    
    Includes CRITICAL fixes for:
    - Object sinking (lifting by half-height 0.02)
    - Robot height relative to object center (lifting by 0.02)
    - Dynamic Gripper Orientation Alignment (matching Expert [1,0,0,0] reference)
    
    Args:
        env: PandaEnv instance
        ik_solver: IKSolver instance (optional). If None, a temporary one is created.
        obs: Observation dictionary containing goal_pos_world and goal_orn_world
        
    Returns:
        goal_image: (H, W, 3) RGB image
    """
    # Save current state
    original_qpos = env.data.qpos.copy()
    original_qvel = env.data.qvel.copy()
    original_ctrl = env.data.ctrl.copy()
    
    # 1. Ensure IK Solver exists (Critical for accurate robot pose)
    if ik_solver is None:
        try:
            # Fallback for when UnifiedPlanner is run without IK for control
            ik_solver = IKSolver(urdf_path="urdf/panda_mujoco_kinematics.urdf")
        except Exception as e:
            log.warning(f"Could not instantiate IK Solver for rendering: {e}")
            # Without IK, we can't position the robot correctly.
            # Fallback: Just place object and leave robot at home (sub-optimal but safe)
            pass

    # Constants matching Expert
    HOVER_HEIGHT = 0.10
    
    # Instantiate temporary expert to use its alignment logic
    dummy_expert = ScriptedExpert(ObjectProfile(size=np.zeros(3), grasp_width_normalized=0.0))

    try:
        goal_pos_world = obs['goal_pos_world']
        goal_orn_world = obs['goal_orn_world']

        # 1. Move object to goal POSE (Position + Orientation)
        obj_addr = env.model.jnt_qposadr[env.object_joint_id]
        
        # FIX 1: Lift object by half-height (0.02) to prevent sinking into table (Z=0.401).
        target_obj_pos = goal_pos_world + np.array([0.0, 0.0, 0.02])
        env.data.qpos[obj_addr:obj_addr+3] = target_obj_pos
        
        # Set Orientation
        goal_orn_wxyz = env._scipy_xyzw_to_mujoco_wxyz(goal_orn_world)
        env.data.qpos[obj_addr+3:obj_addr+7] = goal_orn_wxyz
        
        # 2. Position Robot (only if IK is available)
        if ik_solver is not None:
             # FIX 2: Also lift robot by half-height (0.02) so it hovers relative to 
            # the object's center, not the table surface.
            target_pos = goal_pos_world + np.array([0.0, 0.0, HOVER_HEIGHT + 0.02])
            
            # FIX 3: Expert uses [1, 0, 0, 0] (Rot X 180) as base. [0, 1, 0, 0] causes twisted arm.
            seed_downward_quat = np.array([1.0, 0.0, 0.0, 0.0])
            target_quat = dummy_expert._calculate_aligned_orientation(goal_orn_world, seed_downward_quat)
            
            target_pose_7d = np.concatenate([target_pos, target_quat])
            
            # Use hardcoded home_qpos, same as env.reset()
            home_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
            
            goal_qpos = ik_solver.solve_ik_static(
                target_pose=target_pose_7d, 
                model=env.model, 
                data=env.data, 
                ee_site_id=env.ee_site_id,
                q0=home_qpos 
            )
            
            if goal_qpos is not None:
                env.data.qpos[:7] = goal_qpos
            else:
                env.data.qpos[:7] = home_qpos
                
        # Open Gripper (Retract state has open gripper)
        env.data.qpos[7:9] = 0.04 
    
        # Forward and render
        mujoco.mj_forward(env.model, env.data)
        goal_image = env.render()
        
        return goal_image
    
    finally:
        # Restore State
        env.data.qpos[:] = original_qpos
        env.data.qvel[:] = original_qvel
        env.data.ctrl[:] = original_ctrl
        mujoco.mj_forward(env.model, env.data)


# ==============================================================================
# 2. METRICS LOGGER
# ==============================================================================

class EvaluationLogger:
    """
    Production-grade CSV logger for evaluation metrics.
    
    Logs step-wise telemetry including:
    - Episode metadata
    - Robot/object/goal positions
    - Distances and grasp state
    - Diffusion sampling parameters
    - Success flags
    """
    
    HEADERS = [
        # Episode Meta
        "episode_id", "step", "time_sec",
        # World State
        "ee_x", "ee_y", "ee_z",
        "obj_x", "obj_y", "obj_z",
        "goal_x", "goal_y", "goal_z",
        # Distances
        "dist_ee_obj", "dist_obj_goal",
        # Gripper
        "is_grasped", "gripper_cmd",
        # Diffusion Metrics
        "diffusion_steps",      # DDIM steps used
        "guidance_scale",       # CFG scale
        # Policy Output (first step of chunk)
        "policy_dx", "policy_dy", "policy_dz",
        "policy_grip",
        # Flags
        "success_flag"
    ]
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.file = open(filepath, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.HEADERS)
        log.info(f"CSV logger initialized: {filepath}")
    
    def log_step(self, data: Dict):
        """Log a single step of evaluation."""
        row = [data.get(h, 0.0) for h in self.HEADERS]
        self.writer.writerow(row)
    
    def close(self):
        self.file.close()
        log.info(f"CSV logger closed: {self.filepath}")


# ==============================================================================
# 3. UNIFIED PLANNER EVALUATOR
# ==============================================================================

class UnifiedPlannerEvaluator:
    """
    Evaluator for UnifiedDiffusionPlanner checkpoints.
    
    Runs policy in closed-loop with DDIM sampling, records video with HUD overlay,
    and computes success metrics across multiple episodes.
    """
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Device: {self.device}")
        
        # 1. Load Policy from Checkpoint
        log.info(f"Loading checkpoint: {cfg.checkpoint}")
        self._load_policy(cfg.checkpoint)
        
        # 2. Initialize Environment
        log.info("Initializing PandaEnv...")
        self.env = PandaEnv(
            xml_path=cfg.env.xml_path,
            control_mode="delta",  # CRITICAL: Use delta mode for delta actions
            render_mode="rgb_array"
        )
        
        # 3. Initialize IK Solver (optional, for debugging)
        if cfg.get("use_ik", False):
            log.info("Initializing IK Solver...")
            self.ik_solver = IKSolver(urdf_path=cfg.env.urdf_path)
            
            # === DYNAMIC TUNING SUPPORT ===
            # Default to current "tuned" values if not specified
            ik_kp = getattr(cfg, "ik_kp", 139.0)
            ik_ki = getattr(cfg, "ik_ki", 0.1)
            ik_kd = getattr(cfg, "ik_kd", 3.0)
            
            log.info(f"Setting IK Solver Gains: Kp={ik_kp}, Ki={ik_ki}, Kd={ik_kd}")
            self.ik_solver.set_gains(kp=ik_kp, ki=ik_ki, kd=ik_kd)
        else:
            self.ik_solver = None
        
        # 4. Control calibration
        SIM_SUBSTEPS = 20
        self.effective_dt = self.env.model.opt.timestep * SIM_SUBSTEPS
        
        # 5. Image transform - MUST EXACTLY MATCH TRAINING!
        # From UnifiedPlannerDataset: Resize(224, BICUBIC) + ToTensor() + Normalize(0.5, 0.5) -> [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # 6. Output directory
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        log.info("Unified Planner Evaluator initialized.")
    
    def _load_policy(self, checkpoint_path: str):
        """Load UnifiedDiffusionPlanner from Lightning checkpoint."""
        log.info(f"Loading Lightning checkpoint: {checkpoint_path}")
        
        # Load the full Lightning module
        pl_module = UnifiedPlannerLightningModule.load_from_checkpoint(
            checkpoint_path,
            map_location=self.device,
            strict=False  # Allow missing keys (e.g., optimizer state)
        )
        
        # Extract the  UnifiedDiffusionPlanner model
        self.model: UnifiedDiffusionPlanner = pl_module.model.to(self.device)
        self.model.eval()
        
        log.info(f"Model loaded successfully")
        log.info(f"  Diffusion timesteps: {self.model.cfg.diffusion_timesteps}")
        log.info(f"  Default inference steps: {self.model.cfg.inference_steps}")
        log.info(f"  Default guidance scale: {self.model.cfg.guidance_scale}")
        
        # === DIAGNOSTIC: Print ActionNormalizer statistics ===
        normalizer = self.model.action_normalizer
        if normalizer.fitted:
            log.info("=== ActionNormalizer Diagnostics ===")
            log.info(f"  action_min: {normalizer.action_min.cpu().numpy()}")
            log.info(f"  action_max: {normalizer.action_max.cpu().numpy()}")
            log.info(f"  action_range: {(normalizer.action_max - normalizer.action_min).cpu().numpy()}")
        else:
            log.warning("ActionNormalizer is NOT fitted! This is a critical error.")
    
    def _prepare_batch(
        self,
        prev_img: np.ndarray,
        curr_img: np.ndarray,
        goal_img: np.ndarray,
        proprio: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare observation batch for policy inference.
        
        Args:
            prev_img: (H, W, 3) uint8 RGB
            curr_img: (H, W, 3) uint8 RGB
            goal_img: (H, W, 3) uint8 RGB
            proprio: (proprio_dim,) float
            
        Returns:
            batch: Dict with transformed tensors
        """
        prev_t = self.transform(Image.fromarray(prev_img)).unsqueeze(0).to(self.device)
        curr_t = self.transform(Image.fromarray(curr_img)).unsqueeze(0).to(self.device)
        goal_t = self.transform(Image.fromarray(goal_img)).unsqueeze(0).to(self.device)
        proprio_t = torch.from_numpy(proprio).float().unsqueeze(0).to(self.device)
        
        return {
            "prev_image": prev_t,
            "curr_image": curr_t,
            "goal_image": goal_t,
            "curr_proprio": proprio_t
        }
    
    def run_episode(
        self,
        episode_id: int,
        seed: int,
        video_writer: cv2.VideoWriter,
        logger: EvaluationLogger
    ) -> Tuple[bool, float]:
        """
        Run a single evaluation episode.
        
        Args:
            episode_id: Episode index
            seed: Random seed
            video_writer: OpenCV video writer
            logger: CSV logger
            
        Returns:
            (success, total_reward)
        """
        # Reset environment
        self.env.reset(seed=seed)
        obs = self.env.get_expert_obs()
        
        # Render goal image
        # Render goal image (with robot at goal if IK available)
        goal_img = render_goal_image(self.env, self.ik_solver, obs)
        prev_img = obs['image_primary'].copy()
        
        episode_success = False
        total_reward = 0.0
        success_steps = 0  # Count consecutive steps at goal
        
        log.info(f"=== Episode {episode_id} (seed={seed}) ===")
        
        for step in range(self.cfg.max_steps):
            curr_img = obs['image_primary']
            proprio = obs['proprio']
            
            # 1. Policy inference via DDIM sampling
            batch = self._prepare_batch(prev_img, curr_img, goal_img, proprio)
            
            with torch.no_grad():
                # DDIM sampling returns (B, K, 8) denormalized actions
                sampled_actions = self.model.sample(
                    batch,
                    num_steps=self.cfg.sampling.inference_steps,
                    guidance_scale=self.cfg.sampling.guidance_scale
                )
            
            # Extract first step of the action chunk
            # Shape: (1, K, 8) -> (8,)
            delta_action = sampled_actions[0, 0].cpu().numpy()  # (8,)
            delta_pose = delta_action[:7]  # (dx, dy, dz, dqx, dqy, dqz, dqw)
            
            # Apply action scaling to position deltas (diagnostic tuning)
            action_scale = getattr(self.cfg, 'action_scale', 1.0)
            
            delta_pose[:3] = delta_pose[:3] * action_scale  # Scale dx, dy, dz only
            
            # Safety clip gripper to [-1, 1] range
            gripper_cmd = float(np.clip(delta_action[7], -1.0, 1.0))
            
            # 2. Prepare action for environment
            # PandaEnv expects (8,) action: [7 joint deltas, 1 gripper]
            # Since we have delta pose, we can either:
            #   A) Use IK to convert delta pose -> delta joints (if ik_solver available)
            #   B) Assume env can handle delta pose directly (NOT STANDARD)
            # For robustness, we'll use approach A if IK is available
            
            if self.ik_solver is not None:
                # Convert delta pose to absolute target pose
                current_ee_pose = obs['ee_pose_world']
                # Apply delta to get target (using dataset utility)
                from utils.unified_planner_dataset import apply_delta_pose
                target_pose = apply_delta_pose(current_ee_pose, delta_pose)
                
                # === DIAGNOSTIC: Log every step ===
                # if step % 10 == 0: # Optional: Reduce spam if needed
                log.info(f"--- Step {step} ---")
                log.info(f"  Raw Delta Pose: {delta_pose}")
                log.info(f"  Current EE: {current_ee_pose[:3]}")
                log.info(f"  Target  EE: {target_pose[:3]}")
                
                # Direction check
                obj_pos = obs['object_pos_world']
                ee_to_obj = obj_pos - current_ee_pose[:3]
                dist_to_obj = np.linalg.norm(ee_to_obj)
                dot_product = np.dot(ee_to_obj / (dist_to_obj + 1e-8), delta_pose[:3] / (np.linalg.norm(delta_pose[:3]) + 1e-8))
                log.info(f"  Dist to Obj: {dist_to_obj:.4f} | Dot Prod: {dot_product:.4f}")
                
                # Compute delta joints via IK
                try:
                    delta_joints = self.ik_solver.compute_delta_action(
                        target_ee_pose=target_pose,
                        model=self.env.model,
                        data=self.env.data,
                        ee_site_id=self.env.ee_site_id,
                        joint_qpos_indices=np.arange(7),
                        effective_dt=self.effective_dt,
                        max_dq=self.env.ACTION_SCALING_FACTOR / self.effective_dt
                    )
                    # === DIAGNOSTIC: Log IK result ===
                    log.info(f"  IK Delta Joints: {delta_joints}")
                    log.info(f"  IK Magnitude: {np.linalg.norm(delta_joints):.4f}")
                except Exception as e:
                    log.warning(f"IK failed at step {step}: {e}")
                    delta_joints = np.zeros(7)
            else:
                # No fallback - IK is required to convert delta pose to joint deltas
                raise RuntimeError(
                    "IK solver is REQUIRED to convert delta poses to joint deltas! "
                    f"Delta poses are in Cartesian space but env expects joint space. "
                    f"Set use_ik: true in config file."
                )
            
            # CRITICAL FIX: Compensate for environment's internal scaling
            # The env applies: physical_delta = action * ACTION_SCALING_FACTOR
            # So we must send: action = desired_delta / ACTION_SCALING_FACTOR
            # CORRECTED LOGIC: IK Solver returns normalized action [-1, 1].
            # Do NOT divide by scaling factor again, or we get gain=45x (Teleportation).

            compensated_delta_joints = delta_joints


        #    compensated_delta_joints = delta_joints / self.env.ACTION_SCALING_FACTOR



            action = np.concatenate([compensated_delta_joints, [gripper_cmd]])
            
            # 3. Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            obs = self.env.get_expert_obs()
            total_reward += reward
            
            # 4. Compute metrics
            ee_pos = obs['ee_pose_world'][:3]
            obj_pos = obs['object_pos_world']
            goal_pos = obs['goal_pos_world']
            
            dist_ee_obj = np.linalg.norm(ee_pos - obj_pos)
            dist_obj_goal = np.linalg.norm(obj_pos - goal_pos)
            is_grasped = obs['is_grasped'][0] > 0.5
            
            # === DIAGNOSTIC: Log Post-Step ===
            log.info(f"  Compensated Action Sent: {compensated_delta_joints}")
            log.info(f"  EE after step: {ee_pos}")
            log.info(f"  Move Dist: {np.linalg.norm(ee_pos - obs['ee_pose_world'][:3]):.4f}") # Approx
            log.info(f"  Is Grasped: {is_grasped}")
            
            # Success check: object near goal and stable
            if dist_obj_goal < self.cfg.success_threshold:
                success_steps += 1
                if success_steps >= self.cfg.success_duration_steps:
                    episode_success = True
            else:
                success_steps = 0
            
            # 5. Log metrics
            logger.log_step({
                "episode_id": episode_id,
                "step": step,
                "time_sec": step * self.effective_dt,
                "ee_x": ee_pos[0], "ee_y": ee_pos[1], "ee_z": ee_pos[2],
                "obj_x": obj_pos[0], "obj_y": obj_pos[1], "obj_z": obj_pos[2],
                "goal_x": goal_pos[0], "goal_y": goal_pos[1], "goal_z": goal_pos[2],
                "dist_ee_obj": dist_ee_obj,
                "dist_obj_goal": dist_obj_goal,
                "is_grasped": float(is_grasped),
                "gripper_cmd": gripper_cmd,
                "diffusion_steps": self.cfg.sampling.inference_steps,
                "guidance_scale": self.cfg.sampling.guidance_scale,
                "policy_dx": delta_pose[0],
                "policy_dy": delta_pose[1],
                "policy_dz": delta_pose[2],
                "policy_grip": gripper_cmd,
                "success_flag": float(episode_success)
            })
            
            # 6. Render frame with HUD
            frame = self.env.render()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # HUD Overlay
            self._draw_hud(
                frame, episode_id, step,
                dist_obj_goal, is_grasped, gripper_cmd, episode_success
            )
            
            video_writer.write(frame)
            
            # Update state
            prev_img = curr_img.copy()
            
            # Check termination
            if episode_success or terminated or truncated:
                break
        
        # Add pause frames at end to show final state
        for _ in range(30):
            video_writer.write(frame)
        
        result = "SUCCESS ✓" if episode_success else "FAIL ✗"
        log.info(f"Episode {episode_id} Result: {result} | Steps: {step+1} | Dist to Goal: {dist_obj_goal:.3f}m")
        
        return episode_success, total_reward
    
    def _draw_hud(
        self,
        frame: np.ndarray,
        ep_id: int,
        step: int,
        dist_goal: float,
        is_grasped: bool,
        gripper_cmd: float,
        success: bool
    ):
        """
        Draw HUD overlay on video frame.
        
        Shows:
        - Episode ID and step count
        - Distance to goal (color-coded)
        - Gripper state
        - Grasp indicator
        - Success indicator
        - Diffusion info
        """
        h, w = frame.shape[:2]
        
        # Background bar
        cv2.rectangle(frame, (0, 0), (w, 90), (40, 40, 40), -1)
        
        # Row 1: Episode info
        cv2.putText(frame, f"Ep {ep_id} | Step {step}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Row 1 Right: Diffusion info
        ddim_txt = f"DDIM:{self.cfg.sampling.inference_steps} CFG:{self.cfg.sampling.guidance_scale:.1f}"
        cv2.putText(frame, ddim_txt,
                    (w - 180, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Row 2: Distance to goal (color-coded)
        dist_color = (0, 255, 0) if dist_goal < 0.05 else (0, 165, 255) if dist_goal < 0.1 else (0, 0, 255)
        cv2.putText(frame, f"Goal: {dist_goal*100:.1f}cm",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, dist_color, 1)
        
        # Row 2 Middle: Gripper state
        grip_txt = "CLOSED" if gripper_cmd < 0 else "OPEN"
        grip_color = (0, 255, 0) if gripper_cmd < 0 else (255, 255, 0)
        cv2.putText(frame, f"Grip: {grip_txt}",
                    (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, grip_color, 1)
        
        # Row 3: Grasp indicator
        grasp_txt = "GRASPED" if is_grasped else "NOT GRASPED"
        grasp_color = (0, 255, 0) if is_grasped else (128, 128, 128)
        cv2.putText(frame, grasp_txt,
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, grasp_color, 1)
        
        # Success indicator (large, top right)
        if success:
            cv2.putText(frame, "SUCCESS!",
                        (w - 140, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    def run(self):
        """Run full evaluation across multiple episodes."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_name = Path(self.cfg.checkpoint).stem
        
        # Output paths
        video_path = self.output_dir / f"unified_eval_{ckpt_name}_{timestamp}.mp4"
        csv_path = self.output_dir / f"unified_eval_{ckpt_name}_{timestamp}.csv"
        
        log.info(f"Video will be saved to: {video_path}")
        log.info(f"CSV will be saved to: {csv_path}")
        
        # Setup video writer
        frame = self.env.render()
        h, w, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, 30, (w, h))
        
        # Setup logger
        csv_logger = EvaluationLogger(csv_path)
        
        successes = []
        total_rewards = []
        
        try:
            for ep_idx in range(self.cfg.n_episodes):
                seed = self.cfg.seed + ep_idx
                success, reward = self.run_episode(ep_idx, seed, video_writer, csv_logger)
                successes.append(success)
                total_rewards.append(reward)
        
        finally:
            video_writer.release()
            csv_logger.close()
            self.env.close()
        
        # Summary
        success_rate = sum(successes) / len(successes) * 100 if successes else 0.0
        mean_reward = np.mean(total_rewards) if total_rewards else 0.0
        
        log.info("=" * 70)
        log.info("EVALUATION SUMMARY")
        log.info("=" * 70)
        log.info(f"Checkpoint: {self.cfg.checkpoint}")
        log.info(f"Episodes: {self.cfg.n_episodes}")
        log.info(f"Success Rate: {success_rate:.1f}%")
        log.info(f"Mean Reward: {mean_reward:.2f}")
        log.info(f"Successful Episodes: {sum(successes)}/{len(successes)}")
        log.info(f"Video: {video_path}")
        log.info(f"CSV: {csv_path}")
        log.info("=" * 70)
        
        return success_rate


# ==============================================================================
# 4. MAIN
# ==============================================================================

@hydra.main(version_base=None, config_path="../configs", config_name="eval_unified_planner_config")
def main(cfg: DictConfig):
    """Main entry point for evaluation."""
    log.info("=" * 70)
    log.info("UNIFIED DIFFUSION PLANNER EVALUATION")
    log.info("=" * 70)
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    evaluator = UnifiedPlannerEvaluator(cfg)
    success_rate = evaluator.run()
    
    return success_rate


if __name__ == "__main__":
    main()
