# FILE: evaluate/evaluate_unified_planner_auto.py
# Expert-to-Model Handoff Evaluation for Unified Diffusion Planner

"""
Phase-Based Hybrid Evaluation Script for UnifiedDiffusionPlanner.

This script implements a diagnostic evaluation approach where:
1. The **Expert (DGPOExpert)** controls the robot up to a specified handoff phase
2. The **Model (UnifiedDiffusionPlanner)** takes control from the handoff phase onwards

This allows systematic diagnosis of which phase the model fails at:
- If model fails at APPROACH: Fundamental vision/reaching issue
- If model fails at GRASP:   Precision positioning issue
- If model fails at LIFT:    Post-grasp control issue
- If model fails at PLACE:   Transport/navigation issue

Phases (from EXPERT_PHASE_MAP):
    0: REACH   (MOVE_TO_PRE_GRASP, PREPARE_GRIPPER, DESCEND_TO_GRASP)
    1: GRASP   (GRASP - gripper closure)
    2: LIFT    (LIFT, MOVE_TO_GOAL)
    3: PLACE   (PREPARE_PLACE, DESCEND_TO_PLACE, RELEASE)
    4: RETRACT (RETRACT, DONE)

Usage:
    # Expert controls until GRASP (phase 1), model takes over from LIFT:
    python evaluate/evaluate_unified_planner_auto.py \\
        --checkpoint /path/to/unified_planner.ckpt \\
        --handoff_phase 1 \\
        --n_episodes 5

    # Expert controls until LIFT (phase 2), model takes over from PLACE:
    python evaluate/evaluate_unified_planner_auto.py \\
        --handoff_phase 2
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
# FORCE EGL BACKEND FOR HEADLESS CO-LAB / LINUX RENDERING
# os.environ["MUJOCO_GL"] = "egl"

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
from utils.dgpo_expert import DGPOExpert, DGPOExpertConfig, ObjectProfile, EXPERT_PHASE_MAP
from utils.scripted_expert import ScriptedExpert
from utils.unified_planner_dataset import apply_delta_pose
from scipy.spatial.transform import Rotation as R

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("UnifiedPlanner_Auto")


# ==============================================================================
# PHASE NAMES FOR DISPLAY
# ==============================================================================
PHASE_NAMES = ["REACH", "GRASP", "LIFT", "PLACE", "RETRACT"]


# ==============================================================================
# 1. GOAL IMAGE RENDERING
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
    """CSV logger for evaluation metrics."""
    
    HEADERS = [
        "episode_id", "step", "time_sec",
        "controller",  # "EXPERT" or "MODEL"
        "expert_state", "phase",
        "ee_x", "ee_y", "ee_z",
        "obj_x", "obj_y", "obj_z",
        "goal_x", "goal_y", "goal_z",
        "dist_ee_obj", "dist_obj_goal",
        "is_grasped", "gripper_cmd",
        "policy_dx", "policy_dy", "policy_dz",
        "control_error_x", "control_error_y", "control_error_z",
        "handoff_phase", "handoff_step",
        "success_flag"
    ]
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.file = open(filepath, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.HEADERS)
        log.info(f"CSV logger initialized: {filepath}")
    
    def log_step(self, data: Dict):
        row = [data.get(h, 0.0) for h in self.HEADERS]
        self.writer.writerow(row)
    
    def close(self):
        self.file.close()
        log.info(f"CSV logger closed: {self.filepath}")


# ==============================================================================
# 3. HYBRID EVALUATOR (EXPERT -> MODEL HANDOFF)
# ==============================================================================
class HybridEvaluator:
    """
    Evaluator implementing Expert-to-Model handoff at specified phase.
    
    The Expert controls the robot until it reaches the handoff_phase.
    Then the Model takes over and attempts to complete the task.
    """
    
    def __init__(self, cfg: DictConfig, handoff_phase: int = 1):
        self.cfg = cfg
        self.handoff_phase = handoff_phase
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Device: {self.device}")
        log.info(f"Handoff Phase: {handoff_phase} ({PHASE_NAMES[handoff_phase]})")
        
        # 1. Load Policy
        log.info(f"Loading checkpoint: {cfg.checkpoint}")
        self._load_policy(cfg.checkpoint)
        
        # 2. Initialize Environment
        log.info("Initializing PandaEnv...")
        self.env = PandaEnv(
            xml_path=cfg.env.xml_path,
            control_mode="delta",
            render_mode="rgb_array"
        )
        
        # 3. Initialize IK Solver with tunable gains
        log.info("Initializing IK Solver...")
        self.ik_solver = IKSolver(urdf_path=cfg.env.urdf_path)
        
        # EXPERT-SPECIFIC PID GAINS (must use these when expert is controlling)
        self.expert_kp = 139.0
        self.expert_ki = 0.1
        self.expert_kd = 3.0
        
        # MODEL PID GAINS (use these when model takes over)
        self.model_kp = getattr(cfg, "ik_kp", 500.0)
        self.model_ki = getattr(cfg, "ik_ki", 0.5)
        self.model_kd = getattr(cfg, "ik_kd", 15.0)
        
        # Start with expert gains
        log.info(f"Expert IK Gains: Kp={self.expert_kp}, Ki={self.expert_ki}, Kd={self.expert_kd}")
        log.info(f"Model IK Gains: Kp={self.model_kp}, Ki={self.model_ki}, Kd={self.model_kd}")
        self.ik_solver.set_gains(kp=self.expert_kp, ki=self.expert_ki, kd=self.expert_kd)
        
        # 4. Initialize Expert
        log.info("Initializing DGPOExpert...")
        object_size = np.array([0.04, 0.04, 0.04])  # Default cube size
        object_profile = ObjectProfile(
            size=object_size,
            grasp_width_normalized=0.5
        )
        self.expert = DGPOExpert(object_profile=object_profile, cfg=DGPOExpertConfig())
        
        # 5. Control calibration
        SIM_SUBSTEPS = 20
        self.effective_dt = self.env.model.opt.timestep * SIM_SUBSTEPS
        
        # 6. Image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # 7. Output directory
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        log.info("HybridEvaluator initialized.")
    
    def _load_policy(self, checkpoint_path: str):
        """Load UnifiedDiffusionPlanner from Lightning checkpoint."""
        log.info(f"Loading Lightning checkpoint: {checkpoint_path}")
        
        pl_module = UnifiedPlannerLightningModule.load_from_checkpoint(
            checkpoint_path,
            map_location=self.device,
            strict=False
        )
        
        self.model: UnifiedDiffusionPlanner = pl_module.model.to(self.device)
        self.model.eval()
        
        log.info(f"Model loaded successfully")
        log.info(f"  Diffusion timesteps: {self.model.cfg.diffusion_timesteps}")
        log.info(f"  Inference steps: {self.model.cfg.inference_steps}")
    
    def _prepare_batch(
        self,
        prev_img: np.ndarray,
        curr_img: np.ndarray,
        goal_img: np.ndarray,
        proprio: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        """Prepare observation batch for policy inference."""
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
    
    def _get_expert_phase(self) -> int:
        """Get the current phase from expert state using EXPERT_PHASE_MAP."""
        expert_state = self.expert.get_state()
        return EXPERT_PHASE_MAP.get(expert_state, 0)
    
    def run_episode(
        self,
        episode_id: int,
        seed: int,
        video_writer: cv2.VideoWriter,
        logger: EvaluationLogger
    ) -> Tuple[bool, float, int]:
        """
        Run a single evaluation episode with expert-to-model handoff.
        
        Returns:
            (success, total_reward, handoff_step)
        """
        # Reset environment and expert
        self.env.reset(seed=seed)
        self.expert.reset()
        obs = self.env.get_expert_obs()
        
        # Render goal image
        goal_img = render_goal_image(self.env, self.ik_solver, obs)
        prev_img = obs['image_primary'].copy()
        
        episode_success = False
        total_reward = 0.0
        success_steps = 0
        handoff_step = -1  # Step at which handoff occurred
        controller = "EXPERT"  # Start with expert
        
        log.info(f"=== Episode {episode_id} (seed={seed}) ===")
        log.info(f"  Handoff will occur at phase {self.handoff_phase} ({PHASE_NAMES[self.handoff_phase]})")
        
        for step in range(self.cfg.max_steps):
            if step % 50 == 0:
                print(f"DEBUG: Episode {episode_id} | Step {step}/{self.cfg.max_steps} | Phase {self._get_expert_phase()}")

            curr_img = obs['image_primary']
            proprio = obs['proprio']
            
            # Get current phase from expert
            current_phase = self._get_expert_phase()
            expert_state = self.expert.get_state()
            
            # Check for handoff condition
            if controller == "EXPERT" and current_phase >= self.handoff_phase:
                controller = "MODEL"
                handoff_step = step
                # SWITCH TO MODEL PID GAINS
                self.ik_solver.set_gains(kp=self.model_kp, ki=self.model_ki, kd=self.model_kd)
                self.ik_solver.reset_controller_state()
                log.info(f"  [HANDOFF] Step {step}: Expert -> Model (Phase {current_phase}: {PHASE_NAMES[current_phase]})")
                log.info(f"  [HANDOFF] Switched to Model PID: Kp={self.model_kp}, Ki={self.model_ki}, Kd={self.model_kd}")
            
            try:
                # Initialize control variables
                delta_pose = np.zeros(7)
                gripper_cmd = 1.0
                target_pose = None
                control_error = np.zeros(3)
                
                if controller == "EXPERT":
                    # === EXPERT CONTROL ===
                    target_pose_7d, gripper_cmd, expert_info = self.expert.get_target_pose(obs)
                    target_pose = target_pose_7d
                    
                    # Convert target pose to delta action via IK
                    try:
                        delta_joints = self.ik_solver.compute_delta_action(
                            target_ee_pose=target_pose_7d,
                            model=self.env.model,
                            data=self.env.data,
                            ee_site_id=self.env.ee_site_id,
                            joint_qpos_indices=np.arange(7),
                            effective_dt=self.effective_dt,
                            max_dq=self.env.ACTION_SCALING_FACTOR / self.effective_dt
                        )
                    except Exception as e:
                        log.warning(f"Expert IK failed at step {step}: {e}")
                        delta_joints = np.zeros(7)
                    
                    # Compute control error
                    current_ee_pose = obs['ee_pose_world']
                    control_error = target_pose_7d[:3] - current_ee_pose[:3]
                    delta_pose[:3] = target_pose_7d[:3] - current_ee_pose[:3]
                    
                else:
                    # === MODEL CONTROL ===
                    batch = self._prepare_batch(prev_img, curr_img, goal_img, proprio)
                    
                    with torch.no_grad():
                        sampled_actions = self.model.sample(
                            batch,
                            num_steps=self.cfg.sampling.inference_steps,
                            guidance_scale=self.cfg.sampling.guidance_scale
                        )
                    
                    delta_action = sampled_actions[0, 0].cpu().numpy()
                    delta_pose = delta_action[:7]
                    
                    # Apply action scaling
                    action_scale = getattr(self.cfg, 'action_scale', 1.0)
                    delta_pose[:3] = delta_pose[:3] * action_scale
                    
                    gripper_cmd = float(np.clip(delta_action[7], -1.0, 1.0))
                    
                    # MOD: Safety Clip - Limit model-predicted delta to 5cm per step to prevent instability
                    max_model_delta = 0.05 
                    model_delta_clipped = np.clip(delta_pose[:3], -max_model_delta, max_model_delta)
                    
                    # Convert delta pose to absolute target
                    current_ee_pose = obs['ee_pose_world']
                    # Apply clipped translation
                    target_pose = current_ee_pose.copy()
                    target_pose[:3] += model_delta_clipped
                    # Use unclipped rotation (or apply safety logic if needed, but translation is the primary source of 'explosions')
                    target_pose[3:] = apply_delta_pose(current_ee_pose, delta_pose)[3:]
                    
                    # Convert to joint deltas via IK
                    if getattr(self.cfg, "teleport", False):
                         # --- TELEPORT MODE (DIAGNOSTIC) ---
                         # Bypasses velocity control dynamics. Directly solves static IK and sets state.
                         q_sol = self.ik_solver.solve_ik_static(
                             target_pose=target_pose,
                             model=self.env.model,
                             data=self.env.data,
                             ee_site_id=self.env.ee_site_id,
                             q0=self.env.data.qpos[:7]
                         )
                         if q_sol is not None:
                             # DIRECT STATE SET
                             self.env.data.qpos[:7] = q_sol
                             self.env.data.qvel[:7] = 0.0 # Stop momentum
                             delta_joints = np.zeros(7) # No action needed for step, we moved already
                         else:
                             log.warning(f"Teleport IK failed at step {step}")
                             delta_joints = np.zeros(7)
                    else:
                        # --- DYNAMIC CONTROL MODE (STANDARD) ---
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
                        except Exception as e:
                            log.warning(f"Model IK failed at step {step}: {e}")
                            delta_joints = np.zeros(7)
                    
                    # Compute control error
                    control_error = target_pose[:3] - current_ee_pose[:3]
                
                # Step environment
                action = np.concatenate([delta_joints, [gripper_cmd]])
                obs, reward, terminated, truncated, info = self.env.step(action)
                obs = self.env.get_expert_obs()
                total_reward += reward

            except Exception as e:
                import traceback
                traceback.print_exc()
                log.error(f"CRITICAL ERROR at Step {step}: {e}")
                break
            
            # Compute metrics
            ee_pos = obs['ee_pose_world'][:3]
            obj_pos = obs['object_pos_world']
            goal_pos = obs['goal_pos_world']
            
            dist_ee_obj = np.linalg.norm(ee_pos - obj_pos)
            dist_obj_goal = np.linalg.norm(obj_pos - goal_pos)
            is_grasped = obs['is_grasped'][0] > 0.5
            
            # Success check
            if dist_obj_goal < self.cfg.success_threshold:
                success_steps += 1
                if success_steps >= self.cfg.success_duration_steps:
                    episode_success = True
            else:
                success_steps = 0
            
            # Log metrics
            logger.log_step({
                "episode_id": episode_id,
                "step": step,
                "time_sec": step * self.effective_dt,
                "controller": controller,
                "expert_state": expert_state,
                "phase": current_phase,
                "ee_x": ee_pos[0], "ee_y": ee_pos[1], "ee_z": ee_pos[2],
                "obj_x": obj_pos[0], "obj_y": obj_pos[1], "obj_z": obj_pos[2],
                "goal_x": goal_pos[0], "goal_y": goal_pos[1], "goal_z": goal_pos[2],
                "dist_ee_obj": dist_ee_obj,
                "dist_obj_goal": dist_obj_goal,
                "is_grasped": float(is_grasped),
                "gripper_cmd": gripper_cmd,
                "policy_dx": delta_pose[0],
                "policy_dy": delta_pose[1],
                "policy_dz": delta_pose[2],
                "control_error_x": control_error[0],
                "control_error_y": control_error[1],
                "control_error_z": control_error[2],
                "handoff_phase": self.handoff_phase,
                "handoff_step": handoff_step,
                "success_flag": float(episode_success)
            })
            
            # Render frame with HUD
            frame = self.env.render()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            self._draw_hud(
                frame, episode_id, step,
                dist_obj_goal, is_grasped, gripper_cmd, episode_success,
                controller, current_phase, expert_state, control_error
            )
            
            video_writer.write(frame)
            
            # Update state
            prev_img = curr_img.copy()
            
            # Check termination
            if episode_success or terminated or truncated or self.expert.is_done():
                break
        
        # Add pause frames at end
        for _ in range(30):
            video_writer.write(frame)
        
        result = "SUCCESS ✓" if episode_success else "FAIL ✗"
        log.info(f"Episode {episode_id} Result: {result} | Steps: {step+1} | "
                 f"Handoff at step {handoff_step} | Dist to Goal: {dist_obj_goal:.3f}m")
        
        return episode_success, total_reward, handoff_step
    
    def _draw_hud(
        self,
        frame: np.ndarray,
        ep_id: int,
        step: int,
        dist_goal: float,
        is_grasped: bool,
        gripper_cmd: float,
        success: bool,
        controller: str,
        phase: int,
        expert_state: str,
        control_error: np.ndarray = np.zeros(3)
    ):
        """Draw HUD overlay on video frame."""
        h, w = frame.shape[:2]
        
        # Background bar
        cv2.rectangle(frame, (0, 0), (w, 100), (40, 40, 40), -1)
        
        # Row 1: Episode info
        cv2.putText(frame, f"Ep {ep_id} | Step {step}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Row 1 Right: Controller indicator
        ctrl_color = (0, 255, 255) if controller == "EXPERT" else (255, 165, 0)
        cv2.putText(frame, f"CTRL: {controller}",
                    (w - 140, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ctrl_color, 1)
        
        # Row 2: Phase and Expert State
        phase_txt = PHASE_NAMES[min(phase, 4)]
        cv2.putText(frame, f"Phase: {phase_txt} | {expert_state}",
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Row 2 Right: Handoff info
        cv2.putText(frame, f"Handoff@{PHASE_NAMES[self.handoff_phase]}",
                    (w - 160, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 200), 1)
        
        # Row 3: Distance to goal
        dist_color = (0, 255, 0) if dist_goal < 0.05 else (0, 165, 255) if dist_goal < 0.1 else (0, 0, 255)
        cv2.putText(frame, f"Goal: {dist_goal*100:.1f}cm",
                    (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dist_color, 1)
        
        # Row 3 Middle: Gripper state
        grip_txt = "CLOSED" if gripper_cmd < 0 else "OPEN"
        grip_color = (0, 255, 0) if gripper_cmd < 0 else (255, 255, 0)
        cv2.putText(frame, f"Grip: {grip_txt}",
                    (150, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, grip_color, 1)
        
        # Row 3 Right: Control Error Z
        err_z = control_error[2]
        err_color = (0, 0, 255) if abs(err_z) > 0.05 else (0, 255, 0)
        cv2.putText(frame, f"CtrlErrZ: {err_z*100:.1f}cm",
                    (w - 160, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.45, err_color, 1)
        
        # Row 4: Grasp indicator
        grasp_txt = "GRASPED" if is_grasped else "NOT GRASPED"
        grasp_color = (0, 255, 0) if is_grasped else (128, 128, 128)
        cv2.putText(frame, grasp_txt,
                    (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.5, grasp_color, 1)
        
        # Success indicator
        if success:
            cv2.putText(frame, "SUCCESS!",
                        (w - 120, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def run(self) -> Dict:
        """Run full evaluation across multiple episodes."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use experiment_name if provided (for flat grid search output)
        if hasattr(self.cfg, "experiment_name"):
            filename_base = f"{self.cfg.experiment_name}"
            # Ensure unique timestamp is appended if not running grid search to avoid overwrites
            # But for grid search we often want the name to be exactly as specified
            # Let's append timestamp to be safe but keep it readable
            video_path = self.output_dir / f"{filename_base}.mp4"
            csv_path = self.output_dir / f"{filename_base}.csv"
        else:
            video_path = self.output_dir / f"hybrid_eval_handoff{self.handoff_phase}_{timestamp}.mp4"
            csv_path = self.output_dir / f"hybrid_eval_handoff{self.handoff_phase}_{timestamp}.csv"
        
        # Initialize video writer
        first_obs = self.env.get_expert_obs()
        sample_frame = first_obs['image_primary']
        h, w = sample_frame.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, 30, (w, h))
        
        # Initialize logger
        logger = EvaluationLogger(csv_path)
        
        log.info(f"Starting evaluation with handoff at phase {self.handoff_phase} ({PHASE_NAMES[self.handoff_phase]})")
        log.info(f"  Episodes: {self.cfg.n_episodes}")
        log.info(f"  Max steps: {self.cfg.max_steps}")
        log.info(f"  Video: {video_path}")
        log.info(f"  CSV: {csv_path}")
        
        results = []
        
        for ep_id in range(self.cfg.n_episodes):
            seed = self.cfg.seed + ep_id
            success, reward, handoff_step = self.run_episode(ep_id, seed, video_writer, logger)
            results.append({
                "episode": ep_id,
                "success": success,
                "reward": reward,
                "handoff_step": handoff_step
            })
        
        # Cleanup
        video_writer.release()
        logger.close()
        
        # Calculate statistics
        successes = sum(1 for r in results if r['success'])
        success_rate = 100.0 * successes / len(results) if results else 0.0
        
        log.info("=" * 60)
        log.info(f"EVALUATION COMPLETE")
        log.info(f"  Handoff Phase: {self.handoff_phase} ({PHASE_NAMES[self.handoff_phase]})")
        log.info(f"  Success Rate: {success_rate:.1f}% ({successes}/{len(results)})")
        log.info(f"  Video: {video_path}")
        log.info(f"  CSV: {csv_path}")
        log.info("=" * 60)
        
        return {
            "handoff_phase": self.handoff_phase,
            "success_rate": success_rate,
            "results": results
        }


# ==============================================================================
# 4. MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Expert-to-Model Handoff Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to unified planner checkpoint")
    parser.add_argument("--config", type=str, default="configs/eval_unified_planner_config.yaml",
                        help="Path to config file")
    parser.add_argument("--handoff_phase", type=int, default=1, choices=[0, 1, 2, 3],
                        help="Phase at which to handoff from expert to model (0=REACH, 1=GRASP, 2=LIFT, 3=PLACE)")
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/hybrid_eval")
    parser.add_argument("--ik_kp", type=float, default=100.0)
    parser.add_argument("--ik_ki", type=float, default=0.5)
    parser.add_argument("--ik_kd", type=float, default=15.0, help="IK D Gain")
    parser.add_argument("--action_scale", type=float, default=0.05, help="Policy action scale (Default: 5cm)")
    parser.add_argument("--teleport", action="store_true", help="DIAGNOSTIC: Teleport robot to model target (bypassing dynamics)")
    
    args = parser.parse_args()
    
    # Load or create config
    config_path = ROOT / args.config
    if config_path.exists():
        cfg = OmegaConf.load(config_path)
    else:
        log.warning(f"Config not found, creating minimal config")
        cfg = OmegaConf.create({
            "env": {
                "xml_path": "envs/panda_pick_place.xml",
                "urdf_path": "urdf/panda_mujoco_kinematics.urdf"
            },
            "sampling": {
                "inference_steps": 10,
                "guidance_scale": 1.5
            },
            "success_threshold": 0.05,
            "success_duration_steps": 10
        })
    
    # Override with command line args
    cfg.checkpoint = args.checkpoint
    cfg.n_episodes = args.n_episodes
    cfg.max_steps = args.max_steps
    cfg.seed = args.seed
    cfg.output_dir = args.output_dir
    cfg.ik_kp = args.ik_kp
    cfg.ik_ki = args.ik_ki
    cfg.ik_kd = args.ik_kd
    cfg.action_scale = args.action_scale
    cfg.teleport = args.teleport  # NEW: Pass teleport flag
    
    # Run evaluation
    evaluator = HybridEvaluator(cfg, handoff_phase=args.handoff_phase)
    results = evaluator.run()
    
    return results


if __name__ == "__main__":
    main()
