# FILE: evaluate/evaluate_semantic_planner_v2.py
# (Final, Production-Grade Evaluation with Tunable IK PID)

"""
Semantic Planner Evaluation Script (v2.0 - Final)

This script evaluates a trained SemanticPlanner checkpoint by running it
in the PandaEnv with a properly tuned IK-based controller.

Key Improvements over v1:
- Uses IK solver with TUNABLE PID gains (Kp, Ki, Kd)
- No trajectory smoothing or hysteresis hacks
- Clean architecture based on proven unified planner approach
- Detailed CSV telemetry and video recording with HUD

Architecture:
1. SemanticPlanner outputs absolute pose chunks (B, K, 7)
2. We take the first pose from the chunk as the immediate target
3. IK solver with PID controller computes joint velocities
4. Robot executes the action and we observe the result

Usage:
    python evaluate/evaluate_semantic_planner_v2.py
    
    # With custom config overrides:
    python evaluate/evaluate_semantic_planner_v2.py ik_kp=500 ik_kd=20 n_episodes=5
"""

from __future__ import annotations

import csv
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

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
from train.train_semantic_planner import SemanticPlannerLightningModule
from models.semantic_planner import SemanticPlanner
from utils.ik_solver import IKSolver

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("SemanticPlanner_Eval_v2")


# ==============================================================================
# 1. UTILITY: GOAL IMAGE RENDERING
# ==============================================================================

def render_goal_image(env: PandaEnv, goal_pos: np.ndarray) -> np.ndarray:
    """
    Renders the goal image by teleporting object to goal position.
    
    Args:
        env: PandaEnv instance
        goal_pos: (3,) world position of goal
        
    Returns:
        goal_image: (H, W, 3) RGB image
    """
    # Save current state
    original_qpos = env.data.qpos.copy()
    original_qvel = env.data.qvel.copy()
    original_ctrl = env.data.ctrl.copy()
    
    # Move object to goal
    obj_joint_adr = env.model.jnt_qposadr[env.object_joint_id]
    env.data.qpos[obj_joint_adr:obj_joint_adr + 3] = goal_pos
    env.data.qvel[:] = 0
    
    # Forward and render
    mujoco.mj_forward(env.model, env.data)
    goal_image = env.render()
    
    # Restore state
    env.data.qpos[:] = original_qpos
    env.data.qvel[:] = original_qvel
    env.data.ctrl[:] = original_ctrl
    mujoco.mj_forward(env.model, env.data)
    
    return goal_image


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
    - Model predictions
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
        "is_grasped", "gripper_prob", "gripper_cmd",
        # Model Predictions (target pose)
        "target_x", "target_y", "target_z",
        # Predicted Phase
        "pred_phase",
        # Control Output
        "joint_vel_norm",
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
        row = []
        for h in self.HEADERS:
            val = data.get(h, 0.0)
            if isinstance(val, float):
                row.append(f"{val:.6f}")
            else:
                row.append(val)
        self.writer.writerow(row)
    
    def close(self):
        self.file.close()
        log.info(f"CSV logger closed: {self.filepath}")


# ==============================================================================
# 3. SEMANTIC PLANNER EVALUATOR
# ==============================================================================

class SemanticPlannerEvaluator:
    """
    Evaluator for SemanticPlanner checkpoints (v2.0).
    
    Key Features:
    - Tunable IK PID gains via config
    - Clean closed-loop control without hacks
    - Detailed telemetry logging
    - Video recording with HUD overlay
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
            control_mode="delta",
            render_mode="rgb_array"
        )
        
        # 3. Initialize IK Solver with TUNABLE PID GAINS
        log.info("Initializing IK Solver...")
        self.ik_solver = IKSolver(urdf_path=cfg.env.urdf_path)
        
        # Apply tunable gains from config
        ik_kp = cfg.get("ik_kp", 400.0)
        ik_ki = cfg.get("ik_ki", 0.1)
        ik_kd = cfg.get("ik_kd", 20.0)
        
        log.info(f"Setting IK Solver Gains: Kp={ik_kp}, Ki={ik_ki}, Kd={ik_kd}")
        self.ik_solver.set_gains(kp=ik_kp, ki=ik_ki, kd=ik_kd)
        
        # 4. Control calibration
        SIM_SUBSTEPS = 20
        self.effective_dt = self.env.model.opt.timestep * SIM_SUBSTEPS
        log.info(f"Control dt: {self.effective_dt:.4f}s")
        
        # 5. Image transform - MUST EXACTLY MATCH TRAINING!
        # SemanticPlannerDataset uses: Resize(224) + ToTensor()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor()
        ])
        
        # 6. Output directory
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        log.info("Semantic Planner Evaluator v2.0 initialized.")
    
    def _load_policy(self, checkpoint_path: str):
        """Load SemanticPlanner from Lightning checkpoint."""
        log.info(f"Loading Lightning checkpoint: {checkpoint_path}")
        
        # Load the full Lightning module
        pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
            checkpoint_path,
            map_location=self.device,
            strict=False
        )
        
        # Extract the SemanticPlanner model
        self.model: SemanticPlanner = pl_module.model.to(self.device)
        self.model.eval()
        
        # Store training config for reference
        self.train_cfg = pl_module.cfg
        
        log.info(f"Model loaded successfully")
        log.info(f"  Chunk size: {self.model.cfg.chunk_size}")
        log.info(f"  Num phases: {self.model.cfg.num_task_phases}")
    
    def _prepare_batch(
        self,
        prev_img: np.ndarray,
        curr_img: np.ndarray,
        goal_img: np.ndarray,
        proprio: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare observation batch for policy inference.
        
        The SemanticPlanner expects:
        - prev_image: (B, 3, 224, 224)
        - curr_image: (B, 3, 224, 224)
        - goal_image: (B, 3, 224, 224)
        - curr_proprio: (B, proprio_dim)
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
        
        Returns:
            (success, total_reward)
        """
        # Reset environment and IK controller state
        self.env.reset(seed=seed)
        obs = self.env.get_expert_obs()
        self.ik_solver.reset_controller_state()
        
        # Render goal image
        goal_img = render_goal_image(self.env, obs['goal_pos_world'])
        prev_img = obs['image_primary'].copy()
        
        episode_success = False
        total_reward = 0.0
        success_steps = 0
        
        log.info(f"=== Episode {episode_id} (seed={seed}) ===")
        
        for step in range(self.cfg.max_steps):
            curr_img = obs['image_primary']
            proprio = obs['proprio']
            
            # 1. Policy inference
            batch = self._prepare_batch(prev_img, curr_img, goal_img, proprio)
            
            with torch.no_grad():
                outputs = self.model(batch)
            
            # SemanticPlanner outputs:
            # - pose_chunk: (B, K, 7) - absolute target poses
            # - gripper_chunk: (B, K, 1) - gripper logits
            # - phase_logits: (B, N_phases) - phase classification
            
            pose_chunk = outputs['pose_chunk']      # (1, K, 7)
            gripper_chunk = outputs['gripper_chunk'] # (1, K, 1)
            phase_logits = outputs['phase_logits']   # (1, N_phases)
            
            # Take the FIRST step from the chunk as immediate target
            target_pose = pose_chunk[0, 0, :].cpu().numpy()  # (7,)
            gripper_logit = gripper_chunk[0, 0, 0].item()
            pred_phase = torch.argmax(phase_logits, dim=1).item()
            
            # Convert gripper logit to command
            gripper_prob = torch.sigmoid(torch.tensor(gripper_logit)).item()
            gripper_cmd = -1.0 if gripper_prob > 0.5 else 1.0  # -1=close, 1=open
            
            # 2. Compute delta joints via IK with tuned PID controller
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
                log.warning(f"IK failed at step {step}: {e}")
                delta_joints = np.zeros(7)
            
            # 3. Execute action
            action = np.concatenate([delta_joints, [gripper_cmd]])
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
            
            # Success check: object near goal and grasped
            if dist_obj_goal < self.cfg.success_threshold and is_grasped:
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
                "gripper_prob": gripper_prob,
                "gripper_cmd": gripper_cmd,
                "target_x": target_pose[0],
                "target_y": target_pose[1],
                "target_z": target_pose[2],
                "pred_phase": pred_phase,
                "joint_vel_norm": np.linalg.norm(delta_joints),
                "success_flag": float(episode_success)
            })
            
            # 6. Render frame with HUD
            frame = self.env.render()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self._draw_hud(
                frame, episode_id, step,
                dist_ee_obj, dist_obj_goal, is_grasped, 
                gripper_cmd, pred_phase, episode_success
            )
            video_writer.write(frame)
            
            # Debug logging every 50 steps
            if step % 50 == 0:
                log.info(f"  Step {step}: dist_ee_obj={dist_ee_obj:.3f}m, "
                        f"dist_obj_goal={dist_obj_goal:.3f}m, "
                        f"grasped={is_grasped}, phase={pred_phase}")
            
            # Update state
            prev_img = curr_img.copy()
            
            # Check termination
            if episode_success or terminated or truncated:
                break
        
        # Add pause frames at end
        for _ in range(30):
            video_writer.write(frame)
        
        result = "SUCCESS ✓" if episode_success else "FAIL ✗"
        log.info(f"Episode {episode_id} Result: {result} | Steps: {step+1} | "
                f"Dist to Goal: {dist_obj_goal:.3f}m")
        
        return episode_success, total_reward
    
    def _draw_hud(
        self,
        frame: np.ndarray,
        ep_id: int,
        step: int,
        dist_ee_obj: float,
        dist_goal: float,
        is_grasped: bool,
        gripper_cmd: float,
        phase: int,
        success: bool
    ):
        """Draw HUD overlay on video frame."""
        h, w = frame.shape[:2]
        phase_names = ["REACH", "GRASP", "LIFT", "MOVE", "PLACE"]
        
        # Background bar
        cv2.rectangle(frame, (0, 0), (w, 100), (40, 40, 40), -1)
        
        # Row 1: Episode info
        cv2.putText(frame, f"Ep {ep_id} | Step {step}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Row 1 Right: Phase
        phase_txt = phase_names[min(phase, 4)]
        cv2.putText(frame, f"Phase: {phase_txt}",
                    (w - 150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Row 2: Distances
        ee_color = (0, 255, 0) if dist_ee_obj < 0.05 else (0, 165, 255) if dist_ee_obj < 0.1 else (0, 0, 255)
        cv2.putText(frame, f"EE->Obj: {dist_ee_obj*100:.1f}cm",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ee_color, 1)
        
        goal_color = (0, 255, 0) if dist_goal < 0.05 else (0, 165, 255) if dist_goal < 0.1 else (0, 0, 255)
        cv2.putText(frame, f"Obj->Goal: {dist_goal*100:.1f}cm",
                    (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, goal_color, 1)
        
        # Row 3: Gripper and Grasp
        grip_txt = "CLOSED" if gripper_cmd < 0 else "OPEN"
        grip_color = (0, 255, 0) if gripper_cmd < 0 else (255, 255, 0)
        cv2.putText(frame, f"Grip: {grip_txt}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, grip_color, 1)
        
        grasp_txt = "GRASPED" if is_grasped else "NOT GRASPED"
        grasp_color = (0, 255, 0) if is_grasped else (128, 128, 128)
        cv2.putText(frame, grasp_txt,
                    (150, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, grasp_color, 1)
        
        # Success indicator
        if success:
            cv2.putText(frame, "SUCCESS!",
                        (w - 140, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # PID Gains (for reference)
        gains_txt = f"Kp={self.cfg.get('ik_kp', 400):.0f}"
        cv2.putText(frame, gains_txt,
                    (w - 100, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    def run(self):
        """Run full evaluation across multiple episodes."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_name = Path(self.cfg.checkpoint).stem
        
        # Output paths
        video_path = self.output_dir / f"semantic_eval_{ckpt_name}_{timestamp}.mp4"
        csv_path = self.output_dir / f"semantic_eval_{ckpt_name}_{timestamp}.csv"
        
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
        log.info(f"IK Gains: Kp={self.cfg.get('ik_kp', 400)}, Ki={self.cfg.get('ik_ki', 0.1)}, Kd={self.cfg.get('ik_kd', 20)}")
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

@hydra.main(version_base=None, config_path="../configs", config_name="eval_semantic_planner_v2_config")
def main(cfg: DictConfig):
    """Main entry point for evaluation."""
    log.info("=" * 70)
    log.info("SEMANTIC PLANNER EVALUATION (v2.0 - Final)")
    log.info("=" * 70)
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    evaluator = SemanticPlannerEvaluator(cfg)
    success_rate = evaluator.run()
    
    return success_rate


if __name__ == "__main__":
    main()
