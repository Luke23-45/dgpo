# FILE: evaluate/evaluate_dgpo.py
"""
DGPO Policy Evaluation Script

This script evaluates a trained DGPO checkpoint (SemanticPlanner) by running it
in the PandaEnv and recording a video of the rollout.

Features:
- Loads DGPO checkpoint (policy_state_dict from .pt file)
- Runs closed-loop evaluation with the trained policy
- Records MP4 video with HUD overlay showing:
  - Current phase
  - Gripper state
  - Distance to goal
  - Success/Failure status
- Logs detailed metrics to CSV
- Supports multiple episodes for success rate calculation

Usage:
    python evaluate/evaluate_dgpo.py --checkpoint outputs/dgpo_runs/dgpo_iter_0100.pt --seed 42
    python evaluate/evaluate_dgpo.py --checkpoint outputs/dgpo_runs/dgpo_final.pt --n_episodes 5
"""

import argparse
import csv
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mujoco
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torchvision import transforms

# Project Imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.panda_env import PandaEnv
from models.semantic_planner import SemanticPlanner, SemanticPlannerConfig
from train.train_semantic_planner import SemanticPlannerLightningModule
from utils.ik_solver import IKSolver

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("DGPO_Eval")


# ==============================================================================
# 1. UTILITY: GOAL IMAGE RENDERING
# ==============================================================================

def render_goal_image(env: PandaEnv, goal_pos: np.ndarray) -> np.ndarray:
    """Renders the goal image by teleporting object to goal position."""
    # Save current state
    original_qpos = env.data.qpos.copy()
    original_qvel = env.data.qvel.copy()
    
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
    mujoco.mj_forward(env.model, env.data)
    
    return goal_image


# ==============================================================================
# 2. METRICS LOGGER
# ==============================================================================

class EvaluationLogger:
    """Logs evaluation metrics to CSV file."""
    
    HEADERS = [
        "episode_id", "step", "time_sec",
        "ee_x", "ee_y", "ee_z",
        "obj_x", "obj_y", "obj_z",
        "goal_x", "goal_y", "goal_z",
        "dist_ee_obj", "dist_obj_goal",
        "is_grasped", "gripper_cmd",
        "policy_pose_x", "policy_pose_y", "policy_pose_z",
        "success_flag"
    ]
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.file = open(filepath, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.HEADERS)
    
    def log_step(self, data: Dict):
        """Log a single step of evaluation."""
        row = [data.get(h, 0.0) for h in self.HEADERS]
        self.writer.writerow(row)
    
    def close(self):
        self.file.close()


# ==============================================================================
# 3. DGPO EVALUATOR
# ==============================================================================

class DGPOEvaluator:
    """
    Evaluator for DGPO-trained SemanticPlanner checkpoints.
    
    Runs policy in closed-loop, records video with HUD overlay,
    and computes success metrics.
    """
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Device: {self.device}")
        
        # 1. Load Policy from DGPO Checkpoint
        log.info(f"Loading DGPO checkpoint: {args.checkpoint}")
        self._load_policy(args.checkpoint, args.bc_checkpoint)
        
        # 2. Initialize Environment
        log.info("Initializing PandaEnv...")
        self.env = PandaEnv(
            xml_path=args.xml_path,
            control_mode="delta",
            render_mode="rgb_array"
        )
        
        # 3. Initialize IK Solver
        log.info("Initializing IK Solver...")
        self.ik_solver = IKSolver(urdf_path=args.urdf_path)
        
        # 4. Control calibration
        SIM_SUBSTEPS = 20
        self.effective_dt = self.env.model.opt.timestep * SIM_SUBSTEPS
        self.max_dq = self.env.ACTION_SCALING_FACTOR / self.effective_dt
        
        # 5. Image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor()
        ])
        
        # 6. Output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        log.info("DGPO Evaluator initialized.")
    
    def _load_policy(self, dgpo_checkpoint: str, bc_checkpoint: str):
        """Load policy weights from DGPO checkpoint."""
        # First load the BC checkpoint to get the model architecture
        log.info(f"Loading BC architecture from: {bc_checkpoint}")
        pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
            bc_checkpoint, map_location=self.device, strict=True
        )
        self.policy = pl_module.model.to(self.device)
        
        # Now load DGPO weights if provided
        if dgpo_checkpoint and Path(dgpo_checkpoint).exists():
            log.info(f"Loading DGPO weights from: {dgpo_checkpoint}")
            ckpt = torch.load(dgpo_checkpoint, map_location=self.device)
            
            if 'policy_state_dict' in ckpt:
                self.policy.load_state_dict(ckpt['policy_state_dict'])
                log.info(f"Loaded DGPO weights from iteration {ckpt.get('iteration', '?')}")
            elif 'model_state_dict' in ckpt:
                self.policy.load_state_dict(ckpt['model_state_dict'])
                log.info(f"Loaded DAgger/Model weights from iteration {ckpt.get('iteration', '?')}")
            else:
                log.warning("Checkpoint doesn't contain 'policy_state_dict' or 'model_state_dict'. Using BC weights.")
        else:
            log.warning(f"DGPO checkpoint not found: {dgpo_checkpoint}. Using BC weights.")
        
        self.policy.eval()
    
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
    
    def run_episode(
        self,
        episode_id: int,
        seed: int,
        video_writer: cv2.VideoWriter,
        logger: EvaluationLogger
    ) -> Tuple[bool, float]:
        """
        Run a single evaluation episode.
        
        Returns: (success, total_reward)
        """
        # Reset environment
        self.env.reset(seed=seed)
        obs = self.env.get_expert_obs()
        self.ik_solver.reset_controller_state()
        
        # Render goal image
        goal_img = render_goal_image(self.env, obs['goal_pos_world'])
        prev_img = obs['image_primary'].copy()
        
        episode_success = False
        total_reward = 0.0
        success_steps = 0  # Count consecutive steps at goal
        
        log.info(f"=== Episode {episode_id} (seed={seed}) ===")
        
        for step in range(self.args.max_steps):
            curr_img = obs['image_primary']
            proprio = obs['proprio']
            
            # 1. Policy inference
            batch = self._prepare_batch(prev_img, curr_img, goal_img, proprio)
            with torch.no_grad():
                policy_out = self.policy(batch)
            
            # Extract first step of chunk
            policy_pose = policy_out['pose_chunk'][0, 0].cpu().numpy()
            policy_grip_logit = policy_out['gripper_chunk'][0, 0].cpu().numpy()[0]
            gripper_cmd = -1.0 if policy_grip_logit > 0 else 1.0
            
            # 2. Compute action via IK
            try:
                delta_joints = self.ik_solver.compute_delta_action(
                    target_ee_pose=policy_pose,
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
            
            action = np.concatenate([delta_joints, [gripper_cmd]])
            
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
            
            # Success check: object near goal and stable
            if dist_obj_goal < 0.05:
                success_steps += 1
                if success_steps >= 10:  # Stable for 10 steps
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
                "policy_pose_x": policy_pose[0],
                "policy_pose_y": policy_pose[1],
                "policy_pose_z": policy_pose[2],
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
        
        # Add pause frames at end
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
        """Draw HUD overlay on video frame."""
        h, w = frame.shape[:2]
        
        # Background bar
        cv2.rectangle(frame, (0, 0), (w, 70), (40, 40, 40), -1)
        
        # Episode and step info
        cv2.putText(frame, f"Ep {ep_id} | Step {step}", 
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Distance to goal
        dist_color = (0, 255, 0) if dist_goal < 0.05 else (0, 165, 255) if dist_goal < 0.1 else (0, 0, 255)
        cv2.putText(frame, f"Goal: {dist_goal*100:.1f}cm", 
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, dist_color, 1)
        
        # Gripper state
        grip_txt = "CLOSED" if gripper_cmd < 0 else "OPEN"
        grip_color = (0, 255, 0) if gripper_cmd < 0 else (255, 255, 0)
        cv2.putText(frame, f"Grip: {grip_txt}", 
                    (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, grip_color, 1)
        
        # Grasp indicator
        grasp_txt = "GRASPED" if is_grasped else "NOT GRASPED"
        grasp_color = (0, 255, 0) if is_grasped else (128, 128, 128)
        cv2.putText(frame, grasp_txt, 
                    (150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, grasp_color, 1)
        
        # Success indicator
        if success:
            cv2.putText(frame, "SUCCESS!", 
                        (w - 100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def run(self):
        """Run full evaluation."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_name = Path(self.args.checkpoint).stem if self.args.checkpoint else "bc_only"
        
        # Output paths
        video_path = self.output_dir / f"dgpo_eval_{ckpt_name}_{timestamp}.mp4"
        csv_path = self.output_dir / f"dgpo_eval_{ckpt_name}_{timestamp}.csv"
        
        log.info(f"Video will be saved to: {video_path}")
        log.info(f"CSV will be saved to: {csv_path}")
        
        # Setup video writer
        frame = self.env.render()
        h, w, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, 30, (w, h))
        
        # Setup logger
        logger = EvaluationLogger(csv_path)
        
        successes = []
        total_rewards = []
        
        try:
            for ep_idx in range(self.args.n_episodes):
                seed = self.args.seed + ep_idx
                success, reward = self.run_episode(ep_idx, seed, video_writer, logger)
                successes.append(success)
                total_rewards.append(reward)
        
        finally:
            video_writer.release()
            logger.close()
            self.env.close()
        
        # Summary
        success_rate = sum(successes) / len(successes) * 100
        mean_reward = np.mean(total_rewards)
        
        log.info("=" * 60)
        log.info("EVALUATION SUMMARY")
        log.info("=" * 60)
        log.info(f"Checkpoint: {self.args.checkpoint}")
        log.info(f"Episodes: {self.args.n_episodes}")
        log.info(f"Success Rate: {success_rate:.1f}%")
        log.info(f"Mean Reward: {mean_reward:.2f}")
        log.info(f"Video: {video_path}")
        log.info(f"CSV: {csv_path}")
        log.info("=" * 60)
        
        return success_rate


# ==============================================================================
# 4. MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate DGPO-trained policy")
    
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to DGPO checkpoint (.pt file)"
    )
    parser.add_argument(
        "--bc_checkpoint", type=str, 
        default="/content/drive/MyDrive/pda/bc/bc_backup_epoch_088.ckpt",
        help="Path to BC checkpoint (for model architecture)"
    )
    parser.add_argument(
        "--xml_path", type=str, default="envs/panda_pick_place.xml",
        help="Path to MuJoCo XML file"
    )
    parser.add_argument(
        "--urdf_path", type=str, default="urdf/panda_mujoco_kinematics.urdf",
        help="Path to URDF file for IK"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/dgpo_eval",
        help="Directory to save evaluation outputs"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for first episode"
    )
    parser.add_argument(
        "--n_episodes", type=int, default=1,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--max_steps", type=int, default=800,
        help="Maximum steps per episode"
    )
    
    args = parser.parse_args()
    
    evaluator = DGPOEvaluator(args)
    evaluator.run()


if __name__ == "__main__":
    main()
