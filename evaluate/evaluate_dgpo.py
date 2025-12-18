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
from utils.scripted_expert import ScriptedExpert, ObjectProfile

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("DGPO_Eval")


# ==============================================================================
# 1. UTILITY: GOAL IMAGE RENDERING
# ==============================================================================

def render_goal_image(env: PandaEnv, ik_solver: IKSolver, obs: Dict) -> np.ndarray:
    """
    Renders the goal image by determining the Robot's Hover Pose using Inverse Kinematics,
    matching the Expert Policy's Retract State.
    
    Includes CRITICAL fixes for:
    - Object sinking (lifting by half-height 0.02)
    - Robot height relative to object center (lifting by 0.02)
    - Dynamic Gripper Orientation Alignment (matching Expert [1,0,0,0] reference)
    
    Args:
        env: PandaEnv instance
        ik_solver: IKSolver instance used to calculate the robot's pose
        obs: Observation dictionary containing goal_pos_world and goal_orn_world
        
    Returns:
        goal_image: (H, W, 3) RGB image
    """
    # Save current state
    original_qpos = env.data.qpos.copy()
    original_qvel = env.data.qvel.copy()
    original_ctrl = env.data.ctrl.copy()
    
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
        
        # 2. Calculate Robot Target Pose (Goal Pos + Hover Z)
        # FIX 2: Also lift robot by half-height (0.02) so it hovers relative to 
        # the object's center, not the table surface.
        target_pos = goal_pos_world + np.array([0.0, 0.0, HOVER_HEIGHT + 0.02])
        
        # 3. Calculate DYNAMIC Target Orientation
        # The expert aligns with the goal object. We calculate this alignment relative
        # to the goal orientation we just retrieved.
        # FIX 3: Expert uses [1, 0, 0, 0] (Rot X 180) as base. [0, 1, 0, 0] causes twisted arm.
        seed_downward_quat = np.array([1.0, 0.0, 0.0, 0.0])
        target_quat = dummy_expert._calculate_aligned_orientation(goal_orn_world, seed_downward_quat)
        
        target_pose_7d = np.concatenate([target_pos, target_quat])
        
        # 4. Solve IK for Hover Pose
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
            # Fallback (should ideally not happen with correct seed/pose)
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
        env.data.ctrl[:] = original_ctrl # Original code didn't save ctrl, but ground truth does. Safe to add.
        mujoco.mj_forward(env.model, env.data)


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
        "target_obj_dx", "target_obj_dy", "target_obj_dz",  # Model bias diagnostics
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
        
        # 3. Initialize IK Solver with tunable PID gains
        log.info(f"Initializing IK Solver with PID: Kp={args.ik_kp}, Ki={args.ik_ki}, Kd={args.ik_kd}")
        self.ik_solver = IKSolver(
            urdf_path=args.urdf_path,
            kp=args.ik_kp,
            ki=args.ik_ki,
            kd=args.ik_kd
        )
        
        # 4. Control calibration
        SIM_SUBSTEPS = 20
        self.effective_dt = self.env.model.opt.timestep * SIM_SUBSTEPS
        self.max_dq = self.env.ACTION_SCALING_FACTOR / self.effective_dt
        
        # 5. Image transform - MUST EXACTLY MATCH BC TRAINING!
        # BC uses: Resize(224, BICUBIC) + ToTensor() + Normalize(0.5, 0.5) -> [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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
        goal_img = render_goal_image(self.env, self.ik_solver, obs)
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
            
            # Compute model-vs-object diagnostic (key for detecting bias)
            target_obj_dx = policy_pose[0] - obj_pos[0]
            target_obj_dy = policy_pose[1] - obj_pos[1]
            target_obj_dz = policy_pose[2] - obj_pos[2]
            
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
                "target_obj_dx": target_obj_dx,
                "target_obj_dy": target_obj_dy,
                "target_obj_dz": target_obj_dz,
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
    
    # IK PID Tuning Arguments (use optimal values from grid search)
    parser.add_argument(
        "--ik_kp", type=float, default=500.0,
        help="IK solver proportional gain (default: 500 from grid search)"
    )
    parser.add_argument(
        "--ik_ki", type=float, default=0.5,
        help="IK solver integral gain (default: 0.5 from grid search)"
    )
    parser.add_argument(
        "--ik_kd", type=float, default=15.0,
        help="IK solver derivative gain (default: 15 from grid search)"
    )
    
    args = parser.parse_args()
    
    evaluator = DGPOEvaluator(args)
    evaluator.run()


if __name__ == "__main__":
    main()
