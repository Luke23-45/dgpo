# FILE: evaluate/evaluate_hybrid.py
"""
Hybrid Policy-Expert Evaluation Script

This diagnostic script tests whether the trained policy can handle LIFT and PLACE
phases if the expert handles the initial APPROACH and GRASP phases.

Flow:
1. EXPERT handles: moving to object and grasping it
2. POLICY takes over: lifting, moving to goal, and placing

This helps diagnose whether the policy has learned post-grasp behavior even if
it struggles with the initial approach phase.

Usage:
    python evaluate/evaluate_hybrid.py --checkpoint outputs/dagger_runs/dagger_iter_0009.pt
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
from utils.scripted_expert import ScriptedExpert, ExpertConfig, ObjectProfile

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("Hybrid_Eval")


# ==============================================================================
# 1. UTILITY: GOAL IMAGE RENDERING
# ==============================================================================

def render_goal_image(env: PandaEnv, goal_pos: np.ndarray) -> np.ndarray:
    """Renders the goal image by teleporting object to goal position."""
    original_qpos = env.data.qpos.copy()
    original_qvel = env.data.qvel.copy()
    
    obj_joint_adr = env.model.jnt_qposadr[env.object_joint_id]
    env.data.qpos[obj_joint_adr:obj_joint_adr + 3] = goal_pos
    env.data.qvel[:] = 0
    
    mujoco.mj_forward(env.model, env.data)
    goal_image = env.render()
    
    env.data.qpos[:] = original_qpos
    env.data.qvel[:] = original_qvel
    mujoco.mj_forward(env.model, env.data)
    
    return goal_image


# ==============================================================================
# 2. HYBRID EVALUATOR
# ==============================================================================

class HybridEvaluator:
    """
    Hybrid Evaluator: Expert handles grasp, Policy handles post-grasp.
    
    This diagnostic tool helps determine if the policy learned lift/place behavior
    even if it fails at approach/grasp.
    """
    
    # Control mode: EXPERT or POLICY
    EXPERT_MODE = "EXPERT"
    POLICY_MODE = "POLICY"
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Device: {self.device}")
        
        # 1. Load Policy
        log.info(f"Loading policy checkpoint: {args.checkpoint}")
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
        
        log.info("Hybrid Evaluator initialized.")
    
    def _load_policy(self, checkpoint: str, bc_checkpoint: str):
        """Load policy weights."""
        log.info(f"Loading BC architecture from: {bc_checkpoint}")
        pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
            bc_checkpoint, map_location=self.device, strict=True
        )
        self.policy = pl_module.model.to(self.device)
        
        if checkpoint and Path(checkpoint).exists():
            log.info(f"Loading checkpoint weights from: {checkpoint}")
            ckpt = torch.load(checkpoint, map_location=self.device)
            
            if 'policy_state_dict' in ckpt:
                self.policy.load_state_dict(ckpt['policy_state_dict'])
                log.info(f"Loaded weights from iteration {ckpt.get('iteration', '?')}")
            elif 'model_state_dict' in ckpt:
                self.policy.load_state_dict(ckpt['model_state_dict'])
                log.info(f"Loaded DAgger weights from iteration {ckpt.get('iteration', '?')}")
            else:
                log.warning("No compatible state dict found. Using BC weights.")
        
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
    
    def _get_policy_action(self, obs, prev_img, goal_img):
        """Get action from policy."""
        curr_img = obs['image_primary']
        proprio = obs['proprio']
        
        batch = self._prepare_batch(prev_img, curr_img, goal_img, proprio)
        with torch.no_grad():
            policy_out = self.policy(batch)
        
        policy_pose = policy_out['pose_chunk'][0, 0].cpu().numpy()
        policy_grip_logit = policy_out['gripper_chunk'][0, 0].cpu().numpy()[0]
        gripper_cmd = -1.0 if policy_grip_logit > 0 else 1.0
        
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
            log.warning(f"IK failed: {e}")
            delta_joints = np.zeros(7)
        
        return np.concatenate([delta_joints, [gripper_cmd]]), policy_pose
    
    def _get_expert_action(self, expert, obs):
        """Get action from scripted expert."""
        target_pose, gripper_action, _ = expert.get_target_pose(obs)  # Returns (pose, gripper, info)
        # Expert gripper: -1.0 = open, 1.0 = close
        # Environment expects: 1.0 = open, -1.0 = close  
        # So we can use gripper_action directly (both use same convention now)
        gripper_cmd = gripper_action
        
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
            log.warning(f"Expert IK failed: {e}")
            delta_joints = np.zeros(7)
        
        return np.concatenate([delta_joints, [gripper_cmd]]), target_pose
    
    def run_episode(
        self,
        episode_id: int,
        seed: int,
        video_writer: cv2.VideoWriter,
        handoff_mode: str = "after_grasp"
    ) -> Tuple[bool, str, Dict]:
        """
        Run a single hybrid evaluation episode.
        
        handoff_mode options:
        - "after_grasp": Expert until object is grasped, then policy takes over
        - "after_lift": Expert until object is lifted, then policy takes over
        - "policy_only": Pure policy evaluation (for baseline comparison)
        - "expert_only": Pure expert evaluation (sanity check)
        """
        # Reset environment
        self.env.reset(seed=seed)
        obs = self.env.get_expert_obs()
        self.ik_solver.reset_controller_state()
        
        # Initialize expert
        object_profile = ObjectProfile(
            size=np.array([0.03, 0.03, 0.03]),  # Default cube size
            grasp_width_normalized=0.7
        )
        expert = ScriptedExpert(object_profile, ExpertConfig())
        expert.reset()
        
        # Render goal image
        goal_img = render_goal_image(self.env, obs['goal_pos_world'])
        prev_img = obs['image_primary'].copy()
        
        # State tracking
        current_mode = self.EXPERT_MODE if handoff_mode != "policy_only" else self.POLICY_MODE
        handoff_step = None
        episode_success = False
        success_steps = 0
        grasp_achieved = False
        lift_achieved = False
        
        metrics = {
            "handoff_step": -1,
            "grasp_step": -1,
            "lift_step": -1,
            "final_dist_to_goal": 999,
            "max_lift_height": 0,
            "policy_steps": 0,
            "expert_steps": 0
        }
        
        log.info(f"=== Episode {episode_id} (seed={seed}) | Mode: {handoff_mode} ===")
        
        for step in range(self.args.max_steps):
            curr_img = obs['image_primary']
            
            # Get current state info
            ee_pos = obs['ee_pose_world'][:3]
            obj_pos = obs['object_pos_world']
            goal_pos = obs['goal_pos_world']
            is_grasped = obs['is_grasped'][0] > 0.5
            dist_obj_goal = np.linalg.norm(obj_pos - goal_pos)
            
            # Track grasp and lift
            if is_grasped and not grasp_achieved:
                grasp_achieved = True
                metrics["grasp_step"] = step
                log.info(f"  Step {step}: GRASP ACHIEVED!")
            
            if grasp_achieved and obj_pos[2] > 0.12:  # Lifted above 12cm
                if not lift_achieved:
                    lift_achieved = True
                    metrics["lift_step"] = step
                    log.info(f"  Step {step}: LIFT ACHIEVED! (z={obj_pos[2]:.3f})")
            
            metrics["max_lift_height"] = max(metrics["max_lift_height"], obj_pos[2])
            
            # === HANDOFF LOGIC ===
            if current_mode == self.EXPERT_MODE:
                # Check handoff conditions
                should_handoff = False
                
                if handoff_mode == "after_grasp" and grasp_achieved:
                    should_handoff = True
                elif handoff_mode == "after_lift" and lift_achieved:
                    should_handoff = True
                elif handoff_mode == "expert_only":
                    should_handoff = False  # Never handoff
                
                if should_handoff:
                    current_mode = self.POLICY_MODE
                    handoff_step = step
                    metrics["handoff_step"] = step
                    log.info(f"  Step {step}: === HANDOFF TO POLICY ===")
            
            # === GET ACTION ===
            if current_mode == self.EXPERT_MODE:
                action, pose = self._get_expert_action(expert, obs)
                metrics["expert_steps"] += 1
            else:
                action, pose = self._get_policy_action(obs, prev_img, goal_img)
                metrics["policy_steps"] += 1
            
            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            obs = self.env.get_expert_obs()
            
            # Update metrics
            ee_pos = obs['ee_pose_world'][:3]
            obj_pos = obs['object_pos_world']
            dist_obj_goal = np.linalg.norm(obj_pos - goal_pos)
            is_grasped = obs['is_grasped'][0] > 0.5
            metrics["final_dist_to_goal"] = dist_obj_goal
            
            # Success check
            if dist_obj_goal < 0.05:
                success_steps += 1
                if success_steps >= 10:
                    episode_success = True
            else:
                success_steps = 0
            
            # Render frame with HUD
            frame = self.env.render()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self._draw_hud(
                frame, episode_id, step, current_mode,
                dist_obj_goal, is_grasped, grasp_achieved, lift_achieved,
                episode_success, handoff_step
            )
            video_writer.write(frame)
            
            # Update state
            prev_img = curr_img.copy()
            
            # Check termination
            if episode_success or terminated or truncated:
                break
            
            # Expert done check
            if current_mode == self.EXPERT_MODE and expert.is_done():
                if handoff_mode == "expert_only":
                    break
        
        # Add pause frames
        for _ in range(30):
            video_writer.write(frame)
        
        result = "SUCCESS" if episode_success else "FAIL"
        log.info(f"Episode {episode_id} Result: {result}")
        log.info(f"  Grasp step: {metrics['grasp_step']}, Lift step: {metrics['lift_step']}")
        log.info(f"  Handoff step: {metrics['handoff_step']}")
        log.info(f"  Expert steps: {metrics['expert_steps']}, Policy steps: {metrics['policy_steps']}")
        log.info(f"  Final dist to goal: {metrics['final_dist_to_goal']:.3f}m")
        
        return episode_success, result, metrics
    
    def _draw_hud(
        self, 
        frame: np.ndarray, 
        ep_id: int, 
        step: int,
        mode: str,
        dist_goal: float,
        is_grasped: bool,
        grasp_achieved: bool,
        lift_achieved: bool,
        success: bool,
        handoff_step: Optional[int]
    ):
        """Draw HUD overlay on video frame."""
        h, w = frame.shape[:2]
        
        # Background bar
        cv2.rectangle(frame, (0, 0), (w, 80), (40, 40, 40), -1)
        
        # Mode indicator (prominent)
        mode_color = (0, 200, 255) if mode == self.EXPERT_MODE else (255, 100, 0)
        mode_text = "EXPERT" if mode == self.EXPERT_MODE else "POLICY"
        cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, mode_color, 2)
        
        # Step info
        cv2.putText(frame, f"Ep {ep_id} | Step {step}", 
                    (130, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Distance to goal
        dist_color = (0, 255, 0) if dist_goal < 0.05 else (0, 165, 255) if dist_goal < 0.1 else (100, 100, 255)
        cv2.putText(frame, f"Goal: {dist_goal*100:.1f}cm", 
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dist_color, 1)
        
        # Grasp/Lift status
        status_parts = []
        if grasp_achieved:
            status_parts.append("GRASPED")
        if lift_achieved:
            status_parts.append("LIFTED")
        status_text = " + ".join(status_parts) if status_parts else "Approaching"
        status_color = (0, 255, 0) if lift_achieved else (0, 200, 255) if grasp_achieved else (150, 150, 150)
        cv2.putText(frame, status_text, (130, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Handoff indicator
        if handoff_step is not None:
            cv2.putText(frame, f"Handoff @ {handoff_step}", 
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Success indicator
        if success:
            cv2.putText(frame, "SUCCESS!", 
                        (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    def run(self):
        """Run full hybrid evaluation."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_name = Path(self.args.checkpoint).stem if self.args.checkpoint else "bc_only"
        
        # Output paths
        video_path = self.output_dir / f"hybrid_eval_{ckpt_name}_{timestamp}.mp4"
        
        log.info(f"Video will be saved to: {video_path}")
        
        # Setup video writer
        frame = self.env.render()
        h, w, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, 30, (w, h))
        
        results = []
        
        try:
            for ep_idx in range(self.args.n_episodes):
                seed = self.args.seed + ep_idx
                success, result, metrics = self.run_episode(
                    ep_idx, seed, video_writer, 
                    handoff_mode=self.args.handoff_mode
                )
                results.append({
                    "episode": ep_idx,
                    "success": success,
                    "result": result,
                    **metrics
                })
        
        finally:
            video_writer.release()
            self.env.close()
        
        # Summary
        successes = sum(1 for r in results if r["success"])
        success_rate = successes / len(results) * 100
        
        log.info("=" * 60)
        log.info("HYBRID EVALUATION SUMMARY")
        log.info("=" * 60)
        log.info(f"Checkpoint: {self.args.checkpoint}")
        log.info(f"Handoff Mode: {self.args.handoff_mode}")
        log.info(f"Episodes: {self.args.n_episodes}")
        log.info(f"Success Rate: {success_rate:.1f}% ({successes}/{len(results)})")
        
        # Per-phase analysis
        grasp_achieved = sum(1 for r in results if r["grasp_step"] >= 0)
        lift_achieved = sum(1 for r in results if r["lift_step"] >= 0)
        log.info(f"Grasp Achieved: {grasp_achieved}/{len(results)}")
        log.info(f"Lift Achieved: {lift_achieved}/{len(results)}")
        
        avg_policy_steps = np.mean([r["policy_steps"] for r in results])
        avg_expert_steps = np.mean([r["expert_steps"] for r in results])
        log.info(f"Avg Expert Steps: {avg_expert_steps:.0f}")
        log.info(f"Avg Policy Steps: {avg_policy_steps:.0f}")
        
        log.info(f"Video: {video_path}")
        log.info("=" * 60)
        
        return results


# ==============================================================================
# 3. MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Hybrid Policy-Expert Evaluation")
    
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to policy checkpoint (.pt file)"
    )
    parser.add_argument(
        "--bc_checkpoint", type=str, 
        default="/content/drive/MyDrive/pda/bc/bc_backup_epoch_088.ckpt",
        help="Path to BC checkpoint (for model architecture)"
    )
    parser.add_argument(
        "--handoff_mode", type=str, default="after_grasp",
        choices=["after_grasp", "after_lift", "policy_only", "expert_only"],
        help="When to handoff from expert to policy"
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
        "--output_dir", type=str, default="outputs/hybrid_eval",
        help="Directory to save evaluation outputs"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for first episode"
    )
    parser.add_argument(
        "--n_episodes", type=int, default=3,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--max_steps", type=int, default=800,
        help="Maximum steps per episode"
    )
    
    args = parser.parse_args()
    
    evaluator = HybridEvaluator(args)
    evaluator.run()


if __name__ == "__main__":
    main()
