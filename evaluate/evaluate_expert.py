# FILE: evaluate/evaluate_expert.py
"""
Expert Evaluation Script

This script runs the ScriptedExpert to verify it can complete the full
pick-and-place task. This is a sanity check to ensure the environment
and expert are working correctly before debugging policy issues.

Usage:
    python evaluate/evaluate_expert.py --n_episodes 3 --seed 42
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2
import mujoco
import numpy as np

# Project Imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.panda_env import PandaEnv
from utils.ik_solver import IKSolver
from utils.scripted_expert import ScriptedExpert, ExpertConfig, ObjectProfile

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("Expert_Eval")


class ExpertEvaluator:
    """Evaluates the scripted expert to verify it completes the task."""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        
        # Initialize Environment
        log.info("Initializing PandaEnv...")
        self.env = PandaEnv(
            xml_path=args.xml_path,
            control_mode="delta",
            render_mode="rgb_array"
        )
        
        # Initialize IK Solver
        log.info("Initializing IK Solver...")
        self.ik_solver = IKSolver(urdf_path=args.urdf_path)
        
        # Control calibration
        SIM_SUBSTEPS = 20
        self.effective_dt = self.env.model.opt.timestep * SIM_SUBSTEPS
        self.max_dq = self.env.ACTION_SCALING_FACTOR / self.effective_dt
        
        # Output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        log.info("Expert Evaluator initialized.")
    
    def _get_expert_action(self, expert, obs):
        """Get action from scripted expert."""
        target_pose, gripper_action, info = expert.get_target_pose(obs)
        
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
            log.warning(f"IK failed: {e}")
            delta_joints = np.zeros(7)
        
        return np.concatenate([delta_joints, [gripper_action]]), target_pose, info
    
    def run_episode(self, episode_id: int, seed: int, video_writer: cv2.VideoWriter):
        """Run a single expert evaluation episode."""
        # Reset environment
        self.env.reset(seed=seed)
        obs = self.env.get_expert_obs()
        self.ik_solver.reset_controller_state()
        
        # Initialize expert
        object_profile = ObjectProfile(
            size=np.array([0.03, 0.03, 0.03]),
            grasp_width_normalized=0.7
        )
        expert = ScriptedExpert(object_profile, ExpertConfig())
        expert.reset()
        
        # Track metrics
        episode_success = False
        grasp_achieved = False
        lift_achieved = False
        success_steps = 0
        
        goal_pos = obs['goal_pos_world']
        
        log.info(f"=== Episode {episode_id} (seed={seed}) ===")
        log.info(f"  Object: {obs['object_pos_world']}")
        log.info(f"  Goal: {goal_pos}")
        
        for step in range(self.args.max_steps):
            # Get expert action
            action, target_pose, info = self._get_expert_action(expert, obs)
            expert_state = info.get('expert_state_str', 'UNKNOWN')
            
            # Step environment
            obs, reward, terminated, truncated, _ = self.env.step(action)
            obs = self.env.get_expert_obs()
            
            # Track metrics
            ee_pos = obs['ee_pose_world'][:3]
            obj_pos = obs['object_pos_world']
            is_grasped = obs['is_grasped'][0] > 0.5
            dist_obj_goal = np.linalg.norm(obj_pos - goal_pos)
            
            if is_grasped and not grasp_achieved:
                grasp_achieved = True
                log.info(f"  Step {step}: GRASP ACHIEVED!")
            
            if grasp_achieved and obj_pos[2] > 0.12:
                if not lift_achieved:
                    lift_achieved = True
                    log.info(f"  Step {step}: LIFT ACHIEVED! (z={obj_pos[2]:.3f})")
            
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
            self._draw_hud(frame, episode_id, step, expert_state, dist_obj_goal, 
                          is_grasped, grasp_achieved, lift_achieved, episode_success)
            video_writer.write(frame)
            
            # Check termination
            if episode_success or expert.is_done():
                break
        
        # Add pause frames at end
        for _ in range(30):
            video_writer.write(frame)
        
        result = "SUCCESS ✓" if episode_success else "FAIL ✗"
        log.info(f"Episode {episode_id} Result: {result}")
        log.info(f"  Grasp: {grasp_achieved}, Lift: {lift_achieved}")
        log.info(f"  Final dist to goal: {dist_obj_goal:.3f}m")
        log.info(f"  Final state: {expert_state}")
        
        return episode_success, grasp_achieved, lift_achieved, dist_obj_goal
    
    def _draw_hud(self, frame, ep_id, step, state, dist_goal, 
                  is_grasped, grasp_achieved, lift_achieved, success):
        """Draw HUD overlay on video frame."""
        h, w = frame.shape[:2]
        
        # Background bar
        cv2.rectangle(frame, (0, 0), (w, 80), (40, 40, 40), -1)
        
        # Expert state (prominent)
        cv2.putText(frame, f"EXPERT: {state}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        
        # Episode/step info
        cv2.putText(frame, f"Ep {ep_id} | Step {step}", 
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Distance to goal
        dist_color = (0, 255, 0) if dist_goal < 0.05 else (0, 165, 255) if dist_goal < 0.1 else (100, 100, 255)
        cv2.putText(frame, f"Goal: {dist_goal*100:.1f}cm", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dist_color, 1)
        
        # Status indicators
        status_parts = []
        if grasp_achieved:
            status_parts.append("GRASPED")
        if lift_achieved:
            status_parts.append("LIFTED")
        status_text = " + ".join(status_parts) if status_parts else "Approaching"
        status_color = (0, 255, 0) if lift_achieved else (0, 200, 255) if grasp_achieved else (150, 150, 150)
        cv2.putText(frame, status_text, (180, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Success indicator
        if success:
            cv2.putText(frame, "SUCCESS!", 
                        (w - 110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    def run(self):
        """Run full expert evaluation."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = self.output_dir / f"expert_eval_{timestamp}.mp4"
        
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
                success, grasp, lift, dist = self.run_episode(ep_idx, seed, video_writer)
                results.append({
                    "episode": ep_idx,
                    "success": success,
                    "grasp_achieved": grasp,
                    "lift_achieved": lift,
                    "final_dist": dist
                })
        finally:
            video_writer.release()
            self.env.close()
        
        # Summary
        successes = sum(1 for r in results if r["success"])
        grasps = sum(1 for r in results if r["grasp_achieved"])
        lifts = sum(1 for r in results if r["lift_achieved"])
        success_rate = successes / len(results) * 100
        
        log.info("=" * 60)
        log.info("EXPERT EVALUATION SUMMARY")
        log.info("=" * 60)
        log.info(f"Episodes: {self.args.n_episodes}")
        log.info(f"Success Rate: {success_rate:.1f}% ({successes}/{len(results)})")
        log.info(f"Grasps Achieved: {grasps}/{len(results)}")
        log.info(f"Lifts Achieved: {lifts}/{len(results)}")
        log.info(f"Video: {video_path}")
        log.info("=" * 60)
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Scripted Expert")
    
    parser.add_argument(
        "--xml_path", type=str, default="envs/panda_pick_place.xml",
        help="Path to MuJoCo XML file"
    )
    parser.add_argument(
        "--urdf_path", type=str, default="urdf/panda_mujoco_kinematics.urdf",
        help="Path to URDF file for IK"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/expert_eval",
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
        "--max_steps", type=int, default=1000,
        help="Maximum steps per episode"
    )
    
    args = parser.parse_args()
    
    evaluator = ExpertEvaluator(args)
    evaluator.run()


if __name__ == "__main__":
    main()
