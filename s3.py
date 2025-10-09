# FILE: scripts/deep_diagnostics.py
"""
An exhaustive diagnostic script to debug the expert trajectory generation pipeline.

This script runs a single episode and logs a detailed, step-by-step report of
the entire data flow:
1.  The expert's high-level state and intended target pose.
2.  The robot's actual physical state (EE pose, object pose).
3.  The calculated distance errors between intent and reality.
4.  The final action command sent to the simulation.
5.  The resulting raw sensor data (touch and force) after the action is executed.

The output is saved to 'verification_output/deep_diagnostics_report.txt',
providing the ground truth needed to pinpoint failures in the control loop.
"""
import argparse
import logging
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

# --- Project Imports ---
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from envs.panda_env import PandaEnv
from utils.scripted_expert import ScriptedExpert, ObjectProfile, ExpertConfig
from utils.ik_solver import IKSolver

# --- Setup File Logger ---
log_dir = Path("verification_output")
log_dir.mkdir(parents=True, exist_ok=True)
report_path = log_dir / "deep_diagnostics_report.txt"

# Configure a logger to write to the file
file_handler = logging.FileHandler(report_path, mode='w')
file_handler.setFormatter(logging.Formatter("%(message)s"))
log = logging.getLogger("DEEP_DIAGNOSTICS")
log.setLevel(logging.INFO)
log.addHandler(file_handler)
log.propagate = False # Prevent double-logging to console if root logger is configured

# Also log to console for real-time feedback
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(console_handler)


def format_array(arr, precision=4):
    """Helper to format numpy arrays for clean logging."""
    return np.array2string(arr, precision=precision, suppress_small=True, separator=', ')

def run_diagnostics(args: argparse.Namespace):
    log.info("--- STARTING DEEP DIAGNOSTICS FOR EXPERT GRASPING ---")
    log.info(f"Report will be saved to: {report_path.resolve()}\n")

    object_to_grasp = ObjectProfile(
        size=np.array([0.04, 0.04, 0.04]),
        grasp_width_normalized=1.0 # Command a full close
    )
    
    # --- 1. Initialize Core Components ---
    env = PandaEnv(xml_path=args.xml_path)
    # Use the physics-aware expert
    expert = ScriptedExpert(object_profile=object_to_grasp)
    ik_solver = IKSolver(urdf_path=args.urdf_path)
    
    # --- 2. Run One Full Episode ---
    env.set_object_size(object_to_grasp.size)
    expert.reset()
    obs, _ = env.reset(seed=args.seed)

    try:
        for step_num in range(env.max_episode_steps):
            log.info(f"\n==================== STEP {step_num} ====================")
            
            expert_obs = env.get_expert_obs()
            
            # --- LOG BLOCK 1: STATE BEFORE ACTION ---
            log.info(f"EXPERT STATE: {expert.get_state()}")
            
            current_ee_pos = expert_obs["ee_pose_world"][:3]
            object_pos = expert_obs["object_pos_world"]
            
            log.info(f"  - ACTUAL EE POS:      {format_array(current_ee_pos)}")
            log.info(f"  - ACTUAL OBJECT POS:  {format_array(object_pos)}")
            
            # --- Get Expert Command ---
            target_pose, gripper_action = expert.get_target_pose(
                expert_obs["ee_pose_world"],
                expert_obs["object_pos_world"],
                expert_obs["proprio"],
                expert_obs["goal_pos_world"],
                expert_obs["is_grasped"][0] > 0.5,
            )
            target_pos = target_pose[:3]
            
            # --- LOG BLOCK 2: EXPERT INTENT ---
            log.info(f"EXPERT INTENT:")
            log.info(f"  - TARGET EE POS:      {format_array(target_pos)}")
            log.info(f"  - GRIPPER ACTION:     {gripper_action:.4f}")

            # --- LOG BLOCK 3: CALCULATED METRICS ---
            dist_ee_to_target = np.linalg.norm(current_ee_pos - target_pos)
            dist_ee_to_object = np.linalg.norm(current_ee_pos - object_pos)
            log.info(f"METRICS:")
            log.info(f"  - Dist (EE -> Target):  {dist_ee_to_target:.4f} m")
            log.info(f"  - Dist (EE -> Object):  {dist_ee_to_object:.4f} m")
            
            # --- Calculate and Log Action ---
            base_pos, base_quat = env.get_base_pose()
            R_world_base = R.from_quat(base_quat).inv()
            pos_in_base = R_world_base.apply(target_pose[:3] - base_pos)
            rot_in_base = R_world_base * R.from_quat(target_pose[3:7])
            target_pose_base = np.concatenate([pos_in_base, rot_in_base.as_quat()])

            current_joints = expert_obs["internal_full_proprio"][:7]
            arm_action = ik_solver.compute_action(target_pose_base, current_joints)
            final_action = np.concatenate([arm_action, [gripper_action]])
            
            log.info(f"CALCULATED ACTION (normalized): {format_array(final_action)}")
            
            # --- Step Environment and Log Outcome ---
            obs, _, terminated, truncated, _ = env.step(final_action)
            
            proprio_after_step = obs['proprio']
            left_touch = proprio_after_step[14]
            right_touch = proprio_after_step[15]
            left_force = proprio_after_step[16:19]
            right_force = proprio_after_step[19:22]
            
            log.info(f"OUTCOME:")
            log.info(f"  - Is Grasped Flag:    {obs['is_grasped'][0] > 0.5}")
            log.info(f"  - Left Touch Sensor:  {left_touch:.4f}")
            log.info(f"  - Right Touch Sensor: {right_touch:.4f}")
            log.info(f"  - Left Force Sensor:  {format_array(left_force)}")
            log.info(f"  - Right Force Sensor: {format_array(right_force)}")

            if expert.is_done() or terminated or truncated:
                log.info(f"\n==================== EPISODE END ====================")
                log.info(f"Episode finished at step {step_num}. Final expert state: {expert.get_state()}")
                break

    finally:
        env.close()

    if expert.was_successful():
        log.info("\n✅ DIAGNOSTICS COMPLETE: Expert reported SUCCESS.")
    else:
        log.info("\n❌ DIAGNOSTICS COMPLETE: Expert reported FAILURE.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run deep diagnostics on the expert grasping pipeline.")
    parser.add_argument("--urdf_path", type=str, default="urdf/panda_mujoco_kinematics.urdf")
    parser.add_argument("--xml_path", type=str, default="envs/panda_pick_place.xml")
    parser.add_argument("--seed", type=int, default=123, help="Seed for the environment reset.")
    args = parser.parse_args()
    run_diagnostics(args)