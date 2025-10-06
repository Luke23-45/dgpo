# FILE: debug_scripts/verify_reward_wrapper.py (DEFINITIVE VERSION)

import argparse
import logging
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

# --- Project Imports ---
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from envs.panda_env import PandaEnv
from utils.rl_reward_wrapper import RLRewardWrapper
from utils.scripted_expert import ScriptedExpert, ObjectProfile
from utils.ik_solver import IKSolver

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | [%(name)s] | %(message)s"
)
log = logging.getLogger("VERIFY_REWARD_WRAPPER")


def print_step_diagnostics(step_name: str, obs: dict, info: dict, reward: float):
    """Helper to print a clean, detailed log for each test step."""
    log.info("-" * 80)
    log.info(f"--- VERIFYING SCENARIO: {step_name} ---")
    
    is_grasped_obs = obs.get("is_grasped", np.array([-1.0]))[0]
    log.info(f"Agent Observation['is_grasped']: {is_grasped_obs:.1f}")

    dist_ee = info.get('dist_ee_to_cube', -1)
    dist_goal = info.get('dist_cube_to_goal', -1)
    log.info(f"Distances | EE->Cube: {dist_ee:.4f} | Cube->Goal: {dist_goal:.4f}")

    r_total = reward
    r_reach = info.get('R_reach', 0)
    r_timing = info.get('R_gripper_timing', 0)
    r_grasp = info.get('R_grasp', 0)
    r_lift = info.get('R_lift', 0)
    r_place = info.get('R_place', 0)
    r_success = info.get('R_success', 0)
    log.info(f"Total Reward: {r_total:+.4f}")
    log.info(f"  Breakdown | Reach: {r_reach:+.4f} | Timing: {r_timing:+.4f} | Grasp: {r_grasp:+.1f} | "
             f"Lift: {r_lift:+.1f} | Place: {r_place:+.4f} | Success: {r_success:+.1f}")
    log.info("-" * 80)


def get_expert_action(env, expert, ik_solver):
    """Helper to generate a single expert action for the current state."""
    expert_obs = env.unwrapped.get_expert_obs() # Get rich obs from base env

    target_pose, gripper_action = expert.get_target_pose(
        expert_obs["ee_pose_world"], expert_obs["object_pos_world"],
        expert_obs["object_orn_world"], expert_obs["goal_pos_world"],
        expert_obs["is_grasped"]
    )
    
    base_pos, base_quat = env.unwrapped.get_base_pose()
    R_world_base = R.from_quat(base_quat)
    R_base_world = R_world_base.inv()
    pos_in_base = R_base_world.apply(target_pose[:3] - base_pos)
    rot_in_base = R_base_world * R.from_quat(target_pose[3:7])
    target_pose_base = np.concatenate([pos_in_base, rot_in_base.as_quat()]).astype(np.float32)

    current_joints = expert_obs["internal_full_proprio"][:7]
    arm_action = ik_solver.compute_action(target_pose_base, current_joints)
    
    return np.concatenate([arm_action, [gripper_action]])


def main(args: argparse.Namespace):
    log.info("--- Starting RLRewardWrapper Verification Script (v2) ---")
    
    # 1. CONSTRUCT ALL COMPONENTS
    base_env = PandaEnv(xml_path=args.xml_path, control_mode='absolute')
    env = RLRewardWrapper(base_env) # Wrap it
    object_profile = ObjectProfile(size=np.array([0.04, 0.04, 0.04]), grasp_width_normalized=0.6)
    expert = ScriptedExpert(object_profile)
    ik_solver = IKSolver(urdf_path=args.urdf_path)

    obs, info = env.reset(seed=args.seed)
    expert.reset()
    
    # 2. RUN EPISODE UNTIL GRASP ATTEMPT
    log.info("\n>>> Running expert policy until GRASP state to test reach and timing rewards...")
    last_info, last_reward = {}, 0.0
    for _ in range(50): # Give it 50 steps to reach the cube
        action = get_expert_action(env, expert, ik_solver)
        obs, last_reward, _, _, last_info = env.step(action)
        
        # We stop right when the expert enters the GRASP state, which is the perfect
        # moment to test the R_gripper_timing and R_grasp rewards.
        if expert.get_state() == "GRASP":
            break
            
    print_step_diagnostics("GRASP ATTEMPT", obs, last_info, last_reward)
    log.info("VERIFICATION: R_reach should be positive. R_gripper_timing should be > 0. R_grasp should be > 0.")
    log.info("VERIFICATION: Agent obs['is_grasped'] should be 1.0.")

    # --- VERIFY ONE-TIME BONUSES ---
    log.info("\n>>> Taking one more step to verify one-time bonuses...")
    action = get_expert_action(env, expert, ik_solver) # Will be another grasp action
    obs, last_reward, _, _, last_info = env.step(action)
    print_step_diagnostics("GRASP AGAIN", obs, last_info, last_reward)
    log.info("VERIFICATION: R_gripper_timing and R_grasp should BOTH now be 0.0.")

    # --- VERIFY LIFT ---
    log.info("\n>>> Running expert policy until lift occurs...")
    for _ in range(20): # Give it 20 steps to lift
        action = get_expert_action(env, expert, ik_solver)
        obs, last_reward, _, _, last_info = env.step(action)
        # Use is_lifted from the info dict for verification
        if last_info.get("is_lifted", False):
            break
            
    print_step_diagnostics("LIFT", obs, last_info, last_reward)
    log.info("VERIFICATION: R_lift should be > 0.")

    # --- VERIFY SUCCESS ---
    log.info("\n>>> Running expert until end of episode to test SUCCESS condition...")
    terminated = False
    while not expert.is_done() and not terminated:
        action = get_expert_action(env, expert, ik_solver)
        obs, last_reward, terminated, _, last_info = env.step(action)
        
    print_step_diagnostics("RELEASE AT GOAL", obs, last_info, last_reward)
    log.info("VERIFICATION: R_success should be > 0.")
    log.info(f"Final episode status: Terminated = {terminated} (should be True)")

    env.close()
    log.info("--- Verification script finished. Analyze the logs above. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify the RLRewardWrapper logic.")
    parser.add_argument("--xml_path", type=str, default="envs/panda_pick_place.xml")
    parser.add_argument("--urdf_path", type=str, default="urdf/panda_mujoco_kinematics.urdf")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)