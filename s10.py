# FILE: test_reward_wrapper.py
# Description: A script to diagnose the AdvancedRewardWrapper behavior.

import sys
import logging
from pathlib import Path
import numpy as np
from dataclasses import replace
import copy
import pickle # To potentially load expert episodes
import csv
from typing import List, Dict, Any
# --- Add project root to sys.path ---
# Adjust the number of .parent calls if this script is moved



# --- Project Imports ---
# Wrap imports in try-except for better error messages if paths are wrong
try:
    from envs.panda_env import PandaEnv
    from utils.rl_reward_wrapper import (
        AdvancedRewardWrapper,
        AdvancedRewardConfig,
        CurriculumConfig,
        PhysicalState
    )
    from utils.scripted_expert import ScriptedExpert, ExpertConfig, ObjectProfile
    from utils.ik_solver import IKSolver # Needed for expert execution
except ImportError as e:
    print("\n--- IMPORT ERROR ---")
    print(f"Failed to import necessary modules: {e}")
    print("Please ensure:")
    print(f"1. You are running this script from within the project structure (e.g., from ).")
    print(f"2. The project root directory  is correctly added to PYTHONPATH.")
    print("3. All required files (panda_env.py, rl_reward_wrapper.py, etc.) exist in their expected locations.")
    print("---------------------\n")
    sys.exit(1)

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger("RewardTester")

# --- IMPORTANT: Match these configurations to your RL training run ---
# FILE: test_reward_wrapper.py (s10.py)

# --- REPLACE THIS ENTIRE BLOCK ---
REWARD_CONFIG = AdvancedRewardConfig(
    # --- V2 Master Scale & Discount ---
    potential_scale=5.0,
    gamma=0.99,

    # --- V2 Potential Function Coefficients ---
    # 1. Reach
    k_reach_pos_xy=1.5,
    k_reach_pos_z=1.0,
    k_reach_orn=0.5,

    # 2. Grasp
    k_grasp_pos_z=1.5,
    k_grasp_align=0.5,

    # 3. Lift
    k_lift=2.0,

    # 4. Move
    k_move_pos_xy=2.0,
    k_move_pos_z=0.5,
    k_move_orn=0.5,

    # 5. Place
    k_place_pos_z=2.0,
    k_place_align_xy=1.0,
    k_place_orn_final=0.5,
    k_retract = 3.0,

    # --- Penalty Coefficients (mostly unchanged) ---
    action_penalty=0.00001,
    jerk_penalty=0.0001,
    contact_penalty=2.0,
    drop_penalty=20.0,
    instability_penalty=0.05,

    # --- Sparse Bonuses (unchanged) ---
    grasp_bonus=10.0,
    lift_bonus=15.0,
    place_bonus=25.0,
    success_bonus=100.0,

    # --- Physical & Task Thresholds (new/renamed fields) ---
    hover_height=0.05,
    grasp_descend_height=0.005,
    lift_height_thresh=0.04,
    place_dist_thresh=0.03,
    goal_pos_thresh=0.02,
    goal_orn_thresh=0.1,
    stable_velocity_thresh=0.01,

    # --- V2 Blending Sigmas ---
    sigma_dist_xy=0.15,
    sigma_dist_z=0.01,
    sigma_lift=0.05
)
# --- END OF REPLACEMENT ---

CURRICULUM_CONFIG = CurriculumConfig(
    total_episodes=2000, # Set to your training total_episodes
    # V2 anneals 'potential_scale', not 'dense_reward_weight'
    potential_scale_anneal_end_factor=0.5,
    goal_thresh_anneal_end_factor=0.5
)

# Configuration for the environment and expert (adjust paths as needed)
# Use absolute paths if relative paths cause issues
XML_PATH = str("envs/panda_pick_place.xml")
URDF_PATH = str("urdf/panda_mujoco_kinematics.urdf") # Needed for expert
OBJECT_SIZE = (0.04, 0.04, 0.04)
GRASP_WIDTH = 0.6
CONTROL_MODE = 'delta' # Match the control mode used during RL training

# Expert Config (can use defaults or load yours)
EXPERT_CFG = ExpertConfig()

# --- Helper Functions ---

def print_reward_info(step_num: int, reward: float, info: dict):
    """Prints a formatted summary of the reward components."""
    print(f"\n--- Step {step_num} ---")
    print(f"Total Reward: {reward:.4f}")
    print("Components:")
    for key, value in info.items():
        if key.startswith('r_') or key.startswith('pot_') or key == 'is_success':
            # Format floats nicely, pass through others (like bools)
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    print("-" * (14 + len(str(step_num))))


# FILE: test_reward_wrapper.py

# REPLACE the existing generate_expert_trajectory function
def generate_expert_trajectory(env: PandaEnv, expert: ScriptedExpert, seed: int) -> list:
    """Generates a single expert trajectory in memory, including the expert's FSM state."""
    log.info(f"Generating expert trajectory with seed {seed}...")
    trajectory = []
    expert.reset()
    obs, _ = env.reset(seed=seed)
    env.set_object_size(expert.object.size)

    # Use a fresh IK solver instance to avoid state conflicts
    ik_solver = IKSolver(urdf_path=URDF_PATH)

    N_SUBSTEPS = 20
    effective_dt = env.model.opt.timestep * N_SUBSTEPS
    arm_joint_ids = np.arange(7)
    max_dq = env.ACTION_SCALING_FACTOR / effective_dt

    steps = 0
    while not expert.is_done() and steps < env.max_episode_steps:
        expert_obs = env.get_expert_obs()
        target_ee_pose, gripper_action = expert.get_target_pose(expert_obs)

        delta_arm_action = ik_solver.compute_delta_action(
            target_ee_pose=target_ee_pose,
            model=env.model,
            data=env.data,
            ee_site_id=env.ee_site_id,
            joint_qpos_indices=arm_joint_ids,
            effective_dt=effective_dt,
            max_dq=max_dq
        )
        action_to_take = np.concatenate([delta_arm_action, [gripper_action]])

        # --- PATCHED PART ---
        # Store the observation, the action, AND the expert's state for this step
        trajectory.append({
            'obs': copy.deepcopy(expert_obs),
            'action': action_to_take,
            'expert_state': expert.get_state() # <-- NEW
        })
        # --- END PATCH ---

        obs, _, terminated, truncated, _ = env.step(action_to_take)
        steps += 1
        if terminated or truncated:
            break

    if expert.was_successful():
        log.info(f"Expert trajectory generation successful ({len(trajectory)} steps).")
    else:
        log.warning(f"Expert trajectory generation failed after {len(trajectory)} steps. State: {expert.get_state()}")
    return trajectory



def save_results_to_csv(results_data: List[Dict[str, Any]], output_path: Path):
    """Saves the collected reward analysis data to a CSV file."""
    if not results_data:
        log.warning("No data to save, skipping CSV creation.")
        return

    # Dynamically get headers from the keys of the first data dictionary
    headers = results_data[0].keys()

    log.info(f"Saving reward analysis for {len(results_data)} steps to {output_path}...")
    try:
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results_data)
        log.info(f"Successfully saved analysis to {output_path}")
    except Exception as e:
        log.exception(f"Failed to write to CSV file: {e}")

# FILE: test_reward_wrapper.py


# FILE: test_reward_wrapper.py

# REPLACE the existing replay_trajectory function with this one.
def replay_trajectory(wrapped_env: AdvancedRewardWrapper, trajectory: list, test_seed: int, episode_num: int = 0) -> List[Dict[str, Any]]:
    """
    Replays a trajectory, collects detailed V2 reward/state info, and returns it.
    This version includes seed-matching, potential alignment, and comprehensive logging.
    """
    log.info(f"\n===== REPLAYING AND ANALYZING (Curriculum Episode: {episode_num}, Seed: {test_seed}) =====")
    if not trajectory:
        log.warning("Trajectory is empty, cannot replay.")
        return []

    results_data = []

    # --- 1. Reset Environment and Set Curriculum ---
    wrapped_env._current_episode = episode_num
    log.info(f"Setting curriculum episode to: {wrapped_env._current_episode}")
    obs, info = wrapped_env.reset(seed=test_seed)
    initial_potential_at_reset = wrapped_env._last_potential
    log.info(f"Initial potential at reset (from seed={test_seed}): {initial_potential_at_reset:.6f}")

    # --- 2. Align Wrapper's Potential with Trajectory's True Start State ---
    aligned_potential = None
    if "obs" in trajectory[0]:
        saved_obs0 = trajectory[0]['obs']
        try:
            # Use a zero action for state extraction at the start
            state0 = wrapped_env._extract_state(saved_obs0, np.zeros(wrapped_env.action_space.shape))
            aligned_potential = wrapped_env._calculate_potential(state0)[0]
            log.info(f"Aligning wrapper._last_potential to recorded trajectory's initial potential: {aligned_potential:.6f}")
            wrapped_env._last_potential = aligned_potential
        except Exception as e:
            log.warning(f"Failed to align _last_potential from saved obs: {e}")

    total_reward_accum = 0.0

    # --- 3. Replay Trajectory Step-by-Step ---
    for i, step_data in enumerate(trajectory):
        action = step_data['action']
        expert_state_str = step_data['expert_state']
        
        # Store the potential *before* the step for detailed analysis
        potential_before_step = wrapped_env._last_potential

        # Perform the step
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        total_reward_accum += reward

        # --- 4. Comprehensive Data Collection (using correct V2 info keys) ---
        s = wrapped_env._extract_state(obs, wrapped_env._last_action)
        dense_reward = info.get('r_dense_shaped')

        # Anomaly Detection for the first step
        if i == 0 and dense_reward is not None and abs(dense_reward) > 5.0:
            log.warning(
                f"[ANOMALY DETECTED] Large dense reward at step 0 ({dense_reward:.4f}). "
                f"Initial Potential: {potential_before_step:.4f}, New Potential: {info.get('potential_total'):.4f}. "
                "This may indicate a state mismatch."
            )

        step_results = {
            'step': i,
            'expert_state': expert_state_str,
            'total_reward': reward,
            'r_dense_shaped': dense_reward,
            'r_sparse_event': info.get('r_sparse_event'),
            'r_penalty_total': info.get('r_penalty_total'),
            'potential_before_step': potential_before_step, # NEW DIAGNOSTIC
            'potential_after_step': info.get('potential_total'),  # NEW DIAGNOSTIC

            # Blending Weights
            'w_reach': info.get('w_reach'), 'w_grasp': info.get('w_grasp'),
            'w_lift': info.get('w_lift'), 'w_move': info.get('w_move'), 'w_place': info.get('w_place'),
            'w_retract': info.get('w_retract'), # NEW
            'potential_before_step': wrapped_env._last_potential, # NEW - Add this for clarity
            'potential_after_step': info.get('potential_total'),  # NEW - Renamed for clarity
            # Key State Variables
            'dist_ee_obj_xy': s.dist_ee_obj_xy, 'dist_obj_goal_3d': s.dist_obj_goal_3d,
            'object_lift': s.object_lift_relative, 'is_grasped': s.is_physically_grasped,
            'err_angle_obj_goal': s.angle_obj_goal_alignment,
        }
        results_data.append(step_results)
        # --- End Data Collection ---

        if terminated or truncated:
            log.info(f"Replay terminated/truncated at step {i}.")
            break

    log.info(f"Trajectory replay finished. Total accumulated reward: {total_reward_accum:.4f}")
    if info.get('is_success', False):
         log.info("Replay indicates SUCCESS was achieved.")
    else:
         log.warning("Replay indicates SUCCESS was NOT achieved.")

    return results_data

if __name__ == "__main__":
    log.info("--- Starting Reward Wrapper Test Script ---")

    # --- 1. Initialization ---
    try:
        log.info(f"Initializing PandaEnv (XML: {XML_PATH}, Control: {CONTROL_MODE})...")
        env = PandaEnv(
            xml_path=XML_PATH,
            control_mode=CONTROL_MODE,
            render_mode=None
        )
        env.set_object_size(OBJECT_SIZE)

        log.info("Initializing AdvancedRewardWrapper...")
        wrapped_env = AdvancedRewardWrapper(
            env,
            reward_cfg=replace(REWARD_CONFIG),
            curriculum_cfg=replace(CURRICULUM_CONFIG)
        )

        log.info("Initializing ScriptedExpert...")
        object_profile = ObjectProfile(size=np.array(OBJECT_SIZE), grasp_width_normalized=GRASP_WIDTH)
        expert = ScriptedExpert(object_profile=object_profile, cfg=EXPERT_CFG)

    except Exception as e:
        log.exception(f"ERROR: Failed during initialization: {e}")
        sys.exit(1)

    # --- 2. Generate Expert Trajectory ---
    TEST_SEED = 42 # Use a fixed seed for reproducibility
    try:
        expert_trajectory = generate_expert_trajectory(env, expert, TEST_SEED)
    except Exception as e:
        log.exception(f"ERROR: Failed to generate expert trajectory: {e}")
        expert_trajectory = []

    # --- 3. Replay and Save Results ---
    if expert_trajectory:
        # We will test at a single, representative stage of the curriculum.
        # The middle of training is a good choice. Change this value to test other stages.
        EPISODE_TO_TEST = CURRICULUM_CONFIG.total_episodes // 2
        
        log.info(f"\n===== REPLAYING AND ANALYZING (Curriculum Episode: {EPISODE_TO_TEST}) =====")
        
        # Replay the trajectory to get the data
        replay_data = replay_trajectory(wrapped_env, expert_trajectory, test_seed=TEST_SEED,episode_num=EPISODE_TO_TEST)
        
        # Define the output path and save the results
        output_csv_path = "reward_analysis.csv"
        save_results_to_csv(replay_data, output_csv_path)

    else:
        log.error("Cannot perform replay test because expert trajectory is empty.")

    # --- 4. Cleanup ---
    log.info("Closing environment.")
    wrapped_env.close()

    log.info("--- Reward Wrapper Test Script Finished ---")



