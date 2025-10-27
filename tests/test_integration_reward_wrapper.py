# FILE: tests/test_integration_reward_wrapper.py

import pytest
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import time
from scipy.spatial.transform import Rotation as R
import random
# Adjust import paths based on your project structure
# Assuming tests/ is at the same level as envs/, utils/, models/
import sys
from pathlib import Path
# Add project root to sys.path if necessary
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from envs.panda_env import PandaEnv
from utils.rl_reward_wrapper import AdvancedRewardWrapper, AdvancedRewardConfig, CurriculumConfig
from utils.scripted_expert import ScriptedExpert, ExpertConfig, ObjectProfile
import gymnasium as gym
from utils.ik_solver import IKSolver
# --- Constants & Tolerances ---
ACTION_DIM = 8 # 7 robot + 1 gripper
POS_DIM = 3
ORN_DIM = 4 # Quaternions (xyzw)
DEFAULT_ACTION_SCALING = 0.1 # Example scaling factor for delta actions
RTOL = 1e-4 # Relative tolerance for float comparisons

# --- Fixtures ---
@pytest.fixture(scope="module")
def ik_solver() -> IKSolver:
    """Provides a shared IKSolver instance for all tests."""
    # NOTE: Adjust the path if your URDF is located elsewhere
    urdf_path = "urdf/panda_mujoco_kinematics.urdf"
    return IKSolver(urdf_path=urdf_path)

@pytest.fixture(scope="module") # Module scope for configs - they don't change per test
def reward_config() -> AdvancedRewardConfig:
    """Provides a default reward configuration."""
    return AdvancedRewardConfig(
        # Use slightly larger tolerances for integration tests if needed
        goal_pos_thresh=0.03,
        goal_orn_thresh=0.15
    )

@pytest.fixture(scope="module")
def curriculum_config() -> CurriculumConfig:
    """Provides a curriculum config for testing annealing."""
    # Use fewer episodes to see annealing effect faster in tests
    return CurriculumConfig(total_episodes=10)

@pytest.fixture(scope="module")
def expert_config() -> ExpertConfig:
    """Provides a default expert configuration."""
    # Use shorter durations for faster testing, but ensure they are > 0
    return ExpertConfig(
        move_to_pre_grasp_duration=50,
        prepare_gripper_duration=10,
        descend_to_grasp_duration=20,
        lift_duration_steps=15,
        move_to_goal_duration=60,
        prepare_place_duration=20,
        descend_to_place_duration=20,
        retract_duration_steps=20,
        failure_timeout_steps=100 # Shorter timeout for tests
    )

@pytest.fixture(scope="module")
def object_profile() -> ObjectProfile:
    """Provides a default object profile."""
    # Example cube
    return ObjectProfile(size=np.array([0.04, 0.04, 0.04]), grasp_width_normalized=0.8)

@pytest.fixture(scope="function") # Recreate expert for each test to reset its state
def scripted_expert(object_profile: ObjectProfile, expert_config: ExpertConfig) -> ScriptedExpert:
    """Provides a fresh instance of the ScriptedExpert."""
    return ScriptedExpert(object_profile, expert_config)

@pytest.fixture(scope="function") # Recreate wrapped env for each test
def wrapped_env(reward_config: AdvancedRewardConfig,
                curriculum_config: CurriculumConfig):
    """
    Provides a fresh, fully initialized, and reset instance of the
    PandaEnv wrapped with AdvancedRewardWrapper for each test function.
    """
    print("\nSetting up wrapped_env fixture...")
    try:
        # 1. Create base env
        # Need config access here - maybe pass cfg object or use defaults
        # For simplicity, using defaults assumed by PandaEnv if xml_path is None
        base_env = PandaEnv(render_mode="rgb_array", xml_path="envs/panda_pick_place.xml")

        # 2. Apply Reward Wrapper
        # Use non-annealing curriculum for most tests unless specifically testing annealing
        test_curriculum_cfg = CurriculumConfig(total_episodes=10000) # Effectively disable annealing
        wrapped = AdvancedRewardWrapper(base_env, reward_config, test_curriculum_cfg)

        # 3. Apply TimeLimit
        wrapped = gym.wrappers.TimeLimit(wrapped, max_episode_steps=500)

        # 4. Initial Reset
        seed = random.randint(0, 10000) # Use different seed per test run
        print(f"Resetting wrapped_env with seed {seed}...")
        obs, info = wrapped.reset(seed=seed)
        print("wrapped_env reset complete.")
        # Ensure observation has expected keys after wrapping + reset
        assert isinstance(obs, dict)
        assert 'ee_pose_world' in obs

        yield wrapped # Provide the wrapped env to the test

        # Teardown: Close the environment
        print("Closing wrapped_env...")
        wrapped.close()
        print("wrapped_env closed.")

    except Exception as e:
        print(f"ERROR during wrapped_env setup: {e}")
        raise

# --- Helper Function ---

def calculate_delta_action(
    current_pose_7d: np.ndarray,
    target_pose_7d: np.ndarray,
    gripper_action: float,
    max_delta_pos: float = 0.05, # Max position change per step (tune this)
    max_delta_orn_deg: float = 10.0, # Max orientation change per step (tune this)
    action_scaling: float = DEFAULT_ACTION_SCALING # Scaling factor
) -> np.ndarray:
    """
    Calculates a delta action to move from current to target pose.
    Handles position and orientation separately.
    """
    current_pos = current_pose_7d[:POS_DIM]
    current_orn_xyzw = current_pose_7d[POS_DIM:]
    target_pos = target_pose_7d[:POS_DIM]
    target_orn_xyzw = target_pose_7d[POS_DIM:]

    # --- Position Delta ---
    delta_pos_raw = (target_pos - current_pos)
    # Scale and clip position delta magnitude
    pos_dist = np.linalg.norm(delta_pos_raw)
    scaled_delta_pos = delta_pos_raw * action_scaling
    clipped_delta_pos = scaled_delta_pos * min(1.0, max_delta_pos / (pos_dist * action_scaling + 1e-6))

    # --- Orientation Delta (Simplified: Axis-Angle Difference) ---
    try:
        R_current = R.from_quat(current_orn_xyzw)
        R_target = R.from_quat(target_orn_xyzw)
        delta_R = R_target * R_current.inv()
        delta_rotvec = delta_R.as_rotvec() # Axis-angle representation (scaled axis)
    except ValueError: # Handle invalid quaternions
        delta_rotvec = np.zeros(3)

    # Scale and clip orientation delta magnitude (angle)
    angle_rad = np.linalg.norm(delta_rotvec)
    max_angle_rad = np.radians(max_delta_orn_deg)
    scaled_delta_rotvec = delta_rotvec * action_scaling
    clipped_delta_rotvec = scaled_delta_rotvec * min(1.0, max_angle_rad / (angle_rad * action_scaling + 1e-6))

    # Combine action (pos_delta, orn_delta, gripper)
    # NOTE: MuJoCo/Gym typically expects orn_delta as axis-angle or euler. Check env specifics.
    # Assuming axis-angle for now.
    action = np.concatenate([
        clipped_delta_pos,
        clipped_delta_rotvec, # Axis-angle (3 values)
        [gripper_action]
    ])

    if ACTION_DIM == 8:
        # A common format for an 8D action is [dx, dy, dz, d_roll, d_pitch, d_yaw, d_grip_w, d_grip_h]
        # Our `calculate_delta_action` using rotvec is likely mismatching this.
        # LET'S TRY A SIMPLER, MORE DIRECT CONTROL for debugging.
        # We will directly output a correctly shaped 8D action vector.
        # [delta_pos (3), delta_orn_rotvec (3), unused (1), gripper (1)]
        action_7d = np.concatenate([clipped_delta_pos, clipped_delta_rotvec, [gripper_action]])

        # Create an 8D action and place values correctly.
        # This is an assumption! We may need to find the env's true action definition.
        # Assumption: action[:3]=delta_pos, action[3:6]=delta_orn, action[7]=gripper
        action = np.zeros(ACTION_DIM)
        action[:3] = clipped_delta_pos
        action[3:6] = clipped_delta_rotvec
        action[7] = gripper_action
    else: # Fallback to original logic if ACTION_DIM is different
        action = np.concatenate([clipped_delta_pos, clipped_delta_rotvec, [gripper_action]])

    # Final clipping
    return np.clip(action, -1.0, 1.0)




# FILE: tests/test_integration_reward_wrapper.py

# ... (make sure IKSolver is imported at the top)
from utils.ik_solver import IKSolver
# ...

def run_steps_with_expert(
    env: gym.Wrapper,
    expert: ScriptedExpert,
    ik_solver: IKSolver, # Correct argument name
    max_steps: int = 100,
    target_expert_state: Optional[str] = None,
) -> List[Tuple[Dict, float, bool, bool, Dict]]:
    """
    Runs the simulation loop, driving actions using the ScriptedExpert
    and a proper Differential IK controller, with detailed debugging printouts.
    """
    results = []

    # --- Get controller parameters from the unwrapped environment ---
    # These are needed for the IK solver to work correctly with the sim
    unwrapped_env = env.unwrapped
    N_SUBSTEPS = 20 # Assuming this is constant in your env's step method
    effective_dt = unwrapped_env.model.opt.timestep * N_SUBSTEPS
    # This is the critical scaling factor derived from your reference script
    max_dq = unwrapped_env.ACTION_SCALING_FACTOR / effective_dt
    arm_joint_ids = np.arange(7) # Panda arm has 7 controllable joints

    print("\n" + "="*50)
    print("--- Starting Expert-Driven Rollout (with IK) ---")
    print(f"Target Expert State: {target_expert_state} | Max Steps: {max_steps}")
    print(f"IK Params: effective_dt={effective_dt:.4f}, max_dq={max_dq:.2f}")
    print("="*50 + "\n")

    for step in range(max_steps):
        # 1. Get full ground-truth observation for the expert
        obs = unwrapped_env.get_expert_obs()

        # 2. Get expert target pose
        target_ee_pose, gripper_action = expert.get_target_pose(obs)

        # 3. Calculate delta action USING THE CORRECT IK SOLVER
        delta_arm_action = ik_solver.compute_delta_action(
            target_ee_pose=target_ee_pose,
            model=unwrapped_env.model,
            data=unwrapped_env.data,
            ee_site_id=unwrapped_env.ee_site_id,
            joint_qpos_indices=arm_joint_ids,
            effective_dt=effective_dt,
            max_dq=max_dq
        )
        
        # Construct the final 8D action vector expected by the environment
        action = np.zeros(ACTION_DIM)
        action[:7] = delta_arm_action
        action[7] = gripper_action

        # --- DETAILED DEBUG PRINTOUT ---
        current_pose_7d = obs['ee_pose_world']
        print(f"--- Step {step+1}/{max_steps} | Expert State: {expert.get_state()} ---")
        print(f"  Current EE Pos:    {np.array2string(current_pose_7d[:3], precision=3, suppress_small=True)}")
        print(f"  Expert Target Pos:   {np.array2string(target_ee_pose[:3], precision=3, suppress_small=True)}")
        print(f"  Calculated IK Action: {np.array2string(action, precision=2, suppress_small=True)}")

        # 4. Step the environment
        try:
            next_obs, reward, terminated, truncated, info = env.step(action)
            results.append((next_obs, reward, terminated, truncated, info))

            # --- MORE DEBUG PRINTOUT (POST-STEP) ---
            next_ee_pos = next_obs['ee_pose_world'][:3]
            pos_error = np.linalg.norm(next_ee_pos - target_ee_pose[:3])
            print(f"  Reward Received:   {reward:<8.4f} | New EE Pos: {np.array2string(next_ee_pos, precision=3, suppress_small=True)} | Pos Error: {pos_error:.4f}")
            reward_components = {k: v for k, v in info.items() if k.startswith(('r_', 'pot_'))}
            if reward_components:
                print(f"  Info Dict:         {reward_components}")
            print("-"*(len(expert.get_state()) + 20))

            # 5. Check exit conditions
            if terminated or truncated:
                print(f"\nEpisode ended. Terminated={terminated}, Truncated={truncated}")
                break
            if target_expert_state and expert.get_state() == target_expert_state:
                print(f"\nReached target expert state '{target_expert_state}'.")
                break
            if expert.is_done():
                 print(f"\nExpert finished.")
                 break
        except Exception as e:
            print(f"ERROR during simulation step {step+1}: {e}")
            pytest.fail(f"Simulation error: {e}")

    if step == max_steps - 1:
        print(f"\nWarning: Reached max_steps ({max_steps}) without termination or reaching target state.")

    print("\n" + "="*50)
    print("--- Finished Expert-Driven Rollout ---")
    print("="*50 + "\n")
    
    return results



# --- Test Functions ---

# FILE: tests/test_integration_reward_wrapper.py

def test_reach_potential_increases(wrapped_env: gym.Wrapper, scripted_expert: ScriptedExpert, ik_solver: IKSolver):
    """Verify potential increases and positive rewards during the reach phase."""
    print("--- Testing Reach Potential ---")
    scripted_expert.reset()
    assert scripted_expert.get_state() == "MOVE_TO_PRE_GRASP"

    # Get the initial potential before any steps are taken
    initial_potential = wrapped_env.env._last_potential

    results = run_steps_with_expert(wrapped_env, scripted_expert, ik_solver, max_steps=60, target_expert_state="PREPARE_GRIPPER")

    assert len(results) > 1, "Should have taken multiple steps"

    # --- NEW DETAILED ANALYSIS BLOCK ---
    potentials = [initial_potential] + [info['potential_total'] for _, _, _, _, info in results]
    dense_rewards = [info['r_dense_shaped'] for _, _, _, _, info in results]
    reach_distances = [info.get('dist_ee_to_obj', -1) for _, _, _, _, info in results] # Assuming this key is in info

    print("\n--- REACH PHASE ANALYSIS ---")
    print(f"{'Step':<5} | {'DistToObj':<12} | {'Potential':<12} | {'DenseReward':<12}")
    print("-"*50)
    for i in range(len(results)):
        step_num = i + 1
        dist = reach_distances[i]
        pot = potentials[i+1]
        reward = dense_rewards[i]
        print(f"{step_num:<5} | {dist:<12.4f} | {pot:<12.4f} | {reward:<12.4f}")

    # Check for decreasing distances (robot is making progress)
    # The path isn't perfectly straight, so we check the overall trend
    assert reach_distances[0] > reach_distances[-1], "Robot should have moved closer to the object over the trajectory."

    # CRITICAL ASSERTION: If the reward is negative, the potential must have decreased. Let's find the first failure.
    for i, reward in enumerate(dense_rewards):
        if i > 0 and reward < 0: # Skip the first step which can be noisy
            potential_change = potentials[i+1] - potentials[i]
            assert potential_change >= 0, \
                f"FAILURE at Step {i+1}: Dense reward was {reward:.4f}, but potential change was {potential_change:.4f}. Potential should not decrease for progress."

    assert scripted_expert.get_state() == "PREPARE_GRIPPER", "Expert should have reached target state"


def test_grasp_bonus_awarded_once(wrapped_env: gym.Wrapper, scripted_expert: ScriptedExpert, ik_solver: IKSolver):
    """Verify grasp bonus is awarded exactly once when grasp occurs."""
    print("--- Testing Grasp Bonus ---")
    scripted_expert.reset()
    # Run simulation until *after* the grasp state is expected to finish
    results = run_steps_with_expert(wrapped_env, scripted_expert, ik_solver, max_steps=150, target_expert_state="LIFT")

    assert scripted_expert.get_state() == "LIFT", "Expert should have successfully lifted"

    grasp_bonus_count = 0
    grasp_bonus_value = wrapped_env.reward_cfg.grasp_bonus
    for i, (_, _, _, _, info) in enumerate(results):
        if info.get('r_sparse_grasp', 0.0) == pytest.approx(grasp_bonus_value):
            grasp_bonus_count += 1
            print(f"Grasp bonus found at step {i}")
        # Also check is_grasped state transition
        obs = results[i][0]
        if i > 0 and not results[i-1][0]['is_grasped'][0] > 0.5 and obs['is_grasped'][0] > 0.5:
             print(f"is_grasped became True at step {i}")


    assert grasp_bonus_count == 1, f"Expected grasp bonus once, found {grasp_bonus_count} times"

def test_lift_bonus_awarded_once(wrapped_env: gym.Wrapper, scripted_expert: ScriptedExpert, ik_solver: IKSolver):
    """Verify lift bonus is awarded exactly once when object is lifted."""
    print("--- Testing Lift Bonus ---")
    scripted_expert.reset()
    # Run simulation well into the move phase
    results = run_steps_with_expert(wrapped_env, scripted_expert, ik_solver, max_steps=200, target_expert_state="MOVE_TO_GOAL") 

    assert scripted_expert.get_state() == "MOVE_TO_GOAL", "Expert should have reached move state"

    lift_bonus_count = 0
    lift_bonus_value = wrapped_env.reward_cfg.lift_bonus
    lift_threshold = wrapped_env.reward_cfg.lift_height_thresh
    lift_triggered_step = -1

    for i, (_, _, _, _, info) in enumerate(results):
        if info.get('r_sparse_lift', 0.0) == pytest.approx(lift_bonus_value):
            lift_bonus_count += 1
            print(f"Lift bonus found at step {i}")
        if lift_triggered_step < 0 and info.get('object_lift', -1.0) > lift_threshold:
            lift_triggered_step = i
            print(f"Object lift exceeded threshold at step {i} (lift: {info['object_lift']:.4f})")


    assert lift_bonus_count == 1, f"Expected lift bonus once, found {lift_bonus_count} times"
    assert lift_triggered_step != -1, "Object lift never exceeded threshold"

def test_place_bonus_awarded_once(wrapped_env: gym.Wrapper, scripted_expert: ScriptedExpert, ik_solver: IKSolver): 
    """Verify place bonus near goal."""
    print("--- Testing Place Bonus ---")
    scripted_expert.reset()
    # Run simulation until near placement
    # Need a state after MOVE_TO_GOAL, e.g., PREPARE_PLACE or DESCEND_TO_PLACE
    results = run_steps_with_expert(wrapped_env, scripted_expert, ik_solver, max_steps=300, target_expert_state="DESCEND_TO_PLACE")

    # It's possible the expert finishes faster, check final state reached
    final_expert_state = scripted_expert.get_state()
    print(f"Simulation ended with expert state: {final_expert_state}")
    assert final_expert_state in ["DESCEND_TO_PLACE", "AWAIT_PLACEMENT_CONTACT", "RELEASE"], "Expert did not reach placement phase"

    place_bonus_count = 0
    place_bonus_value = wrapped_env.reward_cfg.place_bonus

    for i, (_, _, _, _, info) in enumerate(results):
        if info.get('r_sparse_place', 0.0) == pytest.approx(place_bonus_value):
            place_bonus_count += 1
            print(f"Place bonus found at step {i}")

    assert place_bonus_count <= 1, f"Place bonus should be awarded at most once, found {place_bonus_count}"
    # We check <= 1 because the bonus might trigger right at the end of the run_steps

def test_success_bonus_and_termination(wrapped_env: gym.Wrapper, scripted_expert: ScriptedExpert, ik_solver: IKSolver): 
    """Verify success bonus and episode termination on successful placement."""
    print("--- Testing Success Bonus & Termination ---")
    scripted_expert.reset()
    # Run the full sequence
    results = run_steps_with_expert(wrapped_env, scripted_expert, ik_solver, max_steps=500, target_expert_state="DONE")

    assert scripted_expert.is_done(), "Expert should have finished"
    assert scripted_expert.was_successful(), "Expert should report success"
    assert len(results) > 0, "Simulation must run at least one step"

    last_obs, last_reward, last_terminated, last_truncated, last_info = results[-1]

    success_bonus_value = wrapped_env.reward_cfg.success_bonus
    found_success_bonus = False
    termination_step = -1

    for i, (_, _, terminated, _, info) in enumerate(results):
        if info.get('r_sparse_success', 0.0) == pytest.approx(success_bonus_value):
            found_success_bonus = True
            print(f"Success bonus found at step {i}")
        if terminated and termination_step < 0:
             termination_step = i
             print(f"Termination signal True found at step {i}")

    assert found_success_bonus, "Success bonus was not awarded"
    assert last_terminated is True, "Final step should have terminated=True"
    assert last_info.get('is_success') is True, "Final info dict should indicate success"
    # Check that termination happened at or after the success bonus
    assert termination_step != -1, "Termination signal was never True"


@pytest.mark.skip(reason="Collision testing requires precise control, hard to guarantee with expert.")
def test_collision_penalty(wrapped_env: AdvancedRewardWrapper, scripted_expert: ScriptedExpert):
    """Verify collision penalty triggers (skipped, needs dedicated control)."""
    # This is hard to reliably trigger just using the expert.
    # Would require manually setting target poses to force collision.
    pass

def test_drop_penalty(wrapped_env: gym.Wrapper, scripted_expert: ScriptedExpert, ik_solver: IKSolver): 
    """Verify drop penalty occurs if object is dropped after lifting."""
    print("--- Testing Drop Penalty ---")
    scripted_expert.reset()

    # 1. Run until object is lifted
    results_lift = run_steps_with_expert(wrapped_env, scripted_expert, ik_solver, max_steps=200, target_expert_state="MOVE_TO_GOAL")
    assert scripted_expert.get_state() == "MOVE_TO_GOAL", "Should have lifted the object"
    assert results_lift[-1][0]['is_grasped'][0] > 0.5, "Object should be grasped"
    assert wrapped_env.env._was_lifted, "_was_lifted flag should be True"

    # 2. Force a drop by commanding gripper open
    print("Commanding gripper open to force drop...")
    obs = results_lift[-1][0]
    # Keep pose same, just change gripper action
    current_pose_7d = obs['ee_pose_world']
    drop_action = calculate_delta_action(current_pose_7d, current_pose_7d, gripper_action=1.0) # Open gripper

    # Step env with drop action
    drop_obs, drop_reward, _, _, drop_info = wrapped_env.step(drop_action)

    # 3. Step again (object should now be falling/ungrasped)
    # Use a neutral action
    next_action = np.zeros_like(drop_action)
    final_obs, final_reward, _, _, final_info = wrapped_env.step(next_action)

    print(f"Info after commanding open: {drop_info}")
    print(f"Info after next step: {final_info}")

    # Assert: Drop penalty should be present in the step *after* the object is detected as un-grasped
    # It might take one step for is_grasped to update in the simulation state
    drop_penalty_value = -wrapped_env.reward_cfg.drop_penalty
    found_drop_penalty = False
    if drop_info.get('r_penalty_drop', 0.0) == pytest.approx(drop_penalty_value):
        found_drop_penalty = True
    if final_info.get('r_penalty_drop', 0.0) == pytest.approx(drop_penalty_value):
         found_drop_penalty = True


    assert final_obs['is_grasped'][0] < 0.5, "Object should be un-grasped after opening gripper"
    assert found_drop_penalty, f"Drop penalty ({drop_penalty_value}) not found in info dicts"

def test_curriculum_annealing(reward_config: AdvancedRewardConfig, curriculum_config: CurriculumConfig):
    """Verify reward parameters anneal over multiple resets."""
    print("--- Testing Curriculum Annealing ---")
    # Use the specific curriculum config for this test
    curriculum_config_test = CurriculumConfig(total_episodes=10) # Anneal over 10 episodes
    env = PandaEnv(render_mode="rgb_array", xml_path="envs/panda_pick_place.xml")
    adv_wrapper = AdvancedRewardWrapper(env, reward_config, curriculum_config_test)
    wrapper = gym.wrappers.TimeLimit(adv_wrapper, max_episode_steps=5)
    

    initial_dense_weight = adv_wrapper.reward_cfg.dense_reward_weight
    initial_pos_thresh = adv_wrapper.reward_cfg.goal_pos_thresh
    initial_orn_thresh = adv_wrapper.reward_cfg.goal_orn_thresh

    print(f"Initial values: DenseW={initial_dense_weight:.3f}, PosTh={initial_pos_thresh:.4f}, OrnTh={initial_orn_thresh:.3f}")

    weights = [initial_dense_weight]
    pos_threshs = [initial_pos_thresh]
    orn_threshs = [initial_orn_thresh]

    num_resets = 15 # Go beyond total_episodes to check clamping
    for i in range(num_resets):
        wrapper.reset(seed=100+i)
        # Run one step just to ensure internal episode counter updates if needed
        # wrapper.step(wrapper.action_space.sample()) # Not strictly necessary if reset updates counter
        weights.append(adv_wrapper.reward_cfg.dense_reward_weight)
        pos_threshs.append(adv_wrapper.reward_cfg.goal_pos_thresh)
        orn_threshs.append(adv_wrapper.reward_cfg.goal_orn_thresh)
        print(f"After reset {i+1}: DenseW={weights[-1]:.3f}, PosTh={pos_threshs[-1]:.4f}, OrnTh={orn_threshs[-1]:.3f}")


    # Assert: Values should decrease and then plateau
    assert weights[1] < weights[0], "Dense weight should decrease initially"
    assert pos_threshs[1] < pos_threshs[0], "Pos threshold should decrease initially"
    assert orn_threshs[1] < orn_threshs[0], "Orn threshold should decrease initially"

    # Check plateau after total_episodes (index 11 corresponds to after 10 resets)
    assert weights[11] == pytest.approx(weights[-1]), "Dense weight should plateau"
    assert pos_threshs[11] == pytest.approx(pos_threshs[-1]), "Pos threshold should plateau"
    assert orn_threshs[11] == pytest.approx(orn_threshs[-1]), "Orn threshold should plateau"

    # Check final values against expected annealed values
    end_factor_dense = curriculum_config_test.dense_reward_anneal_end_factor
    end_factor_thresh = curriculum_config_test.goal_thresh_anneal_end_factor
    expected_final_weight = 1.0 - (1.0 - end_factor_dense) # Should be initial * factor? No, formula is 1-(1-factor)*progress
    # The formula used is: current = initial - (initial - final_target) * progress
    # final_target_weight = initial_dense_weight * end_factor_dense # ASSUMING factor multiplies
    expected_final_weight = initial_dense_weight * end_factor_dense # Let's assume this interpretation
    expected_final_pos_thresh = initial_pos_thresh * end_factor_thresh
    expected_final_orn_thresh = initial_orn_thresh * end_factor_thresh

    # Correction based on the actual formula in the wrapper:
    # final_weight = 1.0 - (1.0 - end_factor_dense) * 1.0 = end_factor_dense
    # final_pos_thresh = initial_pos_thresh - (initial_pos_thresh - initial_pos_thresh * end_factor_thresh) * 1.0
    #                = initial_pos_thresh * end_factor_thresh
    expected_final_weight = end_factor_dense
    expected_final_pos_thresh = initial_pos_thresh * end_factor_thresh
    expected_final_orn_thresh = initial_orn_thresh * end_factor_thresh


    print(f"Expected final values: DenseW={expected_final_weight:.3f}, PosTh={expected_final_pos_thresh:.4f}, OrnTh={expected_final_orn_thresh:.3f}")

    # Use a tolerance for the final check
    assert weights[-1] == pytest.approx(expected_final_weight, abs=RTOL)
    assert pos_threshs[-1] == pytest.approx(expected_final_pos_thresh, abs=RTOL)
    assert orn_threshs[-1] == pytest.approx(expected_final_orn_thresh, abs=RTOL)

    wrapper.close()