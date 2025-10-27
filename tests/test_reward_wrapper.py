# FILE: tests/test_reward_wrapper.py
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R

# Adjust import paths based on your project structure
from utils.rl_reward_wrapper import AdvancedRewardWrapper, AdvancedRewardConfig, CurriculumConfig, PhysicalState
from tests.mocks import MockPandaEnv # Import from the mocks file

# Floating point tolerance
RTOL = 1e-5

# --- Fixtures ---

@pytest.fixture
def mock_env() -> MockPandaEnv:
    """Provides a fresh instance of the mock environment for each test."""
    return MockPandaEnv()

@pytest.fixture
def reward_config() -> AdvancedRewardConfig:
    """Provides a default reward configuration."""
    return AdvancedRewardConfig()

@pytest.fixture
def curriculum_config() -> CurriculumConfig:
    """Provides a default curriculum configuration."""
    # Set total_episodes high to avoid significant annealing during a single test step
    return CurriculumConfig(total_episodes=10000)

@pytest.fixture
def reward_wrapper(mock_env: MockPandaEnv,
                   reward_config: AdvancedRewardConfig,
                   curriculum_config: CurriculumConfig) -> AdvancedRewardWrapper:
    """Provides an instance of the reward wrapper initialized with the mock env."""
    return AdvancedRewardWrapper(mock_env, reward_config, curriculum_config)

# --- Helper to create observation dictionary ---

def create_obs_dict(
    ee_pos: list = [0.5, 0.0, 0.6],
    ee_orn_xyzw: list = [0.0, 1.0, 0.0, 0.0], # Default downward
    obj_pos: list = [0.6, 0.1, 0.42], # On table (z=0.4 + 0.02 half-height)
    obj_orn_xyzw: list = [0.0, 0.0, 0.0, 1.0], # Default identity
    goal_pos: list = [0.7, -0.1, 0.42],
    goal_orn_xyzw: list = [0.0, 0.0, 0.0, 1.0],
    is_grasped: float = 0.0,
    gripper_qpos: list = [0.04, 0.04], # Open
    obj_vel: list = [0.0] * 6,
    robot_base_pos: list = [0.0, 0.0, 0.0]
) -> dict:
    """Creates a valid observation dictionary with default or specified values."""
    proprio = np.zeros(22)
    proprio[20:22] = gripper_qpos # Set gripper joint positions
    # Mock forces (indices 14-20) - assumed zero unless testing grasp force potential
    return {
        'ee_pose_world': np.array(ee_pos + ee_orn_xyzw, dtype=np.float32),
        'object_pos_world': np.array(obj_pos, dtype=np.float32),
        'object_orn_world': np.array(obj_orn_xyzw, dtype=np.float32),
        'goal_pos_world': np.array(goal_pos, dtype=np.float32),
        'goal_orn_world': np.array(goal_orn_xyzw, dtype=np.float32),
        'is_grasped': np.array([is_grasped], dtype=np.float32),
        'proprio': proprio,
        'object_vel': np.array(obj_vel, dtype=np.float32),
        'robot_base_pos_world': np.array(robot_base_pos, dtype=np.float32),
    }

# --- Test Functions ---

def test_initialization(reward_wrapper: AdvancedRewardWrapper):
    """Test if the wrapper initializes correctly."""
    assert reward_wrapper is not None
    assert reward_wrapper._last_potential == 0.0
    assert reward_wrapper.TABLE_Z == pytest.approx(0.4)
    assert reward_wrapper.OBJECT_START_Z == pytest.approx(0.4 + 0.02) # table + half-height
    # Check if robot geom IDs were identified (mock IDs: 2, 3, 4)
    assert set(reward_wrapper.robot_geom_ids) == {2, 3, 4}
    assert reward_wrapper.table_geom_id == 0

def test_reset_updates_state(reward_wrapper: AdvancedRewardWrapper, mock_env: MockPandaEnv):
    """Test if reset correctly initializes internal states based on the first observation."""
    initial_obj_z = 0.42
    obs = create_obs_dict(obj_pos=[0.6, 0.1, initial_obj_z])
    mock_env.set_observation(obs)

    _, info = reward_wrapper.reset()

    assert reward_wrapper._initial_object_z == pytest.approx(initial_obj_z)
    assert reward_wrapper._last_potential != 0.0 # Potential should be calculated
    assert reward_wrapper._was_lifted is False
    assert reward_wrapper._current_episode == 1
    assert not reward_wrapper._given_grasp_bonus
    assert not reward_wrapper._given_lift_bonus
    assert not reward_wrapper._given_place_bonus

def test_potential_shaping_reach(reward_wrapper: AdvancedRewardWrapper, mock_env: MockPandaEnv):
    """Test dense reward during reaching phase."""
    # State 1: EE far from object
    obs1 = create_obs_dict(ee_pos=[0.4, -0.2, 0.7], obj_pos=[0.6, 0.0, 0.42])
    mock_env.set_observation(obs1)
    _, _ = reward_wrapper.reset()
    potential1 = reward_wrapper._last_potential

    # State 2: EE closer to object hover position
    hover_pos = [0.6, 0.0, 0.42 + 0.02 + reward_wrapper.reward_cfg.hover_height] # obj_top_z + hover
    obs2 = create_obs_dict(ee_pos=hover_pos, obj_pos=[0.6, 0.0, 0.42])
    mock_env.set_observation(obs2)
    action = np.zeros(mock_env.action_space.shape)
    _, reward, _, _, info = reward_wrapper.step(action)

    potential2 = reward_wrapper._last_potential
    dense_reward_expected = potential2 - potential1

    # Assert: Potential should increase (dense reward positive) when moving towards hover
    assert potential2 > potential1
    assert reward == pytest.approx(dense_reward_expected) # Only dense reward here
    assert info['r_dense_shaped'] == pytest.approx(dense_reward_expected)
    assert info['r_sparse_event'] == 0.0
    assert info['r_penalty_total'] == 0.0 # Zero action penalty

def test_grasp_bonus(reward_wrapper: AdvancedRewardWrapper, mock_env: MockPandaEnv):
    """Test if the grasp bonus is awarded correctly once."""
    # State 1: Not grasped
    obs1 = create_obs_dict(is_grasped=0.0)
    mock_env.set_observation(obs1)
    _, _ = reward_wrapper.reset()
    potential1 = reward_wrapper._last_potential
    assert not reward_wrapper._given_grasp_bonus

    # State 2: Grasped
    obs2 = create_obs_dict(is_grasped=1.0)
    mock_env.set_observation(obs2)
    action = np.zeros(mock_env.action_space.shape)
    _, reward, _, _, info = reward_wrapper.step(action)
    potential2 = reward_wrapper._last_potential

    dense_reward = potential2 - potential1
    expected_reward = dense_reward + reward_wrapper.reward_cfg.grasp_bonus
    assert reward == pytest.approx(expected_reward)
    assert info['r_sparse_event'] == pytest.approx(reward_wrapper.reward_cfg.grasp_bonus)
    assert 'r_sparse_grasp' in info
    assert reward_wrapper._given_grasp_bonus # Flag should be set

    # State 3: Still grasped (bonus should not be given again)
    obs3 = create_obs_dict(is_grasped=1.0) # Assume negligible potential change
    mock_env.set_observation(obs3)
    _, reward, _, _, info = reward_wrapper.step(action)
    potential3 = reward_wrapper._last_potential
    dense_reward_step3 = potential3 - potential2

    expected_reward_step3 = dense_reward_step3 # No sparse bonus
    assert reward == pytest.approx(expected_reward_step3)
    assert info['r_sparse_event'] == 0.0
    assert 'r_sparse_grasp' not in info # Bonus not given again

def test_lift_bonus_and_flag(reward_wrapper: AdvancedRewardWrapper, mock_env: MockPandaEnv):
    """Test lift bonus and the _was_lifted flag."""
    initial_obj_z = 0.42
    # State 1: Grasped on table
    obs1 = create_obs_dict(obj_pos=[0.6, 0.1, initial_obj_z], is_grasped=1.0)
    mock_env.set_observation(obs1)
    _, _ = reward_wrapper.reset()
    potential1 = reward_wrapper._last_potential
    # Need to trigger grasp bonus first to make logic simpler here
    reward_wrapper._given_grasp_bonus = True
    assert not reward_wrapper._was_lifted

    # State 2: Lifted slightly (below threshold)
    lifted_z_low = initial_obj_z + reward_wrapper.reward_cfg.lift_height_thresh * 0.5
    obs2 = create_obs_dict(obj_pos=[0.6, 0.1, lifted_z_low], is_grasped=1.0)
    mock_env.set_observation(obs2)
    action = np.zeros(mock_env.action_space.shape)
    _, reward, _, _, info = reward_wrapper.step(action)
    potential2 = reward_wrapper._last_potential

    # Assert: Potential increases, dense reward positive, no bonus yet, _was_lifted still False
    assert potential2 > potential1
    assert info['r_sparse_event'] == 0.0
    assert not reward_wrapper._given_lift_bonus
    assert not reward_wrapper._was_lifted # Not set until *after* step processing

    # State 3: Lifted above threshold
    lifted_z_high = initial_obj_z + reward_wrapper.reward_cfg.lift_height_thresh * 1.5
    obs3 = create_obs_dict(obj_pos=[0.6, 0.1, lifted_z_high], is_grasped=1.0)
    mock_env.set_observation(obs3)
    _, reward, _, _, info = reward_wrapper.step(action)
    potential3 = reward_wrapper._last_potential

    dense_reward = potential3 - potential2
    expected_reward = dense_reward + reward_wrapper.reward_cfg.lift_bonus
    assert reward == pytest.approx(expected_reward)
    assert info['r_sparse_event'] == pytest.approx(reward_wrapper.reward_cfg.lift_bonus)
    assert 'r_sparse_lift' in info
    assert reward_wrapper._given_lift_bonus
    assert reward_wrapper._was_lifted # Should be set now (checked *after* step)

def test_place_bonus(reward_wrapper: AdvancedRewardWrapper, mock_env: MockPandaEnv):
    """Test place bonus."""
    goal_pos = [0.7, -0.1, 0.42]
    # State 1: Lifted, far from goal
    obs1 = create_obs_dict(obj_pos=[0.6, 0.1, 0.5], goal_pos=goal_pos, is_grasped=1.0)
    mock_env.set_observation(obs1)
    _, _ = reward_wrapper.reset()
    potential1 = reward_wrapper._last_potential
    reward_wrapper._given_grasp_bonus = True # Assume already grasped/lifted
    reward_wrapper._given_lift_bonus = True
    reward_wrapper._was_lifted = True
    assert not reward_wrapper._given_place_bonus

    # State 2: Near goal, still grasped (within place_dist_thresh)
    place_dist = reward_wrapper.reward_cfg.place_dist_thresh
    near_goal_pos = np.array(goal_pos) + np.array([0, 0, place_dist * 0.5]) # Slightly above
    obs2 = create_obs_dict(obj_pos=near_goal_pos.tolist(), goal_pos=goal_pos, is_grasped=1.0)
    mock_env.set_observation(obs2)
    action = np.zeros(mock_env.action_space.shape)
    _, reward, _, _, info = reward_wrapper.step(action)
    potential2 = reward_wrapper._last_potential

    dense_reward = potential2 - potential1
    expected_reward = dense_reward + reward_wrapper.reward_cfg.place_bonus
    assert reward == pytest.approx(expected_reward)
    assert info['r_sparse_event'] == pytest.approx(reward_wrapper.reward_cfg.place_bonus)
    assert 'r_sparse_place' in info
    assert reward_wrapper._given_place_bonus

def test_success_bonus_and_termination(reward_wrapper: AdvancedRewardWrapper, mock_env: MockPandaEnv):
    """Test success bonus and termination condition."""
    goal_pos = [0.7, -0.1, 0.42]
    goal_dist = reward_wrapper.reward_cfg.goal_pos_thresh
    # State 1: Very near goal, still grasped
    near_goal_pos = np.array(goal_pos) + np.array([0, 0, goal_dist * 0.5])
    obs1 = create_obs_dict(obj_pos=near_goal_pos.tolist(), goal_pos=goal_pos, is_grasped=1.0)
    mock_env.set_observation(obs1)
    _, _ = reward_wrapper.reset()
    potential1 = reward_wrapper._last_potential
    reward_wrapper._given_grasp_bonus = True
    reward_wrapper._given_lift_bonus = True
    reward_wrapper._given_place_bonus = True # Assume place bonus already given
    reward_wrapper._was_lifted = True

    # State 2: At goal, released
    at_goal_pos = goal_pos
    # Need to set object orientation close to goal too
    obs2 = create_obs_dict(
        obj_pos=at_goal_pos, goal_pos=goal_pos, is_grasped=0.0,
        obj_orn_xyzw=[0,0,0,1], goal_orn_xyzw=[0,0,0,1] # Assume simple goal orn
    )
    mock_env.set_observation(obs2)
    action = np.zeros(mock_env.action_space.shape) # Assume release action here
    _, reward, terminated, truncated, info = reward_wrapper.step(action)
    potential2 = reward_wrapper._last_potential

    dense_reward = potential2 - potential1
    expected_reward = dense_reward + reward_wrapper.reward_cfg.success_bonus
    assert reward == pytest.approx(expected_reward)
    assert info['r_sparse_event'] == pytest.approx(reward_wrapper.reward_cfg.success_bonus)
    assert 'r_sparse_success' in info
    assert terminated is True # Success should terminate the episode
    assert info.get('is_success') is True

def test_penalties(reward_wrapper: AdvancedRewardWrapper, mock_env: MockPandaEnv):
    """Test various penalty conditions."""
    # Setup initial state (e.g., grasped and lifted)
    initial_obj_z = 0.42
    lifted_z = initial_obj_z + 0.1
    obs = create_obs_dict(obj_pos=[0.6, 0.1, lifted_z], is_grasped=1.0)
    mock_env.set_observation(obs)
    _, _ = reward_wrapper.reset()
    reward_wrapper._was_lifted = True # Manually set for drop penalty test
    potential1 = reward_wrapper._last_potential

    # --- Test Action/Jerk Penalty ---
    action = np.array([0.5] * mock_env.action_space.shape[0]) # Non-zero action
    _, reward, _, _, info = reward_wrapper.step(action)
    potential2 = reward_wrapper._last_potential
    dense_reward = potential2 - potential1
    action_penalty = -reward_wrapper.reward_cfg.action_penalty * np.sum(np.square(action))
    jerk_penalty = -reward_wrapper.reward_cfg.jerk_penalty * np.sum(np.square(action - 0.0)) # Jerk from zero
    expected_reward = dense_reward + action_penalty + jerk_penalty
    assert reward == pytest.approx(expected_reward)
    assert info['r_penalty_total'] == pytest.approx(action_penalty + jerk_penalty)
    assert info['r_penalty_action'] == pytest.approx(action_penalty)
    assert info['r_penalty_jerk'] == pytest.approx(jerk_penalty)

    # --- Test Collision Penalty ---
    # Set mock collision between robot body (geom 2) and table (geom 0)
    mock_env.set_collisions([(2, 0)])
    # Reset last potential etc for clean penalty check
    reward_wrapper._last_potential = potential2
    reward_wrapper._last_action = action
    # Take another step (action value doesn't matter much here)
    _, reward, _, _, info = reward_wrapper.step(action * 0.1)
    potential3 = reward_wrapper._last_potential
    dense_reward = potential3 - potential2
    action_penalty = -reward_wrapper.reward_cfg.action_penalty * np.sum(np.square(action*0.1))
    jerk_penalty = -reward_wrapper.reward_cfg.jerk_penalty * np.sum(np.square(action*0.1 - action))
    collision_penalty = -reward_wrapper.reward_cfg.contact_penalty
    expected_reward = dense_reward + action_penalty + jerk_penalty + collision_penalty
    assert reward == pytest.approx(expected_reward)
    assert info['r_penalty_total'] == pytest.approx(action_penalty + jerk_penalty + collision_penalty)
    assert info['r_penalty_contact'] == pytest.approx(collision_penalty)

    # --- Test Drop Penalty ---
    mock_env.set_collisions([]) # Clear collisions
    obs_dropped = create_obs_dict(obj_pos=[0.6, 0.1, lifted_z - 0.05], is_grasped=0.0) # Dropped
    mock_env.set_observation(obs_dropped)
    # Reset last potential etc.
    reward_wrapper._last_potential = potential3
    reward_wrapper._last_action = action * 0.1
    # Take step
    _, reward, _, _, info = reward_wrapper.step(np.zeros_like(action))
    potential4 = reward_wrapper._last_potential
    dense_reward = potential4 - potential3
    drop_penalty = -reward_wrapper.reward_cfg.drop_penalty
    # Action/jerk penalties are also active
    action_penalty = 0.0
    jerk_penalty = -reward_wrapper.reward_cfg.jerk_penalty * np.sum(np.square(0.0 - action*0.1))
    expected_reward = dense_reward + drop_penalty + action_penalty + jerk_penalty
    assert reward == pytest.approx(expected_reward)
    assert info['r_penalty_total'] == pytest.approx(drop_penalty + action_penalty + jerk_penalty)
    assert info['r_penalty_drop'] == pytest.approx(drop_penalty)

    # --- Test Instability Penalty ---
    obs_unstable = create_obs_dict(
        obj_pos=[0.6, 0.1, lifted_z],
        is_grasped=1.0,
        obj_vel=[0.1, -0.05, 0.02, 0, 0, 0] # Non-zero velocity
    )
    mock_env.set_observation(obs_unstable)
    # Reset last potential etc. (Object didn't drop, so _was_lifted is still true)
    reward_wrapper._last_potential = potential4
    reward_wrapper._last_action = np.zeros_like(action)
    # Take step
    _, reward, _, _, info = reward_wrapper.step(np.zeros_like(action))
    potential5 = reward_wrapper._last_potential
    dense_reward = potential5 - potential4
    obj_vel_norm = np.linalg.norm(np.array([0.1, -0.05, 0.02])) + np.linalg.norm(np.array([0,0,0]))
    instability_penalty = -reward_wrapper.reward_cfg.instability_penalty * obj_vel_norm
    expected_reward = dense_reward + instability_penalty # Action/jerk = 0
    assert reward == pytest.approx(expected_reward)
    assert info['r_penalty_total'] == pytest.approx(instability_penalty)
    assert info['r_penalty_instability'] == pytest.approx(instability_penalty)

# Add more tests as needed, e.g., for orientation potentials, curriculum updates etc.