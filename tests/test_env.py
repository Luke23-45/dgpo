# tests/test_panda_env.py

import pytest
import numpy as np
import gymnasium as gym
from envs.panda_env import PandaEnv # Adjust this import if your path is different

# =================
#  Pytest Fixture
# =================

@pytest.fixture
def env():
    """
    A pytest fixture to create and tear down the PandaEnv for each test.
    This ensures tests are independent and resources are cleaned up.
    """
    # Setup: Create the environment instance
    environment = PandaEnv(
        xml_path="envs/panda_pick_place.xml", # Make sure this path is correct from your project root
        render_mode="rgb_array",
        enable_domain_randomization=True
    )
    yield environment # The test runs at this point
    # Teardown: Clean up the environment
    environment.close()

# ==========================
#  1. API & Sanity Checks
# ==========================

def test_env_creation_and_spaces(env):
    """
    Tests that the environment can be created and that its observation
    and action spaces have the expected structure, shape, and dtype.
    """
    assert isinstance(env, PandaEnv)
    assert isinstance(env.observation_space, gym.spaces.Dict)
    assert isinstance(env.action_space, gym.spaces.Box)
    
    # Check observation space details
    obs_space = env.observation_space.spaces
    assert obs_space["image_primary"].shape == (256, 256, 3)
    assert obs_space["image_primary"].dtype == np.uint8
    assert obs_space["image_wrist"].shape == (128, 128, 3)
    assert obs_space["image_wrist"].dtype == np.uint8
    assert obs_space["proprio"].shape == (14,)
    assert obs_space["proprio"].dtype == np.float32

    # Check action space details
    assert env.action_space.shape == (8,)
    assert env.action_space.dtype == np.float32

def test_reset_api_compliance(env):
    """
    Tests that the reset() method returns an observation and info dict
    and that the observation conforms to the observation space.
    """
    obs, info = env.reset()
    assert isinstance(obs, dict)
    assert isinstance(info, dict)
    
    # Create a "standard" observation dict by filtering the expert obs
    standard_obs = {key: obs[key] for key in env.observation_space.keys()}
    assert env.observation_space.contains(standard_obs)

    # Check for expert-specific keys in the full observation from reset()
    assert "ee_pose_world" in obs
    assert obs["ee_pose_world"].shape == (7,)

    
def test_step_api_compliance(env):
    """
    Tests that the step() method returns the correct 5-tuple
    (obs, reward, terminated, truncated, info) with correct types.
    """
    env.reset()
    random_action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(random_action)

    assert isinstance(obs, dict)
    # Create a "standard" observation dict by filtering the expert obs
    standard_obs = {key: obs[key] for key in env.observation_space.keys()}
    assert env.observation_space.contains(standard_obs)

    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
# ==================================
#  2. Logical Correctness Tests
# ==================================

@pytest.mark.parametrize("reset_run", range(5)) # Run this test 5 times for robustness
def test_visibility_guarantee_on_reset(env, reset_run):
    """
    CRITICAL TEST: Verifies that after a reset, both the object and the goal
    are visible within the frame of the primary camera. This tests the core logic
    of the data generation pipeline.
    """
    obs, _ = env.reset()
    
    # Get the ground-truth positions of the object and goal
    object_pos = obs["object_pos_world"]
    goal_pos = obs["goal_pos_world"]

    # Use the environment's internal check to verify visibility
    is_object_visible, debug_obj = env._is_pos_in_camera_view(object_pos, "fixed_camera")
    is_goal_visible, debug_goal = env._is_pos_in_camera_view(goal_pos, "fixed_camera")

    assert is_object_visible, f"Object is not visible on reset run {reset_run}. Debug: {debug_obj}"
    assert is_goal_visible, f"Goal is not visible on reset run {reset_run}. Debug: {debug_goal}"

def test_domain_randomization_changes_state(env):
    """
    Tests that domain randomization is active by checking that key environment
    properties (camera pose, light color) are different across two resets.
    """
    # --- First Reset ---
    env.reset()
    cam_pos_1 = env.model.cam_pos[env.camera_id].copy()
    light_diffuse_1 = env.model.light_diffuse[env.light_id].copy()
    table_mat_id_1 = env.model.geom_matid[env.table_geom_id]

    # --- Second Reset ---
    env.reset()
    cam_pos_2 = env.model.cam_pos[env.camera_id].copy()
    light_diffuse_2 = env.model.light_diffuse[env.light_id].copy()
    table_mat_id_2 = env.model.geom_matid[env.table_geom_id]

    # Assert that the properties have changed.
    # We expect these to be different due to randomization.
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(cam_pos_1, cam_pos_2, "Camera position did not change between resets.")

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(light_diffuse_1, light_diffuse_2, "Light color did not change between resets.")
    
    # It's possible for the texture to be the same by chance, but less likely
    # over many runs. A simple inequality check is fine here.
    assert table_mat_id_1 != table_mat_id_2 or len(env.dr_config.table_textures) <= 1, \
        "Table texture did not change. (This could fail by chance; run again to confirm)"


# =================================
#  3. Component Functionality Tests
# =================================

def test_rendering_outputs(env):
    """
    Tests that rendering from both primary and wrist cameras returns a
    correctly shaped and typed numpy array without crashing.
    """
    env.reset()
    primary_img = env.render(camera_name="fixed_camera")
    wrist_img = env.render(camera_name="wrist_camera")

    assert isinstance(primary_img, np.ndarray)
    assert primary_img.shape == (256, 256, 3)
    assert primary_img.dtype == np.uint8

    assert isinstance(wrist_img, np.ndarray)
    assert wrist_img.shape == (128, 128, 3)
    assert wrist_img.dtype == np.uint8

def test_expert_obs_keys(env):
    """
    Verifies that the observation dictionary returned for the expert contains
    all the necessary ground-truth keys.
    """
    obs = env.get_expert_obs()
    
    required_keys = [
        "image_primary", "image_wrist", "proprio", "task_completed",
        "ee_pose_world", "object_pos_world", "goal_pos_world",
        "internal_full_proprio"
    ]

    for key in required_keys:
        assert key in obs, f"Expert observation is missing required key: '{key}'"


if __name__ == "__main__":
    """
    This block allows the test file to be run directly as a script
    using `python -m tests.test_panda_env`.
    
    It programmatically invokes pytest on this file, including any
    command-line arguments you might pass to it.
    """
    import sys
    # This is the magic line that makes the file a self-executing test suite.
    # It tells pytest to start a session and uses sys.argv to pass along
    # any arguments like -v, --json-report, etc.
    # We explicitly add `__file__` to ensure pytest targets this file.
    exit_code = pytest.main([__file__] + sys.argv[1:])
    sys.exit(exit_code)