# test_panda_env_robust.py
import unittest
import numpy as np
import mujoco
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

# The class we are testing
from envs.panda_env import PandaEnv

class TestPandaEnvRobust(unittest.TestCase):
    """
    A comprehensive, production-grade test suite for the custom PandaEnv.

    This suite verifies:
    1. Full compliance with the Gymnasium API.
    2. The precise structure and properties of the observation and action spaces.
    3. The correct behavior of the custom `reset()` and `step()` methods.
    4. The correctness of the custom `get_base_pose()` helper method.
    """

    @classmethod
    def setUpClass(cls):
        """Load the environment once for all tests in this class for efficiency."""
        try:
            cls.env = PandaEnv()
        except Exception as e:
            cls.fail(f"Failed to initialize PandaEnv in setUpClass. Error: {e}")

    def test_A_gymnasium_compliance(self):
        """
        [Primary Check] Run the official Gymnasium environment checker.
        This is the most critical test for ensuring API compatibility.
        """
        try:
            # The check_env function is the gold standard for validating a custom env.
            # It performs a wide range of checks on reset(), step(), spaces, etc.
            check_env(self.env.unwrapped)
            print("\n✅ [TestPandaEnv] Gymnasium compliance check passed.")
        except Exception as e:
            self.fail(f"Gymnasium compliance check failed: {e}")

    def test_B_space_definitions(self):
        """
        [Contract Check] Verify the specific structure of the observation and action spaces.
        This ensures the environment's "contract" with the agent is correct.
        """
        # 1. Test Action Space
        self.assertIsInstance(self.env.action_space, spaces.Box, "Action space should be a Box.")
        self.assertEqual(self.env.action_space.shape, (8,), "Action space should have shape (8,).")

        # 2. Test Observation Space
        self.assertIsInstance(self.env.observation_space, spaces.Dict, "Observation space should be a Dict.")
        
        # Define the expected structure and properties of the observation space
        expected_obs_structure = {
            "image_primary": (spaces.Box, (256, 256, 3), np.uint8),
            "image_wrist": (spaces.Box, (128, 128, 3), np.uint8),
            # MODIFIED: Change expectation to Box with the correct shape and dtype
            "task_completed": (spaces.Box, (4,), np.int64),
            "timestep": (spaces.Box, (1,), np.int32),
            "internal_full_proprio": (spaces.Box, (14,), np.float32),
            "timestep_pad_mask": (spaces.Discrete, (), np.int64),
        }
        for key, (space_type, shape, dtype) in expected_obs_structure.items():
            with self.subTest(key=key):
                self.assertIn(key, self.env.observation_space.spaces, f"Key '{key}' missing from observation space.")
                space = self.env.observation_space.spaces[key]
                self.assertIsInstance(space, space_type, f"Space for '{key}' has wrong type.")
                if shape: # Discrete spaces don't have a .shape attribute
                    self.assertEqual(space.shape, shape, f"Space for '{key}' has wrong shape.")
                self.assertEqual(space.dtype, dtype, f"Space for '{key}' has wrong dtype.")
        
        print("✅ [TestPandaEnv] Observation and action spaces are correctly defined.")

        self.assertIn("pad_mask_dict", self.env.observation_space.spaces)
        pad_mask_space = self.env.observation_space.spaces["pad_mask_dict"]
        self.assertIsInstance(pad_mask_space, spaces.Dict)

        expected_pad_mask_structure = {
            "image_primary": (spaces.Discrete, (), np.int64),
            "image_wrist": (spaces.Discrete, (), np.int64),
            "timestep": (spaces.Discrete, (), np.int64),
        }

        for key, (space_type, shape, dtype) in expected_pad_mask_structure.items():
            with self.subTest(key=f"pad_mask_dict.{key}"):
                self.assertIn(key, pad_mask_space.spaces)
                space = pad_mask_space.spaces[key]
                self.assertIsInstance(space, space_type)
                self.assertEqual(space.dtype, dtype)

        print("✅ [TestPandaEnv] Observation and action spaces are correctly defined.")

    def test_C_reset_and_step_behavior(self):
        """
        [Behavior Check] Verify the core logic of reset() and step().
        """
        # 1. Test reset()
        obs, info = self.env.reset()
        self.assertIsInstance(info, dict, "reset() should return a dictionary for the info object.")
        self.assertTrue(self.env.observation_space.contains(obs), "The observation from reset() is not contained in the observation space.")
        self.assertEqual(self.env.timestep, 0, "Environment timestep should be reset to 0.")

        # 2. Test step()
        # Take a random valid action
        action = self.env.action_space.sample()
        step_result = self.env.step(action)
        self.assertEqual(len(step_result), 5, "step() should return a 5-tuple.")
        
        next_obs, reward, terminated, truncated, info = step_result
        self.assertTrue(self.env.observation_space.contains(next_obs), "The observation from step() is not contained in the observation space.")
        self.assertIsInstance(reward, float, "Reward should be a float.")
        self.assertIsInstance(terminated, bool, "Terminated flag should be a bool.")
        self.assertIsInstance(truncated, bool, "Truncated flag should be a bool.")
        self.assertEqual(self.env.timestep, 1, "Environment timestep should increment to 1 after one step.")
        
        print("✅ [TestPandaEnv] reset() and step() methods behave correctly.")

    def test_D_get_base_pose_method(self):
        """
        [Custom Method Check] Verify the get_base_pose() helper method.
        """
        pos, quat_xyzw = self.env.get_base_pose()

        # Check types and shapes
        self.assertIsInstance(pos, np.ndarray)
        self.assertIsInstance(quat_xyzw, np.ndarray)
        self.assertEqual(pos.shape, (3,), "Position should have shape (3,)")
        self.assertEqual(quat_xyzw.shape, (4,), "Quaternion should have shape (4,)")

        # Check values
        expected_pos = np.array([0., 0., 0.])
        np.testing.assert_allclose(pos, expected_pos, atol=1e-6, err_msg="Base position should be at the world origin.")

        # Add a sanity check for the quaternion
        self.assertAlmostEqual(np.linalg.norm(quat_xyzw), 1.0, places=6, msg="Quaternion should be a unit quaternion (norm ≈ 1.0).")
        
        print("✅ [TestPandaEnv] get_base_pose() method is correct and robust.")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up the environment once after all tests are done."""
        cls.env.close()

if __name__ == '__main__':
    unittest.main(verbosity=2)