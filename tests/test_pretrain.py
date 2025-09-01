# tests/test_pretrain.py
import unittest
import os
import tempfile
import numpy as np
import torch
from unittest.mock import patch, MagicMock
import logging
import gymnasium
from gymnasium import spaces
import gc

# The script we are testing
# We use a try-except block to handle different project structures
try:
    import pretrain_policy as pretrain
except ModuleNotFoundError:
    from .. import pretrain_policy as pretrain

# Suppress logger output for clean test results
logging.disable(logging.CRITICAL)


# --- Mock Classes for Dependencies ---

import gymnasium
from gymnasium import spaces

class MockPandaEnv(gymnasium.Env):
    """A fake PandaEnv that correctly inherits from gymnasium.Env for SB3 compatibility."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Define the required observation and action spaces for SB3
        self.observation_space = spaces.Dict({
            "image_primary": spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
            "proprio": spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
        })
        self.action_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)

    def reset(self, seed=None, options=None): # Updated signature for Gymnasium
        super().reset(seed=seed)
        obs = self.observation_space.sample()
        return obs, {}

    def get_base_pose(self):
        # Return a fixed, predictable base pose
        return np.array([0.1, 0.2, 0.3]), np.array([0.0, 0.0, 0.0, 1.0])

    def step(self, action): # Add a dummy step method, which is required
        obs, _ = self.reset()
        return obs, 0.0, False, False, {}

    def close(self):
        pass

class MockIKSolver:
    """A fake IKSolver that returns a predictable action."""
    def __init__(self, *args, **kwargs):
        pass # Ignore any arguments like urdf_path

    def compute_action(self, target_pose, current_joints):
        # Return a fixed 8D action vector
        return np.arange(8, dtype=np.float32)

class MockOctoModel:
    """A fake OctoModel that returns a predictable target pose."""
    def create_tasks(self, texts):
        return MagicMock() # The task object itself doesn't need to do anything

    def sample_actions(self, obs, task, rng):
        # Return a fixed 7D pose (pos, quat_xyzw)
        pose = np.arange(7, dtype=np.float32) / 10.0
        # Return in the shape OCTO provides: (batch, history, dims)
        return pose[np.newaxis, np.newaxis, :]

    @classmethod
    def load_pretrained(cls, name):
        return cls()


class TestPretrainScript(unittest.TestCase):
    """A test suite for the pretrain.py script."""

    def setUp(self):
        """Create a temporary directory for dataset and model outputs."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = os.path.join(self.temp_dir.name, "dataset")

    def tearDown(self):
        """Clean up the temporary directory."""
        self.temp_dir.cleanup()

    @classmethod
    def tearDownClass(cls):
        """Re-enable logging after all tests are done."""
        logging.disable(logging.NOTSET)

    @patch('pretrain_policy.PandaEnv', new=MockPandaEnv)
    @patch('pretrain_policy.IKSolver', new=MockIKSolver)
    @patch('pretrain_policy.OctoModel', new=MockOctoModel)
    def test_A_generate_dataset_streaming(self):
        """Test the dataset generation in streaming mode (saving to disk)."""
        num_samples = 10
        result = pretrain.generate_synthetic_dataset(
            num_samples=num_samples,
            output_dir=self.output_dir,
            stream_to_disk_threshold=5, # Force streaming mode
            urdf_path="dummy.urdf",
            octo_model_name="dummy-octo",
            quat_format="xyzw",
            write_placeholders_on_fail=True,
            jax_seed=0,
            retry_limit=1,
            deterministic_env_seed=0
        )

        # 1. Check the return value
        self.assertEqual(result['mode'], 'disk')
        self.assertEqual(result['n'], num_samples)
        self.assertTrue(os.path.exists(result['images_path']))
        self.assertTrue(os.path.exists(result['proprio_path']))
        self.assertTrue(os.path.exists(result['actions_path']))

        # 2. Check the saved files' content and format
        images = np.load(result['images_path'])
        actions = np.load(result['actions_path'])
        
        self.assertEqual(images.shape, (num_samples, 128, 128, 3))
        self.assertEqual(images.dtype, np.uint8)
        self.assertEqual(actions.shape[0], num_samples)
        
        # Verify the content of the action is what our MockIKSolver produces
        expected_action = np.arange(8, dtype=np.float32)
        self.assertTrue(np.allclose(actions[0, :8], expected_action))
        print("\n✅ [TestPretrain] Dataset generation (streaming to disk) is correct.")

    @patch('pretrain_policy.PandaEnv', new=MockPandaEnv)
    @patch('pretrain_policy.IKSolver', new=MockIKSolver)
    @patch('pretrain_policy.OctoModel', new=MockOctoModel)
    def test_B_pretrain_from_dataset(self):
        """Test the pre-training function using a generated disk dataset."""
        # First, generate a small dataset to use for pre-training
        dataset_info = pretrain.generate_synthetic_dataset(
            num_samples=20,
            output_dir=self.output_dir,
            stream_to_disk_threshold=5,
            urdf_path="dummy.urdf",
            octo_model_name="dummy-octo",
            quat_format="xyzw",
            write_placeholders_on_fail=True,
            jax_seed=0,
            retry_limit=1,
            deterministic_env_seed=0
        )

        save_path = os.path.join(self.temp_dir.name, "test_policy.zip")
        
        # Run pre-training for just one epoch to test the logic
        result_path = pretrain.pretrain_policy_from_dataset(
            dataset_obj=dataset_info,
            epochs=1,
            batch_size=4,
            save_path=save_path,
            lr=1e-4
        )

        # Check that the model was saved
        self.assertEqual(result_path, save_path)
        self.assertTrue(os.path.exists(save_path))
        print("✅ [TestPretrain] Pre-training function runs and saves a model correctly.")


if __name__ == '__main__':
    unittest.main(verbosity=2)