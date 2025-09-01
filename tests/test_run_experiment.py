# test_run_experiment.py
import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import torch
import gymnasium
import logging

# The script we are testing
import run_experiment

# Suppress the logger output during tests to keep the test results clean
logging.disable(logging.CRITICAL)


class TestRunExperiment(unittest.TestCase):
    """
    A test suite for the experiment runner script (run_experiment.py).

    This suite uses mocking to test the integration logic of the script
    without needing to create real environments or train models, making the
    tests fast and reliable.
    """

    def setUp(self):
        """Create a temporary directory for each test to use for file outputs."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.run_name = "test_run"
        # Define the expected save directory based on the script's logic
        self.save_dir = os.path.join(self.temp_dir.name, "trained_models", self.run_name)
        # The script creates this directory, so we don't need to os.makedirs here.

    def tearDown(self):
        """Clean up the temporary directory after each test."""
        self.temp_dir.cleanup()

    def _create_dummy_bc_checkpoint(self, path):
        """Helper to create a fake BC checkpoint file for testing the loading logic."""
        dummy_state_dict = {"layer1.weight": torch.randn(10, 10)}
        # Save in the format the robust loader expects
        torch.save({"model_state_dict": dummy_state_dict}, path)
    
    @classmethod
    def tearDownClass(cls):
        """Re-enable logging after all tests in this class are done."""
        logging.disable(logging.NOTSET)

    @patch('torch.cuda.is_available')
    def test_A_resolve_device(self, mock_cuda_available):
        """Unit test for the resolve_device helper function."""
        # Case 1: CUDA is available
        mock_cuda_available.return_value = True
        self.assertEqual(run_experiment.resolve_device("auto"), "cuda")

        # Case 2: CUDA is not available
        mock_cuda_available.return_value = False
        self.assertEqual(run_experiment.resolve_device("auto"), "cpu")

        # Case 3: A specific device is passed through
        self.assertEqual(run_experiment.resolve_device("cpu"), "cpu")
        self.assertEqual(run_experiment.resolve_device("cuda:1"), "cuda:1")
        print("\n✅ [TestRunner] `resolve_device` helper is correct.")

    @patch('run_experiment.transfer_bc_weights')
    @patch('run_experiment.PPO')
    @patch('run_experiment.make_vec_env')
    def test_B_run_experiment_full_flow(self, mock_make_vec_env, mock_PPO, mock_transfer_weights):
        """
        Integration test for the main run_experiment function's logic.
        This test mocks the environment, agent, and training to run quickly.
        """
        # --- 1. Setup Mocks ---
        mock_env = MagicMock()
        mock_env.envs = [MagicMock()]
        mock_env.envs[0].observation_space = gymnasium.spaces.Dict({
            "state": gymnasium.spaces.Box(low=-1, high=1, shape=(10,))
        })
        mock_env.envs[0].action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(8,))
        mock_make_vec_env.return_value = mock_env

        mock_agent_instance = MagicMock()
        mock_PPO.return_value = mock_agent_instance

        # Create the checkpoint in the temporary directory
        bc_model_path = os.path.join(self.temp_dir.name, "dummy_bc.pth")
        self._create_dummy_bc_checkpoint(bc_model_path)

        # --- 2. Run the experiment function ---
        original_cwd = os.getcwd()
        try:
            # Change CWD so that relative paths are created inside our temp folder
            os.chdir(self.temp_dir.name)
            
            run_experiment.run_experiment(
                xml_path="fake.xml",
                # The checkpoint path now needs to be absolute since we changed directories
                bc_model_path=bc_model_path,
                total_timesteps=100,
                run_name=self.run_name,
                save_freq=50,
                seed=42,
                device_arg="cpu",
                n_envs=1
            )
        finally:
            # Always change back to the original directory to not affect other tests
            os.chdir(original_cwd)

        # --- 3. Assertions: Verify the integration logic ---
        mock_make_vec_env.assert_called_once()

        mock_PPO.assert_called_once()
        call_args = mock_PPO.call_args.kwargs
        self.assertEqual(call_args['policy'], 'MultiInputPolicy')
        self.assertEqual(call_args['device'], 'cpu')

        mock_transfer_weights.assert_called_once()

        mock_agent_instance.learn.assert_called_once()
        self.assertEqual(mock_agent_instance.learn.call_args.kwargs['total_timesteps'], 100)

        # The script calls save() with a relative path. We verify that path.
        expected_relative_save_path = os.path.join("trained_models", self.run_name, "final_policy.zip")
        mock_agent_instance.save.assert_called_once_with(expected_relative_save_path)
        print("✅ [TestRunner] Main experiment flow integrates components correctly.")
if __name__ == '__main__':
    unittest.main(verbosity=2)