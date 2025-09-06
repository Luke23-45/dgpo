# test_expert_dataset_robust.py
import unittest
import torch
import numpy as np
from torch.utils.data import DataLoader

# The class we are testing
from utils.expert_dataset import ExpertDataset

# Define constants for the test
URDF_PATH = "urdf/panda_mujoco_kinematics.urdf"
BATCH_SIZE = 2
NUM_SAMPLES_TO_GENERATE = 4 # Should be a multiple of batch size
BASE_SEED = 42

class TestExpertDatasetRobust(unittest.TestCase):
    """
    A comprehensive, production-grade test suite for the ExpertDataset.

    This suite verifies:
    1. Correct initialization with different configurations.
    2. Successful batch generation via a DataLoader.
    3. The precise shape, dtype, and value ranges of the output tensors.
    4. Correct device placement when requested.
    """

    def _run_test_for_config(self, move_to_device: bool, device_str: str):
        """Helper function to run the core tests for a given configuration."""
        device = torch.device(device_str)

        # 1. Initialization
        try:
            dataset = ExpertDataset(
                urdf_path=URDF_PATH,
                max_samples_per_epoch=NUM_SAMPLES_TO_GENERATE,
                base_seed=BASE_SEED,
                move_to_device=move_to_device,
                device=device
            )
            # The length should be defined when max_samples_per_epoch is set
            self.assertEqual(len(dataset), NUM_SAMPLES_TO_GENERATE)
        except Exception as e:
            self.fail(f"Dataset initialization failed for config (move_to_device={move_to_device}). Error: {e}")

        # 2. Data Loading
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)
        try:
            # Fetch all batches to ensure the full dataset can be generated
            all_batches = list(loader)
            self.assertEqual(len(all_batches), NUM_SAMPLES_TO_GENERATE // BATCH_SIZE)
        except Exception as e:
            self.fail(f"Fetching batches failed for config (move_to_device={move_to_device}). Error: {e}")

        # 3. Batch Content Verification (using the first batch)
        obs_batch, action_batch = all_batches[0]

        # --- Assertions for the observation dictionary ---
        self.assertIn("image_primary", obs_batch)
        self.assertIn("proprio", obs_batch)

        img_tensor = obs_batch["image_primary"]
        proprio_tensor = obs_batch["proprio"]

        # Check shapes
        self.assertEqual(img_tensor.shape, (BATCH_SIZE, 3, 256, 256))
        self.assertEqual(proprio_tensor.shape, (BATCH_SIZE, 14))

        # Check dtypes
        self.assertEqual(img_tensor.dtype, torch.float32)
        self.assertEqual(proprio_tensor.dtype, torch.float32)

        # --- Assertions for the action tensor ---
        self.assertEqual(action_batch.shape, (BATCH_SIZE, 8))
        self.assertEqual(action_batch.dtype, torch.float32)

        # --- Rigorous Value Range Checks ---
        # Image tensor should be normalized to [-1, 1]
        self.assertTrue(torch.all(img_tensor >= -1.0) and torch.all(img_tensor <= 1.0),
                        "Image tensor values are not correctly normalized to the [-1, 1] range.")
        
        # Action tensor should be clipped to [-1, 1]
        self.assertTrue(torch.all(action_batch >= -1.0) and torch.all(action_batch <= 1.0),
                        "Action tensor values are not correctly clipped to the [-1, 1] range.")

        # --- Device Placement Check ---
        if move_to_device:
            self.assertEqual(img_tensor.device.type, device.type, "Image tensor is on the wrong device.")
            self.assertEqual(action_batch.device.type, device.type, "Action tensor is on the wrong device.")
        else:
            # Default device should be CPU
            self.assertEqual(img_tensor.device.type, "cpu", "Image tensor should be on CPU by default.")
            self.assertEqual(action_batch.device.type, "cpu", "Action tensor should be on CPU by default.")

    def test_A_data_generation_cpu(self):
        """Test the full pipeline with tensors remaining on the CPU (default)."""
        print("\n[TestExpertDataset] Running test for CPU-only configuration...")
        self._run_test_for_config(move_to_device=False, device_str="cpu")
        print("✅ [TestExpertDataset] CPU-only configuration passed.")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available, skipping GPU test.")
    def test_B_data_generation_gpu(self):
        """Test the full pipeline with `move_to_device=True` to a CUDA device."""
        print("\n[TestExpertDataset] Running test for GPU (move_to_device=True) configuration...")
        self._run_test_for_config(move_to_device=True, device_str="cuda")
        print("✅ [TestExpertDataset] GPU configuration passed.")


if __name__ == '__main__':
    unittest.main(verbosity=2)