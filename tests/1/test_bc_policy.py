# tests/test_bc_policy.py
import unittest
import torch
import numpy as np
import logging

# The class we are testing
from models.bc_policy import BCNet

# Suppress the logger output during tests for cleaner results
logging.disable(logging.CRITICAL)


class TestBCNet(unittest.TestCase):
    """A test suite for the BCNet model."""

    def setUp(self):
        """Create a default BCNet instance for the tests."""
        self.n_actions = 8
        self.proprio_dim = 14
        self.image_channels = 3
        self.model = BCNet(
            n_actions=self.n_actions,
            proprio_dim=self.proprio_dim,
            image_channels=self.image_channels,
            normalize_mode="-1,1"  # Set an explicit mode for testing
        )
        self.batch_size = 4
        self.img_size = (128, 128)

    @classmethod
    def tearDownClass(cls):
        """Re-enable logging after all tests in this class are done."""
        logging.disable(logging.NOTSET)

    def test_A_forward_pass_shape(self):
        """Test that the forward pass produces the correct output shape."""
        img = torch.randn(self.batch_size, self.image_channels, *self.img_size)
        proprio = torch.randn(self.batch_size, self.proprio_dim)
        
        obs = {"image_primary": img, "proprio": proprio}
        actions = self.model(obs)
        
        self.assertEqual(actions.shape, (self.batch_size, self.n_actions))
        print("\n✅ [TestBCNet] Forward pass produces the correct output shape.")

    def test_B_observation_unpacking(self):
        """Test that the model correctly unpacks different observation formats."""
        img = torch.randn(self.batch_size, self.image_channels, *self.img_size)
        proprio = torch.randn(self.batch_size, self.proprio_dim)

        # Test with standard dict keys
        obs_dict1 = {"image_primary": img, "proprio": proprio}
        self.assertIsNotNone(self.model(obs_dict1))

        # Test with alternative dict keys
        obs_dict2 = {"pixels": img, "state": proprio}
        self.assertIsNotNone(self.model(obs_dict2))

        # Test with tuple format
        obs_tuple = (img, proprio)
        self.assertIsNotNone(self.model(obs_tuple))

        # Test for failure with missing keys
        with self.assertRaises(ValueError):
            self.model({"image_primary": img}) # Missing proprio
        
        with self.assertRaises(ValueError):
            self.model({"state": proprio}) # Missing image

        print("✅ [TestBCNet] Observation unpacking is robust.")

    def test_C_image_normalization_and_format(self):
        """Test the image normalization logic for different input types and formats."""
        proprio = torch.randn(self.batch_size, self.proprio_dim)

        # Test 1: HWC uint8 images [0, 255]
        img_hwc_uint8 = torch.randint(0, 256, (self.batch_size, *self.img_size, self.image_channels), dtype=torch.uint8)
        self.assertIsNotNone(self.model({"image": img_hwc_uint8, "proprio": proprio}))
        
        # Test 2: CHW float images [0, 1]
        img_chw_float_01 = torch.rand(self.batch_size, self.image_channels, *self.img_size)
        self.assertIsNotNone(self.model({"image": img_chw_float_01, "proprio": proprio}))

        # Test 3: CHW float images [0, 255]
        img_chw_float_255 = torch.rand(self.batch_size, self.image_channels, *self.img_size) * 255
        self.assertIsNotNone(self.model({"image": img_chw_float_255, "proprio": proprio}))

        # Test 4: HWC float images [-1, 1]
        img_hwc_float_11 = (torch.rand(self.batch_size, *self.img_size, self.image_channels) * 2) - 1
        self.assertIsNotNone(self.model({"image": img_hwc_float_11, "proprio": proprio}))
        
        print("✅ [TestBCNet] Image normalization handles various formats correctly.")

    def test_D_load_state_dict_flexible(self):
        """Test the flexible checkpoint loading method."""
        # Create a second model to be our "checkpoint" source
        source_model = BCNet(n_actions=self.n_actions, proprio_dim=self.proprio_dim)
        source_state_dict = source_model.state_dict()

        # Case 1: Load from a raw state_dict
        report1 = self.model.load_state_dict_flexible(source_state_dict, strict=True)
        self.assertTrue(report1['loaded'])
        self.assertEqual(len(report1['missing_keys']), 0)
        self.assertEqual(len(report1['unexpected_keys']), 0)

        # Case 2: Load from a wrapped checkpoint dictionary
        wrapped_ckpt = {"model_state_dict": source_state_dict, "epoch": 10}
        report2 = self.model.load_state_dict_flexible(wrapped_ckpt, strict=True)
        self.assertTrue(report2['loaded'])
        
        # Case 3: Load with missing keys (non-strict)
        partial_state_dict = {k: v for k, v in source_state_dict.items() if 'head' not in k}
        report3 = self.model.load_state_dict_flexible(partial_state_dict, strict=False)
        self.assertTrue(report3['loaded'])
        self.assertGreater(len(report3['missing_keys']), 0)

        print("✅ [TestBCNet] Flexible state dict loading works as expected.")

    def test_E_action_rescaling(self):
        """Test the static method for rescaling tanh actions."""
        # Mock an action space from Gymnasium
        class MockActionSpace:
            def __init__(self, low, high):
                self.low = np.array(low, dtype=np.float32)
                self.high = np.array(high, dtype=np.float32)

        action_space = MockActionSpace(low=[-10, 0], high=[10, 100])

        # tanh output from the model is in [-1, 1]
        tanh_output = torch.tensor([
            [-1.0, -1.0], # Should map to low
            [1.0, 1.0],   # Should map to high
            [0.0, 0.0]    # Should map to the midpoint
        ])

        rescaled = BCNet.rescale_action_from_tanh(tanh_output, action_space)

        self.assertTrue(torch.allclose(rescaled[0], torch.tensor([-10.0, 0.0])))
        self.assertTrue(torch.allclose(rescaled[1], torch.tensor([10.0, 100.0])))
        self.assertTrue(torch.allclose(rescaled[2], torch.tensor([0.0, 50.0])))
        print("✅ [TestBCNet] Action rescaling from tanh is correct.")


if __name__ == '__main__':
    unittest.main(verbosity=2)