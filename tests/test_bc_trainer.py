# tests/test_bc_trainer.py
import unittest
import os
import tempfile
import torch
from torch.utils.data import Dataset, DataLoader
from unittest.mock import patch, MagicMock
import logging

# The components we are testing
from training.bc_trainer import BCTrainer, _default_collate, set_seeds
from models.bc_policy import BCNet

# Suppress the logger output during tests for cleaner results
logging.disable(logging.CRITICAL)


class MockExpertDataset(Dataset):
    """A fake dataset that returns predictable data for testing."""
    def __init__(self, length=100, n_actions=8, proprio_dim=14):
        self.length = length
        self.n_actions = n_actions
        self.proprio_dim = proprio_dim
        set_seeds(0) # Ensure mock data is deterministic

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Return data as a dictionary, mimicking one of the formats
        # the collate function is designed to handle.
        obs = {
            "image_primary": torch.randn(3, 128, 128),
            "proprio": torch.randn(self.proprio_dim)
        }
        action = torch.randn(self.n_actions)
        return {"obs": obs, "action": action}


class TestBCTrainer(unittest.TestCase):
    """A test suite for the BCTrainer class."""

    def setUp(self):
        """Set up a temporary directory and a default trainer configuration."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = {
            "n_actions": 8,
            "proprio_dim": 14,
            "learning_rate": 1e-4,
            "batch_size": 4,
            "num_epochs": 2,
            "device": "cpu",
            "urdf_path": "dummy.urdf", # Required by _get_dataloader
            "steps_per_epoch": 10,     # Required by _get_dataloader
            "save_path": os.path.join(self.temp_dir.name, "final_model.pth"),
        }

    def tearDown(self):
        """Clean up the temporary directory."""
        self.temp_dir.cleanup()

    @classmethod
    def tearDownClass(cls):
        """Re-enable logging after tests are done."""
        logging.disable(logging.NOTSET)

    def test_A_initialization(self):
        """Test that the BCTrainer initializes correctly from a config."""
        trainer = BCTrainer(self.config)
        self.assertIsInstance(trainer.model, BCNet)
        self.assertIsInstance(trainer.optimizer, torch.optim.Optimizer)
        self.assertEqual(trainer.device, torch.device("cpu"))
        self.assertEqual(trainer.start_epoch, 0)
        print("\n✅ [TestBCTrainer] Initialization is correct.")

    def test_B_save_and_load_checkpoint(self):
        """Test the save and load checkpoint functionality."""
        trainer1 = BCTrainer(self.config)
        trainer1.best_loss = 0.123
        checkpoint_path = os.path.join(self.temp_dir.name, "ckpt.pth")
        trainer1.save_checkpoint(checkpoint_path, epoch_completed=5, loss=0.123)
        
        self.assertTrue(os.path.exists(checkpoint_path))

        trainer2 = BCTrainer(self.config)
        trainer2.load_checkpoint(checkpoint_path)

        self.assertEqual(trainer2.start_epoch, 5)
        self.assertAlmostEqual(trainer2.best_loss, 0.123)
        
        state_dict1 = trainer1.model.state_dict()
        state_dict2 = trainer2.model.state_dict()
        for key in state_dict1:
            self.assertTrue(torch.equal(state_dict1[key], state_dict2[key]))
            
        print("✅ [TestBCTrainer] Checkpointing (save/load) is correct.")

    @patch('training.bc_trainer.BCTrainer._get_dataloader')
    @patch('training.bc_trainer.BCTrainer._train_one_epoch')
    def test_C_train_loop_logic(self, mock_train_one_epoch, mock_get_dataloader):
        """Test the main training loop's logic, like early stopping and saving."""
        mock_get_dataloader.return_value = [1, 2, 3] # Dummy iterable
        mock_train_one_epoch.side_effect = [0.5, 0.3, 0.1, 0.2, 0.2]

        self.config["num_epochs"] = 5
        self.config["early_stop_patience"] = 2
        trainer = BCTrainer(self.config)
        trainer.train()

        # Loop should run 5 times: best loss at epoch 3 (loss 0.1), then no improvement 
        # at epoch 4 (loss 0.2) and epoch 5 (loss 0.2). Patience=2 is met.
        self.assertEqual(mock_train_one_epoch.call_count, 5)
        
        self.assertAlmostEqual(trainer.best_loss, 0.1)
        
        best_path = os.path.join(self.temp_dir.name, "best_model.pth")
        self.assertTrue(os.path.exists(best_path))
        
        best_ckpt = torch.load(best_path)
        self.assertEqual(best_ckpt['epoch'], 3)

        print("✅ [TestBCTrainer] Training loop logic (early stopping, best model save) is correct.")

    def test_D_default_collate_function(self):
        """Test the custom collate function with a mock dataset."""
        dataset = MockExpertDataset(
            length=self.config['batch_size'],
            n_actions=self.config['n_actions'],
            proprio_dim=self.config['proprio_dim']
        )
        batch = [dataset[i] for i in range(len(dataset))]
        
        obs, actions = _default_collate(batch)
        
        self.assertIsInstance(obs, dict)
        self.assertIn("image_primary", obs)
        self.assertIn("proprio", obs)
        
        self.assertEqual(obs['image_primary'].shape, (self.config['batch_size'], 3, 128, 128))
        self.assertEqual(obs['proprio'].shape, (self.config['batch_size'], self.config['proprio_dim']))
        self.assertEqual(actions.shape, (self.config['batch_size'], self.config['n_actions']))
        print("✅ [TestBCTrainer] Custom collate function is correct.")


if __name__ == '__main__':
    unittest.main(verbosity=2)