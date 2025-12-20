import unittest
import torch
import numpy as np
import shutil
from pathlib import Path
from unittest.mock import MagicMock
from omegaconf import OmegaConf
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from train.train_dgpo_robust import MetricsLogger, DGPOTrainer, RolloutBuffer

class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.log_dir = Path("test_logs")
        if self.log_dir.exists():
            shutil.rmtree(self.log_dir, ignore_errors=True)
        self.csv_path = self.log_dir / "test_metrics.csv"
        
    def tearDown(self):
        if self.log_dir.exists():
            shutil.rmtree(self.log_dir, ignore_errors=True)

    def test_logger_headers(self):
        logger = MetricsLogger(self.csv_path)
        expected = [
            "blend_alpha", "exec_pos_div", "grad_norm_p", "grad_norm_v", 
            "explained_var", "clip_frac"
        ]
        
        with open(self.csv_path, 'r') as f:
            header = f.readline().strip()
            for key in expected:
                self.assertIn(key, header, f"Header missing {key}")
        logger.close()

    def test_logger_write(self):
        logger = MetricsLogger(self.csv_path)
        stats = {
            "mean_reward": 1.0, "success_rate": 0.5, "n_episodes": 10,
            "shadow_pos_div": 0.1, "shadow_orn_div": 0.01, "grip_agreement": 0.9,
            "blend_alpha": 0.5, "exec_pos_div": 0.05
        }
        update_stats = {
            "policy_loss": 0.1, "value_loss": 0.2, "bc_loss": 0.3,
            "lr_policy": 1e-4, "lr_value": 2e-4, "entropy": 0.5,
            "kl_divergence": 0.01, "kl_beta": 1.0, "grad_norm_policy": 0.5,
            "grad_norm_value": 0.6, "explained_variance": 0.8, "clip_fraction": 0.1,
            "adv_mean": 0.0, "loss_ppo": 0.1, "loss_value": 0.2, "loss_bc": 0.3,
            "loss_entropy": 0.01, "loss_phase": 0.02, "loss_smoothness": 0.03
        }
        
        import csv
        
        logger.log_step(1, stats, update_stats, 100)
        logger.close()  # Essential for Windows file release
        
        with open(self.csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            row = next(reader)
            
            # Check values (converting strings back to floats)
            self.assertAlmostEqual(float(row['grad_norm_p']), 0.5)
            self.assertAlmostEqual(float(row['explained_var']), 0.8)

if __name__ == "__main__":
    unittest.main()
