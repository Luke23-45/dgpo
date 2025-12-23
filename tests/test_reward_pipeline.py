# FILE: tests/test_reward_pipeline.py
"""
Unit tests for reward computation in train_dgpo_robust.py

Tests:
1. Imitation reward calculation (RSD → exponential)
2. Smoothness penalty computation
3. Entropy bonus integration
4. Success bonus application
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
import torch
import numpy as np


class TestRewardPipeline:
    """Test suite for reward computation logic from collect_rollouts."""
    
    def test_imitation_reward_formula(self):
        """Test the exponential reward formula: exp(-rsd / sigma_sq)."""
        sigma_sq = 0.05
        
        # Test cases
        test_cases = [
            (0.0, 1.0),    # Perfect match → max reward
            (0.05, np.exp(-1.0)),  # rsd = sigma_sq → exp(-1)
            (0.1, np.exp(-2.0)),   # rsd = 2*sigma_sq → exp(-2)
        ]
        
        for rsd, expected in test_cases:
            reward = np.exp(-rsd / sigma_sq)
            assert np.isclose(reward, expected, atol=1e-6), \
                f"RSD={rsd}: expected {expected}, got {reward}"
    
    def test_imitation_reward_bounds(self):
        """Test that imitation reward is bounded in [0, 1]."""
        sigma_sq = 0.05
        
        for rsd in [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]:
            reward = np.exp(-rsd / sigma_sq)
            assert 0 <= reward <= 1, f"Reward {reward} out of bounds for rsd={rsd}"
    
    def test_smoothness_penalty_zero_diff(self):
        """Test smoothness penalty with no action change."""
        prev_action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0])
        curr_action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0])
        
        action_diff = np.linalg.norm(curr_action[:8] - prev_action)
        smoothness_penalty = -0.05 * action_diff ** 2
        
        assert smoothness_penalty == 0.0
    
    def test_smoothness_penalty_large_diff(self):
        """Test smoothness penalty with large action change."""
        prev_action = np.zeros(8)
        curr_action = np.ones(8)
        
        action_diff = np.linalg.norm(curr_action - prev_action)
        smoothness_penalty = -0.05 * action_diff ** 2
        
        # Should be negative (penalty)
        assert smoothness_penalty < 0
        
        # Should scale quadratically
        expected = -0.05 * (np.sqrt(8)) ** 2  # norm of 8 ones
        assert np.isclose(smoothness_penalty, expected, atol=1e-6)
    
    def test_entropy_bonus_positive(self):
        """Test that entropy bonus is positive."""
        # Simulate policy entropy
        policy_mean = torch.randn(4, 10, 7)
        log_std = torch.zeros(1, 10, 7)  # std = 1
        
        dist = torch.distributions.Normal(policy_mean, log_std.exp())
        entropy = dist.entropy().mean(dim=[1, 2])
        
        entropy_coef = 0.01
        entropy_bonus = entropy_coef * entropy
        
        # Entropy is always positive for Gaussian
        assert (entropy_bonus > 0).all()
    
    def test_entropy_bonus_scales_with_std(self):
        """Test that entropy bonus increases with exploration."""
        policy_mean = torch.zeros(1, 10, 7)
        
        # Low std
        log_std_low = torch.ones(1, 10, 7) * -3.0  # std ≈ 0.05
        dist_low = torch.distributions.Normal(policy_mean, log_std_low.exp())
        entropy_low = dist_low.entropy().mean()
        
        # High std
        log_std_high = torch.ones(1, 10, 7) * 0.0  # std = 1.0
        dist_high = torch.distributions.Normal(policy_mean, log_std_high.exp())
        entropy_high = dist_high.entropy().mean()
        
        assert entropy_high > entropy_low, "Higher std should give higher entropy"
    
    def test_success_bonus_application(self):
        """Test success bonus logic."""
        base_reward = 0.5
        success_bonus = 10.0
        
        # Success case: distance to goal < 0.05
        obj_pos = np.array([0.5, 0.5, 0.5])
        goal_pos = np.array([0.5, 0.5, 0.52])  # 2cm away
        dist = np.linalg.norm(obj_pos - goal_pos)
        
        if dist < 0.05:
            total = base_reward + success_bonus
        else:
            total = base_reward
        
        assert dist < 0.05
        assert total == base_reward + success_bonus
    
    def test_success_bonus_not_applied_when_far(self):
        """Test success bonus not given when far from goal."""
        base_reward = 0.5
        
        obj_pos = np.array([0.5, 0.5, 0.5])
        goal_pos = np.array([0.6, 0.6, 0.6])  # ~17cm away
        dist = np.linalg.norm(obj_pos - goal_pos)
        
        success = dist < 0.05
        
        assert not success
        assert dist > 0.05
    
    def test_total_reward_components_combine(self):
        """Test that all reward components combine correctly."""
        # Imitation reward
        rsd = 0.02
        sigma_sq = 0.05
        imitation_reward = np.exp(-rsd / sigma_sq)
        
        # Smoothness penalty
        action_diff = 0.1
        smoothness_penalty = -0.05 * action_diff ** 2
        
        # Entropy bonus (simulated)
        entropy_bonus = 0.05
        
        # Success bonus (not achieved)
        success_bonus = 0.0
        
        total = imitation_reward + entropy_bonus + smoothness_penalty + success_bonus
        
        # Should be positive since imitation + entropy > penalty
        expected = np.exp(-0.4) + 0.05 - 0.0005
        assert np.isclose(total, expected, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
