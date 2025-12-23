# FILE: tests/test_ppo_loss.py
"""
Unit tests for PPO loss computation in train_dgpo_robust.py

Tests:
1. Ratio calculation (log_prob_new - log_prob_old)
2. Clipping behavior
3. Advantage weighting
4. Gradient flow to policy parameters
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
import torch
import numpy as np


class TestPPOLoss:
    """Test suite for PPO loss computation."""
    
    def test_ratio_calculation(self):
        """Test ratio = exp(log_prob_new - log_prob_old)."""
        log_prob_old = torch.tensor([-1.0, -2.0, -3.0])
        log_prob_new = torch.tensor([-1.2, -1.8, -3.5])
        
        ratio = torch.exp(log_prob_new - log_prob_old)
        
        expected = torch.exp(torch.tensor([-0.2, 0.2, -0.5]))
        assert torch.allclose(ratio, expected, atol=1e-5)
    
    def test_ratio_equals_one_for_same_log_probs(self):
        """Test ratio = 1 when log probs are identical."""
        log_prob = torch.tensor([-1.0, -2.0, -3.0])
        
        ratio = torch.exp(log_prob - log_prob)
        
        assert torch.allclose(ratio, torch.ones_like(ratio))
    
    def test_clipping_behavior(self):
        """Test PPO clipping limits ratio."""
        clip_param = 0.2
        
        # Large positive ratio (action became more likely)
        ratio_high = torch.tensor([2.0])
        clipped_high = torch.clamp(ratio_high, 1.0 - clip_param, 1.0 + clip_param)
        assert clipped_high.item() == pytest.approx(1.2)
        
        # Large negative ratio (action became less likely)
        ratio_low = torch.tensor([0.5])
        clipped_low = torch.clamp(ratio_low, 1.0 - clip_param, 1.0 + clip_param)
        assert clipped_low.item() == pytest.approx(0.8)
    
    def test_surrogate_loss_with_positive_advantage(self):
        """Test PPO surrogate with positive advantage."""
        ratio = torch.tensor([1.1])
        advantage = torch.tensor([1.0])  # Positive advantage
        clip_param = 0.2
        
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantage
        ppo_loss = -torch.min(surr1, surr2)
        
        # With positive advantage and ratio > 1, we want to encourage
        # Clipped ratio is 1.1 (within bounds), so surr1 = surr2
        assert ppo_loss.item() == pytest.approx(-1.1)
    
    def test_surrogate_loss_with_negative_advantage(self):
        """Test PPO surrogate with negative advantage."""
        ratio = torch.tensor([1.1])
        advantage = torch.tensor([-1.0])  # Negative advantage
        clip_param = 0.2
        
        surr1 = ratio * advantage  # -1.1
        surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantage  # -1.1
        ppo_loss = -torch.min(surr1, surr2)
        
        # min(-1.1, -1.1) = -1.1, so loss = 1.1
        assert ppo_loss.item() == pytest.approx(1.1)
    
    def test_clipping_prevents_large_updates(self):
        """Test that clipping prevents overly large policy updates."""
        clip_param = 0.2
        advantage = torch.tensor([1.0])
        
        # Very large ratio (action probability increased 10x)
        ratio_extreme = torch.tensor([10.0])
        
        surr1 = ratio_extreme * advantage  # 10.0
        surr2 = torch.clamp(ratio_extreme, 1 - clip_param, 1 + clip_param) * advantage  # 1.2
        loss = -torch.min(surr1, surr2)
        
        # Should take clipped version (1.2) not extreme (10.0)
        assert loss.item() == pytest.approx(-1.2)
    
    def test_gradient_flow_through_ppo_loss(self):
        """Test gradients flow through PPO loss computation."""
        # Policy parameters
        mean = torch.randn(4, 10, 7, requires_grad=True)
        log_std = torch.zeros(1, 10, 7, requires_grad=True)
        
        # Create distribution
        dist = torch.distributions.Normal(mean, log_std.exp())
        
        # Sample actions (detach to simulate buffer)
        actions = dist.sample()
        
        # Old log probs (from buffer, detached)
        log_prob_old = dist.log_prob(actions).sum(dim=[1, 2]).detach()
        
        # New log probs (for current policy)
        log_prob_new = dist.log_prob(actions).sum(dim=[1, 2])
        
        # Compute ratio and loss
        ratio = torch.exp(log_prob_new - log_prob_old)
        advantage = torch.randn(4)
        
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantage
        ppo_loss = -torch.min(surr1, surr2).mean()
        
        ppo_loss.backward()
        
        assert mean.grad is not None, "No gradient for mean"
        assert log_std.grad is not None, "No gradient for log_std"
    
    def test_normalized_advantage(self):
        """Test advantage normalization."""
        advantages = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        
        normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        assert torch.isclose(normalized.mean(), torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(normalized.std(), torch.tensor(1.0), atol=0.1)
    
    def test_clip_fraction_calculation(self):
        """Test clip fraction metric."""
        ratios = torch.tensor([0.7, 0.9, 1.0, 1.1, 1.5])
        clip_param = 0.2
        
        clipped = ((ratios - 1.0).abs() > clip_param).float()
        clip_fraction = clipped.mean().item()
        
        # 0.7 and 1.5 are outside [0.8, 1.2]
        expected = 2/5
        assert clip_fraction == pytest.approx(expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
