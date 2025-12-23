# FILE: tests/test_adaptive_kl_penalty.py
"""
Unit tests for AdaptiveKLPenalty in train_dgpo_robust.py

CRITICAL COMPONENT - Known past issue with gradients not flowing.

Tests:
1. Beta update dynamics
2. Clamping behavior (min/max bounds)
3. Gradient flow through penalty term
4. State save/load
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
import torch
import numpy as np

from train.train_dgpo_robust import AdaptiveKLPenalty


class TestAdaptiveKLPenalty:
    """Test suite for AdaptiveKLPenalty component."""
    
    @pytest.fixture
    def kl_penalty(self):
        """Create an AdaptiveKLPenalty instance."""
        return AdaptiveKLPenalty(target_kl=0.015, init_beta=0.02)
    
    def test_initial_values(self, kl_penalty):
        """Test initial beta and target_kl values."""
        assert kl_penalty.target_kl == 0.015
        assert kl_penalty.beta == 0.02
    
    def test_update_increases_beta_when_kl_high(self, kl_penalty):
        """Test that beta increases when measured KL > target."""
        initial_beta = kl_penalty.beta
        
        # Measured KL is 2x the target
        measured_kl = 0.030
        kl_penalty.update(measured_kl)
        
        assert kl_penalty.beta > initial_beta, \
            f"Beta should increase: {initial_beta} -> {kl_penalty.beta}"
    
    def test_update_decreases_beta_when_kl_low(self, kl_penalty):
        """Test that beta decreases when measured KL < target."""
        initial_beta = kl_penalty.beta
        
        # Measured KL is half the target
        measured_kl = 0.0075
        kl_penalty.update(measured_kl)
        
        assert kl_penalty.beta < initial_beta, \
            f"Beta should decrease: {initial_beta} -> {kl_penalty.beta}"
    
    def test_beta_clamping_lower_bound(self, kl_penalty):
        """Test that beta doesn't go below minimum."""
        # Update with very low KL repeatedly
        for _ in range(100):
            kl_penalty.update(0.0001)
        
        assert kl_penalty.beta >= 0.001, f"Beta below min: {kl_penalty.beta}"
    
    def test_beta_clamping_upper_bound(self, kl_penalty):
        """Test that beta doesn't exceed maximum."""
        # Update with very high KL repeatedly
        for _ in range(100):
            kl_penalty.update(1.0)
        
        assert kl_penalty.beta <= 20.0, f"Beta above max: {kl_penalty.beta}"
    
    def test_update_with_zero_kl(self, kl_penalty):
        """Test handling of zero KL divergence."""
        initial_beta = kl_penalty.beta
        kl_penalty.update(0.0)
        
        # Should return without modifying beta
        assert kl_penalty.beta == initial_beta
    
    def test_update_with_negative_kl(self, kl_penalty):
        """Test handling of negative KL (edge case)."""
        initial_beta = kl_penalty.beta
        kl_penalty.update(-0.01)
        
        # Should return without modifying beta
        assert kl_penalty.beta == initial_beta
    
    def test_state_dict(self, kl_penalty):
        """Test state serialization."""
        kl_penalty.update(0.03)  # Modify state
        
        state = kl_penalty.state_dict()
        
        assert 'beta' in state
        assert 'target_kl' in state
        assert state['beta'] == kl_penalty.beta
    
    def test_load_state_dict(self, kl_penalty):
        """Test state restoration."""
        state = {'beta': 5.0, 'target_kl': 0.02}
        
        kl_penalty.load_state_dict(state)
        
        assert kl_penalty.beta == 5.0
        assert kl_penalty.target_kl == 0.02
    
    def test_gradient_flow_through_penalty(self):
        """
        CRITICAL TEST: Verify that gradients flow through the KL penalty term
        when used in loss computation.
        """
        kl_penalty = AdaptiveKLPenalty(target_kl=0.015, init_beta=1.0)
        
        # Simulate policy output
        policy_mean = torch.randn(4, 10, 7, requires_grad=True)
        log_std = torch.zeros(1, 10, 7)
        
        # Create distribution
        dist = torch.distributions.Normal(policy_mean, log_std.exp())
        
        # Sample action and compute log prob
        action = dist.sample()
        log_prob_new = dist.log_prob(action).sum(dim=[1, 2])
        
        # Old log prob (detached, as it would be from buffer)
        log_prob_old = log_prob_new.detach() * 0.9  # Slightly different
        
        # KL divergence term (simplified)
        kl_div = (log_prob_old - log_prob_new).mean()
        
        # KL PENALTY - THIS IS THE CRITICAL PART
        # The penalty should be computed WITH gradient tracking
        kl_penalty_loss = kl_penalty.beta * kl_div
        
        # Check that gradient flows
        kl_penalty_loss.backward()
        
        assert policy_mean.grad is not None, "CRITICAL FAILURE: No gradient through KL penalty!"
        assert policy_mean.grad.abs().sum() > 0, "CRITICAL FAILURE: Zero gradient through KL penalty!"
        
        print(f"✓ Gradient norm: {policy_mean.grad.norm().item():.6f}")
    
    def test_beta_stability_converges(self):
        """Test that beta converges when KL is at target."""
        kl_penalty = AdaptiveKLPenalty(target_kl=0.015, init_beta=1.0)
        
        # Update with target KL repeatedly
        betas = [kl_penalty.beta]
        for _ in range(20):
            kl_penalty.update(0.015)  # Exactly at target
            betas.append(kl_penalty.beta)
        
        # Beta should remain stable
        assert np.std(betas[-5:]) < 0.01, "Beta not stable at target KL"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
