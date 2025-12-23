# FILE: tests/test_gae_computation.py
"""
Unit tests for compute_gae in train_dgpo_robust.py

Tests:
1. GAE computation correctness with known inputs
2. Done flag handling
3. Edge cases (zeros, ones, single step)
4. Numerical stability
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
import numpy as np

from train.train_dgpo_robust import compute_gae


class TestComputeGAE:
    """Test suite for compute_gae function."""
    
    def test_basic_gae_shape(self):
        """Test that output shapes match input."""
        T = 10
        rewards = np.random.randn(T).astype(np.float32)
        values = np.random.randn(T).astype(np.float32)
        dones = np.zeros(T, dtype=bool)
        
        advantages, returns = compute_gae(rewards, values, dones)
        
        assert advantages.shape == (T,)
        assert returns.shape == (T,)
    
    def test_gae_single_step(self):
        """Test GAE with single step."""
        rewards = np.array([1.0])
        values = np.array([0.5])
        dones = np.array([True])
        
        advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
        
        # With done=True, advantage = reward - value
        expected_adv = 1.0 - 0.5
        assert np.isclose(advantages[0], expected_adv, atol=0.1)
    
    def test_gae_no_dones(self):
        """Test GAE without any episode terminations."""
        T = 5
        rewards = np.ones(T)
        values = np.zeros(T)
        dones = np.zeros(T, dtype=bool)
        
        advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
        
        # All advantages should be positive since reward > value
        assert np.all(advantages > 0)
        
        # Returns should also be positive
        assert np.all(returns > 0)
    
    def test_gae_with_terminal_state(self):
        """Test GAE resets at episode boundaries."""
        rewards = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        values = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        dones = np.array([False, False, True, False, False])
        
        advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
        
        # Advantage at step 2 (done) should not look ahead
        # because mask = 0
        adv_at_done = advantages[2]
        expected_at_done = rewards[2] - values[2]  # delta when next_value masked out
        
        # Should be close to simple delta
        assert np.isclose(adv_at_done, expected_at_done, atol=0.5)
    
    def test_gae_returns_equal_advantages_plus_values(self):
        """Test that returns = advantages + values."""
        T = 10
        rewards = np.random.randn(T).astype(np.float32)
        values = np.random.randn(T).astype(np.float32)
        dones = np.zeros(T, dtype=bool)
        
        advantages, returns = compute_gae(rewards, values, dones)
        
        reconstructed = advantages + values
        assert np.allclose(returns, reconstructed, atol=1e-5)
    
    def test_gae_zero_rewards(self):
        """Test GAE with zero rewards."""
        T = 5
        rewards = np.zeros(T)
        values = np.ones(T)
        dones = np.zeros(T, dtype=bool)
        
        advantages, returns = compute_gae(rewards, values, dones)
        
        # Should not produce NaN
        assert np.isfinite(advantages).all()
        assert np.isfinite(returns).all()
    
    def test_gae_gamma_zero(self):
        """Test GAE with gamma=0 (myopic)."""
        rewards = np.array([1.0, 2.0, 3.0])
        values = np.array([0.0, 0.0, 0.0])
        dones = np.zeros(3, dtype=bool)
        
        advantages, returns = compute_gae(rewards, values, dones, gamma=0.0, lam=0.95)
        
        # With gamma=0, advantage = reward - value (no future consideration)
        expected_adv = rewards - values
        assert np.allclose(advantages, expected_adv, atol=0.1)
    
    def test_gae_lambda_zero(self):
        """Test GAE with lambda=0 (TD(0))."""
        rewards = np.array([1.0, 1.0, 1.0])
        values = np.array([0.5, 0.5, 0.5])
        dones = np.zeros(3, dtype=bool)
        
        advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, lam=0.0)
        
        # With lambda=0, advantage = delta (one-step TD error only)
        assert np.isfinite(advantages).all()
    
    def test_gae_numerical_stability(self):
        """Test numerical stability with extreme values."""
        T = 10
        rewards = np.random.randn(T) * 1000  # Large rewards
        values = np.random.randn(T) * 1000
        dones = np.zeros(T, dtype=bool)
        
        advantages, returns = compute_gae(rewards, values, dones)
        
        assert np.isfinite(advantages).all(), "NaN/Inf with large values"
        assert np.isfinite(returns).all(), "NaN/Inf with large values"
    
    def test_gae_all_dones(self):
        """Test GAE when every step is terminal."""
        T = 5
        rewards = np.ones(T)
        values = np.zeros(T)
        dones = np.ones(T, dtype=bool)
        
        advantages, returns = compute_gae(rewards, values, dones)
        
        # Each step is isolated, advantage should be reward - value
        expected = rewards - values
        assert np.allclose(advantages, expected, atol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
