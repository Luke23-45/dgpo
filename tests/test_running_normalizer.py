# FILE: tests/test_running_normalizer.py
"""
Unit tests for RunningNormalizer in train_dgpo_robust.py

Tests:
1. Normalization behavior - correct mean/var tracking
2. Clipping bounds - ensures output stays within bounds
3. State save/load - serialization works correctly
4. Numerical stability - handles edge cases
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
import numpy as np

from train.train_dgpo_robust import RunningNormalizer


class TestRunningNormalizer:
    """Test suite for RunningNormalizer component."""
    
    @pytest.fixture
    def normalizer(self):
        """Create a RunningNormalizer instance."""
        return RunningNormalizer(epsilon=1e-8, gamma=0.99)
    
    def test_initial_state(self, normalizer):
        """Test initial state values."""
        assert normalizer.mean == 0.0
        assert normalizer.var == 1.0
        assert normalizer.count == 0
    
    def test_first_normalization_updates_stats(self, normalizer):
        """Test that first call updates running stats."""
        rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _ = normalizer.normalize(rewards, clip_range=5.0, update_stats=True)
        
        assert normalizer.count == 1
        assert normalizer.mean == np.mean(rewards)
        assert normalizer.var == np.var(rewards)
    
    def test_normalization_output_range(self, normalizer):
        """Test that normalized output is within clip range."""
        # Seed with initial stats
        rewards1 = np.array([0.0, 1.0, 2.0])
        normalizer.normalize(rewards1, clip_range=5.0)
        
        # Now normalize extreme values
        rewards2 = np.array([-100.0, 0.0, 100.0])
        normalized = normalizer.normalize(rewards2, clip_range=5.0)
        
        assert np.all(normalized >= -5.0), f"Values below -5: {normalized}"
        assert np.all(normalized <= 5.0), f"Values above 5: {normalized}"
    
    def test_update_stats_false(self, normalizer):
        """Test that update_stats=False doesn't modify stats."""
        rewards1 = np.array([1.0, 2.0, 3.0])
        normalizer.normalize(rewards1, update_stats=True)
        
        saved_mean = normalizer.mean
        saved_var = normalizer.var
        saved_count = normalizer.count
        
        rewards2 = np.array([100.0, 200.0, 300.0])
        normalizer.normalize(rewards2, update_stats=False)
        
        assert normalizer.mean == saved_mean
        assert normalizer.var == saved_var
        assert normalizer.count == saved_count
    
    def test_ema_update(self, normalizer):
        """Test exponential moving average updates correctly."""
        # First batch
        rewards1 = np.array([0.0, 0.0, 0.0])
        normalizer.normalize(rewards1)
        
        # Second batch with different values
        rewards2 = np.array([10.0, 10.0, 10.0])
        normalizer.normalize(rewards2)
        
        # Mean should be blended (gamma=0.99)
        expected_mean = 0.99 * 0.0 + 0.01 * 10.0
        assert np.isclose(normalizer.mean, expected_mean, atol=1e-6), \
            f"Expected {expected_mean}, got {normalizer.mean}"
    
    def test_state_dict_save_load(self, normalizer):
        """Test state serialization and restoration."""
        rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalizer.normalize(rewards)
        
        state = normalizer.state_dict()
        
        new_normalizer = RunningNormalizer()
        new_normalizer.load_state_dict(state)
        
        assert new_normalizer.mean == normalizer.mean
        assert new_normalizer.var == normalizer.var
        assert new_normalizer.count == normalizer.count
    
    def test_inverse_normalize(self, normalizer):
        """Test inverse normalization recovers original scale."""
        rewards = np.array([5.0, 10.0, 15.0])
        normalizer.normalize(rewards)  # Set up stats
        
        test_values = np.array([1.0, 2.0, 3.0])
        normalized = normalizer.normalize(test_values, update_stats=False)
        recovered = normalizer.inverse_normalize(normalized)
        
        # Should approximately recover original
        assert np.allclose(recovered, test_values, atol=0.1), \
            f"Expected {test_values}, got {recovered}"
    
    def test_numerical_stability_zero_variance(self, normalizer):
        """Test handling of zero variance (all same values)."""
        rewards = np.array([5.0, 5.0, 5.0, 5.0])
        normalized = normalizer.normalize(rewards)
        
        assert np.isfinite(normalized).all(), "NaN/Inf with zero variance"
    
    def test_empty_array_handling(self, normalizer):
        """Test with edge case inputs."""
        # Single value
        rewards = np.array([1.0])
        normalized = normalizer.normalize(rewards)
        assert np.isfinite(normalized).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
