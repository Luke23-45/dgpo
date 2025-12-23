# FILE: tests/test_temporal_ensemble.py
"""
Unit tests for TemporalEnsemble in train_dgpo_robust.py

Tests:
1. Action smoothing behavior - validates weighted averaging
2. Cache management - tests add/reset operations
3. Edge cases - empty cache, single chunk, etc.
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
import numpy as np

from train.train_dgpo_robust import TemporalEnsemble


class TestTemporalEnsemble:
    """Test suite for TemporalEnsemble component."""
    
    @pytest.fixture
    def ensemble(self):
        """Create a TemporalEnsemble instance."""
        return TemporalEnsemble(cache_size=5, chunk_size=10, action_dim=7)
    
    def test_empty_cache_returns_zeros(self, ensemble):
        """Test that empty cache returns zero action."""
        action = ensemble.get_smoothed_action()
        
        assert action.shape == (7,)
        assert np.allclose(action, np.zeros(7))
    
    def test_single_chunk_returns_first_step(self, ensemble):
        """Test with single chunk in cache."""
        chunk = np.random.randn(10, 7)
        ensemble.add(chunk)
        
        action = ensemble.get_smoothed_action()
        
        assert action.shape == (7,)
        # With single chunk, should return first step of that chunk
        assert np.allclose(action, chunk[0])
    
    def test_cache_size_limit(self, ensemble):
        """Test that cache doesn't exceed max size."""
        for i in range(10):
            chunk = np.random.randn(10, 7)
            ensemble.add(chunk)
        
        assert len(ensemble.cache) == 5, f"Cache size: {len(ensemble.cache)}"
    
    def test_smoothing_with_multiple_chunks(self, ensemble):
        """Test weighted averaging with multiple chunks."""
        # Add known chunks
        chunk1 = np.ones((10, 7)) * 0.0
        chunk2 = np.ones((10, 7)) * 1.0
        
        ensemble.add(chunk1)
        ensemble.add(chunk2)
        
        action = ensemble.get_smoothed_action()
        
        # Should be weighted average (more recent = higher weight)
        assert action.shape == (7,)
        # With exponential weights, second chunk has more weight
        assert np.all(action > 0.0) and np.all(action < 1.0), \
            f"Expected blended action, got {action[0]}"
    
    def test_reset_clears_cache(self, ensemble):
        """Test that reset clears all cached chunks."""
        for i in range(3):
            ensemble.add(np.random.randn(10, 7))
        
        assert len(ensemble.cache) == 3
        
        ensemble.reset()
        
        assert len(ensemble.cache) == 0
    
    def test_weight_normalization(self, ensemble):
        """Test that weights sum to 1."""
        assert np.isclose(ensemble.weights.sum(), 1.0), \
            f"Weights sum to {ensemble.weights.sum()}"
    
    def test_exponential_weights_ordering(self, ensemble):
        """Test that more recent chunks have higher weights."""
        # Weights should increase (index 0 is oldest)
        for i in range(len(ensemble.weights) - 1):
            assert ensemble.weights[i] <= ensemble.weights[i + 1], \
                "Weights should increase for more recent chunks"
    
    def test_add_copies_chunk(self, ensemble):
        """Test that add() copies the input array."""
        chunk = np.ones((10, 7))
        ensemble.add(chunk)
        
        # Modify original
        chunk[0, 0] = 999.0
        
        # Cached chunk should be unchanged
        assert ensemble.cache[0][0, 0] == 1.0
    
    def test_partial_cache_weighted_average(self, ensemble):
        """Test weighted average with fewer chunks than cache_size."""
        # Only add 2 chunks to cache of size 5
        chunk1 = np.zeros((10, 7))
        chunk2 = np.ones((10, 7))
        
        ensemble.add(chunk1)
        ensemble.add(chunk2)
        
        action = ensemble.get_smoothed_action()
        
        # Should use only the 2 chunks with renormalized weights
        assert action.shape == (7,)
        assert np.all(action > 0) and np.all(action < 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
