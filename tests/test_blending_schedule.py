# FILE: tests/test_blending_schedule.py
"""
Unit tests for BlendingSchedule in train_dgpo_robust.py

Tests:
1. Warmup phase behavior (α=0.1)
2. Rampup phase progression (linear increase)
3. Plateau phase (α=max_alpha)
4. Boundary conditions
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
import numpy as np

from train.train_dgpo_robust import BlendingSchedule


class TestBlendingSchedule:
    """Test suite for BlendingSchedule component."""
    
    @pytest.fixture
    def schedule(self):
        """Create a BlendingSchedule instance."""
        return BlendingSchedule(
            warmup_iters=20,
            rampup_iters=100,
            max_alpha=0.9
        )
    
    def test_warmup_phase_returns_01(self, schedule):
        """Test that warmup phase returns α=0.1."""
        for iteration in [0, 5, 10, 19]:
            alpha = schedule.get_alpha(iteration)
            assert alpha == 0.1, f"Iteration {iteration}: expected 0.1, got {alpha}"
    
    def test_rampup_starts_after_warmup(self, schedule):
        """Test that rampup phase starts at warmup_iters."""
        alpha_warmup_end = schedule.get_alpha(19)  # Last warmup
        alpha_rampup_start = schedule.get_alpha(20)  # First rampup
        alpha_rampup_mid = schedule.get_alpha(70)  # Mid rampup
        
        assert alpha_warmup_end == 0.1
        # At iteration 20 (start of rampup), alpha may still be at base (0.1)
        # because progress = (20-20)/100 = 0, so alpha = 0.1 + 0*(0.9-0.1) = 0.1
        assert alpha_rampup_start >= 0.1
        # By mid rampup, alpha should be increasing
        assert alpha_rampup_mid > 0.1, f"Mid rampup should be increasing, got {alpha_rampup_mid}"
    
    def test_rampup_linear_progression(self, schedule):
        """Test that alpha increases linearly during rampup."""
        alphas = []
        for i in range(20, 120):
            alphas.append(schedule.get_alpha(i))
        
        # Check monotonic increase
        for i in range(len(alphas) - 1):
            assert alphas[i] <= alphas[i + 1], \
                f"Alpha should increase: {alphas[i]} -> {alphas[i+1]}"
        
        # Check roughly linear (differences should be approximately equal)
        diffs = np.diff(alphas)
        assert np.std(diffs) < 0.01, "Progression not linear"
    
    def test_plateau_phase(self, schedule):
        """Test that alpha reaches max_alpha after rampup."""
        # After warmup (20) + rampup (100) = iteration 120+
        for iteration in [120, 150, 200, 500]:
            alpha = schedule.get_alpha(iteration)
            assert alpha == 0.9, f"Iteration {iteration}: expected 0.9, got {alpha}"
    
    def test_rampup_end_value(self, schedule):
        """Test value at end of rampup phase."""
        # At iteration 119 (last rampup step)
        alpha = schedule.get_alpha(119)
        
        # Should be close to max_alpha
        assert np.isclose(alpha, 0.9, atol=0.02), f"Expected ~0.9, got {alpha}"
    
    def test_custom_parameters(self):
        """Test with different warmup/rampup parameters."""
        custom = BlendingSchedule(
            warmup_iters=10,
            rampup_iters=50,
            max_alpha=0.8
        )
        
        assert custom.get_alpha(9) == 0.1  # Warmup
        assert custom.get_alpha(59) > 0.7  # End of rampup
        assert custom.get_alpha(60) == 0.8  # Plateau
    
    def test_zero_warmup(self):
        """Test with zero warmup iterations."""
        no_warmup = BlendingSchedule(
            warmup_iters=0,
            rampup_iters=100,
            max_alpha=0.9
        )
        
        # Should start ramping immediately
        alpha_0 = no_warmup.get_alpha(0)
        assert alpha_0 >= 0.1
    
    def test_negative_iteration_handling(self, schedule):
        """Test handling of negative iteration (edge case)."""
        # Negative iteration should behave like warmup
        alpha = schedule.get_alpha(-1)
        assert alpha == 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
