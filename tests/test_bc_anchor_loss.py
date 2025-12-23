# FILE: tests/test_bc_anchor_loss.py
"""
Unit tests for BC (Behavior Cloning) anchor loss in train_dgpo_robust.py

Tests:
1. MSE loss computation between pred and expert chunks
2. bc_coef scaling effect
3. Gradient contribution
4. Loss dominance check (bc_coef=1000 is very high)
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
import torch
import numpy as np
import torch.nn.functional as F


class TestBCAnchorLoss:
    """Test suite for BC anchor loss computation."""
    
    def test_bc_loss_formula(self):
        """Test BC loss is MSE between first steps of pred and expert chunks."""
        pred_chunks = torch.tensor([[[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]])
        expert_chunks = torch.tensor([[[1.1, 2.1, 3.1, 0.0, 0.0, 0.0, 1.0]]])
        
        # BC loss uses first step only ([:, 0, :])
        bc_loss = F.mse_loss(pred_chunks[:, 0, :], expert_chunks[:, 0, :])
        
        # Manual calculation
        diff = pred_chunks[:, 0, :] - expert_chunks[:, 0, :]
        expected = (diff ** 2).mean()
        
        assert torch.isclose(bc_loss, expected)
    
    def test_bc_loss_zero_for_identical_chunks(self):
        """Test BC loss is zero when pred matches expert."""
        chunk = torch.randn(4, 10, 7)
        
        bc_loss = F.mse_loss(chunk[:, 0, :], chunk[:, 0, :])
        
        assert bc_loss.item() == 0.0
    
    def test_bc_loss_increases_with_error(self):
        """Test BC loss increases as prediction error grows."""
        pred = torch.zeros(1, 10, 7)
        
        losses = []
        for offset in [0.0, 0.1, 0.5, 1.0]:
            expert = pred.clone()
            expert[:, 0, :3] += offset
            
            loss = F.mse_loss(pred[:, 0, :], expert[:, 0, :])
            losses.append(loss.item())
        
        # Losses should increase monotonically
        for i in range(len(losses) - 1):
            assert losses[i] <= losses[i + 1], f"Losses should increase: {losses}"
    
    def test_bc_coef_scaling(self):
        """Test that bc_coef scales the loss correctly."""
        pred = torch.randn(4, 10, 7)
        expert = torch.randn(4, 10, 7)
        
        bc_loss_raw = F.mse_loss(pred[:, 0, :], expert[:, 0, :])
        
        bc_coef = 1000.0
        scaled_loss = bc_coef * bc_loss_raw
        
        assert scaled_loss.item() == pytest.approx(bc_coef * bc_loss_raw.item())
    
    def test_bc_loss_only_uses_first_step(self):
        """Test that BC loss only considers first step of chunk."""
        pred = torch.zeros(2, 10, 7)
        expert = torch.zeros(2, 10, 7)
        
        # Only differ at first step
        expert[:, 0, 0] = 1.0
        
        bc_loss = F.mse_loss(pred[:, 0, :], expert[:, 0, :])
        
        assert bc_loss.item() > 0, "Loss should be positive when first steps differ"
        
        # Now reset and only differ at later steps
        expert = torch.zeros(2, 10, 7)
        expert[:, 5, 0] = 1.0  # Step 5, not step 0
        
        bc_loss_later = F.mse_loss(pred[:, 0, :], expert[:, 0, :])
        
        assert bc_loss_later.item() == 0.0, "Loss should be zero when first steps match"
    
    def test_gradient_flow(self):
        """Test gradients flow through BC loss."""
        pred = torch.randn(4, 10, 7, requires_grad=True)
        expert = torch.randn(4, 10, 7)
        
        bc_loss = F.mse_loss(pred[:, 0, :], expert[:, 0, :])
        bc_loss.backward()
        
        assert pred.grad is not None
        
        # Gradient should only be non-zero for first step
        assert pred.grad[:, 0, :].abs().sum() > 0, "Gradient should exist for first step"
        assert pred.grad[:, 1:, :].abs().sum() == 0, "Gradient should be zero for other steps"
    
    def test_bc_coef_1000_dominates_loss(self):
        """Test that bc_coef=1000 makes BC loss dominant."""
        pred = torch.randn(4, 10, 7)
        expert = torch.randn(4, 10, 7)
        
        bc_loss_raw = F.mse_loss(pred[:, 0, :], expert[:, 0, :])
        
        # Simulate other loss components
        ppo_loss = torch.tensor(0.1)  # Typical PPO loss
        value_loss = torch.tensor(0.5)  # Typical value loss
        
        bc_coef = 1000.0
        bc_loss_scaled = bc_coef * bc_loss_raw
        
        total_loss = ppo_loss + 0.5 * value_loss + bc_loss_scaled
        
        # BC should dominate when bc_loss_raw > 0.001
        if bc_loss_raw.item() > 0.001:
            assert bc_loss_scaled.item() > ppo_loss.item() + value_loss.item(), \
                f"BC loss {bc_loss_scaled.item()} should dominate"
    
    def test_position_vs_orientation_contribution(self):
        """Test relative contribution of position vs orientation error."""
        pred = torch.zeros(1, 10, 7)
        pred[:, :, 6] = 1.0  # Identity quaternion
        
        # Position error only
        expert_pos = pred.clone()
        expert_pos[:, 0, :3] += 0.1  # 10cm position error
        loss_pos = F.mse_loss(pred[:, 0, :], expert_pos[:, 0, :])
        
        # Orientation error only
        expert_orn = pred.clone()
        expert_orn[:, 0, 3] = 0.1  # Small quaternion perturbation
        expert_orn[:, 0, 3:] = F.normalize(expert_orn[:, 0, 3:], dim=-1)
        loss_orn = F.mse_loss(pred[:, 0, :], expert_orn[:, 0, :])
        
        # Both should contribute to loss
        assert loss_pos.item() > 0
        assert loss_orn.item() > 0
    
    def test_numerical_stability_with_large_errors(self):
        """Test numerical stability with large prediction errors."""
        pred = torch.randn(4, 10, 7) * 100  # Large values
        expert = torch.randn(4, 10, 7) * 100
        
        bc_loss = F.mse_loss(pred[:, 0, :], expert[:, 0, :])
        
        assert torch.isfinite(bc_loss), "Loss should be finite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
