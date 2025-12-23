# FILE: tests/test_vision_critic.py
"""
Unit tests for VisionCritic in train_dgpo_robust.py

Tests:
1. Forward pass shapes - validates output dimensions
2. Gradient flow - ensures gradients propagate through the network
3. Value estimation - numerically tests value prediction behavior
"""
import sys
import os
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
import torch
import numpy as np

from train.train_dgpo_robust import VisionCritic


class TestVisionCritic:
    """Test suite for VisionCritic component."""
    
    @pytest.fixture
    def critic(self):
        """Create a VisionCritic instance for testing."""
        return VisionCritic(
            vision_feature_dim=768,
            proprio_dim=22,
            hidden_dim=256
        )
    
    def test_forward_output_shape(self, critic):
        """Test that forward pass produces correct output shape."""
        batch_size = 4
        visual_emb = torch.randn(batch_size, 768)
        proprio = torch.randn(batch_size, 22)
        
        output = critic(visual_emb, proprio)
        
        assert output.shape == (batch_size,), f"Expected ({batch_size},), got {output.shape}"
    
    def test_forward_single_sample(self, critic):
        """Test forward pass with single sample."""
        visual_emb = torch.randn(1, 768)
        proprio = torch.randn(1, 22)
        
        output = critic(visual_emb, proprio)
        
        assert output.shape == (1,)
        assert torch.isfinite(output).all(), "Output contains NaN or Inf"
    
    def test_gradient_flow(self, critic):
        """Test that gradients flow through the entire network."""
        visual_emb = torch.randn(4, 768, requires_grad=True)
        proprio = torch.randn(4, 22, requires_grad=True)
        
        output = critic(visual_emb, proprio)
        loss = output.mean()
        loss.backward()
        
        # Check gradients exist for input
        assert visual_emb.grad is not None, "No gradient for visual_emb"
        assert proprio.grad is not None, "No gradient for proprio"
        
        # Check gradients exist for network parameters
        for name, param in critic.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Invalid gradient for {name}"
    
    def test_value_prediction_numerical_stability(self, critic):
        """Test numerical stability with extreme inputs."""
        # Test with zeros
        visual_emb_zeros = torch.zeros(2, 768)
        proprio_zeros = torch.zeros(2, 22)
        output_zeros = critic(visual_emb_zeros, proprio_zeros)
        assert torch.isfinite(output_zeros).all(), "Output not finite with zero input"
        
        # Test with large values
        visual_emb_large = torch.randn(2, 768) * 100
        proprio_large = torch.randn(2, 22) * 100
        output_large = critic(visual_emb_large, proprio_large)
        assert torch.isfinite(output_large).all(), "Output not finite with large input"
    
    def test_deterministic_output(self, critic):
        """Test that same input produces same output."""
        visual_emb = torch.randn(2, 768)
        proprio = torch.randn(2, 22)
        
        critic.eval()
        output1 = critic(visual_emb, proprio)
        output2 = critic(visual_emb, proprio)
        
        assert torch.allclose(output1, output2), "Output is not deterministic"
    
    def test_device_compatibility(self, critic):
        """Test CPU compatibility (CUDA tested separately if available)."""
        visual_emb = torch.randn(2, 768)
        proprio = torch.randn(2, 22)
        
        critic_cpu = critic.cpu()
        output = critic_cpu(visual_emb.cpu(), proprio.cpu())
        
        assert output.device.type == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
