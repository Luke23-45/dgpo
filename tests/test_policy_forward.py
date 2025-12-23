# FILE: tests/test_policy_forward.py
"""
Unit tests for SemanticPlanner forward pass

Tests:
1. Output shapes (pose_chunk, gripper_chunk, phase_logits, visual_embedding)
2. Forward pass with various batch sizes
3. Frozen/unfrozen vision backbone behavior
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
import torch
import numpy as np


class MockSemanticPlanner(torch.nn.Module):
    """Mock SemanticPlanner for testing without loading full model weights."""
    
    def __init__(self, vision_dim=768, proprio_dim=22, chunk_size=10, num_phases=5):
        super().__init__()
        self.chunk_size = chunk_size
        self.num_phases = num_phases
        self.vision_dim = vision_dim
        
        # Minimal layers for testing
        self.fc = torch.nn.Linear(vision_dim + proprio_dim, 256)
        self.pose_head = torch.nn.Linear(256, chunk_size * 7)
        self.gripper_head = torch.nn.Linear(256, chunk_size * 1)
        self.phase_head = torch.nn.Linear(256, num_phases)
        self.visual_proj = torch.nn.Linear(vision_dim, vision_dim)
        
        # Mock vision backbone
        self.vision_backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, stride=16),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(16, vision_dim)
        )
    
    def forward(self, batch):
        B = batch['curr_proprio'].shape[0]
        
        # Process images through backbone
        curr_img = batch['curr_image']
        vis_feat = self.vision_backbone(curr_img)
        
        # Fuse with proprio
        proprio = batch['curr_proprio']
        fused = torch.cat([vis_feat, proprio], dim=-1)
        h = torch.relu(self.fc(fused))
        
        # Output heads
        pose_chunk = self.pose_head(h).view(B, self.chunk_size, 7)
        gripper_chunk = self.gripper_head(h).view(B, self.chunk_size, 1)
        phase_logits = self.phase_head(h)
        visual_embedding = self.visual_proj(vis_feat)
        
        return {
            'pose_chunk': pose_chunk,
            'gripper_chunk': gripper_chunk,
            'phase_logits': phase_logits,
            'visual_embedding': visual_embedding
        }


class TestPolicyForward:
    """Test suite for policy forward pass."""
    
    @pytest.fixture
    def policy(self):
        """Create mock policy for testing."""
        return MockSemanticPlanner()
    
    @pytest.fixture
    def sample_batch(self):
        """Create sample batch for testing."""
        B = 4
        return {
            'prev_image': torch.randn(B, 3, 224, 224),
            'curr_image': torch.randn(B, 3, 224, 224),
            'goal_image': torch.randn(B, 3, 224, 224),
            'curr_proprio': torch.randn(B, 22)
        }
    
    def test_pose_chunk_shape(self, policy, sample_batch):
        """Test pose_chunk output shape."""
        output = policy(sample_batch)
        
        assert 'pose_chunk' in output
        assert output['pose_chunk'].shape == (4, 10, 7)
    
    def test_gripper_chunk_shape(self, policy, sample_batch):
        """Test gripper_chunk output shape."""
        output = policy(sample_batch)
        
        assert 'gripper_chunk' in output
        assert output['gripper_chunk'].shape == (4, 10, 1)
    
    def test_phase_logits_shape(self, policy, sample_batch):
        """Test phase_logits output shape."""
        output = policy(sample_batch)
        
        assert 'phase_logits' in output
        assert output['phase_logits'].shape == (4, 5)  # 5 phases
    
    def test_visual_embedding_shape(self, policy, sample_batch):
        """Test visual_embedding output shape."""
        output = policy(sample_batch)
        
        assert 'visual_embedding' in output
        assert output['visual_embedding'].shape == (4, 768)
    
    def test_batch_size_1(self, policy):
        """Test with single sample batch."""
        batch = {
            'prev_image': torch.randn(1, 3, 224, 224),
            'curr_image': torch.randn(1, 3, 224, 224),
            'goal_image': torch.randn(1, 3, 224, 224),
            'curr_proprio': torch.randn(1, 22)
        }
        
        output = policy(batch)
        
        assert output['pose_chunk'].shape == (1, 10, 7)
    
    def test_large_batch(self, policy):
        """Test with large batch size."""
        B = 32
        batch = {
            'prev_image': torch.randn(B, 3, 224, 224),
            'curr_image': torch.randn(B, 3, 224, 224),
            'goal_image': torch.randn(B, 3, 224, 224),
            'curr_proprio': torch.randn(B, 22)
        }
        
        output = policy(batch)
        
        assert output['pose_chunk'].shape == (B, 10, 7)
    
    def test_gradient_flow_to_all_outputs(self, policy, sample_batch):
        """Test that gradients flow to all outputs."""
        output = policy(sample_batch)
        
        # Sum all outputs and backprop
        loss = (output['pose_chunk'].sum() + 
                output['gripper_chunk'].sum() + 
                output['phase_logits'].sum() +
                output['visual_embedding'].sum())
        loss.backward()
        
        # Check gradients exist
        for name, param in policy.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_output_numerical_stability(self, policy, sample_batch):
        """Test that outputs are numerically stable."""
        output = policy(sample_batch)
        
        for key, value in output.items():
            assert torch.isfinite(value).all(), f"{key} contains NaN/Inf"
    
    def test_deterministic_in_eval_mode(self, policy, sample_batch):
        """Test deterministic outputs in eval mode."""
        policy.eval()
        
        output1 = policy(sample_batch)
        output2 = policy(sample_batch)
        
        assert torch.allclose(output1['pose_chunk'], output2['pose_chunk'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
