# FILE: tests/test_riemannian_divergence.py
"""
Unit tests for Riemannian divergence computation in utils/riemannian_diff.py

Tests:
1. pose7d_to_matrix conversion
2. SE(3) log map computation
3. Phase-weighted divergence
4. AMP (mixed precision) compatibility
5. Numerical stability
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
import torch
import numpy as np

from utils.riemannian_diff import (
    pose7d_to_matrix,
    se3_log_map,
    compute_riemannian_divergence
)


class TestPose7dToMatrix:
    """Test suite for pose7d_to_matrix function."""
    
    def test_output_shape(self):
        """Test output shape is (B, 4, 4)."""
        pose = torch.randn(4, 7)
        pose[:, 3:] = torch.nn.functional.normalize(pose[:, 3:], dim=-1)
        
        matrix = pose7d_to_matrix(pose)
        
        assert matrix.shape == (4, 4, 4)
    
    def test_identity_quaternion(self):
        """Test identity quaternion [0, 0, 0, 1] produces identity rotation."""
        pose = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        
        matrix = pose7d_to_matrix(pose)
        
        # Rotation part should be identity
        expected_rot = torch.eye(3)
        assert torch.allclose(matrix[0, :3, :3], expected_rot, atol=1e-5)
    
    def test_translation_preserved(self):
        """Test that translation vector is preserved."""
        pose = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]])
        
        matrix = pose7d_to_matrix(pose)
        
        assert torch.allclose(matrix[0, :3, 3], pose[0, :3], atol=1e-5)
    
    def test_homogeneous_row(self):
        """Test that last row is [0, 0, 0, 1]."""
        pose = torch.randn(2, 7)
        pose[:, 3:] = torch.nn.functional.normalize(pose[:, 3:], dim=-1)
        
        matrix = pose7d_to_matrix(pose)
        
        expected_last_row = torch.tensor([0., 0., 0., 1.])
        assert torch.allclose(matrix[:, 3, :], expected_last_row.unsqueeze(0).expand(2, -1), atol=1e-5)
    
    def test_rotation_orthogonality(self):
        """Test that rotation matrix is orthogonal."""
        pose = torch.randn(3, 7)
        pose[:, 3:] = torch.nn.functional.normalize(pose[:, 3:], dim=-1)
        
        matrix = pose7d_to_matrix(pose)
        R = matrix[:, :3, :3]
        
        # R @ R.T should be identity
        identity = torch.eye(3).unsqueeze(0).expand(3, -1, -1)
        RRT = R @ R.transpose(-2, -1)
        
        assert torch.allclose(RRT, identity, atol=1e-4)


class TestSE3LogMap:
    """Test suite for se3_log_map function."""
    
    def test_output_shape(self):
        """Test output shape is (B, 6)."""
        T = torch.eye(4).unsqueeze(0).expand(3, -1, -1)
        
        twist = se3_log_map(T)
        
        assert twist.shape == (3, 6)
    
    def test_identity_transform_gives_zero_twist(self):
        """Test that identity transform gives zero twist."""
        T = torch.eye(4).unsqueeze(0)
        
        twist = se3_log_map(T)
        
        assert torch.allclose(twist, torch.zeros(1, 6), atol=1e-4)
    
    def test_pure_translation(self):
        """Test log map of pure translation."""
        T = torch.eye(4).unsqueeze(0)
        T[0, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
        
        twist = se3_log_map(T)
        
        # Translation part should be preserved
        assert torch.allclose(twist[0, :3], torch.tensor([1.0, 2.0, 3.0]), atol=1e-4)
        # Rotation part should be zero
        assert torch.allclose(twist[0, 3:], torch.zeros(3), atol=1e-4)
    
    def test_numerical_stability(self):
        """Test numerical stability with near-identity rotations."""
        # Small rotation angle
        T = torch.eye(4).unsqueeze(0).expand(5, -1, -1).clone()
        
        twist = se3_log_map(T)
        
        assert torch.isfinite(twist).all()


class TestComputeRiemannianDivergence:
    """Test suite for compute_riemannian_divergence function."""
    
    def test_output_shape(self):
        """Test output shape is (B,)."""
        B, K = 4, 10
        pred_chunk = torch.randn(B, K, 7)
        pred_chunk[..., 3:] = torch.nn.functional.normalize(pred_chunk[..., 3:], dim=-1)
        
        expert_chunk = torch.randn(B, K, 7)
        expert_chunk[..., 3:] = torch.nn.functional.normalize(expert_chunk[..., 3:], dim=-1)
        
        phase_scores = torch.randn(B, 5)  # 5 phases
        
        divergence = compute_riemannian_divergence(pred_chunk, expert_chunk, phase_scores)
        
        assert divergence.shape == (B,)
    
    def test_zero_divergence_for_identical_poses(self):
        """Test that identical poses give zero divergence."""
        B, K = 2, 10
        chunk = torch.zeros(B, K, 7)
        chunk[..., 6] = 1.0  # Identity quaternion
        
        phase_scores = torch.zeros(B, 5)
        phase_scores[:, 0] = 10.0  # High logit for phase 0
        
        divergence = compute_riemannian_divergence(chunk, chunk.clone(), phase_scores)
        
        assert torch.allclose(divergence, torch.zeros(B), atol=1e-3)
    
    def test_divergence_increases_with_error(self):
        """Test that divergence increases as poses differ."""
        B, K = 1, 10
        
        pred_chunk = torch.zeros(B, K, 7)
        pred_chunk[..., 6] = 1.0
        
        # Expert with increasing position offset
        divergences = []
        for offset in [0.0, 0.1, 0.5, 1.0]:
            expert_chunk = pred_chunk.clone()
            expert_chunk[..., 0] += offset  # X offset
            
            phase_scores = torch.zeros(B, 5)
            
            div = compute_riemannian_divergence(pred_chunk, expert_chunk, phase_scores)
            divergences.append(div.item())
        
        # Divergence should increase monotonically
        for i in range(len(divergences) - 1):
            assert divergences[i] <= divergences[i + 1], \
                f"Divergence should increase: {divergences}"
    
    def test_phase_weighting_affects_divergence(self):
        """Test that different phases produce different weights."""
        B, K = 1, 10
        
        pred_chunk = torch.zeros(B, K, 7)
        pred_chunk[..., 6] = 1.0
        
        expert_chunk = pred_chunk.clone()
        expert_chunk[..., 0] += 0.1  # Same position error
        
        # Phase 0 (Approach) vs Phase 1 (Grasp)
        phase_0 = torch.zeros(B, 5)
        phase_0[:, 0] = 10.0
        
        phase_1 = torch.zeros(B, 5)
        phase_1[:, 1] = 10.0
        
        div_0 = compute_riemannian_divergence(pred_chunk, expert_chunk, phase_0)
        div_1 = compute_riemannian_divergence(pred_chunk, expert_chunk, phase_1)
        
        # Different phases should give different divergence (due to different weights)
        # Phase 1 (Grasp) has higher weights, so divergence should be higher
        assert div_1.item() > div_0.item()
    
    def test_amp_compatibility(self):
        """Test compatibility with Automatic Mixed Precision."""
        B, K = 2, 10
        pred_chunk = torch.randn(B, K, 7, dtype=torch.float16)
        pred_chunk[..., 3:] = torch.nn.functional.normalize(pred_chunk[..., 3:].float(), dim=-1).half()
        
        expert_chunk = torch.randn(B, K, 7, dtype=torch.float16)
        expert_chunk[..., 3:] = torch.nn.functional.normalize(expert_chunk[..., 3:].float(), dim=-1).half()
        
        phase_scores = torch.randn(B, 5, dtype=torch.float16)
        
        # Should not throw dtype mismatch error
        try:
            divergence = compute_riemannian_divergence(pred_chunk, expert_chunk, phase_scores)
            assert torch.isfinite(divergence).all()
        except RuntimeError as e:
            pytest.fail(f"AMP compatibility issue: {e}")
    
    def test_gradient_flow(self):
        """Test that gradients flow through divergence computation."""
        B, K = 2, 10
        pred_chunk = torch.randn(B, K, 7, requires_grad=True)
        
        expert_chunk = torch.randn(B, K, 7)
        expert_chunk[..., 3:] = torch.nn.functional.normalize(expert_chunk[..., 3:], dim=-1)
        
        phase_scores = torch.randn(B, 5)
        
        divergence = compute_riemannian_divergence(pred_chunk, expert_chunk, phase_scores)
        loss = divergence.sum()
        loss.backward()
        
        assert pred_chunk.grad is not None
        assert pred_chunk.grad.abs().sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
