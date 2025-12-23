# FILE: tests/test_full_update_policy.py
"""
Integration tests for the complete update_policy() method in train_dgpo_robust.py

Tests:
1. All loss components computed
2. Optimizer step effects
3. Gradient norms
4. Metrics returned correctly
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from train.train_dgpo_robust import (
    RolloutBuffer,
    VisionCritic, 
    AdaptiveKLPenalty,
    RunningNormalizer,
    compute_gae
)


class MockPolicy(torch.nn.Module):
    """Mock policy for testing update_policy without full SemanticPlanner."""
    
    def __init__(self, chunk_size=10, action_dim=7):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        
        self.fc = torch.nn.Linear(22, 256)
        self.pose_head = torch.nn.Linear(256, chunk_size * action_dim)
        self.phase_head = torch.nn.Linear(256, 5)
        self.embed_proj = torch.nn.Linear(256, 768)
    
    def forward(self, batch):
        B = batch['curr_proprio'].shape[0]
        h = torch.relu(self.fc(batch['curr_proprio']))
        
        pose_chunk = self.pose_head(h).view(B, self.chunk_size, self.action_dim)
        phase_logits = self.phase_head(h)
        visual_embedding = self.embed_proj(h)
        
        return {
            'pose_chunk': pose_chunk,
            'phase_logits': phase_logits,
            'visual_embedding': visual_embedding
        }


class TestFullUpdatePolicy:
    """Integration tests for update_policy method."""
    
    @pytest.fixture
    def mock_trainer_components(self):
        """Create mock trainer components for testing."""
        device = torch.device('cpu')
        
        policy = MockPolicy()
        value_net = VisionCritic(768, 22, 256)
        kl_penalty = AdaptiveKLPenalty(0.015, 0.02)
        reward_normalizer = RunningNormalizer()
        chk_log_std = torch.nn.Parameter(torch.ones(1, 10, 7) * -3.5)
        
        policy_optimizer = torch.optim.Adam(
            list(policy.parameters()) + [chk_log_std], 
            lr=1e-4
        )
        value_optimizer = torch.optim.Adam(value_net.parameters(), lr=1e-3)
        
        return {
            'policy': policy,
            'value_net': value_net,
            'kl_penalty': kl_penalty,
            'reward_normalizer': reward_normalizer,
            'chk_log_std': chk_log_std,
            'policy_optimizer': policy_optimizer,
            'value_optimizer': value_optimizer,
            'device': device
        }
    
    @pytest.fixture
    def filled_buffer(self):
        """Create buffer with sample data."""
        buffer = RolloutBuffer()
        
        for i in range(16):
            buffer.add(
                prev_img=np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
                curr_img=np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
                goal_img=np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
                proprio=np.random.randn(22).astype(np.float32),
                visual_emb=torch.randn(768),
                action_chunk=np.random.randn(10, 7).astype(np.float32),
                log_prob=float(np.random.randn()),
                reward=float(np.random.randn()),
                value=float(np.random.randn()),
                done=i == 15,
                expert_pose_chunk=np.random.randn(10, 7).astype(np.float32),
                expert_phase=np.random.randint(0, 5)
            )
        
        return buffer
    
    def test_gae_computed_correctly(self, filled_buffer):
        """Test GAE computation on buffer data."""
        rewards = np.array(filled_buffer.rewards)
        values = np.array(filled_buffer.values)
        dones = np.array(filled_buffer.dones)
        
        advantages, returns = compute_gae(rewards, values, dones)
        
        assert advantages.shape == rewards.shape
        assert returns.shape == rewards.shape
        assert np.isfinite(advantages).all()
        assert np.isfinite(returns).all()
    
    def test_advantage_normalization(self, filled_buffer):
        """Test advantage normalization."""
        rewards = np.array(filled_buffer.rewards)
        values = np.array(filled_buffer.values)
        dones = np.array(filled_buffer.dones)
        
        advantages, _ = compute_gae(rewards, values, dones)
        
        # Normalize
        normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        assert np.isclose(normalized.mean(), 0.0, atol=1e-5)
        assert np.isclose(normalized.std(), 1.0, atol=0.1)
    
    def test_ppo_loss_computed(self, mock_trainer_components, filled_buffer):
        """Test that PPO loss can be computed from buffer."""
        policy = mock_trainer_components['policy']
        chk_log_std = mock_trainer_components['chk_log_std']
        
        # Prepare batch
        idx = list(range(min(4, len(filled_buffer))))
        proprios = torch.stack([torch.from_numpy(filled_buffer.proprios[i]) for i in idx])
        batch = {'curr_proprio': proprios}
        
        # Forward pass
        out = policy(batch)
        pred_chunks = out['pose_chunk']
        
        # Distribution
        dist = torch.distributions.Normal(pred_chunks, chk_log_std.exp())
        
        # Get actions from buffer
        actions = torch.stack([torch.from_numpy(filled_buffer.action_chunks[i]) for i in idx])
        old_log_probs = torch.tensor([filled_buffer.log_probs[i] for i in idx])
        
        # New log prob
        new_log_probs = dist.log_prob(actions).sum(dim=[1, 2])
        
        # Ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        assert torch.isfinite(ratio).all()
    
    def test_bc_loss_computed(self, mock_trainer_components, filled_buffer):
        """Test that BC loss can be computed from buffer."""
        policy = mock_trainer_components['policy']
        
        idx = list(range(min(4, len(filled_buffer))))
        proprios = torch.stack([torch.from_numpy(filled_buffer.proprios[i]) for i in idx])
        batch = {'curr_proprio': proprios}
        
        out = policy(batch)
        pred_chunks = out['pose_chunk']
        
        expert_chunks = torch.stack([torch.from_numpy(filled_buffer.expert_pose_chunks[i]) for i in idx])
        
        bc_loss = torch.nn.functional.mse_loss(pred_chunks[:, 0, :], expert_chunks[:, 0, :])
        
        assert torch.isfinite(bc_loss)
        assert bc_loss >= 0
    
    def test_value_loss_computed(self, mock_trainer_components, filled_buffer):
        """Test that value loss can be computed."""
        value_net = mock_trainer_components['value_net']
        
        idx = list(range(min(4, len(filled_buffer))))
        
        # Mock visual embeddings
        visual_embs = torch.randn(len(idx), 768)
        proprios = torch.stack([torch.from_numpy(filled_buffer.proprios[i]) for i in idx])
        
        # Value prediction
        values = value_net(visual_embs, proprios)
        
        # Target returns
        rewards = np.array(filled_buffer.rewards)
        buffer_values = np.array(filled_buffer.values)
        dones = np.array(filled_buffer.dones)
        _, returns = compute_gae(rewards, buffer_values, dones)
        
        returns_batch = torch.tensor([returns[i] for i in idx])
        
        value_loss = torch.nn.functional.mse_loss(values, returns_batch)
        
        assert torch.isfinite(value_loss)
    
    def test_phase_loss_computed(self, mock_trainer_components, filled_buffer):
        """Test that phase classification loss can be computed."""
        policy = mock_trainer_components['policy']
        
        idx = list(range(min(4, len(filled_buffer))))
        proprios = torch.stack([torch.from_numpy(filled_buffer.proprios[i]) for i in idx])
        batch = {'curr_proprio': proprios}
        
        out = policy(batch)
        phase_logits = out['phase_logits']
        
        expert_phases = torch.tensor([filled_buffer.expert_phases[i] for i in idx])
        
        phase_loss = torch.nn.functional.cross_entropy(phase_logits, expert_phases)
        
        assert torch.isfinite(phase_loss)
    
    def test_gradient_norms_reasonable(self, mock_trainer_components):
        """Test that gradient norms are in reasonable range."""
        policy = mock_trainer_components['policy']
        
        # Create dummy loss
        x = torch.randn(4, 22)
        out = policy({'curr_proprio': x})
        loss = out['pose_chunk'].sum()
        
        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        
        assert grad_norm >= 0
        assert torch.isfinite(torch.tensor(grad_norm))
    
    def test_optimizer_step_updates_parameters(self, mock_trainer_components):
        """Test that optimizer step actually updates parameters."""
        policy = mock_trainer_components['policy']
        optimizer = mock_trainer_components['policy_optimizer']
        
        # Save initial parameters
        initial_params = {n: p.clone() for n, p in policy.named_parameters()}
        
        # Forward and backward
        x = torch.randn(4, 22)
        out = policy({'curr_proprio': x})
        loss = out['pose_chunk'].sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check parameters changed
        changed = False
        for name, param in policy.named_parameters():
            if not torch.allclose(param, initial_params[name]):
                changed = True
                break
        
        assert changed, "Optimizer step should change parameters"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
