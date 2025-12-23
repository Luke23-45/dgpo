# FILE: tests/test_rollout_buffer.py
"""
Unit tests for RolloutBuffer in train_dgpo_robust.py

Tests:
1. Add operation - data integrity
2. Clear operation - proper cleanup
3. Length tracking - correct counting
4. Data type preservation
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
import torch
import numpy as np

from train.train_dgpo_robust import RolloutBuffer


class TestRolloutBuffer:
    """Test suite for RolloutBuffer component."""
    
    @pytest.fixture
    def buffer(self):
        """Create a RolloutBuffer instance."""
        return RolloutBuffer()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return {
            'prev_img': np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            'curr_img': np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            'goal_img': np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            'proprio': np.random.randn(22).astype(np.float32),
            'visual_emb': torch.randn(768),
            'action_chunk': np.random.randn(10, 7).astype(np.float32),
            'log_prob': np.random.randn(),
            'reward': 1.5,
            'value': 0.8,
            'done': False,
            'expert_pose_chunk': np.random.randn(10, 7).astype(np.float32),
            'expert_phase': 2
        }
    
    def test_initial_empty_buffer(self, buffer):
        """Test that buffer starts empty."""
        assert len(buffer) == 0
        assert len(buffer.rewards) == 0
        assert len(buffer.values) == 0
    
    def test_add_single_transition(self, buffer, sample_data):
        """Test adding a single transition."""
        buffer.add(**sample_data)
        
        assert len(buffer) == 1
        assert len(buffer.prev_images) == 1
        assert len(buffer.curr_images) == 1
        assert len(buffer.goal_images) == 1
        assert len(buffer.proprios) == 1
        assert len(buffer.action_chunks) == 1
        assert len(buffer.log_probs) == 1
        assert len(buffer.rewards) == 1
        assert len(buffer.values) == 1
        assert len(buffer.dones) == 1
        assert len(buffer.expert_pose_chunks) == 1
        assert len(buffer.expert_phases) == 1
    
    def test_add_multiple_transitions(self, buffer, sample_data):
        """Test adding multiple transitions."""
        for i in range(10):
            data = sample_data.copy()
            data['reward'] = float(i)
            buffer.add(**data)
        
        assert len(buffer) == 10
        
        # Verify rewards are stored correctly
        for i, r in enumerate(buffer.rewards):
            assert r == float(i)
    
    def test_clear_empties_buffer(self, buffer, sample_data):
        """Test that clear removes all data."""
        for _ in range(5):
            buffer.add(**sample_data)
        
        assert len(buffer) == 5
        
        buffer.clear()
        
        assert len(buffer) == 0
        assert len(buffer.prev_images) == 0
        assert len(buffer.rewards) == 0
    
    def test_data_type_preservation_numpy(self, buffer, sample_data):
        """Test that numpy arrays are preserved."""
        buffer.add(**sample_data)
        
        assert isinstance(buffer.prev_images[0], np.ndarray)
        assert isinstance(buffer.proprios[0], np.ndarray)
        assert isinstance(buffer.action_chunks[0], np.ndarray)
        assert isinstance(buffer.expert_pose_chunks[0], np.ndarray)
    
    def test_data_type_preservation_torch(self, buffer, sample_data):
        """Test that torch tensors are preserved."""
        buffer.add(**sample_data)
        
        assert isinstance(buffer.visual_embeddings[0], torch.Tensor)
    
    def test_data_type_preservation_primitives(self, buffer, sample_data):
        """Test that primitive types are preserved."""
        buffer.add(**sample_data)
        
        assert isinstance(buffer.rewards[0], float)
        assert isinstance(buffer.values[0], float)
        assert isinstance(buffer.dones[0], bool)
        assert isinstance(buffer.expert_phases[0], int)
    
    def test_log_prob_storage(self, buffer, sample_data):
        """Test log probability storage."""
        sample_data['log_prob'] = -5.5
        buffer.add(**sample_data)
        
        assert buffer.log_probs[0] == -5.5
    
    def test_done_flag_storage(self, buffer, sample_data):
        """Test done flag storage."""
        sample_data['done'] = True
        buffer.add(**sample_data)
        
        assert buffer.dones[0] == True
        
        sample_data['done'] = False
        buffer.add(**sample_data)
        
        assert buffer.dones[1] == False
    
    def test_image_shapes(self, buffer, sample_data):
        """Test that image shapes are preserved."""
        buffer.add(**sample_data)
        
        assert buffer.prev_images[0].shape == (256, 256, 3)
        assert buffer.curr_images[0].shape == (256, 256, 3)
        assert buffer.goal_images[0].shape == (256, 256, 3)
    
    def test_chunk_shapes(self, buffer, sample_data):
        """Test that chunk shapes are preserved."""
        buffer.add(**sample_data)
        
        assert buffer.action_chunks[0].shape == (10, 7)
        assert buffer.expert_pose_chunks[0].shape == (10, 7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
