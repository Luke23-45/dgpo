# FILE: utils/online_buffer.py
# (Definitive, SOTA, RAM-Optimized Replay Buffer)

"""
Online Replay Buffer for DAgger (Dataset Aggregation).

This module implements a high-performance, RAM-efficient buffer to store
and serve "Correction" data generated during the online phase of training.

Key Features:
1.  **Ring Buffer Semantics**: Uses `deque(maxlen=N)` to automatically discard
    stale data when capacity is reached, preventing memory leaks.
2.  **Lazy Transformation**: Stores raw uint8 numpy arrays to minimize memory footprint,
    applying Torch transforms only on-the-fly during `__getitem__`.
3.  **Synthetic Advantage**: Allows injecting a high static advantage value for
    correction samples to force the AWR loss to learn them.
4.  **Schema Compliance**: Output dictionary perfectly mimics `SemanticPlannerDataset`.
"""

from __future__ import annotations

import logging
import pickle
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Setup Logger
log = logging.getLogger(__name__)

class OnlineReplayBuffer(Dataset):
    """
    A First-In-First-Out (FIFO) Replay Buffer compatible with PyTorch DataLoaders.
    Stores transition tuples (State, ExpertAction) for Online Learning.
    """

    def __init__(self, 
                 capacity: int = 10000, 
                 resize_size: Tuple[int, int] = (224, 224),
                 default_advantage: float = 10.0):
        """
        Args:
            capacity: Max number of frames to store. Oldest are dropped first.
            resize_size: Target image size for the Vision Transformer.
            default_advantage: The scalar 'score' assigned to expert corrections.
                               Should be high enough to generate large AWR weights.
        """
        self.capacity = capacity
        self.default_advantage = default_advantage
        
        # The Core Storage
        self.buffer = deque(maxlen=capacity)
        
        # Transform Pipeline (Matches SemanticPlannerDataset)
        self.transform = transforms.Compose([
            transforms.Resize(resize_size, antialias=True),
            transforms.ToTensor()
            # Note: No random augmentation here by default to preserve
            # the exact pixel geometry of the specific correction case.
        ])
        
        log.info(f"OnlineReplayBuffer initialized. Capacity: {capacity} frames.")

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, 
            initial_image: np.ndarray, 
            goal_image: np.ndarray,
            task_phase: int,
            current_proprio: np.ndarray,
            gt_pose: np.ndarray,
            gt_gripper: float,
            advantage: Optional[float] = None):
        """
        Add a single timestep 'Correction' to the buffer.
        
        Args:
            initial_image: (H, W, 3) uint8 (Current View)
            goal_image: (H, W, 3) uint8 (Target View)
            task_phase: int (0-5)
            current_proprio: (D,) float32 (Current joint state)
            gt_pose: (7,) float32 (Expert's intended Pose)
            gt_gripper: float (0.0 or 1.0) (Expert's intended Gripper)
            advantage: Override for the advantage score (default: self.default_advantage)
        """
        
        # Data Sanitization & Compression
        # We ensure data is stored as standard Python/Numpy types to avoid
        # keeping Torch computation graphs alive in RAM (Memory Leak protection).
        
        sample = {
            # Inputs
            'initial_image': initial_image.astype(np.uint8),
            'goal_image': goal_image.astype(np.uint8),
            'task_phase': int(task_phase),
            'current_proprio': current_proprio.astype(np.float32),
            
            # Targets (The Expert Correction)
            'gt_pose': gt_pose.astype(np.float32),
            'gt_gripper': float(gt_gripper),
            
            # Metadata
            'advantage': float(advantage) if advantage is not None else self.default_advantage
        }
        
        self.buffer.append(sample)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieve a sample formatted for the SemanticPlanner model.
        """
        sample = self.buffer[idx]
        
        # 1. Process Images (On-the-fly)
        # Convert numpy -> PIL -> Tensor
        initial_img_pil = Image.fromarray(sample['initial_image'])
        goal_img_pil = Image.fromarray(sample['goal_image'])
        
        initial_img_t = self.transform(initial_img_pil)
        goal_img_t = self.transform(goal_img_pil)
        
        # 2. Convert Metadata to Tensors
        # We explicitly create new tensors to ensure memory isolation
        task_phase_t = torch.tensor(sample['task_phase'], dtype=torch.long)
        proprio_t = torch.tensor(sample['current_proprio'], dtype=torch.float32)
        
        gt_pose_t = torch.tensor(sample['gt_pose'], dtype=torch.float32)
        gt_gripper_t = torch.tensor([sample['gt_gripper']], dtype=torch.float32)
        advantage_t = torch.tensor([sample['advantage']], dtype=torch.float32)
        
        # 3. Assemble Output
        # Keys must match SemanticPlannerDataset.__getitem__ EXACTLY
        return {
            # Inputs
            'initial_image': initial_img_t,
            'goal_image': goal_img_t,
            'task_phase': task_phase_t,
            'current_proprio': proprio_t,
            
            # Targets
            'ground_truth_subgoal_pose': gt_pose_t,
            'ground_truth_gripper_state': gt_gripper_t,
            
            # AWR Weight
            'advantage': advantage_t
        }

    def save_to_disk(self, path: str):
        """Persist buffer to disk for debugging or resumption."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, 'wb') as f:
            pickle.dump(list(self.buffer), f)
        log.info(f"Saved {len(self)} samples to {path}")

    def load_from_disk(self, path: str):
        """Load a buffer from disk."""
        path_obj = Path(path)
        if not path_obj.exists():
            log.warning(f"Buffer path {path} not found. Starting empty.")
            return
        
        with open(path_obj, 'rb') as f:
            data_list = pickle.load(f)
        
        self.buffer.clear()
        self.buffer.extend(data_list)
        log.info(f"Loaded {len(self)} samples from {path}")