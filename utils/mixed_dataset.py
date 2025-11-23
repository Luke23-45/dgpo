# FILE: utils/mixed_dataset.py
# (Definitive, SOTA, Hybrid Data Loader)

"""
Mixed Dataset Strategy for DAgger (Dataset Aggregation).

This module implements a `MixedDataset` wrapper that seamlessly blends
a massive, disk-based Static Dataset (Offline Expert) with a smaller,
RAM-based Online Buffer (Student Corrections) while maintaining high I/O throughput.

Architectural Solves:
1.  **Preservation of I/O Locality**: It is designed to work with `EpisodeAwareSampler`.
    The sampler dictates a sequential access pattern for the Static Dataset (LMDB).
    This wrapper intercepts those indices. Most of the time, it honors the sequential
    read (keeping LMDB fast). Occasionally, it swaps the read for a RAM-based
    Online sample (instant access). This avoids "seeking" on disk.
2.  **Virtual Attribute Proxy**: It masquerades as the underlying Static Dataset,
    exposing necessary attributes (like `expert_reader`) so that samplers and
    trainers remain agnostic to the mixing logic.
3.  **Dynamic Ratio**: Allows defining a fixed probability of seeing Online samples,
    ensuring the "Correction" signal remains strong even as the Static dataset dwarfs
    the Online buffer in size.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
# Setup Logger
log = logging.getLogger(__name__)


class MixedDataset(Dataset):
    """
    A dataset wrapper that stochastically interleaves samples from two sources.
    
    Logic:
        - Driving Source (Length & Indexing): The Static Dataset.
        - Injection Source: The Online Buffer.
    
    The length of this dataset equals the length of the Static Dataset.
    During iteration, an index `i` typically retrieves `static[i]`.
    However, with probability `mix_ratio`, `static[i]` is skipped (not read from disk),
    and a random sample from `online` is returned instead.
    """

    def __init__(self, 
                 static_dataset: Dataset, 
                 online_dataset: Dataset, 
                 mix_ratio: float = 0.3):
        """
        Args:
            static_dataset: The massive, LMDB-backed expert dataset.
            online_dataset: The RAM-based OnlineReplayBuffer.
            mix_ratio: Probability (0.0 to 1.0) of serving an Online sample.
                       e.g., 0.3 means 30% of the batch will be Online data.
        """
        super().__init__()
        self.static_dataset = static_dataset
        self.online_dataset = online_dataset
        self.mix_ratio = mix_ratio

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            # Note: We assume inputs are already Tensors.
            # If not, we'd add ToTensor(), but buffers usually store Tensors or Arrays
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        
        # Validate sources
        if not hasattr(static_dataset, '__getitem__'):
            raise ValueError("Static dataset must implement __getitem__")
        if not hasattr(online_dataset, '__getitem__'):
            raise ValueError("Online dataset must implement __getitem__")
            
        log.info(f"MixedDataset initialized. Base Length: {len(self.static_dataset)}. Mix Ratio: {self.mix_ratio}")

    def __len__(self) -> int:
        """
        Returns the length of the Static dataset.
        We treat one 'Epoch' as a full pass over the Static data, 
        interspersed with random Online samples.
        """
        return len(self.static_dataset)

    def __getattr__(self, name: str) -> Any:
        """
        SOTA Proxy Mechanism.
        Allows external samplers (like EpisodeAwareSampler) to inspect the 
        underlying static dataset (e.g., accessing `expert_reader`).
        """
        if hasattr(self.static_dataset, name):
            return getattr(self.static_dataset, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a sample, handling mixing, resizing, and strict Type Conversion.
        """
        use_online = False
        online_len = len(self.online_dataset)

        # 1. Probabilistic switch
        if online_len > 0 and random.random() < self.mix_ratio:
            use_online = True

        if use_online:
            rand_idx = random.randint(0, online_len - 1)
            sample = self.online_dataset[rand_idx]
            
            # --- [FIX] Uniform Type Conversion & Resizing ---
            
            # A. Images: Ensure Tensor + Resize + Normalize
            for key in ['curr_image', 'prev_image', 'goal_image']:
                if key in sample:
                    # Convert to Tensor if Numpy
                    if not isinstance(sample[key], torch.Tensor):
                         sample[key] = torch.from_numpy(sample[key])
                    
                    # Force 224x224 and Normalize
                    sample[key] = self.transform(sample[key])
            
            # B. Scalars/Arrays: Convert to Tensors to match Static Dataset types
            
            # Proprioception -> Float32
            if 'curr_proprio' in sample and not isinstance(sample['curr_proprio'], torch.Tensor):
                sample['curr_proprio'] = torch.tensor(sample['curr_proprio'], dtype=torch.float32)

            # Trajectory Targets -> Float32
            for key in ['gt_pose_chunk', 'gt_grip_chunk']:
                if key in sample and not isinstance(sample[key], torch.Tensor):
                    sample[key] = torch.tensor(sample[key], dtype=torch.float32)

            # Phase Label -> Long (Int64)
            if 'gt_phase_label' in sample and not isinstance(sample['gt_phase_label'], torch.Tensor):
                sample['gt_phase_label'] = torch.tensor(sample['gt_phase_label'], dtype=torch.long)

            # Advantage -> Float32 Tensor with shape [1]
            if 'advantage' in sample:
                val = sample['advantage']
                if not isinstance(val, torch.Tensor):
                    sample['advantage'] = torch.tensor([val], dtype=torch.float32)
            
            return sample
        else:
            # Static Dataset sample
            sample = self.static_dataset[idx]
            
            # [Safety] Enforce Image Size for Static Data too (in case of 256x256 inputs)
            for key in ['curr_image', 'prev_image', 'goal_image']:
                if key in sample:
                    if sample[key].shape[-1] != 224:
                         resizer = transforms.Resize((224, 224), antialias=True)
                         sample[key] = resizer(sample[key])
            
            return sample

    def update_ratio(self, new_ratio: float):
        """Allows curriculum learning (increasing online ratio over time)."""
        self.mix_ratio = np.clip(new_ratio, 0.0, 1.0)
        log.info(f"MixedDataset ratio updated to {self.mix_ratio}")