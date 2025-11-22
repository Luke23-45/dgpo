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
        Retrieves a sample.
        
        Branch Logic:
        1. If Online Buffer is empty -> Force Static.
        2. Else -> Roll Dice vs `mix_ratio`.
           - Win: Sample random from Online Buffer (RAM, fast).
           - Lose: Sample `idx` from Static Dataset (Disk, sequential cache hit).
        """
        use_online = False
        online_len = len(self.online_dataset)

        # 1. Safety check: Can we use online data?
        if online_len > 0:
            # 2. Probabilistic switch
            if random.random() < self.mix_ratio:
                use_online = True

        if use_online:
            # Random sampling from buffer (Replacment is allowed/implied by randomint)
            # We act as a "Infinite Reservoir" for the online data within the epoch structure of static data.
            rand_idx = random.randint(0, online_len - 1)
            try:
                sample = self.online_dataset[rand_idx]
                # Inject a flag for debugging if needed, though Tensor structure must match exactly
                return sample
            except Exception as e:
                # Fallback to static on corruption/error to prevent crash
                log.warning(f"Failed to fetch online sample {rand_idx}: {e}. Falling back to static.")
                return self.static_dataset[idx]
        else:
            # Passthrough to static dataset using the sequential index provided by sampler
            return self.static_dataset[idx]

    def update_ratio(self, new_ratio: float):
        """Allows curriculum learning (increasing online ratio over time)."""
        self.mix_ratio = np.clip(new_ratio, 0.0, 1.0)
        log.info(f"MixedDataset ratio updated to {self.mix_ratio}")