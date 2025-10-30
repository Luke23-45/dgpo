# FILE: utils/samplers.py
# SOTA Samplers for Efficient, Cache-Aware Data Loading (Final, Corrected Version)

import torch
from torch.utils.data import Sampler, Dataset, Subset
from typing import Iterator, Sized, List, Dict, Tuple
import logging
import numpy as np
import random
log = logging.getLogger(__name__)

class EpisodeAwareSampler(Sampler[int]):
    """
    State-of-the-art Sampler that yields indices in an episode-contiguous manner.

    This SOTA version is robustly designed to work seamlessly with both full datasets
    and `torch.utils.data.Subset` objects created by `random_split`.

    It sacrifices perfect global shuffling for vastly improved data loading performance
    by maximizing cache hits. It shuffles episodes, then yields all indices from one
    episode before moving to the next.
    """
    def __init__(self, dataset: Sized, shuffle: bool = True, seed: int = 42):
        super().__init__()
        
        # --- SOTA PATCH: ROBUST SUBSET HANDLING ---
        self.is_subset = isinstance(dataset, Subset)
        if self.is_subset:
            self.subset_indices = dataset.indices
            self.full_dataset = dataset.dataset
        else:
            self.full_dataset = dataset
            self.subset_indices = None
        
        self.num_samples = len(dataset)
        # --- END OF SOTA PATCH ---

        self.shuffle = shuffle
        self.seed = seed
        self.generator = torch.Generator().manual_seed(self.seed)

        # The sampler requires access to the full dataset's index structure.
        if not hasattr(self.full_dataset, 'expert_reader') or not hasattr(self.full_dataset.expert_reader, '_cumulative_chunks'):
            raise ValueError("EpisodeAwareSampler requires the underlying dataset to expose "
                             "`expert_reader._cumulative_chunks` attribute.")
        
        self.cumulative_chunks = self.full_dataset.expert_reader._cumulative_chunks
        self.num_episodes = len(self.cumulative_chunks)

        # --- SOTA PATCH: PRE-COMPUTE EPISODE-TO-INDEX MAPPING ---
        # This is the key to making Subset handling efficient.
        # We create a map: {ep_idx: [list of sample indices in this episode]}
        self.episode_to_indices_map: Dict[int, List[int]] = {i: [] for i in range(self.num_episodes)}
        
        # Determine which indices belong to the sampler (all if not a subset)
        indices_to_process = self.subset_indices if self.is_subset else range(len(self.full_dataset))
        
        log.info("Building episode-to-index map for sampler...")
        for sample_idx in indices_to_process:
            # Find which episode this sample_idx belongs to
            ep_idx = np.searchsorted(self.cumulative_chunks, sample_idx, side='right')
            self.episode_to_indices_map[ep_idx].append(sample_idx)
        log.info("Map building complete.")
        # --- END OF SOTA PATCH ---

    def __iter__(self) -> Iterator[int]:
        # 1. Create a list of episode indices that have samples in them.
        episode_order = [ep_idx for ep_idx, indices in self.episode_to_indices_map.items() if indices]
        
        if self.shuffle:
            # Shuffle the order of episodes to be processed.
            random.Random(self.seed).shuffle(episode_order)
        
        # 2. Iterate through the shuffled episodes.
        for ep_idx in episode_order:
            # 3. Get the list of all valid sample indices for this episode.
            episode_indices = self.episode_to_indices_map[ep_idx]
            
            # 4. (Optional) Shuffle the samples *within* the episode.
            if self.shuffle:
                random.Random(self.seed + ep_idx).shuffle(episode_indices)
            
            # 5. Yield all indices from this episode.
            yield from episode_indices

    def __len__(self) -> int:
        return self.num_samples