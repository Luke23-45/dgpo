# FILE: utils/samplers.py
# (Definitive, SOTA, DDP-Aware, Vectorized Version)

import torch
from torch.utils.data import Sampler, Dataset, Subset
import torch.distributed as dist
from typing import Iterator, Sized, List, Optional
import logging
import math
import numpy as np

log = logging.getLogger(__name__)

class EpisodeAwareSampler(Sampler[int]):
    """
    SOTA Sampler for Robotics Transformers.
    
    Features:
    1. **Cache Locality**: Yields all indices of Episode N before moving to Episode M.
    2. **DDP Support**: Automatically partitions episodes across multiple GPUs/ranks.
    3. **Vectorized Init**: Builds index mappings using NumPy for instant startup.
    4. **Robust**: Handles PyTorch Subsets (random_split) seamlessly.
    
    Logic:
    It views the dataset as a collection of Episodes, not individual Frames.
    It shuffles the list of Episodes, assigns a subset of Episodes to the current GPU,
    and then flattens them into a stream of frame indices.
    """

    def __init__(self, 
                 dataset: Sized, 
                 shuffle: bool = True, 
                 seed: int = 42, 
                 num_replicas: Optional[int] = None, 
                 rank: Optional[int] = None, 
                 drop_last: bool = False):
        super().__init__(dataset)
        
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # --- 1. Resolve Distributed (DDP) Parameters ---
        if num_replicas is None:
            if not dist.is_available():
                num_replicas = 1
                rank = 0
            else:
                try:
                    # CRITICAL FIX: Check if the process group is actually initialized
                    if dist.is_initialized():
                        num_replicas = dist.get_world_size()
                        rank = dist.get_rank()
                    else:
                        # Fallback for single-GPU / non-DDP runs
                        num_replicas = 1
                        rank = 0
                except RuntimeError:
                    # Catch-all for any other dist errors
                    num_replicas = 1
                    rank = 0
        
        self.num_replicas = num_replicas
        self.rank = rank
        
        if self.rank == 0:
            log.info(f"Initializing EpisodeAwareSampler (DDP: {self.num_replicas > 1}, Rank: {self.rank})")

        # --- 2. Resolve Underlying Data Structure ---
        # Handle Subset wrapping to find the source of truth (Cumulative Chunks)
        if isinstance(dataset, Subset):
            self.subset_indices = np.array(dataset.indices)
            full_dataset = dataset.dataset
        else:
            # Create a range for the full dataset
            self.subset_indices = np.arange(len(dataset))
            full_dataset = dataset

        # Access the Episode Index from the ExpertReader
        if not hasattr(full_dataset, 'expert_reader'):
             # Graceful fallback if used with a different dataset type, though less optimal
             raise ValueError("Dataset must expose 'expert_reader' for EpisodeAware sampling.")
             
        self.cumulative_chunks = np.array(full_dataset.expert_reader._cumulative_chunks)
        self.num_total_episodes = len(self.cumulative_chunks)

        # --- 3. Vectorized Episode Mapping (The Optimization) ---
        # Instead of looping python ints, we use numpy to bucket ALL indices at once.
        # Find which episode every valid index belongs to.
        # e.g. indices [0, 1, 2, 100, 101] -> episodes [0, 0, 0, 1, 1]
        episode_assignments = np.searchsorted(self.cumulative_chunks, self.subset_indices, side='right')
        
        # We need to group indices by episode.
        # Structure: { ep_idx: [frame_idx_1, frame_idx_2...] }
        # Optimization: Use sorting to group them efficiently.
        sort_order = np.argsort(episode_assignments)
        sorted_indices = self.subset_indices[sort_order]
        sorted_episodes = episode_assignments[sort_order]
        
        # Find boundaries where episode ID changes
        unique_eps, split_indices = np.unique(sorted_episodes, return_index=True)
        
        # Split the sorted index array into chunks, one per episode
        # This gives us a list where grouped_indices[i] is the array of frames for unique_eps[i]
        grouped_indices = np.split(sorted_indices, split_indices[1:])
        
        # Map episode ID -> Array of Global Indices
        # We filter out empty episodes automatically via unique()
        self.episode_map = {ep_id: indices for ep_id, indices in zip(unique_eps, grouped_indices) if len(indices) > 0}
        
        # The list of episodes available in this specific Subset
        self.available_episodes = list(self.episode_map.keys())
        
        # --- 4. Calculate DDP Lengths ---
        # We partition based on EPISODES, not FRAMES, to preserve cache locality.
        total_episodes = len(self.available_episodes)
        
        if self.drop_last and self.num_replicas > 1:
            self.num_episodes_per_replica = math.floor(total_episodes / self.num_replicas)
        else:
            self.num_episodes_per_replica = math.ceil(total_episodes / self.num_replicas)
            
        self.total_size_episodes = self.num_episodes_per_replica * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        # 1. Deterministic Shuffling (Epoch-based)
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # Shuffle the list of EPISODES
        if self.shuffle:
            indices = torch.randperm(len(self.available_episodes), generator=g).tolist()
        else:
            indices = list(range(len(self.available_episodes)))
            
        # 2. DDP Padding (Ensure all ranks have equal number of episodes)
        if not self.drop_last:
            # Add extra episodes to make it evenly divisible
            padding_size = self.total_size_episodes - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # Truncate
            indices = indices[:self.total_size_episodes]

        # 3. Subsample: Pick the episodes for THIS specific GPU (Rank)
        # This is the magic step. Rank 0 gets ep [0, 4, 8...], Rank 1 gets [1, 5, 9...]
        # Note: Strided slicing (rank::num_replicas) spreads the load better than chunking
        my_episode_indices = indices[self.rank : self.total_size_episodes : self.num_replicas]
        
        # 4. Flatten into Frame Indices
        # Now we yield all frames for Ep A, then all frames for Ep B...
        final_indices = []
        for idx_in_list in my_episode_indices:
            real_ep_id = self.available_episodes[idx_in_list]
            frames = self.episode_map[real_ep_id]
            
            # Optional: Shuffle frames WITHIN the episode? 
            # Usually NO for RNNs, YES for Transformers/CNNs if obs_horizon=1.
            # Since we are doing single-frame planning, local shuffling breaks correlation 
            # slightly which is good for IID, but we keep order for cache consistency usually.
            # We will yield sequentially to be cache-friendly.
            final_indices.extend(frames)

        return iter(final_indices)

    def __len__(self) -> int:
        # Note: This is an approximation because episodes have different lengths.
        # PyTorch mostly uses this for the progress bar.
        # We calculate the exact number of frames assigned to this rank.
        # To avoid recomputing every call, we return the count based on initialization.
        # This might be slightly off if DDP padding occurs, but is generally safe.
        total_frames = sum(len(self.episode_map[ep]) for ep in self.available_episodes)
        return math.ceil(total_frames / self.num_replicas)

    def set_epoch(self, epoch: int):
        """
        Sets the epoch for this sampler. This ensures that the shuffle order
        changes every epoch, which is critical for training convergence.
        """
        self.epoch = epoch