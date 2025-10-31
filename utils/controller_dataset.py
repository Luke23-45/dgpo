# FILE: utils/controller_dataset.py
# SOTA Hierarchical Controller Dataset for ViDHiS

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple

# We import our state-of-the-art reader, which will do all the heavy lifting.
from utils.expert_dataset import ExpertTrajectoryDataset, collate_fn

log = logging.getLogger(__name__)

class HierarchicalControllerDataset(Dataset):
    """
    State-of-the-art Dataset for training the ViDHiS Controller.

    This class acts as a smart wrapper around the highly optimized `ExpertTrajectoryDataset`.
    It leverages the underlying reader's virtual indexing, SoA format, on-the-fly
    decompression, and per-worker LRU caching to achieve maximum data loading performance.

    Its primary role is to orchestrate the loading of three key data components for each sample:
    1. A history of observations (obs_history).
    2. A future trajectory of actions (action_trajectory).
    3. The visual subgoal image corresponding to the end of the action trajectory.
    """

    def __init__(self,
                 dataset_path: str,
                 observation_horizon: int,
                 action_horizon: int,
                 subgoal_horizon_k: int):
        """
        Args:
            dataset_path (str): Path to the enhanced expert LMDB dataset file.
            observation_horizon (int): Number of observation steps to stack (H_o).
            action_horizon (int): Number of action steps to predict (H_a).
            subgoal_horizon_k (int): The future timestep `t+k` from which to draw the
                                     visual subgoal image.
        """
        super().__init__()
        log.info(f"Initializing HierarchicalControllerDataset with H_o={observation_horizon}, H_a={action_horizon}, k={subgoal_horizon_k}")

        self.expert_reader = ExpertTrajectoryDataset(
            demo_path=dataset_path,
            observation_horizon=observation_horizon,
            action_horizon=action_horizon
        )

        self.subgoal_horizon_k = subgoal_horizon_k
        self.observation_horizon = observation_horizon
        self.action_horizon = action_horizon
        
        # --- NEW: Build a robust virtual index that respects ALL horizons ---
        
        # The maximum lookahead required is the largest of the action horizon or subgoal horizon.
        # We also need to account for the observation history.
        max_future_horizon = max(self.action_horizon, self.subgoal_horizon_k)
        
        self.episode_chunks = []
        for ep_meta in self.expert_reader.episode_metadata:
            # The number of valid starting points `t` for an action sequence in an
            # episode of length L is determined by the need to have a full
            # observation history AND a full future action/subgoal trajectory.
            num_valid_chunks = ep_meta['length'] - (self.observation_horizon - 1) - max_future_horizon
            
            if num_valid_chunks > 0:
                self.episode_chunks.append(num_valid_chunks)
            else:
                self.episode_chunks.append(0)
                log.warning(
                    f"Episode {ep_meta['episode_id']} has length {ep_meta['length']} but requires "
                    f"at least {(self.observation_horizon - 1) + max_future_horizon + 1} steps. "
                    "This episode will be skipped."
                )

        self.cumulative_chunks = np.cumsum(self.episode_chunks)
        self.total_chunks = self.cumulative_chunks[-1] if len(self.cumulative_chunks) > 0 else 0
        
        if self.subgoal_horizon_k != self.action_horizon:
            log.warning(f"Configuration mismatch: subgoal_horizon_k ({self.subgoal_horizon_k}) does not equal action_horizon ({self.action_horizon}). Ensure this is intentional.")

        log.info(f"Successfully initialized. Found {self.total_chunks} valid controller samples across {len(self.episode_chunks)} episodes.")

    def __len__(self) -> int:
        """Returns the total number of valid data chunks in the dataset."""
        return self.total_chunks


    def __getitem__(self, idx: int) -> Tuple[Tuple[Dict[str, np.ndarray], np.ndarray], np.ndarray]:
        """
        Retrieves a complete training sample for the Controller. This version uses
        the robust internal index map to guarantee all lookaheads are valid.
        """
        if not (0 <= idx < self.total_chunks):
            raise IndexError(f"Index {idx} out of range for dataset with {self.total_chunks} valid chunks.")

        try:
            # 1. Find the correct episode and local index using our robust index.
            # This is a fast O(log N) binary search.
            ep_idx = np.searchsorted(self.cumulative_chunks, idx, side='right')
            ep_start_chunk_idx = self.cumulative_chunks[ep_idx - 1] if ep_idx > 0 else 0
            local_chunk_idx = idx - ep_start_chunk_idx

            # 2. Calculate the starting timestep 't' of the action trajectory.
            timestep_t = (self.observation_horizon - 1) + local_chunk_idx

            # 3. Use the underlying expert_reader to load the obs/action chunks.
            # We need to translate our robust local_chunk_idx to the reader's global index.
            reader_global_start_idx = self.expert_reader._cumulative_chunks[ep_idx - 1] if ep_idx > 0 else 0
            reader_idx = reader_global_start_idx + local_chunk_idx
            obs_chunk, action_chunk = self.expert_reader[reader_idx]

            # 4. Calculate the subgoal timestep and load the subgoal image.
            # This is now guaranteed to be within the episode bounds.
            subgoal_t = timestep_t + self.subgoal_horizon_k
            
            ep_meta = self.expert_reader.episode_metadata[ep_idx]
            img_primary_meta = ep_meta["modalities"]["image_primary"]
            full_image_array = self.expert_reader._get_full_modality_array(
                img_primary_meta["key"],
                img_primary_meta["compression"],
                img_primary_meta["dtype"],
                tuple(img_primary_meta["shape"])
            )
            subgoal_image = full_image_array[subgoal_t]

            # 5. Assemble and return the final sample.
            return (obs_chunk, subgoal_image), action_chunk

        except Exception as e:
            log.error(f"Error loading data for index {idx}. Error: {e}", exc_info=True)
            raise


def hierarchical_collate_fn(batch):
    """
    Custom collate_fn for the HierarchicalControllerDataset.
    It correctly handles the nested structure and converts data to PyTorch Tensors.
    """
    # Deconstruct the batch of samples
    obs_chunks = [item[0][0] for item in batch]
    subgoal_images = [item[0][1] for item in batch]
    action_chunks = [item[1] for item in batch]

    # Batch the obs_chunks
    batched_obs_chunk = {}
    obs_keys = obs_chunks[0].keys()
    for key in obs_keys:
        # Stack numpy arrays for each observation modality
        numpy_array = np.stack([obs[key] for obs in obs_chunks])
        # CONVERT TO TENSOR
        batched_obs_chunk[key] = torch.from_numpy(numpy_array)

    # Batch the other components
    numpy_subgoals = np.stack(subgoal_images)
    numpy_actions = np.stack(action_chunks)
    
    # CONVERT TO TENSOR
    batched_subgoal_image = torch.from_numpy(numpy_subgoals)
    batched_action_chunk = torch.from_numpy(numpy_actions)

    # Reconstruct the final batched sample in the expected nested format
    return (batched_obs_chunk, batched_subgoal_image), batched_action_chunk


# --- Example Usage and Validation Block ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s - %(message)s')
    
    # !!! IMPORTANT !!!
    # Replace this path with the actual path to your ENHANCED expert LMDB dataset.
    DUMMY_DATASET_PATH = "/path/to/your/expert_YYYYMMDD_..._samples.lmdb"

    if not Path(DUMMY_DATASET_PATH).exists():
        print("\n" + "="*80)
        print(f"!!! VALIDATION FAILED: Dataset not found at '{DUMMY_DATASET_PATH}'")
        print("Please update the `DUMMY_DATASET_PATH` variable in `utils/controller_dataset.py` to point to your")
        print("actual enhanced expert dataset file (.lmdb) before running this script for validation.")
        print("="*80 + "\n")
    else:
        log.info("--- Running Validation for HierarchicalControllerDataset ---")
        
        # Configuration matching a typical setup
        H_OBS = 2
        H_ACT = 8
        SUBGOAL_K = 8

        dataset = HierarchicalControllerDataset(
            dataset_path=DUMMY_DATASET_PATH,
            observation_horizon=H_OBS,
            action_horizon=H_ACT,
            subgoal_horizon_k=SUBGOAL_K
        )

        log.info(f"Dataset length: {len(dataset)}")
        
        if len(dataset) > 0:
            # Get a random sample to inspect
            sample_idx = np.random.randint(0, len(dataset))
            log.info(f"Fetching sample at index: {sample_idx}")
            
            (obs_chunk, subgoal_image), action_chunk = dataset[sample_idx]
            
            log.info("--- Sample Structure and Shapes ---")
            log.info("Observation Chunk Keys: " + str(obs_chunk.keys()))
            for key, value in obs_chunk.items():
                log.info(f"  - obs_chunk['{key}'].shape: {value.shape} | dtype: {value.dtype}")
            log.info(f"Subgoal Image Shape: {subgoal_image.shape} | dtype: {subgoal_image.dtype}")
            log.info(f"Action Chunk Shape: {action_chunk.shape} | dtype: {action_chunk.dtype}")
            
            # Verify horizons
            assert obs_chunk['image_primary'].shape[0] == H_OBS
            assert obs_chunk['proprio'].shape[0] == H_OBS
            assert action_chunk.shape[0] == H_ACT
            log.info("Horizons verified. [✔]")

            # --- Test with DataLoader and Collate Function ---
            log.info("\n--- Testing DataLoader Integration ---")
            # Use num_workers=0 for simple debugging, set > 0 to test multiprocessing
            dataloader = DataLoader(
                dataset,
                batch_size=4,
                shuffle=True,
                num_workers=0,
                collate_fn=collate_fn # Use the same collate_fn as the base dataset
            )
            
            (batch_obs_chunk, batch_subgoal_image), batch_action_chunk = next(iter(dataloader))
            
            log.info("--- Batch Structure and Shapes ---")
            log.info("Batch Observation Chunk Keys: " + str(batch_obs_chunk.keys()))
            for key, value in batch_obs_chunk.items():
                log.info(f"  - batch_obs_chunk['{key}'].shape: {value.shape} | dtype: {value.dtype}")
            log.info(f"Batch Subgoal Image Shape: {batch_subgoal_image.shape} | dtype: {batch_subgoal_image.dtype}")
            log.info(f"Batch Action Chunk Shape: {batch_action_chunk.shape} | dtype: {batch_action_chunk.dtype}")
            
            # Verify batch dimensions
            assert batch_obs_chunk['image_primary'].shape[1] == H_OBS
            assert batch_action_chunk.shape[1] == H_ACT
            log.info("Batching and collation verified. [✔]")
            log.info("\nValidation successful!")