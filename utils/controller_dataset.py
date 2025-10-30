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
                                     visual subgoal image. For consistency, this should
                                     typically match the Planner's training horizon.
        """
        super().__init__()
        log.info(f"Initializing HierarchicalControllerDataset with H_o={observation_horizon}, H_a={action_horizon}, k={subgoal_horizon_k}")

        # --- SOTA Principle: Composition over Re-implementation ---
        # We instantiate our powerful reader to handle all low-level data access.
        # This dataset becomes a lightweight orchestrator.
        self.expert_reader = ExpertTrajectoryDataset(
            demo_path=dataset_path,
            observation_horizon=observation_horizon,
            action_horizon=action_horizon
        )

        self.subgoal_horizon_k = subgoal_horizon_k
        self.total_chunks = len(self.expert_reader)

        # Basic validation
        if self.subgoal_horizon_k < 1:
            raise ValueError("subgoal_horizon_k must be a positive integer.")
        
        # Verify that the reader has enough runway to sample subgoals
        # We check the first episode as a proxy.
        if len(self.expert_reader.episode_metadata) > 0:
            first_ep_len = self.expert_reader.episode_metadata[0]['length']
            max_horizon = max(observation_horizon - 1, action_horizon, subgoal_horizon_k)
            if first_ep_len <= max_horizon:
                log.warning(
                    f"Episodes may be too short for the specified horizons. "
                    f"Example episode length: {first_ep_len}, Max required lookahead: {max_horizon}. "
                    f"This may result in a smaller-than-expected dataset size."
                )
        log.info(f"Successfully initialized. Found {self.total_chunks} valid controller samples.")

    def __len__(self) -> int:
        """Returns the total number of valid data chunks in the dataset."""
        return self.total_chunks

    def __getitem__(self, idx: int) -> Tuple[Tuple[Dict[str, np.ndarray], np.ndarray], np.ndarray]:
        """
        Retrieves a complete training sample for the Controller.

        Returns:
            A tuple containing:
            - A tuple of inputs: (observation_chunk, subgoal_image)
            - The target: action_chunk
        """
        if not (0 <= idx < self.total_chunks):
            raise IndexError(f"Index {idx} out of range for dataset with {self.total_chunks} chunks.")

        try:
            # --- 1. Delegate Primary Data Loading ---
            # This single call efficiently loads the observation history and action
            # trajectory using all the SOTA features of the underlying reader.
            obs_chunk, action_chunk = self.expert_reader[idx]

            # --- 2. Determine Subgoal Location ---
            # We reuse the reader's internal virtual index to find the episode and timestep
            # corresponding to this flat index `idx`. This is an O(log N) operation.
            ep_idx = np.searchsorted(self.expert_reader._cumulative_chunks, idx, side='right')
            ep_start_chunk_idx = self.expert_reader._cumulative_chunks[ep_idx - 1] if ep_idx > 0 else 0
            local_chunk_idx = idx - ep_start_chunk_idx
            # This is the 't' that defines the start of the action trajectory
            timestep_t = (self.expert_reader.observation_horizon - 1) + local_chunk_idx
            
            # The subgoal is at a future timestep t+k
            subgoal_t = timestep_t + self.subgoal_horizon_k
            ep_meta = self.expert_reader.episode_metadata[ep_idx]
            
            # Sanity check to ensure subgoal_t is within the episode bounds
            if subgoal_t >= ep_meta["length"]:
                # This should theoretically not happen if the reader's total_chunks is calculated correctly,
                # but it's good defensive programming.
                raise IndexError(
                    f"Calculated subgoal timestep {subgoal_t} is out of bounds for "
                    f"episode {ep_idx} with length {ep_meta['length']}."
                )

            # --- 3. Load the Subgoal Image via the Reader's Cache ---
            # We get the full 'image_primary' array for the episode. This call is
            # extremely fast on subsequent accesses due to the reader's LRU cache.
            img_primary_meta = ep_meta["modalities"]["image_primary"]
            full_image_array = self.expert_reader._get_full_modality_array(
                img_primary_meta["key"],
                img_primary_meta["compression"],
                img_primary_meta["dtype"],
                tuple(img_primary_meta["shape"])
            )
            
            # Slice the single subgoal image from the full array. This is a near-zero-cost view.
            subgoal_image = full_image_array[subgoal_t]

            # --- 4. Assemble and Return the Final Sample ---
            return (obs_chunk, subgoal_image), action_chunk

        except Exception as e:
            log.error(f"Error loading data for index {idx}. This may indicate a corrupt dataset or a bug. Error: {e}", exc_info=True)
            # To prevent training crashes, we could return a dummy sample,
            # but raising the error is better for debugging.
            raise


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