# FILE: utils/ego_planner_dataset.py
# State-of-the-Art, High-Performance Dataset for the Ego-Planner Model (Fully Patched)

"""
This module provides the definitive dataset and dataloader pipeline for training
the unified Ego-Planner model. This version has been fully patched and audited
to align with the high-performance ExpertTrajectoryDataset format.

Architectural Philosophy (The "Orchestrator" Pattern):
This dataset acts as a smart "Orchestrator" on top of the highly-optimized
`ExpertTrajectoryDataset`. It delegates all low-level data access (LMDB reads,
decompression, caching) to the underlying reader, while its sole responsibility
is to assemble the specific, multi-part data samples required by the Ego-Planner.

Key SOTA Features & Patches in this Definitive Version:
-   **Explicit Sample Index Map**: Constructs a definitive map of all valid
    (episode, timestep) samples at initialization, ensuring robust and unambiguous
    indexing, eliminating fragile reverse-lookups.
-   **Correct API Contract**: The `_get_image_primary_at` helper has been patched to
    correctly call the private API of the underlying `ExpertTrajectoryDataset`,
    preventing data loading crashes.
-   **Resilient Collate Function**: The collate function is replaced with a
    battle-tested version that correctly handles and filters out corrupted
    samples within a batch, preventing training interruptions.
-   **Preserved SOTA Augmentation**: Retains the excellent, temporally-consistent
    data augmentation logic that applies the same random transform to all images
    in an observation history.
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

# SOTA: We import the powerful base dataset we are building upon.
from utils.expert_dataset import ExpertTrajectoryDataset

# Setup a logger for the module
log = logging.getLogger(__name__)


# ==============================================================================
# SECTION 1: THE DEFINITIVE EGO-PLANNER DATASET
# ==============================================================================

class EgoPlannerDataset(Dataset):
    """
    The definitive, high-performance dataset for the Ego-Planner. It acts as a
    smart orchestrator on top of the ExpertTrajectoryDataset, providing fully
    preprocessed, correctly augmented, and model-ready data samples.
    """
    def __init__(self,
                 dataset_path: str,
                 obs_horizon: int,
                 action_horizon: int,
                 use_aug: bool = False):
        super().__init__()
        log.info(f"Initializing EgoPlannerDataset (Definitive Patched Version, use_aug={use_aug})")

        # 1. Composition: We use the ExpertTrajectoryDataset as our core data engine.
        self.expert_reader = ExpertTrajectoryDataset(
            demo_path=dataset_path,
            observation_horizon=obs_horizon,
            action_horizon=action_horizon
        )
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.use_aug = use_aug

        # 2. Define image transformation pipelines.
        self.transform_primary = transforms.Compose([transforms.Resize((224, 224), antialias=True), transforms.ToTensor()])
        self.transform_wrist = transforms.Compose([transforms.Resize((128, 128), antialias=True), transforms.ToTensor()])
        
        # SOTA augmentation modules using functional transforms.
        self.aug_color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        self.aug_random_apply_p = 0.8
        self.aug_random_grayscale_p = 0.1

        # --- [DEFINITIVE PATCH 1: BUILD AN EXPLICIT SAMPLE INDEX MAP] ---
        # This creates a robust, unambiguous mapping from a flat index to a
        # specific (episode, timestep) coordinate.
        self.samples: List[Tuple[int, int]] = []
        log.info("Building explicit sample index map for robust indexing...")
        for ep_idx in range(self.expert_reader.get_num_episodes()):
            ep_len = self.expert_reader.get_episode_length(ep_idx)
            # A valid sample can start at timestep `t` if there are `obs_horizon`
            # frames before it (inclusive) and `action_horizon` actions after it.
            start_t = self.obs_horizon - 1
            end_t = ep_len - self.action_horizon
            for t in range(start_t, end_t + 1):
                self.samples.append((ep_idx, t))
        log.info(f"Successfully initialized. Found {len(self.samples)} valid samples across {self.expert_reader.get_num_episodes()} episodes.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        [DEFINITIVE, FULLY PATCHED VERSION]
        Orchestrates the retrieval of a complete, processed sample for the Ego-Planner.
        """
        if not (0 <= idx < len(self)):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self)} samples.")

        try:
            # --- 1. Get Data from Underlying Reader ---
            # The global index `idx` directly maps to a valid chunk.
            obs_history_chunk_np, action_chunk_np = self.expert_reader[idx]

            # Get the episode index for this sample.
            ep_idx, _ = self.samples[idx]

            # Get the strategic images: the very first and very last frames of the episode.
            initial_image_np = self._get_image_primary_at(ep_idx, 0)
            goal_image_np = self.expert_reader.get_goal_image(ep_idx)

            # --- 2. Preprocessing & SOTA Correlated Augmentation ---
            initial_image = self.transform_primary(Image.fromarray(initial_image_np))
            goal_image = self.transform_primary(Image.fromarray(goal_image_np))

            observation_history = {}
            for key, val in obs_history_chunk_np.items():
                if 'image' in key:
                    img_stack_pil = [Image.fromarray(img) for img in val]
                    
                    if self.use_aug:
                        # Decide ONCE if jitter will be applied to this sample's history.
                        if random.random() < self.aug_random_apply_p:
                            # Sample ColorJitter parameters ONCE.
                            jitter_params = self.aug_color_jitter.get_params(
                                self.aug_color_jitter.brightness, self.aug_color_jitter.contrast,
                                self.aug_color_jitter.saturation, self.aug_color_jitter.hue
                            )
                            # Apply the SAME sampled parameters to all images in the stack.
                            img_stack_pil = [TF.functional_pil_color_jitter(img, *jitter_params) for img in img_stack_pil]

                        # Decide ONCE if grayscale will be applied.
                        if random.random() < self.aug_random_grayscale_p:
                            img_stack_pil = [TF.to_grayscale(img, num_output_channels=3) for img in img_stack_pil]

                    # Apply the non-random base transforms (Resize -> ToTensor).
                    transform_fn = self.transform_wrist if 'wrist' in key else self.transform_primary
                    observation_history[key] = torch.stack([transform_fn(img) for img in img_stack_pil])
                else:
                    observation_history[key] = torch.from_numpy(val.copy()).float()
            
            action_chunk = torch.from_numpy(action_chunk_np.copy()).float()
            
            return {
                'initial_image': initial_image,
                'goal_image': goal_image,
                'observation_history': observation_history,
                'action_chunk': action_chunk,
            }

        except Exception as e:
            log.error(f"Error loading data for sample index {idx}. Error: {e}", exc_info=False)
            return None # Returning None allows the robust collate_fn to handle this.

    def _get_image_primary_at(self, ep_idx: int, timestep_t: int) -> np.ndarray:
        """
        [DEFINITIVE PATCH 2: CORRECT API USAGE]
        Private helper to get a single primary image frame from an episode.
        This version correctly calls the underlying reader's API.
        """
        ep_meta = self.expert_reader.episode_metadata[ep_idx]
        img_meta = ep_meta["modalities"]["image_primary"]
        
        # This call correctly unpacks the metadata and hits the per-worker LRU cache.
        full_image_array = self.expert_reader._get_full_modality_array(
            key=img_meta["key"],
            compression=img_meta["compression"],
            dtype_str=img_meta["dtype"],
            shape_list=tuple(img_meta["shape"])
        )
        return full_image_array[timestep_t]

    # These delegate calls are useful for external utilities and samplers.
    def get_num_episodes(self) -> int:
        return self.expert_reader.get_num_episodes()
    
    def get_episode_length(self, episode_idx: int) -> int:
        return self.expert_reader.get_episode_length(episode_idx)


# ==============================================================================
# SECTION 2: THE DEFINITIVE ROBUST COLLATE FUNCTION
# ==============================================================================

def ego_planner_collate_fn(batch: List[Optional[Dict[str, any]]]) -> Dict[str, any]:
    """
    [DEFINITIVE PATCH 3: RESILIENT BATCHING]
    A purpose-built, robust collate function for the EgoPlannerDataset.

    Its primary job is to filter out any `None` samples that may have been
    returned by `__getitem__` due to data loading errors. This prevents a single
    bad data point from crashing an entire training batch.
    """
    # 1. Filter out failed samples (None values).
    valid_samples = [s for s in batch if s is not None]

    # If the entire batch failed, return a special dictionary indicating this.
    if not valid_samples:
        print("An entire batch of data loading failed. Skipping batch.")
        return {"batch_failed": True}

    # 2. Use PyTorch's default collate to stack the valid samples.
    # This is a highly optimized function that correctly handles dictionaries and nested tensors.
    return torch.utils.data.dataloader.default_collate(valid_samples)