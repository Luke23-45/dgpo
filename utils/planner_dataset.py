# FILE: utils/planner_dataset.py
# SOTA Hierarchical Planner Dataset (Corrected, Final Version)

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
from torchvision import transforms
import random
from PIL import Image
import time
import os
# --- SOTA Imports ---
# We import our SOTA reader, which will handle all the complex data loading.
from utils.expert_dataset import ExpertTrajectoryDataset

log = logging.getLogger(__name__)

class HierarchicalPlannerDataset(Dataset):
    """
    State-of-the-art Dataset for training the ViDHiS Visual Planner.

    This class is a lightweight, intelligent wrapper around the ExpertTrajectoryDataset.
    It leverages the underlying reader's SOTA features (virtual indexing, SoA format,
    caching) to efficiently sample the specific data required for planner training:
    (current_image, goal_image, progress) -> ground_truth_subgoal_image.
    """

    def __init__(self,
                 dataset_path: str,
                 subgoal_horizon_k: int,
                 image_size: Tuple[int, int] = (224, 224),
                 img_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 img_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
                 use_random_aug: bool = False
                 ):
        super().__init__()
        
        # --- SOTA PATCH 1: COMPOSITION ---
        # Instantiate the expert reader. This is the workhorse.
        # We use minimal horizons because we will access data directly by index, not by chunking.
        log.info("Initializing underlying ExpertTrajectoryDataset reader...")
        self.expert_reader = ExpertTrajectoryDataset(
            demo_path=dataset_path,
            observation_horizon=1,
            action_horizon=1
        )
        
        self.subgoal_horizon_k = subgoal_horizon_k
        self.image_size = image_size
        
        log.info(f"Loaded index for {len(self.expert_reader.episode_metadata)} episodes.")

        # --- SOTA PATCH 2: ROBUST SAMPLE PRE-COMPUTATION ---
        # Pre-compute a list of all valid (episode_idx, timestep_t) tuples.
        self.valid_samples = self._precompute_valid_samples()
        log.info(f"Precomputed {len(self.valid_samples)} valid planner samples (k={subgoal_horizon_k}).")

        # --- Image Transformations (Correctly Implemented) ---
        transform_steps = [transforms.ToPILImage()]

        # Add augmentation transforms if enabled
        if use_random_aug:
            augment_transforms = [
                transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.1),
                # SOTA Addition: Add random horizontal flip for more robustness
                transforms.RandomHorizontalFlip(p=0.5),
            ]
            transform_steps.extend(augment_transforms)

        # Add base transforms (resize, and convert back to tensor) at the end
        base_transforms = [
            transforms.Resize(image_size, antialias=True),
            transforms.ToTensor(), # PIL Image -> CHW:float[0,1]
        ]
        transform_steps.extend(base_transforms)

        # The final composed transform pipeline
        self.transform = transforms.Compose(transform_steps)

        # Normalization is applied separately AFTER the main transform
        self.normalize_transform = transforms.Normalize(mean=img_mean, std=img_std)

    def _precompute_valid_samples(self) -> List[Tuple[int, int]]:
        """ Creates a list of (episode_idx, timestep_t) valid for sampling. """
        valid_samples = []
        for ep_idx, ep_info in enumerate(self.expert_reader.episode_metadata):
            ep_len = ep_info['length']
            # A valid timestep 't' must allow for a future subgoal at 't+k'.
            for t in range(ep_len - self.subgoal_horizon_k):
                valid_samples.append((ep_idx, t))
        return valid_samples

    def __len__(self) -> int:
        return len(self.valid_samples)

    # --- SOTA PATCH 3: DELEGATED AND EFFICIENT DATA LOADING ---
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        log.debug(f"[Worker {os.getpid()}] Starting __getitem__ for index {idx}")
        """
        Retrieves a complete training sample by orchestrating calls to the expert reader.
        """
        if not (0 <= idx < len(self.valid_samples)):
            raise IndexError("Index out of bounds")

        ep_idx, timestep_t = self.valid_samples[idx]
        ep_info = self.expert_reader.episode_metadata[ep_idx]
        ep_len = ep_info['length']

        try:
            # --- START OF SOTA DIAGNOSTIC PATCH (continued) ---
            log.debug(f"  - Index {idx} maps to (ep={ep_idx}, time={timestep_t})")
            log.debug("  - Loading current_image_np...")
            # --- END OF SOTA DIAGNOSTIC PATCH (continued) ---
            current_image_np = self._get_image_primary_at(ep_idx, timestep_t)

            # --- START OF SOTA DIAGNOSTIC PATCH (continued) ---
            log.debug("  - Loading final_goal_image_np...")
            # --- END OF SOTA DIAGNOSTIC PATCH (continued) ---
            final_goal_image_np = self.expert_reader.get_goal_image(ep_idx)

            # --- START OF SOTA DIAGNOSTIC PATCH (continued) ---
            log.debug("  - Loading gt_subgoal_image_np...")
            # --- END OF SOTA DIAGNOSTIC PATCH (continued) ---
            subgoal_t = timestep_t + self.subgoal_horizon_k
            gt_subgoal_image_np = self._get_image_primary_at(ep_idx, subgoal_t)

            # --- START OF SOTA DIAGNOSTIC PATCH (continued) ---
            log.debug("  - All data loaded from disk. Starting transforms...")
            # --- END OF SOTA DIAGNOSTIC PATCH (continued) ---

        except Exception as e:
            log.error(f"Error loading data for index {idx} (ep {ep_idx}, t {timestep_t}): {e}", exc_info=True)
            dummy_img = torch.zeros((3, *self.image_size), dtype=torch.float32)
            return {
                'current_image': dummy_img.clone(),
                'goal_image': dummy_img.clone(),
                'progress': torch.tensor(0.0, dtype=torch.float32),
                'gt_subgoal_image': dummy_img.clone(),
            }

        # 4. Calculate progress
        progress = torch.tensor(timestep_t / max(1, ep_len - 1), dtype=torch.float32)

        # 5. Apply transformations
        current_image = self.normalize_transform(self.transform(current_image_np))
        final_goal_image = self.normalize_transform(self.transform(final_goal_image_np))
        gt_subgoal_image = self.normalize_transform(self.transform(gt_subgoal_image_np))
        log.debug(f"  - Transforms complete. Returning sample for index {idx}.")

        return {
            'current_image': current_image,
            'goal_image': final_goal_image,
            'progress': progress,
            'gt_subgoal_image': gt_subgoal_image,
        }

    def _get_image_primary_at(self, ep_idx: int, timestep_t: int) -> np.ndarray:
        """
        Private helper to get a single primary image frame.
        It leverages the expert_reader's caching for efficiency.
        """
        ep_meta = self.expert_reader.episode_metadata[ep_idx]
        img_meta = ep_meta["modalities"]["image_primary"]
        
        # This call to the reader's internal method is key. It will hit the LRU cache.
        full_image_array = self.expert_reader._get_full_modality_array(
            img_meta["key"], img_meta["compression"], img_meta["dtype"], tuple(img_meta["shape"])
        )
        
        # Return the specific frame.
        return full_image_array[timestep_t]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Replace with your actual enhanced dataset path
    DUMMY_DATASET_PATH = "/path/to/your/enhanced_expert_dataset.lmdb"
    if not Path(DUMMY_DATASET_PATH).exists():
         print(f"Please update DUMMY_DATASET_PATH in planner_dataset.py to your actual LMDB file.")
    else:
        dataset = HierarchicalPlannerDataset(
            dataset_path=DUMMY_DATASET_PATH,
            subgoal_horizon_k=8,
            image_size=(128, 128), # Smaller size for faster testing
            use_random_aug=True
        )
        print(f"Dataset length: {len(dataset)}")
        if len(dataset) > 0:
            sample = dataset[random.randint(0, len(dataset) - 1)]
            print("Sample keys:", sample.keys())
            print("Current image shape:", sample['current_image'].shape)
            print("Goal image shape:", sample['goal_image'].shape)
            print("Progress value:", sample['progress'].item())
            print("GT Subgoal image shape:", sample['gt_subgoal_image'].shape)

            # Check normalization
            print("Current image mean:", sample['current_image'].mean(dim=[1,2]))
            print("Current image std:", sample['current_image'].std(dim=[1,2]))


        # Test DataLoader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0) # Set num_workers > 0 for real use
        batch = next(iter(dataloader))
        print("\nBatch keys:", batch.keys())
        print("Batch current image shape:", batch['current_image'].shape)
        print("Batch progress shape:", batch['progress'].shape)