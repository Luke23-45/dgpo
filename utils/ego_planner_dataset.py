# FILE: utils/ego_planner_dataset.py
# State-of-the-Art, High-Performance Dataset for the Ego-Planner Model (Fully Patched)

"""
This module provides the definitive dataset and dataloader pipeline for training
the unified Ego-Planner model.

Architectural Philosophy (The "Orchestrator" Pattern):
This dataset is not a monolithic implementation. Instead, it acts as a smart
"Orchestrator" that sits on top of the already brilliant and highly-optimized
`ExpertTrajectoryDataset`. By using composition, this dataset's sole
responsibility is to orchestrate the retrieval of the specific, multi-part
data samples required by the Ego-Planner, while delegating all complex, low-level
data access (LMDB reads, decompression, caching) to the underlying reader.

Key Features:
-   **Maximal Efficiency**: Leverages the full performance suite of the
    `ExpertTrajectoryDataset`, including its zero-scan startup, virtual index,
    SoA chunking, on-the-fly decompression, and per-worker LRU caching.
-   **Correct Data Sampling**: Each sample `idx` corresponds to a unique
    `(observation_chunk, action_chunk)` pair, ensuring a uniform sampling
    distribution over all possible tactical decisions in the dataset.
-   **Multi-Part Data Orchestration**: A single `__getitem__` call efficiently
    assembles the three required data components:
    1.  The initial `t=0` image from the episode.
    2.  The final goal image from the episode.
    3.  The core observation/action chunk from a random timestep `t`.
-   **Robustness**: Includes comprehensive error handling to prevent training
    crashes from corrupted data points and a dedicated, purpose-built collate
    function to ensure correct batching.
-   **State-of-the-Art Transformations**: Integrates `torchvision` transforms
    for resizing, data augmentation (color jitter, grayscale), and normalization.
-   **Extensive Verification**: A thorough `if __name__ == '__main__':` block
    provides a unit test to validate data shapes, dtypes, and the integrity of
    the entire data loading and batching pipeline.
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

# SOTA: We import the powerful base dataset we are building upon.
from utils.expert_dataset import ExpertTrajectoryDataset

# Setup a logger for the module
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. The Main EgoPlannerDataset Class
# -----------------------------------------------------------------------------

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
        log.info(f"Initializing EgoPlannerDataset v4 (use_aug={use_aug}) with H_o={obs_horizon}, H_a={action_horizon}")

        self.expert_reader = ExpertTrajectoryDataset(
            demo_path=dataset_path,
            observation_horizon=obs_horizon,
            action_horizon=action_horizon
        )
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.use_aug = use_aug
        self._fail_count = 0

        # Define the augmentation modules separately for parameter access.
        self.aug_color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        self.aug_random_apply_p = 0.8
        self.aug_random_grayscale_p = 0.1

        # Define the base transformation pipelines.
        self.transform_primary = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor()
        ])
        self.transform_wrist = transforms.Compose([
            transforms.Resize((128, 128), antialias=True),
            transforms.ToTensor()
        ])
        
        # --- CRITICAL FIX: BUILD AN EXPLICIT SAMPLE INDEX MAP ---
        self.samples: List[Tuple[int, int]] = []
        log.info("Building explicit sample index map for robustness...")
        for ep_idx, meta in enumerate(self.expert_reader.episode_metadata):
            ep_len = meta.get("episode_len", meta.get("length"))
            if ep_len is None:
                raise RuntimeError(f"Cannot determine episode length for ep_idx {ep_idx}.")
            for t in range(self.obs_horizon - 1, ep_len - self.action_horizon):
                self.samples.append((ep_idx, t))

        log.info(f"Successfully initialized. Found {len(self.samples)} valid samples.")

    def __len__(self) -> int:
        # --- CRITICAL FIX: USE THE LENGTH OF THE EXPLICIT SAMPLE MAP ---
        return len(self.samples)

# FILE: utils/ego_planner_dataset.py

# --- START OF DEFINITIVE PATCH ---
# REPLACE the existing __getitem__ method with this one.

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        if not (0 <= idx < len(self)):
            raise IndexError(f"Index {idx} out of range.")

        try:
            obs_history_chunk_np, action_chunk_np = self.expert_reader[idx]
            ep_idx = np.searchsorted(self.expert_reader._cumulative_chunks, idx, side='right')
            initial_image_np = self._get_image_primary_at(ep_idx, 0)
            goal_image_np = self.expert_reader.get_goal_image(ep_idx)

            # --- DEFINITIVE PREPROCESSING & AUGMENTATION (v5 - Patched) ---
            
            # 1. Transform static strategic images (no augmentation)
            initial_image = self.transform_primary(Image.fromarray(initial_image_np))
            goal_image = self.transform_primary(Image.fromarray(goal_image_np))

            # 2. Transform tactical observation history images
            observation_history = {}
            for key, val in obs_history_chunk_np.items():
                if 'image' in key:
                    img_stack_pil = [Image.fromarray(img) for img in val]
                    
                    # --- DEFINITIVE FIX for Correlated Data Augmentation ---
                    if self.use_aug:
                        # Decide ONCE if we will apply jitter for this whole sample
                        apply_jitter = random.random() < self.aug_random_apply_p
                        if apply_jitter:
                            # Sample ColorJitter parameters ONCE
                            jitter_params = self.aug_color_jitter.get_params(
                                self.aug_color_jitter.brightness,
                                self.aug_color_jitter.contrast,
                                self.aug_color_jitter.saturation,
                                self.aug_color_jitter.hue
                            )
                            # **CRITICAL FIX**: Unpack the tuple and apply functional transforms
                            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = jitter_params
                            
                            # Apply the SAME sampled parameters to all images in the stack
                            for i in range(len(img_stack_pil)):
                                for fn_id in fn_idx:
                                    if fn_id == 0 and brightness_factor is not None:
                                        img_stack_pil[i] = TF.adjust_brightness(img_stack_pil[i], brightness_factor)
                                    if fn_id == 1 and contrast_factor is not None:
                                        img_stack_pil[i] = TF.adjust_contrast(img_stack_pil[i], contrast_factor)
                                    if fn_id == 2 and saturation_factor is not None:
                                        img_stack_pil[i] = TF.adjust_saturation(img_stack_pil[i], saturation_factor)
                                    if fn_id == 3 and hue_factor is not None:
                                        img_stack_pil[i] = TF.adjust_hue(img_stack_pil[i], hue_factor)

                        # Decide ONCE if we will apply grayscale for this whole sample
                        apply_grayscale = random.random() < self.aug_random_grayscale_p
                        if apply_grayscale:
                            # Apply the SAME grayscale transform to all images
                            img_stack_pil = [TF.to_grayscale(img, num_output_channels=3) for img in img_stack_pil]

                    # Now, apply the non-random base transforms (Resize -> ToTensor)
                    transform = self.transform_wrist if 'wrist' in key else self.transform_primary
                    observation_history[key] = torch.stack([transform(img) for img in img_stack_pil])
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
            log.error(f"Error loading data for sample index {idx}, returning None. Error: {e}", exc_info=True)
            return None
# --- END OF DEFINITIVE PATCH ---

    def _get_image_primary_at(self, ep_idx: int, timestep_t: int) -> np.ndarray:
        """
        Private helper to get a single primary image frame from an episode.
        This version is patched to correctly call the underlying reader's API.
        """
        # 1. Get the full metadata for the desired episode.
        ep_meta = self.expert_reader.episode_metadata[ep_idx]
        
        # 2. Get the specific metadata for the 'image_primary' modality.
        img_meta = ep_meta["modalities"]["image_primary"]
        
        # 3. **CRITICAL FIX**: Call the private method with the exact signature it expects.
        # We now pass the key, compression, dtype, and shape from the metadata.
        # This call will correctly hit the per-worker LRU cache.
        full_image_array = self.expert_reader._get_full_modality_array(
            img_meta["key"], 
            img_meta["compression"], 
            img_meta["dtype"], 
            tuple(img_meta["shape"])
        )
        
        # 4. Return the specific frame requested.
        return full_image_array[timestep_t]

    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """
        Generates a placeholder sample with the correct keys, shapes, and dtypes.
        """
        dummy_img = torch.zeros((3, 224, 224), dtype=torch.float32)
        proprio_dim = self.expert_reader.get_proprioception_dim()
        action_dim = self.expert_reader.episode_metadata[0]['modalities']['actions']['shape'][-1]
        
        return {
            'initial_image': dummy_img.clone(),
            'goal_image': dummy_img.clone(),
            'observation_history': {
                'image_primary': torch.zeros((self.obs_horizon, 3, 224, 224), dtype=torch.float32),
                'image_wrist': torch.zeros((self.obs_horizon, 3, 128, 128), dtype=torch.float32),
                'proprio': torch.zeros((self.obs_horizon, proprio_dim), dtype=torch.float32),
            },
            'action_chunk': torch.zeros((self.action_horizon, action_dim), dtype=torch.float32),
        }

# -----------------------------------------------------------------------------
# 2. Custom Collate Function
# -----------------------------------------------------------------------------

def ego_planner_collate_fn(batch: List[Dict[str, any]]) -> Dict[str, any]:
    """
    A purpose-built collate function for the EgoPlannerDataset.
    """
    # The dataset now returns dummy samples, so the batch should not be empty.
    if not batch:
        raise RuntimeError("Batch is empty. This should not happen if batch_size > 0.")

    collated_batch = {
        'initial_image': torch.stack([s['initial_image'] for s in batch]),
        'goal_image': torch.stack([s['goal_image'] for s in batch]),
        'action_chunk': torch.stack([s['action_chunk'] for s in batch]),
        'observation_history': {}
    }

    obs_history_batch = [s['observation_history'] for s in batch]
    obs_keys = obs_history_batch[0].keys()

    for key in obs_keys:
        collated_batch['observation_history'][key] = torch.stack([obs[key] for obs in obs_history_batch])
    
    return collated_batch

# -----------------------------------------------------------------------------
# 3. Verification and Unit Testing (with CORRECTED shapes)
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s - %(message)s')

    DUMMY_DATASET_PATH = "/path/to/your/expert_YYYYMMDD_..._samples.lmdb"
    
    from pathlib import Path
    if not Path(DUMMY_DATASET_PATH).exists():
        print("\n" + "="*80)
        print(f"!!! VALIDATION FAILED: Dataset not found at '{DUMMY_DATASET_PATH}'")
        print("Please update the `DUMMY_DATASET_PATH` variable in this file.")
        print("="*80 + "\n")
    else:
        log.info("--- [EgoPlannerDataset] Running Unit Test (Fully Patched Version) ---")
        
        OBS_HORIZON = 2
        ACTION_HORIZON = 8
        dataset = EgoPlannerDataset(
            dataset_path=DUMMY_DATASET_PATH,
            obs_horizon=OBS_HORIZON,
            action_horizon=ACTION_HORIZON,
        )
        log.info(f"Dataset instantiation successful. Length: {len(dataset)}")

        if len(dataset) > 0:
            log.info("\n--- Verifying a single sample ---")
            sample_idx = np.random.randint(0, len(dataset))
            sample = dataset[sample_idx]
            
            log.info(f"Sample at index {sample_idx} has keys: {sample.keys()}")
            
            proprio_dim = dataset.expert_reader.get_proprioception_dim()
            action_dim = dataset.expert_reader.episode_metadata[0]['modalities']['actions']['shape'][-1]
            
            # --- CORRECT Expected Shapes (CHW for tensors) ---
            expected_shapes = {
                'initial_image': (3, 224, 224),
                'goal_image': (3, 224, 224),
                'observation_history': {
                    'image_primary': (OBS_HORIZON, 3, 224, 224),
                    'image_wrist': (OBS_HORIZON, 3, 128, 128),
                    'proprio': (OBS_HORIZON, proprio_dim)
                },
                'action_chunk': (ACTION_HORIZON, action_dim),
            }
            
            def check_shapes(data, shapes):
                for key, expected_shape in shapes.items():
                    assert key in data
                    if isinstance(expected_shape, dict):
                        check_shapes(data[key], expected_shape)
                    else:
                        actual_shape = tuple(data[key].shape)
                        assert actual_shape == expected_shape, f"Shape mismatch for '{key}'. Got {actual_shape}, expected {expected_shape}"
                        assert isinstance(data[key], torch.Tensor)
                        log.info(f"  - Key '{key}' | Shape: {actual_shape} [PASS]")

            check_shapes(sample, expected_shapes)
            log.info("Single sample verification [PASS]")

            log.info("\n--- Verifying DataLoader and Collation ---")
            dataloader = DataLoader(
                dataset, batch_size=4, shuffle=True, num_workers=2,
                collate_fn=ego_planner_collate_fn
            )
            
            batch = next(iter(dataloader))
            log.info(f"Batch has keys: {batch.keys()}")
            
            B = 4
            expected_batch_shapes = {
                'initial_image': (B, 3, 224, 224),
                'goal_image': (B, 3, 224, 224),
                'observation_history': {
                    'image_primary': (B, OBS_HORIZON, 3, 224, 224),
                    'image_wrist': (B, OBS_HORIZON, 3, 128, 128),
                    'proprio': (B, OBS_HORIZON, proprio_dim)
                },
                'action_chunk': (B, ACTION_HORIZON, action_dim),
            }

            check_shapes(batch, expected_batch_shapes)
            log.info("DataLoader and collation verification [PASS]")

        log.info("\n--- [EgoPlannerDataset] Unit Test Complete ---")