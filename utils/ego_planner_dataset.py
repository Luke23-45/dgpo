# FILE: utils/ego_planner_dataset.py
# (This is the final, fully corrected file content)

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
    The definitive, high-performance dataset for the Ego-Planner. This version
    has been fully patched and audited to be resilient and efficient.
    """
    def __init__(self,
                 dataset_path: str,
                 obs_horizon: int,
                 action_horizon: int,
                 use_aug: bool = False):
        super().__init__()
        log.info(f"Initializing EgoPlannerDataset (Definitive Patched Version, use_aug={use_aug})")

        self.expert_reader = ExpertTrajectoryDataset(
            demo_path=dataset_path,
            observation_horizon=obs_horizon,
            action_horizon=action_horizon
        )
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.use_aug = use_aug

        self.transform_primary = transforms.Compose([transforms.Resize((224, 224), antialias=True), transforms.ToTensor()])
        self.transform_wrist = transforms.Compose([transforms.Resize((128, 128), antialias=True), transforms.ToTensor()])
        
        self.aug_color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        self.aug_random_apply_p = 0.8
        self.aug_random_grayscale_p = 0.1

        self.samples: List[Tuple[int, int]] = []
        log.info("Building explicit sample index map for robust indexing...")
        for ep_idx in range(self.expert_reader.get_num_episodes()):
            ep_len = self.expert_reader.get_episode_length(ep_idx)
            start_t = self.obs_horizon - 1
            end_t = ep_len - self.action_horizon
            # Ensure start_t is not greater than end_t for short episodes
            if start_t <= end_t:
                for t in range(start_t, end_t + 1):
                    self.samples.append((ep_idx, t))
        log.info(f"Successfully initialized. Found {len(self.samples)} valid samples across {self.expert_reader.get_num_episodes()} episodes.")

    def __len__(self) -> int:
        return len(self.samples)


    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        [DEFINITIVE, SOTA, TASK-PHASE-AWARE, CORRECTED-KEY VERSION]
        Orchestrates the retrieval of a complete, processed sample for the
        enhanced Ego-Planner. This version includes the `task_phase` for
        sequentially-guided planning with the corrected dictionary key.
        """
        if not (0 <= idx < len(self)):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self)} samples.")

        ep_idx, timestep_t = -1, -1
        try:
            # --- 1. Get Timestep Info and Chunked Data ---
            ep_idx, timestep_t = self.samples[idx]
            obs_history_chunk_np, action_chunk_np = self.expert_reader[idx]
            ep_meta = self.expert_reader.episode_metadata[ep_idx]

            def get_full_modality(modality_name: str):
                meta = ep_meta["modalities"][modality_name]
                return self.expert_reader._get_full_modality_array(
                    key=meta["key"], compression=meta["compression"],
                    dtype_str=meta["dtype"], shape_list=tuple(meta["shape"])
                )

            # --- 2. Get All Required Data Modalities ---
            all_primary_images = get_full_modality("image_primary")
            initial_image_np = all_primary_images[0]
            goal_image_np = all_primary_images[-1]
            all_task_phases = get_full_modality("task_phases")
            current_task_phase = all_task_phases[timestep_t]
            
            # --- 3. Preprocessing & SOTA Correlated Augmentation ---
            initial_image = self.transform_primary(Image.fromarray(initial_image_np))
            goal_image = self.transform_primary(Image.fromarray(goal_image_np))

            observation_history = {}
            for key, val in obs_history_chunk_np.items():
                if 'image' in key:
                    img_stack_pil = [Image.fromarray(img) for img in val]
                    if self.use_aug:
                        if random.random() < self.aug_random_apply_p:
                            jitter_params = self.aug_color_jitter.get_params(
                                self.aug_color_jitter.brightness, self.aug_color_jitter.contrast,
                                self.aug_color_jitter.saturation, self.aug_color_jitter.hue
                            )
                            fn_idx, brightness, contrast, saturation, hue = jitter_params
                            for i in range(len(img_stack_pil)):
                                img = img_stack_pil[i]
                                if 0 in fn_idx: img = TF.adjust_brightness(img, brightness)
                                if 1 in fn_idx: img = TF.adjust_contrast(img, contrast)
                                if 2 in fn_idx: img = TF.adjust_saturation(img, saturation)
                                if 3 in fn_idx: img = TF.adjust_hue(img, hue)
                                img_stack_pil[i] = img
                        if random.random() < self.aug_random_grayscale_p:
                            img_stack_pil = [TF.to_grayscale(img, num_output_channels=3) for img in img_stack_pil]
                    transform_fn = self.transform_wrist if 'wrist' in key else self.transform_primary
                    observation_history[key] = torch.stack([transform_fn(img) for img in img_stack_pil])
                else:
                    observation_history[key] = torch.from_numpy(val.copy()).float()
            
            action_chunk = torch.from_numpy(action_chunk_np.copy()).float()
            task_phase_tensor = torch.tensor(current_task_phase, dtype=torch.long)
            
            # --- 4. Assemble the Final Output Dictionary ---
            return {
                'initial_image': initial_image,
                'goal_image': goal_image,
                'observation_history': observation_history,
                'action_chunk': action_chunk,
                'task_phase': task_phase_tensor, 
            }

        except Exception as e:
            log.error(f"Error loading data for sample index {idx} (ep: {ep_idx}, t: {timestep_t}). Error: {e}", exc_info=False)
            return None


    def _get_image_primary_at(self, ep_idx: int, timestep_t: int) -> np.ndarray:
        """
        [DEFINITIVE, CORRECT API USAGE]
        Private helper to get a single primary image frame from an episode.
        """
        ep_meta = self.expert_reader.episode_metadata[ep_idx]
        img_meta = ep_meta["modalities"]["image_primary"]
        
        full_image_array = self.expert_reader._get_full_modality_array(
            key=img_meta["key"],
            compression=img_meta["compression"],
            dtype_str=img_meta["dtype"],
            shape_list=tuple(img_meta["shape"])
        )
        return full_image_array[timestep_t]

    def get_num_episodes(self) -> int:
        return self.expert_reader.get_num_episodes()
    
    def get_episode_length(self, episode_idx: int) -> int:
        return self.expert_reader.get_episode_length(episode_idx)


# ==============================================================================
# SECTION 2: THE DEFINITIVE ROBUST COLLATE FUNCTION
# ==============================================================================

def ego_planner_collate_fn(batch: List[Optional[Dict[str, any]]]) -> Dict[str, any]:
    """
    [DEFINITIVE, RESILIENT BATCHING]
    Filters out any `None` samples that may have been returned by `__getitem__`
    due to data loading errors, preventing a single bad data point from
    crashing an entire training batch.
    """
    valid_samples = [s for s in batch if s is not None]

    if not valid_samples:
        log.warning("An entire batch of data loading failed. Skipping batch.")
        return {"batch_failed": True}

    return torch.utils.data.dataloader.default_collate(valid_samples)