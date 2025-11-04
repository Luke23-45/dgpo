# FILE: utils/vip_c_dataset.py
#
# Definitive, SOTA, High-Performance Dataset for the ViP-C Framework.
# This script provides the data loading pipeline for training the unified
# Planner-Controller model.
#

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from models.vip_c import LinearNormalizer
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


from utils.expert_dataset import ExpertTrajectoryDataset

logger = logging.getLogger(__name__)

# --- The Main ViP-C Dataset Class ---

class ViPCDataset(Dataset):
    """
    The definitive, high-performance dataset for the ViP-C model.

    This class acts as a smart "Orchestrator" on top of the ExpertTrajectoryDataset.
    Its primary responsibility is to assemble the complex, multi-part data samples
    required by the ViP-C's dual-objective (Planner + Controller) training,
    while delegating all low-level, high-performance data access (LMDB reads,
    decompression, caching) to the underlying reader.

    Each `__getitem__` call returns a complete dictionary containing supervisory
    signals for both the Planner and the Controller.
    """

    def __init__(self,
                 enhanced_dataset_path: str,
                 obs_horizon: int,
                 action_horizon: int,
                 action_normalizer: LinearNormalizer,
                 proprio_normalizer: LinearNormalizer):
        super().__init__()
        logger.info(f"Initializing ViPCDataset (SOTA Orchestrator) with H_o={obs_horizon}, H_a={action_horizon}")

        # 1. Composition: We use the ExpertTrajectoryDataset as our core data engine.
        #    It handles all the complex, optimized LMDB access.
        self.expert_reader = ExpertTrajectoryDataset(
            demo_path=enhanced_dataset_path,
            observation_horizon=obs_horizon,
            action_horizon=action_horizon
        )
        self.action_normalizer = action_normalizer
        self.proprio_normalizer = proprio_normalizer
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        # 2. Define the image transformation pipelines.
        #    These are standard pipelines for pre-trained vision models.
        #    The Planner's ViT requires ImageNet normalization.
        self.transform_planner_img = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # The Controller's ResNets are trained from scratch, so they only need
        # to be converted to tensors. No normalization is applied here.
        self.transform_controller_primary = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
        ])
        self.transform_controller_wrist = transforms.Compose([
            transforms.Resize((128, 128), antialias=True),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        # The total number of valid samples is determined by the underlying reader.
        return len(self.expert_reader)


    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        [DEFINITIVE, DUAL-MODE VERSION]
        Assembles a complete sample, operating in 'stat computation' mode or
        'training' mode based on whether the passed normalizers are fitted.
        """
        ep_idx, timestep_t = -1, -1
        try:
            ep_idx, timestep_t = self.expert_reader.get_episode_and_timestep(idx)
            ep_meta = self.expert_reader.episode_metadata[ep_idx]
            obs_chunk_np, action_chunk_np = self.expert_reader[idx]

            current_image_np = obs_chunk_np["image_primary"][-1]
            
            def get_full_modality(modality_name: str):
                meta = ep_meta["modalities"][modality_name]
                return self.expert_reader._get_full_modality_array(
                    key=meta["key"], compression=meta["compression"],
                    dtype_str=meta["dtype"], shape_list=tuple(meta["shape"])
                )

            all_phases = get_full_modality("task_phases")
            current_task_phase = all_phases[timestep_t]
            all_heatmaps_uint8 = get_full_modality("subgoal_heatmaps")
            gt_heatmap_uint8 = all_heatmaps_uint8[timestep_t]

            # goal_image_np = self.expert_reader.get_goal_image(ep_idx)
            goal_image_np = get_full_modality("image_primary")[-1]

            # --- Data Transformation ---
            current_image = self.transform_planner_img(Image.fromarray(current_image_np))
            goal_image = self.transform_planner_img(Image.fromarray(goal_image_np))
            task_phase = torch.tensor(current_task_phase, dtype=torch.long)
            
            proprio_raw = torch.from_numpy(obs_chunk_np["proprio"].copy()).float()
            action_raw = torch.from_numpy(action_chunk_np.copy()).float()
            
            is_stat_computation_mode = self.action_normalizer.min is None

            observation_history = {
                "image_primary": torch.stack([self.transform_controller_primary(Image.fromarray(img)) for img in obs_chunk_np["image_primary"]]),
                "image_wrist": torch.stack([self.transform_controller_wrist(Image.fromarray(img)) for img in obs_chunk_np["image_wrist"]]),
                "proprio": self.proprio_normalizer.normalize(proprio_raw) if not is_stat_computation_mode else proprio_raw,
            }
            action_chunk = self.action_normalizer.normalize(action_raw) if not is_stat_computation_mode else action_raw
            
            gt_heatmap = torch.from_numpy(gt_heatmap_uint8.copy()).float() / 255.0
            gt_heatmap = gt_heatmap.squeeze().unsqueeze(0)

            result = {
                "planner_current_image": current_image, "planner_goal_image": goal_image,
                "planner_task_phase": task_phase, "ground_truth_subgoal_heatmap": gt_heatmap,
                "controller_observation_history": observation_history,
                "ground_truth_action_chunk": action_chunk,
            }
            
            if is_stat_computation_mode:
                result['ground_truth_action_chunk_raw'] = action_raw
                result['controller_observation_history']['proprio_raw'] = proprio_raw
            
            return result

        except Exception as e:
            logger.error(f"Error loading data for sample index {idx} (ep: {ep_idx}, t: {timestep_t}). Error: {e}", exc_info=False)
            return None


# --- Custom Collate Function for Robust Batching ---

def vip_c_collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    A purpose-built, robust collate function for the ViPCDataset.

    Its primary job is to filter out any `None` samples that may have been
    returned by `__getitem__` due to data loading errors. This prevents a single
    bad data point from crashing the entire training batch.
    """
    # 1. Filter out failed samples (None values).
    valid_samples = [s for s in batch if s is not None]

    if not valid_samples:
        # If the entire batch failed, return a special dictionary indicating this.
        # The training loop should be prepared to handle this case by skipping the batch.
        logger.warning("An entire batch of data loading failed. Skipping batch.")
        return {"batch_failed": True}

    # 2. Use PyTorch's default collate to stack the valid samples.
    #    This is a highly optimized function that correctly handles dictionaries
    #    and nested tensors.
    return torch.utils.data.dataloader.default_collate(valid_samples)