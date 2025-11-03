# FILE: utils/vip_c_dataset.py
#
# Definitive, SOTA, High-Performance Dataset for the ViP-C Framework.
# This script provides the data loading pipeline for training the unified
# Planner-Controller model.
#

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# We build upon the powerful, existing ExpertTrajectoryDataset for all low-level I/O.
# NOTE: This assumes ExpertTrajectoryDataset has been extended with a method
# to map a global index back to (episode, timestep).
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
                 action_horizon: int):
        super().__init__()
        logger.info(f"Initializing ViPCDataset (SOTA Orchestrator) with H_o={obs_horizon}, H_a={action_horizon}")

        # 1. Composition: We use the ExpertTrajectoryDataset as our core data engine.
        #    It handles all the complex, optimized LMDB access.
        self.expert_reader = ExpertTrajectoryDataset(
            demo_path=enhanced_dataset_path,
            observation_horizon=obs_horizon,
            action_horizon=action_horizon
        )

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
        The orchestrator method. Assembles a complete training sample for ViP-C.
        """
        ep_idx, timestep_t = -1, -1 # Initialize for robust error logging
        try:
            # 1. Map the global sample index `idx` to its episode and timestep coordinate.
            #    This is a fast, O(log N) lookup thanks to the reader's virtual index.
            #    NOTE: This requires `get_episode_and_timestep` to be implemented in ExpertTrajectoryDataset.
            ep_idx, timestep_t = self.expert_reader.get_episode_and_timestep(idx)
            ep_meta = self.expert_reader.episode_metadata[ep_idx]

            # 2. Fetch the chunked data for the Controller.
            #    This is one of the main, efficient reads from the database.
            obs_chunk_np, action_chunk_np = self.expert_reader[idx]

            # 3. Fetch the single-timestep supervisory signals for the Planner.
            #    These calls are fast and cached by the reader's LRU mechanism.
            
            # The "current image" for the planner is the last image in the history chunk.
            current_image_np = obs_chunk_np["image_primary"][-1]
            
            # Get the ground-truth task phase for the current timestep.
            def get_full_modality(modality_name: str):
                """Helper to robustly call the reader with correct, hashable arguments."""
                meta = ep_meta["modalities"][modality_name]
                return self.expert_reader._get_full_modality_array(
                    key=meta["key"],
                    compression=meta["compression"],
                    dtype_str=meta["dtype"],
                    shape_list=tuple(meta["shape"]) # Ensure hashable tuple
                )

            # Get the ground-truth task phase for the current timestep.
            all_phases = get_full_modality("task_phases")
            current_task_phase = all_phases[timestep_t]

            # Get the ground-truth subgoal heatmap.
            all_heatmaps_uint8 = get_full_modality("subgoal_heatmaps")
            gt_heatmap_uint8 = all_heatmaps_uint8[timestep_t]

            # 4. Fetch the episode-level goal image for the Planner.
            all_primary_images = get_full_modality("image_primary")
            
            # 2. The goal image is the last one in this sequence.
            goal_image_np = all_primary_images[-1]


            # --- Data Transformation (NumPy/PIL to PyTorch Tensors) ---

            # A. Transform Planner inputs
            current_image = self.transform_planner_img(Image.fromarray(current_image_np))
            goal_image = self.transform_planner_img(Image.fromarray(goal_image_np))
            task_phase = torch.tensor(current_task_phase, dtype=torch.long)
            
            # B. Transform Controller inputs
            observation_history = {
                "image_primary": torch.stack([
                    self.transform_controller_primary(Image.fromarray(img))
                    for img in obs_chunk_np["image_primary"]
                ]),
                "image_wrist": torch.stack([
                    self.transform_controller_wrist(Image.fromarray(img))
                    for img in obs_chunk_np["image_wrist"]
                ]),
                "proprio": torch.from_numpy(obs_chunk_np["proprio"].copy()).float(),
            }
            action_chunk = torch.from_numpy(action_chunk_np.copy()).float()
            
            # C. Transform the ground-truth heatmap.
            #    Convert from uint8 [0, 255] back to a float32 [0.0, 1.0] tensor
            #    and add the channel dimension.
            gt_heatmap = torch.from_numpy(gt_heatmap_uint8.copy()).float() / 255.0
            gt_heatmap = gt_heatmap.squeeze().unsqueeze(0) # Ensure shape [1, H, W]

            # 5. Assemble and return the final, model-ready dictionary.
            return {
                # --- Planner Supervisory Data ---
                "planner_current_image": current_image,
                "planner_goal_image": goal_image,
                "planner_task_phase": task_phase,
                "ground_truth_subgoal_heatmap": gt_heatmap,
                
                # --- Controller Supervisory Data ---
                "controller_observation_history": observation_history,
                "ground_truth_action_chunk": action_chunk,
            }

        except Exception as e:
            # If anything goes wrong, log the error and return None.
            # The custom collate_fn will handle this gracefully.
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