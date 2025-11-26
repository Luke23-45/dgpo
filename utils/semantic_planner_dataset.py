# FILE: utils/semantic_planner_dataset.py
# (Definitive, SOTA, Explicit Oracle Version v9.0)

"""
Semantic Planner Dataset (v9.0 - Action Chunking & Phase Prediction).

This module defines the high-performance data pipeline for the "Strategist" model.
It transforms raw expert demonstrations into a robust training signal that solves
the "Covariate Shift" and "Causal Confusion" problems inherent in standard BC.

Architectural Upgrades:
1.  **Action Chunking**: Instead of single-step regression, we serve `k` steps
    of future trajectory. This forces the model to learn smooth, temporally
    consistent motions (Zhao et al., RSS 2023).
2.  **Temporal Context**: Serves (t) and (t-1) frames to allow internal velocity
    estimation, distinguishing "Stopping" from "Moving" visually.
3.  **Proprioceptive Noise**: Injects Gaussian noise into joint states during
    training. This serves as a lightweight DAgger, teaching the model to
    recover from small servoing errors.
4.  **Phase Self-Supervision**: Extracts Ground Truth phases as *targets* rather
    than *inputs*, forcing the vision encoder to learn semantic awareness.
"""

from __future__ import annotations
from torchvision import transforms 
import logging
import random
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
from PIL import Image
from torch.utils.data import default_collate

# Import base class
from utils.ego_planner_dataset import EgoPlannerDataset

# Setup logger
log = logging.getLogger(__name__)


class SemanticPlannerDataset(EgoPlannerDataset):
    """
    A specialized dataset for training the v9.0 Semantic Planner.
    
    Serves:
    - Inputs: [Prev_Img, Curr_Img, Goal_Img, Noisy_Proprio]
    - Targets: [Pose_Trajectory_Chunk, Gripper_Trajectory_Chunk, Phase_Label]
    """

    def __init__(self, 
                 dataset_path: str, 
                 use_aug: bool = False,
                 chunk_size: int = 10,          # Prediction Horizon (e.g., 0.3s - 0.5s)
                 proprio_noise: float = 0.02):  # Noise Std Dev (approx 2cm / 0.02rad)
        
        log.info(f"Initializing SemanticPlannerDataset (v9.0 SOTA). Path: {dataset_path}")
        log.info(f"Config: Chunk Size={chunk_size}, Proprio Noise={proprio_noise}, History=T-1")

        # Initialize base class with updated horizons
        # obs_horizon=2 ensures the reader caches at least t and t-1
        # action_horizon=chunk_size ensures the reader caches enough future steps
        super().__init__(
            dataset_path=dataset_path,
            obs_horizon=2,              
            action_horizon=chunk_size,  
            use_aug=use_aug
        )
        
        self.chunk_size = chunk_size
        self.proprio_noise = proprio_noise
        self.transform_primary = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(), # Converts [0, 255] -> [0.0, 1.0]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Converts [0, 1] -> [-1, 1]
        ])

        # --- SOTA VALIDATION (Explicit Keys) ---
        if self.expert_reader.get_num_episodes() > 0:
            first_ep_meta = self.expert_reader.episode_metadata[0]
            modalities = first_ep_meta["modalities"]
            
            # Verify all required signals exist in the LMDB
            required_keys = [
                "advantages", 
                "gt_phase", 
                "gt_gripper", 
                "ee_pose_world", 
                "image_primary",
                "proprio"
            ]
            missing_keys = [key for key in required_keys if key not in modalities]
            
            if missing_keys:
                raise RuntimeError(
                    f"Dataset is missing required modalities: {missing_keys}. "
                    "Please regenerate data with the updated Expert and Advantage Calculator."
                )
        
        log.info("Dataset loaded successfully. Ready for Trajectory & Phase training.")

    def _get_trajectory_chunk(self, 
                              current_t: int, 
                              all_poses: np.ndarray, 
                              all_grippers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Slices the next K steps for Action Chunking.
        Handles edge cases where the episode ends before K steps by padding with the final state.
        
        Args:
            current_t: Current timestep index.
            all_poses: Full episode poses [L, 7].
            all_grippers: Full episode gripper states [L, 1].
            
        Returns:
            pose_chunk: [K, 7]
            grip_chunk: [K, 1]
        """
        episode_len = len(all_poses)
        
        # We want to predict actions for t+1, t+2, ... t+k
        # (The target for state t is the action to get to t+1)
        start_t = current_t + 1
        end_t = min(start_t + self.chunk_size, episode_len)
        
        # Slice available future
        pose_chunk = all_poses[start_t : end_t]
        grip_chunk = all_grippers[start_t : end_t]
        
        # Calculate missing steps (Padding)
        pad_len = self.chunk_size - len(pose_chunk)
        
        if pad_len > 0:
            # Pad Poses: Repeat the last valid pose (Stabilization at end of task)
            # If start_t >= episode_len (edge case), use the very last frame of episode
            last_pose = pose_chunk[-1] if len(pose_chunk) > 0 else all_poses[-1]
            padding_poses = np.tile(last_pose, (pad_len, 1))
            pose_chunk = np.concatenate([pose_chunk, padding_poses], axis=0)
            
            # Pad Grippers
            last_grip = grip_chunk[-1] if len(grip_chunk) > 0 else all_grippers[-1]
            padding_grips = np.tile(last_grip, (pad_len, 1))
            grip_chunk = np.concatenate([grip_chunk, padding_grips], axis=0)
            
        return pose_chunk, grip_chunk

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves a v9.0 training sample with Noise Injection and History.
        """
        if not (0 <= idx < len(self)):
            raise IndexError(f"Index {idx} out of range.")

        try:
            # 1. Resolve Index
            ep_idx, timestep_t = self.samples[idx]
            ep_meta = self.expert_reader.episode_metadata[ep_idx]

            # 2. Helper for Fast Access (Leverages LRU Cache in Reader)
            def get_mod(name):
                meta = ep_meta["modalities"][name]
                return self.expert_reader._get_full_modality_array(
                    key=meta["key"], compression=meta["compression"],
                    dtype_str=meta["dtype"], shape_list=tuple(meta["shape"])
                )

            # 3. Load Raw Data
            all_images = get_mod("image_primary")
            all_poses = get_mod("ee_pose_world")
            all_advantages = get_mod("advantages")
            all_proprio = get_mod("proprio")
            all_gt_phases = get_mod("gt_phase")
            all_gt_grippers = get_mod("gt_gripper")

            # 4. Extract Inputs (Visual History T=2)
            
            # Current Frame (t)
            curr_image_np = all_images[timestep_t]
            
            # Previous Frame (t-1) - Handle Start of Episode
            prev_t = max(0, timestep_t - 1)
            prev_image_np = all_images[prev_t]
            
            # Goal Frame (T)
            goal_image_np = all_images[-1]

            # 5. Extract & Augment Proprioception
            # We copy to ensure we don't modify the cached array
            current_proprio_np = all_proprio[timestep_t].copy()
            
            if self.use_aug:
                # [SOTA FIX] Noise Injection
                # Simulates tracking error/drift, forcing model to rely on Vision + Goals
                # rather than memorizing exact joint coordinate strings.
                noise = np.random.normal(0, self.proprio_noise, size=current_proprio_np.shape)
                current_proprio_np += noise.astype(np.float32)

            # 6. Extract Targets (Action Chunking & Semantic Phase)
            
            # Target A: Trajectory Chunk (The "How")
            gt_pose_chunk, gt_grip_chunk = self._get_trajectory_chunk(
                timestep_t, all_poses, all_gt_grippers
            )
            
            # Target B: Phase Label (The "What")
            # Used for Auxiliary Classification Loss to enforce semantic awareness
            gt_phase_val = int(all_gt_phases[timestep_t].item())
            
            # Weighting
            advantage_val = all_advantages[timestep_t]

            # 7. Visual Transforms
            curr_image_pil = Image.fromarray(curr_image_np)
            prev_image_pil = Image.fromarray(prev_image_np)
            goal_image_pil = Image.fromarray(goal_image_np)

            if self.use_aug:
                # Independent Jitter to force feature robustness
                if random.random() < self.aug_random_apply_p:
                    curr_image_pil = self.aug_color_jitter(curr_image_pil)
                if random.random() < self.aug_random_apply_p:
                    prev_image_pil = self.aug_color_jitter(prev_image_pil)
                if random.random() < self.aug_random_apply_p:
                    goal_image_pil = self.aug_color_jitter(goal_image_pil)

            curr_image_tensor = self.transform_primary(curr_image_pil)
            prev_image_tensor = self.transform_primary(prev_image_pil)
            goal_image_tensor = self.transform_primary(goal_image_pil)

            # 8. Assemble v9.0 Output Dictionary
            return {
                # --- INPUTS ---
                'prev_image': prev_image_tensor.float(),        # (3, H, W)
                'curr_image': curr_image_tensor.float(),        # (3, H, W)
                'goal_image': goal_image_tensor.float(),        # (3, H, W)
                'curr_proprio': torch.tensor(current_proprio_np, dtype=torch.float32),
                
                # --- TARGETS ---
                # Trajectory Regression Targets
                'gt_pose_chunk': torch.tensor(gt_pose_chunk, dtype=torch.float32), # (K, 7)
                'gt_grip_chunk': torch.tensor(gt_grip_chunk, dtype=torch.float32), # (K, 1)
                
                # Classification Target (Self-Supervision)
                'gt_phase_label': torch.tensor(gt_phase_val, dtype=torch.long),    # Scalar
                
                # --- WEIGHTING ---
                'advantage': torch.tensor([advantage_val], dtype=torch.float32)
            }

        except Exception as e:
            log.error(f"Error loading sample {idx} (Ep {ep_idx}, T {timestep_t}): {e}", exc_info=False)
            return None


def semantic_planner_collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Robust collation function that filters out failed samples (None) 
    to prevent crashing the entire training batch.
    """
    valid_samples = [s for s in batch if s is not None]
    if not valid_samples: 
        log.warning("Batch collation failed: All samples were None.")
        return {}
    return default_collate(valid_samples)