# FILE: utils/semantic_planner_dataset.py
# (Definitive, SOTA, Production-Grade, Robust Version)

"""
Semantic Planner Dataset.

This module defines the data pipeline for the Advantage-Weighted Semantic Planner (AWSP).
It is engineered to serve (State, Subgoal, Advantage) tuples with high efficiency
and robustness, leveraging the underlying architecture of the EgoPlannerDataset.

This dataset transforms the raw trajectory data into a Goal-Conditioned,
Value-Weighted learning formulation suitable for AWR.

Key Features:
1.  **Goal-Conditioned Logic**: Unlike behavior cloning policies which map state->action,
    this dataset structures learning around *discrete semantic subgoals* derived
    from the `task_phases` annotations.
2.  **Future-Keyframe Lookup**: Implements robust logic to identify the "Next Subgoal"
    by scanning the future task phases of an episode to find the next transition boundary.
3.  **Advantage Integration**: Natively loads the pre-calculated `advantages` modality
    to drive the AWR loss function, enabling value-weighted policy improvement.
4.  **Data Sanitization**: Automatically corrects invalid phase labels (e.g., -1 from DONE)
    to prevent CUDA errors during embedding lookups.
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import default_collate

# Import base class and reader
from utils.ego_planner_dataset import EgoPlannerDataset

# Setup logger
log = logging.getLogger(__name__)


class SemanticPlannerDataset(EgoPlannerDataset):
    """
    A specialized dataset for training the Semantic Planner (Strategist).
    
    It overrides the `__getitem__` method of `EgoPlannerDataset` to serve the
    specific data requirements of the hierarchical planning objective:
    - Inputs: Initial Image, Goal Image, Current Task Phase, Current Proprio.
    - Targets: Next Subgoal Pose (3D+Quat), Gripper State (Open/Close).
    - Weight: Calculated Advantage ($A_t$).
    """

    def __init__(self, 
                 dataset_path: str, 
                 use_aug: bool = False):
        """
        Initialize the SemanticPlannerDataset.

        Args:
            dataset_path (str): Path to the LMDB dataset (must include 'advantages').
            use_aug (bool): Whether to apply visual augmentations.
        """
        log.info(f"Initializing SemanticPlannerDataset (SOTA Version). Path: {dataset_path}")
        
        # Initialize base class with horizon=1.
        super().__init__(
            dataset_path=dataset_path,
            obs_horizon=1,
            action_horizon=1,
            use_aug=use_aug
        )
        
        # --- SOTA VALIDATION ---
        if self.expert_reader.get_num_episodes() > 0:
            first_ep_meta = self.expert_reader.episode_metadata[0]
            modalities = first_ep_meta["modalities"]
            
            required_keys = ["advantages", "task_phases", "ee_pose_world", "actions", "proprio"]
            missing_keys = [key for key in required_keys if key not in modalities]
            
            if missing_keys:
                raise RuntimeError(
                    f"Dataset at {dataset_path} is missing required modalities: {missing_keys}. "
                    "Please run 'scripts/preprocess_advantages.py' and/or 'utils/dataset_enhancer.py' first."
                )
        
        log.info("SemanticPlannerDataset validation successful. Ready for AWR training.")

    def _find_next_subgoal_pose(self, 
                                current_t: int, 
                                task_phases: np.ndarray, 
                                ee_poses: np.ndarray) -> np.ndarray:
        """
        [SOTA Logic] Robustly identifies the pose of the robot at the start of the 
        *next* semantic phase.

        Algorithm:
        1. Identify current phase $P_t$.
        2. Scan forward from $t+1$.
        3. Find first $t'$ where $P_{t'} \neq P_t$.
        4. Return $Pose_{t'}$.
        5. Fallback: If current phase extends to end of episode, return final pose.
        """
        current_phase = task_phases[current_t]
        
        # Slice forward to find transitions
        future_phases = task_phases[current_t + 1:]
        
        # Find indices where the phase changes
        transition_indices = np.where(future_phases != current_phase)[0]
        
        if len(transition_indices) > 0:
            # Found a transition. Convert relative index to absolute.
            next_phase_start_t = (current_t + 1) + transition_indices[0]
            return ee_poses[next_phase_start_t]
        else:
            # No future transition (end of episode or single phase).
            # Return final pose as the subgoal.
            return ee_poses[-1]

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves a complete training sample for the Semantic Planner.
        Includes SOTA Data Sanitization to prevent embedding crashes.
        """
        if not (0 <= idx < len(self)):
            raise IndexError(f"Index {idx} out of range.")

        try:
            # 1. Resolve Index
            ep_idx, timestep_t = self.samples[idx]
            ep_meta = self.expert_reader.episode_metadata[ep_idx]

            # 2. Helper to access data (leveraging LRU cache)
            def get_mod(name):
                meta = ep_meta["modalities"][name]
                return self.expert_reader._get_full_modality_array(
                    key=meta["key"], compression=meta["compression"],
                    dtype_str=meta["dtype"], shape_list=tuple(meta["shape"])
                )

            # 3. Load Full Modalities
            all_images = get_mod("image_primary")
            all_phases = get_mod("task_phases")
            all_poses = get_mod("ee_pose_world")
            all_actions = get_mod("actions")
            all_advantages = get_mod("advantages")
            all_proprio = get_mod("proprio")

            # 4. Extract & Sanitize Inputs
            
            # A. Initial/Goal Images
            initial_image_np = all_images[0]
            goal_image_np = all_images[-1]

            # B. Task Phase (CRITICAL FIX)
            # The enhancer maps 'DONE' to -1. nn.Embedding cannot handle -1.
            # We clamp -1 to the last valid phase (4: RETRACT) or 0.
            # Assuming phases 0..4 are valid.
            current_phase_val = int(all_phases[timestep_t])
            if current_phase_val < 0:
                current_phase_val = 4 # SOTA: Map invalid/DONE to final phase
            
            # C. Proprioception
            current_proprio_np = all_proprio[timestep_t]

            # 5. Extract Ground Truth Targets
            
            # A. Subgoal Pose
            gt_subgoal_pose_np = self._find_next_subgoal_pose(timestep_t, all_phases, all_poses)
            
            # B. Gripper State
            current_gripper_action = all_actions[timestep_t][-1]
            gt_gripper_state_val = 1.0 if current_gripper_action < -0.1 else 0.0
            
            # C. Advantage
            advantage_val = all_advantages[timestep_t]

            # 6. Visual Processing
            initial_image_pil = Image.fromarray(initial_image_np)
            goal_image_pil = Image.fromarray(goal_image_np)

            if self.use_aug:
                if random.random() < self.aug_random_apply_p:
                    initial_image_pil = self.aug_color_jitter(initial_image_pil)
                if random.random() < self.aug_random_apply_p:
                    goal_image_pil = self.aug_color_jitter(goal_image_pil)

            initial_image_tensor = self.transform_primary(initial_image_pil)
            goal_image_tensor = self.transform_primary(goal_image_pil)

            # 7. Assemble Output with Copy for Writability
            # torch.from_numpy on read-only LMDB buffers triggers warnings.
            # .float() creates a copy, making it writable and safe.
            return {
                'initial_image': initial_image_tensor.float(),
                'goal_image': goal_image_tensor.float(),
                'task_phase': torch.tensor(current_phase_val, dtype=torch.long),
                'current_proprio': torch.from_numpy(current_proprio_np).float(),
                
                'ground_truth_subgoal_pose': torch.from_numpy(gt_subgoal_pose_np).float(),
                'ground_truth_gripper_state': torch.tensor([gt_gripper_state_val], dtype=torch.float32),
                'advantage': torch.tensor([advantage_val], dtype=torch.float32)
            }

        except Exception as e:
            log.error(f"Error loading sample {idx} (Ep {ep_idx}, T {timestep_t}): {e}", exc_info=False)
            return None


def semantic_planner_collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Filters out None samples to prevent training crashes from single bad data points.
    """
    valid_samples = [s for s in batch if s is not None]

    if not valid_samples:
        log.warning("An entire batch of data loading failed. Skipping batch.")
        return {} 

    return default_collate(valid_samples)