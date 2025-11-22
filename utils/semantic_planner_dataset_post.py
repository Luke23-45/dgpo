"""
Semantic Planner Dataset (Explicit Oracle & History-Aware).

Wraps the raw ExpertTrajectoryDataset to provide:
1.  **History Chunks**: Sliding windows of observations [t-H+1 : t+1].
2.  **Action History**: Sequence of past actions [t-H : t] for trajectory embedding.
3.  **Stable Subgoal Targets**: Scans future 'gt_phase' to find the next semantic transition.
4.  **Explicit Labels**: Uses 'gt_phase' and 'gt_gripper' directly from expert metadata.

Dependencies:
- utils.expert_dataset.ExpertTrajectoryDataset (The Reader)
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.expert_dataset import ExpertTrajectoryDataset

log = logging.getLogger(__name__)

class SemanticPlannerDataset(Dataset):
    def __init__(
        self, 
        dataset_path: str, 
        history_horizon: int = 10, 
        use_aug: bool = False
    ):
        self.dataset_path = dataset_path
        self.history_horizon = history_horizon
        self.use_aug = use_aug
        
        self.reader = ExpertTrajectoryDataset(
            demo_path=dataset_path, 
            observation_horizon=history_horizon, 
            action_horizon=1 
        )
        
        self.valid_indices = list(range(len(self.reader)))
        
        self.jitter_params = {
            'brightness': 0.3, 'contrast': 0.3, 'saturation': 0.3, 'hue': 0.05
        }
        self.resize_norm = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
        ])
        
        log.info(f"SemanticPlannerDataset loaded. Samples: {len(self.valid_indices)}, History: {history_horizon}")

    def __len__(self):
        return len(self.valid_indices)

    def _apply_consistent_aug(self, images: List[Image.Image]) -> torch.Tensor:
        """Applies the SAME random jitter to all frames in the history stack."""
        if not self.use_aug or random.random() > 0.5:
            return torch.stack([self.resize_norm(img) for img in images])

        fn_idx, b, c, s, h = transforms.ColorJitter.get_params(
            brightness=(max(0, 1-0.3), 1+0.3), 
            contrast=(max(0, 1-0.3), 1+0.3),
            saturation=(max(0, 1-0.3), 1+0.3),
            hue=(-0.05, 0.05)
        )

        processed = []
        for img in images:
            img = transforms.functional.adjust_brightness(img, b)
            img = transforms.functional.adjust_contrast(img, c)
            img = transforms.functional.adjust_saturation(img, s)
            img = transforms.functional.adjust_hue(img, h)
            processed.append(self.resize_norm(img))
            
        return torch.stack(processed)

    def _get_future_target(self, ep_idx: int, current_t: int, ep_len: int, phases: np.ndarray, poses: np.ndarray) -> np.ndarray:
        """Scans future phases to find stable subgoal."""
        current_phase = phases[current_t]
        target_t = ep_len - 1
        
        for t in range(current_t + 1, ep_len):
            if phases[t] != current_phase:
                next_phase = phases[t]
                phase_start = t
                phase_end = ep_len - 1
                for k in range(phase_start + 1, ep_len):
                    if phases[k] != next_phase:
                        phase_end = k
                        break
                duration = phase_end - phase_start
                target_t = phase_start + int(duration * 0.8)
                break
        
        target_t = min(target_t, ep_len - 1)
        return poses[target_t]

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        try:
            ep_idx, t_end = self.reader.get_episode_and_timestep(idx)
            ep_meta = self.reader.episode_metadata[ep_idx]
            ep_len = ep_meta['length']

            # --- 1. Helper for Fast Loading ---
            def get_full_arr(name):
                meta = ep_meta["modalities"][name]
                return self.reader._get_full_modality_array(
                    meta["key"], meta["compression"], meta["dtype"], tuple(meta["shape"])
                )

            # --- 2. Load Full Arrays (Cached) ---
            phases = get_full_arr("gt_phase")
            poses = get_full_arr("ee_pose_world")
            actions = get_full_arr("actions")
            grippers = get_full_arr("gt_gripper")
            if "advantages" in ep_meta["modalities"]:
                advs = get_full_arr("advantages")
            else:
                advs = np.ones(ep_len, dtype=np.float32)

            # --- 3. Extract History Windows ---
            # A. Actions History
            act_start = t_end - self.history_horizon
            act_end = t_end
            
            if act_start < 0:
                pad_len = abs(act_start)
                act_slice = actions[0:act_end]
                first_act = actions[0]
                act_pad = np.tile(first_act, (pad_len, 1))
                action_hist = np.concatenate([act_pad, act_slice], axis=0)
            else:
                action_hist = actions[act_start:act_end]

            # B. Observations (Images/Proprio) via Reader Chunking
            chunk_data, _ = self.reader[idx]
            imgs_np = chunk_data['image_primary']
            proprio_hist = chunk_data['proprio']

            # --- 4. Process Images (Consistent Augmentation) ---
            img_list = [Image.fromarray(img) for img in imgs_np]
            history_tensor = self._apply_consistent_aug(img_list) 
            
            # --- FIX: Goal Image Handling with Fallback ---
            try:
                # Try to get the explicit goal image
                goal_img_np = self.reader.get_goal_image(ep_idx)
            except (KeyError, RuntimeError, ValueError):
                # FALLBACK: If 'goal_image_primary' is missing from index, 
                # load the LAST frame of 'image_primary' for this episode.
                meta_img = ep_meta["modalities"]["image_primary"]
                full_imgs = self.reader._get_full_modality_array(
                    meta_img["key"], meta_img["compression"], 
                    meta_img["dtype"], tuple(meta_img["shape"])
                )
                goal_img_np = full_imgs[-1]

            goal_tensor = self.resize_norm(Image.fromarray(goal_img_np))

            # --- 5. Targets ---
            current_phase = phases[t_end]
            gt_subgoal = self._get_future_target(ep_idx, t_end, ep_len, phases, poses)
            gt_gripper = grippers[t_end]
            current_adv = advs[t_end]

            return {
                'initial_image': history_tensor,       
                'goal_image': goal_tensor,             
                'proprio_hist': torch.from_numpy(proprio_hist).float(), 
                'action_hist': torch.from_numpy(action_hist).float(),   
                'task_phase': torch.tensor(int(current_phase), dtype=torch.long),
                'ground_truth_subgoal_pose': torch.from_numpy(gt_subgoal).float(),
                'ground_truth_gripper_state': torch.tensor([float(gt_gripper)], dtype=torch.float32),
                'advantage': torch.tensor([float(current_adv)], dtype=torch.float32)
            }

        except Exception as e:
            log.error(f"Error loading sample {idx}: {e}")
            return None