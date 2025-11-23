# FILE: utils/online_buffer.py

"""
Online Data Management for Interactive Imitation Learning.
Implements efficient Sliding Window buffers for history-aware inference
and Prioritized Replay Buffers for DAgger training.

Key Features:
- Zero-Copy History Management: Uses `collections.deque` for O(1) updates.
- Dual-Stream Storage: Separates 'Success' and 'Correction' data.
- RAM Optimization: Stores images as uint8 numpy arrays, converts to float tensors on sampling.
"""

import collections
import logging
import random
import pickle
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch

log = logging.getLogger(__name__)

class SlidingWindowBuffer:
    """
    Real-time History Buffer for Inference.
    Maintains the last `horizon` frames for the agent's context.
    """
    def __init__(self, horizon: int, proprio_dim: int, action_dim: int):
        self.horizon = horizon
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        
        # Deques for O(1) push/pop
        self.img_queue = collections.deque(maxlen=horizon)
        self.proprio_queue = collections.deque(maxlen=horizon)
        self.action_queue = collections.deque(maxlen=horizon)
        
    def reset(self):
        self.img_queue.clear()
        self.proprio_queue.clear()
        self.action_queue.clear()

    def add(self, image: np.ndarray, proprio: np.ndarray, action: np.ndarray):
        """
        Adds a single timestep.
        Args:
            image: (H, W, C) uint8
            proprio: (D_prop,) float32
            action: (D_act,) float32 - The action taken at this step
        """
        self.img_queue.append(image)
        self.proprio_queue.append(proprio)
        self.action_queue.append(action)

    def get_history(self) -> Dict[str, torch.Tensor]:
        """
        Returns stacked history tensors ready for Model Inference.
        Padding: Repeats the first frame if history < horizon.
        """
        if len(self.img_queue) == 0:
            raise RuntimeError("Cannot get history from empty buffer.")

        # 1. Snapshot current state
        imgs = list(self.img_queue)
        props = list(self.proprio_queue)
        acts = list(self.action_queue)
        
        current_len = len(imgs)
        missing = self.horizon - current_len

        # 2. Auto-Padding (Repeat First Frame)
        if missing > 0:
            imgs = [imgs[0]] * missing + imgs
            props = [props[0]] * missing + props
            acts = [acts[0]] * missing + acts
            
        # 3. Stack & Format
        # Image: (T, H, W, C) -> (T, C, H, W) -> Float -> Normalize
        img_stack = np.stack(imgs)
        if img_stack.shape[-1] == 3: # HWC -> CHW
            img_stack = np.transpose(img_stack, (0, 3, 1, 2))
            
        # Create Tensors
        img_tensor = torch.from_numpy(img_stack).float().div_(255.0)
        prop_tensor = torch.from_numpy(np.stack(props)).float()
        act_tensor = torch.from_numpy(np.stack(acts)).float()

        # Add Batch Dimension (B=1)
        return {
            "initial_image": img_tensor.unsqueeze(0), 
            "proprio_hist": prop_tensor.unsqueeze(0), 
            "action_hist": act_tensor.unsqueeze(0)
        }


class OnlineReplayBuffer:
    """
    Dual-Stream Replay Buffer for DAgger.
    """
    def __init__(self, capacity: int = 10000, default_advantage: float = 2.0):
        self.capacity = capacity
        self.success_buffer: List[Dict] = []
        self.correction_buffer: List[Dict] = []
        self.total_added = 0
        self.default_advantage = default_advantage
    
    def add(self, sample: Dict[str, Any], is_correction: bool):
        """
        Stores a training sample. 
        Expects inputs to be CPU numpy arrays or CPU tensors.
        Images (float 0-1) are compressed to uint8 (0-255) for RAM efficiency.
        """
        target_list = self.correction_buffer if is_correction else self.success_buffer
        
        # FIFO Eviction
        limit = self.capacity // 2
        if len(target_list) >= limit:
            target_list.pop(0)
            
        processed_sample = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu()
                # Compress Images: Float (0-1) -> Uint8 (0-255)
                # Matches keys in SemanticPlannerDataset and DAggerCollector
                if k in ["curr_image", "goal_image", "prev_image"] and v.dtype == torch.float32:
                   v = (v * 255.0).to(torch.uint8)
            processed_sample[k] = v
            
        target_list.append(processed_sample)
        self.total_added += 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a single sample by index (virtual concatenated list).
        Necessary for MixedDataset compatibility.
        """
        # Virtual list: [ ...success_buffer..., ...correction_buffer... ]
        n_success = len(self.success_buffer)
        
        if idx < n_success:
            sample = self.success_buffer[idx]
        else:
            corr_idx = idx - n_success
            if corr_idx >= len(self.correction_buffer):
                raise IndexError(f"Index {idx} out of range for OnlineReplayBuffer")
            sample = self.correction_buffer[corr_idx]

        # Decompress and format
        out_sample = {}
        for k, v in sample.items():
            # Decompress: Uint8 -> Float (0-1)
            if isinstance(v, torch.Tensor) and v.dtype == torch.uint8 and k in ["curr_image", "goal_image", "prev_image"]:
                out_sample[k] = v.float().div_(255.0)
            # If it was numpy uint8 (from env), convert to Tensor Float
            elif isinstance(v, np.ndarray) and v.dtype == np.uint8 and k in ["curr_image", "goal_image", "prev_image"]:
                 out_sample[k] = torch.from_numpy(v).float().div_(255.0)
                 if out_sample[k].ndim == 3 and out_sample[k].shape[-1] == 3: # HWC -> CHW
                     out_sample[k] = out_sample[k].permute(2, 0, 1)
            else:
                out_sample[k] = v
        return out_sample

    def __len__(self):
        return len(self.success_buffer) + len(self.correction_buffer)
        
    def save_to_disk(self, path: str):
        """Saves buffer to disk using Pickle."""
        data = {
            "success_buffer": self.success_buffer,
            "correction_buffer": self.correction_buffer,
            "total_added": self.total_added
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        log.info(f"OnlineBuffer saved to {path} ({len(self)} samples)")

    def load_from_disk(self, path: str):
        """Loads buffer from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.success_buffer = data["success_buffer"]
        self.correction_buffer = data["correction_buffer"]
        self.total_added = data["total_added"]
        log.info(f"OnlineBuffer loaded from {path} ({len(self)} samples)")