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
from typing import Dict, List, Tuple, Any

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
            # For actions, padding with zeros or first action is a design choice.
            # Repeating first action is usually safer for continuity.
            acts = [acts[0]] * missing + acts
            
        # 3. Stack & Format
        # Image: (T, H, W, C) -> (T, C, H, W) -> Float -> Normalize
        img_stack = np.stack(imgs)
        if img_stack.shape[-1] == 3: # HWC -> CHW
            img_stack = np.transpose(img_stack, (0, 3, 1, 2))
            
        # Create Tensors
        # Note: We don't need gradients for inference input
        img_tensor = torch.from_numpy(img_stack).float().div_(255.0)
        prop_tensor = torch.from_numpy(np.stack(props)).float()
        act_tensor = torch.from_numpy(np.stack(acts)).float()

        # Add Batch Dimension (B=1)
        return {
            "initial_image": img_tensor.unsqueeze(0), # (1, T, C, H, W) - Model takes last or seq
            "proprio_hist": prop_tensor.unsqueeze(0), # (1, T, D_prop)
            "action_hist": act_tensor.unsqueeze(0)    # (1, T, D_act)
        }


class OnlineReplayBuffer:
    """
    Dual-Stream Replay Buffer for DAgger.
    Optimized for RAM: Stores images as uint8 (CPU), converts to float (GPU) on sample.
    """
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.success_buffer: List[Dict] = []
        self.correction_buffer: List[Dict] = []
        self.total_added = 0
    
    def add(self, sample: Dict[str, Any], is_correction: bool):
        """
        Stores a training sample.
        Crucial: Expects inputs to be CPU numpy arrays or CPU tensors to save VRAM.
        """
        target_list = self.correction_buffer if is_correction else self.success_buffer
        
        # FIFO Eviction (Split capacity between buffers)
        limit = self.capacity // 2
        if len(target_list) >= limit:
            target_list.pop(0)
            
        # Lightweight processing before storage (ensure CPU)
        processed_sample = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu()
                # Compress Images: Float (0-1) -> Uint8 (0-255)
                if k in ["initial_image", "goal_image"] and v.dtype == torch.float32:
                    v = (v * 255.0).to(torch.uint8)
            processed_sample[k] = v
            
        target_list.append(processed_sample)
        self.total_added += 1

    def sample_batch(self, batch_size: int, correction_ratio: float = 0.5, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """
        Samples a mixed batch, decompresses images, and moves to device.
        """
        n_correct = int(batch_size * correction_ratio)
        n_success = batch_size - n_correct

        # Handle empty buffer edge cases
        if not self.correction_buffer:
            n_success = batch_size
            n_correct = 0
        elif not self.success_buffer:
            n_success = 0
            n_correct = batch_size
            
        if n_correct == 0 and n_success == 0:
            return {}

        # Sampling
        batch_samples = []
        if n_correct > 0:
            batch_samples.extend(random.choices(self.correction_buffer, k=n_correct))
        if n_success > 0:
            batch_samples.extend(random.choices(self.success_buffer, k=n_success))

        # Collate & Decompress
        collated = {}
        # Get keys from first sample
        keys = batch_samples[0].keys()
        
        for key in keys:
            # Stack into (B, ...)
            tensor_stack = torch.stack([s[key] for s in batch_samples])
            
            # Decompress Images: Uint8 -> Float
            if key in ["initial_image", "goal_image"] and tensor_stack.dtype == torch.uint8:
                tensor_stack = tensor_stack.float().div_(255.0)
            
            collated[key] = tensor_stack.to(device)
            
        return collated
        
    def __len__(self):
        return len(self.success_buffer) + len(self.correction_buffer)