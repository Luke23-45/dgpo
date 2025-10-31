# FILE: utils/data_utils.py
# Shared data structures and utilities for the ViDHiS project.

import numpy as np
import torch
from torchvision import transforms
from typing import Dict, List
import logging

log = logging.getLogger(__name__)

# --- SOTA Observation History Buffer ---
# This buffer is essential for both training and evaluation.
# It now lives in a central utility file.
class ObsHistoryBuffer:
    def __init__(self, n_envs: int, history_len: int, obs_space: Dict):
        log.info(f"Initializing ObsHistoryBuffer with history_len={history_len}")
        self.history_len = history_len
        self.n_envs = n_envs
        
        self.buffers = {}
        for key, space in obs_space.items():
            shape = (n_envs, history_len) + space.shape
            self.buffers[key] = np.zeros(shape, dtype=space.dtype)
        
        self.pos = np.zeros(n_envs, dtype=np.int64)
        self.is_full = np.zeros(n_envs, dtype=bool)

    def reset(self, env_idx: int, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        self.pos[env_idx] = 0
        self.is_full[env_idx] = False
        for key in self.buffers.keys():
            self.buffers[key][env_idx, :] = 0
        return self.append_and_get_stacked(env_idx, obs)

    def append(self, env_idx: int, obs: Dict[str, np.ndarray]):
        for key, val in obs.items():
            if key in self.buffers:
                self.buffers[key][env_idx, self.pos[env_idx]] = val
        
        self.pos[env_idx] += 1
        if self.pos[env_idx] >= self.history_len:
            self.pos[env_idx] = 0
            self.is_full[env_idx] = True

    def get_stacked(self, env_idx: int) -> Dict[str, np.ndarray]:
        stacked_obs = {}
        if self.is_full[env_idx]:
            # If buffer is full, roll to get the most recent history at the end
            # and then return the full buffer.
            idx = np.arange(self.history_len)
            roll_idx = np.roll(idx, -self.pos[env_idx])
            for key in self.buffers.keys():
                stacked_obs[key] = self.buffers[key][env_idx][roll_idx]
        else:
            # If buffer is not full, just return the valid part.
            current_len = self.pos[env_idx]
            for key in self.buffers.keys():
                stacked_obs[key] = self.buffers[key][env_idx]
        return stacked_obs
        
    def append_and_get_stacked(self, env_idx: int, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        self.append(env_idx, obs)
        return self.get_stacked(env_idx)