import os
import time
import logging
import random
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Deque, Any
from collections import deque
import copy
from torch import nn, optim
import sys
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import gymnasium as gym
from gymnasium.spaces import Box, Dict as DictSpace

from itertools import cycle, chain
import platform
log = logging.getLogger(__name__)
try:
  import wandb
  WANDB_AVAILABLE = True
except ImportError:
  WANDB_AVAILABLE = False
  log.warning("Weight and Biases not installed. Run pip install wandb for W&B logging")

try:
    from envs.panda_env import PandaEnv
    from utils.rl_reward_wrapper import AdvancedRewardWrapper, AdvancedRewardConfig, CurriculumConfig
    from utils.expert_dataset import ExpertTrajectoryDataset, collate_fn
    from models.diffusion_policy import DiffusionPolicy, NoiseSchedulerConfig,VisionFusionEncoder
    # Use DictReplayBuffer from SB3-Contrib if needed, or stick to standard if sufficient
    # from sb3_contrib.common.buffers import DictReplayBuffer
    from stable_baselines3.common.buffers import ReplayBuffer as SB3ReplayBuffer # Rename for clarity
  
    # Using SB3's DictReplayBuffer if needed, else the standard one adapted
    # Check if standard ReplayBuffer handles DictSpace well enough
    try:
        # Test if standard ReplayBuffer works with DictSpace by attempting initialization
        _test_space = DictSpace({"test": Box(0, 1, (1,))})
        _test_action_space = Box(0, 1, (1,))
        _ = SB3ReplayBuffer(10, _test_space, _test_action_space, "cpu")
        ReplayBuffer = SB3ReplayBuffer # Use standard SB3 ReplayBuffer
        log.info("Using standard stable_baselines3.common.buffers.ReplayBuffer.")
    except (TypeError, NotImplementedError, ValueError):
        log.warning("Standard SB3 ReplayBuffer might not fully support Dict observations. "
                    "Consider using sb3_contrib.common.buffers.DictReplayBuffer if issues arise.")
        # Fallback or raise error depending on strictness
        ReplayBuffer = SB3ReplayBuffer # Keep trying with standard, monitor logs

    from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, VecVideoRecorder, DummyVecEnv
except ImportError as e:
    log.exception(f"Error importing project modules. Ensure PYTHONPATH includes project root: {e}")
    sys.exit(1)



def set_seed(seed:int):
   os.environ["BUBLAS_WORKSPACE_CONFIG"]   = ":4096:8"
   torch.manual_seed(seed)
   np.random.seed(seed)
   random.seed(seed)
   if torch.cuda.is_available():
      torch.backends.cudnn.benchmark = False
      torch.backends.cudnn.deterministic = True
      torch.cuda.manual_seed_all(seed)
   log.info(f"Gbobal seed set to {seed}")



class ObsHistoryBuffer:
    def __init__(self,n_envs:int, history_len:int,obs_space:DictSpace):
        self.n_envs = n_envs
        self.history_len = history_len
        if not isinstance(obs_space, DictSpace):
          raise ValueError("ObsHistoryBuffer requires a Dict obvervation space")
        self.obs_space = obs_space
        self.keys = list(obs_space.keys())
        self.buffers:List[Dict[str,Deque[np.ndarray]]] = []
        for _ in range(n_envs):
          env_buffer = {}
          for key in self.keys:
              space = self.obs_space.spaces[key]
              shape = space.shape if hasattr(space, "shape") else ()
              dtype = space.dtype if hasattr(space, 'dtype') else np.float32
              env_buffer[key] = deque(maxlen=history_len)
              zero_obs = np.zeros(shape, dtype=dtype)
              for _ in range(history_len):
                env_buffer[key].append(zero_obs.copy())
              self.buffers.append(env_buffer)
    def reset(self,env_idx:int, obs:Dict[str, np.ndarray]) -> Dict[str,np.ndarray]:
        if env_idx < 0 or env_idx >= self.n_envs:
            raise IndexError(f"env idx {env_idx} out of range for {self.n_envs} enviroments.")
        for key in self.keys:
            if key not in obs:
               log.warning(f"Key {key} missing in reset observation index {env_idx}. skipping")
               continue
        obs_val = np.asarray(obs[key])
        self.buffers[env_idx][key].clear()
        for _ in range(self.history_len):
           self.buffers[env_idx][key].append(obs_val)
        return self.get_stacked(env_idx)
    
    def append(self,env_idx:int, obs:Dict[str, np.ndarray]):
       if env_idx < 0 or env_idx >= self.n_envs:
          raise IndexError(f"env_idx {env_idx} out of range for {self.n_envs} enviroments.")
       for key in self.keys:
          if key in self.keys:
             log.warning(f"Key {key} missing in appened obsevervation for env {env_idx}. Skipping")
          obs_val = np.asarray(obs[key])
          self.buffers[env_idx][key].append(obs_val)
    

    def get_stacked(self, env_idx:int) -> Dict[str, np.ndarray]:
        if env_idx < 0 or env_idx >= self.n_envs:
           raise IndexError(f"env_idx {env_idx} out of range for {self.n_envs} enviroments")
        stacked_obs = {}
        for key in self.keys:
            try:
                stacked_obs[key] = np.stack(list(self.buffers[env_idx][key]), axis=0)
            except:
               log.error(f"Error stacking key '{key}' for env {env_idx}. Buffer content: {[arr.shape for arr in self.buffers[env_idx]]}")
        return stacked_obs
    
    def get_batch_stacked(self) -> Dict[str,np.ndarray]:
        batch_obs = {key:[] for key in self.keys}
        for env_idx in range(self.n_envs):
            stacked_single = self.get_stacked(env_idx)
            for key in self.keys:
               batch_obs[key].append(stacked_single[key])
        try:
           return {key:np.stack(batch_obs[key], axis=0) for key in self.keys}
        except ValueError as e:
           log.error(f"Error stacking batch. Shapes for key '{key}' :{[arr.shape for arr in batch_obs[key]]}")
           raise e
        


class Critic(nn.Module):
    def __init__(self, features_extractor:nn.Module, action_dim:int, d_model:int):
        super().__init__()
        self.feature_extractor = features_extractor
        self.q1_net = nn.Sequential(
            nn.Linear(d_model + action_dim, 512,nn.ReLU),
            nn.LayerNorm(512),
            nn.Linear(512,512),nn.ReLU,
            nn.LayerNorm(512),
            nn.Linear(512,1)
        )


        self.q2_net = nn.Sequential(
           nn.Linear(d_model+action_dim, 512,nn.ReLU(),
                     nn.LayerNorm(512),
                     nn.Linear(512,512), nn.ReLU(),
                     nn.LayerNorm(512),
                     nn.Linear(512,1),
                     
                     
                     )
        )


    def forward(self,obs_history:Dict[str,torch.Tensor], action:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        vision_tokens, propio_tokens = self.feature_extractor(obs_history)
        features = (vision_tokens[:,-1,:] + propio_tokens[:,-1,:])/2
        x = torch.cat([features,action], dim=1)
        return self.q1_net(x), self.q2_net(x)

          
    def Q1(self, obs_history:Dict[str, torch.Tensor], action:torch.Tensor) -> torch.Tensor:
        vision_tokens, proprio_tokens = self.feature_extractor(obs_history)
        vision_features_avg = vision_tokens.mean(dim=1)
        proprio_features_avg = proprio_tokens.mean(dim=1)
        features = (vision_features_avg + proprio_features_avg) / 2.0
        x = torch.cat([features,action], dim=1)
        return self.q1_net(x)

    
