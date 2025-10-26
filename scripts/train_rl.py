# FILE: train_rl.py
# (State-of-the-Art, Hydra-Configurable, TD3+BC, CFG-Aware, History-Managed, Optimized Eval Version)

"""
State-of-the-art RL fine-tuning script for the advanced, pre-trained Diffusion Policy.

This script implements a complete training framework for fine-tuning the new
state-of-the-art diffusion actor (featuring ResNet, Cross-Attention, and AdaLN-Zero)
using online reinforcement learning, regularized by offline expert data.

This version incorporates SOTA practices including:
  - **Robust Observation History Management**: Uses a custom `ObsHistoryBuffer`
    to manage observation sequences (`H_o`) needed by the Transformer policy.
  - **Hydra Configuration**: For flexible and reproducible experiments[cite: 804].
  - **TD3+BC Algorithm**: Stable offline-to-online fine-tuning with adaptive
    BC weight[cite: 806, 854].
  - **Advanced Diffusion Policy Actor**: Integrates the SOTA diffusion model
    with Classifier-Free Guidance (CFG) for high-quality action generation [cite: 807-809].
  - **Representation Sharing**: Critic leverages the actor's frozen vision
    encoder for sample efficiency[cite: 810, 818].
  - **Comprehensive Logging**: TensorBoard and optional W&B integration[cite: 811].
  - **Optimized Evaluation & Video Logging**: Decouples statistics collection
    from video recording for potential performance gains and records exactly
    one evaluation episode video directly to W&B when triggered[cite: 813].
  - **Reproducibility**: Deterministic seeding and checkpointing[cite: 814].
"""

# -------------------------
# 1. Imports
# -------------------------
# Standard Library
import os
import time
import logging
import random
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Deque
from collections import deque
import copy
import sys
import shutil
# Third-Party
import numpy as np
import torch
from torch import nn, optim
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
import functools 

log = logging.getLogger(__name__)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    log.warning("Weights & Biases not installed. Run `pip install wandb` for W&B logging.")

# Project-Specific
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


# Setup a logger for the script (Hydra usually configures handlers)
log = logging.getLogger(__name__)

# -------------------------
# 2. Helper Functions & Classes
# -------------------------

def set_seed(seed: int):
    """Sets the seed for all relevant random number generators for reproducibility."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # For deterministic CUDA ops
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Try to make CUDA deterministic (may impact performance)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed) # Important for multi-GPU
    # torch.use_deterministic_algorithms(True) # Can cause issues with some operations
    log.info(f"Global seed set to {seed}")

class ObsHistoryBuffer:
    """
    Manages observation history (frame stacking) for Dict observation spaces.
    (Unchanged from previous version - already SOTA)
    """
    def __init__(self, n_envs: int, history_len: int, obs_space: DictSpace):
        self.n_envs = n_envs
        self.history_len = history_len
        # Ensure obs_space is usable
        if not isinstance(obs_space, DictSpace):
             raise ValueError("ObsHistoryBuffer requires a Dict observation space.")
        self.obs_space = obs_space
        self.keys = list(obs_space.keys())

        self.buffers: List[Dict[str, Deque[np.ndarray]]] = []
        for _ in range(n_envs):
            env_buffer = {}
            for key in self.keys:
                # Ensure space shape is tuple
                space = self.obs_space.spaces[key]
                shape = space.shape if hasattr(space, 'shape') else ()
                dtype = space.dtype if hasattr(space, 'dtype') else np.float32

                # Create deque for this key
                env_buffer[key] = deque(maxlen=history_len)

                 # Pre-fill with zeros matching the space shape and dtype
                zero_obs = np.zeros(shape, dtype=dtype)
                for _ in range(history_len):
                    env_buffer[key].append(zero_obs.copy())

            self.buffers.append(env_buffer)


    def reset(self, env_idx: int, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Resets the history for a specific env, padding with the initial obs."""
        if env_idx < 0 or env_idx >= self.n_envs:
            raise IndexError(f"env_idx {env_idx} out of range for {self.n_envs} environments.")
        for key in self.keys:
            if key not in obs:
                 log.warning(f"Key {key} missing in reset observation for env {env_idx}. Skipping.")
                 continue
            # Ensure obs[key] is numpy array before appending
            obs_val = np.asarray(obs[key])
            self.buffers[env_idx][key].clear()
            for _ in range(self.history_len):
                self.buffers[env_idx][key].append(obs_val)
        return self.get_stacked(env_idx)

    def append(self, env_idx: int, obs: Dict[str, np.ndarray]):
        """Appends a new observation to the history for a specific env."""
        if env_idx < 0 or env_idx >= self.n_envs:
            raise IndexError(f"env_idx {env_idx} out of range for {self.n_envs} environments.")
        for key in self.keys:
             if key not in obs:
                 log.warning(f"Key {key} missing in append observation for env {env_idx}. Skipping.")
                 continue
             # Ensure obs[key] is numpy array before appending
             obs_val = np.asarray(obs[key])
             self.buffers[env_idx][key].append(obs_val)


    def get_stacked(self, env_idx: int) -> Dict[str, np.ndarray]:
        """Returns the stacked observation history for a single env."""
        if env_idx < 0 or env_idx >= self.n_envs:
            raise IndexError(f"env_idx {env_idx} out of range for {self.n_envs} environments.")
        stacked_obs = {}
        for key in self.keys:
            try:
                stacked_obs[key] = np.stack(list(self.buffers[env_idx][key]), axis=0)
            except ValueError as e:
                log.error(f"Error stacking key '{key}' for env {env_idx}. Buffer content: {[arr.shape for arr in self.buffers[env_idx][key]]}")
                raise e
        return stacked_obs


    def get_batch_stacked(self) -> Dict[str, np.ndarray]:
        """Returns the stacked observation history for *all* envs as a batch."""
        batch_obs = {key: [] for key in self.keys}
        for env_idx in range(self.n_envs):
            stacked_single = self.get_stacked(env_idx)
            for key in self.keys:
                 batch_obs[key].append(stacked_single[key])

        # Stack along the new batch dimension (axis=0)
        try:
            return {key: np.stack(batch_obs[key], axis=0) for key in self.keys}
        except ValueError as e:
            log.error(f"Error stacking batch. Shapes for key '{key}': {[arr.shape for arr in batch_obs[key]]}")
            raise e



class Critic(nn.Module):
    """Twin Critic network for TD3."""
    def __init__(self,
                 features_extractor: nn.Module,
                 action_dim: int,
                 d_model: int,
                 hidden_dims: List[int] = [512, 512], # From config
                 use_layernorm: bool = True,          # From config
                 use_last_feature: bool = False      # From config
                 ):
        super().__init__()
        self.features_extractor = features_extractor
        self.use_last_feature = use_last_feature
        self.d_model = d_model

        # Build Q-networks based on config
        self.q1_net = self._build_mlp(d_model + action_dim, 1, hidden_dims, use_layernorm)
        self.q2_net = self._build_mlp(d_model + action_dim, 1, hidden_dims, use_layernorm)
        log.info(f"Critic MLP built with hidden_dims={hidden_dims}, use_layernorm={use_layernorm}")
        log.info(f"Critic using features from {'last timestep' if use_last_feature else 'averaged history'}.")


    def _build_mlp(self, input_dim: int, output_dim: int, hidden_dims: List[int], use_layernorm: bool) -> nn.Sequential:
        """Helper to build MLP layers."""
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)

    def _extract_features(self, obs_history: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extracts features, applying configured logic (last vs avg)."""
        # Gradients must flow for actor update, so parameter freezing is handled in train_step.
        vision_tokens, proprio_tokens = self.features_extractor(obs_history)

        if self.use_last_feature:
            # --- Use features from the LAST timestep ---
            # This assumes both vision_tokens and proprio_tokens have shape (B, H_o, D).
            # If your VisionFusionEncoder has a more complex output, this logic may need adjustment.
            if vision_tokens.shape[1] == proprio_tokens.shape[1]: # Check for history dimension
                 last_vision_feat = vision_tokens[:, -1, :]
                 last_proprio_feat = proprio_tokens[:, -1, :]
                 features = (last_vision_feat + last_proprio_feat) / 2.0
            else:
                 log.warning("Unexpected feature shapes for 'last_feature' mode. Falling back to averaging.")
                 features = (vision_tokens.mean(dim=1) + proprio_tokens.mean(dim=1)) / 2.0
        else:
            # --- Average features over history (default method) ---
            vision_features_avg = vision_tokens.mean(dim=1)
            proprio_features_avg = proprio_tokens.mean(dim=1)
            features = (vision_features_avg + proprio_features_avg) / 2.0

        return features

    def forward(self, obs_history: Dict[str, torch.Tensor], action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes Q-values."""
        features = self._extract_features(obs_history)
        x = torch.cat([features, action], dim=1)
        return self.q1_net(x), self.q2_net(x)

    def Q1(self, obs_history: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        """Computes the Q-value from the first critic only."""
        features = self._extract_features(obs_history)
        x = torch.cat([features, action], dim=1)
        return self.q1_net(x)


class DiffusionActor(nn.Module):
    """Actor wrapper for the DiffusionPolicy, supporting CFG."""
    def __init__(self, diffusion_policy: DiffusionPolicy, guidance_scale: float, sampling_steps: int):
        super().__init__()
        self.diffusion_policy = diffusion_policy
        self.guidance_scale = guidance_scale
        self.sampling_steps = sampling_steps
        log.info(f"DiffusionActor initialized with guidance_scale={guidance_scale}, sampling_steps={sampling_steps}")

    @torch.no_grad()
    def act(self, obs_history: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Selects an action for inference/environment interaction (no gradients)."""
        self.diffusion_policy.eval() # Ensure model is in eval mode
        
        sampled_actions = self.diffusion_policy.sample(
            obs_history,
            steps=self.sampling_steps,
            guidance_scale=self.guidance_scale,
            use_ema=True # Use EMA weights for inference
        )
        # Return the first action in the predicted sequence
        return sampled_actions[:, 0, :]

    def forward(self, obs_history: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Selects an action with gradients enabled through sampling.
        WARNING: Backpropagating through diffusion sampling is computationally
        expensive and potentially unstable. This method should only be used
        with the original TD3-style actor loss, not the preferred re-weighted BC loss.
        """
        self.diffusion_policy.train() # Ensure model is in train mode for training forward pass
        
        # Use non-EMA weights for the training forward pass
        sampled_actions = self.diffusion_policy.sample(
            obs_history,
            steps=self.sampling_steps,
            guidance_scale=self.guidance_scale,
            use_ema=False 
        )
        return sampled_actions[:, 0, :]


# -------------------------
# 3. Main Trainer Class
# -------------------------
class RLFineTuner:
    """Encapsulates the entire TD3+BC fine-tuning pipeline."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.start_time = time.time()

        # --- Setup Environment and Logging ---
        self.device = torch.device(cfg.device)
        self.output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        log.info(f"Output directory: {self.output_dir}")
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True) # For saving checkpoints

        self.writer = SummaryWriter(log_dir=self.output_dir / "tensorboard")
        self.use_wandb = WANDB_AVAILABLE and cfg.logging.use_wandb
        if self.use_wandb:
            wandb.init(
                project=cfg.logging.wandb_project,
                name=cfg.logging.get("wandb_run_name", self.output_dir.name),
                config=OmegaConf.to_container(cfg, resolve=True),
                dir=str(self.output_dir), # W&B expects string path
                sync_tensorboard=True, # Sync TB logs to W&B
                monitor_gym=True, # Automatically log videos and stats
                save_code=True, # Save main script to W&B
            )

        # --- Vectorized Environments ---
        log.info(f"Initializing {cfg.environment.n_envs} parallel training environments...")
        self.env = self._make_vec_env(is_eval=False)

        # --- Evaluation Environment (No Video Recorder Wrapper Here) ---
        # We need the unwrapped env for manual rendering during video generation
        n_eval_envs = self.cfg.logging.n_eval_episodes
        log.info(f"Initializing {n_eval_envs} parallel envs for evaluation stats...")
        self.eval_env = self._make_vec_env(is_eval=True, n_envs_override=n_eval_envs)

        # [SOTA PATCH] Create a single, separate env for reliable video rendering.
        log.info("Initializing single environment for video recording...")
        self.video_env = self._make_vec_env(is_eval=True, n_envs_override=1)

        # --- Action Space properties ---
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])

        # --- Observation History & Replay Buffer ---
        self.history_len = self.cfg.model.observation_horizon
        self.n_envs = self.cfg.environment.n_envs

        single_step_obs_space = self.env.observation_space
        history_obs_space = self._create_history_obs_space(single_step_obs_space, self.history_len)

        log.info("Initializing Replay Buffer...")
        # Ensure DictReplayBuffer is used if standard ReplayBuffer fails
        try:
             # Use DictReplayBuffer for robustness with complex observations
             from stable_baselines3.common.buffers import DictReplayBuffer
             self.replay_buffer = DictReplayBuffer(
                 buffer_size=self.cfg.rl_algorithm.buffer_size,
                 observation_space=history_obs_space,
                 action_space=self.env.action_space,
                 device=self.device,
                 n_envs=self.n_envs,
                 handle_timeout_termination=False, # Important for PBRS
             )
             log.info("Using sb3_contrib.common.buffers.DictReplayBuffer.")
        except ImportError:
              log.warning("sb3_contrib not found. Falling back to standard ReplayBuffer. "
                         "Install sb3_contrib (`pip install sb3-contrib`) for robust Dict observation handling.")
              raise ImportError("Please install sb3-contrib to use DictReplayBuffer - pip install sb3-contrib" )



        self.obs_history = ObsHistoryBuffer(self.n_envs, self.history_len, single_step_obs_space)
        # [SOTA PATCH] Size the eval history buffer for the *parallel* evaluation environment.
        self.eval_obs_history = ObsHistoryBuffer(n_eval_envs, self.history_len, single_step_obs_space)

        # --- Expert Dataloader ---
        self.expert_loader = self._make_expert_loader()
        self.expert_iterator = cycle(self.expert_loader)

        # --- Build Models ---
        self._build_models_and_optimizers(history_obs_space)

        # --- State Tracking ---
        self.total_timesteps = 0
        self.timesteps_since_eval = 0
        self.best_eval_success_rate = -1.0 # Track best model based on success rate
        self.grad_accumulation_counter = 0 # For gradient accumulation

        # --- Checkpointing ---
        self.checkpoint_path = self.output_dir / "checkpoints" / "last_checkpoint.pth"
        self._load_checkpoint() # Attempt to load if exists

    def _create_history_obs_space(self, obs_space: DictSpace, history_len: int) -> DictSpace:
        """Takes a single-step Dict obs space and adds the history dimension."""
        if not isinstance(obs_space, DictSpace):
             raise ValueError("_create_history_obs_space requires a Dict observation space.")

        history_spaces = {}
        for key, space in obs_space.spaces.items():
            if isinstance(space, Box):
                # Ensure dtype is valid NumPy type
                try:
                     actual_dtype = np.dtype(space.dtype)
                except TypeError:
                     log.warning(f"Invalid dtype '{space.dtype}' for key '{key}'. Defaulting to float32.")
                     actual_dtype = np.float32

                history_shape = (history_len,) + space.shape # Prepend history len

                # Create low/high arrays with the history dimension
                low_with_history = np.repeat(np.expand_dims(space.low, axis=0), history_len, axis=0)
                high_with_history = np.repeat(np.expand_dims(space.high, axis=0), history_len, axis=0)

                history_spaces[key] = Box(
                    low=low_with_history,
                    high=high_with_history,
                    shape=history_shape,
                    dtype=actual_dtype
                )
            else:
                 log.warning(f"Unsupported space type {type(space)} for key '{key}' in history creation. Skipping.")
                 # Or handle other types like Discrete if necessary
                 # For now, we only support Box spaces within the Dict.

        return DictSpace(history_spaces)



    def _build_env_worker(rank: int, cfg: DictConfig, is_eval: bool = False) -> gym.Env:
        """
        Worker function to create a single env instance in a subprocess.
        This function is standalone to be pickleable.
        """
        log = logging.getLogger(__name__)
        env_cfg = cfg.environment

        base_seed = int(getattr(cfg, "seed", 0))
        seed = base_seed + rank + (1000 if is_eval else 0)

        try:
            env = PandaEnv(
                xml_path=env_cfg.get("xml_path", None),
                control_mode=env_cfg.get("control_mode", "delta"),
                render_mode="rgb_array",
            )
        except Exception as e:
            log.exception(f"Failed to instantiate PandaEnv (rank={rank}): {e}")
            raise

        try:
            reward_cfg = None
            curriculum_cfg = None
            if hasattr(cfg, "reward"):
                reward_cfg = AdvancedRewardConfig(**cfg.reward) if cfg.reward else None
            if hasattr(cfg, "curriculum"):
                curriculum_cfg = CurriculumConfig(**cfg.curriculum) if cfg.curriculum else None

            if reward_cfg or curriculum_cfg:
                env = AdvancedRewardWrapper(env, reward_cfg=reward_cfg, curriculum_cfg=curriculum_cfg)
        except Exception as e:
            log.exception(f"Failed to apply reward/curriculum wrappers (rank={rank}): {e}")
            raise

        try:
            max_steps = int(env_cfg.get("max_episode_steps", 1000))
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
        except Exception:
            pass

        try:
            try:
                env.reset(seed=seed)
            except TypeError:
                if hasattr(env, "seed"):
                    env.seed(seed)
                else:
                    np.random.seed(seed)
        except Exception as e:
            log.exception(f"Failed to seed/reset environment (rank={rank}, seed={seed}): {e}")
            raise

        log.info(f"[ENV WORKER INIT] rank={rank} is_eval={is_eval} seed={seed}")
        return env  

    def _make_vec_env(self, is_eval: bool = False, n_envs_override: Optional[int] = None):
        """
        Creates a vectorized environment using the standalone worker function to
        ensure pickleability for SubprocVecEnv.
        """
        log = logging.getLogger(__name__)

        if n_envs_override is not None:
            n_envs = n_envs_override
        else:
            n_envs = 1 if is_eval else int(getattr(self.cfg.environment, "n_envs", 1))
        
        if n_envs < 1: n_envs = 1

        os_name = platform.system()
        vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv

        if os_name in ["Windows", "Darwin"] and n_envs > 1:
            log.warning(f"[SAFEGUARD] Detected {os_name}. Falling back to DummyVecEnv.")
            vec_env_cls = DummyVecEnv
        elif os_name == "Linux" and torch.cuda.is_available() and n_envs > 1:
            try:
                torch.cuda.init()
                log.info("[INFO] CUDA initialized in parent process before spawning workers.")
            except Exception:
                log.warning("[WARN] torch.cuda.init() failed.")
        
        # Use functools.partial to create pickleable callables that pass the config
        env_fns = [
            functools.partial(self._build_env_worker, rank=i, cfg=self.cfg, is_eval=is_eval)
            for i in range(n_envs)
        ]
        
        t0 = time.time()
        try:
            env = vec_env_cls(env_fns)
        except Exception as e:
            log.error(f"[ERROR] VecEnv creation failed with {vec_env_cls.__name__}: {e}", exc_info=True)
            if vec_env_cls is not DummyVecEnv:
                log.warning("[RECOVERY] Retrying with DummyVecEnv fallback.")
                try:
                    env = DummyVecEnv(env_fns)
                    vec_env_cls = DummyVecEnv
                except Exception as e2:
                    log.exception(f"[FATAL] DummyVecEnv also failed: {e2}")
                    raise
            else:
                raise

        
        return env

    def _make_vec_env(self, is_eval: bool = False, n_envs_override: Optional[int] = None):
        """
        Creates a vectorized environment for training or evaluation.
        - Platform-aware: falls back to DummyVecEnv on Windows/macOS for stability.
        - On Linux+GPU, initializes CUDA context prior to forking subprocesses.
        - Returns a SB3 VecEnv instance.
        """
        log = logging.getLogger(__name__)

        def make_env(rank: int):
            def _init():
                # Note: _build_single_env handles seeding & reset internally.
                env = self._build_single_env(rank, is_eval=is_eval)
                return env
            return _init

        if n_envs_override is not None:
            n_envs = n_envs_override
        else:
            n_envs = 1 if is_eval else int(getattr(self.cfg.environment, "n_envs", 1))
        
        if n_envs < 1:
            n_envs = 1

        os_name = platform.system()
        # default choice
        vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv

        # Platform-specific safe-guards
        if os_name == "Windows" and n_envs > 1:
            log.warning("[SAFEGUARD] Detected Windows OS. SubprocVecEnv uses 'spawn' multiprocessing "
                        "which often breaks with PyTorch/CuDNN/MuJoCo. Falling back to DummyVecEnv.")
            vec_env_cls = DummyVecEnv
        elif os_name == "Darwin" and n_envs > 1:
            log.warning("[SAFEGUARD] Detected macOS. Multiprocessing with GL/MuJoCo may deadlock. "
                        "Falling back to DummyVecEnv.")
            vec_env_cls = DummyVecEnv
        elif os_name == "Linux" and torch.cuda.is_available() and n_envs > 1:
            try:
                # Warm up CUDA context to avoid duplicate-factory registration in children
                torch.cuda.init()
                log.info("[INFO] CUDA initialized in parent process before spawning workers.")
            except Exception:
                # If init fails, we still continue — worker creation may still work
                log.warning("[WARN] torch.cuda.init() failed or was a no-op.")

        env_fns = [make_env(i) for i in range(n_envs)]
        t0 = time.time()
        try:
            env = vec_env_cls(env_fns)
        except Exception as e:
            log.error(f"[ERROR] VecEnv creation failed with {vec_env_cls.__name__}: {e}", exc_info=True)
            # Fallback: try DummyVecEnv as recovery
            if vec_env_cls is not DummyVecEnv:
                log.warning("[RECOVERY] Retrying with DummyVecEnv fallback.")
                try:
                    env = DummyVecEnv(env_fns)
                    vec_env_cls = DummyVecEnv
                except Exception as e2:
                    log.exception(f"[FATAL] DummyVecEnv also failed: {e2}")
                    raise
            else:
                raise

        creation_time = time.time() - t0
        log.info(f"[ENV BUILDER] Created {n_envs} env(s) using {vec_env_cls.__name__} in {creation_time:.3f}s on {os_name} | CUDA={'ON' if torch.cuda.is_available() else 'OFF'}")

        # Optional: micro-benchmark to report env stepping performance (enable in cfg)
        try:
            if getattr(self.cfg.environment, "diagnose_env_speed", False):
                # perform a tiny benchmark (non-invasive)
                obs = env.reset()
                sample_action = None
                # build one sample action if possible
                try:
                    sample_action = env.action_space.sample()
                except Exception:
                    sample_action = None
                steps = 5
                t1 = time.time()
                for _ in range(steps):
                    if sample_action is not None:
                        obs, rewards, dones, infos = env.step([sample_action] * n_envs) if n_envs > 1 else env.step(sample_action)
                    else:
                        obs = env.reset()
                t2 = time.time()
                fps = (steps * max(1, n_envs)) / max(1e-6, (t2 - t1))
                log.info(f"[DIAGNOSTIC] Env step speed ≈ {fps:.1f} steps/sec across {n_envs} env(s).")
        except Exception as e:
            log.warning(f"[DIAGNOSTIC] Env speed diagnostic failed: {e}")

        return env



    def _make_expert_loader(self) -> DataLoader:
        """Factory for the offline expert data loader."""
        try:
            dataset = ExpertTrajectoryDataset(
                demo_path=self.cfg.dataset.path,
                observation_horizon=self.cfg.model.observation_horizon,
                action_horizon=self.cfg.model.action_horizon,
            )
        except Exception as e:
            log.exception(f"Error creating ExpertTrajectoryDataset from {self.cfg.dataset.path}: {e}")
            raise

        return DataLoader(
            dataset,
            batch_size=self.cfg.rl_algorithm.batch_size,
            shuffle=True,
            num_workers=self.cfg.dataset.num_workers,
            collate_fn=collate_fn, # Ensure this handles Dict observations correctly
            pin_memory=self.device.type == 'cuda',
            drop_last=True # Important for consistent batch sizes
        )

    def _build_models_and_optimizers(self, history_obs_space: DictSpace):
        """Initializes actor, critic, target networks, and optimizers."""
        log.info("Building RL models and loading pre-trained actor...")

        # --- Infer proprioception dimension from the expert dataset ---
        proprio_dim = self.expert_loader.dataset.get_proprioception_dim()
        log.info(f"Inferred proprioception dimension from expert dataset: {proprio_dim}")

        # --- Create Diffusion Policy (Actor Core) ---
        model_cfg = self.cfg.model
        policy_kwargs = {
            "proprio_dim": proprio_dim,
            "H_o": model_cfg.observation_horizon,
            "H_a": model_cfg.action_horizon,
            "action_dim": self.action_dim,
            "image_feat_dim": model_cfg.image_feat_dim,
            "scheduler_cfg": NoiseSchedulerConfig(**self.cfg.scheduler),
            "d_model": model_cfg.d_model,
            "denoiser_layers": model_cfg.denoiser_layers,
            "denoiser_heads": model_cfg.denoiser_heads,
            "cfg_p_uncond": 0.0,
            "ema_decay": None,
            "device": self.device
        }
        diffusion_policy = DiffusionPolicy(**policy_kwargs)

        # --- Load Pre-trained Weights ---
        if self.cfg.pretrained_policy_path:
            pretrained_path = Path(self.cfg.pretrained_policy_path)
            if pretrained_path.exists():
                diffusion_policy.load(pretrained_path)
                log.info(f"Successfully loaded pre-trained diffusion policy from {pretrained_path}")
            else:
                log.warning(f"Pretrained policy path not found: {pretrained_path}. Actor may be untrained.")
        else:
            log.warning("No pretrained policy path specified. Actor may be untrained.")
        
        sampling_steps = self.cfg.rl_algorithm.get("sampling_steps", 10)

        # --- Create Actor and Target Actor ---
        self.actor = DiffusionActor(
            diffusion_policy,
            self.cfg.rl_algorithm.guidance_scale,
            sampling_steps
        ).to(self.device)
        target_diffusion_policy = DiffusionPolicy(**policy_kwargs)
        target_diffusion_policy.load_state_dict(diffusion_policy.state_dict())
        self.actor_target = DiffusionActor(
            target_diffusion_policy,
            self.cfg.rl_algorithm.guidance_scale,
            sampling_steps
        ).to(self.device)
        log.info("Actor and Target Actor created.")

        # --- Create Critic and Target Critic ---
        critic_feature_extractor = VisionFusionEncoder(
            image_feat_dim=model_cfg.image_feat_dim,
            proprio_dim=proprio_dim,
            d_model=model_cfg.d_model
        )
        critic_feature_extractor.load_state_dict(
            self.actor.diffusion_policy.vision_fusion_encoder.state_dict()
        )
        # Use new config options for Critic
        self.critic = Critic(
            features_extractor=critic_feature_extractor,
            action_dim=self.action_dim,
            d_model=self.cfg.model.d_model,
            hidden_dims=self.cfg.rl_algorithm.critic_net_arch,
            use_layernorm=self.cfg.rl_algorithm.use_critic_layernorm,
            use_last_feature=self.cfg.rl_algorithm.critic_use_last_feature
        ).to(self.device)

        critic_target_feature_extractor = VisionFusionEncoder(
            image_feat_dim=model_cfg.image_feat_dim,
            proprio_dim=proprio_dim,
            d_model=model_cfg.d_model
        )
        critic_target_feature_extractor.load_state_dict(
            self.actor_target.diffusion_policy.vision_fusion_encoder.state_dict()
        )
        self.critic_target = Critic(
            features_extractor=critic_target_feature_extractor,
            action_dim=self.action_dim,
            d_model=self.cfg.model.d_model,
            hidden_dims=self.cfg.rl_algorithm.critic_net_arch,
            use_layernorm=self.cfg.rl_algorithm.use_critic_layernorm,
            use_last_feature=self.cfg.rl_algorithm.critic_use_last_feature
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        log.info("Critic and Target Critic created with DECOUPLED encoders.")

        # --- Optimizers ---
        opt_cfg = self.cfg.optimizer
        # CORRECT: Actor optimizer updates all actor parameters
        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(),
            lr=opt_cfg.actor_lr,
            weight_decay=opt_cfg.get("actor_weight_decay", 1e-4)
        )
        # CORRECT: Critic optimizer updates ONLY the Q-network parameters
        critic_q_net_params = chain(self.critic.q1_net.parameters(), self.critic.q2_net.parameters())
        self.critic_optimizer = optim.AdamW(
            critic_q_net_params,
            lr=opt_cfg.critic_lr,
            weight_decay=opt_cfg.get("critic_weight_decay", 1e-4)
        )
        log.info("Optimizers created. Critic optimizer targets Q-networks only.")

        # --- Schedulers (Adjusted for Gradient Accumulation) ---
        num_optimizer_steps = self.cfg.training.total_timesteps // self.cfg.training.gradient_accumulation_steps
        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, T_max=num_optimizer_steps)
        self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.critic_optimizer, T_max=num_optimizer_steps)
        log.info("CosineAnnealingLR schedulers created.")

        # --- Optional: torch.compile for PyTorch 2.0+ ---
        if self.cfg.training.use_torch_compile and hasattr(torch, "compile"):
             log.info("Applying torch.compile (mode='reduce-overhead')...")
             try:
                 self.actor = torch.compile(self.actor, mode="reduce-overhead")
                 self.critic = torch.compile(self.critic, mode="reduce-overhead")
                 log.info("torch.compile applied successfully.")
             except Exception as e:
                 log.warning(f"torch.compile failed: {e}. Continuing without compilation.")
  
    def select_action(self, obs_history_batch: Dict[str, np.ndarray]) -> np.ndarray:
        """Selects a batch of actions from the actor, adding exploration noise."""
        obs_torch = {
             k: torch.as_tensor(v, device=self.device).float()
             for k, v in obs_history_batch.items()
        }

        with torch.no_grad():
            # Use the dedicated inference method for clarity and safety
            actions = self.actor.act(obs_torch) # <-- CHANGE HERE

        if self.cfg.rl_algorithm.exploration_noise > 0:
            noise_scale = self.cfg.rl_algorithm.exploration_noise
            noise = torch.randn_like(actions) * noise_scale
            actions = actions + noise
        
        with torch.no_grad():
            actions = actions.clamp(-self.max_action, self.max_action)

        return actions.cpu().numpy()


    def _update_critic(self, obs, next_obs, actions, rewards, dones) -> torch.Tensor:
        """Performs the critic update step."""
        # The feature extractor is NOT in the critic_optimizer, so we don't need to freeze it here.
        # Its gradients are generated during the actor update.

        with torch.no_grad():
            policy_noise_scale = self.cfg.rl_algorithm.policy_noise
            noise_clip = self.cfg.rl_algorithm.noise_clip
            noise = (torch.randn_like(actions) * policy_noise_scale).clamp(-noise_clip, noise_clip)

            next_action = (self.actor_target.act(next_obs) + noise).clamp(-self.max_action, self.max_action)

            target_q1, target_q2 = self.critic_target(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2)

            gamma = self.cfg.rl_algorithm.gamma
            target_q = rewards + (1.0 - dones) * gamma * target_q

        current_q1, current_q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Normalize loss for gradient accumulation
        critic_loss = critic_loss / self.cfg.training.gradient_accumulation_steps
        critic_loss.backward()

        return critic_loss * self.cfg.training.gradient_accumulation_steps

    def _update_actor(self, obs, expert_obs, expert_actions_full, expert_actions_first) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs the actor update step (delayed)."""
        # Freeze critic parameters to avoid unnecessary gradient computation
        for p in self.critic.parameters():
            p.requires_grad = False

        actor_loss_rl = torch.tensor(0.0, device=self.device)
        
        if self.cfg.training.use_reweighted_bc_actor_loss:
            # SOTA: Advantage-weighted BC loss
            with torch.no_grad():
                q_expert_actions = self.critic.Q1(expert_obs, expert_actions_first)
                beta = self.cfg.rl_algorithm.actor_loss_beta
                weights = torch.exp(beta * q_expert_actions).clamp(max=100.0)
                weights = (weights / weights.mean()).detach()

            # NOTE: This assumes your DiffusionPolicy.compute_loss can accept weights.
            # If not, you must modify it to compute a per-sample loss, which you then
            # multiply by weights before taking the mean.
            # For this guide, we assume it's possible or proceed with an un-weighted version if not.
            try:
                bc_loss, _ = self.actor.diffusion_policy.compute_loss(
                    expert_actions_full, expert_obs, weights=weights
                )
            except TypeError:
                log.warning("DiffusionPolicy.compute_loss does not accept `weights`. Using un-weighted BC loss for actor update.")
                bc_loss, _ = self.actor.diffusion_policy.compute_loss(expert_actions_full, expert_obs)

            actor_loss = bc_loss # The total loss is just the weighted BC loss
            alpha = torch.tensor(0.0, device=self.device) # Not used

        else:
            # Original TD3+BC actor loss
            actor_online_actions = self.actor(obs) # Calls forward() with grads
            q1_values_for_actor_loss = self.critic.Q1(obs, actor_online_actions)
            actor_loss_rl = -q1_values_for_actor_loss.mean()

            bc_loss, _ = self.actor.diffusion_policy.compute_loss(expert_actions_full, expert_obs)

            with torch.no_grad():
                q_expert_actions = self.critic.Q1(expert_obs, expert_actions_first)
                alpha_denom = torch.mean(torch.abs(q_expert_actions)).detach() + 1e-6
                alpha = (1.0 / alpha_denom).clamp(0.01, 100.0)
            
            actor_loss = actor_loss_rl + alpha * bc_loss

        # Normalize loss for gradient accumulation and backpropagate
        actor_loss = actor_loss / self.cfg.training.gradient_accumulation_steps
        actor_loss.backward()

        # Unfreeze critic
        for p in self.critic.parameters():
            p.requires_grad = True

        return actor_loss * self.cfg.training.gradient_accumulation_steps, actor_loss_rl, bc_loss

    def train_step(self):
        """Performs a single gradient update step, accumulating gradients if configured."""
        # 1. Sample Data
        replay_data = self.replay_buffer.sample(self.cfg.rl_algorithm.batch_size)
        expert_obs_chunk, expert_action_chunk = next(self.expert_iterator)

        # 2. Prepare Data
        obs = {k: v.to(self.device).float() for k, v in replay_data.observations.items()}
        next_obs = {k: v.to(self.device).float() for k, v in replay_data.next_observations.items()}
        actions = replay_data.actions.to(self.device).float()
        rewards = replay_data.rewards.to(self.device).float()
        dones = replay_data.dones.to(self.device).float()
        expert_obs = {k: v.to(self.device).float() for k, v in expert_obs_chunk.items()}
        expert_actions_full = expert_action_chunk.to(self.device).float()
        expert_actions_first = expert_actions_full[:, 0, :]

        # 3. Update Critic (computes loss and gradients)
        critic_loss = self._update_critic(obs, next_obs, actions, rewards, dones)

        actor_loss, actor_loss_rl, bc_loss = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
        
        # 4. Delayed Actor Update (computes loss and gradients)
        if self.total_timesteps % self.cfg.rl_algorithm.policy_delay == 0:
            actor_loss, actor_loss_rl, bc_loss = self._update_actor(
                obs, expert_obs, expert_actions_full, expert_actions_first
            )

        # 5. Gradient Accumulation & Optimizer Step
        self.grad_accumulation_counter += 1
        if self.grad_accumulation_counter % self.cfg.training.gradient_accumulation_steps == 0:
            # Critic optimizer step
            if self.cfg.optimizer.critic_grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    chain(self.critic.q1_net.parameters(), self.critic.q2_net.parameters()),
                    self.cfg.optimizer.critic_grad_clip_norm
                )
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad(set_to_none=True) # More efficient

            # Actor optimizer step (if actor was updated)
            if self.total_timesteps % self.cfg.rl_algorithm.policy_delay == 0:
                if self.cfg.optimizer.actor_grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.optimizer.actor_grad_clip_norm)
                self.actor_optimizer.step()
                self.actor_optimizer.zero_grad(set_to_none=True)

                # Target Network Updates (Polyak)
                tau = self.cfg.rl_algorithm.tau
                with torch.no_grad():
                    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                        target_param.data.mul_(1.0 - tau)
                        target_param.data.add_(tau * param.data)
                    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.mul_(1.0 - tau)
                        target_param.data.add_(tau * param.data)

                # Step LR schedulers
                self.actor_scheduler.step()
                self.critic_scheduler.step()

        # 6. Logging (only on actual optimizer steps)
        if (self.grad_accumulation_counter % self.cfg.training.gradient_accumulation_steps == 0) and \
           (self.total_timesteps % self.cfg.logging.log_interval_steps == 0):
            
            metrics = {
                "train/critic_loss": critic_loss.item(),
                "train/actor_lr": self.actor_scheduler.get_last_lr()[0],
                "train/critic_lr": self.critic_scheduler.get_last_lr()[0],
            }
            if self.total_timesteps % self.cfg.rl_algorithm.policy_delay == 0:
                 metrics.update({
                    "train/actor_loss_total": actor_loss.item(),
                    "train/actor_loss_rl": actor_loss_rl.item(),
                    "train/bc_loss": bc_loss.item(),
                 })

            if self.use_wandb: wandb.log(metrics, step=self.total_timesteps)
            for k, v in metrics.items(): self.writer.add_scalar(k, v, self.total_timesteps)


    def run(self):
        """The main entry point to start the training process."""
        log.info("Starting RL fine-tuning...")
        # Reset envs and initialize observation history buffer
        try:
            # SB3 VecEnv reset returns observations for all envs
            raw_obs_list = self.env.reset() # This should be Dict[str, np.ndarray]
            # Handle potential discrepancies if reset doesn't return dict directly
            if not isinstance(raw_obs_list, dict):
                 # Attempt to convert if it looks like SB3 format (list of dicts, etc.)
                 if isinstance(raw_obs_list, (list, tuple)) and len(raw_obs_list) == self.n_envs:
                      # Assuming list of dicts, reform into batch dict
                      keys = raw_obs_list[0].keys()
                      raw_obs_list = {k: np.stack([raw_obs_list[i][k] for i in range(self.n_envs)]) for k in keys}
                 else:
                     raise TypeError(f"VecEnv.reset() returned unexpected type: {type(raw_obs_list)}")

            # Initialize history buffer correctly
            for i in range(self.n_envs):
                env_obs = {k: v[i] for k, v in raw_obs_list.items()}
                self.obs_history.reset(i, env_obs)
        except Exception as e:
            log.exception(f"Error during initial environment reset: {e}")
            raise

        # Determine total number of training loops
        # Each loop processes self.n_envs steps
        num_loops = int(self.cfg.training.total_timesteps) // self.n_envs
        timings = {
            "select_action": [],
            "env_step": [],
            "buffer_add_etc": [],
            "train_step": [],
            "total_loop": []
        }
        for loop_idx in tqdm(range(num_loops), desc="Total Timesteps"):
            t_loop_start = time.time()
            current_loop_timestep = loop_idx * self.n_envs

            # --- Action Selection ---
            t0 = time.time()
            if current_loop_timestep < self.cfg.rl_algorithm.learning_starts:
                # Sample random actions before learning starts
                action = np.array([self.env.action_space.sample() for _ in range(self.n_envs)])
            else:
                # Get stacked history for policy inference
                stacked_obs_batch = self.obs_history.get_batch_stacked()
                action = self.select_action(stacked_obs_batch)
            timings["select_action"].append(time.time() - t0) # PROFILING

            # --- Environment Interaction ---
            t0 = time.time() # PROFILING
            # --- Environment Interaction ---
            try:
                next_raw_obs_list, rewards, dones, infos = self.env.step(action)
                # Handle potential discrepancies if step doesn't return dict directly
                if not isinstance(next_raw_obs_list, dict):
                    if isinstance(next_raw_obs_list, (list, tuple)) and len(next_raw_obs_list) == self.n_envs:
                         keys = next_raw_obs_list[0].keys()
                         next_raw_obs_list = {k: np.stack([next_raw_obs_list[i][k] for i in range(self.n_envs)]) for k in keys}
                    else:
                        raise TypeError(f"VecEnv.step() returned unexpected type for observations: {type(next_raw_obs_list)}")

            except Exception as e:
                log.exception(f"Error during environment step: {e}")
                # Decide how to handle env errors: continue, break, etc.
                continue # Skip this step
            timings["env_step"].append(time.time() - t0) # PROFILING

            # --- Buffer and History Management ---
            t0 = time.time() # PROFILING
            try:
                # 1. Get s_t (history *before* this step's observation is added).
                # This is already a batched dictionary of numpy arrays.
                obs_history_t = self.obs_history.get_batch_stacked()

                # 2. Update the history buffer for all environments to get s_{t+1}.
                # The 'infos' dict will contain the "real" next observation for terminal states.
                # SB3's replay buffer uses this `infos` dict to store the correct terminal observation.
                # Therefore, we can safely update our history buffer with the `next_raw_obs_list`
                # and then handle terminal resets separately.
                for i in range(self.n_envs):
                    env_next_obs = {k: v[i] for k, v in next_raw_obs_list.items()}
                    self.obs_history.append(i, env_next_obs)
                
                # Note: For the replay buffer, the "next_obs" is handled internally by SB3
                # using the `infos` array for terminal states. We don't need to manually create `s_{t+1}`.
                # We simply add the current history (`s_t`) and the raw `next_raw_obs_list`. The buffer
                # is smart enough to use `infos[i]["terminal_observation"]` when `dones[i]` is True.
                # However, the sb3-contrib DictReplayBuffer *does* expect the full next_obs dictionary.
                # The logic below is more robust for both buffer types.

                # Let's prepare the next_obs properly for the buffer.
                # It's mostly the updated history, but for dones, it's special.
                # For simplicity and correctness with sb3-contrib, we just pass the *updated* history.
                # The buffer's `handle_timeout_termination=False` ensures it stores what we give it.
                obs_history_t_plus_1 = self.obs_history.get_batch_stacked()

                # 3. Add the entire batch of transitions to the replay buffer in ONE call.
                self.replay_buffer.add(
                    obs=obs_history_t,
                    next_obs=obs_history_t_plus_1,
                    action=action,          # Shape (n_envs, action_dim)
                    reward=rewards,         # Shape (n_envs,)
                    done=dones,             # Shape (n_envs,)
                    infos=infos,            # List of info dicts, length n_envs
                )

                # 4. Handle Episode Terminations (Reset History Buffers for next loop iteration).
                # This must happen *after* adding to the buffer.
                for i in range(self.n_envs):
                    if dones[i]:
                        env_info = infos[i] if isinstance(infos, (list, tuple)) else infos
                        if "terminal_observation" in env_info:
                            terminal_obs = env_info["terminal_observation"]
                            self.obs_history.reset(i, terminal_obs)
                        else:
                            log.warning(f"No 'terminal_observation' in info dict for env {i} on done. History buffer reset might be inaccurate.")
                            last_obs_for_env_i = {k: v[i] for k, v in next_raw_obs_list.items()}
                            self.obs_history.reset(i, last_obs_for_env_i)
            
            except Exception as e:
                log.exception(f"Error processing vectorized step and adding to buffer: {e}")
                continue # Skip this entire batch if an error occurs
            timings["buffer_add_etc"].append(time.time() - t0) # PROFILING

            # --- Update Timestep Counter ---
            # Correctly increment based on number of parallel environments
            self.total_timesteps = (loop_idx + 1) * self.n_envs
            t0 = time.time() # PROFILING
            # --- Training Step ---
            if self.total_timesteps >= self.cfg.rl_algorithm.learning_starts:
                # Perform gradient updates using sampled data
                self.train_step()
            timings["train_step"].append(time.time() - t0)

            # --- Evaluation and Checkpointing ---
            if self.total_timesteps - self.timesteps_since_eval >= self.cfg.logging.eval_freq:
                self.evaluate()
                self._save_checkpoint() # Save checkpoint after evaluation
                self.timesteps_since_eval = self.total_timesteps
            timings["total_loop"].append(time.time() - t_loop_start)
            if (loop_idx + 1) % 1 == 0: # Print stats every 200 loops
                log.info("\n----------- PROFILING STATS (avg ms per loop) -----------")
                for key, val in timings.items():
                    avg_time_ms = np.mean(val) * 1000
                    log.info(f"{key:<20}: {avg_time_ms:.2f} ms")
                log.info("---------------------------------------------------------")
        # --- Final Save ---
        log.info("Training finished. Saving final checkpoint.")
        self._save_checkpoint(is_final=True)
        self.env.close()
        self.eval_env.close()
        self.video_env.close()
        if self.use_wandb:
            wandb.finish()
        self.writer.close()

    def evaluate(self):
        """
        [SOTA: Hybrid Parallel/Sequential Evaluation]
        Gathers statistics by running episodes in parallel for maximum efficiency.
        Then, separately records a single, clean video episode if required.
        """
        log.info(f"Starting evaluation phase at timestep {self.total_timesteps}...")
        self.actor.eval()

        # ===================================================================
        # 1. PARALLEL STATISTICS GATHERING (uses self.eval_env)
        # ===================================================================
        log.info(f"Running {self.cfg.logging.n_eval_episodes} episodes in parallel for statistics...")
        start_eval_time = time.time()
        
        all_ep_rewards = []
        all_successes = []
        episodes_completed = 0
        
        num_eval_envs = self.eval_env.num_envs
        current_rewards = np.zeros(num_eval_envs)

        try:
            raw_obs_batch = self.eval_env.reset()
            for i in range(num_eval_envs):
                self.eval_obs_history.reset(i, {k: v[i] for k, v in raw_obs_batch.items()})
        except Exception as e:
            log.exception("Error resetting parallel evaluation environment. Skipping evaluation.")
            self.actor.train()
            return
            
        while episodes_completed < self.cfg.logging.n_eval_episodes:
            # Get histories for only the active parallel evaluation envs
            stacked_obs_batch = self.eval_obs_history.get_batch_stacked()
            active_obs_batch = {k: v[:num_eval_envs] for k, v in stacked_obs_batch.items()}

            with torch.no_grad():
                actions = self.select_action(active_obs_batch)

            try:
                next_raw_obs_batch, rewards, dones, infos = self.eval_env.step(actions)
            except Exception as e:
                log.exception("Error stepping parallel evaluation environment. Ending stats collection early.")
                break

            for i in range(num_eval_envs):
                if episodes_completed >= self.cfg.logging.n_eval_episodes: break # Early exit if another env finished
                
                self.eval_obs_history.append(i, {k: v[i] for k, v in next_raw_obs_batch.items()})
                current_rewards[i] += rewards[i]

                if dones[i]:
                    episodes_completed += 1
                    all_ep_rewards.append(current_rewards[i])
                    all_successes.append(infos[i].get('is_success', 0.0))
                    
                    if "terminal_observation" in infos[i]:
                        self.eval_obs_history.reset(i, infos[i]["terminal_observation"])
                    current_rewards[i] = 0.0

        stats_duration = time.time() - start_eval_time
        mean_reward = np.mean(all_ep_rewards) if all_ep_rewards else 0.0
        success_rate = np.mean(all_successes) if all_successes else 0.0
        log.info(f"Statistics gathering complete ({stats_duration:.2f}s): Success Rate={success_rate:.3f}, Mean Reward={mean_reward:.2f}")

        metrics = {"eval/mean_reward": mean_reward, "eval/success_rate": success_rate}
        if self.use_wandb: wandb.log(metrics, step=self.total_timesteps)
        for k, v in metrics.items(): self.writer.add_scalar(k, v, self.total_timesteps)

        # ===================================================================
        # 2. SEQUENTIAL VIDEO RECORDING (uses self.video_env)
        # ===================================================================
        current_eval_phase = (self.total_timesteps // self.cfg.logging.eval_freq) if self.cfg.logging.eval_freq > 0 else 0
        if self.use_wandb and self.cfg.logging.video_log_freq > 0 and current_eval_phase % self.cfg.logging.video_log_freq == 0:
            log.info("Starting video recording episode...")
            video_frames = []
            try:
                raw_obs_dict_tuple = self.video_env.reset()
                # DummyVecEnv with n=1 still returns a dict of batched arrays (batch_size=1)
                raw_obs_dict = {k: v[0] for k, v in raw_obs_dict_tuple.items()}
                # Use a single slot (index 0) of the history buffer for the video env
                current_hist_obs = self.eval_obs_history.reset(0, raw_obs_dict)
                done = False
            except Exception as e:
                log.exception("Error resetting video environment. Aborting video recording.")
                done = True

            while not done:
                try:
                    # render() on DummyVecEnv returns a list of frames
                    frame = self.video_env.render()
                    video_frames.append(frame.copy())
                except Exception as e:
                    log.warning(f"Could not render frame for video: {e}")

                stacked_obs_batch = {k: v[np.newaxis, ...] for k, v in current_hist_obs.items()}
                with torch.no_grad():
                    action = self.select_action(stacked_obs_batch)
                
                try:
                    next_raw_obs_list, _, dones, _ = self.video_env.step(action)
                    done = dones[0]
                    env_next_obs = {k: v[0] for k, v in next_raw_obs_list.items()}
                    self.eval_obs_history.append(0, env_next_obs)
                    current_hist_obs = self.eval_obs_history.get_stacked(0)
                except Exception as e:
                    log.exception("Error stepping video environment. Aborting episode.")
                    break
            
            if video_frames:
                video_np = np.stack(video_frames)
                video_np = np.transpose(video_np, (0, 3, 1, 2)) # T, C, H, W for W&B
                wandb.log({"eval/video": wandb.Video(video_np, fps=self.cfg.logging.get("video_fps", 20))}, step=self.total_timesteps)
                log.info("Logged video to W&B.")

        # ===================================================================
        # 3. CHECKPOINTING AND CLEANUP
        # ===================================================================
        if success_rate > self.best_eval_success_rate:
            log.info(f"New best eval success rate: {success_rate:.3f}. Saving best checkpoint.")
            self.best_eval_success_rate = success_rate
            self._save_checkpoint(is_best=True)

        self.actor.train()
        log.info("Evaluation phase finished.")




    # --- Checkpointing Methods ---
    def _save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """Saves a training checkpoint atomically."""
        state = {
            'total_timesteps': self.total_timesteps,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'best_eval_success_rate': self.best_eval_success_rate,
            'np_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'torch_rng_state': torch.get_rng_state(),
            'torch_cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'config_dict': OmegaConf.to_container(self.cfg, resolve=True), # Use dict for saving
        }

        # Save latest checkpoint
        latest_path = self.output_dir / "checkpoints" / "latest_checkpoint.pth"
        try:
            temp_latest_path = latest_path.with_suffix(".pth.tmp")
            torch.save(state, temp_latest_path)
            os.replace(temp_latest_path, latest_path) # Atomic operation
            log.info(f"Saved latest checkpoint to {latest_path} at timestep {self.total_timesteps}")
        except Exception as e:
             log.exception(f"Error saving latest checkpoint: {e}")
             if temp_latest_path.exists(): os.remove(temp_latest_path) # Cleanup temp

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "checkpoints" / "best_checkpoint.pth"
            try:
                shutil.copyfile(latest_path, best_path) # Copy latest if it's the best
                log.info(f"Saved best checkpoint to {best_path}")
            except Exception as e:
                 log.exception(f"Error saving best checkpoint: {e}")

        # Save final checkpoint
        if is_final:
            final_path = self.output_dir / "checkpoints" / "final_checkpoint.pth"
            try:
                 shutil.copyfile(latest_path, final_path)
                 log.info(f"Saved final checkpoint to {final_path}")
            except Exception as e:
                 log.exception(f"Error saving final checkpoint: {e}")


    def _load_checkpoint(self):
        """Loads training state from the last checkpoint if it exists."""
        load_path = self.output_dir / "checkpoints" / "latest_checkpoint.pth"
        if load_path.exists():
            log.info(f"Attempting to load checkpoint from {load_path}")
            try:
                state = torch.load(load_path, map_location=self.device)

                # Load models
                self.actor.load_state_dict(state['actor_state_dict'])
                self.critic.load_state_dict(state['critic_state_dict'])
                self.actor_target.load_state_dict(state['actor_target_state_dict'])
                self.critic_target.load_state_dict(state['critic_target_state_dict'])

                # Load optimizers
                self.actor_optimizer.load_state_dict(state['actor_optimizer_state_dict'])
                self.critic_optimizer.load_state_dict(state['critic_optimizer_state_dict'])

                # Load training progress
                self.total_timesteps = state.get('total_timesteps', 0)
                self.best_eval_success_rate = state.get('best_eval_success_rate', -1.0)
                self.timesteps_since_eval = 0 # Reset eval timer after loading

                # Restore RNG states
                np.random.set_state(state['np_rng_state'])
                random.setstate(state['random_rng_state'])
                torch.set_rng_state(state['torch_rng_state'])
                if torch.cuda.is_available() and state.get('torch_cuda_rng_state'):
                    torch.cuda.set_rng_state_all(state['torch_cuda_rng_state'])

                # Config check (optional but recommended)
                # loaded_cfg_dict = state.get('config_dict')
                # if loaded_cfg_dict:
                #     # Compare loaded config with current cfg - warn on major discrepancies
                #     pass

                log.info(f"Successfully loaded checkpoint. Resuming from timestep {self.total_timesteps}.")

            except Exception as e:
                log.exception(f"Error loading checkpoint from {load_path}. Training will start from scratch.")
                # Ensure state variables are reset
                self.total_timesteps = 0
                self.best_eval_success_rate = -1.0
        else:
            log.info("No checkpoint found at specified path. Starting training from scratch.")


# -------------------------
# 4. Hydra Main Entry Point
# -------------------------
@hydra.main(version_base=None, config_path="../configs", config_name="finetune_rl_config")
def main(cfg: DictConfig):
    # Setup logging (Hydra manages output dir and basic setup)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    log.info("----------- RL Fine-tuning Configuration -----------")
    log.info(OmegaConf.to_yaml(cfg))
    log.info("--------------------------------------------------")
    # Additional check: Ensure observation horizon matches dataset if possible
    # (Requires loading a sample or metadata, maybe add later)

    try:
        trainer = RLFineTuner(cfg)
        trainer.run()
    except Exception as e:
        log.exception("An error occurred during the RL fine-tuning run.")
        # Optionally, re-raise or exit with error code
        # raise e
        sys.exit(1) # Exit with error code on failure

if __name__ == "__main__":
    main()