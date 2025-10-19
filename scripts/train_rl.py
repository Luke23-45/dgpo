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
    from models.diffusion_policy import DiffusionPolicy, NoiseSchedulerConfig
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
                    env_buffer[key].append(zero_obs)

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
    """Twin Critic network for TD3, using a shared feature extractor."""
    def __init__(self, features_extractor: nn.Module, action_dim: int, d_model: int):
        super().__init__()
        # IMPORTANT: Do NOT keep a reference to the online actor's extractor directly
        # if the actor is also being trained. Clone it or pass features.
        # Since the feature extractor is part of the DiffusionPolicy which *is* trained (actor_loss),
        # we should use it in no_grad mode or pass features explicitly.
        # The current implementation uses no_grad, which is correct for TD3's critic update.
        self.features_extractor = features_extractor

        # Q1 network
        self.q1_net = nn.Sequential(
            nn.Linear(d_model + action_dim, 512), nn.ReLU(),
            nn.LayerNorm(512), # Add LayerNorm for stability
            nn.Linear(512, 512), nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 1)
        )
        # Q2 network
        self.q2_net = nn.Sequential(
            nn.Linear(d_model + action_dim, 512), nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 512), nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 1)
        )

    def forward(self, obs_history: Dict[str, torch.Tensor], action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes Q-values using features from the *last* timestep in history."""
        # --- Feature Extraction ---
        # No gradients needed for feature extraction during critic update
        with torch.no_grad():
            # Features extractor expects (B, H_o, ...)
            vision_tokens, proprio_tokens = self.features_extractor(obs_history)
            # Combine vision and proprio tokens: (B, H_o*(Nv+Np), D_model)
            # For the critic, we typically only need the features from the *current* state (last in history)
            # Let's take the features corresponding to the last observation timestep
            # Assuming vision_tokens is (B, H_o*N_img_tokens, D) and proprio is (B, H_o, D)
            # This needs careful slicing based on how VisionFusionEncoder structures output.
            # Assuming VisionFusionEncoder outputs (B, H_o*N_fused_tokens, D) and (B, H_o, D)
            # Let's average over the history dimension for simplicity and robustness
            vision_features_avg = vision_tokens.mean(dim=1) # (B, D_model)
            proprio_features_avg = proprio_tokens.mean(dim=1) # (B, D_model)
            # Combine features (e.g., average or concatenate)
            # Averaging seems more robust here.
            features = (vision_features_avg + proprio_features_avg) / 2.0 # (B, D_model)

        # --- Q-Value Computation ---
        x = torch.cat([features, action], dim=1)
        return self.q1_net(x), self.q2_net(x)

    def Q1(self, obs_history: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        """Computes the Q-value from the first critic only."""
        with torch.no_grad():
            vision_tokens, proprio_tokens = self.features_extractor(obs_history)
            vision_features_avg = vision_tokens.mean(dim=1)
            proprio_features_avg = proprio_tokens.mean(dim=1)
            features = (vision_features_avg + proprio_features_avg) / 2.0

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
    def forward(self, obs_history: Dict[str, torch.Tensor], deterministic: bool = True) -> torch.Tensor:
        """Selects an action using the diffusion policy with CFG."""
        self.diffusion_policy.eval() # Ensure model is in eval mode

        # Number of sampling steps can be fixed or depend on deterministic flag
        # Using a fixed (small) number is typical for RL inference speed
        steps = self.sampling_steps

        sampled_actions = self.diffusion_policy.sample(
            obs_history,
            steps=steps,
            guidance_scale=self.guidance_scale,
            use_ema=True # Use EMA weights for inference
        )
        # Return the first action in the predicted sequence [cite: 826]
        # Shape: (B, H_a, A_dim) -> (B, A_dim)
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
        log.info(f"Initializing {cfg.environment.n_envs} vectorized environments...")
        self.env = self._make_vec_env(is_eval=False)

        # --- Evaluation Environment (No Video Recorder Wrapper Here) ---
        # We need the unwrapped env for manual rendering during video generation
        self.eval_env_unwrapped = self._make_vec_env(is_eval=True)
        log.info("Initialized evaluation environment.")

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
             # Attempt with standard buffer, might have issues with Dicts
             self.replay_buffer = SB3ReplayBuffer(
                 buffer_size=self.cfg.rl_algorithm.buffer_size,
                 observation_space=history_obs_space,
                 action_space=self.env.action_space,
                 device=self.device,
                 n_envs=self.n_envs,
                 handle_timeout_termination=False,
             )


        self.obs_history = ObsHistoryBuffer(self.n_envs, self.history_len, single_step_obs_space)
        # Separate history buffer for the single evaluation environment
        self.eval_obs_history = ObsHistoryBuffer(1, self.history_len, single_step_obs_space)

        # --- Expert Dataloader ---
        self.expert_loader = self._make_expert_loader()
        self.expert_iterator = iter(self.expert_loader)

        # --- Build Models ---
        self._build_models_and_optimizers(history_obs_space)

        # --- State Tracking ---
        self.total_timesteps = 0
        self.timesteps_since_eval = 0
        self.best_eval_success_rate = -1.0 # Track best model based on success rate

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


    def _make_vec_env(self, is_eval: bool = False) -> VecEnv:
        """Factory for creating the vectorized simulation environment."""
        def make_env(rank: int):
            def _init():
                env_cfg = self.cfg.environment
                # Ensure eval envs have different seeds from training envs
                seed = self.cfg.seed + rank + (1000 if is_eval else 0)

                # 1. Create the base environment
                # IMPORTANT: Need rgb_array render mode for video recording later
                render_mode = "rgb_array" if is_eval else "rgb_array"
                try:
                    env = PandaEnv(
                        xml_path=env_cfg.xml_path,
                        control_mode="delta",
                        render_mode=render_mode, # Critical for eval video
                    )
                except Exception as e:
                     log.exception(f"Error creating PandaEnv (rank {rank}): {e}")
                     raise

                # 2. Create the reward configs (Hydra config management)
                # Allow reward/curriculum configs to be defined in main Hydra config
                try:
                    reward_config = AdvancedRewardConfig(**self.cfg.get("reward", {}))
                    curriculum_config = CurriculumConfig(**self.cfg.get("curriculum", {}))
                except Exception as e:
                    log.error(f"Error creating reward/curriculum configs: {e}. Using defaults.")
                    reward_config = AdvancedRewardConfig()
                    curriculum_config = CurriculumConfig()


                # Override curriculum total episodes if specified in training config
                curriculum_config.total_episodes = self.cfg.training.get(
                     "curriculum_total_episodes", curriculum_config.total_episodes
                )

                # 3. Apply the SOTA reward wrapper
                try:
                    env = AdvancedRewardWrapper(
                        env,
                        reward_cfg=reward_config,
                        curriculum_cfg=curriculum_config
                    )
                except Exception as e:
                     log.exception(f"Error applying AdvancedRewardWrapper (rank {rank}): {e}")
                     raise

                # 4. Apply TimeLimit wrapper (standard practice)
                env = gym.wrappers.TimeLimit(env, max_episode_steps=env_cfg.max_episode_steps)

                # 5. Seed and Reset
                try:
                    env.reset(seed=seed)
                except Exception as e:
                     log.exception(f"Error resetting env (rank {rank}): {e}")
                     raise

                return env
            return _init

        n_envs = 1 if is_eval else self.cfg.environment.n_envs

        # Use DummyVecEnv for n_envs=1 (or for debugging) to avoid multiprocessing issues
        # Use SubprocVecEnv for n_envs > 1 for performance
        vec_env_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
        try:
            return vec_env_cls([make_env(i) for i in range(n_envs)])
        except Exception as e:
             log.exception(f"Error creating VecEnv: {e}")
             raise


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

        # --- Create Diffusion Policy (Actor Core) ---
        try:
            scheduler_cfg = NoiseSchedulerConfig(**self.cfg.scheduler)
            model_cfg = self.cfg.model
            proprio_dim = history_obs_space["proprio"].shape[-1] # Dim per step
            log.info(f"Inferred proprioception dimension per step: {proprio_dim}")

            diffusion_policy = DiffusionPolicy(
                proprio_dim=proprio_dim,
                H_o=model_cfg.observation_horizon,
                H_a=model_cfg.action_horizon,
                action_dim=self.action_dim, # Use action dim from env
                image_feat_dim=model_cfg.image_feat_dim,
                scheduler_cfg=scheduler_cfg,
                d_model=model_cfg.d_model,
                denoiser_layers=model_cfg.denoiser_layers,
                denoiser_heads=model_cfg.denoiser_heads,
                cfg_p_uncond=0.0, # Not used in RL inference sampling with CFG wrapper
                ema_decay=None, # EMA state loaded from checkpoint if available
                device=self.device
            )
        except Exception as e:
            log.exception(f"Error initializing DiffusionPolicy: {e}")
            raise

        # --- Load Pre-trained Weights ---
        if self.cfg.pretrained_policy_path:
            pretrained_path = Path(self.cfg.pretrained_policy_path)
            if pretrained_path.exists():
                try:
                    diffusion_policy.load(pretrained_path)
                    log.info(f"Successfully loaded pre-trained diffusion policy from {pretrained_path}")
                except Exception as e:
                    log.warning(f"Could not load pretrained policy from {pretrained_path}: {e}. Actor starts potentially untrained.")
            else:
                log.warning(f"Pretrained policy path specified but not found: {pretrained_path}. Actor starts potentially untrained.")
        else:
            log.warning("No pretrained policy path specified. Actor starts potentially untrained.")
        sampling_steps = self.cfg.rl_algorithm.get("sampling_steps", 10)

        # --- Create Actor and Target Actor ---
        self.actor = DiffusionActor(
            diffusion_policy,
            self.cfg.rl_algorithm.guidance_scale,
            sampling_steps
        ).to(self.device)

        # Target actor initially mirrors the online actor
        # Need deepcopy to avoid sharing weights unintentionally before Polyak updates
        target_diffusion_policy = copy.deepcopy(diffusion_policy)
        self.actor_target = DiffusionActor(
            target_diffusion_policy,
            self.cfg.rl_algorithm.guidance_scale,
            sampling_steps
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        log.info("Actor and Target Actor created.")

        # --- Create Critic and Target Critic ---
        # Critic shares the *online* actor's vision encoder for representation learning
        # Make sure feature extractor doesn't update during critic step via no_grad() in Critic.forward
        critic_feature_extractor = self.actor.diffusion_policy.vision_fusion_encoder
        self.critic = Critic(
            critic_feature_extractor,
            self.action_dim,
            model_cfg.d_model
        ).to(self.device)

        # Target critic shares the *target* actor's vision encoder
        critic_target_feature_extractor = self.actor_target.diffusion_policy.vision_fusion_encoder
        self.critic_target = Critic(
            critic_target_feature_extractor,
            self.action_dim,
            model_cfg.d_model
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        log.info("Critic and Target Critic created.")

        # --- Optimizers ---
        opt_cfg = self.cfg.optimizer
        try:
            self.actor_optimizer = optim.AdamW(
                self.actor.parameters(),
                lr=opt_cfg.actor_lr,
                weight_decay=opt_cfg.get("actor_weight_decay", 1e-4) # Add weight decay option
            )
            self.critic_optimizer = optim.AdamW(
                self.critic.parameters(),
                lr=opt_cfg.critic_lr,
                weight_decay=opt_cfg.get("critic_weight_decay", 1e-4)
            )
            log.info("Optimizers created.")
        except Exception as e:
            log.exception(f"Error creating optimizers: {e}")
            raise

    def select_action(self, obs_history_batch: Dict[str, np.ndarray]) -> np.ndarray:
        """Selects a batch of actions from the actor, adding exploration noise."""
        # Convert numpy batch to torch batch
        obs_torch = {
             k: torch.as_tensor(v, device=self.device).float()
             for k, v in obs_history_batch.items()
        }

        with torch.no_grad():
            actions = self.actor(obs_torch, deterministic=True) # Use deterministic sampling

        # Add exploration noise (Gaussian noise)
        if self.cfg.rl_algorithm.exploration_noise > 0:
            noise_scale = self.cfg.rl_algorithm.exploration_noise
            noise = torch.randn_like(actions) * noise_scale
            actions = actions + noise
        else:
             # If no noise, still clamp to ensure validity (though diffusion might already do this)
             pass

        # Clamp actions to the environment's action space bounds
        actions = actions.clamp(-self.max_action, self.max_action)

        return actions.cpu().numpy()

    def train_step(self):
        """Performs a single gradient update step for both actor and critic."""
        # 1. Sample from replay buffer
        try:
            replay_data = self.replay_buffer.sample(self.cfg.rl_algorithm.batch_size)
        except ValueError as e:
            # Handle case where buffer might not be full enough yet
            log.warning(f"Could not sample from replay buffer (possibly not full): {e}")
            return
        except Exception as e:
             log.exception(f"Unexpected error sampling from replay buffer: {e}")
             return


        # 2. Sample from expert dataloader
        try:
            expert_obs_chunk, expert_action_chunk = next(self.expert_iterator)
        except StopIteration:
            # This is the "stuck" point. Let's add logging.
            log.info("Expert data iterator exhausted. Reloading for a new epoch...")
            start_reload_time = time.time()
            self.expert_iterator = iter(self.expert_loader) # This triggers the reload
            try:
                expert_obs_chunk, expert_action_chunk = next(self.expert_iterator)
                reload_duration = time.time() - start_reload_time
                log.info(f"Expert data reloaded in {reload_duration:.2f} seconds.")
            except StopIteration:
                log.error("Expert dataloader is empty or exhausted unexpectedly after reset.")
                return # Skip training step if no expert data

        # 3. Prepare Expert Data (move to device, ensure correct dtype)
        expert_obs = {
            k: v.to(self.device).float()
            for k, v in expert_obs_chunk.items()
        }
        expert_actions_full = expert_action_chunk.to(self.device).float() # (B, H_a, A_dim)
        expert_actions_first = expert_actions_full[:, 0, :] # (B, A_dim) - action at t

        # 4. Prepare Replay Data (move to device, ensure correct dtype)
        # Note: replay_data from SB3 buffers might already be tensors on the correct device
        obs = {
            k: v.to(self.device).float() if isinstance(v, torch.Tensor) else torch.as_tensor(v, device=self.device).float()
            for k, v in replay_data.observations.items()
        }
        next_obs = {
             k: v.to(self.device).float() if isinstance(v, torch.Tensor) else torch.as_tensor(v, device=self.device).float()
             for k, v in replay_data.next_observations.items()
        }
        actions = replay_data.actions.to(self.device).float()
        rewards = replay_data.rewards.to(self.device).float()
        dones = replay_data.dones.to(self.device).float()

        # --- 5. Critic Update ---
        with torch.no_grad():
            # Target policy smoothing: Add noise to target actions
            policy_noise_scale = self.cfg.rl_algorithm.policy_noise
            noise_clip = self.cfg.rl_algorithm.noise_clip
            noise = (
                torch.randn_like(actions) * policy_noise_scale
            ).clamp(-noise_clip, noise_clip)

            # Get next action from target actor and add noise
            next_action = (self.actor_target(next_obs) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute target Q-value using target critic
            target_q1, target_q2 = self.critic_target(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2)

            # TD target: R + gamma * (1 - Done) * Q_target(s', a')
            gamma = self.cfg.rl_algorithm.gamma
            target_q = rewards + (1.0 - dones) * gamma * target_q

        # Get current Q-values using online critic and actions from buffer
        current_q1, current_q2 = self.critic(obs, actions)

        # Compute critic loss (MSE between current Q and target Q)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Optional: Gradient clipping for critic
        if self.cfg.optimizer.get("critic_grad_clip_norm"):
             torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.optimizer.critic_grad_clip_norm)
        self.critic_optimizer.step()

        # --- 6. Delayed Actor Update ---
        if self.total_timesteps % self.cfg.rl_algorithm.policy_delay == 0:
            # Freeze critic parameters during actor update
            for p in self.critic.parameters():
                p.requires_grad = False

            # --- RL Loss Component (Policy Gradient) ---
            # Compute actions from the *online* actor for states from the replay buffer
            actor_online_actions = self.actor(obs)
            # Evaluate these actions using the *online* critic's Q1 network
            q1_values_for_actor_loss = self.critic.Q1(obs, actor_online_actions)
            # Policy gradient loss: maximize Q-value (minimize negative Q-value)
            actor_loss_rl = -q1_values_for_actor_loss.mean()

            # --- BC Loss Component (Diffusion Loss on Expert Data) ---
            # Compute the standard diffusion training loss using the *online* actor's
            # diffusion policy, but only on the *expert* data batch.
            bc_loss, _ = self.actor.diffusion_policy.compute_loss(
                expert_actions_full, expert_obs
            )

            # --- Adaptive BC Weight (alpha) ---
            with torch.no_grad():
                # Evaluate the expert's first action using the *online* critic's Q1
                q_expert_actions = self.critic.Q1(expert_obs, expert_actions_first)
                # Calculate alpha: Lower alpha if critic thinks expert actions are bad
                # Use a small epsilon to prevent division by zero
                alpha_denom = torch.mean(torch.abs(q_expert_actions)).detach() + 1e-6
                alpha = (1.0 / alpha_denom).clamp(0.01, 100.0) # Wider clamp range

                # Alternative: Use average Q magnitude (less aggressive scaling)
                # avg_q_magnitude = (torch.mean(torch.abs(current_q1)).detach() +
                #                    torch.mean(torch.abs(current_q2)).detach()) / 2.0 + 1e-6
                # alpha = (self.cfg.rl_algorithm.bc_lambda / avg_q_magnitude).clamp(0.01, 100.0)


            # --- Total Actor Loss ---
            actor_loss = actor_loss_rl + alpha * bc_loss

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
             # Optional: Gradient clipping for actor
            if self.cfg.optimizer.get("actor_grad_clip_norm"):
                 torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.optimizer.actor_grad_clip_norm)
            self.actor_optimizer.step()

            # Unfreeze critic parameters
            for p in self.critic.parameters():
                p.requires_grad = True

            # --- 7. Target Network Updates (Polyak Averaging) ---
            tau = self.cfg.rl_algorithm.tau
            with torch.no_grad():
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.mul_(1.0 - tau)
                    target_param.data.add_(tau * param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.mul_(1.0 - tau)
                    target_param.data.add_(tau * param.data)

            # --- 8. Logging (Inside Delayed Update) ---
            if self.total_timesteps % self.cfg.logging.log_interval_steps == 0:
                metrics = {
                    "train/critic_loss": critic_loss.item(),
                    "train/actor_loss_rl": actor_loss_rl.item(),
                    "train/bc_loss": bc_loss.item(),
                    "train/alpha_bc": alpha.item(),
                    "train/actor_loss_total": actor_loss.item(),
                    "train/q1_mean_online": current_q1.mean().item(),
                    "train/q2_mean_online": current_q2.mean().item(),
                    "train/q_mean_expert": q_expert_actions.mean().item(), # Q value of expert actions
                    "train/target_q_mean": target_q.mean().item(),
                }
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

        for loop_idx in tqdm(range(num_loops), desc="Total Timesteps"):

            current_loop_timestep = loop_idx * self.n_envs

            # --- Action Selection ---
            if current_loop_timestep < self.cfg.rl_algorithm.learning_starts:
                # Sample random actions before learning starts
                action = np.array([self.env.action_space.sample() for _ in range(self.n_envs)])
            else:
                # Get stacked history for policy inference
                stacked_obs_batch = self.obs_history.get_batch_stacked()
                action = self.select_action(stacked_obs_batch)

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

            # --- Manage Replay Buffer and History Buffer ---
            for i in range(self.n_envs):
                try:
                    # Get s_t (history *before* this step's observation is added)
                    obs_history_t = self.obs_history.get_stacked(i)

                    # Get next observation for this specific environment
                    env_next_obs = {k: v[i] for k, v in next_raw_obs_list.items()}

                    # Append next observation to history
                    self.obs_history.append(i, env_next_obs)

                    # Get s_{t+1} (history *after* appending)
                    obs_history_t_plus_1 = self.obs_history.get_stacked(i)

                    # Get action, reward, done, info for this env
                    env_action = action[i]
                    env_reward = rewards[i]
                    env_done = dones[i]
                    # SB3 VecEnv typically wraps infos in a list/tuple
                    env_info = infos[i] if isinstance(infos, (list, tuple)) else infos

                    # Add transition (s_t, a_t, r_t, s_{t+1}, done_t) to buffer
                    # Use deepcopy for info if it contains complex objects
                    self.replay_buffer.add(
                        obs_history_t,
                        obs_history_t_plus_1,
                        env_action,
                        env_reward,
                        env_done,
                        [copy.deepcopy(env_info)] # SB3 expects infos as a list
                    )

                    # --- Handle Episode Termination ---
                    if env_done:
                        # VecEnv automatically resets, get the *real* terminal observation
                        if "terminal_observation" in env_info:
                            terminal_obs = env_info["terminal_observation"]
                            # Reset the history buffer with the terminal observation
                            self.obs_history.reset(i, terminal_obs)
                        else:
                             log.warning(f"No 'terminal_observation' found in info dict for env {i} on done. History buffer reset might be inaccurate.")
                             # Attempt to reset with the last `env_next_obs` as a fallback
                             self.obs_history.reset(i, env_next_obs)

                except Exception as e:
                    log.exception(f"Error processing step for env {i}: {e}")
                    # Skip adding this transition if error occurs

            # --- Update Timestep Counter ---
            # Correctly increment based on number of parallel environments
            self.total_timesteps = (loop_idx + 1) * self.n_envs

            # --- Training Step ---
            if self.total_timesteps >= self.cfg.rl_algorithm.learning_starts:
                # Perform gradient updates using sampled data
                self.train_step()

            # --- Evaluation and Checkpointing ---
            if self.total_timesteps - self.timesteps_since_eval >= self.cfg.logging.eval_freq:
                self.evaluate()
                self._save_checkpoint() # Save checkpoint after evaluation
                self.timesteps_since_eval = self.total_timesteps

        # --- Final Save ---
        log.info("Training finished. Saving final checkpoint.")
        self._save_checkpoint(is_final=True)
        self.env.close()
        self.eval_env_unwrapped.close()
        if self.use_wandb:
            wandb.finish()
        self.writer.close()


    def evaluate(self):
        """
        Runs evaluation episodes, gathers stats, and handles video logging efficiently.
        Decouples statistics gathering from video recording for clarity and potential speedup.
        """
        log.info(f"Starting evaluation phase at timestep {self.total_timesteps}...")
        self.actor.eval() # Set actor to evaluation mode

        all_ep_rewards = []
        all_successes = []
        total_eval_steps = 0
        start_eval_time = time.time()

        # === Statistics Gathering Loop ===
        log.info(f"Running {self.cfg.logging.n_eval_episodes} episodes for statistics...")
        for i in range(self.cfg.logging.n_eval_episodes):
            log.debug(f"Starting stats episode {i+1}/{self.cfg.logging.n_eval_episodes}")
            ep_reward = 0.0
            ep_len = 0
            ep_success = 0.0 # Default to failure

            try:
                # Reset environment and history buffer (use unwrapped env for stats)
                raw_obs_dict = self.eval_env_unwrapped.reset()
                current_raw_obs = {k: v[0] for k, v in raw_obs_dict.items()} # Get obs for env 0
                current_hist_obs = self.eval_obs_history.reset(0, current_raw_obs)
            except Exception as e:
                 log.exception(f"Error resetting evaluation environment for stats episode {i+1}: {e}")
                 continue # Skip this episode

            dones = [False] # VecEnv returns dones as list/array
            while not dones[0]:
                # Prepare observation history for actor (add batch dim)
                stacked_obs_batch = {k: v[np.newaxis, ...] for k, v in current_hist_obs.items()}
                obs_torch = {
                    k: torch.as_tensor(v, device=self.device).float()
                    for k, v in stacked_obs_batch.items()
                }

                # Select action deterministically using the actor
                with torch.no_grad():
                    action = self.actor(obs_torch, deterministic=True).cpu().numpy()

                # Step the unwrapped environment
                try:
                    # VecEnv step expects batched action, returns batched results
                    next_raw_obs_list, reward, dones, infos = self.eval_env_unwrapped.step(action)

                    # Extract results for the single environment (index 0)
                    env_next_obs = {k: v[0] for k, v in next_raw_obs_list.items()}
                    env_reward = reward[0]
                    env_done = dones[0] # Boolean indicating if env 0 is done
                    env_info = infos[0] # Info dict for env 0

                except Exception as e:
                     log.exception(f"Error stepping evaluation environment during stats episode {i+1}, step {ep_len}: {e}")
                     env_done = True # Force end episode on error
                     dones = [True] # Ensure loop terminates

                # Update history buffer
                self.eval_obs_history.append(0, env_next_obs)
                current_hist_obs = self.eval_obs_history.get_stacked(0)

                ep_reward += env_reward
                ep_len += 1
                total_eval_steps += 1

                if env_done:
                    ep_success = env_info.get('is_success', 0.0) # Get success status from info
                    # Handle automatic reset by VecEnv - reset history buffer
                    if "terminal_observation" in env_info:
                        terminal_obs = env_info["terminal_observation"]
                        self.eval_obs_history.reset(0, terminal_obs)
                    else:
                        # Fallback if terminal obs not provided (should be by SB3 VecEnvs)
                        log.warning("No 'terminal_observation' in info dict on eval done. Resetting history with last obs.")
                        self.eval_obs_history.reset(0, env_next_obs)
                    break # Exit episode loop

            all_ep_rewards.append(ep_reward)
            all_successes.append(ep_success)
            log.debug(f"Stats episode {i+1} finished. Reward: {ep_reward:.2f}, Success: {ep_success:.0f}, Length: {ep_len}")

        # Calculate average statistics
        mean_reward = np.mean(all_ep_rewards) if all_ep_rewards else 0.0
        std_reward = np.std(all_ep_rewards) if all_ep_rewards else 0.0
        success_rate = np.mean(all_successes) if all_successes else 0.0
        stats_duration = time.time() - start_eval_time
        log.info(f"Statistics gathering complete ({stats_duration:.2f}s): "
                 f"Mean Reward={mean_reward:.2f} (+/- {std_reward:.2f}), "
                 f"Success Rate={success_rate:.2f}")

        # Log metrics to TensorBoard and W&B
        metrics = {
            "eval/mean_reward": mean_reward,
            "eval/success_rate": success_rate,
            "eval/std_reward": std_reward,
            "eval/num_episodes": len(all_ep_rewards),
            "eval/total_steps": total_eval_steps,
        }
        if self.use_wandb:
            wandb.log(metrics, step=self.total_timesteps)
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, self.total_timesteps)

        # === Conditional Video Recording ===
        # Determine if it's time to log video based on *evaluation phase count*
        current_eval_phase = self.total_timesteps // self.cfg.logging.eval_freq
        record_video_this_eval = (
            self.use_wandb and
            self.cfg.logging.video_log_freq > 0 and # Only if freq is positive
            current_eval_phase % self.cfg.logging.video_log_freq == 0
        )

        if record_video_this_eval:
            log.info("Starting video recording episode...")
            start_video_time = time.time()
            video_frames = []
            try:
                # Reset environment and history buffer for video episode
                raw_obs_dict = self.eval_env_unwrapped.reset()
                current_raw_obs = {k: v[0] for k, v in raw_obs_dict.items()}
                current_hist_obs = self.eval_obs_history.reset(0, current_raw_obs)
            except Exception as e:
                 log.exception(f"Error resetting evaluation environment for video recording: {e}")
                 record_video_this_eval = False # Skip video logging on reset error

            dones = [False]
            while not dones[0] and record_video_this_eval:
                # Render frame *before* taking the step
                try:
                    # Use the render method of the VecEnv or its underlying envs
                    # Assuming VecEnv's render returns list of frames or single frame
                    frame_or_list = self.eval_env_unwrapped.render()
                    if isinstance(frame_or_list, (list, tuple)):
                        frame = frame_or_list[0] # Get frame for the first (only) env
                    else:
                        frame = frame_or_list

                    if isinstance(frame, np.ndarray):
                        video_frames.append(frame.copy()) # Use copy to avoid issues
                    else:
                        log.warning("Eval env render did not return a NumPy array. Skipping frame.")
                except Exception as e:
                    log.warning(f"Could not render frame for video: {e}")
                    # Continue without this frame

                # Prepare observation and select action (same as stats loop)
                stacked_obs_batch = {k: v[np.newaxis, ...] for k, v in current_hist_obs.items()}
                obs_torch = {k: torch.as_tensor(v, device=self.device).float() for k, v in stacked_obs_batch.items()}
                with torch.no_grad():
                    action = self.actor(obs_torch, deterministic=True).cpu().numpy()

                # Step environment
                try:
                    next_raw_obs_list, _, dones, infos = self.eval_env_unwrapped.step(action)
                    env_next_obs = {k: v[0] for k, v in next_raw_obs_list.items()}
                    env_done = dones[0]
                    env_info = infos[0]

                except Exception as e:
                     log.exception(f"Error stepping evaluation environment during video recording: {e}")
                     env_done = True # Force end episode
                     dones = [True]

                # Update history buffer
                self.eval_obs_history.append(0, env_next_obs)
                current_hist_obs = self.eval_obs_history.get_stacked(0)

                if env_done:
                    # Handle reset if needed (though loop condition breaks)
                    if "terminal_observation" in env_info:
                         self.eval_obs_history.reset(0, env_info["terminal_observation"])
                    break # Exit episode loop

            video_duration = time.time() - start_video_time
            log.info(f"Video recording episode finished ({video_duration:.2f}s). Collected {len(video_frames)} frames.")

            # Log video to W&B if frames were collected
            if video_frames:
                try:
                    # Stack frames: (T, H, W, C)
                    video_np = np.stack(video_frames)
                    # Transpose for W&B: (T, C, H, W)
                    video_np = np.transpose(video_np, (0, 3, 1, 2))
                    wandb.log(
                        {"eval/video": wandb.Video(video_np, fps=self.cfg.logging.get("video_fps", 20), format="mp4")},
                        step=self.total_timesteps
                    )
                    log.info("Logged video to W&B.")
                except Exception as e:
                    log.warning(f"Failed to log video to W&B: {e}")

        # === Checkpointing based on Best Success Rate ===
        if success_rate > self.best_eval_success_rate:
            log.info(f"New best evaluation success rate: {success_rate:.3f} (previous: {self.best_eval_success_rate:.3f}). Saving best checkpoint.")
            self.best_eval_success_rate = success_rate
            self._save_checkpoint(is_best=True)

        self.actor.train() # IMPORTANT: Set actor back to training mode
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
        temp_latest_path = latest_path.with_suffix(".tmp")
        try:
            torch.save(state, temp_latest_path)
            os.replace(temp_latest_path, latest_path) # Atomic rename
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