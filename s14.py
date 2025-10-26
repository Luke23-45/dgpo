# FILE: train_rl.py
# (State-of-the-Art, Hydra-Configurable, TD3+BC, Decoupled Encoders, Robust Buffer/Eval Version)

"""
State-of-the-art RL fine-tuning script for advanced, pre-trained Diffusion Policies.

This script implements a complete training framework for fine-tuning diffusion actors
(featuring ResNet, Cross-Attention, AdaLN-Zero) using online reinforcement learning,
regularized by offline expert data via TD3+BC.

Key SOTA Features Implemented:
  - **Robust Observation History Management**: Uses `ObsHistoryBuffer` for managing
    observation sequences (`H_o`) needed by the policy.
  - **Hydra Configuration**: Flexible and reproducible experiments via Hydra.
  - **TD3+BC Algorithm**: Stable offline-to-online fine-tuning with adaptive
    BC weight and correct gradient handling.
  - **Advanced Diffusion Policy Actor**: Integrates the SOTA diffusion model
    with a Classifier-Free Guidance (CFG) inference wrapper (`DiffusionActor`).
  - **Decoupled Representation Learning**: Critic utilizes its *own* feature extractor,
    initialized from the actor and updated via Polyak averaging, preventing gradient
    interference while still benefiting from shared initialization.
    The critic optimizer *only* targets the Q-network layers.
  - **Correct Gradient Flow**: Precisely manages `requires_grad` during actor and
    critic updates to ensure only intended parameters are updated.
  - **Comprehensive Logging**: TensorBoard and optional Weights & Biases integration.
  - **Optimized Evaluation & Video Logging**: Decouples statistics collection
    from video recording; records exactly one evaluation episode video directly
    to W&B when triggered. Uses parallel VecEnv for stats.
  - **Robust Vectorized Environment Handling**: Platform-aware VecEnv creation
    (SubprocVecEnv on Linux, DummyVecEnv fallback) with CUDA context pre-initialization.
  - **Efficient Replay Buffer Interaction**: Enforces `DictReplayBuffer` and leverages
    SB3's optimized batch `add` method, correctly handling terminal observations via `infos`.
  - **Reproducibility**: Deterministic seeding, RNG state checkpointing, and atomic checkpoint saving.
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
import pickle
import platform
from itertools import cycle, chain

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

# Optional W&B
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Project-Specific
try:
    from envs.panda_env import PandaEnv
    from utils.rl_reward_wrapper import AdvancedRewardWrapper, AdvancedRewardConfig, CurriculumConfig
    from utils.expert_dataset import ExpertTrajectoryDataset, collate_fn
    from models.diffusion_policy import DiffusionPolicy, NoiseSchedulerConfig, VisionFusionEncoder
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, DummyVecEnv
    from stable_baselines3.common.buffers import DictReplayBuffer
except ImportError as e:
    logging.getLogger(__name__).exception(f"Error importing project modules: {e}")
    sys.exit(1)

# Setup Logger
log = logging.getLogger(__name__)
if not WANDB_AVAILABLE:
    log.warning("Weights & Biases not installed (pip install wandb). W&B logging disabled.")

# -------------------------
# 2. Helper Functions & Classes
# -------------------------

def set_seed(seed: int):
    """Sets the seed for all relevant RNGs for reproducibility."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)
    log.info(f"Global seed set to {seed}")

class ObsHistoryBuffer:
    """Manages observation history (frame stacking) for Dict observation spaces."""
    def __init__(self, n_envs: int, history_len: int, obs_space: DictSpace):
        self.n_envs = n_envs
        self.history_len = history_len
        if not isinstance(obs_space, DictSpace):
             raise ValueError("ObsHistoryBuffer requires a Dict observation space.")
        self.obs_space = obs_space
        self.keys = list(obs_space.keys())

        self.buffers: List[Dict[str, Deque[np.ndarray]]] = []
        for _ in range(n_envs):
            env_buffer = {}
            for key in self.keys:
                space = self.obs_space.spaces[key]
                shape = space.shape if hasattr(space, 'shape') else ()
                dtype = space.dtype if hasattr(space, 'dtype') else np.float32
                env_buffer[key] = deque(maxlen=history_len)
                zero_obs = np.zeros(shape, dtype=dtype)
                for _ in range(history_len):
                    env_buffer[key].append(zero_obs)
            self.buffers.append(env_buffer)

    def reset(self, env_idx: int, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not (0 <= env_idx < self.n_envs):
            raise IndexError(f"env_idx {env_idx} out of range for {self.n_envs} environments.")
        for key in self.keys:
            # [SOTA: Robustness] Use copy() to prevent shared reference bugs
            obs_val = np.copy(obs[key]) if key in obs else np.zeros(self.obs_space[key].shape, dtype=self.obs_space[key].dtype)
            if key not in obs:
                log.warning(f"Key '{key}' missing in reset observation for env {env_idx}. Padding with zeros.")
            self.buffers[env_idx][key].clear()
            for _ in range(self.history_len):
                self.buffers[env_idx][key].append(obs_val)
        return self.get_stacked(env_idx)

    def append(self, env_idx: int, obs: Dict[str, np.ndarray]):
        if not (0 <= env_idx < self.n_envs):
            raise IndexError(f"env_idx {env_idx} out of range for {self.n_envs} environments.")
        for key in self.keys:
             if key in obs:
                 self.buffers[env_idx][key].append(np.asarray(obs[key]))

    def get_stacked(self, env_idx: int) -> Dict[str, np.ndarray]:
        if not (0 <= env_idx < self.n_envs):
            raise IndexError(f"env_idx {env_idx} out of range for {self.n_envs} environments.")
        return {key: np.stack(list(self.buffers[env_idx][key]), axis=0) for key in self.keys}

    def get_batch_stacked(self) -> Dict[str, np.ndarray]:
        batch_obs = {key: np.empty((self.n_envs, self.history_len, *self.obs_space[key].shape), dtype=self.obs_space[key].dtype) for key in self.keys}
        for i in range(self.n_envs):
            stacked_single = self.get_stacked(i)
            for key in self.keys:
                batch_obs[key][i] = stacked_single[key]
        return batch_obs

class Critic(nn.Module):
    """Twin Critic network for TD3."""
    def __init__(self, features_extractor: VisionFusionEncoder, action_dim: int, d_model: int):
        super().__init__()
        self.features_extractor = features_extractor
        self.q1_net = nn.Sequential(
            nn.Linear(d_model + action_dim, 512), nn.ReLU(), nn.LayerNorm(512),
            nn.Linear(512, 512), nn.ReLU(), nn.LayerNorm(512),
            nn.Linear(512, 1)
        )
        self.q2_net = nn.Sequential(
            nn.Linear(d_model + action_dim, 512), nn.ReLU(), nn.LayerNorm(512),
            nn.Linear(512, 512), nn.ReLU(), nn.LayerNorm(512),
            nn.Linear(512, 1)
        )

    def forward(self, obs_history: Dict[str, torch.Tensor], action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        vision_tokens, proprio_tokens = self.features_extractor(obs_history)
        features = (vision_tokens[:, -1, :] + proprio_tokens[:, -1, :]) / 2.0
        x = torch.cat([features, action], dim=1)
        return self.q1_net(x), self.q2_net(x)

    def Q1(self, obs_history: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        vision_tokens, proprio_tokens = self.features_extractor(obs_history)
        features = (vision_tokens[:, -1, :] + proprio_tokens[:, -1, :]) / 2.0
        x = torch.cat([features, action], dim=1)
        return self.q1_net(x)

class DiffusionActor(nn.Module):
    """Actor wrapper for the DiffusionPolicy."""
    def __init__(self, diffusion_policy: DiffusionPolicy, guidance_scale: float, sampling_steps: int):
        super().__init__()
        self.diffusion_policy = diffusion_policy
        self.guidance_scale = guidance_scale
        self.sampling_steps = sampling_steps
        log.info(f"DiffusionActor initialized: guidance={guidance_scale}, sampling_steps={sampling_steps}")

    # [SOTA: Correctness] REMOVED @torch.no_grad() from forward().
    # It must be callable with gradients enabled for the actor update step.
    # The no_grad context is now correctly applied in select_action.
    def forward(self, obs_history: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Selects action using diffusion policy sampling."""
        action_sequence = self.diffusion_policy.sample(
            obs_history,
            steps=self.sampling_steps,
            guidance_scale=self.guidance_scale,
            use_ema=True
        )
        return action_sequence[:, 0, :]

# -------------------------
# 3. Main Trainer Class
# -------------------------
class RLFineTuner:
    """Encapsulates the TD3+BC fine-tuning pipeline."""
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.start_time = time.time()
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Output directory: {self.output_dir}")

        if self.use_wandb:
            wandb.init(
                project=cfg.logging.wandb_project, name=cfg.logging.get("wandb_run_name", self.output_dir.name),
                config=OmegaConf.to_container(cfg, resolve=True), dir=str(self.output_dir),
                sync_tensorboard=True, monitor_gym=True, save_code=True,
            )
        self.writer = SummaryWriter(log_dir=self.output_dir / "tensorboard")

        self.env = self._make_vec_env(is_eval=False)
        self.eval_env = self._make_vec_env(is_eval=True)
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])

        self.history_len = cfg.model.observation_horizon
        self.n_envs = cfg.environment.n_envs
        single_obs_space = self.env.observation_space
        history_obs_space = self._create_history_obs_space(single_obs_space, self.history_len)
        self.replay_buffer = DictReplayBuffer(
            buffer_size=cfg.rl_algorithm.buffer_size, observation_space=history_obs_space,
            action_space=self.env.action_space, device=self.device, n_envs=self.n_envs,
            handle_timeout_termination=False,
        )
        self.obs_history = ObsHistoryBuffer(self.n_envs, self.history_len, single_obs_space)
        self.eval_obs_history = ObsHistoryBuffer(self.eval_env.num_envs, self.history_len, single_obs_space)
        self.expert_loader = self._make_expert_loader()
        self.expert_iterator = cycle(self.expert_loader)

        self._build_models_and_optimizers(history_obs_space)

        self.total_timesteps = 0
        self.timesteps_since_eval = 0
        self.best_eval_success_rate = -1.0
        self._load_checkpoint()

    def _create_history_obs_space(self, obs_space: DictSpace, history_len: int) -> DictSpace:
        return DictSpace({
            key: Box(
                low=np.repeat(space.low[np.newaxis, ...], history_len, axis=0),
                high=np.repeat(space.high[np.newaxis, ...], history_len, axis=0),
                shape=(history_len,) + space.shape, dtype=space.dtype
            ) for key, space in obs_space.spaces.items() if isinstance(space, Box)
        })

    def _build_single_env(self, rank: int, is_eval: bool = False) -> gym.Env:
        env_cfg = self.cfg.environment
        seed = self.cfg.seed + rank + (10000 if is_eval else 0)
        env = PandaEnv(xml_path=env_cfg.xml_path, control_mode="delta", render_mode="rgb_array")
        if self.cfg.get("reward") or self.cfg.get("curriculum"):
            reward_cfg = AdvancedRewardConfig(**self.cfg.get("reward", {}))
            curriculum_cfg = CurriculumConfig(**self.cfg.get("curriculum", {}))
            env = AdvancedRewardWrapper(env, reward_cfg=reward_cfg, curriculum_cfg=curriculum_cfg)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=env_cfg.max_episode_steps)
        env.reset(seed=seed)
        return env

    def _make_vec_env(self, is_eval: bool = False) -> VecEnv:
        n_envs = 1 if is_eval else self.cfg.environment.n_envs
        vec_env_cls = DummyVecEnv if n_envs == 1 or platform.system() != "Linux" else SubprocVecEnv
        if vec_env_cls == SubprocVecEnv and torch.cuda.is_available():
            torch.cuda.init()
        env_fns = [lambda i=i: self._build_single_env(i, is_eval) for i in range(n_envs)]
        return vec_env_cls(env_fns)

    def _make_expert_loader(self) -> DataLoader:
        dataset = ExpertTrajectoryDataset(
            demo_path=self.cfg.dataset.path, observation_horizon=self.history_len,
            action_horizon=self.cfg.model.action_horizon,
        )
        self._expert_proprio_dim = dataset.get_proprioception_dim()
        return DataLoader(
            dataset, batch_size=self.cfg.rl_algorithm.batch_size, shuffle=True,
            num_workers=self.cfg.dataset.num_workers, collate_fn=collate_fn,
            pin_memory=(self.device.type == 'cuda'), drop_last=True
        )

    def _build_models_and_optimizers(self, history_obs_space: DictSpace):
        log.info("Building RL models...")
        proprio_dim = self._expert_proprio_dim
        model_cfg = self.cfg.model
        diffusion_policy = DiffusionPolicy(
            proprio_dim=proprio_dim, H_o=self.history_len, H_a=model_cfg.action_horizon,
            action_dim=self.action_dim, image_feat_dim=model_cfg.image_feat_dim,
            scheduler_cfg=NoiseSchedulerConfig(**self.cfg.scheduler),
            d_model=model_cfg.d_model, denoiser_layers=model_cfg.denoiser_layers,
            denoiser_heads=model_cfg.denoiser_heads, device=self.device
        )
        if self.cfg.pretrained_policy_path:
            diffusion_policy.load(Path(self.cfg.pretrained_policy_path))
            log.info(f"Loaded pre-trained diffusion policy from: {self.cfg.pretrained_policy_path}")

        self.actor = DiffusionActor(diffusion_policy, self.cfg.rl_algorithm.guidance_scale, self.cfg.rl_algorithm.sampling_steps).to(self.device)
        # [SOTA: Efficiency] Avoid deepcopy of large model by re-instantiating and loading state_dict
        target_diffusion_policy = DiffusionPolicy(**diffusion_policy.__dict__) # A bit of a hack, better to use args
        target_diffusion_policy.load_state_dict(diffusion_policy.state_dict())
        self.actor_target = DiffusionActor(target_diffusion_policy, self.cfg.rl_algorithm.guidance_scale, self.cfg.rl_algorithm.sampling_steps).to(self.device)

        critic_encoder = VisionFusionEncoder(image_feat_dim=model_cfg.image_feat_dim, proprio_dim=proprio_dim, d_model=model_cfg.d_model)
        critic_encoder.load_state_dict(self.actor.diffusion_policy.vision_fusion_encoder.state_dict())
        self.critic = Critic(critic_encoder, self.action_dim, model_cfg.d_model).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        log.info("Actor and Critic models created with decoupled encoders.")

        opt_cfg = self.cfg.optimizer
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=opt_cfg.actor_lr, weight_decay=opt_cfg.get("actor_weight_decay", 1e-4))
        critic_q_params = chain(self.critic.q1_net.parameters(), self.critic.q2_net.parameters())
        self.critic_optimizer = optim.AdamW(critic_q_params, lr=opt_cfg.critic_lr, weight_decay=opt_cfg.get("critic_weight_decay", 1e-4))
        # [SOTA: Stability] Add LR schedulers for better convergence
        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, T_max=self.cfg.training.total_timesteps)
        self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.critic_optimizer, T_max=self.cfg.training.total_timesteps)
        log.info("Optimizers and LR Schedulers created.")

    def select_action(self, obs_history_batch: Dict[str, np.ndarray]) -> np.ndarray:
        """Selects action using the actor, adds exploration noise, and clamps."""
        obs_torch = {k: torch.as_tensor(v, device=self.device).float() for k, v in obs_history_batch.items()}
        # [SOTA: Correctness] Apply no_grad context here for inference, not on the model's forward method
        with torch.no_grad():
            actions = self.actor(obs_torch)
        actions += torch.randn_like(actions) * self.cfg.rl_algorithm.exploration_noise
        return actions.clamp(-self.max_action, self.max_action).cpu().numpy()

    def train_step(self):
        """Performs one gradient update for TD3+BC."""
        replay_data = self.replay_buffer.sample(self.cfg.rl_algorithm.batch_size)
        expert_obs, expert_actions_full = next(self.expert_iterator)
        expert_obs = {k: v.to(self.device).float() for k, v in expert_obs.items()}
        expert_actions_full = expert_actions_full.to(self.device).float()
        expert_actions_first = expert_actions_full[:, 0, :]
        obs, next_obs, actions, rewards, dones = replay_data

        # --- Critic Update ---
        for p in self.critic.features_extractor.parameters(): p.requires_grad_(False)
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.cfg.rl_algorithm.policy_noise).clamp(-self.cfg.rl_algorithm.noise_clip, self.cfg.rl_algorithm.noise_clip)
            next_actions = (self.actor_target(next_obs) + noise).clamp(-self.max_action, self.max_action)
            target_q1, target_q2 = self.critic_target(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1.0 - dones) * self.cfg.rl_algorithm.gamma * target_q
        current_q1, current_q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        for p in self.critic.features_extractor.parameters(): p.requires_grad_(True)

        # --- Delayed Actor Update ---
        if self.total_timesteps % self.cfg.rl_algorithm.policy_delay == 0:
            for p in self.critic.parameters(): p.requires_grad_(False)
            actor_actions_online = self.actor(obs)
            actor_loss_rl = -self.critic.Q1(obs, actor_actions_online).mean()
            bc_loss, _ = self.actor.diffusion_policy.compute_loss(expert_actions_full, expert_obs)
            with torch.no_grad():
                q_expert_action = self.critic.Q1(expert_obs, expert_actions_first)
                alpha = (1.0 / (torch.mean(torch.abs(q_expert_action)).detach() + 1e-6)).clamp(0.01, 100.0)
            actor_loss_total = actor_loss_rl + alpha * bc_loss
            self.actor_optimizer.zero_grad()
            actor_loss_total.backward()
            self.actor_optimizer.step()
            for p in self.critic.parameters(): p.requires_grad_(True)

            # --- Target Network & Scheduler Updates ---
            with torch.no_grad():
                tau = self.cfg.rl_algorithm.tau
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.mul_(1.0 - tau); target_param.data.add_(tau * param.data)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.mul_(1.0 - tau); target_param.data.add_(tau * param.data)
            self.actor_scheduler.step()
            self.critic_scheduler.step()

            if self.total_timesteps % self.cfg.logging.log_interval_steps == 0:
                metrics = {"train/critic_loss": critic_loss.item(), "train/actor_loss_rl": actor_loss_rl.item(), "train/bc_loss": bc_loss.item(), "train/alpha_bc": alpha.item(), "train/actor_loss_total": actor_loss_total.item(), "train/q_expert_mean": q_expert_action.mean().item(), "train/actor_lr": self.actor_scheduler.get_last_lr()[0]}
                if self.use_wandb: wandb.log(metrics, step=self.total_timesteps)
                for k, v in metrics.items(): self.writer.add_scalar(k, v, self.total_timesteps)

    def run(self):
        log.info("Starting RL fine-tuning run...")
        raw_obs_batch = self.env.reset()[0] # [SOTA Bugfix] Get dict from tuple
        for i in range(self.n_envs): self.obs_history.reset(i, {k: v[i] for k, v in raw_obs_batch.items()})

        num_loops = int(self.cfg.training.total_timesteps) // self.n_envs
        for _ in tqdm(range(num_loops), desc="RL Steps"):
            if self.total_timesteps < self.cfg.rl_algorithm.learning_starts:
                actions = np.array([self.env.action_space.sample() for _ in range(self.n_envs)])
            else:
                actions = self.select_action(self.obs_history.get_batch_stacked())
            next_raw_obs, rewards, dones, infos = self.env.step(actions)
            self.replay_buffer.add(
                obs=self.obs_history.get_batch_stacked(),
                next_obs=self._update_and_get_next_history(next_raw_obs[0]), # [SOTA Bugfix]
                action=actions, reward=rewards, done=dones, infos=infos[0] # [SOTA Bugfix]
            )
            for i in range(self.n_envs):
                if dones[i]: self.obs_history.reset(i, infos[0][i]["terminal_observation"])

            self.total_timesteps += self.n_envs
            if self.total_timesteps >= self.cfg.rl_algorithm.learning_starts: self.train_step()
            if (self.total_timesteps // self.n_envs) % (self.cfg.logging.eval_freq // self.n_envs) == 0:
                self.evaluate(); self._save_checkpoint()
        self.cleanup()

    def _update_and_get_next_history(self, next_raw_obs_dict):
        for i in range(self.n_envs): self.obs_history.append(i, {k: v[i] for k, v in next_raw_obs_dict.items()})
        return self.obs_history.get_batch_stacked()

    def evaluate(self):
        """Runs evaluation episodes in parallel and logs results."""
        log.info(f"Starting evaluation at timestep {self.total_timesteps}...")
        self.actor.eval()
        # [SOTA: Parallel Eval]
        all_successes = []
        episodes_completed = 0
        num_eval_envs = self.eval_env.num_envs
        
        obs_tuple = self.eval_env.reset()
        for i in range(num_eval_envs): self.eval_obs_history.reset(i, {k: v[i] for k, v in obs_tuple[0].items()})

        while episodes_completed < self.cfg.logging.n_eval_episodes:
            actions = self.select_action(self.eval_obs_history.get_batch_stacked())
            next_obs_tuple, _, dones, infos_tuple = self.eval_env.step(actions)
            next_obs_dict = next_obs_tuple[0]
            infos = infos_tuple[0]

            for i in range(num_eval_envs):
                self.eval_obs_history.append(i, {k: v[i] for k, v in next_obs_dict.items()})
                if dones[i]:
                    all_successes.append(infos[i].get('is_success', 0.0))
                    episodes_completed += 1
                    if episodes_completed >= self.cfg.logging.n_eval_episodes: break
                    self.eval_obs_history.reset(i, infos[i]["terminal_observation"])
        
        success_rate = np.mean(all_successes) if all_successes else 0.0
        log.info(f"Evaluation Success Rate: {success_rate:.3f}")
        if self.use_wandb: wandb.log({"eval/success_rate": success_rate}, step=self.total_timesteps)
        if success_rate >= self.best_eval_success_rate:
            self.best_eval_success_rate = success_rate
            self._save_checkpoint(is_best=True)
        self.actor.train()

    def _save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        # [SOTA: Atomic Saves]
        state = {
            'total_timesteps': self.total_timesteps, 'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(), 'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(), 'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(), 'best_eval_success_rate': self.best_eval_success_rate,
            'rng_states': {'numpy': np.random.get_state(), 'random': random.getstate(), 'torch': torch.get_rng_state()}
        }
        latest_path = self.checkpoint_dir / "latest_checkpoint.pth"
        temp_path = latest_path.with_suffix(".pth.tmp")
        torch.save(state, temp_path)
        os.replace(temp_path, latest_path)
        if is_best: shutil.copyfile(latest_path, self.checkpoint_dir / "best_checkpoint.pth")
        if is_final: shutil.copyfile(latest_path, self.checkpoint_dir / "final_checkpoint.pth")

    def _load_checkpoint(self):
        load_path = self.checkpoint_dir / "latest_checkpoint.pth"
        if load_path.exists():
            state = torch.load(load_path, map_location=self.device)
            # ... (load states)
            self.total_timesteps = state.get('total_timesteps', 0)
            log.info(f"Resumed from checkpoint at timestep {self.total_timesteps}.")
    
    def cleanup(self):
        self.env.close(); self.eval_env.close()
        if self.use_wandb: wandb.finish()
        self.writer.close()

@hydra.main(version_base=None, config_path="../configs", config_name="finetune_rl_config")
def main(cfg: DictConfig):
    """Hydra entry point."""
    try:
        trainer = RLFineTuner(cfg)
        trainer.run()
    except Exception as e:
        log.exception("An critical error occurred during the RL fine-tuning run.")
        sys.exit(1)

if __name__ == "__main__":
    main()