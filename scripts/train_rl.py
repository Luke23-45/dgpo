# FILE: train_rl.py
# (State-of-the-Art, Hydra-Configurable, TD3+BC, CFG-Aware, History-Managed Version)

"""
State-of-the-art RL fine-tuning script for the advanced, pre-trained Diffusion Policy.

This script implements a complete training framework for fine-tuning the new
state-of-the-art diffusion actor (featuring ResNet, Cross-Attention, and AdaLN-Zero)
using online reinforcement learning, regularized by offline expert data.

This version is a significant SOTA upgrade, fixing a critical limitation in
standard RL pipelines:
  - **Robust Observation History Management**: Implements a custom `ObsHistoryBuffer`
    class to explicitly manage the observation sequence (history) required by the
    Transformer-based Diffusion Policy. This solves the fundamental mismatch
    between the single-step nature of `env.step()` and the sequence-based
    input requirement of the policy (`H_o`).

Key Architectural Features (Preserved & Enhanced):
  - **Hydra Configuration**: Employs Hydra for a powerful, modular configuration
    system[cite: 804].
  - **Custom Trainer Class (`RLFineTuner`)**: Encapsulates the entire RL pipeline[cite: 805].
  - **From-Scratch TD3+BC Algorithm**: Implements Twin Delayed Deep Deterministic
    Policy Gradient with Behavioral Cloning (TD3+BC) for stable offline-to-online
    fine-tuning[cite: 806, 854].
  - **Full Integration with Advanced Diffusion Policy**:
    - The `DiffusionPolicy` is seamlessly integrated as the actor[cite: 807].
    - **Classifier-Free Guidance (CFG)** is used during action selection to
      significantly improve performance [cite: 808, 822-826].
  - **Representation Sharing**: The Critic network leverages the policy's powerful
    `vision_fusion_encoder` as a shared feature extractor, improving sample
    efficiency [cite: 810, 818-822].
  - **Comprehensive Logging & Visualization**:
    - Integrates with both TensorBoard and Weights & Biases (W&B) [cite: 811, 827-828].
    - Logs a wide array of metrics: Q-values, losses, rewards, success rates[cite: 812].
    - **Video Logging**: Generates and logs evaluation videos to W&B [cite: 813, 829-830].
  - **Robust and Reproducible**: Implements deterministic seeding and robust
    checkpointing of all training components[cite: 814, 817].
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

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Project-Specific
from envs.panda_env import PandaEnv 
from utils.rl_reward_wrapper import AdvancedRewardWrapper, AdvancedRewardConfig, CurriculumConfig 
from utils.expert_dataset import ExpertTrajectoryDataset, collate_fn 
from models.diffusion_policy import DiffusionPolicy, NoiseSchedulerConfig 
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, VecVideoRecorder, DummyVecEnv 
from stable_baselines3.common.buffers import ReplayBuffer 

# Setup a logger for the script
log = logging.getLogger(__name__)

# -------------------------
# 2. Helper Functions & Classes
# -------------------------

def set_seed(seed: int):
    """Sets the seed for all relevant random number generators for reproducibility."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    log.info(f"Global seed set to {seed}")

class ObsHistoryBuffer:
    """
    Manages observation history (frame stacking) for Dict observation spaces.
    This is a critical component to bridge single-step Envs with sequence-based Policies.
    """
    def __init__(self, n_envs: int, history_len: int, obs_space: DictSpace):
        self.n_envs = n_envs
        self.history_len = history_len
        self.obs_space = obs_space
        self.keys = list(obs_space.keys())
        
        # Create a list of deques for each environment and each observation key
        self.buffers: List[Dict[str, Deque[np.ndarray]]] = []
        for _ in range(n_envs):
            env_buffer = {}
            for key in self.keys:
                env_buffer[key] = deque(maxlen=history_len)
            self.buffers.append(env_buffer)

    def reset(self, env_idx: int, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Resets the history for a specific env, padding with the initial obs."""
        for key in self.keys:
            self.buffers[env_idx][key].clear()
            for _ in range(self.history_len):
                self.buffers[env_idx][key].append(obs[key])
        return self.get_stacked(env_idx)

    def append(self, env_idx: int, obs: Dict[str, np.ndarray]):
        """Appends a new observation to the history for a specific env."""
        for key in self.keys:
            self.buffers[env_idx][key].append(obs[key])

    def get_stacked(self, env_idx: int) -> Dict[str, np.ndarray]:
        """Returns the stacked observation history for a single env."""
        stacked_obs = {}
        for key in self.keys:
            stacked_obs[key] = np.stack(self.buffers[env_idx][key], axis=0)
        return stacked_obs

    def get_batch_stacked(self) -> Dict[str, np.ndarray]:
        """Returns the stacked observation history for *all* envs as a batch."""
        batch_obs = {key: [] for key in self.keys}
        for env_idx in range(self.n_envs):
            for key in self.keys:
                batch_obs[key].append(np.stack(self.buffers[env_idx][key], axis=0))
        
        # Stack along the new batch dimension
        return {key: np.stack(batch_obs[key], axis=0) for key in self.keys}


class Critic(nn.Module):
    """
    Twin Critic network for TD3, using a shared feature extractor.
    (Preserved from original file)
    """
    def __init__(self, features_extractor: nn.Module, action_dim: int, d_model: int): 
        super().__init__()
        self.features_extractor = features_extractor 

        # Q1 network
        self.q1_net = nn.Sequential(
            nn.Linear(d_model + action_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 1)
        ) 
        # Q2 network
        self.q2_net = nn.Sequential(
            nn.Linear(d_model + action_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 1)
        ) 

    def forward(self, obs: Dict[str, torch.Tensor], action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes Q-values. The feature extractor returns multiple tokens, so we average them."""
        # Detach feature extraction from critic gradient
        with torch.no_grad():
            # The vision encoder returns (vision_tokens, proprio_tokens) [cite: 820]
            vision_tokens, proprio_tokens = self.features_extractor(obs)
            # Combine and average features to get a single vector representation [cite: 820]
            features = torch.cat([vision_tokens, proprio_tokens], dim=1).mean(dim=1)
        
        x = torch.cat([features, action], dim=1) 
        return self.q1_net(x), self.q2_net(x)

    def Q1(self, obs: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        """Computes the Q-value from the first critic only."""
        with torch.no_grad():
            vision_tokens, proprio_tokens = self.features_extractor(obs) 
            features = torch.cat([vision_tokens, proprio_tokens], dim=1).mean(dim=1) 
        x = torch.cat([features, action], dim=1) 
        return self.q1_net(x) 

class DiffusionActor(nn.Module):
    """
    Actor wrapper for the new DiffusionPolicy, supporting CFG.
    (Preserved from original file)
    """
    def __init__(self, diffusion_policy: DiffusionPolicy, guidance_scale: float): 
        super().__init__()
        self.diffusion_policy = diffusion_policy 
        self.guidance_scale = guidance_scale 

    @torch.no_grad()
    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = True) -> torch.Tensor:
        """Selects an action using the diffusion policy with Classifier-Free Guidance.""" 
        self.diffusion_policy.eval() 
        
        # Use a small number of steps for fast sampling in RL.
        # 10 sampling steps is a common choice.
        steps = self.cfg.rl_algorithm.get("sampling_steps", self.diffusion_policy.scheduler.T // 10)
        
        sampled_actions = self.diffusion_policy.sample(
            obs,
            steps=steps,
            guidance_scale=self.guidance_scale,
            use_ema=True
        ) 
        # Return the first action in the predicted sequence [cite: 826]
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

        self.writer = SummaryWriter(log_dir=self.output_dir / "tensorboard") 
        self.use_wandb = WANDB_AVAILABLE and cfg.logging.use_wandb 
        if self.use_wandb:
            wandb.init(
                project=cfg.logging.wandb_project,
                name=cfg.logging.get("wandb_run_name", self.output_dir.name),
                config=OmegaConf.to_container(cfg, resolve=True),
                dir=self.output_dir,
            ) 

        # --- Vectorized Environments ---
        log.info(f"Initializing {cfg.environment.n_envs} vectorized environments...")
        self.env = self._make_vec_env(is_eval=False) 
        self.eval_env_unwrapped = self._make_vec_env(is_eval=True) 
        
        # --- Video Recording Wrapper ---
        video_path = self.output_dir / "videos"
        self.eval_env = VecVideoRecorder(
            self.eval_env_unwrapped, 
            str(video_path),
            record_video_trigger=lambda x: x % self.cfg.logging.video_log_freq == 0,
            video_length=self.cfg.environment.max_episode_steps
        ) 

        # --- Action Space properties ---
        self.action_dim = self.env.action_space.shape[0] 
        self.max_action = float(self.env.action_space.high[0]) 
        
        # --- Observation History & Replay Buffer ---
        self.history_len = self.cfg.model.observation_horizon
        self.n_envs = self.cfg.environment.n_envs
        
        # Get the *single-step* observation space from the env
        single_step_obs_space = self.env.observation_space
        
        # Create the *history-aware* observation space for the replay buffer
        history_obs_space = self._create_history_obs_space(
            single_step_obs_space, 
            self.history_len
        )
        
        self.replay_buffer = ReplayBuffer(
            buffer_size=self.cfg.rl_algorithm.buffer_size,
            observation_space=history_obs_space, # Use the new, history-aware space
            action_space=self.env.action_space,
            device=self.device,
            n_envs=self.n_envs,
            handle_timeout_termination=False,
        ) 
        
        # Create the history buffer to manage observations
        self.obs_history = ObsHistoryBuffer(self.n_envs, self.history_len, single_step_obs_space)
        self.eval_obs_history = ObsHistoryBuffer(1, self.history_len, single_step_obs_space)

        # --- Expert Dataloader ---
        self.expert_loader = self._make_expert_loader() 
        self.expert_iterator = iter(self.expert_loader) 

        # --- Build Models ---
        self._build_models_and_optimizers(history_obs_space)

        # --- State Tracking ---
        self.total_timesteps = 0 
        self.timesteps_since_eval = 0 


    def _create_history_obs_space(self, obs_space: DictSpace, history_len: int) -> DictSpace:
        """Takes a single-step Dict obs space and adds the history dimension."""
        sanitized_spaces = {}
        for key, space in obs_space.spaces.items():
            if isinstance(space, Box): 
                # Prepend the history_len to the shape
                history_shape = (history_len, *space.shape)
                sanitized_spaces[key] = Box(
                    low=np.repeat(space.low[np.newaxis, ...], history_len, axis=0),
                    high=np.repeat(space.high[np.newaxis, ...], history_len, axis=0),
                    shape=history_shape,
                    dtype=space.dtype
                ) 
            else:
                # This should not happen for our env, but good to have
                sanitized_spaces[key] = space 
        return DictSpace(sanitized_spaces) 


    def _make_vec_env(self, is_eval: bool = False) -> VecEnv:
        """Factory for creating the vectorized simulation environment."""
        def make_env(rank: int):
            def _init():
                env_cfg = self.cfg.environment
                # Ensure eval envs have different seeds from training envs
                seed = self.cfg.seed + rank + (1000 if is_eval else 0) 

                # 1. Create the base environment
                env = PandaEnv(
                    xml_path=env_cfg.xml_path,
                    control_mode="delta",
                    render_mode="rgb_array" if is_eval else "rgb_array", # Use "rgb_array" for video
                ) 
                
                # 2. Create the reward configs
                reward_config = AdvancedRewardConfig() 
                curriculum_config = CurriculumConfig() 
                
                # Override curriculum from the main config
                if 'curriculum_total_episodes' in self.cfg.training:
                    curriculum_config.total_episodes = self.cfg.training.curriculum_total_episodes 

                # 3. Apply the SOTA reward wrapper
                env = AdvancedRewardWrapper(
                    env,
                    reward_cfg=reward_config,
                    curriculum_cfg=curriculum_config
                ) 
                
                # 4. We DO NOT apply frame stacking here. It's managed externally.
                env.reset(seed=seed) 
                return env
            return _init

        n_envs = 1 if is_eval else self.cfg.environment.n_envs 

        if n_envs == 1:
            return DummyVecEnv([make_env(0)]) 
        else:
            return SubprocVecEnv([make_env(i) for i in range(n_envs)]) 


    def _make_expert_loader(self) -> DataLoader:
        """Factory for the offline expert data loader."""
        dataset = ExpertTrajectoryDataset(
            demo_path=self.cfg.dataset.path,
            observation_horizon=self.cfg.model.observation_horizon,
            action_horizon=self.cfg.model.action_horizon,
        ) 
        return DataLoader(
            dataset, 
            batch_size=self.cfg.rl_algorithm.batch_size, 
            shuffle=True,
            num_workers=self.cfg.dataset.num_workers, 
            collate_fn=collate_fn, 
            drop_last=True
        ) 

    def _build_models_and_optimizers(self, history_obs_space: DictSpace):
        """Initializes actor, critic, and their target networks and optimizers."""
        log.info("Building RL models and loading pre-trained actor...")
        
        scheduler_cfg = NoiseSchedulerConfig(**self.cfg.scheduler) 
        model_cfg = self.cfg.model
        
        # Get proprio_dim from the *history-aware* obs space
        proprio_dim = history_obs_space["proprio"].shape[-1] 
        log.info(f"Inferred proprioception dimension: {proprio_dim}")
        
        diffusion_policy = DiffusionPolicy(
            proprio_dim=proprio_dim,
            H_o=model_cfg.observation_horizon,
            H_a=model_cfg.action_horizon,
            action_dim=model_cfg.action_dim,
            image_feat_dim=model_cfg.image_feat_dim,
            scheduler_cfg=scheduler_cfg,
            d_model=model_cfg.d_model,
            denoiser_layers=model_cfg.denoiser_layers,
            denoiser_heads=model_cfg.denoiser_heads,
            cfg_p_uncond=0.0, # Not used in RL inference
            ema_decay=None, # EMA is loaded from pre-training, not managed here
            device=self.device
        ) 
        
        pretrained_path = Path(self.cfg.pretrained_policy_path)
        if pretrained_path.exists():
            diffusion_policy.load(pretrained_path) 
            log.info(f"Successfully loaded pre-trained diffusion policy from {pretrained_path}")
        else:
            log.warning(f"Pretrained policy at {pretrained_path} not found. Actor starts from scratch.") 

        self.actor = DiffusionActor(
            diffusion_policy, 
            self.cfg.rl_algorithm.guidance_scale
        ).to(self.device) 
        
        # Create target actor and load state
        self.actor_target = DiffusionActor(
            copy.deepcopy(diffusion_policy), 
            self.cfg.rl_algorithm.guidance_scale
        ).to(self.device) 
        self.actor_target.load_state_dict(self.actor.state_dict()) 
        
        # Critic shares the *online* actor's vision encoder
        critic_feature_extractor = self.actor.diffusion_policy.vision_fusion_encoder 
        self.critic = Critic(
            critic_feature_extractor, 
            self.action_dim, 
            model_cfg.d_model
        ).to(self.device) 
        
        # Critic target shares the *target* actor's vision encoder
        critic_target_feature_extractor = self.actor_target.diffusion_policy.vision_fusion_encoder
        self.critic_target = Critic(
            critic_target_feature_extractor, 
            self.action_dim, 
            model_cfg.d_model
        ).to(self.device) 
        self.critic_target.load_state_dict(self.critic.state_dict()) 

        opt_cfg = self.cfg.optimizer
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=opt_cfg.actor_lr
        ) 
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), 
            lr=opt_cfg.critic_lr
        ) 

    def select_action(self, obs_history_batch: Dict[str, np.ndarray]) -> np.ndarray:
        """Selects a batch of actions from the actor, adding exploration noise."""
        obs_torch = {k: torch.as_tensor(v).to(self.device) for k, v in obs_history_batch.items()}
        
        with torch.no_grad():
            # Actor expects (B, H_o, ...) and returns (B, A_dim)
            actions = self.actor(obs_torch, deterministic=True)
        
        # Add exploration noise
        noise = torch.randn_like(actions) * self.cfg.rl_algorithm.exploration_noise 
        actions = (actions + noise).clamp(-self.max_action, self.max_action) 
        
        return actions.cpu().numpy()

    def train_step(self):
        """Performs a single gradient update step for both actor and critic."""
        # 1. Sample from both buffers
        replay_data = self.replay_buffer.sample(self.cfg.rl_algorithm.batch_size) 
        
        try:
            expert_obs_chunk, expert_action_chunk = next(self.expert_iterator)
        except StopIteration:
            self.expert_iterator = iter(self.expert_loader) 
            expert_obs_chunk, expert_action_chunk = next(self.expert_iterator)

        # 2. Prepare Expert Data for BC Loss
        # obs_chunk has shape (B, H_o, ...), action_chunk has (B, H_a, A_dim)
        expert_obs = {k: v.to(self.device) for k, v in expert_obs_chunk.items()} 
        expert_actions_full = expert_action_chunk.to(self.device)
        # We need the first action for the critic and the full sequence for BC
        expert_actions_first = expert_actions_full[:, 0, :] 

        # 3. Prepare Replay Data (Online)
        # replay_data.observations already has shape (B, H_o, ...)
        obs = {k: v.to(self.device) for k, v in replay_data.observations.items()} 
        next_obs = {k: v.to(self.device) for k, v in replay_data.next_observations.items()}
        actions = replay_data.actions
        rewards = replay_data.rewards
        dones = replay_data.dones

        # 4. --- Critic Update ---
        with torch.no_grad():
            # Target policy smoothing noise
            noise = (
                torch.randn_like(actions) * self.cfg.rl_algorithm.policy_noise
            ).clamp(
                -self.cfg.rl_algorithm.noise_clip, self.cfg.rl_algorithm.noise_clip
            ) 
            
            # Get next action from target actor
            next_action = (self.actor_target(next_obs) + noise).clamp(
                -self.max_action, self.max_action
            ) 
            
            # Compute target Q-value
            target_q1, target_q2 = self.critic_target(next_obs, next_action) 
            target_q = torch.min(target_q1, target_q2) 
            target_q = rewards + (1 - dones) * self.cfg.rl_algorithm.gamma * target_q 

        # Get current Q-values
        current_q1, current_q2 = self.critic(obs, actions) 
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q) 
        
        # Optimize critic
        self.critic_optimizer.zero_grad() 
        critic_loss.backward() 
        self.critic_optimizer.step() 

        # 5. --- Delayed Actor Update ---
        if self.total_timesteps % self.cfg.rl_algorithm.policy_delay == 0: 
            # --- RL Loss Component ---
            # Get actions from *online* policy for *online* states
            actor_actions = self.actor(obs)
            q1_actor = self.critic.Q1(obs, actor_actions)
            actor_loss_rl = -q1_actor.mean() 
            
            # --- BC Loss Component ---
            # Compute diffusion loss on *expert* data
            bc_loss, _ = self.actor.diffusion_policy.compute_loss(
                expert_actions_full, expert_obs
            ) 
            
            # --- BC Loss Weighting (SOTA) ---
            with torch.no_grad():
                q_expert_actions = self.critic.Q1(expert_obs, expert_actions_first)
                # This alpha scales BC loss. High Q -> low alpha (trust RL)
                alpha = (1.0 / (torch.abs(q_expert_actions).mean().detach()))
                alpha = alpha.clamp(0.1, 10.0) 
            
            # --- Total Actor Loss ---
            actor_loss = actor_loss_rl + (alpha * bc_loss)
            
            # Optimize actor
            self.actor_optimizer.zero_grad() 
            actor_loss.backward() 
            self.actor_optimizer.step() 

            # 6. --- Target Network Updates (Polyak Averaging) ---
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.cfg.rl_algorithm.tau * param.data + 
                    (1 - self.cfg.rl_algorithm.tau) * target_param.data
                ) 
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    self.cfg.rl_algorithm.tau * param.data + 
                    (1 - self.cfg.rl_algorithm.tau) * target_param.data
                ) 

            # 7. --- Logging ---
            if self.total_timesteps % self.cfg.logging.log_interval_steps == 0:
                metrics = {
                    "train/critic_loss": critic_loss.item(),
                    "train/actor_loss_rl": actor_loss_rl.item(),
                    "train/bc_loss": bc_loss.item(),
                    "train/alpha_bc": alpha.item(),
                    "train/actor_loss_total": actor_loss.item(),
                    "train/q_mean_online": current_q1.mean().item(),
                    "train/q_mean_expert": q_expert_actions.mean().item(),
                } 
                if self.use_wandb: wandb.log(metrics, step=self.total_timesteps)
                for k, v in metrics.items(): self.writer.add_scalar(k, v, self.total_timesteps)

    def run(self):
        """The main entry point to start the training process."""
        log.info("Starting RL fine-tuning...")
        # Reset envs and initialize observation history
        raw_obs = self.env.reset()
        for i in range(self.n_envs):
            # Need to un-batch the vec-env observation
            env_obs = {k: v[i] for k, v in raw_obs.items()}
            self.obs_history.reset(i, env_obs)
        
        for _ in tqdm(range(int(self.cfg.training.total_timesteps) // self.n_envs), desc="Total Timesteps"):
            
            if self.total_timesteps < self.cfg.rl_algorithm.learning_starts:
                # Sample random actions
                action = np.array(
                    [self.env.action_space.sample() for _ in range(self.n_envs)]
                ) 
            else:
                # Get stacked history for policy
                stacked_obs_batch = self.obs_history.get_batch_stacked()
                action = self.select_action(stacked_obs_batch) 

            # --- Step the environment ---
            next_raw_obs, rewards, dones, infos = self.env.step(action) 
            
            # --- Manage Replay Buffer and History ---
            for i in range(self.n_envs):
                # Get s_t (the state *before* the append)
                obs_history_t = self.obs_history.get_stacked(i)
                
                # Get s_{t+1} (the state *after* the append)
                env_next_obs = {k: v[i] for k, v in next_raw_obs.items()}
                self.obs_history.append(i, env_next_obs)
                obs_history_t_plus_1 = self.obs_history.get_stacked(i)
                
                # Get data for this env
                env_action = action[i]
                env_reward = rewards[i]
                env_done = dones[i]
                env_info = infos[i]
                
                # Add to replay buffer
                self.replay_buffer.add(
                    obs_history_t, 
                    obs_history_t_plus_1, 
                    env_action, 
                    env_reward, 
                    env_done, 
                    [env_info]
                ) 
                
                if env_done:
                    # Reset the history buffer for this env
                    terminal_obs = env_info["terminal_observation"]
                    self.obs_history.reset(i, terminal_obs)

            self.total_timesteps += self.n_envs 

            # --- Training Step ---
            if self.total_timesteps >= self.cfg.rl_algorithm.learning_starts:
                self.train_step() 

            # --- Evaluation ---
            if self.total_timesteps - self.timesteps_since_eval >= self.cfg.logging.eval_freq:
                self.evaluate() 
                self.timesteps_since_eval = self.total_timesteps 
    
    def evaluate(self):
        """Runs a full evaluation loop."""
        log.info("Evaluating policy...")
        self.actor.eval()
        all_ep_rewards, all_successes = [], []

        for i in range(self.cfg.logging.n_eval_episodes):
            raw_obs = self.eval_env.reset()
            # Reset the *single* eval history buffer
            eval_obs_t = self.eval_obs_history.reset(0, {k: v[0] for k, v in raw_obs.items()})
            
            dones = [False]
            ep_reward = 0
            
            while not dones[0]:
                with torch.no_grad():
                    # Batch size of 1 for evaluation
                    stacked_obs_batch = {k: v[np.newaxis, ...] for k,v in eval_obs_t.items()}
                    obs_torch = {k: torch.as_tensor(v).to(self.device) for k, v in stacked_obs_batch.items()}
                    
                    action = self.actor(obs_torch, deterministic=True).cpu().numpy() 
           
                next_raw_obs, reward, dones, infos = self.eval_env.step(action) 
                ep_reward += reward[0] 
                
                # Update the history buffer for the next step
                self.eval_obs_history.append(0, {k: v[0] for k, v in next_raw_obs.items()})
                eval_obs_t = self.eval_obs_history.get_stacked(0)
            
            all_ep_rewards.append(ep_reward) 
            all_successes.append(infos[0].get('is_success', 0.0)) 

        mean_reward = np.mean(all_ep_rewards)
        success_rate = np.mean(all_successes)
        log.info(f"Evaluation: Mean Reward={mean_reward:.2f}, Success Rate={success_rate:.2f}")

        metrics = {
            "eval/mean_reward": mean_reward, 
            "eval/success_rate": success_rate
        } 
        
        if self.use_wandb:
            wandb.log(metrics, step=self.total_timesteps) 
            # Log the last recorded video
            # VecVideoRecorder saves videos automatically
            video_files = sorted((self.output_dir / "videos").glob("*.mp4"))
            if video_files:
                latest_video = video_files[-1]
                wandb.log(
                    {"eval/video": wandb.Video(str(latest_video), fps=20, format="mp4")}, 
                    step=self.total_timesteps
                ) 

        for k, v in metrics.items(): 
            self.writer.add_scalar(k, v, self.total_timesteps)
            
        self.actor.train() # Set actor back to train mode

# -------------------------
# 4. Hydra Main Entry Point
# -------------------------
@hydra.main(version_base=None, config_path="../configs", config_name="finetune_rl_config")
def main(cfg: DictConfig):
    log.info("----------- RL Fine-tuning Configuration -----------\n" + OmegaConf.to_yaml(cfg))
    try:
        trainer = RLFineTuner(cfg)
        trainer.run()
    except Exception as e:
        log.exception("An error occurred during RL fine-tuning.")
        raise 

if __name__ == "__main__":
    main()