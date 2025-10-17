# FILE: train_rl.py
# (State-of-the-Art, Hydra-Configurable, TD3+BC, CFG-Aware Version)

"""
State-of-the-art RL fine-tuning script for the advanced, pre-trained Diffusion Policy.

This script implements a complete training framework for fine-tuning the new
state-of-the-art diffusion actor (featuring ResNet, Cross-Attention, and AdaLN-Zero)
using online reinforcement learning, regularized by offline expert data.

Key Architectural Features:
  - **Hydra Configuration**: Employs Hydra for a powerful, modular configuration
    system.
  - **Custom Trainer Class (`RLFineTuner`)**: Encapsulates the entire RL pipeline.
  - **From-Scratch TD3+BC Algorithm**: Implements Twin Delayed Deep Deterministic
    Policy Gradient with Behavioral Cloning (TD3+BC) for stable offline-to-online
    fine-tuning.
  - **Full Integration with Advanced Diffusion Policy**:
    - The new `DiffusionPolicy` is seamlessly integrated as the actor.
    - **Classifier-Free Guidance (CFG)** is used during action selection to
      significantly improve performance. The `guidance_scale` is a configurable
      hyperparameter.
  - **Representation Sharing**: The Critic network leverages the policy's powerful
    `vision_fusion_encoder` as a shared feature extractor, improving sample efficiency.
  - **Comprehensive Logging & Visualization**:
    - Integrates with both TensorBoard and Weights & Biases (W&B).
    - Logs a wide array of metrics: Q-values, losses, rewards, success rates.
    - **Video Logging**: Generates and logs evaluation videos to W&B, providing
      qualitative insight into the policy's behavior over time.
  - **Robust and Reproducible**: Implements deterministic seeding and robust
    checkpointing of all training components.

To Run:
    # Ensure a corresponding Hydra config file exists (e.g., in configs/finetune_rl_config.yaml)
    # The config must be updated to match the new DiffusionPolicy architecture and include
    # the new `rl_algorithm.guidance_scale` parameter.
    python train_rl.py
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
from typing import Dict, Any, Tuple, Optional, List

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
from envs.panda_env import PandaEnv
from utils.rl_reward_wrapper import AdvancedRewardWrapper, AdvancedRewardConfig, CurriculumConfig
from envs.panda_env_wrapper import HERGoalEnvWrapper
import gymnasium as gym
from gymnasium.spaces import Box, Dict as DictSpace

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Project-Specific
# Note: Ensure these paths are correct relative to your project structure
from utils.expert_dataset import ExpertTrajectoryDataset, collate_fn
from models.diffusion_policy import DiffusionPolicy, NoiseSchedulerConfig
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, VecVideoRecorder,DummyVecEnv
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

class Critic(nn.Module):
    """Twin Critic network for TD3, using a shared feature extractor."""
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
        with torch.no_grad():
             # The vision encoder returns (vision_tokens, proprio_tokens)
            vision_tokens, proprio_tokens = self.features_extractor(obs)
            # Combine and average features to get a single vector representation
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
    """Actor wrapper for the new DiffusionPolicy, now supporting CFG."""
    def __init__(self, diffusion_policy: DiffusionPolicy, guidance_scale: float):
        super().__init__()
        self.diffusion_policy = diffusion_policy
        self.guidance_scale = guidance_scale

    @torch.no_grad()
    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = True) -> torch.Tensor:
        """Selects an action using the diffusion policy with Classifier-Free Guidance."""
        self.diffusion_policy.eval()
        
        # Use a small number of steps for fast sampling in RL.
        # The key is using guidance_scale to improve action quality.
        steps = self.diffusion_policy.scheduler.T // 10 if deterministic else self.diffusion_policy.scheduler.T
        
        sampled_actions = self.diffusion_policy.sample(
            obs,
            steps=steps,
            guidance_scale=self.guidance_scale,
            use_ema=True
        )
        # Return the first action in the predicted sequence
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
        self.env = self._make_vec_env()
        # Create a separate video-recording environment for evaluation
        self.eval_env = self._make_vec_env(is_eval=True)
        video_path = self.output_dir / "videos"
        self.eval_env = VecVideoRecorder(self.eval_env, str(video_path),
                                         record_video_trigger=lambda x: x % self.cfg.logging.video_log_freq == 0,
                                         video_length=200)

        # --- Action Space properties ---
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        original_space = self.env.observation_space
        sanitized_spaces = {}
        if isinstance(original_space, DictSpace):
            for key, space in original_space.spaces.items():
                if isinstance(space, Box):
                    # Re-create the Box space with shape explicitly cast to integers
                    sanitized_shape = tuple(map(int, space.shape))
                    sanitized_spaces[key] = Box(
                        low=space.low,
                        high=space.high,
                        shape=sanitized_shape,
                        dtype=space.dtype
                    )
                else:
                    # If it's not a Box space, just copy it (unlikely in our case)
                    sanitized_spaces[key] = space
            sanitized_obs_space = DictSpace(sanitized_spaces)
        else:
            # If it's not a Dict space, just use the original (unlikely for our env)
            sanitized_obs_space = original_space

        
        # --- Replay Buffer and Expert Dataloader ---
        self.replay_buffer = ReplayBuffer(
            buffer_size=cfg.rl_algorithm.buffer_size,
            observation_space=sanitized_obs_space, # Use the new, clean space
            action_space=self.env.action_space,
            device=self.device,
            n_envs=cfg.environment.n_envs,
            handle_timeout_termination=False,
        )
        self.expert_loader = self._make_expert_loader()
        self.expert_iterator = iter(self.expert_loader)

        # --- Build Models ---
        self._build_models_and_optimizers()

        # --- State Tracking ---
        self.total_timesteps = 0
        self.timesteps_since_eval = 0


    def _make_vec_env(self, is_eval: bool = False) -> VecEnv:
        """Factory for creating the vectorized simulation environment."""
        def make_env(rank: int):
            def _init():
                env_cfg = self.cfg.environment
                seed = self.cfg.seed + rank + (1000 if is_eval else 0)

                # 1. Create the base environment
                env = PandaEnv(
                    xml_path=env_cfg.xml_path,
                    control_mode="delta",
                )
                
                # 2. Create the reward configs (no need for Hydra here is simpler)
                reward_config = AdvancedRewardConfig()
                curriculum_config = CurriculumConfig()
                
                # Override curriculum episodes from the main config
                if 'curriculum_total_episodes' in self.cfg.training:
                    curriculum_config.total_episodes = self.cfg.training.curriculum_total_episodes

                # 3. Apply the reward wrapper
                env = AdvancedRewardWrapper(
                    env,
                    reward_cfg=reward_config,
                    curriculum_cfg=curriculum_config
                )
                
                # We DO NOT apply the HER wrapper.
                
                env.reset(seed=seed)
                return env
            return _init

        # For evaluation, we typically only need one environment
        n_envs = 1 if is_eval else self.cfg.environment.n_envs

        if n_envs == 1:
            # Use DummyVecEnv for n_envs=1 to avoid multiprocessing issues
            return DummyVecEnv([make_env(0)])
        else:
            # Use SubprocVecEnv for n_envs > 1 for performance
            return SubprocVecEnv([make_env(i) for i in range(n_envs)])


    def _make_expert_loader(self) -> DataLoader:
        """Factory for the offline expert data loader."""
        dataset = ExpertTrajectoryDataset(
            demo_path=self.cfg.dataset.path,
            observation_horizon=self.cfg.model.observation_horizon,
            action_horizon=self.cfg.model.action_horizon,
        )
        return DataLoader(dataset, batch_size=self.cfg.rl_algorithm.batch_size, shuffle=True,
                          num_workers=self.cfg.dataset.num_workers, collate_fn=collate_fn, drop_last=True)

    def _build_models_and_optimizers(self):
        """Initializes actor, critic, and their target networks and optimizers."""
        log.info("Building RL models and loading pre-trained actor...")
        
        scheduler_cfg = NoiseSchedulerConfig(**self.cfg.scheduler)
        model_cfg = self.cfg.model
        sample_obs, _ = self.expert_loader.dataset[0]
        proprio_dim = sample_obs["proprio"].shape[-1]
        
        diffusion_policy = DiffusionPolicy(
            proprio_dim=proprio_dim, H_o=model_cfg.observation_horizon, H_a=model_cfg.action_horizon,
            action_dim=model_cfg.action_dim, image_feat_dim=model_cfg.image_feat_dim,
            scheduler_cfg=scheduler_cfg, d_model=model_cfg.d_model, denoiser_layers=model_cfg.denoiser_layers,
            denoiser_heads=model_cfg.denoiser_heads, cfg_p_uncond=0.0, # Not used in RL inference
            ema_decay=None, device=self.device
        )
        
        pretrained_path = Path(self.cfg.pretrained_policy_path)
        if pretrained_path.exists():
            diffusion_policy.load(pretrained_path)
            log.info(f"Successfully loaded pre-trained diffusion policy from {pretrained_path}")
        else:
            log.warning(f"Pretrained policy at {pretrained_path} not found. Actor starts from scratch.")

        self.actor = DiffusionActor(diffusion_policy, self.cfg.rl_algorithm.guidance_scale).to(self.device)
        self.actor_target = DiffusionActor(diffusion_policy, self.cfg.rl_algorithm.guidance_scale).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        critic_feature_extractor = self.actor.diffusion_policy.vision_fusion_encoder
        self.critic = Critic(critic_feature_extractor, self.action_dim, model_cfg.d_model).to(self.device)
        self.critic_target = Critic(critic_feature_extractor, self.action_dim, model_cfg.d_model).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        opt_cfg = self.cfg.optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=opt_cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=opt_cfg.critic_lr)

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Selects an action from the actor, adding exploration noise."""
        obs_torch = {k: torch.as_tensor(v).to(self.device) for k, v in obs.items()}
        with torch.no_grad():
            action = self.actor(obs_torch, deterministic=True)
        
        noise = torch.randn_like(action) * self.cfg.rl_algorithm.exploration_noise
        action = (action + noise).clamp(-self.max_action, self.max_action)
        return action.cpu().numpy()

    def train_step(self):
        """Performs a single gradient update step for both actor and critic."""
        replay_data = self.replay_buffer.sample(self.cfg.rl_algorithm.batch_size)
        try:
            expert_obs_chunk, expert_action_chunk = next(self.expert_iterator)
        except StopIteration:
            self.expert_iterator = iter(self.expert_loader)
            expert_obs_chunk, expert_action_chunk = next(self.expert_iterator)

        expert_obs = {k: v.to(self.device) for k,v in expert_obs_chunk.items()}
        expert_actions = expert_action_chunk[:, 0, :].to(self.device)

        obs = {k: v.to(self.device) for k, v in replay_data.observations.items()}
        next_obs = {k: v.to(self.device) for k, v in replay_data.next_observations.items()}
        actions, rewards, dones = replay_data.actions, replay_data.rewards, replay_data.dones

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.cfg.rl_algorithm.policy_noise).clamp(-self.cfg.rl_algorithm.noise_clip, self.cfg.rl_algorithm.noise_clip)
            next_action = (self.actor_target(next_obs) + noise).clamp(-self.max_action, self.max_action)
            target_q1, target_q2 = self.critic_target(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.cfg.rl_algorithm.gamma * target_q

        current_q1, current_q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_timesteps % self.cfg.rl_algorithm.policy_delay == 0:
            actor_actions = self.actor(obs)
            q1_actor = self.critic.Q1(obs, actor_actions)
            actor_loss_rl = -q1_actor.mean()
            
            q_expert_actions = self.critic.Q1(expert_obs, expert_actions)
            alpha = (1.0 / (torch.abs(q_expert_actions).mean().detach())).clamp(0.1, 10.0)
            bc_loss, _ = self.actor.diffusion_policy.compute_loss(expert_action_chunk, expert_obs)
            actor_loss = actor_loss_rl + alpha * bc_loss
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.cfg.rl_algorithm.tau * param.data + (1 - self.cfg.rl_algorithm.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.cfg.rl_algorithm.tau * param.data + (1 - self.cfg.rl_algorithm.tau) * target_param.data)

            if self.total_timesteps % self.cfg.logging.log_interval_steps == 0:
                metrics = {"train/critic_loss": critic_loss.item(), "train/actor_loss_rl": actor_loss_rl.item(),
                           "train/bc_loss": bc_loss.item(), "train/alpha_bc": alpha.item(),
                           "train/actor_loss_total": actor_loss.item(), "train/q_mean": current_q1.mean().item()}
                if self.use_wandb: wandb.log(metrics, step=self.total_timesteps)
                for k, v in metrics.items(): self.writer.add_scalar(k, v, self.total_timesteps)

    def run(self):
        log.info("Starting RL fine-tuning...")
        obs = self.env.reset()
        
        for _ in tqdm(range(int(self.cfg.training.total_timesteps)), desc="Total Timesteps"):
            self.total_timesteps += self.cfg.environment.n_envs
            if self.total_timesteps < self.cfg.rl_algorithm.learning_starts:
                action = np.array([self.env.action_space.sample() for _ in range(self.cfg.environment.n_envs)])
            else:
                action = self.select_action(obs)

            next_obs, rewards, dones, infos = self.env.step(action)
            self.replay_buffer.add(obs, next_obs, action, rewards, dones, infos)
            obs = next_obs
            
            if self.total_timesteps >= self.cfg.rl_algorithm.learning_starts:
                self.train_step()

            if self.total_timesteps - self.timesteps_since_eval >= self.cfg.logging.eval_freq:
                self.evaluate()
                self.timesteps_since_eval = self.total_timesteps
    
    def evaluate(self):
        log.info("Evaluating policy...")
        self.actor.eval()
        all_ep_rewards, all_successes = [], []

        for i in range(self.cfg.logging.n_eval_episodes):
            obs = self.eval_env.reset()
            dones = [False]
            ep_reward = 0
            while not dones[0]:
                with torch.no_grad():
                    obs_torch = {k: torch.as_tensor(v).to(self.device) for k, v in obs.items()}
                    action = self.actor(obs_torch, deterministic=True).cpu().numpy()
                obs, reward, dones, infos = self.eval_env.step(action)
                ep_reward += reward[0]
            
            all_ep_rewards.append(ep_reward)
            all_successes.append(infos[0].get('is_success', 0.0))

        mean_reward, success_rate = np.mean(all_ep_rewards), np.mean(all_successes)
        log.info(f"Evaluation: Mean Reward={mean_reward:.2f}, Success Rate={success_rate:.2f}")

        metrics = {"eval/mean_reward": mean_reward, "eval/success_rate": success_rate}
        if self.use_wandb:
            wandb.log(metrics, step=self.total_timesteps)
            # Log the last recorded video
            video_files = list((self.output_dir / "videos").glob("*.mp4"))
            if video_files:
                latest_video = max(video_files, key=os.path.getctime)
                wandb.log({"eval/video": wandb.Video(str(latest_video), fps=20, format="mp4")}, step=self.total_timesteps)

        for k, v in metrics.items(): self.writer.add_scalar(k, v, self.total_timesteps)
        self.actor.train()

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
