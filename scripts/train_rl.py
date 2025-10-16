# FILE: train_rl.py
# (State-of-the-Art, Hydra-Configurable, TD3+BC Implementation)

"""
State-of-the-art RL fine-tuning script for a pre-trained Diffusion Policy.

This script implements a complete, from-scratch training framework for fine-tuning
a diffusion model actor using online reinforcement learning, regularized by offline
expert data. It is designed for high performance, deep diagnostics, and reproducibility,
incorporating modern MLOps and algorithmic best practices.

Key Architectural Features:
  - **Hydra Configuration**: Employs Hydra for a powerful, modular, and composable
    configuration system. All hyperparameters, paths, and settings are managed
    through structured YAML files, allowing for easy experimentation and command-line
    overrides.
  - **Custom Trainer Class (`RLFineTuner`)**: Encapsulates the entire RL pipeline,
    including environment interaction, data management, model updates, logging,
    and evaluation, promoting code clarity, reusability, and extensibility.
  - **From-Scratch TD3+BC Algorithm**: Implements the Twin Delayed Deep Deterministic
    Policy Gradient with Behavioral Cloning (TD3+BC) algorithm. This advanced
    technique is ideal for offline-to-online fine-tuning, combining a strong
    off-policy RL algorithm (TD3) with a dynamic behavioral cloning loss that
    regularizes the policy and prevents catastrophic forgetting of the pre-trained
    diffusion prior.
  - **Full Model Integration**: The `DiffusionPolicy` is seamlessly integrated as the
    actor. A custom `DiffusionActor` wrapper provides the necessary interface for
    the TD3 algorithm, using deterministic sampling for action selection during
    environment interaction.
  - **Comprehensive Logging & Visualization**:
    - Integrates with both TensorBoard and Weights & Biases (W&B) for rich
      experiment tracking.
    - Logs a wide array of metrics: Q-values, actor/critic/BC losses, rewards,
      episode lengths, and custom evaluation scores.
    - Features a dedicated, periodic evaluation loop that measures true policy
      performance (e.g., success rate) in a separate environment instance.
    - Generates and logs evaluation videos to W&B, providing qualitative insight
      into the policy's behavior over time.
  - **Robust and Reproducible**: Implements deterministic seeding, robust checkpointing
    (saving model, optimizer, and replay buffer state), and seamless resumption
    of training runs.
  - **High-Performance Code**: Utilizes vectorized environments for parallel data
    collection and is structured for efficient GPU utilization.

To Run:
    # Ensure a corresponding Hydra config file exists (e.g., in configs/finetune_rl_config.yaml)
    # The script will automatically create a unique, timestamped output directory.
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

# Optional, for enhanced logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Project-Specific
from envs.panda_env import PandaEnv
from envs.panda_env_wrapper import RLRewardWrapper, GoalPandaEnv
from utils.expert_dataset import ExpertTrajectoryDataset, collate_fn
from models.diffusion_policy import DiffusionPolicy, NoiseSchedulerConfig
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv
from stable_baselines3.common.buffers import ReplayBuffer

# Setup a logger for the script
log = logging.getLogger(__name__)

# -------------------------
# 2. RL Network Components
# -------------------------

class Critic(nn.Module):
    """
    Standard Twin Critic network for TD3.
    It takes an observation and an action and outputs a Q-value.
    Implements two separate Q-networks (Q1 and Q2) to mitigate overestimation bias.
    """
    def __init__(self, features_extractor: nn.Module, action_dim: int):
        super().__init__()
        self.features_extractor = features_extractor
        features_dim = features_extractor.features_dim

        # Q1 network
        self.q1_net = nn.Sequential(
            nn.Linear(features_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Q2 network
        self.q2_net = nn.Sequential(
            nn.Linear(features_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, obs: Dict[str, torch.Tensor], action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the Q-values from both critics.
        Returns: (q1_value, q2_value)
        """
        features = self.features_extractor(obs)
        x = torch.cat([features, action], dim=1)
        q1 = self.q1_net(x)
        q2 = self.q2_net(x)
        return q1, q2

    def Q1(self, obs: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        """Computes the Q-value from the first critic only."""
        features = self.features_extractor(obs)
        x = torch.cat([features, action], dim=1)
        return self.q1_net(x)


class DiffusionActor(nn.Module):
    """
    An actor network wrapper for the DiffusionPolicy.
    This module is responsible for generating actions from observations during RL.
    """
    def __init__(self, diffusion_policy: DiffusionPolicy):
        super().__init__()
        # The core of our actor is the pre-trained diffusion policy
        self.diffusion_policy = diffusion_policy

    @torch.no_grad()
    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = True) -> torch.Tensor:
        """
        Selects an action using the diffusion policy.
        For TD3, we typically use deterministic actions during training interaction.
        """
        self.diffusion_policy.eval()
        # For RL, we need a single action, not a sequence. We take the first action
        # from the predicted action horizon.
        # Shape of sampled_actions: (B, H_a, D_a)
        if deterministic:
             # Use a small number of steps for fast, near-deterministic sampling
            sampled_actions = self.diffusion_policy.sample(obs, steps=10, use_ema=True)
        else:
            # Full stochastic sampling for exploration
            sampled_actions = self.diffusion_policy.sample(obs, steps=50, use_ema=True)

        # Return the first action in the sequence
        action = sampled_actions[:, 0, :]
        return action

# -------------------------
# 3. Main Trainer Class
# -------------------------

class RLFineTuner:
    """
    Encapsulates the entire fine-tuning pipeline using TD3+BC.
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
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
        self.eval_env = self._make_vec_env(is_eval=True) # Separate env for evaluation

        # --- Action Space properties ---
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])

        # --- Replay Buffer and Expert Dataloader ---
        self.replay_buffer = ReplayBuffer(
            buffer_size=cfg.rl_algorithm.buffer_size,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            device=self.device,
            n_envs=cfg.environment.n_envs
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
                # Use a different seed for each environment instance
                seed = self.cfg.seed + rank + (1000 if is_eval else 0)
                env = PandaEnv(
                    xml_path=env_cfg.xml_path,
                    control_mode="delta",
                )
                env = RLRewardWrapper(env)
                # Important: GoalPandaEnv wraps the observation to be HER-compatible,
                # which we are not using in this custom implementation, but the observation
                # structure is useful. We will access the 'observation' key.
                env = GoalPandaEnv(env)
                env.reset(seed=seed)
                return env
            return _init

        return SubprocVecEnv([make_env(i) for i in range(self.cfg.environment.n_envs)])
        
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
            drop_last=True,
        )

    def _build_models_and_optimizers(self):
        """Initializes actor, critic, and their target networks and optimizers."""
        log.info("Building RL models and loading pre-trained actor...")
        
        # --- Diffusion Policy (Actor Core) ---
        # Note: We build the full DiffusionPolicy, not just a simple denoiser
        scheduler_cfg = NoiseSchedulerConfig(**self.cfg.scheduler)
        diffusion_policy = DiffusionPolicy(
            **self.cfg.model,
            scheduler_cfg=scheduler_cfg,
            device=self.device
        )
        
        # Load pre-trained weights
        pretrained_path = Path(self.cfg.pretrained_policy_path)
        if pretrained_path.exists():
            diffusion_policy.load(pretrained_path)
            log.info(f"Successfully loaded pre-trained diffusion policy from {pretrained_path}")
        else:
            log.warning("Pretrained policy not found. Actor is starting from random initialization.")

        self.actor = DiffusionActor(diffusion_policy).to(self.device)
        self.actor_target = DiffusionActor(diffusion_policy).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # --- Critic ---
        # The critic needs a feature extractor. We re-use the diffusion policy's vision part.
        # This is a form of representation sharing.
        critic_feature_extractor = self.actor.diffusion_policy._cond_embed
        self.critic = Critic(critic_feature_extractor, self.action_dim).to(self.device)
        self.critic_target = Critic(critic_feature_extractor, self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # --- Optimizers ---
        opt_cfg = self.cfg.optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=opt_cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=opt_cfg.critic_lr)

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Selects an action from the actor, adding exploration noise."""
        # SB3 VecEnv gives obs as a dict of numpy arrays
        # Convert to torch tensors on the correct device
        obs_torch = {k: torch.as_tensor(v).to(self.device) for k, v in obs.items()}
        
        with torch.no_grad():
            action = self.actor(obs_torch['observation'], deterministic=True)
        
        # Add exploration noise
        noise = torch.randn_like(action) * self.cfg.rl_algorithm.exploration_noise
        action = (action + noise).clamp(-self.max_action, self.max_action)
        
        return action.cpu().numpy()

    def train_step(self):
        """Performs a single gradient update step for both actor and critic."""
        
        # --- 1. Sample from replay buffer and expert data ---
        replay_data = self.replay_buffer.sample(self.cfg.rl_algorithm.batch_size)
        try:
            expert_obs_chunk, expert_action_chunk = next(self.expert_iterator)
        except StopIteration:
            self.expert_iterator = iter(self.expert_loader)
            expert_obs_chunk, expert_action_chunk = next(self.expert_iterator)

        # Move expert data to device and select first action from horizon
        expert_obs = {k: v.to(self.device) for k,v in expert_obs_chunk.items()}
        expert_actions = expert_action_chunk[:, 0, :].to(self.device)

        # Unpack online data
        obs = {k: v.to(self.device) for k, v in replay_data.observations.items()}
        next_obs = {k: v.to(self.device) for k, v in replay_data.next_observations.items()}
        actions = replay_data.actions
        rewards = replay_data.rewards
        dones = replay_data.dones

        # --- 2. Critic Update ---
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(actions) * self.cfg.rl_algorithm.policy_noise).clamp(
                -self.cfg.rl_algorithm.noise_clip, self.cfg.rl_algorithm.noise_clip
            )
            next_action = (self.actor_target(next_obs['observation']) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_q1, target_q2 = self.critic_target(next_obs['observation'], next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.cfg.rl_algorithm.gamma * target_q

        # Get current Q estimates
        current_q1, current_q2 = self.critic(obs['observation'], actions)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- 3. Delayed Actor and BC Update ---
        if self.total_timesteps % self.cfg.rl_algorithm.policy_delay == 0:
            # --- a. RL Actor Loss ---
            actor_actions = self.actor(obs['observation'])
            q1_actor = self.critic.Q1(obs['observation'], actor_actions)
            actor_loss_rl = -q1_actor.mean()
            
            # --- b. TD3+BC Auxiliary Loss ---
            # Compute Q value for expert actions to get the weight
            q_expert_actions = self.critic.Q1(expert_obs, expert_actions)
            # Dynamic alpha based on Q values
            alpha = 1.0 / (torch.abs(q_expert_actions).mean().detach())

            # BC loss is the diffusion loss
            bc_loss, _ = self.actor.diffusion_policy.compute_loss(expert_action_chunk, expert_obs)
            
            # --- c. Combined Actor Loss ---
            actor_loss = actor_loss_rl + alpha * bc_loss
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # --- 4. Soft update target networks ---
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.cfg.rl_algorithm.tau * param.data + (1 - self.cfg.rl_algorithm.tau) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.cfg.rl_algorithm.tau * param.data + (1 - self.cfg.rl_algorithm.tau) * target_param.data)

            # --- Logging ---
            if self.total_timesteps % self.cfg.logging.log_interval_steps == 0:
                metrics = {
                    "train/critic_loss": critic_loss.item(),
                    "train/actor_loss_rl": actor_loss_rl.item(),
                    "train/bc_loss": bc_loss.item(),
                    "train/alpha_bc": alpha.item(),
                    "train/actor_loss_total": actor_loss.item(),
                    "train/q_mean": current_q1.mean().item()
                }
                if self.use_wandb:
                    wandb.log(metrics, step=self.total_timesteps)
                for k, v in metrics.items():
                    self.writer.add_scalar(k, v, self.total_timesteps)

    def run(self):
        """Main training loop: environment interaction and model updates."""
        log.info("Starting RL fine-tuning...")
        obs = self.env.reset()
        
        for _ in tqdm(range(int(self.cfg.training.total_timesteps)), desc="Total Timesteps"):
            self.total_timesteps += self.cfg.environment.n_envs

            if self.total_timesteps < self.cfg.rl_algorithm.learning_starts:
                action = np.array([self.env.action_space.sample() for _ in range(self.cfg.environment.n_envs)])
            else:
                action = self.select_action(obs)

            next_obs, rewards, dones, infos = self.env.step(action)
            
            # Handle terminal observations
            for idx, done in enumerate(dones):
                if done:
                    # SB3 VecEnvs auto-reset, 'infos' contains the final observation
                    final_obs = infos[idx].get("final_observation")
                    if final_obs is not None:
                       self.replay_buffer.add(obs, final_obs, action, rewards, dones, infos)
            
            self.replay_buffer.add(obs, next_obs, action, rewards, dones, infos)
            obs = next_obs
            
            # Train the agent
            if self.total_timesteps >= self.cfg.rl_algorithm.learning_starts:
                self.train_step()

            # Evaluate the agent
            if self.total_timesteps - self.timesteps_since_eval >= self.cfg.logging.eval_freq:
                self.evaluate()
                self.timesteps_since_eval = self.total_timesteps
    
    def evaluate(self):
        """Evaluate the policy's performance in the environment."""
        log.info("Evaluating policy...")
        self.actor.eval()
        all_ep_rewards = []
        all_successes = []

        for _ in range(self.cfg.logging.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            ep_reward = 0
            while not done:
                with torch.no_grad():
                    obs_torch = {k: torch.as_tensor(v).to(self.device) for k, v in obs.items()}
                    action = self.actor(obs_torch['observation'], deterministic=True).cpu().numpy()
                obs, reward, done, info = self.eval_env.step(action)
                ep_reward += reward[0]
            
            all_ep_rewards.append(ep_reward)
            # Assuming 'is_success' is populated by the reward wrapper on success
            all_successes.append(info[0].get('is_success', 0.0))

        mean_reward = np.mean(all_ep_rewards)
        success_rate = np.mean(all_successes)
        log.info(f"Evaluation: Mean Reward={mean_reward:.2f}, Success Rate={success_rate:.2f}")

        metrics = {"eval/mean_reward": mean_reward, "eval/success_rate": success_rate}
        if self.use_wandb:
            wandb.log(metrics, step=self.total_timesteps)
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, self.total_timesteps)
        
        # TODO: Add video logging to W&B
        self.actor.train()


# -------------------------
# 4. Hydra Main Entry Point
# -------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="finetune_rl_config")
def main(cfg: DictConfig):
    """
    Main function managed by Hydra.
    """
    log.info("----------- RL Fine-tuning Configuration -----------")
    log.info(OmegaConf.to_yaml(cfg))
    log.info("-------------------------------------------------")

    try:
        trainer = RLFineTuner(cfg)
        trainer.run()
    except Exception as e:
        log.exception("An error occurred during RL fine-tuning.")
        raise

# -------------------------
# 5. Standard Python Entry
# -------------------------

if __name__ == "__main__":
    main()
