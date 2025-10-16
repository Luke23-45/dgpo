"""
train_rl.py

High-quality RL fine-tuning script integrating:
- HER for sample efficiency
- TQC (distributional off-policy RL)
- Auxiliary Advantage‐Weighted BC (AWBC) regularization
- Pretrained diffusion actor warm start
- Full logging, checkpointing, reproducibility

Usage:
    python train_rl.py --config path/to/finetune_config.yaml
"""

from __future__ import annotations
import os
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import gymnasium as gym
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.her import HerReplayBuffer

# SB3 TQC import
from sb3_contrib import TQC
from sb3_contrib.tqc.tqc import TQCPolicy

# Project imports (adjust as needed)
from envs.panda_env import PandaEnv
from envs.panda_env_wrapper import GoalPandaEnv
from utils.rl_reward_wrapper import RLRewardWrapper
from utils.expert_dataset import ExpertTrajectoryDataset, collate_fn
from models.custom_sb3_extractor import BCFeaturesExtractor
from models.diffusion_policy import DiffusionPolicy

logger = logging.getLogger("train_rl")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


class DiffusionActor(nn.Module):
    """
    Actor wrapper for SB3 that uses a DiffusionPolicy as the action generator.
    This class routes observations through the diffusion model to sample or
    deterministically get actions.
    """
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        features_extractor: nn.Module,
        features_dim: int,
        diffusion_policy: DiffusionPolicy,
    ):
        super().__init__()
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self.action_dim = action_space.shape[0]
        self.mu = diffusion_policy  # diffusion actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to produce an action. Called by SB3 policy.
        We interpret this as the *deterministic* action for the policy.
        """
        # Extract features (observations to cond)
        # features_extractor should return a dict or tensor appropriate for diffusion_policy
        cond = self.features_extractor(obs)
        with torch.no_grad():
            # Use deterministic sampling for exploitation
            a = self.mu.sample_deterministic(cond, steps=1)
        return a

    def compute_bc_loss(self, obs: Dict[str, torch.Tensor], expert_actions: torch.Tensor) -> torch.Tensor:
        """
        Compute the BC loss (diffusion MSE / noise prediction loss) on given expert actions.
        """
        return self.mu(expert_actions, obs)[0]  # loss, _ = forward

    def get_log_prob(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
        """
        Proxy log probability for RL. Approximation via diffusion log_prob_approx.
        """
        return self.mu.log_prob_approx(actions, obs)


class DiffusionTQCPolicy(TQCPolicy):
    """
    Custom TQCPolicy that uses our DiffusionActor as the actor branch.
    """
    def __init__(self, *args, actor_kwargs: Dict[str, Any], **kwargs):
        super().__init__(*args, **kwargs)
        self.actor_kwargs = actor_kwargs

    def make_actor(self, features_extractor: nn.Module) -> DiffusionActor:
        actor = DiffusionActor(
            observation_space=self.observation_space,
            action_space=self.action_space,
            features_extractor=features_extractor,
            features_dim=self.features_extractor.features_dim,
            diffusion_policy=self.actor_kwargs["diffusion_policy"],
        )
        return actor


class DiffusionTQC(TQC):
    """
    TQC with integrated Advantage-Weighted Behavior Cloning (AWBC) auxiliary loss.
    The actor updates include a weighted BC loss toward expert data.
    """

    def __init__(
        self,
        *args,
        offline_expert_loader: DataLoader,
        lambda_bc: float = 1.0,
        bc_clip: float = 100.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.offline_expert_loader = offline_expert_loader
        self._offline_iter = iter(self.offline_expert_loader)
        self.lambda_bc = lambda_bc
        self.bc_clip = bc_clip

    def _get_expert_batch(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        try:
            obs, actions = next(self._offline_iter)
        except StopIteration:
            self._offline_iter = iter(self.offline_expert_loader)
            obs, actions = next(self._offline_iter)
        return obs, actions

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """
        Override TQC.train to insert BC updates after standard RL updates.
        """
        # Perform TQC’s standard actor/critic update
        super().train(gradient_steps, batch_size)

        # Now auxiliary BC updates
        actor = self.policy.actor
        device = self.device

        actor.train()  # ensure actor in train mode

        for _ in range(gradient_steps):
            expert_obs, expert_actions = self._get_expert_batch()
            expert_actions = expert_actions.to(device)
            # Move obs to correct device
            expert_obs = {k: v.to(device) for k, v in expert_obs.items()}

            # Compute Q(s, a) for expert transitions via critic target(s)
            with torch.no_grad():
                # critic_target returns quantiles shape (B, n_critics, n_quantiles)
                q_targs = self.policy.critic_target(expert_obs, expert_actions)
                # Flatten: (B, total_quantiles) then take min over critics or median
                q_concat = torch.cat(q_targs, dim=1)  # may be shape (B, sum_q)
                # A simple baseline: subtract average value OR minimum quantile
                q_min = q_concat.min(dim=1, keepdim=True)[0]
                # Use baseline as min → advantage = q - q_min
                advantages = q_concat.mean(dim=1, keepdim=True) - q_min
                # Normalize adv
                adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Weight BC loss
            weights = torch.exp(adv_norm * (1.0 / self.lambda_bc))
            weights = torch.clamp(weights, max=self.bc_clip)

            # Compute BC loss (diffusion loss)
            bc_loss = actor.compute_bc_loss(expert_obs, expert_actions)
            weighted_loss = (bc_loss * weights).mean()

            # Optimize actor
            self.policy.actor.optimizer.zero_grad()
            weighted_loss.backward()
            # Optionally gradient clip
            torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), max_norm=1.0)
            self.policy.actor.optimizer.step()

            logger.debug(f"BC auxiliary step: bc_loss={bc_loss.mean().item():.6f}, weights mean={weights.mean().item():.4f}")

        actor.eval()


def train_rl(
    config: Dict[str, Any],
):
    """
    Main driver to setup RL fine-tuning from config dict.
    """
    # Setup paths
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=4))

    # Seeds
    seed = int(config.get("seed", 0))
    set_random_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")

    # 1. Build vectorized environments
    def make_env_fn():
        env = PandaEnv(xml_path=config["xml_path"], control_mode="delta", max_episode_steps=config["max_episode_steps"])
        env = RLRewardWrapper(env)
        env = GoalPandaEnv(env)
        return env

    env = SubprocVecEnv([make_env_fn for _ in range(config["n_envs"])])

    # 2. Offline expert loader
    offline_ds = ExpertTrajectoryDataset(Path(config["demo_path"]))
    offline_loader = DataLoader(
        offline_ds,
        batch_size=config["bc_batch_size"],
        shuffle=True,
        num_workers=config.get("bc_num_workers", 4),
        collate_fn=collate_fn,
        drop_last=True
    )

    # 3. Policy kwargs (actor + critic)
    # First, build diffusion policy instance
    diffusion_policy = DiffusionPolicy(
        denoiser=config["actor"]["denoiser"],
        action_dim=config["env_action_dim"],
        cond_embed_fn=config["actor"]["cond_embed_fn"],
        scheduler=config["actor"]["scheduler"],
        device=device,
        ema_decay=config["actor"].get("ema_decay", 0.0),
    )

    policy_kwargs = {
        "actor_kwargs": {
            "diffusion_policy": diffusion_policy
        },
        "net_arch": dict(pi=config["net_arch"]["pi"], qf=config["net_arch"]["qf"]),
        "features_extractor_class": BCFeaturesExtractor,
        "features_extractor_kwargs": {"observation_space": env.observation_space["observation"]},
    }

    # 4. Instantiate DiffusionTQC
    agent = DiffusionTQC(
        policy=DiffusionTQCPolicy,
        env=env,
        offline_expert_loader=offline_loader,
        lambda_bc=config["lambda_bc"],
        bc_clip=config.get("bc_clip", 100.0),
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": config["her_n_sampled_goal"],
            "goal_selection_strategy": config["her_strategy"],
            "online_sampling": True,
            "max_episode_length": config["max_episode_steps"],
        },
        buffer_size=config["buffer_size"],
        learning_starts=config["learning_starts"],
        batch_size=config["rl_batch_size"],
        learning_rate=config["rl_lr"],
        gamma=config["gamma"],
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(output_dir / "logs"),
        seed=seed,
        device=device,
        verbose=config.get("verbose", 1),
    )

    # 5. Load pretrained actor weights (if available)
    pretrained_path = Path(config["pretrained_policy_path"])
    if pretrained_path.exists():
        state = torch.load(pretrained_path, map_location=device)
        agent.policy.actor.mu.load_state_dict(state)
        logger.info("Loaded pretrained diffusion actor weights.")
    else:
        logger.warn("Pretrained actor not found; starting from scratch.")

    # 6. Checkpoint callback
    ckpt_callback = CheckpointCallback(
        save_freq=config.get("save_freq", 50000),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="rl_model",
        save_replay_buffer=True
    )

    # 7. Begin RL learning
    logger.info("Starting RL fine-tuning...")
    agent.learn(total_timesteps=config["total_timesteps"], callback=ckpt_callback)
    final_path = output_dir / "final_model.zip"
    agent.save(final_path)
    logger.info(f"Training done; model saved to {final_path}")

    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL fine-tuning with Diffusion + TQC + AWBC")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    train_rl(cfg)


if __name__ == "__main__":
    main()
