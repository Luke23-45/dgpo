# FILE: algos/diffusion_ppo.py
# DEFINITIVE, COMPLETE VERSION

"""
This file defines the DiffusionPPO algorithm, a custom PPO agent, and its
corresponding DiffusionActorCriticPolicy, designed to fine-tune a DiffusionPolicy
with a dual-stream auxiliary Behavioral Cloning (BC) loss.
"""

import logging
from typing import Iterator, Dict, Tuple, Any

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule

# Assuming diffusion_policy is in models/diffusion_policy.py
from models.diffusion_policy import DiffusionPolicy 

# Configure logging
logger = logging.getLogger(__name__)

# --- 1. The Custom Policy Class ---

class DiffusionActorCriticPolicy(ActorCriticPolicy):
    """
    A custom Actor-Critic policy for SB3 that uses our DiffusionPolicy as the actor.
    
    This class is the bridge between the SB3 framework and our custom generative model.
    It ensures that the DiffusionPolicy is correctly instantiated and that its output
    is compatible with the PPO algorithm's expectations.
    """
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule: Schedule,
        # Custom arguments for our diffusion actor
        diffusion_policy_class: type[DiffusionPolicy],
        diffusion_policy_kwargs: Dict[str, Any],
        **kwargs,
    ):
        self.diffusion_policy_class = diffusion_policy_class
        self.diffusion_policy_kwargs = diffusion_policy_kwargs
        
        # We must call the parent constructor first.
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        
        # We disable the default SB3 actor network, as we are replacing it.
        self.action_net = None 
        
        # Now, create and assign our custom diffusion actor
        self._build_actor()

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Builds the policy network. This is called by the parent __init__.
        We let the parent build the feature extractor and value network as usual.
        """
        super()._build(lr_schedule)
        # The actor is built separately in __init__ after super call.

    def _build_actor(self) -> None:
        """Instantiates our custom DiffusionPolicy as the actor."""
        # The diffusion policy needs the feature extractor, which is created in super()._build()
        self.diffusion_policy_kwargs['obs_feature_extractor'] = self.features_extractor
        self.actor = self.diffusion_policy_class(**self.diffusion_policy_kwargs)
        # SB3 uses self.action_net for some operations, so we link it.
        self.action_net = self.actor

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The main forward pass for both training and inference.
        """
        # Get latent features for the value function (critic)
        _, latent_vf, _ = self._get_latent(obs)
        values = self.value_net(latent_vf)
        
        # Get actions directly from our diffusion actor's differentiable forward pass.
        # The actor takes the raw observation dictionary.
        actions = self.actor(obs)
        
        # PPO requires a log_prob for its loss calculation. We create a "pseudo-distribution"
        # centered on our deterministic action. The standard deviation can be a learned parameter.
        action_log_std = self.log_std.expand_as(actions)
        action_dist = torch.distributions.Normal(actions, torch.exp(action_log_std))
        
        # For the PPO loss, we need the log probability of the action we just took.
        log_prob = action_dist.log_prob(actions).sum(axis=-1)
        
        return actions, values, log_prob
        
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions and compute entropy. This is used by PPO during training.
        """
        # Get latent features for the value function
        _, latent_vf, _ = self._get_latent(obs)
        values = self.value_net(latent_vf)

        # The actor's forward pass gives the mean of our pseudo-distribution.
        mean_actions = self.actor(obs)
        
        action_log_std = self.log_std.expand_as(mean_actions)
        action_dist = torch.distributions.Normal(mean_actions, torch.exp(action_log_std))
        
        # Calculate the log probability of the actions that were *actually taken* in the rollout.
        log_prob = action_dist.log_prob(actions).sum(axis=-1)
        entropy = action_dist.entropy().sum(axis=-1)
        
        return values, log_prob, entropy

    def compute_loss(self, obs: Dict[str, torch.Tensor], clean_action: torch.Tensor) -> torch.Tensor:
        """A proxy method to call the underlying diffusion actor's BC loss function."""
        # Ensure the policy is in training mode for the BC loss calculation
        self.train()
        return self.actor.compute_loss(obs, clean_action)

# --- 2. The Custom PPO Agent ---

class DiffusionPPO(PPO):
    """
    A PPO agent that fine-tunes a DiffusionPolicy with a dual-stream auxiliary BC loss.
    """
    def __init__(
        self,
        policy,
        env,
        offline_expert_loader: DataLoader,
        online_expert_loader: DataLoader,
        lambda_bc: float = 1.0,
        lambda_annealing_steps: int = 1_000_000,
        **kwargs,
    ):
        # We must specify our custom policy class here
        if policy == "DiffusionActorCriticPolicy":
            policy = DiffusionActorCriticPolicy
        
        super().__init__(policy=policy, env=env, **kwargs)
        
        self.offline_expert_loader = offline_expert_loader
        self.online_expert_loader = online_expert_loader
        self._offline_iter: Iterator = iter(self.offline_expert_loader)
        self._online_iter: Iterator = iter(self.online_expert_loader)
        
        self.initial_lambda_bc = lambda_bc
        self.current_lambda_bc = lambda_bc
        self.lambda_annealing_steps = lambda_annealing_steps

    def _get_expert_batch(self, loader_type: str) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Safely retrieves the next batch from the specified expert dataloader."""
        if loader_type == 'offline':
            try:
                return next(self._offline_iter)
            except StopIteration:
                self._offline_iter = iter(self.offline_expert_loader)
                return next(self._offline_iter)
        else:
            return next(self._online_iter)

    def train(self) -> None:
        """
        Overrides the standard SB3 `train` method to add the auxiliary BC loss.
        """
        # Update schedules
        self._update_learning_rate(self.policy.optimizer)
        progress = self.num_timesteps / self.lambda_annealing_steps
        self.current_lambda_bc = self.initial_lambda_bc * max(0.0, 1.0 - progress)

        # 1. Standard PPO Update Step
        super().train()

        # 2. Auxiliary BC Regularization Step
        if self.current_lambda_bc > 0:
            bc_losses = []
            for _ in range(self.n_epochs):
                offline_obs, offline_actions = self._get_expert_batch('offline')
                online_obs, online_actions = self._get_expert_batch('online')

                offline_obs = {k: v.to(self.device) for k, v in offline_obs.items()}
                offline_actions = offline_actions.to(self.device)
                online_obs = {k: v.to(self.device) for k, v in online_obs.items()}
                online_actions = online_actions.to(self.device)
                
                bc_loss_offline = self.policy.compute_loss(offline_obs, offline_actions)
                bc_loss_online = self.policy.compute_loss(online_obs, online_actions)
                
                bc_loss = 0.5 * (bc_loss_offline + bc_loss_online)
                total_aux_loss = self.current_lambda_bc * bc_loss
                
                self.policy.optimizer.zero_grad()
                total_aux_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                
                bc_losses.append(bc_loss.item())

            # 3. Logging
            mean_bc_loss = sum(bc_losses) / len(bc_losses)
            self.logger.record("train/bc_loss", mean_bc_loss)
            self.logger.record("train/lambda_bc", self.current_lambda_bc)