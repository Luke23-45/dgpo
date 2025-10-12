import torch
from torch import nn
from typing import Dict, Any
from sb3_contrib import TQC
from sb3_contrib.tqc.policies import TQCPolicy, Actor
from stable_baselines3.common.type_aliases import Schedule
from models.diffusion_policy import DiffusionPolicy # Assumes this is our high-quality implementation

class DiffusionActor(Actor):
    """A custom SB3 Actor that uses our DiffusionPolicy."""
    def __init__(self, observation_space, action_space, features_extractor, features_dim, **kwargs):
        super().__init__(observation_space, action_space, features_extractor, features_dim)
        
        # The actor's network is our entire DiffusionPolicy
        self.mu = DiffusionPolicy(**kwargs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # The forward pass is the differentiable one-step sample
        return self.mu(obs)
    
    def compute_loss(self, obs: Dict[str, torch.Tensor], clean_action: torch.Tensor) -> torch.Tensor:
        """Proxy to the diffusion model's BC loss."""
        return self.mu.compute_loss(obs, clean_action)

class DiffusionTQCPolicy(TQCPolicy):
    """A custom TQC Policy that knows how to use our DiffusionActor."""
    def __init__(self, observation_space, action_space, lr_schedule: Schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def make_actor(self, features_extractor: nn.Module) -> DiffusionActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return DiffusionActor(self.observation_space, self.action_space, features_extractor=features_extractor, **actor_kwargs)

class DiffusionTQC(TQC):
    """
    A custom TQC algorithm that adds an Advantage-Weighted Auxiliary BC Loss.
    """
    def __init__(self, *args, offline_expert_loader, lambda_bc=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.offline_expert_loader = offline_expert_loader
        self._offline_iter = iter(self.offline_expert_loader)
        self.lambda_bc = lambda_bc

    def _get_expert_batch(self):
        try:
            return next(self._offline_iter)
        except StopIteration:
            self._offline_iter = iter(self.offline_expert_loader)
            return next(self._offline_iter)

    def train(self, gradient_steps: int, batch_size: int):
        # First, run the standard TQC training step
        super().train(gradient_steps, batch_size)

        # Now, perform the auxiliary BC update
        self.policy.set_training_mode(True)
        for _ in range(gradient_steps):
            # 1. Sample expert data
            expert_obs, expert_actions = self._get_expert_batch()
            expert_obs = {k: v.to(self.device) for k, v in expert_obs.items()}
            expert_actions = expert_actions.to(self.device)
            
            # 2. Get Q-value estimates for the expert actions (the "Advantage")
            with torch.no_grad():
                # Use the critic target to get a stable Q estimate
                q_values = torch.cat(self.policy.critic_target(expert_obs, expert_actions), dim=1)
                # Take the minimum across critics and average across quantiles for a scalar Q
                q_value_estimate = q_values.min(dim=1, keepdim=True)[0].mean(dim=2, keepdim=True)

            # 3. Calculate Q-weighted BC loss
            # This is a simplified form of AWAC: weight the loss by exponentiated Q-values
            weights = torch.exp(q_value_estimate / self.lambda_bc).detach()
            weights = torch.clamp(weights, max=100.0) # Clamp for stability
            
            bc_loss = self.policy.actor.compute_loss(expert_obs, expert_actions)
            weighted_loss = (bc_loss * weights).mean()
            
            # 4. Optimize the actor
            self.policy.actor.optimizer.zero_grad()
            weighted_loss.backward()
            self.policy.actor.optimizer.step()