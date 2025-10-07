import torch
import torch.nn as nn
from typing import Dict, Tuple, Type
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from models.bc_policy import BCNet
from models.custom_sb3_extractor import BCFeaturesExtractor

class ResidualNet(nn.Module):
    """
    A small MLP that learns a residual correction.
    The output is tanh-activated and scaled to ensure initial corrections are small.
    """
    def __init__(self, latent_dim: int, action_dim: int, initial_scale: float = 0.01):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        # Learnable scaling factor, initialized small
        self.log_scale = nn.Parameter(torch.log(torch.tensor([initial_scale])), requires_grad=True)

    def forward(self, latent_features: torch.Tensor) -> torch.Tensor:
        residual = self.net(latent_features)
        return residual * torch.exp(self.log_scale)

# In models/residual_policy.py

class AdaptiveResidualPolicy(ActorCriticPolicy):
    """
    An SB3 Actor-Critic Policy that combines a frozen BC base with a learned RL residual.
    
    1.  It uses the standard `BCFeaturesExtractor` to get latent features from observations.
    2.  The value function (`vf_net`) is a standard MLP head.
    3.  The policy function is composed of two parts:
        a. A frozen `BCNet` that provides a base action (`bc_action`).
        b. A `ResidualNet` that learns a correction (`residual_action`).
    4.  The final action is `final_action = bc_action + residual_action`.
    """
    def __init__(self, observation_space, action_space, lr_schedule, bc_model: BCNet, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule,
                         features_extractor_class=BCFeaturesExtractor,
                         features_extractor_kwargs=kwargs.pop("features_extractor_kwargs", {}),
                         **kwargs)

        # <--- START OF FIX: MANUALLY SET action_dim --->
        # The parent class stores the action_space, not action_dim. We extract it here.
        self.action_dim = action_space.shape[0]
        # <--- END OF FIX --->
        
        self.bc_model = bc_model
        # Freeze the BC model's parameters
        for param in self.bc_model.parameters():
            param.requires_grad = False
        
        # RL components for residual and value function
        self.residual_net = ResidualNet(self.features_dim, self.action_dim)
        self.value_net = nn.Sequential(
            nn.Linear(self.features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.action_net = self.residual_net # For clarity in SB3 hooks
        self.latent_dim_pi = self.features_dim
        self.latent_dim_vf = self.features_dim