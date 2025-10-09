# models/diffusion_policy.py
"""
Implements a conditional Diffusion Policy based on SOTA practices.

This file contains the core components for a Diffusion Policy:
1.  A sinusoidal time embedding module.
2.  A conditional denoiser network (ε_θ).
3.  The main DiffusionPolicy class which orchestrates the components and provides:
    - A differentiable, one-step action sampling method for RL inference.
    - A BC denoising loss method for pre-training and fine-tuning.

The architecture is designed to be modular and is inspired by best practices from
the original Diffusion Policy papers and production-grade libraries like Hugging Face Diffusers.
"""

import math
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.custom_sb3_extractor import BCFeaturesExtractor # Re-use our robust obs encoder

# --- Helper Modules ---

class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embeddings for diffusion timesteps.
    From: https://github.com/lucidrains/denoising-diffusion-pytorch
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """A simple residual block for the denoiser MLP."""
    def __init__(self, size: int):
        super().__init__()
        self.fc = nn.Linear(size, size)
        self.ln = nn.LayerNorm(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.mish(self.ln(self.fc(x)))

# --- Denoiser Network ---

class ConditionalDenoiser(nn.Module):
    """
    The core denoiser network (ε_θ). Predicts the noise added to an action.
    """
    def __init__(self, action_dim: int, obs_feature_dim: int, time_emb_dim: int = 16):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.Mish(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # The main MLP that processes the concatenated inputs
        input_dim = obs_feature_dim + action_dim + time_emb_dim
        self.mid_mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Mish(),
            ResidualBlock(256),
            nn.Linear(256, 256),
            nn.Mish(),
            ResidualBlock(256),
        )
        self.final_mlp = nn.Linear(256, action_dim)

    def forward(
        self,
        obs_features: torch.Tensor,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            obs_features: (B, obs_feature_dim) Conditioning vector from the obs encoder.
            noisy_action: (B, action_dim) The noisy action at step t (x_t).
            timestep: (B,) The current diffusion timestep.

        Returns:
            (B, action_dim) The predicted noise (ε_θ).
        """
        t = self.time_mlp(timestep)
        x = torch.cat([obs_features, noisy_action, t], dim=-1)
        x = self.mid_mlp(x)
        return self.final_mlp(x)


# --- Main Diffusion Policy Class ---

class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        obs_feature_extractor: BCFeaturesExtractor,
        denoiser: ConditionalDenoiser,
        action_dim: int,
        num_diffusion_iters: int = 100,
        noise_schedule: str = 'squaredcos_cap_v2',
    ):
        super().__init__()
        self.obs_feature_extractor = obs_feature_extractor
        self.denoiser = denoiser
        self.action_dim = action_dim
        self.num_diffusion_iters = num_diffusion_iters

        # --- Set up the noise schedule (adapted from Hugging Face Diffusers) ---
        if noise_schedule == 'linear':
            betas = torch.linspace(1e-4, 0.02, num_diffusion_iters)
        elif noise_schedule == 'squaredcos_cap_v2':
            betas = self._squaredcos_cap_v2_schedule(num_diffusion_iters)
        else:
            raise ValueError(f"Unknown noise schedule: {noise_schedule}")
            
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    @staticmethod
    def _squaredcos_cap_v2_schedule(timesteps, s=0.008):
        """Cosine schedule from Improved DDPM paper, with clamping for stability."""
        t = torch.linspace(0, timesteps, timesteps + 1)
        alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def _extract_obs_features(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Helper to run the observation through the feature extractor."""
        return self.obs_feature_extractor(obs)

    # --- Methods for Training (BC Loss) ---

    def compute_loss(self, obs: Dict[str, torch.Tensor], clean_action: torch.Tensor) -> torch.Tensor:
        """
        Computes the Denoising Diffusion Probabilistic Model (DDPM) loss for a batch
        of expert observations and actions.

        Returns:
            The MSE loss between the true and predicted noise.
        """
        B = clean_action.shape[0]
        
        # 1. Sample a random timestep t for each sample in the batch
        t = torch.randint(0, self.num_diffusion_iters, (B,), device=clean_action.device).long()
        
        # 2. Sample a random noise vector ε from a standard normal distribution
        noise = torch.randn_like(clean_action)
        
        # 3. Create the noisy action x_t using the forward diffusion formula
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        noisy_action = (sqrt_alpha_cumprod_t * clean_action + 
                        sqrt_one_minus_alpha_cumprod_t * noise)
        
        # 4. Get the observation features
        obs_features = self._extract_obs_features(obs)
        
        # 5. Predict the noise using the denoiser network
        predicted_noise = self.denoiser(obs_features, noisy_action, t)
        
        # 6. Calculate the loss
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    # --- Methods for Inference (Action Sampling) ---
    
    @torch.no_grad()
    def sample(self, obs: Dict[str, torch.Tensor], num_steps: int = None) -> torch.Tensor:
        """
        Full DDPM sampling loop for inference. This is non-differentiable.
        Used for evaluation or when full generative power is needed.
        """
        if num_steps is None:
            num_steps = self.num_diffusion_iters
        
        B = next(iter(obs.values())).shape[0]
        device = next(self.parameters()).device
        
        # Start with pure noise
        action = torch.randn((B, self.action_dim), device=device)
        obs_features = self._extract_obs_features(obs)

        for t in reversed(range(num_steps)):
            timestep = torch.full((B,), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.denoiser(obs_features, action, timestep)
            
            # Get schedule constants for this step
            alpha_t = 1.0 - self.betas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            alpha_cumprod_t_prev = self.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0, device=device)
            
            # DDPM sampling formula
            term1 = 1.0 / torch.sqrt(alpha_t)
            term2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)
            
            action = term1 * (action - term2 * predicted_noise)
            
            if t > 0:
                noise = torch.randn_like(action)
                variance = self.betas[t] * (1. - alpha_cumprod_t_prev) / (1. - alpha_cumprod_t)
                action += torch.sqrt(variance) * noise
                
        return action
        
    def forward(self, obs: Dict[str, torch.Tensor], inference_step: int = None) -> torch.Tensor:
        """
        The main forward pass for the policy, implementing the differentiable,
        one-step prediction. This is the method that will be used by the PPO algorithm.
        
        Args:
            obs: The observation dictionary.
            inference_step: The specific timestep to start denoising from.
                            Defaults to the last step for maximum signal.
        
        Returns:
            A differentiable tensor representing the predicted clean action.
        """
        if inference_step is None:
            inference_step = self.num_diffusion_iters - 1
            
        B = next(iter(obs.values())).shape[0]
        device = next(self.parameters()).device
        
        # 1. Start with pure noise
        noisy_action = torch.randn((B, self.action_dim), device=device)
        
        # 2. Get observation features
        obs_features = self._extract_obs_features(obs)
        
        # 3. Predict the noise for the chosen timestep
        t = torch.full((B,), inference_step, device=device, dtype=torch.long)
        predicted_noise = self.denoiser(obs_features, noisy_action, t)
        
        # 4. Predict the clean action x_0 using the differentiable formula
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        pred_clean_action = (noisy_action - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / sqrt_alpha_cumprod_t
        
        # Clamp action to a reasonable range [-2.0, 2.0] as a safety measure,
        # then apply tanh to ensure it's in the policy's [-1, 1] output range.
        pred_clean_action = torch.clamp(pred_clean_action, -2.0, 2.0)
        return torch.tanh(pred_clean_action)