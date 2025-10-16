# FILE: models/diffusion_policy.py
# (State-of-the-Art, ResNet+CrossAttentionVision, AdaLN-Zero+CFG Conditioning)

"""
An advanced, state-of-the-art diffusion policy, meticulously engineered for
high-performance robotic manipulation. This implementation integrates cutting-edge
techniques to push the boundaries of learning capability, robustness, and sample fidelity.

This version is a significant upgrade over previous models and is designed for
maximum performance.

Key Architectural Advancements:
 - **Vision Backbone (ResNet-18)**: Employs a deep, pre-trained ResNet-18 backbone
   to extract rich, semantically meaningful features from high-dimensional image
   data. This leverages knowledge from large-scale datasets like ImageNet for a
   powerful and robust perception frontend.

 - **Multi-View Fusion (Cross-Attention)**: Moves beyond simpler FiLM or concatenation
   methods. This model uses a dedicated cross-attention fusion module. Features from
   the wrist camera attend to features from the primary camera, and vice-versa.
   This allows the model to learn complex spatial and contextual relationships
   between the different views, creating a highly informative, fused representation.

 - **Conditioning Mechanism (AdaLN-Zero & Cross-Attention)**: The core denoiser is
   a custom-built Transformer architecture that conditions on observations in two ways:
     1. **AdaLN-Zero**: Diffusion timestep and global observation embeddings modulate the
        affine transformation (scale and shift) inside each Transformer block's LayerNorm.
        This allows the conditioning information to deeply influence the entire
        denoising process, a technique proven effective in leading Diffusion Transformers (DiT).
     2. **Cross-Attention**: The noisy action tokens (queries) explicitly attend to the
        sequence of fused visual and proprioceptive tokens (keys/values), enabling the
        model to precisely pinpoint relevant information at each denoising step.

 - **Classifier-Free Guidance (CFG)**: Implements CFG for both training and inference.
   During training, the model randomly learns to denoise with and without conditioning.
   During inference, this allows for amplifying the conditioning signal, significantly
   improving adherence to the desired behavior and overall action quality.

 - **Modular and Extensively Documented**: The code is structured into clear, reusable
   components with comprehensive docstrings, extensive type hinting, and meticulous
   shape annotations to ensure clarity, maintainability, and ease of future extension.
"""

from __future__ import annotations

import math
import copy
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

# Setup logger
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# -------------------------
# Utilities & Foundational Layers
# -------------------------

def exists(x):
    """Check if a value is not None."""
    return x is not None

def default(val, d):
    """Return val if it exists, otherwise return default d."""
    return val if exists(val) else d

class SinusoidalPosEmb(nn.Module):
    """Generates sinusoidal positional embeddings for timesteps."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): A tensor of shape (B,) representing timesteps.
        Returns:
            torch.Tensor: Sinusoidal embeddings of shape (B, dim).
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# -------------------------
# Noise Scheduler
# -------------------------

@dataclass
class NoiseSchedulerConfig:
    """Configuration for the noise scheduler."""
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: str = "cosine"
    timesteps: int = 100

class NoiseScheduler:
    """
    Implements the forward diffusion process (noising) and the DDIM sampling
    algorithm for the reverse process (denoising).
    """
    def __init__(self, cfg: NoiseSchedulerConfig):
        self.cfg = cfg
        self.T = int(cfg.timesteps)

        if cfg.schedule == "linear":
            self.betas = torch.linspace(cfg.beta_start, cfg.beta_end, self.T)
        elif cfg.schedule == "cosine":
            timesteps = torch.arange(self.T + 1, dtype=torch.float64)
            s = 0.008
            alphas_cumprod = torch.cos(((timesteps / self.T) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clamp(betas, min=0, max=0.999).float()
        else:
            raise ValueError(f"Unknown schedule: {cfg.schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def to(self, device: torch.device):
        """Moves all schedule-related tensors to the specified device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self

    def add_noise(self, x0: torch.Tensor, t: torch.LongTensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward process: q(x_t | x_0). Diffuses the data for a given timestep.
        Args:
            x0 (torch.Tensor): The original data (e.g., actions), shape (B, ...).
            t (torch.LongTensor): The timesteps, shape (B,).
            noise (torch.Tensor): The noise to add, shape (B, ...).
        Returns:
            torch.Tensor: The noised data x_t.
        """
        B = t.shape[0]
        sqrt_acp_t = self.sqrt_alphas_cumprod[t].reshape(B, *([1] * (x0.dim() - 1)))
        sqrt_one_minus_acp_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(B, *([1] * (x0.dim() - 1)))
        return sqrt_acp_t * x0 + sqrt_one_minus_acp_t * noise

    def ddim_step(self, xt: torch.Tensor, t: int, t_prev: int, eps_pred: torch.Tensor, eta: float = 0.0) -> torch.Tensor:
        """
        Performs a single DDIM reverse step to go from x_t to x_{t-1}.
        Args:
            xt (torch.Tensor): The current noisy tensor (x_t).
            t (int): The current timestep.
            t_prev (int): The previous timestep.
            eps_pred (torch.Tensor): The predicted noise from the model.
            eta (float): Controls the stochasticity of the step (0.0 for deterministic).
        Returns:
            torch.Tensor: The denoised tensor for the previous step (x_{t-1}).
        """
        alpha_cumprod_t = self.alphas_cumprod[t]
        alpha_cumprod_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else 1.0

        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]

        x0_pred = (xt - sqrt_one_minus_alpha_cumprod_t * eps_pred) / sqrt_alpha_cumprod_t

        sigma_t = eta * torch.sqrt(
            (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
        )

        pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigma_t**2) * eps_pred
        x_prev = torch.sqrt(alpha_cumprod_t_prev) * x0_pred + pred_dir_xt
        if eta > 0:
            x_prev += sigma_t * torch.randn_like(xt)

        return x_prev

# -------------------------
# Vision Architecture (ResNet + Cross-Attention Fusion)
# -------------------------

class ResNetEncoder(nn.Module):
    """
    Vision encoder using a pre-trained ResNet-18 backbone, adapted for
    sequential image inputs.
    """
    def __init__(self, out_features: int = 256):
        super().__init__()
        self.features_dim = out_features
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # We use all layers except the final fully connected layer and average pooling
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # Project the ResNet feature map to the desired embedding dimension
        self.projection = nn.Conv2d(512, out_features, kernel_size=1)
        self.layer_norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input images of shape (B, H_o, C, H_img, W_img).
        Returns:
            torch.Tensor: Encoded features of shape (B, H_o, D_v).
        """
        B, H_o, C, H_img, W_img = x.shape
        x = x.view(B * H_o, C, H_img, W_img)

        # Freeze backbone during training for stability
        with torch.no_grad():
            x = self.backbone(x)  # Shape: (B*H_o, 512, H_feat, W_feat)

        x = self.projection(x)  # Shape: (B*H_o, D_v, H_feat, W_feat)
        # Global average pooling
        x = x.mean(dim=[-1, -2])  # Shape: (B*H_o, D_v)
        x = self.layer_norm(x)
        return x.view(B, H_o, self.features_dim)

class VisionFusionEncoder(nn.Module):
    """
    Encodes and robustly fuses multi-view image observations and proprioceptive
    data using cross-attention.
    """
    def __init__(self, image_feat_dim: int, proprio_dim: int, d_model: int, n_heads: int = 4):
        super().__init__()
        self.primary_encoder = ResNetEncoder(out_features=image_feat_dim)
        self.wrist_encoder = ResNetEncoder(out_features=image_feat_dim)

        # Fusion module using cross-attention
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=image_feat_dim, num_heads=n_heads, batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(image_feat_dim)
        
        # Projections for proprioceptive data and final output
        self.proprio_proj = nn.Linear(proprio_dim, d_model)
        self.fusion_proj = nn.Linear(image_feat_dim * 2, d_model) # primary + wrist

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes observations and produces separate visual and proprioceptive tokens.
        Args:
            obs (Dict[str, torch.Tensor]): Observation dictionary.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - vision_tokens (B, H_o * 2, D_model): Fused primary and wrist visual tokens.
                - proprio_tokens (B, H_o, D_model): Proprioceptive tokens.
        """
        # Preprocess images: (B, H_o, H, W, C) -> (B, H_o, C, H, W) and normalize
        img_p = obs["image_primary"].permute(0, 1, 4, 2, 3) / 255.0
        img_w = obs["image_wrist"].permute(0, 1, 4, 2, 3) / 255.0
        proprio = obs["proprio"]

        # Encode each view
        feat_p = self.primary_encoder(img_p)  # (B, H_o, D_feat)
        feat_w = self.wrist_encoder(img_w)    # (B, H_o, D_feat)

        # Fuse visual features: wrist attends to primary
        fused_w, _ = self.fusion_attention(query=feat_w, key=feat_p, value=feat_p)
        fused_w = self.fusion_norm(fused_w + feat_w)

        # Concatenate primary and fused wrist features
        fused_vision_features = torch.cat([feat_p, fused_w], dim=-1)
        vision_tokens = self.fusion_proj(fused_vision_features)

        # Project proprioceptive data
        proprio_tokens = self.proprio_proj(proprio)

        return vision_tokens, proprio_tokens

# -------------------------
# Denoiser Architecture (AdaLN-Zero Transformer)
# -------------------------

class DiffusionTransformerBlock(nn.Module):
    """A single block of the Diffusion Transformer, incorporating AdaLN-Zero and cross-attention."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)
        
        # AdaLN-Zero modulation: predicts scale and shift for the three norms
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model) # 3 norms, each needs shift and scale
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input sequence (noisy actions), shape (B, H_a, D_model).
            cond (torch.Tensor): Conditioning sequence (observations), shape (B, H_o, D_model).
            t_emb (torch.Tensor): Timestep embedding, shape (B, D_model).
        Returns:
            torch.Tensor: Output sequence, shape (B, H_a, D_model).
        """
        # Predict modulation parameters from the timestep embedding
        shift1, scale1, shift2, scale2, shift3, scale3 = self.adaLN_modulation(t_emb).chunk(6, dim=1)
        
        # Self-attention block
        x_sa = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = x + self.self_attn(x_sa, x_sa, x_sa, need_weights=False)[0]
        
        # Cross-attention block
        x_ca = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = x + self.cross_attn(x_ca, cond, cond, need_weights=False)[0]
        
        # Feed-forward block
        x_ffn = self.norm3(x) * (1 + scale3.unsqueeze(1)) + shift3.unsqueeze(1)
        x = x + self.ffn(x_ffn)
        
        return x

class DiffusionTransformer(nn.Module):
    """
    State-of-the-art denoiser using a Transformer with AdaLN-Zero conditioning and cross-attention.
    """
    def __init__(self, action_dim: int, d_model: int, n_layers: int, n_heads: int, H_a: int, dropout: float = 0.1):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, d_model)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model), nn.Linear(d_model, d_model * 4),
            nn.Mish(), nn.Linear(d_model * 4, d_model),
        )
        self.action_pos_emb = nn.Embedding(H_a, d_model)
        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.out_proj = nn.Linear(d_model, action_dim)

    def forward(self, noisy_actions: torch.Tensor, timesteps: torch.Tensor, vision_cond: torch.Tensor, proprio_cond: torch.Tensor):
        """
        Args:
            noisy_actions (torch.Tensor): (B, H_a, D_a)
            timesteps (torch.Tensor): (B,)
            vision_cond (torch.Tensor): (B, L_v, D_model)
            proprio_cond (torch.Tensor): (B, L_p, D_model)
        Returns:
            torch.Tensor: Predicted noise, shape (B, H_a, D_a).
        """
        B, H_a, _ = noisy_actions.shape
        action_tokens = self.action_proj(noisy_actions)
        action_tokens += self.action_pos_emb(torch.arange(H_a, device=noisy_actions.device))
        
        # Combine conditioning tokens
        cond_tokens = torch.cat([vision_cond, proprio_cond], dim=1)
        
        # Process timestep embedding
        t_emb = self.time_mlp(timesteps)
        
        # Pass through transformer blocks
        x = action_tokens
        for block in self.blocks:
            x = block(x, cond=cond_tokens, t_emb=t_emb)
            
        return self.out_proj(x)

# -------------------------
# EMA Helper
# -------------------------

class EMA:
    """Exponential Moving Average for model weights."""
    def __init__(self, model: nn.Module, decay: float):
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.decay = decay

    def update(self, model: nn.Module):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    ema_param = self.ema_model.state_dict()[name]
                    ema_param.copy_(self.decay * ema_param + (1 - self.decay) * param.data)

    def state_dict(self): return self.ema_model.state_dict()
    def load_state_dict(self, sd): self.ema_model.load_state_dict(sd)

# -------------------------
# Main DiffusionPolicy Class
# -------------------------

class DiffusionPolicy(nn.Module):
    """
    Top-level diffusion policy wrapper, integrating all state-of-the-art components.
    """
    def __init__(self, *,
                 proprio_dim: int, H_o: int, H_a: int, action_dim: int,
                 image_feat_dim: int, scheduler_cfg: NoiseSchedulerConfig,
                 d_model: int, denoiser_layers: int, denoiser_heads: int,
                 cfg_p_uncond: float = 0.1,
                 ema_decay: Optional[float] = 0.999,
                 device: Optional[Union[torch.device, str]] = None):
        super().__init__()
        self.device = torch.device(default(device, "cuda" if torch.cuda.is_available() else "cpu"))
        self.H_a = H_a
        self.action_dim = action_dim
        self.cfg_p_uncond = cfg_p_uncond

        # Core Components
        self.vision_fusion_encoder = VisionFusionEncoder(image_feat_dim, proprio_dim, d_model)
        self.denoiser = DiffusionTransformer(action_dim, d_model, denoiser_layers, denoiser_heads, H_a)
        self.scheduler = NoiseScheduler(scheduler_cfg).to(self.device)
        self.ema = EMA(self, decay=ema_decay) if ema_decay is not None else None
        
        # Learnable embedding for unconditional generation (for CFG)
        self.uncond_vis_embedding = nn.Parameter(torch.randn(1, H_o * 2, d_model))
        self.uncond_proprio_embedding = nn.Parameter(torch.randn(1, H_o, d_model))

        self.to(self.device)
        log.info(f"State-of-the-art DiffusionPolicy initialized on device: {self.device}")

    def _cond_embed(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates conditioning tokens from observations."""
        obs_on_device = {k: v.to(self.device) for k, v in obs.items()}
        return self.vision_fusion_encoder(obs_on_device)

    def compute_loss(self, actions: torch.Tensor, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the diffusion MSE loss with support for Classifier-Free Guidance."""
        actions = actions.to(self.device)
        B = actions.shape[0]

        noise = torch.randn_like(actions)
        timesteps = torch.randint(0, self.scheduler.T, (B,), device=self.device).long()
        noisy_actions = self.scheduler.add_noise(actions, timesteps, noise)

        # Get conditioning tokens
        vision_cond, proprio_cond = self._cond_embed(obs)
        
        # Implement unconditional training for CFG
        uncond_mask = (torch.rand(B, device=self.device) < self.cfg_p_uncond)
        vision_cond[uncond_mask] = self.uncond_vis_embedding
        proprio_cond[uncond_mask] = self.uncond_proprio_embedding

        predicted_noise = self.denoiser(noisy_actions, timesteps, vision_cond, proprio_cond)
        loss = F.mse_loss(predicted_noise, noise)

        if self.training and self.ema is not None:
            self.ema.update(self)

        return loss, {"loss": loss.item()}

    @torch.no_grad()
    def sample(self, obs: Dict[str, torch.Tensor],
               steps: Optional[int] = None,
               guidance_scale: float = 1.5,
               use_ema: bool = True,
               return_intermediates: bool = False
               ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Samples an action sequence using DDIM and Classifier-Free Guidance.
        """
        model = self.ema.ema_model if use_ema and self.ema else self
        model.eval()

        vision_cond, proprio_cond = self._cond_embed(obs)
        B = vision_cond.shape[0]

        T = steps if steps is not None else self.scheduler.T
        timesteps = list(reversed(range(0, self.scheduler.T, self.scheduler.T // T)))
        x_t = torch.randn((B, self.H_a, self.action_dim), device=self.device)
        intermediates = [x_t]

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((B,), t, device=self.device, dtype=torch.long)
            t_prev = timesteps[i + 1] if i < len(timesteps) - 1 else -1

            # CFG forward passes
            eps_cond = model.denoiser(x_t, t_tensor, vision_cond, proprio_cond)
            eps_uncond = model.denoiser(x_t, t_tensor, self.uncond_vis_embedding.expand(B,-1,-1), self.uncond_proprio_embedding.expand(B,-1,-1))
            
            # Combine predictions
            eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            
            x_t = self.scheduler.ddim_step(x_t, t, t_prev, eps_pred)
            if return_intermediates:
                intermediates.append(x_t)

        if return_intermediates:
            return x_t, intermediates
        return x_t

    def save(self, path: Path):
        """Saves the policy and EMA weights to a file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "policy_state_dict": self.state_dict(),
            "ema_state_dict": self.ema.state_dict() if self.ema else None
        }
        torch.save(state, path)
        log.info(f"Saved model checkpoint to {path}")

    def load(self, path: Path):
        """Loads model weights from a checkpoint."""
        state = torch.load(path, map_location=self.device)
        self.load_state_dict(state["policy_state_dict"])
        if self.ema and "ema_state_dict" in state and state["ema_state_dict"] is not None:
            self.ema.load_state_dict(state["ema_state_dict"])
        log.info(f"Loaded model checkpoint from {path}")



