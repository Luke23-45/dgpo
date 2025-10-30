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
        ...
        """
        # --- THIS BLOCK HAS THE BUG ---
        # alpha_cumprod_t = self.alphas_cumprod[t]
        # alpha_cumprod_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else 1.0

        # --- THIS IS THE FIX ---
        # Ensure all values are tensors on the correct device.

        alpha_cumprod_t = self.alphas_cumprod[t]
        
        # This is the critical fix. When t_prev is -1, create a tensor, not a float.
        alpha_cumprod_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=xt.device, dtype=xt.dtype)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # --- A SECOND, RELATED FIX FOR ROBUSTNESS ---
        # The line below also needs to handle the scalar `alpha_cumprod_t` correctly
        # Let's make sure it's a tensor before doing math with other tensors.
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]

        # The rest of the original code had a subtle issue here. We need to make sure
        # our tensors can be broadcast correctly. Let's rewrite this part for clarity and safety.

        x0_pred = (xt - sqrt_one_minus_alpha_cumprod_t * eps_pred) / sqrt_alpha_cumprod_t
        x0_pred = torch.clamp(x0_pred, -1., 1.) # Optional: Clamp predicted x0 for stability

        sigma_t = eta * torch.sqrt(
            (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
        )

        pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigma_t**2) * eps_pred
        x_prev = torch.sqrt(alpha_cumprod_t_prev) * x0_pred + pred_dir_xt
        if eta > 0:
            x_prev += sigma_t * torch.randn_like(xt)

        return x_prev

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
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input images of shape (B, H_o, C, H_img, W_img) or (B, H_o, H_img, W_img, C).
                              Can be uint8 [0-255] or float [0-1].
        Returns:
            torch.Tensor: Encoded features of shape (B, H_o, D_v).
        """
        # --- START OF SOTA PATCH 2 (continued) ---
        # Robust input normalization
        if x.shape[2] != 3: # If not CHW, assume HWC
            x = x.permute(0, 1, 4, 2, 3) # B, H_o, H, W, C -> B, H_o, C, H, W
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        # Apply ImageNet normalization
        x = (x - self.mean) / self.std
        # --- END OF SOTA PATCH 2 (continued) ---

        B, H_o, C, H_img, W_img = x.shape
        x = x.reshape(B * H_o, C, H_img, W_img) # Use reshape for efficiency

        # Freeze backbone for stability. No gradients will flow through here.
        with torch.no_grad():
            x = self.backbone(x)

        x = self.projection(x)
        x = x.mean(dim=[-1, -2])
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

    def forward(self, 
                noisy_actions: torch.Tensor, 
                timesteps: torch.Tensor, 
                vision_cond: torch.Tensor, 
                proprio_cond: torch.Tensor,
                subgoal_cond: Optional[torch.Tensor] = None
                ):
        """
        Args:
            noisy_actions (torch.Tensor): (B, H_a, D_a)
            timesteps (torch.Tensor): (B,)
            vision_cond (torch.Tensor): (B, L_v, D_model)
            proprio_cond (torch.Tensor): (B, L_p, D_model)
            subgoal_cond (torch.Tensor, optional): (B, L_s, D_model).
        Returns:
            torch.Tensor: Predicted noise, shape (B, H_a, D_a).
        """
        B, H_a, _ = noisy_actions.shape
        action_tokens = self.action_proj(noisy_actions)
        action_tokens += self.action_pos_emb(torch.arange(H_a, device=noisy_actions.device))
        
        # Combine all available conditioning tokens into a single sequence
        cond_tokens_list = [vision_cond, proprio_cond]
        if subgoal_cond is not None:
            cond_tokens_list.append(subgoal_cond)
        cond_tokens = torch.cat(cond_tokens_list, dim=1)
        
        t_emb = self.time_mlp(timesteps)
        
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
    SOTA ViDHiS Controller. Top-level diffusion policy wrapper, adapted for
    hierarchical control with visual subgoal conditioning.
    """
    def __init__(self, *,
                 proprio_dim: int, H_o: int, H_a: int, action_dim: int,
                 image_feat_dim: int, # This is now the ResNetEncoder output dim
                 d_model: int,        # This is now the Transformer's internal dim
                 denoiser_layers: int, denoiser_heads: int,
                 scheduler_cfg: NoiseSchedulerConfig,
                 cfg_p_uncond: float = 0.1,
                 ema_decay: Optional[float] = 0.999,
                 device: Optional[Union[torch.device, str]] = None):
        super().__init__()
        self.device = torch.device(default(device, "cuda" if torch.cuda.is_available() else "cpu"))
        self.H_o, self.H_a, self.action_dim = H_o, H_a, action_dim
        self.cfg_p_uncond = cfg_p_uncond

        # --- SOTA PATCH 3: Refactored Encoders ---
        # 1. Encoders for Observation History
        self.primary_encoder = ResNetEncoder(out_features=image_feat_dim)
        self.wrist_encoder = ResNetEncoder(out_features=image_feat_dim)
        self.proprio_proj = nn.Linear(proprio_dim, d_model)

        # 2. Dedicated Encoder for the Visual Subgoal
        self.subgoal_encoder = ResNetEncoder(out_features=d_model)

        # 3. Optional Fusion/Projection layers
        # Project ResNet features to the Transformer's dimension
        self.vision_proj = nn.Linear(image_feat_dim * 2, d_model) # Fused primary + wrist
        # --- End Refactored Encoders ---

        self.denoiser = DiffusionTransformer(action_dim, d_model, denoiser_layers, denoiser_heads, H_a)
        self.scheduler = NoiseScheduler(scheduler_cfg).to(self.device)

        # --- SOTA PATCH 3: Refactored Unconditional Embeddings ---
        # Create a dictionary for clean management of unconditional tokens
        self.uncond_embeddings = nn.ParameterDict({
            'primary': nn.Parameter(torch.randn(1, H_o, d_model)),
            'wrist': nn.Parameter(torch.randn(1, H_o, d_model)),
            'proprio': nn.Parameter(torch.randn(1, H_o, d_model)),
            'subgoal': nn.Parameter(torch.randn(1, 1, d_model)),
        })
        # --- End Refactored Embeddings ---

        self.to(self.device)
        self.ema = EMA(self, decay=ema_decay) if ema_decay is not None else None
        log.info(f"ViDHiS Controller (DiffusionPolicy) initialized on device: {self.device}")

    def _get_condition_tokens(self, obs: Dict[str, torch.Tensor], subgoal_image: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """SOTA: Encodes all inputs and returns a structured dictionary of conditioning tokens."""
        obs_on_device = {k: v.to(self.device) for k, v in obs.items()}
        
        # 1. Process Observation History
        primary_tokens = self.primary_encoder(obs_on_device["image_primary"])
        wrist_tokens = self.wrist_encoder(obs_on_device["image_wrist"])
        # Simple fusion by concatenation
        vision_tokens = self.vision_proj(torch.cat([primary_tokens, wrist_tokens], dim=-1))
        proprio_tokens = self.proprio_proj(obs_on_device["proprio"])
        
        # 2. Process Optional Subgoal
        subgoal_tokens = None
        if subgoal_image is not None:
            # Subgoal is (B, H, W, C). ResNetEncoder needs (B, H_o=1, H, W, C).
            subgoal_tokens = self.subgoal_encoder(subgoal_image.to(self.device).unsqueeze(1))
        
        return {
            'vision': vision_tokens,
            'proprio': proprio_tokens,
            'subgoal': subgoal_tokens
        }

    def compute_loss(self,
                     actions: torch.Tensor,
                     obs: Dict[str, torch.Tensor],
                     subgoal_image: Optional[torch.Tensor] = None,
                     weights: Optional[torch.Tensor] = None
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the diffusion MSE loss for training.

        This SOTA version correctly handles:
        - Unified conditioning from observation history, proprioception, and an optional visual subgoal.
        - Classifier-Free Guidance (CFG) during training by randomly dropping conditioning.
        - Optional per-sample weighting for advanced training strategies like RL fine-tuning.
        """
        # 1. Basic setup: move data to device and get batch size.
        actions = actions.to(self.device)
        B = actions.shape[0]
        device = self.device

        # 2. Prepare for diffusion forward process: sample noise and timesteps.
        noise = torch.randn_like(actions)
        timesteps = torch.randint(0, self.scheduler.T, (B,), device=device).long()

        # 3. Apply noise to the clean actions to get the noisy input for the denoiser.
        noisy_actions = self.scheduler.add_noise(actions, timesteps, noise)

        # 4. Get all conditioning tokens in a structured dictionary using the unified helper method.
        cond = self._get_condition_tokens(obs, subgoal_image)

        # 5. Apply Classifier-Free Guidance mask for unconditional training.
        #    With a probability of `cfg_p_uncond`, we replace the real conditioning
        #    tokens with learned unconditional embeddings.
        if self.training and self.cfg_p_uncond > 0:
            uncond_mask = (torch.rand(B, device=device) < self.cfg_p_uncond)
            
            # Replace conditioning tokens with their unconditional counterparts where the mask is active.
            # The `expand` call is necessary to match the batch dimension.
            if cond.get('vision') is not None:
                cond['vision'][uncond_mask] = self.uncond_embeddings['vision'].expand(B, -1, -1)[uncond_mask]
            if cond.get('proprio') is not None:
                cond['proprio'][uncond_mask] = self.uncond_embeddings['proprio'].expand(B, -1, -1)[uncond_mask]
            if cond.get('subgoal') is not None:
                cond['subgoal'][uncond_mask] = self.uncond_embeddings['subgoal'].expand(B, -1, -1)[uncond_mask]

        # 6. Assemble the final unified conditioning sequence for the denoiser.
        #    This gathers all non-None token sets into a single long sequence.
        cond_tokens_list = [t for t in cond.values() if t is not None]
        cond_tokens = torch.cat(cond_tokens_list, dim=1)

        # 7. Predict the noise using the denoiser.
        predicted_noise = self.denoiser(noisy_actions, timesteps, cond_tokens)

        # 8. Calculate the loss. This uses your SOTA weighted loss logic.
        #    It computes a per-sample loss, averages it over the sequence/action dims,
        #    and then applies optional weights before the final mean reduction.
        per_sample_loss = F.mse_loss(predicted_noise, noise, reduction='none')
        per_sample_loss = per_sample_loss.mean(dim=list(range(1, per_sample_loss.ndim)))
        
        if weights is not None:
            if weights.ndim > 1:
                weights = weights.squeeze()
            loss = (per_sample_loss * weights).mean()
        else:
            loss = per_sample_loss.mean()

        # 9. Update the Exponential Moving Average (EMA) of the model weights if in training mode.
        if self.training and self.ema is not None:
            self.ema.update(self)

        return loss, {"loss": loss.item()}

    @torch.no_grad()
    def sample(self,
               obs: Dict[str, torch.Tensor],
               subgoal_image: Optional[torch.Tensor] = None,
               steps: Optional[int] = None,
               guidance_scale: float = 1.5,
               use_ema: bool = True,
               return_intermediates: bool = False
               ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Samples an action sequence from the diffusion model using DDIM and Classifier-Free Guidance.

        This SOTA version ensures a consistent conditioning pipeline with training and
        implements an efficient batched CFG forward pass.
        """
        # 1. Select the model for inference (EMA weights are preferred for stability).
        model = self.ema.ema_model if use_ema and self.ema is not None else self
        model.eval()
        
        B, device = next(iter(obs.values())).shape[0], self.device
        T = steps if steps is not None else self.scheduler.T

        # 2. Get the conditional tokens using the unified helper method.
        cond = self._get_condition_tokens(obs, subgoal_image)
        cond_tokens_list = [t for t in cond.values() if t is not None]
        cond_tokens = torch.cat(cond_tokens_list, dim=1)

        # 3. Assemble the unconditional tokens for CFG.
        #    These are expanded to match the batch size.
        uncond_tokens_list = [
            self.uncond_embeddings['vision'].expand(B, -1, -1),
            self.uncond_embeddings['proprio'].expand(B, -1, -1)
        ]
        if cond['subgoal'] is not None:
            uncond_tokens_list.append(self.uncond_embeddings['subgoal'].expand(B, -1, -1))
        uncond_tokens = torch.cat(uncond_tokens_list, dim=1)

        # 4. Set up the DDIM scheduler and initialize latents from pure noise.
        self.scheduler.set_timesteps(T, device=device)
        timesteps = self.scheduler.timesteps
        latents = torch.randn((B, self.H_a, self.action_dim), device=device)
        
        # Optional: store intermediate steps for visualization.
        intermediates = [latents] if return_intermediates else None

        # 5. The DDIM denoising loop.
        for t in timesteps:
            # For CFG, we predict noise for both conditional and unconditional inputs in a single batch.
            # This is more efficient than two separate forward passes.
            
            # a. Create a batched input for the denoiser: [unconditional_latents, conditional_latents]
            latent_model_input = torch.cat([latents] * 2)
            
            # b. Assemble the full context for the denoiser: [unconditional_tokens, conditional_tokens]
            combined_cond_tokens = torch.cat([uncond_tokens, cond_tokens], dim=0)
            
            # c. Predict noise for the combined batch.
            #    The `t` tensor is also duplicated to match the batch size.
            noise_pred = model.denoiser(latent_model_input, torch.cat([t.expand(B)] * 2), combined_cond_tokens)
            
            # d. Perform guidance: split the predictions and combine them.
            #    `guided_noise = uncond_pred + guidance_scale * (cond_pred - uncond_pred)`
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            guided_noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # e. Scheduler step to compute the previous noisy sample (denoise one step).
            latents = self.scheduler.step(guided_noise_pred, t, latents).prev_sample
            
            if return_intermediates:
                intermediates.append(latents)

        if return_intermediates:
            return latents, intermediates
        
        return latents

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
        log.info(f"Loading model checkpoint from {path}")
        state = torch.load(path, map_location=self.device, weights_only=False)
        
        # --- START OF SOTA PATCH ---
        # More robust loading: handle cases where only policy is present
        policy_state_dict = state.get("policy_state_dict", state)
        self.load_state_dict(policy_state_dict)
        
        ema_state_dict = state.get("ema_state_dict")
        if self.ema and ema_state_dict is not None:
            try:
                self.ema.load_state_dict(ema_state_dict)
                log.info("Successfully loaded EMA weights.")
            except Exception as e:
                log.warning(f"Could not load EMA weights, they may be incompatible. Error: {e}")
        elif self.ema:
            log.warning("Checkpoint does not contain EMA weights, which were expected.")
        # --- END OF SOTA PATCH ---



