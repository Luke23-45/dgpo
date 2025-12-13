# FILE: models/unified_diffusion_planner.py
# State-of-the-Art Unified Diffusion Planner for Robot Manipulation
# Research-backed implementation combining:
# - Diffusion Policy (Chi et al., 2023) - Handles multimodal action distributions
# - ACT (Zhao et al., 2023) - Action chunking reduces compounding errors
# - DiT (Peebles & Xie, 2023) - Diffusion Transformer with AdaLN-Zero

"""
UnifiedDiffusionPlanner: A SOTA robot manipulation policy architecture.

Key features:
1. Diffusion-based action prediction (handles multimodality)
2. Delta action representation (more robust than absolute poses)
3. SigLIP vision backbone (proven in OpenVLA, Pi-0)
4. AdaLN-Zero conditioning (from DiT paper)
5. Action chunking (K=8 steps, reduces compounding errors)
6. DDIM sampling for fast inference (10 steps)

Architecture:
    Input: prev_image, curr_image, goal_image, proprio
    → VisionContextEncoder (SigLIP + fusion transformer)
    → DiffusionActionHead (DiT blocks with cross-attention)
    → Output: delta_pose_chunk (B, K, 7), gripper_chunk (B, K, 1)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipVisionModel

log = logging.getLogger(__name__)


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

@dataclass
class UnifiedDiffusionConfig:
    """
    Configuration for the Unified Diffusion Planner.
    
    All hyperparameters are research-backed:
    - action_chunk_size=8: ACT paper shows optimal range is 4-16
    - diffusion_timesteps=100: Standard for DDIM
    - inference_steps=10: DDIM allows 10x reduction in steps
    - d_model=512: Matches Ego Planner, good balance of capacity/speed
    """
    # Action space
    action_dim: int = 7  # delta pose: dx, dy, dz, dqx, dqy, dqz, dqw
    action_chunk_size: int = 8  # K future steps to predict
    proprio_dim: int = 22  # Proprioception dimension
    
    # Vision backbone
    vision_backbone_model: str = "google/siglip-base-patch16-224"
    vision_feature_dim: int = 768  # SigLIP-base output dim
    
    # Fusion transformer
    fusion_layers: int = 4
    fusion_heads: int = 8
    
    # Diffusion head
    d_model: int = 512  # Dimension of diffusion transformer
    denoiser_layers: int = 6
    denoiser_heads: int = 8
    dropout: float = 0.1
    
    # Diffusion process
    diffusion_timesteps: int = 100
    inference_steps: int = 10  # DDIM sampling steps
    beta_schedule: str = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    
    # Classifier-free guidance
    p_uncond: float = 0.1  # Probability of unconditional training
    guidance_scale: float = 1.5  # CFG scale during inference
    
    # Phase prediction (from Semantic Planner - auxiliary loss for semantic understanding)
    num_task_phases: int = 5  # Number of task phases to predict
    use_phase_prediction: bool = True  # Enable phase prediction auxiliary loss
    phase_loss_weight: float = 0.1  # Weight for phase classification loss
    
    # Separate heads (from Semantic Planner - disentangled outputs)
    use_separate_heads: bool = True  # Separate heads for pose, gripper, phase


# =============================================================================
# 1.5 ACTION NORMALIZER (CRITICAL FOR DIFFUSION)
# =============================================================================

class ActionNormalizer(nn.Module):
    """
    Normalizes actions to [-1, 1] range for diffusion policy.
    
    Research shows that DDIM clipping to [-1, 1] requires actions 
    to be pre-normalized to this range for optimal performance.
    
    This normalizer tracks running statistics and can be fitted
    on the training dataset before training.
    """
    
    def __init__(self, action_dim: int = 8, eps: float = 1e-6):
        super().__init__()
        self.action_dim = action_dim
        self.eps = eps
        
        # Register buffers for normalization stats
        self.register_buffer('action_min', torch.zeros(action_dim))
        self.register_buffer('action_max', torch.ones(action_dim))
        self.register_buffer('fitted', torch.tensor(False))
    
    def fit(self, actions: torch.Tensor):
        """
        Fit normalizer on a batch of actions.
        
        Args:
            actions: (N, K, action_dim) or (N, action_dim) - actions to fit on
        """
        # Flatten to (N, action_dim)
        if actions.dim() == 3:
            actions = actions.reshape(-1, actions.shape[-1])
        
        self.action_min = actions.min(dim=0).values
        self.action_max = actions.max(dim=0).values
        
        # Add small margin to prevent edge cases
        margin = 0.01 * (self.action_max - self.action_min).clamp_min(0.01)
        self.action_min = self.action_min - margin
        self.action_max = self.action_max + margin
        
        self.fitted = torch.tensor(True)
        log.info(f"ActionNormalizer fitted: min={self.action_min.tolist()}, max={self.action_max.tolist()}")
    
    def fit_from_stats(self, action_min: torch.Tensor, action_max: torch.Tensor):
        """Set normalization stats directly."""
        self.action_min = action_min.to(self.action_min.device)
        self.action_max = action_max.to(self.action_max.device)
        self.fitted = torch.tensor(True)
    
    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Normalize actions from original range to [-1, 1].
        
        Args:
            actions: (..., action_dim) - actions in original range
            
        Returns:
            Normalized actions in [-1, 1] range
        """
        if not self.fitted:
            log.warning("ActionNormalizer not fitted, using identity")
            return actions
        
        # Ensure buffers are on the same device as input
        action_min = self.action_min.to(actions.device)
        action_max = self.action_max.to(actions.device)
        
        # Scale to [0, 1] then to [-1, 1]
        range_val = (action_max - action_min).clamp_min(self.eps)
        normalized = (actions - action_min) / range_val  # [0, 1]
        normalized = normalized * 2.0 - 1.0  # [-1, 1]
        return normalized
    
    def denormalize(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Denormalize actions from [-1, 1] to original range.
        
        Args:
            actions: (..., action_dim) - actions in [-1, 1] range
            
        Returns:
            Actions in original range
        """
        if not self.fitted:
            log.warning("ActionNormalizer not fitted, using identity")
            return actions
        
        # Ensure buffers are on the same device as input
        action_min = self.action_min.to(actions.device)
        action_max = self.action_max.to(actions.device)
        
        # Scale from [-1, 1] to [0, 1] then to original
        normalized_01 = (actions + 1.0) / 2.0  # [0, 1]
        range_val = action_max - action_min
        return normalized_01 * range_val + action_min


# =============================================================================
# 2. NOISE SCHEDULER (DDIM)
# =============================================================================

class CosineNoiseScheduler(nn.Module):
    """
    Cosine noise schedule with DDIM sampling.
    
    Based on "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021).
    DDIM sampling allows 10x fewer steps while maintaining quality.
    """
    
    def __init__(self, timesteps: int = 100, beta_schedule: str = "cosine"):
        super().__init__()
        self.T = timesteps
        
        if beta_schedule == "cosine":
            # Cosine schedule from improved DDPM paper
            steps = torch.arange(self.T + 1, dtype=torch.float64)
            s = 0.008  # Small offset to prevent singularity
            alphas_cumprod = torch.cos(((steps / self.T) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, min=0.0, max=0.999).float()
        else:
            # Linear schedule (fallback)
            betas = torch.linspace(1e-4, 0.02, self.T)
        
        # Pre-compute all schedule tensors
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Register as buffers (move to device with model)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
    
    def add_noise(self, x0: torch.Tensor, t: torch.LongTensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0).
        
        Args:
            x0: Clean data (B, K, D)
            t: Timesteps (B,)
            noise: Gaussian noise same shape as x0
            
        Returns:
            x_t: Noisy data at timestep t
        """
        B = t.shape[0]
        # Reshape for broadcasting: (B, 1, 1)
        sqrt_acp = self.sqrt_alphas_cumprod[t].view(B, 1, 1)
        sqrt_one_minus_acp = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1)
        return sqrt_acp * x0 + sqrt_one_minus_acp * noise
    
    def ddim_step(
        self, 
        x_t: torch.Tensor, 
        t: int, 
        t_prev: int, 
        noise_pred: torch.Tensor,
        eta: float = 0.0  # eta=0 is deterministic DDIM
    ) -> torch.Tensor:
        """
        DDIM reverse step: predict x_{t-1} from x_t.
        
        Args:
            x_t: Noisy sample at timestep t
            t: Current timestep
            t_prev: Previous timestep (t_prev < t)
            noise_pred: Predicted noise at timestep t
            eta: Stochasticity (0 = deterministic DDIM)
            
        Returns:
            x_{t-1}: Denoised sample
        """
        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x_t.device)
        
        # Predict x_0 from x_t and noise
        x0_pred = (x_t - self.sqrt_one_minus_alphas_cumprod[t] * noise_pred) / self.sqrt_alphas_cumprod[t].clamp_min(1e-8)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)  # Clip for stability
        
        # Compute sigma for stochastic sampling (eta > 0)
        if eta > 0 and t_prev >= 0:
            sigma_t = eta * torch.sqrt(
                (1 - alpha_prev) / (1 - alpha_t).clamp_min(1e-8) *
                (1 - alpha_t / alpha_prev.clamp_min(1e-8))
            )
        else:
            sigma_t = 0.0
        
        # Direction pointing to x_t
        pred_dir = torch.sqrt((1 - alpha_prev - sigma_t**2).clamp_min(0.0)) * noise_pred
        
        # Compute x_{t-1}
        x_prev = torch.sqrt(alpha_prev) * x0_pred + pred_dir
        
        if eta > 0:
            x_prev = x_prev + sigma_t * torch.randn_like(x_t)
        
        return x_prev


# =============================================================================
# 3. POSITIONAL EMBEDDINGS
# =============================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for diffusion timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timesteps (B,)
            
        Returns:
            Embeddings (B, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# =============================================================================
# 4. VISION CONTEXT ENCODER
# =============================================================================

class VisionContextEncoder(nn.Module):
    """
    SigLIP-based vision encoder with temporal fusion.
    
    Processes prev_image, curr_image, goal_image and fuses them
    into a context sequence for the diffusion head.
    
    Design choices:
    - SigLIP frozen (proven in OpenVLA, Pi-0)
    - Token-type embeddings for temporal distinction
    - Learnable fusion transformer
    - Global context pooling for AdaLN conditioning
    """
    
    def __init__(self, cfg: UnifiedDiffusionConfig):
        super().__init__()
        self.cfg = cfg
        
        # 1. Load SigLIP vision backbone (PARTIAL UNFREEZE like SemanticPlanner)
        log.info(f"Loading SigLIP backbone: {cfg.vision_backbone_model}")
        self.vision_backbone = SiglipVisionModel.from_pretrained(cfg.vision_backbone_model)
        
        # A. Freeze EVERYTHING first
        for param in self.vision_backbone.parameters():
            param.requires_grad = False
        
        # B. Unfreeze the Last 3 Encoder Layers
        # This allows the model to learn "Geometry" without forgetting "Objects"
        last_layers = self.vision_backbone.vision_model.encoder.layers[-3:]
        for layer in last_layers:
            for param in layer.parameters():
                param.requires_grad = True
        
        # C. Unfreeze the Final LayerNorm (Crucial for feature scaling)
        if hasattr(self.vision_backbone.vision_model, 'post_layernorm'):
            for param in self.vision_backbone.vision_model.post_layernorm.parameters():
                param.requires_grad = True
        
        log.info("SigLIP: Bottom layers FROZEN, top 3 layers UNFROZEN for geometric adaptation")
        
        # D. Enable gradient checkpointing for memory efficiency (optional)
        self.use_gradient_checkpointing = True
        if self.use_gradient_checkpointing and hasattr(self.vision_backbone, 'gradient_checkpointing_enable'):
            self.vision_backbone.gradient_checkpointing_enable()
            log.info("SigLIP: Gradient checkpointing ENABLED for memory efficiency")
        
        # Get patch info
        vit_config = self.vision_backbone.config
        self.num_patches = (vit_config.image_size // vit_config.patch_size) ** 2
        log.info(f"SigLIP: {vit_config.image_size}x{vit_config.image_size}, {self.num_patches} patches")
        
        # 2. Learnable embeddings
        # Spatial positional embedding (shared across all frames)
        self.spatial_pos_emb = nn.Parameter(torch.randn(1, self.num_patches, cfg.vision_feature_dim) * 0.02)
        
        # Token type embeddings (prev=0, curr=1, goal=2, proprio=3)
        self.token_type_emb = nn.Embedding(4, cfg.vision_feature_dim)
        
        # 3. Proprioception encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(cfg.proprio_dim, cfg.vision_feature_dim),
            nn.LayerNorm(cfg.vision_feature_dim),
            nn.GELU(),
            nn.Linear(cfg.vision_feature_dim, cfg.vision_feature_dim)
        )
        
        # 4. Fusion transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.vision_feature_dim,
            nhead=cfg.fusion_heads,
            dim_feedforward=cfg.vision_feature_dim * 4,
            dropout=cfg.dropout,
            activation='gelu',
            batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.fusion_layers)
        
        # 5. Global context projection (for AdaLN conditioning)
        self.global_context_proj = nn.Sequential(
            nn.LayerNorm(cfg.vision_feature_dim),
            nn.Linear(cfg.vision_feature_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model)
        )
        
        # 6. Context projection (for cross-attention)
        self.context_proj = nn.Linear(cfg.vision_feature_dim, cfg.d_model)
        
        # 7. Unconditional embedding for CFG
        self.uncond_embedding = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
    
    def forward(
        self, 
        prev_image: torch.Tensor,
        curr_image: torch.Tensor,
        goal_image: torch.Tensor,
        proprio: torch.Tensor,
        return_uncond: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode visual observations into context for diffusion head.
        
        Args:
            prev_image: (B, 3, 224, 224)
            curr_image: (B, 3, 224, 224)
            goal_image: (B, 3, 224, 224)
            proprio: (B, proprio_dim)
            return_uncond: If True, return unconditional embeddings for CFG
            
        Returns:
            context_tokens: (B, L_ctx, d_model) - for cross-attention
            global_context: (B, d_model) - for AdaLN conditioning
        """
        B = curr_image.shape[0]
        device = curr_image.device
        
        if return_uncond:
            # Return unconditional embeddings for classifier-free guidance
            uncond_ctx = self.uncond_embedding.expand(B, -1, -1)
            uncond_global = torch.zeros(B, self.cfg.d_model, device=device)
            return uncond_ctx, uncond_global
        
        # 1. Encode all images through SigLIP (frozen)
        with torch.no_grad():
            # Stack for efficiency: (3*B, 3, 224, 224)
            stacked = torch.cat([prev_image, curr_image, goal_image], dim=0).float()
            outputs = self.vision_backbone(stacked, output_hidden_states=False)
            all_tokens = outputs.last_hidden_state  # (3*B, N_patches, D)
        
        # Split back: each (B, N_patches, D)
        prev_tokens, curr_tokens, goal_tokens = torch.chunk(all_tokens, 3, dim=0)
        
        # 2. Add spatial positional embeddings
        prev_tokens = prev_tokens + self.spatial_pos_emb
        curr_tokens = curr_tokens + self.spatial_pos_emb
        goal_tokens = goal_tokens + self.spatial_pos_emb
        
        # 3. Add token type embeddings
        prev_tokens = prev_tokens + self.token_type_emb(torch.tensor(0, device=device))
        curr_tokens = curr_tokens + self.token_type_emb(torch.tensor(1, device=device))
        goal_tokens = goal_tokens + self.token_type_emb(torch.tensor(2, device=device))
        
        # 4. Encode proprioception
        proprio_token = self.proprio_encoder(proprio).unsqueeze(1)  # (B, 1, D)
        proprio_token = proprio_token + self.token_type_emb(torch.tensor(3, device=device))
        
        # 5. Concatenate all tokens: [proprio, prev, curr, goal]
        fused_input = torch.cat([proprio_token, prev_tokens, curr_tokens, goal_tokens], dim=1)
        
        # 6. Apply fusion transformer
        fused_output = self.fusion_transformer(fused_input)
        
        # 7. Extract global context (pool from first token - proprioception)
        global_context = self.global_context_proj(fused_output[:, 0, :])  # (B, d_model)
        
        # 8. Project all tokens to d_model for cross-attention
        context_tokens = self.context_proj(fused_output)  # (B, L_ctx, d_model)
        
        return context_tokens, global_context


# =============================================================================
# 5. DIFFUSION TRANSFORMER BLOCK (AdaLN-Zero)
# =============================================================================

class DiffusionTransformerBlock(nn.Module):
    """
    Single block of the Diffusion Transformer (DiT) with AdaLN-Zero.
    
    Architecture:
    1. AdaLN → Self-Attention → Residual
    2. AdaLN → Cross-Attention → Residual  
    3. AdaLN → FFN → Residual
    
    AdaLN-Zero: Initialize modulation to output zeros at the start,
    creating a clean residual path for training stability.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Cross-attention (to context)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer norms (no learnable affine - controlled by AdaLN)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)
        
        # AdaLN modulation: outputs 6 * d_model parameters
        # (scale1, shift1, scale2, shift2, scale3, shift3)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model)
        )
        
        # Zero initialization for training stability
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        context: torch.Tensor, 
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Action tokens (B, K, d_model)
            context: Vision context (B, L_ctx, d_model)
            condition: Combined time + global context embedding (B, d_model)
            
        Returns:
            x: Processed action tokens (B, K, d_model)
        """
        # Predict modulation parameters
        mod = self.adaLN_modulation(condition)  # (B, 6*d_model)
        scale1, shift1, scale2, shift2, scale3, shift3 = mod.chunk(6, dim=-1)
        
        # 1. Self-Attention with AdaLN
        x_norm = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        sa_out, _ = self.self_attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + sa_out
        
        # 2. Cross-Attention with AdaLN
        x_norm = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        ca_out, _ = self.cross_attn(x_norm, context, context, need_weights=False)
        x = x + ca_out
        
        # 3. FFN with AdaLN
        x_norm = self.norm3(x) * (1 + scale3.unsqueeze(1)) + shift3.unsqueeze(1)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        
        return x


# =============================================================================
# 6. DIFFUSION ACTION HEAD
# =============================================================================

class DiffusionActionHead(nn.Module):
    """
    Diffusion Transformer head for action denoising.
    
    Takes noisy actions and denoises them conditioned on:
    - Visual context (cross-attention)
    - Timestep embedding (AdaLN)
    - Global context embedding (AdaLN)
    """
    
    def __init__(self, cfg: UnifiedDiffusionConfig):
        super().__init__()
        self.cfg = cfg
        
        # 1. Action embedding
        self.action_proj = nn.Linear(cfg.action_dim + 1, cfg.d_model)  # +1 for gripper
        
        # 2. Positional embedding for action chunk
        self.action_pos_emb = nn.Embedding(cfg.action_chunk_size, cfg.d_model)
        
        # 3. Timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.SiLU(),
            nn.Linear(cfg.d_model * 2, cfg.d_model)
        )
        
        # 4. Denoiser blocks
        self.denoiser_blocks = nn.ModuleList([
            DiffusionTransformerBlock(cfg.d_model, cfg.denoiser_heads, cfg.dropout)
            for _ in range(cfg.denoiser_layers)
        ])
        
        # 5. Output projection (SEPARATE HEADS like Semantic Planner for disentanglement)
        self.out_norm = nn.LayerNorm(cfg.d_model)
        
        if cfg.use_separate_heads:
            # Separate pose head (7D delta pose) - deeper for precise geometry
            self.pose_head = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model),
                nn.GELU(),
                nn.LayerNorm(cfg.d_model),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_model, cfg.d_model),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_model, cfg.action_dim)  # 7D pose
            )
            
            # Separate gripper head (1D) - simpler for binary decision
            self.gripper_head = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model // 2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_model // 2, 1)  # 1D gripper
            )
            
            self.out_proj = None  # Not used with separate heads
        else:
            # Single combined output (original design)
            self.out_proj = nn.Linear(cfg.d_model, cfg.action_dim + 1)
            self.pose_head = None
            self.gripper_head = None
        
        # Initialize output projections to small values
        if cfg.use_separate_heads:
            nn.init.zeros_(self.pose_head[-1].bias)
            nn.init.normal_(self.pose_head[-1].weight, std=0.02)
            nn.init.zeros_(self.gripper_head[-1].bias)
            nn.init.normal_(self.gripper_head[-1].weight, std=0.02)
        else:
            nn.init.zeros_(self.out_proj.bias)
            nn.init.normal_(self.out_proj.weight, std=0.02)
    
    def forward(
        self, 
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        global_context: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise in the noisy actions.
        
        Args:
            noisy_actions: (B, K, action_dim + 1) - noisy delta pose + gripper
            timesteps: (B,) - diffusion timesteps
            context: (B, L_ctx, d_model) - vision context
            global_context: (B, d_model) - pooled context for AdaLN
            
        Returns:
            noise_pred: (B, K, action_dim + 1) - predicted noise
        """
        B, K = noisy_actions.shape[:2]
        device = noisy_actions.device
        
        # 1. Project actions to d_model
        action_tokens = self.action_proj(noisy_actions)  # (B, K, d_model)
        
        # 2. Add positional embeddings
        pos_indices = torch.arange(K, device=device)
        action_tokens = action_tokens + self.action_pos_emb(pos_indices)
        
        # 3. Compute timestep embedding
        time_emb = self.time_mlp(timesteps)  # (B, d_model)
        
        # 4. Combine timestep and global context for AdaLN
        condition = time_emb + global_context  # (B, d_model)
        
        # 5. Apply denoiser blocks
        x = action_tokens
        for block in self.denoiser_blocks:
            x = block(x, context, condition)
        
        # 6. Project to output (separate heads or combined)
        x = self.out_norm(x)
        
        if self.cfg.use_separate_heads:
            # Separate heads for disentangled gradients (like Semantic Planner)
            pose_pred = self.pose_head(x)  # (B, K, 7)
            grip_pred = self.gripper_head(x)  # (B, K, 1)
            noise_pred = torch.cat([pose_pred, grip_pred], dim=-1)  # (B, K, 8)
        else:
            # Combined output
            noise_pred = self.out_proj(x)  # (B, K, action_dim + 1)
        
        return noise_pred


# =============================================================================
# 7. UNIFIED DIFFUSION PLANNER (Main Model)
# =============================================================================

class UnifiedDiffusionPlanner(nn.Module):
    """
    State-of-the-Art Unified Diffusion Planner for Robot Manipulation.
    
    Combines:
    - Diffusion Policy for multimodal action generation
    - ACT-style action chunking for temporal consistency
    - DiT-style AdaLN-Zero for stable conditioning
    - Delta action representation for robustness
    
    Training:
    - Simple MSE loss on noise prediction
    - Optional classifier-free guidance training
    
    Inference:
    - DDIM sampling (10 steps for speed)
    - Optional classifier-free guidance
    """
    
    def __init__(self, cfg: UnifiedDiffusionConfig):
        super().__init__()
        self.cfg = cfg
        
        # 1. Vision context encoder
        self.vision_encoder = VisionContextEncoder(cfg)
        
        # 2. Diffusion action head
        self.action_head = DiffusionActionHead(cfg)
        
        # 3. Noise scheduler
        self.noise_scheduler = CosineNoiseScheduler(
            timesteps=cfg.diffusion_timesteps,
            beta_schedule=cfg.beta_schedule
        )
        
        # 4. Action normalizer (CRITICAL for diffusion stability)
        self.action_normalizer = ActionNormalizer(action_dim=cfg.action_dim + 1)
        
        # 5. Phase prediction head (from Semantic Planner - auxiliary loss for semantic understanding)
        if cfg.use_phase_prediction:
            self.phase_head = nn.Sequential(
                nn.LayerNorm(cfg.d_model),
                nn.Linear(cfg.d_model, cfg.d_model),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_model, cfg.num_task_phases)
            )
            log.info(f"  - Phase head: {cfg.num_task_phases} phases (auxiliary loss)")
        else:
            self.phase_head = None
        
        log.info(f"UnifiedDiffusionPlanner initialized with {self._count_parameters()} parameters")
        log.info(f"  - Vision encoder: {sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad):,} trainable")
        log.info(f"  - Action head: {sum(p.numel() for p in self.action_head.parameters()):,} params")
        log.info(f"  - Separate heads: {cfg.use_separate_heads}")
    
    def _count_parameters(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"{trainable:,} trainable / {total:,} total"
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Training forward pass: compute diffusion loss.
        
        Args:
            batch: Dictionary containing:
                - prev_image: (B, 3, 224, 224)
                - curr_image: (B, 3, 224, 224)
                - goal_image: (B, 3, 224, 224)
                - curr_proprio: (B, proprio_dim)
                - gt_delta_actions: (B, K, action_dim + 1) - ground truth delta pose + gripper
                
        Returns:
            loss: Scalar diffusion loss
        """
        B = batch['curr_image'].shape[0]
        device = batch['curr_image'].device
        
        # 1. Get ground truth actions and normalize to [-1, 1]
        gt_actions_raw = batch['gt_delta_actions']  # (B, K, 8)
        gt_actions = self.action_normalizer.normalize(gt_actions_raw)
        
        # 2. Sample random timesteps
        t = torch.randint(0, self.cfg.diffusion_timesteps, (B,), device=device)
        
        # 3. Sample noise and add to normalized ground truth
        noise = torch.randn_like(gt_actions)
        noisy_actions = self.noise_scheduler.add_noise(gt_actions, t, noise)
        
        # 4. Encode visual context (BATCHED - much more efficient)
        # Create mask for unconditional training (CFG)
        use_uncond = torch.rand(B, device=device) < self.cfg.p_uncond if self.training else torch.zeros(B, device=device, dtype=torch.bool)
        
        # Encode all samples conditionally first
        context_cond, global_cond = self.vision_encoder(
            batch['prev_image'],
            batch['curr_image'],
            batch['goal_image'],
            batch['curr_proprio'],
            return_uncond=False
        )
        
        # Get unconditional embeddings
        context_uncond, global_uncond = self.vision_encoder(
            batch['prev_image'],
            batch['curr_image'],
            batch['goal_image'],
            batch['curr_proprio'],
            return_uncond=True
        )
        
        # Mix conditional and unconditional based on mask
        use_uncond_expanded = use_uncond.view(B, 1, 1)
        context = torch.where(use_uncond_expanded, context_uncond, context_cond)
        
        use_uncond_1d = use_uncond.view(B, 1)
        global_context = torch.where(use_uncond_1d, global_uncond, global_cond)
        
        # 5. Predict noise
        noise_pred = self.action_head(noisy_actions, t, context, global_context)
        
        # 6. Compute diffusion loss (simple MSE)
        diffusion_loss = F.mse_loss(noise_pred, noise)
        
        # 7. Compute phase prediction loss (auxiliary - from Semantic Planner)
        total_loss = diffusion_loss
        if self.cfg.use_phase_prediction and self.phase_head is not None:
            # Use global context for phase prediction
            phase_logits = self.phase_head(global_cond)  # (B, num_phases)
            
            # Get ground truth phase if available
            if 'gt_phase' in batch:
                gt_phase = batch['gt_phase'].long()  # (B,)
                phase_loss = F.cross_entropy(phase_logits, gt_phase)
                total_loss = diffusion_loss + self.cfg.phase_loss_weight * phase_loss
        
        return total_loss
    
    @torch.no_grad()
    def sample(
        self, 
        batch: Dict[str, torch.Tensor],
        num_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Generate action chunk using DDIM sampling.
        
        Args:
            batch: Dictionary containing:
                - prev_image, curr_image, goal_image, curr_proprio
            num_steps: Number of DDIM steps (default: cfg.inference_steps)
            guidance_scale: CFG scale (default: cfg.guidance_scale)
            
        Returns:
            actions: (B, K, action_dim + 1) - delta pose + gripper
        """
        B = batch['curr_image'].shape[0]
        device = batch['curr_image'].device
        num_steps = num_steps or self.cfg.inference_steps
        guidance_scale = guidance_scale or self.cfg.guidance_scale
        
        # 1. Encode conditional context
        context_cond, global_cond = self.vision_encoder(
            batch['prev_image'],
            batch['curr_image'],
            batch['goal_image'],
            batch['curr_proprio'],
            return_uncond=False
        )
        
        # 2. Encode unconditional context (for CFG)
        context_uncond, global_uncond = self.vision_encoder(
            batch['prev_image'],
            batch['curr_image'],
            batch['goal_image'],
            batch['curr_proprio'],
            return_uncond=True
        )
        
        # 3. Start from pure noise
        x_t = torch.randn(B, self.cfg.action_chunk_size, self.cfg.action_dim + 1, device=device)
        
        # 4. Create timestep schedule
        timesteps = torch.linspace(self.cfg.diffusion_timesteps - 1, 0, num_steps, device=device).round().long()
        
        # 5. DDIM sampling loop
        for i in range(num_steps):
            t = timesteps[i]
            t_prev = timesteps[i + 1] if i < num_steps - 1 else -1
            
            t_batch = t.expand(B)
            
            # Predict noise (conditional)
            noise_cond = self.action_head(x_t, t_batch, context_cond, global_cond)
            
            if guidance_scale > 1.0:
                # Predict noise (unconditional)
                noise_uncond = self.action_head(x_t, t_batch, context_uncond, global_uncond)
                
                # Apply CFG
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = noise_cond
            
            # DDIM step
            x_t = self.noise_scheduler.ddim_step(x_t, int(t), int(t_prev), noise_pred)
        
        # 6. Denormalize actions from [-1, 1] back to original range
        actions = self.action_normalizer.denormalize(x_t)
        
        return actions
    
    def get_action(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method to get delta pose and gripper action separately.
        
        Returns:
            delta_pose: (B, 7) - First step delta pose
            gripper: (B, 1) - First step gripper command
        """
        actions = self.sample(batch)  # (B, K, 8)
        delta_pose = actions[:, 0, :7]  # First step, pose only
        gripper = actions[:, 0, 7:]  # First step, gripper
        return delta_pose, gripper


# =============================================================================
# 8. UNIT TEST
# =============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Running unit test on {device}")
    
    # Create config
    cfg = UnifiedDiffusionConfig()
    
    # Create model
    model = UnifiedDiffusionPlanner(cfg).to(device)
    
    # Create dummy batch
    B = 2
    batch = {
        'prev_image': torch.randn(B, 3, 224, 224, device=device),
        'curr_image': torch.randn(B, 3, 224, 224, device=device),
        'goal_image': torch.randn(B, 3, 224, 224, device=device),
        'curr_proprio': torch.randn(B, cfg.proprio_dim, device=device),
        'gt_delta_actions': torch.randn(B, cfg.action_chunk_size, cfg.action_dim + 1, device=device),
    }
    
    # Test training forward pass
    log.info("Testing training forward pass...")
    model.train()
    loss = model(batch)
    log.info(f"  Loss: {loss.item():.4f}")
    assert loss.shape == (), f"Expected scalar loss, got {loss.shape}"
    
    # Test inference
    log.info("Testing inference (DDIM sampling)...")
    model.eval()
    actions = model.sample(batch)
    log.info(f"  Actions shape: {actions.shape}")
    assert actions.shape == (B, cfg.action_chunk_size, cfg.action_dim + 1), f"Unexpected shape: {actions.shape}"
    
    # Test get_action
    log.info("Testing get_action...")
    delta_pose, gripper = model.get_action(batch)
    log.info(f"  Delta pose shape: {delta_pose.shape}")
    log.info(f"  Gripper shape: {gripper.shape}")
    
    log.info("All tests passed!")
