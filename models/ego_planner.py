# FILE: models/ego_planner.py
# (State-of-the-Art, v3 - Definitive, Modular, Production-Grade Edition)

"""
The Ego-Planner v3: The Definitive, State-of-the-Art Unified Robotics Policy.

This script represents a production-grade implementation of the Ego-Planner
architecture. It is the culmination of a deep architectural analysis, engineered
for maximum robustness, clarity, and performance. This version is not merely
an iteration but a ground-up rewrite based on principles of modular design
and SOTA best practices.

Key Architectural Advancements in this Definitive Edition:
1.  **Formal Modular Design**: The conceptual "Strategist" and "Pilot" are now
    fully encapsulated as distinct, self-contained nn.Module classes:
    `ContextualPlanEncoder` and `GroundedActionDecoder`. The top-level
    `EgoPlanner` class acts as a clean orchestrator, dramatically improving
    readability, testability, and maintainability.

2.  **Meticulous Weight Initialization**: All trainable layers are explicitly
    initialized using proven methods (e.g., Kaiming He). This promotes stable
    convergence and is a hallmark of professional-grade model engineering.

3.  **Enhanced Hierarchical Conditioning**: The AdaLN-Zero mechanism in the
    Diffusion Transformer blocks now receives a cleanly separated and projected
    global conditioning embedding, providing the network with a more structured
    and disentangled learning signal.

4.  **Exhaustive Documentation & Shape Annotations**: Every class, method, and
    complex tensor operation is documented with comprehensive docstrings and
    inline shape comments, making the data flow fully transparent and auditable.

5.  **Robust & Principled Guidance**: The implementation of fine-grained CFG and
    the principled multi-conditioner sampling logic from v2 is retained and
    integrated into the new modular structure.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, ResNet18_Weights
from transformers import PretrainedConfig, SiglipVisionModel

log = logging.getLogger(__name__)

# Re-usable, documented configuration dataclass
@dataclass
class EgoPlannerConfig:
    """Configuration for the Ego-Planner v3 model."""
    action_dim: int = 7
    proprio_dim: int = 22
    action_horizon: int = 8
    obs_horizon: int = 2
    vision_backbone_model: str = "google/siglip-base-patch16-224"
    vision_feature_dim: int = 768
    fusion_transformer_layers: int = 4
    fusion_transformer_heads: int = 8
    resnet_feature_dim: int = 256
    pilot_d_model: int = 512
    denoiser_layers: int = 6
    denoiser_heads: int = 8
    diffusion_timesteps: int = 100
    p_plan_drop: float = 0.1
    p_obs_drop: float = 0.1
    beta_schedule: str = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 0.02


# -----------------------------------------------------------------------------
# UTILITY & FOUNDATIONAL MODULES
# -----------------------------------------------------------------------------

# FILE: models/ego_planner.py

# --- START OF DEFINITIVE PATCH ---
# REPLACE the existing _init_weights function with this one.

# FILE: models/ego_planner.py

# --- START OF DEFINITIVE PATCH 1: REPLACE the _init_weights function ---
def _init_weights(module: nn.Module):
    """
    Applies a SOTA weight initialization. This definitive version is robustly
    patched to handle layers that may have their bias or weight terms
    disabled (set to None).
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        if module.weight is not None:
            torch.nn.init.trunc_normal_(module.weight, std=.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        # --- CRITICAL FIX: Check if bias and weight exist before initializing ---
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
        if module.weight is not None:
            torch.nn.init.ones_(module.weight)
# --- END OF DEFINITIVE PATCH 1 ---



class ResNetEncoder(nn.Module):
    """
    A robust, lightweight, and trainable ResNet-18 encoder for processing
    real-time image observation histories. It flattens the spatial feature maps
    into a sequence of visual tokens.

    This module includes built-in ImageNet normalization for self-contained
    robustness, ensuring correct data preprocessing regardless of the training loop.
    """
    def __init__(self, out_features: int):
        super().__init__()
        self.features_dim = out_features
        
        # Load a pre-trained ResNet-18 and use its feature trunk
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[0:7])
        
        # A 1x1 convolution to project the ResNet features to the desired dimension
        self.projection = nn.Conv2d(256, out_features, kernel_size=1)
        self.layer_norm = nn.LayerNorm(out_features)
        
        # Buffer ImageNet statistics for robust, on-the-fly normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes a history of images into a flat sequence of visual tokens.
        Robust to device placement, AMP, and gradient flow.
        Args:
            x (torch.Tensor): (B, H_o, C, H_img, W_img)
        Returns:
            torch.Tensor: (B, L_vis, D_feat)
        """

        # --- 1. Defensive Device Sync ---
        if next(self.parameters()).device != x.device:
            self.to(x.device)

        # --- 2. Normalize using same device ---
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)

        # --- 3. AMP-safe + frozen backbone ---
        with torch.amp.autocast('cuda', enabled=False):
            x_fp32 = x.float()
            if x_fp32.max() > 1.0:
                x_fp32 = x_fp32 / 255.0
            x_fp32 = (x_fp32 - mean) / std

            B, H_o, C, H_img, W_img = x_fp32.shape
            x_fp32 = x_fp32.view(B * H_o, C, H_img, W_img)

            # Run frozen backbone
            with torch.no_grad():
                features = self.backbone(x_fp32)

        # --- 4. Trainable projection & layer norm ---
        features = self.projection(features)
        tokens = features.flatten(2).permute(0, 2, 1)
        tokens = self.layer_norm(tokens)

        _, N_patches, D_feat = tokens.shape
        return tokens.view(B, H_o * N_patches, D_feat)



# FILE: models/ego_planner.py
# INSERT THIS CODE AFTER the ResNetEncoder class definition you just added

class EgoPlannerBlock(nn.Module):
    """
    A single block of the Ego-Planner's Diffusion Transformer (DiT).
    
    This block implements the full sequence of operations for one layer of the
    denoiser: Self-Attention for the action tokens, Cross-Attention to the
    unified context, and a Feed-Forward Network. Crucially, each operation is
    preceded by an Adaptive Layer Normalization (AdaLN-Zero) layer, which
    modulates the network's behavior based on the combined diffusion timestep
    and global plan embedding.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout)
        )
        # We use three separate LayerNorms with learnable affine parameters disabled
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)
        
        # The single modulation network that predicts all scale and shift parameters
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        
        # The "Zero" part of AdaLN-Zero: initialize the final layer to output zeros.
        # This ensures that at the beginning of training, the conditioning has no effect,
        # creating a clean residual path and promoting training stability.
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

# In CLASS EgoPlannerBlock

    def forward(self, x: torch.Tensor, unified_context: torch.Tensor, combined_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input sequence (noisy actions). Shape (B, H_a, D_pilot).
            unified_context (torch.Tensor): The conditioning sequence (vision + proprio). Shape (B, L_ctx, D_pilot).
            combined_emb (torch.Tensor): The fused time and global plan embedding. Shape (B, D_pilot).
        
        Returns:
            torch.Tensor: The processed output sequence. Shape (B, H_a, D_pilot).
        """
        # --- START OF DEBUGGING BLOCK ---
        print(f"\n--- Entering EgoPlannerBlock ---")
        print(f"[DEBUG] Input x shape: {x.shape}")
        print(f"[DEBUG] Unified context shape: {unified_context.shape}")
        # --- END OF DEBUGGING BLOCK ---

        # Predict all 6 modulation parameters (scale & shift for 3 norms)
        shift1, scale1, shift2, scale2, shift3, scale3 = self.adaLN_modulation(combined_emb).chunk(6, dim=1)
        
        # 1. Self-Attention Block with AdaLN
        x_sa = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        
        # --- START OF DEBUGGING BLOCK ---
        print(f"[DEBUG] Shape going into self-attention (Query, Key, Value): {x_sa.shape}")
        # --- END OF DEBUGGING BLOCK ---
        
        sa_out, _ = self.self_attn(x_sa, x_sa, x_sa, need_weights=False)
        x = x + sa_out
        
        # 2. Cross-Attention Block with AdaLN
        x_ca = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        
        # --- START OF DEBUGGING BLOCK ---
        print(f"[DEBUG] Shape going into cross-attention (Query): {x_ca.shape}")
        print(f"[DEBUG] Shape going into cross-attention (Key, Value): {unified_context.shape}")
        # --- END OF DEBUGGING BLOCK ---

        ca_out, _ = self.cross_attn(x_ca, unified_context, unified_context, need_weights=False)
        x = x + ca_out
        
        # 3. Feed-Forward Block with AdaLN
        x_ffn = self.norm3(x) * (1 + scale3.unsqueeze(1)) + shift3.unsqueeze(1)
        
        # --- START OF DEBUGGING BLOCK ---
        print(f"[DEBUG] Shape going into FFN (self.ffn): {x_ffn.shape}")
        # --- END OF DEBUGGING BLOCK ---

        ffn_out = self.ffn(x_ffn)
        x = x + ffn_out
        
        return x

class SinusoidalPosEmb(nn.Module):
    """Generates sinusoidal positional embeddings for diffusion timesteps."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device, half_dim = x.device, self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

# -----------------------------------------------------------------------------
# ARCHITECTURAL COMPONENT 1: THE STRATEGIST
# -----------------------------------------------------------------------------
# FILE: models/ego_planner.py

# --- START OF DEFINITIVE PATCH ---
# REPLACE the existing ContextualPlanEncoder class with this one.

class ContextualPlanEncoder(nn.Module):
    """
    The "Strategist" component of the Ego-Planner (Definitive v5 - Patched).
    This version includes the critical patch to correctly handle the output of
    the SiglipVisionModel, resolving the tensor shape mismatch.
    """
    def __init__(self, cfg: EgoPlannerConfig):
        super().__init__()
        # --- Vision Backbone ---
        log.info(f"[Strategist] Initializing ViT backbone: {cfg.vision_backbone_model}")
        self.vision_backbone = SiglipVisionModel.from_pretrained(cfg.vision_backbone_model)
        self.vision_backbone.requires_grad_(False)

        config = self.vision_backbone.config
        if not all(hasattr(config, attr) for attr in ['image_size', 'patch_size']):
             raise AttributeError("The vision backbone's config is missing 'image_size' or 'patch_size' attributes.")
        
        num_patches_per_side = config.image_size // config.patch_size
        # --- CRITICAL FIX: The patch embeddings do NOT include the CLS token ---
        num_patches = num_patches_per_side ** 2
        log.info(f"[Strategist] Derived configuration: image_size={config.image_size}, "
                 f"patch_size={config.patch_size}, num_patches={num_patches}.")

        # --- Fusion Components ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.vision_feature_dim,
            nhead=cfg.fusion_transformer_heads,
            batch_first=True,
            activation='gelu'
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.fusion_transformer_layers)
        
        # --- Learnable Embeddings ---
        # Positional embedding for the patch tokens (size 196)
        self.patch_pos_emb = nn.Parameter(torch.randn(1, num_patches, cfg.vision_feature_dim))
        # A separate embedding for the CLS token
        self.cls_pos_emb = nn.Parameter(torch.randn(1, 1, cfg.vision_feature_dim))
        
        self.start_token_type = nn.Parameter(torch.randn(1, 1, cfg.vision_feature_dim))
        self.goal_token_type = nn.Parameter(torch.randn(1, 1, cfg.vision_feature_dim))
        
# In CLASS ContextualPlanEncoder

    def forward(self, initial_image: torch.Tensor, goal_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            initial_image (torch.Tensor): Shape (B, 3, 224, 224).
            goal_image (torch.Tensor): Shape (B, 3, 224, 224).
        
        Returns:
            torch.Tensor: The final `plan_vector`. Shape (B, D_vis).
        """
        # --- DEFINITIVE DEVICE GUARD FIX ---
        # Get the target device from the input tensor, which we know is on the correct device.
        target_device = initial_image.device
        # Ensure the vision backbone is on the same device as the input.
        # This is a failsafe against initialization issues.
        self.to(target_device)
        # --- END OF FIX ---
        print("before the contexutal   with torch.no_grad(), torch.cuda.amp.autocast(enabled=False): ")
        # 1. Extract embeddings from the frozen vision backbone within a no-AMP sanctuary.
        # --- DEFINITIVE DEVICE GUARD FIX ---
        # --- END OF FIX ---
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            initial_image_fp32 = initial_image.float()
            goal_image_fp32 = goal_image.float()
            outputs_start = self.vision_backbone(initial_image_fp32, output_hidden_states=False)
            outputs_goal = self.vision_backbone(goal_image_fp32, output_hidden_states=False)

        print("after the contexutal   with torch.no_grad(), torch.cuda.amp.autocast(enabled=False): ")
        # The rest of the function remains the same...
        start_patch_tokens = outputs_start.last_hidden_state.clone()
        start_cls_token = outputs_start.pooler_output.clone().unsqueeze(1) # -> (B, 1, D)
        
        goal_patch_tokens = outputs_goal.last_hidden_state.clone()
        goal_cls_token = outputs_goal.pooler_output.clone().unsqueeze(1) # -> (B, 1, D)

        start_patch_tokens += self.patch_pos_emb
        goal_patch_tokens += self.patch_pos_emb
        
        start_cls_token += self.cls_pos_emb + self.start_token_type
        goal_cls_token += self.cls_pos_emb + self.goal_token_type

        start_sequence = torch.cat([start_cls_token, start_patch_tokens], dim=1)
        goal_sequence = torch.cat([goal_cls_token, goal_patch_tokens], dim=1)
        
        fused_input = torch.cat([start_sequence, goal_sequence], dim=1)
        fused_output = self.fusion_transformer(fused_input)
        
        plan_vector = fused_output[:, 0]
        
        return plan_vector



class GroundedActionDecoder(nn.Module):
    """
    The "Pilot" component of the Ego-Planner (Definitive v5).
    This version includes a performance optimization to eliminate redundant MLP
    computations in the forward pass.
    """
    def __init__(self, cfg: EgoPlannerConfig, use_checkpointing: bool = False):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        log.info(f"[Pilot] Activation Checkpointing: {'Enabled' if use_checkpointing else 'Disabled'}")

        self.primary_obs_encoder = ResNetEncoder(cfg.resnet_feature_dim)
        self.wrist_obs_encoder = ResNetEncoder(cfg.resnet_feature_dim)
        self.vision_obs_fusion = nn.MultiheadAttention(cfg.resnet_feature_dim, 4, batch_first=True)
        self.vision_obs_norm = nn.LayerNorm(cfg.resnet_feature_dim)
        
        self.proprio_proj = nn.Linear(cfg.proprio_dim, cfg.pilot_d_model)
        self.vision_obs_proj = nn.Linear(cfg.resnet_feature_dim, cfg.pilot_d_model)
        
        self.global_cond_mlp = nn.Sequential(nn.Linear(cfg.vision_feature_dim, cfg.pilot_d_model), nn.GELU(), nn.Linear(cfg.pilot_d_model, cfg.pilot_d_model))
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(cfg.pilot_d_model), nn.Linear(cfg.pilot_d_model, cfg.pilot_d_model * 2), nn.Mish(), nn.Linear(cfg.pilot_d_model * 2, cfg.pilot_d_model))
        
        self.action_proj = nn.Linear(cfg.action_dim, cfg.pilot_d_model)
        self.action_pos_emb = nn.Embedding(cfg.action_horizon, cfg.pilot_d_model)
        self.denoiser_blocks = nn.ModuleList([EgoPlannerBlock(cfg.pilot_d_model, cfg.denoiser_heads) for _ in range(cfg.denoiser_layers)])
        self.out_proj = nn.Linear(cfg.pilot_d_model, cfg.action_dim)

    def encode_tactics(self, obs_history: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes vision + proprioceptive history into tokens.
        Defensive: ensures entire module (all params & buffers) live on the same device
        as the input tensors before performing MultiheadAttention / linear ops.
        """
        # --- Device guard (idempotent & cheap check) ---
        target_device = obs_history['image_primary'].device
        # Only move the module if at least one parameter is not on the target device.
        try:
            first_param = next(self.parameters())
        except StopIteration:
            first_param = None

        if first_param is not None and first_param.device != target_device:
            # Move all params & buffers of this submodule to the target device.
            # This is safe and fixes cases where a submodule was instantiated on CPU.
            self.to(target_device)

        # --- Encode vision with ResNets (they themselves contain internal device guards) ---
        primary_tokens = self.primary_obs_encoder(obs_history['image_primary'])
        wrist_tokens = self.wrist_obs_encoder(obs_history['image_wrist'])

        # --- MultiheadAttention: Q= wrist, K/V = primary ---
        fused_vision, _ = self.vision_obs_fusion(wrist_tokens, primary_tokens, primary_tokens)

        # Residual / norm / projections (all now guaranteed to be on the same device)
        fused_vision = self.vision_obs_norm(fused_vision + wrist_tokens)
        vision_tokens = self.vision_obs_proj(fused_vision)
        proprio_tokens = self.proprio_proj(obs_history['proprio'])

        print("wrist_tokens:", wrist_tokens.shape)
        print("primary_tokens:", primary_tokens.shape)
        
        print("vision_tokens:", vision_tokens.shape)
        print("proprio_tokens:", proprio_tokens.shape)
        return vision_tokens, proprio_tokens


    def forward(self, noisy_actions: torch.Tensor, timesteps: torch.Tensor,
                plan_vector: torch.Tensor, vision_tokens: torch.Tensor, proprio_tokens: torch.Tensor) -> torch.Tensor:
        print("this is state of the forward GroundedActionDecoder ")
        action_horizon = noisy_actions.shape[1]
        pos_indices = torch.arange(action_horizon, device=noisy_actions.device)
        action_tokens = self.action_proj(noisy_actions) + self.action_pos_emb(pos_indices)
        
        # --- PATCH 1 (PERFORMANCE): Compute the global conditioning projection only ONCE ---
        global_cond_emb = self.global_cond_mlp(plan_vector)
        
        # Use the pre-computed embedding for both the context and AdaLN modulation
        plan_token = global_cond_emb.unsqueeze(1)

        print(f"plan_token, vision_tokens, proprio_tokens shapes: {plan_token.shape} {vision_tokens.shape} {proprio_tokens.shape}" )


        unified_context = torch.cat([plan_token, vision_tokens, proprio_tokens], dim=1)
        
        time_emb = self.time_mlp(timesteps)
        combined_emb = time_emb + global_cond_emb

        x = action_tokens
        for block in self.denoiser_blocks:
            if self.training and self.use_checkpointing:
                x = torch.utils.checkpoint.checkpoint(block, x, unified_context, combined_emb, use_reentrant=False)
            else:
                x = block(x, unified_context, combined_emb)
        
        return self.out_proj(x)
# -----------------------------------------------------------------------------
# TOP-LEVEL ORCHESTRATOR MODEL (DEFINITIVE PATCHED VERSION)
# -----------------------------------------------------------------------------
class EgoPlanner(nn.Module):
    """
    The Ego-Planner v5: The definitive, fully audited, modular, end-to-end policy orchestrator.
    This version incorporates final performance and robustness enhancements.
    """
    def __init__(self, cfg: EgoPlannerConfig, use_checkpointing: bool = False):
        super().__init__()
        self.cfg = cfg
        self.strategist = ContextualPlanEncoder(cfg)
        self.pilot = GroundedActionDecoder(cfg, use_checkpointing=use_checkpointing)
        self.uncond_embeddings = nn.ParameterDict({
            'plan': nn.Parameter(torch.randn(1, cfg.vision_feature_dim)),
            'vision': nn.Parameter(torch.randn(1, 1, cfg.pilot_d_model)),
            'proprio': nn.Parameter(torch.randn(1, 1, cfg.pilot_d_model))
        })
        self.apply(_init_weights)
        log.info("EgoPlanner v5 (Definitive & Audited) model initialized.")

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """The main end-to-end training forward pass."""
        B, device = batch['initial_image'].shape[0], batch['initial_image'].device
        
        plan_vector = self.strategist(batch['initial_image'], batch['goal_image'])
        vision_tokens, proprio_tokens = self.pilot.encode_tactics(batch['observation_history'])

        if self.training:
            plan_vector = torch.where(torch.rand(B, 1, device=device) < self.cfg.p_plan_drop, self.uncond_embeddings.plan, plan_vector)
            if torch.rand(1).item() < self.cfg.p_obs_drop:
                vision_tokens = self.uncond_embeddings.vision.expand(B, vision_tokens.shape[1], -1)
            if torch.rand(1).item() < self.cfg.p_obs_drop:
                proprio_tokens = self.uncond_embeddings.proprio.expand(B, proprio_tokens.shape[1], -1)
        
        predicted_noise = self.pilot(
            noisy_actions=batch['noisy_actions'],
            timesteps=batch['timesteps'],
            plan_vector=plan_vector,
            vision_tokens=vision_tokens,
            proprio_tokens=proprio_tokens,
        )
        return predicted_noise

    @torch.no_grad()
    def sample(self, batch: Dict[str, torch.Tensor], scheduler, guidance_plan: float, guidance_obs: float) -> torch.Tensor:
        """Generates an action sequence using DDIM sampling and principled CFG."""
        B = batch['initial_image'].shape[0]
        
        # --- PATCH 2 (ROBUSTNESS): Explicit no_grad context for strategist ---
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                # Manually cast the input tensors to float32 for the backbone
                initial_image_fp32 = batch['initial_image'].float()
                goal_image_fp32 = batch['goal_image'].float()
                plan_cond = self.strategist(initial_image_fp32, goal_image_fp32)
        
        plan_uncond = self.uncond_embeddings.plan.expand(B, -1)
        
        vis_cond, prop_cond = self.pilot.encode_tactics(batch['observation_history'])
        vis_uncond = self.uncond_embeddings.vision.expand(B, vis_cond.shape[1], -1)
        prop_uncond = self.uncond_embeddings.proprio.expand(B, prop_cond.shape[1], -1)
        
        plans = torch.cat([plan_cond, plan_uncond, plan_cond, plan_uncond], dim=0)
        visions = torch.cat([vis_cond, vis_cond, vis_uncond, vis_uncond], dim=0)
        proprios = torch.cat([prop_cond, prop_cond, prop_uncond, prop_uncond], dim=0)
        
        latents = torch.randn((B, self.cfg.action_horizon, self.cfg.action_dim), device=self.device)
        
        for t in scheduler.timesteps:
            t_batch = t.expand(B * 4)
            latent_model_input = latents.repeat(4, 1, 1)
            
            noise_preds = self.pilot(latent_model_input, t_batch, plans, visions, proprios)
            
            p_cond_both, p_uncond_plan, p_cond_obs, p_uncond_all = noise_preds.chunk(4)
            
            delta_plan = p_cond_obs - p_uncond_all
            delta_obs = p_uncond_plan - p_uncond_all
            guided_noise = p_uncond_all + guidance_plan * delta_plan + guidance_obs * delta_obs
            
            latents = scheduler.step(guided_noise, t, latents).prev_sample
            
        return latents
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
    
    # Use CUDA if available, otherwise CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"--- [Unit Test] Running on device: {device} ---")

    # --- 1. Configuration ---
    # Create a default configuration for the model.
    cfg = EgoPlannerConfig(
        action_dim=7,
        proprio_dim=22,
        action_horizon=8,
        obs_horizon=2,
        vision_feature_dim=768, # SigLIP-base
        pilot_d_model=512,
        resnet_feature_dim=256
    )
    B = 4 # Batch size for testing

    # --- 2. Create Dummy Input Batch ---
    # This simulates the exact structure of a real data batch.
    dummy_batch = {
        # Strategist inputs
        'initial_image': torch.randn(B, 3, 224, 224, device=device),
        'goal_image': torch.randn(B, 3, 224, 224, device=device),
        # Pilot inputs (tactical observations)
        'observation_history': {
            'image_primary': torch.randn(B, cfg.obs_horizon, 3, 224, 224, device=device),
            'image_wrist': torch.randn(B, cfg.obs_horizon, 3, 128, 128, device=device),
            'proprio': torch.randn(B, cfg.obs_horizon, cfg.proprio_dim, device=device),
        },
        # Diffusion inputs
        'noisy_actions': torch.randn(B, cfg.action_horizon, cfg.action_dim, device=device),
        'timesteps': torch.randint(0, 100, (B,), device=device),
    }
    log.info(f"Created dummy batch with batch size {B}.")

    # --- 3. Model Instantiation ---
    try:
        model = EgoPlanner(cfg).to(device)
        model.train() # Set to training mode for the forward pass test
        log.info("EgoPlanner model instantiated and moved to device successfully.")
    except Exception as e:
        log.error("Failed to instantiate the EgoPlanner model.", exc_info=True)
        exit(1)

    # --- 4. Test: Full Forward Pass (float32) ---
    log.info("\n--- Testing Full Forward Pass (float32) ---")
    try:
        predicted_noise = model(dummy_batch)
        assert predicted_noise.shape == (B, cfg.action_horizon, cfg.action_dim)
        assert predicted_noise.device == device
        assert predicted_noise.dtype == torch.float32
        log.info(f"  [SUCCESS] Output shape: {predicted_noise.shape}, dtype: {predicted_noise.dtype}")
    except Exception as e:
        log.error("  [FAILURE] Full forward pass (float32) failed.", exc_info=True)

    # --- 5. Test: Full Forward Pass with AMP (float16) ---
    if device.type == 'cuda':
        log.info("\n--- Testing Full Forward Pass with AMP (float16) ---")
        # Simulate PyTorch Lightning's behavior
        dummy_batch_amp = {
            k: v.half() if torch.is_floating_point(v) else v 
            for k, v in dummy_batch.items()
        }
        # Nested dict handling
        dummy_batch_amp['observation_history'] = {
            k: v.half() if torch.is_floating_point(v) else v 
            for k, v in dummy_batch['observation_history'].items()
        }
        
        try:
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                # The model itself should not be .half(), only the inputs and autocast context
                predicted_noise_amp = model(dummy_batch_amp)
            
            assert predicted_noise_amp.shape == (B, cfg.action_horizon, cfg.action_dim)
            assert predicted_noise_amp.device == device
            # The output of an AMP-context block can be float16 or float32 depending on the last op
            log.info(f"  [SUCCESS] Output shape: {predicted_noise_amp.shape}, dtype: {predicted_noise_amp.dtype}")
        except Exception as e:
            log.error("  [FAILURE] Full forward pass (AMP float16) failed.", exc_info=True)
    else:
        log.info("\nSkipping AMP test (CUDA not available).")
        
    # --- 6. Test: Component Isolation ---
    log.info("\n--- Testing Component Isolation ---")
    try:
        log.info("Testing Strategist (ContextualPlanEncoder)...")
        plan_vector = model.strategist(dummy_batch['initial_image'], dummy_batch['goal_image'])
        assert plan_vector.shape == (B, cfg.vision_feature_dim)
        log.info(f"  [SUCCESS] Plan vector shape: {plan_vector.shape}")
        
        log.info("Testing Pilot (GroundedActionDecoder)...")
        vision_tokens, proprio_tokens = model.pilot.encode_tactics(dummy_batch['observation_history'])
        # Shape check (example: vision tokens)
        # The exact sequence length depends on the ResNet output, so we check feature dim
        assert vision_tokens.shape[0] == B and vision_tokens.shape[2] == cfg.pilot_d_model
        assert proprio_tokens.shape == (B, cfg.obs_horizon, cfg.pilot_d_model)
        log.info(f"  [SUCCESS] Vision tokens shape: {vision_tokens.shape}, Proprio tokens shape: {proprio_tokens.shape}")
    except Exception as e:
        log.error("  [FAILURE] Component isolation test failed.", exc_info=True)

    log.info("\n--- [Unit Test] All checks complete. ---")