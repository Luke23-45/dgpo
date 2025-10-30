# FILE: models/planner.py
# SOTA Visual Planner Model (CLIP-Diffusion Hybrid)

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional, Dict, Any

# SOTA Libraries for components
try:
    # Use CLIP for rich semantic vision encoding
    from transformers import CLIPVisionModel, ViTConfig
    from diffusers import (
        UNet2DConditionModel,
        DDIMScheduler, # Use SOTA scheduler
        DDIMScheduler, # Keep for compatibility if needed
    )
    from diffusers.configuration_utils import ConfigMixin
    from diffusers.models.modeling_utils import ModelMixin
except ImportError:
    raise ImportError("Please install transformers and diffusers: pip install transformers diffusers")

from tqdm import tqdm
import logging
import torch.nn.functional as F

log = logging.getLogger(__name__)


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Helper module for SOTA sinusoidal position embeddings,
    used to encode the continuous 'progress' scalar.
    """
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Embedding dimension dim ({dim}) must be even.")
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time (torch.Tensor): A 1D tensor of progress values, shape (B,).
                                 Values should be in the range [0, 1].
        Returns:
            torch.Tensor: Sinusoidal embeddings, shape (B, dim).
        """
        device = time.device
        # Scale time from [0, 1] to a larger range (e.g., 0-1000)
        # for a better distribution in the sinusoidal space.
        time = time * 1000.0

        half_dim = self.dim // 2
        # Calculate embedding frequencies
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        # Calculate sin/cos embeddings
        # time[:, None] expands (B,) to (B, 1)
        # embeddings[None, :] expands (half_dim,) to (1, half_dim)
        # Resulting shape is (B, half_dim)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings


class VisualPlannerDiffusion( ModelMixin, ConfigMixin):
    """
    SOTA Goal-Conditioned Visual Subgoal Generator.
    [cite: 36]
    Generates a future subgoal image based on current image, goal image, and progress.
    [cite: 37]
    Implements Classifier-Free Guidance (CFG) for high-quality, goal-adherent generation.
    Uses a CLIP vision encoder for SOTA semantic understanding.
    """
    def __init__(self,
                 image_size: int = 224,
                 # SOTA Default: Use CLIP-Large for its powerful semantic features
                 vit_model_name: str = 'openai/clip-vit-large-patch14',
                 vit_feature_dim: int = 768, # CLIP-Large feature dimension is 768
                 freeze_vit: bool = True,
                 
                 progress_embed_dim: int = 64,
                 
                 unet_block_out_channels: Tuple[int, ...] = (128, 128, 256, 256, 512, 512),
                 
                 unet_down_block_types: Tuple[str, ...] = ("DownBlock2D",)*2 + ("CrossAttnDownBlock2D",)*4,
                 
                 unet_up_block_types: Tuple[str, ...] = ("CrossAttnUpBlock2D",)*4 + ("UpBlock2D",)*2,
                 
                 unet_cross_attention_dim: Optional[int] = None,
                 
                 unet_attention_head_dim: int = 8, # Explicitly set attention heads
                 num_diffusion_timesteps: int = 100,
                 
                 condition_drop_prob: float = 0.1, # Dropout probability for SOTA Classifier-Free Guidance
                 ):
        super().__init__()
        # Register config for saving/loading with diffusers .save_pretrained()
        
        self.register_to_config(
             image_size=image_size, vit_model_name=vit_model_name, vit_feature_dim=vit_feature_dim,
             freeze_vit=freeze_vit, progress_embed_dim=progress_embed_dim,
             unet_block_out_channels=unet_block_out_channels, unet_down_block_types=unet_down_block_types,
             unet_up_block_types=unet_up_block_types, unet_cross_attention_dim=unet_cross_attention_dim,
             num_diffusion_timesteps=num_diffusion_timesteps,
             condition_drop_prob=condition_drop_prob,
             unet_attention_head_dim=unet_attention_head_dim,
        )

        log.info(f"Initializing SOTA VisualPlannerDiffusion with Vision Encoder: {vit_model_name}")
        
        # --- 1. SOTA Vision Encoder (CLIP) ---
        # We use CLIPVisionModel for its superior semantic feature extraction
        self.vision_encoder = CLIPVisionModel.from_pretrained(vit_model_name)
        if freeze_vit:
            log.info("Freezing Vision Encoder (CLIP) weights.")
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        else:
             log.info("Fine-tuning Vision Encoder (CLIP) weights.")

        # --- 2. SOTA Progress Encoder (Sinusoidal) ---
        # Use sinusoidal embeddings for the continuous progress value [0, 1]
        self.progress_embedding = SinusoidalPositionEmbeddings(progress_embed_dim)
        # Followed by a robust MLP to project into the final embedding space
        self.progress_encoder = nn.Sequential(
            nn.Linear(progress_embed_dim, progress_embed_dim * 4),
            nn.GELU(), # Use GELU activation (more modern than ReLU)
            nn.Linear(progress_embed_dim * 4, progress_embed_dim)
        )
        actual_vit_feature_dim = self.vision_encoder.config.hidden_size
        
        # 2. Add a diagnostic print to expose the mismatch.
        if actual_vit_feature_dim != vit_feature_dim:
            log.warning("="*80)
            log.warning(f"Configuration Mismatch Detected in VisualPlannerDiffusion:")
            log.warning(f"  - Config `vit_feature_dim` was set to: {vit_feature_dim}")
            log.warning(f"  - The loaded model '{vit_model_name}' actually has a feature dim of: {actual_vit_feature_dim}")
            log.warning("  - Proceeding by using the actual model's dimension.")
            log.warning("="*80)
        # --- 3. Conditioning Fusion & CFG ---
        # Total dimension of conditioning vector
        self.config.vit_feature_dim = actual_vit_feature_dim
        
        condition_dim = (actual_vit_feature_dim * 2) + progress_embed_dim
        # Use config value if provided, otherwise use calculated dim
        cross_attn_dim = unet_cross_attention_dim if unet_cross_attention_dim is not None else condition_dim

        # Optional projection layer if condition_dim doesn't match cross_attn_dim
        self.condition_proj = nn.Linear(condition_dim, cross_attn_dim) if condition_dim != cross_attn_dim else nn.Identity()
        
        log.info(f"Conditioning dim: {condition_dim}, U-Net Cross Attention dim: {cross_attn_dim}")

        # SOTA: Learned embedding for Classifier-Free Guidance (unconditional state)
        # This single embedding will be used when we drop conditioning during training
        self.uncond_embedding = nn.Parameter(torch.randn(1, 1, cross_attn_dim))

        # --- 4. Diffusion U-Net ---
        log.info("Initializing Diffusion U-Net...")
        self.unet = UNet2DConditionModel(
            sample_size=image_size,
            in_channels=3, # Input noisy image
            out_channels=3, # Output predicted noise
            block_out_channels=unet_block_out_channels,
            
            down_block_types=unet_down_block_types,
            
            up_block_types=unet_up_block_types,
            
            cross_attention_dim=cross_attn_dim, # Dimension of conditioning
            attention_head_dim=unet_attention_head_dim,
        )

        # --- 5. SOTA Noise Scheduler (DPM-Solver++) ---
        # This scheduler provides SOTA results in few inference steps
        print("before noise_scheduler::::::")
        log.info("Using DDIMScheduler for robust CPU compatibility.")
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_diffusion_timesteps,
            beta_schedule='squaredcos_cap_v2',
            prediction_type='epsilon'
        )
        log.info("SOTA Planner components initialized.")

    def encode_condition(self, current_image: torch.Tensor, goal_image: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        """... docstring ..."""
        batch_size = current_image.shape[0]

        # --- Process images with SOTA (CLIP) Vision Encoder ---
        with torch.no_grad() if self.config.freeze_vit else torch.enable_grad():
            
            # --- HIGH-PRECISION DEBUGGING ---
            print("        [encode_condition] ==> ABOUT TO CALL vision_encoder for current_image...")
            outputs_current = self.vision_encoder(pixel_values=current_image)
            print("        [encode_condition] ==> SUCCESS: vision_encoder for current_image COMPLETED.")
            
            print("        [encode_condition] ==> ABOUT TO CALL vision_encoder for goal_image...")
            outputs_goal = self.vision_encoder(pixel_values=goal_image)
            print("        [encode_condition] ==> SUCCESS: vision_encoder for goal_image COMPLETED.")
            # ---------------------------------

        current_features = outputs_current.pooler_output
        goal_features = outputs_goal.pooler_output

        # --- Process progress scalar ---
        progress_emb = self.progress_encoder(self.progress_embedding(progress))

        # --- Concatenate all features ---
        condition = torch.cat([current_features, goal_features, progress_emb], dim=-1)
        condition = self.condition_proj(condition)

        return condition.unsqueeze(1)

# In models/planner.py -> VisualPlannerDiffusion class

    def forward(self,
                gt_subgoal_image: torch.Tensor,
                current_image: torch.Tensor,
                goal_image: torch.Tensor,
                progress: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """... docstring ..."""
        device = gt_subgoal_image.device
        batch_size = gt_subgoal_image.shape[0]

        # --- 1. Encode conditioning vectors ---
        # --- HIGH-PRECISION DEBUGGING ---
        print("        [model.forward] ==> ABOUT TO CALL self.encode_condition...")
        condition = self.encode_condition(current_image, goal_image, progress)
        print("        [model.forward] ==> SUCCESS: self.encode_condition COMPLETED.")
        # ---------------------------------

        # --- 2. Implement CFG (Training) ---
        drop_mask = (torch.rand(batch_size, 1, 1, device=device) < self.config.condition_drop_prob)
        uncond_condition = self.uncond_embedding.expand(batch_size, -1, -1)
        final_condition = torch.where(drop_mask, uncond_condition, condition)

        # --- 3. Standard Diffusion Training Steps ---
        epsilon = torch.randn_like(gt_subgoal_image)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
        noisy_subgoal = self.noise_scheduler.add_noise(gt_subgoal_image, epsilon, timesteps)

        # --- HIGH-PRECISION DEBUGGING ---
        print(f"        [model.forward] ==> ABOUT TO CALL self.unet... (Input shape: {noisy_subgoal.shape})")
        predicted_epsilon = self.unet(
            sample=noisy_subgoal,
            timestep=timesteps,
            encoder_hidden_states=final_condition
        ).sample
        print("        [model.forward] ==> SUCCESS: self.unet COMPLETED.")
        # ---------------------------------

        return predicted_epsilon, epsilon


# In models/planner.py -> VisualPlannerDiffusion class

    @torch.no_grad()
    def sample(self,
               current_image: torch.Tensor,
               goal_image: torch.Tensor,
               progress: torch.Tensor,
               num_inference_steps: int = 20,
               guidance_scale: float = 7.5,
               generator: Optional[torch.Generator] = None
              ) -> torch.Tensor:
        """... docstring ..."""
        device = current_image.device
        batch_size = current_image.shape[0]

        # 1. Set inference timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.noise_scheduler.timesteps

        # 2. Encode condition
        cond_condition = self.encode_condition(current_image, goal_image, progress)
        uncond_condition = self.uncond_embedding.expand(batch_size, -1, -1)
        condition = torch.cat([uncond_condition, cond_condition], dim=0)

        # 3. Initialize latents
        latents = torch.randn((batch_size, 3, self.config.image_size, self.config.image_size),
                              generator=generator, device=device, dtype=condition.dtype)
        latents = latents * self.noise_scheduler.init_noise_sigma
        
        print("        [model.sample] ==> SETUP COMPLETE. About to enter denoising loop...")

        # 4. SOTA Denoising loop (with CFG)
        for i, t in enumerate(tqdm(timesteps, desc="Planner Sampling", leave=False, disable=True)):
            print(f"        [model.sample] ==> LOOP START: Iteration {i}, Timestep {t.item()}")

            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)
            
            print(f"        [model.sample] ==> ABOUT TO CALL self.unet (inside loop)...")
            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=condition
            ).sample
            print(f"        [model.sample] ==> SUCCESS: self.unet (inside loop) COMPLETED.")

            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            guided_noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            latents = self.noise_scheduler.step(guided_noise_pred, t, latents).prev_sample
            print(f"        [model.sample] ==> LOOP END: Iteration {i}")

        print("        [model.sample] ==> SUCCESS: Denoising loop finished.")
        image = latents
        return image


# Example Usage / Unit Test
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
    print(f"Using device: {device}")

    # Example config
    planner = VisualPlannerDiffusion(
        image_size=128, # Smaller for testing
        vit_model_name='openai/clip-vit-base-patch32', # Use smaller CLIP for testing
        vit_feature_dim=512, # CLIP-Base-32 dim
        freeze_vit=True,
        num_diffusion_timesteps=50, # Fewer steps for faster testing

        condition_drop_prob=0.1
    ).to(device)

    # Test training forward pass
    bs = 4
    current = torch.randn(bs, 3, 128, 128, device=device)
    goal = torch.randn(bs, 3, 128, 128, device=device)

    prog = torch.rand(bs, device=device) # Progress values [0, 1]

    gt_subgoal = torch.randn(bs, 3, 128, 128, device=device)


    print("Testing training forward pass (with CFG dropout)...")
    pred_noise, true_noise = planner(gt_subgoal, current, goal, prog)
    print("Training forward pass output shapes:")
    print("Predicted Noise:", pred_noise.shape)
    print("True Noise:", true_noise.shape)
    loss = F.mse_loss(pred_noise, true_noise)
    print("Example Loss:", loss.item())

    # Test inference pass (with CFG guidance)
    print("\nTesting inference pass (with CFG guidance)...")
    generated_image = planner.sample(
        current,
        goal,
        prog,
        num_inference_steps=10,
        guidance_scale=4.0 # Use some guidance for testing
    )
    print("Inference pass output shape:")
    print("Generated Image:", generated_image.shape)

    # Test saving/loading config (if using ModelMixin/ConfigMixin correctly)
    # [cite: 64]
    # import tempfile
    # with tempfile.TemporaryDirectory() as tmpdir:
    #     planner.save_pretrained(tmpdir)
    #     loaded_planner = VisualPlannerDiffusion.from_pretrained(tmpdir).to(device)
    #     print("\nModel config saved and loaded successfully.")
    #     # Test loaded model
    #     loaded_image = loaded_planner.sample(current, goal, prog, num_inference_steps=2)
    #     print("Loaded model generated image shape:", loaded_image.shape)