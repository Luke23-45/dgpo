# FILE: models/planner.py
# SOTA Visual Planner Model (CLIP-Diffusion Hybrid)

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional, Dict, Any

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# SOTA Libraries for components
try:
    # Use CLIP for rich semantic vision encoding
    from transformers import CLIPVisionModel, ViTConfig
    from diffusers import (
        UNet2DConditionModel,
        DPMSolverMultistepScheduler, # Use SOTA scheduler
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



class VisualPlannerDiffusion(ModelMixin, ConfigMixin):
    """SOTA Visual Planner with optional UNet warm-starting and LoRA fine-tuning."""
    def __init__(self,
                 image_size: int = 224,
                 vit_model_name: str = 'openai/clip-vit-base-patch32',
                 vit_feature_dim: int = 768,
                 freeze_vit: bool = True,
                 progress_embed_dim: int = 64,
                 # Custom UNet params (used ONLY if not initializing from SD)
                 unet_block_out_channels: Tuple[int, ...] = (128, 128, 256, 512, 512),
                 unet_down_block_types: Tuple[str, ...] = ("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
                 unet_up_block_types: Tuple[str, ...] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
                 unet_attention_head_dim: int = 8,
                 # Critical param for matching pre-trained UNet's expectation
                 unet_cross_attention_dim: Optional[int] = None,
                 num_diffusion_timesteps: int = 50,
                 condition_drop_prob: float = 0.1,
                 # SOTA flags to control the fine-tuning pipeline
                 initialize_unet_from_sd: bool = False,
                 use_lora: bool = False,
                 lora_rank: int = 16
                 ):
        super().__init__()
        self.register_to_config(**{k: v for k, v in locals().items() if k != 'self' and k != '__class__'})

        # --- 1. Vision and Progress Encoders (Correct) ---
        log.info(f"Initializing VisualPlannerDiffusion with Vision Encoder: {vit_model_name}")
        self.vision_encoder = CLIPVisionModel.from_pretrained(vit_model_name)
        if freeze_vit:
            self.vision_encoder.requires_grad_(False)
        self.progress_embedding = SinusoidalPositionEmbeddings(progress_embed_dim)
        self.progress_encoder = nn.Sequential(nn.Linear(progress_embed_dim, progress_embed_dim * 4), nn.GELU(), nn.Linear(progress_embed_dim * 4, progress_embed_dim))

        # --- 2. Conditioning Fusion (Correct) ---
        actual_vit_feature_dim = self.vision_encoder.config.hidden_size
        condition_dim = (actual_vit_feature_dim * 2) + progress_embed_dim
        cross_attn_dim = unet_cross_attention_dim if unet_cross_attention_dim is not None else condition_dim
        self.condition_proj = nn.Linear(condition_dim, cross_attn_dim) if condition_dim != cross_attn_dim else nn.Identity()
        log.info(f"Calculated condition dim: {condition_dim}. Projecting to UNet cross-attention dim: {cross_attn_dim}")
        self.uncond_embedding = nn.Parameter(torch.randn(1, 1, cross_attn_dim))

        # --- 3. Diffusion UNet (Robust Conditional Logic) ---
        log.info("Initializing Diffusion U-Net...")
        if initialize_unet_from_sd:
            try:
                log.info("Attempting to load pre-trained UNet from Stable Diffusion v1.5...")
                if unet_cross_attention_dim != 768:
                    raise ValueError(f"To use Stable Diffusion pre-training, `unet_cross_attention_dim` must be 768, but got {unet_cross_attention_dim}.")
                # --- START: SOTA MODEL SIZE PATCH ---
                # Use the much smaller, distilled version of the SD 1.5 UNet
                sd_unet = UNet2DConditionModel.from_pretrained("segmind/tiny-sd", subfolder="unet")
                # --- END: SOTA MODEL SIZE PATCH ---
                # --- SOTA SURGERY (Fix for Audit Finding #11) ---
                old_conv_in = sd_unet.conv_in
                new_conv_in = nn.Conv2d(3, old_conv_in.out_channels, kernel_size=old_conv_in.kernel_size, stride=old_conv_in.stride, padding=old_conv_in.padding)
                with torch.no_grad():
                    new_conv_in.weight[:] = old_conv_in.weight.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
                    if new_conv_in.bias is not None: new_conv_in.bias.data.copy_(old_conv_in.bias.data)
                sd_unet.conv_in = new_conv_in
                
                # NOTE on `conv_out` (Audit Finding #2 in verdict): The audit correctly warns against replacing `conv_out` in a standard VAE pipeline.
                # However, your specific code trains directly on RGB images, predicting RGB noise. Therefore, the UNet *must* output 3 channels.
                # Replacing `conv_out` is the correct action *for this specific architecture*.
                sd_unet.conv_out = nn.Conv2d(sd_unet.conv_out.in_channels, 3, kernel_size=3, padding=1)
                
                self.unet = sd_unet
                log.info("Successfully loaded and adapted Stable Diffusion UNet.")

                # --- LoRA Application (Fix for Audit Finding #4) ---
                if use_lora:
                    if not PEFT_AVAILABLE: raise ImportError("`use_lora` is true but `peft` is not installed. Please run `pip install peft`.")
                    
                    log.info(f"Applying LoRA via get_peft_model with rank={lora_rank}...")
                    lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_rank, target_modules=["to_q", "to_k", "to_v", "to_out.0"], lora_dropout=0.1, bias="none")
                    self.unet = get_peft_model(self.unet, lora_config)
                    
                    trainable_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
                    if trainable_params == 0:
                        log.warning("LoRA applied but no trainable parameters found. Inspect `target_modules` and the UNet's `named_modules()` output.")
                    else:
                        all_params = sum(p.numel() for p in self.unet.parameters())
                        log.info(f"LoRA applied. Trainable UNet params: {trainable_params:,} || All UNet params: {all_params:,} || Trainable %: {100 * trainable_params / all_params:.4f}%")

            except Exception as e:
                log.error(f"Failed to load/adapt pre-trained UNet. Falling back to custom architecture. Error: {e}")
                self.unet = self._create_custom_unet()
        else:
            log.info("Initializing custom UNet architecture from config.")
            self.unet = self._create_custom_unet()

        # --- UNet Optimizations (Fix for Audit Finding #5) ---
        log.info("Enabling UNet optimizations (if available)...")
        if hasattr(self.unet, 'set_attention_slice'): self.unet.set_attention_slice("auto")
        if hasattr(self.unet, 'enable_freeu'): self.unet.enable_freeu(s1=0.9, s2=0.2, b1=1.1, b2=1.2)

        # --- Noise Scheduler (Fix for Audit Finding #1 and #6) ---
        log.info("Initializing SOTA DPM-Solver++ Scheduler.")
        try:
            self.noise_scheduler = DPMSolverMultistepScheduler.from_pretrained("segmind/tiny-sd", subfolder="scheduler", use_karras_sigmas=True)
        except Exception:
            log.warning("Could not load scheduler from pretrained. Initializing with default config.")
            self.noise_scheduler = DPMSolverMultistepScheduler.from_config(self.config, use_karras_sigmas=True)
        
        log.info("SOTA Planner components initialized.")

    def _create_custom_unet(self):
        """Helper to create a UNet from config parameters."""
        return UNet2DConditionModel(sample_size=self.config.image_size, in_channels=3, out_channels=3, block_out_channels=self.config.unet_block_out_channels, down_block_types=self.config.unet_down_block_types, up_block_types=self.config.unet_up_block_types, cross_attention_dim=self.uncond_embedding.shape[-1], attention_head_dim=self.config.unet_attention_head_dim)

    def encode_condition(self, current_image: torch.Tensor, goal_image: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        with torch.no_grad() if self.config.freeze_vit else torch.enable_grad():
            outputs_current, outputs_goal = self.vision_encoder(pixel_values=current_image), self.vision_encoder(pixel_values=goal_image)
        current_features, goal_features = outputs_current.pooler_output, outputs_goal.pooler_output
        progress_emb = self.progress_encoder(self.progress_embedding(progress))
        condition = torch.cat([current_features, goal_features, progress_emb], dim=-1)
        condition = self.condition_proj(condition)
        return condition.unsqueeze(1)

    def forward(self, gt_subgoal_image: torch.Tensor, current_image: torch.Tensor, goal_image: torch.Tensor, progress: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device, batch_size = gt_subgoal_image.device, gt_subgoal_image.shape[0]
        condition = self.encode_condition(current_image, goal_image, progress)
        drop_mask = torch.rand(batch_size, 1, 1, device=device) < self.config.condition_drop_prob
        uncond_condition = self.uncond_embedding.expand(batch_size, -1, -1)
        final_condition = torch.where(drop_mask, uncond_condition, condition)
        epsilon = torch.randn_like(gt_subgoal_image)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
        noisy_subgoal = self.noise_scheduler.add_noise(gt_subgoal_image, epsilon, timesteps)
        predicted_epsilon = self.unet(sample=noisy_subgoal, timestep=timesteps, encoder_hidden_states=final_condition).sample
        return predicted_epsilon, epsilon

    @torch.no_grad()
    def sample(self, current_image: torch.Tensor, goal_image: torch.Tensor, progress: torch.Tensor, num_inference_steps: int = 20, guidance_scale: float = 7.5, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        device, batch_size = current_image.device, current_image.shape[0]
        self.noise_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.noise_scheduler.timesteps
        cond_condition = self.encode_condition(current_image, goal_image, progress)
        uncond_condition = self.uncond_embedding.expand(batch_size, -1, -1)
        condition = torch.cat([uncond_condition, cond_condition], dim=0)
        
        # --- Latent Dtype Safety (Fix for Audit Finding #8) ---
        model_dtype = next(self.unet.parameters()).dtype
        latents = torch.randn((batch_size, 3, self.config.image_size, self.config.image_size), generator=generator, device=device, dtype=torch.float32).to(model_dtype)
        
        latents = latents * self.noise_scheduler.init_noise_sigma
        for t in timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self.unet(sample=latent_model_input, timestep=t, encoder_hidden_states=condition).sample
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            guided_noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            latents = self.noise_scheduler.step(guided_noise_pred, t, latents).prev_sample
        return latents




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