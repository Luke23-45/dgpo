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
                 initialize_unet_from_sd: bool = True,
                 
                 # Controls whether to apply LoRA for efficient fine-tuning
                 use_lora: bool = True
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
             initialize_unet_from_sd=initialize_unet_from_sd, # Add this
             use_lora=use_lora                          # Add this
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

        if initialize_unet_from_sd:
            try:
                log.info("Attempting to load pre-trained UNet weights from Stable Diffusion Inpainting...")
                pretrained_unet = UNet2DConditionModel.from_pretrained(
                    "runwayml/stable-diffusion-inpainting",
                    subfolder="unet",
                    torch_dtype=torch.float32 # Use float32 for CPU/GPU compatibility during load
                )
                # Load all weights that match in name and shape, skip others (like the input conv layer)
                self.unet.load_state_dict(pretrained_unet.state_dict(), strict=False)
                log.info("Successfully transferred weights from pre-trained UNet.")
            except Exception as e:
                log.error(f"Failed to load pre-trained UNet. The UNet will be trained from scratch. Error: {e}")

        # 3.2: Apply LoRA for Parameter-Efficient Fine-Tuning (PEFT)
        if use_lora:
            if PEFT_AVAILABLE:
                log.info("Applying LoRA to the UNet for efficient fine-tuning...")
                lora_config = LoraConfig(
                    r=16,  # Rank of the adapter matrices. Higher rank = more parameters, more capacity.
                    lora_alpha=32, # A scaling factor.
                    target_modules=["to_q", "to_k", "to_v", "to_out.0"], # Target the attention projections
                    lora_dropout=0.1,
                )
                # Wrap the UNet to create a PEFT model
                self.unet = get_peft_model(self.unet, lora_config)
                log.info("LoRA applied. Trainable parameters:")
                self.unet.print_trainable_parameters()
            else:
                log.warning("`peft` library not found. Cannot apply LoRA. Training the full UNet.")

        # 3.3: Freeze early layers of the UNet to preserve general features
        log.info("Freezing early UNet down_blocks to preserve pre-trained features.")
        # Note: If using LoRA, most of the UNet is already frozen by default.
        # This is an additional safety measure and is critical if NOT using LoRA.
        for name, param in self.unet.named_parameters():
            if "down_blocks.0" in name or "down_blocks.1" in name:
                # If LoRA is active, we only want to freeze non-LoRA parameters
                if 'lora' not in name:
                    param.requires_grad = False

        # 3.4: Apply memory and stability hacks
        log.info("Enabling UNet optimizations: Sliced Attention and FreeU.")
        self.unet.set_attention_slice("auto") # Use less memory during attention
        self.unet.enable_freeu(s1=0.9, s2=0.2, b1=1.1, b2=1.2) # Improve sample quality

        # --- 5. SOTA Noise Scheduler (DPM-Solver++) ---
        # This scheduler provides SOTA results in few inference steps
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_diffusion_timesteps,
            beta_schedule='squaredcos_cap_v2', # A common SOTA schedule
            
            prediction_type='epsilon' # Predict noise
            
        )
        log.info("SOTA Planner components initialized.")

    def encode_condition(self, current_image: torch.Tensor, goal_image: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        """
        Encodes inputs into the conditioning vector for the U-Net.
        [cite: 48]
        Args:
            current_image (torch.Tensor): Batch of current images, shape (B, 3, H, W).
            goal_image (torch.Tensor): Batch of goal images, shape (B, 3, H, W).
            progress (torch.Tensor): Batch of progress scalars, shape (B,).
        Returns:
            torch.Tensor: Combined conditioning vector, shape (B, 1, cross_attn_dim).
        """
        batch_size = current_image.shape[0]

        # --- Process images with SOTA (CLIP) Vision Encoder ---
        with torch.no_grad() if self.config.freeze_vit else torch.enable_grad():
            outputs_current = self.vision_encoder(pixel_values=current_image)
            outputs_goal = self.vision_encoder(pixel_values=goal_image)
            

        # Use the standard CLIP 'pooler_output' (from [CLS] token)
        # This is semantically richer than mean pooling 
        current_features = outputs_current.pooler_output # (B, vit_feature_dim)
        goal_features = outputs_goal.pooler_output     # (B, vit_feature_dim)

        # --- Process progress scalar with SOTA (Sinusoidal) Encoder ---
        # (B,) -> (B, progress_embed_dim) -> (B, progress_embed_dim)
        progress_emb = self.progress_encoder(self.progress_embedding(progress))

        # --- Concatenate all features ---
      
        condition = torch.cat([current_features, goal_features, progress_emb], dim=-1) # (B, condition_dim)

        # Project to final cross-attention dimension
        condition = self.condition_proj(condition) # (B, cross_attn_dim)

        # U-Net expects conditioning shape (B, sequence_length, cross_attn_dim)
        return condition.unsqueeze(1) # (B, 1, cross_attn_dim)

    def forward(self,
                gt_subgoal_image: torch.Tensor,
                
                current_image: torch.Tensor,
                
                goal_image: torch.Tensor,
                
                progress: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        SOTA Training forward pass with Classifier-Free Guidance (CFG) logic.
        Predicts noise given noisy ground truth subgoal.
        [cite: 52]
        Returns:
            predicted_epsilon (torch.Tensor): The noise predicted by the U-Net.
            [cite: 52]
            epsilon (torch.Tensor): The actual noise added to the image.
            [cite: 53]
        """
        device = gt_subgoal_image.device
        batch_size = gt_subgoal_image.shape[0]

        # --- 1. Encode conditioning vectors ---
        condition = self.encode_condition(current_image, goal_image, progress)

        # --- 2. SOTA: Implement CFG (Training) ---
        # Create a random mask for dropping conditions
        drop_mask = (torch.rand(batch_size, 1, 1, device=device) < self.config.condition_drop_prob)

        # Get the learned unconditional embedding, expanded to batch size
        uncond_condition = self.uncond_embedding.expand(batch_size, -1, -1)

        # Select between conditional and unconditional embeddings based on the mask
        # This is the final conditioning tensor for the U-Net
        final_condition = torch.where(drop_mask, uncond_condition, condition)

        # --- 3. Standard Diffusion Training Steps ---
        # 3a. Sample noise
        epsilon = torch.randn_like(gt_subgoal_image)

        # 3b. Sample timesteps
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()

        # 3c. Create noisy image
        
        noisy_subgoal = self.noise_scheduler.add_noise(gt_subgoal_image, epsilon, timesteps)

        # 3d. Predict noise using U-Net
        predicted_epsilon = self.unet(
            sample=noisy_subgoal,            # Noisy image input
            timestep=timesteps,              # Timestep conditioning
            
            encoder_hidden_states=final_condition  # CFG-ready conditioning
            
        ).sample # Get the predicted noise tensor

        return predicted_epsilon, epsilon


    @torch.no_grad()
    def sample(self,
               current_image: torch.Tensor,
               goal_image: torch.Tensor,
               progress: torch.Tensor,
               num_inference_steps: int = 20,
               guidance_scale: float = 7.5,
               generator: Optional[torch.Generator] = None
              ) -> torch.Tensor:
        """
        SOTA Inference pass with Classifier-Free Guidance (CFG).
        """
        device = current_image.device
        batch_size = current_image.shape[0]

        # 1. Set inference timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.noise_scheduler.timesteps

        # 2. Encode condition (and unconditional)
        cond_condition = self.encode_condition(current_image, goal_image, progress)
        uncond_condition = self.uncond_embedding.expand(batch_size, -1, -1)
        # SOTA: Combine for CFG. This part is correct.
        condition = torch.cat([uncond_condition, cond_condition], dim=0)

        # 3. Initialize latents (noisy image)
        # --- START OF SOTA PATCH: CORRECT LATENT HANDLING ---
        # The `latents` variable should ALWAYS have the original batch size (e.g., 16).
        latents = torch.randn((batch_size, 3, self.config.image_size, self.config.image_size),
                              generator=generator, device=device, dtype=condition.dtype)
        # --- END OF SOTA PATCH ---
        
        latents = latents * self.noise_scheduler.init_noise_sigma

        # 4. SOTA Denoising loop (with CFG)
        for t in tqdm(timesteps, desc="Planner Sampling", leave=False, disable=True):
            # --- START OF SOTA PATCH (continued) ---
            # a. Create a temporary, duplicated input for the model.
            #    `latents` (shape [16,...]) is duplicated to `latent_model_input` (shape [32,...])
            latent_model_input = torch.cat([latents] * 2)
            # --- END OF SOTA PATCH (continued) ---
            
            latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)

            # b. Predict noise for *both* cond and uncond in one pass. Output is shape [32, ...]
            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=condition
            ).sample

            # c. Perform Classifier-Free Guidance.
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            # `guided_noise_pred` is the final noise, shape is back to [16, ...]
            guided_noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # --- START OF SOTA PATCH (continued) ---
            # d. Scheduler step.
            #    Crucially, it uses the guided noise (shape [16,...]) and the
            #    original, non-duplicated latents (shape [16,...]).
            #    The shapes now match.
            latents = self.noise_scheduler.step(guided_noise_pred, t, latents).prev_sample
            # --- END OF SOTA PATCH (continued) ---

        # 5. We no longer need to chunk the final latents, as they are already the correct shape.
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