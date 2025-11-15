# To be placed in models/uhp.py

from __future__ import annotations

import logging
import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, ResNet18_Weights
import torch.nn.functional as F
from transformers import SiglipVisionModel
from typing import Tuple, Any
# Initialize logger for this module
logger = logging.getLogger(__name__)



class LinearNormalizer:
    """
    A stateful class for robust data normalization, adapted from SOTA Diffusion
    Policy implementations. It is not an nn.Module.

    This utility learns the min/max statistics from a dataset and provides
    methods to normalize PyTorch tensors to the [-1, 1] range and un-normalize
    them back to their original scale. This is essential for stable training
    and for generating actions in the correct physical units during inference.
    """

    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, data: np.ndarray):
        """
        Computes and stores the min and max values from the training data.
        This should be called once before training begins.

        Args:
            data: A NumPy array of shape (N_samples, D_features).
        """
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)
        logger.info(f"LinearNormalizer fitted. Min shape: {self.min.shape}, Max shape: {self.max.shape}")

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalizes a PyTorch tensor to the [-1, 1] range using the fitted statistics.

        Args:
            data (torch.Tensor): The input tensor to normalize.
        Returns:
            torch.Tensor: The normalized tensor.
        """
        if self.min is None or self.max is None:
            raise RuntimeError("Normalizer must be fitted before use.")

        min_t = torch.tensor(self.min, dtype=data.dtype, device=data.device)
        max_t = torch.tensor(self.max, dtype=data.dtype, device=data.device)

        range_t = max_t - min_t
        # Add a small epsilon for features with no variance to prevent division by zero.
        return 2 * (data - min_t) / (range_t + 1e-8) - 1

    def unnormalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Un-normalizes a PyTorch tensor from [-1, 1] back to its original scale.

        Args:
            data (torch.Tensor): The normalized tensor.
        Returns:
            torch.Tensor: The tensor restored to its original scale.
        """
        if self.min is None or self.max is None:
            raise RuntimeError("Normalizer must be fitted before use.")

        min_t = torch.tensor(self.min, dtype=data.dtype, device=data.device)
        max_t = torch.tensor(self.max, dtype=data.dtype, device=data.device)

        range_t = max_t - min_t
        return (data + 1) / 2 * range_t + min_t


class SinusoidalPosEmb(nn.Module):
    """
    [DEFINITIVE, CORRECTED VERSION]
    A standard nn.Module for creating sinusoidal positional embeddings, used for
    encoding continuous values like diffusion timesteps into a high-dimensional space.
    This version includes a critical fix to ensure correct tensor broadcasting.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): A 1D tensor of continuous values (e.g., timesteps) of shape [B].
        Returns:
            torch.Tensor: The positional embeddings of shape [B, dim].
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        # This is the critical fix: ensure `x` has a trailing dimension of 1
        # so that it can be correctly broadcast with the embedding frequencies.
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        emb = x * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ResNetEncoder(nn.Module):
    """
    [SOTA, SELF-CONTAINED VERSION]
    A robust, trainable ResNet-18 encoder for processing image observation histories.
    It flattens spatial feature maps into a sequence of visual tokens.

    Key Features:
    - Self-contained ImageNet normalization for preprocessing robustness.
    - Defensive device placement to prevent hardware mismatch errors.
    - AMP-safe forward pass for stable mixed-precision training.
    """
    def __init__(self, out_features: int):
        super().__init__()
        self.features_dim = out_features

        # Load a pre-trained ResNet-18 and use its feature trunk (frozen).
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[0:7])
        self.backbone.requires_grad_(False) # Freeze the backbone.

        # A trainable 1x1 convolution to project ResNet features to the desired dimension.
        self.projection = nn.Conv2d(256, out_features, kernel_size=1)
        self.layer_norm = nn.LayerNorm(out_features)

        # Buffer ImageNet statistics for on-the-fly normalization.
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Processes a history of images into a flat sequence of visual tokens.

        Args:
            images (torch.Tensor): Shape `[B, H_obs, 3, H_img, W_img]`.
        Returns:
            torch.Tensor: Shape `[B, L_tokens, D_features]`.
        """
        # Defensive device placement for the entire module.
        if next(self.parameters()).device != images.device:
            self.to(images.device)

        # AMP-safe normalization and feature extraction.
        with torch.amp.autocast(device_type='cuda', enabled=False):
            images_fp32 = images.float()
            # Normalize to [0,1] if input is uint8 [0,255].
            if images_fp32.max() > 1.0:
                images_fp32 = images_fp32 / 255.0

            normalized_images = (images_fp32 - self.mean) / self.std

            B, H_obs, C, H_img, W_img = normalized_images.shape
            # Reshape for batch processing by the ResNet.
            reshaped_images = normalized_images.view(B * H_obs, C, H_img, W_img)

            # Feature extraction with the frozen backbone.
            with torch.no_grad():
                features = self.backbone(reshaped_images)

        # Trainable projection, flattening, and normalization.
        projected_features = self.projection(features.to(self.projection.weight.dtype))
        tokens = projected_features.flatten(2).permute(0, 2, 1)
        tokens = self.layer_norm(tokens)

        # Reshape back to include the batch dimension and observation horizon.
        num_patches = tokens.shape[1]
        return tokens.view(B, H_obs * num_patches, self.features_dim)


class EgoPlannerBlock(nn.Module):
    """
    [SOTA, AdaLN-Zero VERSION]
    A single block of a Diffusion Transformer (DiT), serving as the core of the Executor.

    This block implements a Pre-LN transformer architecture with AdaLN-Zero
    conditioning and cross-attention. The AdaLN-Zero mechanism allows the
    diffusion timestep and subgoal embedding to deeply modulate the denoising
    process, which is critical for high-performance conditional generation.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

        # LayerNorms with learnable affine parameters disabled for AdaLN.
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)

        # A single modulation network that predicts all scale and shift parameters.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model)  # 3 norms, each needs a scale and shift.
        )
        # Initialize the final layer to zero for the "Zero" part of AdaLN-Zero.
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, unified_context: torch.Tensor, combined_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input sequence (noisy actions). Shape `[B, H_action, D_pilot]`.
            unified_context (torch.Tensor): The conditioning sequence (vision + proprio). Shape `[B, L_context, D_pilot]`.
            combined_emb (torch.Tensor): The fused time and subgoal embedding. Shape `[B, D_pilot]`.

        Returns:
            torch.Tensor: The processed output sequence. Shape `[B, H_action, D_pilot]`.
        """
        # Predict all scale and shift parameters from the combined embedding.
        shift1, scale1, shift2, scale2, shift3, scale3 = self.adaLN_modulation(combined_emb).chunk(6, dim=1)

        # 1. Self-Attention Block (Pre-LN with AdaLN modulation)
        x_sa = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        sa_out, _ = self.self_attn(query=x_sa, key=x_sa, value=x_sa, need_weights=False)
        x = x + sa_out

        # 2. Cross-Attention Block
        x_ca = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        ca_out, _ = self.cross_attn(query=x_ca, key=unified_context, value=unified_context, need_weights=False)
        x = x + ca_out

        # 3. Feed-Forward Block
        x_ffn = self.norm3(x) * (1 + scale3.unsqueeze(1)) + shift3.unsqueeze(1)
        ffn_out = self.ffn(x_ffn)
        x = x + ffn_out

        return x
    




class Sequencer(nn.Module):
    """
    [SOTA, Hybrid-Supervision VERSION]
    The Sequencer is the high-level planner for the UHP architecture.

    It uses a frozen vision backbone to encode the current and goal images,
    and a trainable Fusion Transformer to reason about them in the context of
    the current task phase. It produces two outputs in parallel:
    1.  A dense `subgoal_embedding` (the command) for the Executor.
    2.  A `predicted_heatmap` (the teaching tool) for auxiliary loss during training.
    """

    def __init__(self,
                 vision_backbone_model: str = "google/siglip-base-patch16-224",
                 num_task_phases: int = 5,
                 fusion_transformer_layers: int = 4,
                 fusion_transformer_heads: int = 8,
                 subgoal_dim: int = 768
                 ):
        super().__init__()
        logger.info(f"Initializing Sequencer with backbone: {vision_backbone_model}")

        # --- 2.1. Vision Backbone (Frozen) ---
        self.vision_backbone = SiglipVisionModel.from_pretrained(vision_backbone_model)
        self.vision_backbone.requires_grad_(False)

        # Dynamically determine the hidden dimension from the backbone's config.
        # This makes the model robust to different vision backbone sizes.
        config = self.vision_backbone.config
        self.vision_dim = getattr(config, 'hidden_size', 768)
        logger.info(f"Determined vision model hidden dimension: {self.vision_dim}")

        # --- 2.2. Learnable Embeddings ---
        self.task_phase_embedding = nn.Embedding(num_task_phases, self.vision_dim)
        # 0: current_image tokens, 1: goal_image tokens, 2: task_phase token
        self.token_type_embeddings = nn.Embedding(3, self.vision_dim)

        # --- 2.3. Fusion Transformer (Trainable) ---
        # Using a Pre-LN (norm_first=True) architecture for improved stability.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.vision_dim,
            nhead=fusion_transformer_heads,
            dim_feedforward=self.vision_dim * 4,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=fusion_transformer_layers
        )

        # --- 2.4. Output Heads (Dual, Parallel) ---

        # 2.4.1. Primary Head (produces the command for the Executor)
        self.subgoal_head = nn.Sequential(
            nn.Linear(self.vision_dim, self.vision_dim // 2),
            nn.GELU(),
            nn.Linear(self.vision_dim // 2, subgoal_dim)
        )




    def forward(self,
                current_image: torch.Tensor,
                goal_image: torch.Tensor,
                task_phase: torch.Tensor
                ) -> torch.Tensor:
        """
        Performs the full, end-to-end planning pass.

        Args:
            current_image (torch.Tensor): The current visual observation [B, 3, 224, 224].
            goal_image (torch.Tensor): The final goal image [B, 3, 224, 224].
            task_phase (torch.Tensor): The current task phase enum [B].

        Returns:
            A tuple containing:
            - subgoal_embedding (torch.Tensor): The dense command vector [B, D_subgoal].
        """
        B = current_image.shape[0]
        device = current_image.device

        # Defensive device placement for the vision backbone.
        if self.vision_backbone.device != device:
            self.vision_backbone.to(device)

        # --- Step 1: Encode Images with Frozen Backbone ---
        with torch.no_grad():
            current_img_outputs = self.vision_backbone(pixel_values=current_image.to(self.vision_backbone.dtype))
            goal_img_outputs = self.vision_backbone(pixel_values=goal_image.to(self.vision_backbone.dtype))

        # Get tokens in their native dtype (likely float32 from the frozen model).
        current_tokens = current_img_outputs.last_hidden_state
        goal_tokens = goal_img_outputs.last_hidden_state

        # --- Step 2: Prepare Tokens for Fusion ---
        # Embed the task phase. Shape: [B, 1, D_vision].
        phase_token = self.task_phase_embedding(task_phase).unsqueeze(1)
        
        # --- [START OF FINAL PATCH] ---
        # Add token type embeddings for differentiation. This is CRITICAL for the
        # transformer to distinguish the source of each token.
        current_tokens += self.token_type_embeddings(torch.zeros(1, 1, dtype=torch.long, device=device))
        goal_tokens += self.token_type_embeddings(torch.ones(1, 1, dtype=torch.long, device=device))
        phase_token += self.token_type_embeddings(torch.full((1, 1), 2, dtype=torch.long, device=device))
        # --- [END OF FINAL PATCH] ---

        # --- Step 3: Fuse Information in the Transformer ---
        full_sequence = torch.cat([current_tokens, goal_tokens, phase_token], dim=1)

        # SOTA PATCH: Ensure absolute type consistency before the transformer.
        target_dtype = self.fusion_transformer.layers[0].linear1.weight.dtype
        fused_sequence = self.fusion_transformer(full_sequence.to(target_dtype))

        # --- Step 4: Decode Outputs from the Dual Heads ---
        # 4.1. Primary Path: Generate the `subgoal_embedding`.
        cls_token_output = fused_sequence[:, 0, :]
        subgoal_embedding = self.subgoal_head(cls_token_output)

        return subgoal_embedding
    

class Executor(nn.Module):
    """
    [SOTA, Diffusion Transformer VERSION]
    The Executor is the low-level diffusion policy for the UHP architecture.

    It is a conditional denoising model that takes a noisy action sequence and,
    conditioned on a high-level `subgoal_embedding` and a history of local
    observations, predicts the noise to be removed. It uses a state-of-the-art
    Transformer architecture with distinct conditioning mechanisms:
    - Cross-Attention for incorporating local observation history.
    - AdaLN-Zero for modulating the network based on the global subgoal command.
    """

    def __init__(self,
                 action_dim: int,
                 proprio_dim: int,
                 action_horizon: int,
                 obs_horizon: int,
                 subgoal_dim: int = 768,
                 pilot_hidden_dim: int = 512,
                 resnet_feature_dim: int = 256,
                 denoiser_layers: int = 6,
                 denoiser_heads: int = 8
                 ):
        super().__init__()
        logger.info(f"Initializing Executor with hidden_dim: {pilot_hidden_dim}")
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.action_dim = action_dim

        # --- 3.1. Tactical Observation Encoders ---
        self.primary_obs_encoder = ResNetEncoder(out_features=resnet_feature_dim)
        self.wrist_obs_encoder = ResNetEncoder(out_features=resnet_feature_dim)

        self.vision_obs_fusion = nn.MultiheadAttention(
            embed_dim=resnet_feature_dim,
            num_heads=4,
            batch_first=True
        )
        self.vision_obs_norm = nn.LayerNorm(resnet_feature_dim)

        # Final projections to the Executor's main hidden dimension (D_pilot).
        self.vision_obs_proj = nn.Linear(resnet_feature_dim, pilot_hidden_dim)
        self.proprio_proj = nn.Linear(proprio_dim, pilot_hidden_dim)

        # --- 3.2. Conditioning Encoders ---
        # Projects the high-level command from the Sequencer.
        self.subgoal_proj = nn.Linear(subgoal_dim, pilot_hidden_dim)

        # Encodes the diffusion timestep.
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(pilot_hidden_dim),
            nn.Linear(pilot_hidden_dim, pilot_hidden_dim * 4),
            nn.Mish(),
            nn.Linear(pilot_hidden_dim * 4, pilot_hidden_dim),
        )

        # --- 3.3. Denoising Core (Diffusion Transformer) ---
        self.action_proj = nn.Linear(action_dim, pilot_hidden_dim)
        self.action_pos_emb = nn.Embedding(action_horizon, pilot_hidden_dim)

        self.denoiser_blocks = nn.ModuleList([
            EgoPlannerBlock(d_model=pilot_hidden_dim, n_heads=denoiser_heads)
            for _ in range(denoiser_layers)
        ])

        # --- 3.4. Output Head ---
        self.out_proj = nn.Linear(pilot_hidden_dim, action_dim)

    def forward(self,
                observation_history: Dict[str, torch.Tensor],
                subgoal_embedding: torch.Tensor,
                noisy_action_sequence: torch.Tensor,
                diffusion_timestep: torch.Tensor
                ) -> torch.Tensor:
        """
        Performs the full, end-to-end denoising pass for the Executor.

        Args:
            observation_history: Dict of sensor data, e.g., {'image_primary': [B, H_obs, ...], ...}
            subgoal_embedding: The dense command from the Sequencer. Shape `[B, D_subgoal]`.
            noisy_action_sequence: The noisy action trajectory. Shape `[B, H_action, D_action]`.
            diffusion_timestep: The current denoising timestep. Shape `[B]`.

        Returns:
            torch.Tensor: The predicted noise. Shape `[B, H_action, D_action]`.
        """
        # --- Step 1: Encode Tactical Observations (The "How") ---
        # Process image history through ResNets.
        primary_tokens_raw = self.primary_obs_encoder(observation_history['image_primary'])
        wrist_tokens_raw = self.wrist_obs_encoder(observation_history['image_wrist'])

        # Fuse visual tokens (Query=wrist, Key=Value=primary) and project.
        fused_vision, _ = self.vision_obs_fusion(
            query=wrist_tokens_raw, key=primary_tokens_raw, value=primary_tokens_raw
        )
        fused_vision = self.vision_obs_norm(fused_vision + wrist_tokens_raw) # Residual connection
        vision_tokens = self.vision_obs_proj(fused_vision)

        # Process proprioceptive history.
        proprio_tokens = self.proprio_proj(observation_history['proprio'])

        # Assemble the cross-attention context.
        unified_context = torch.cat([vision_tokens, proprio_tokens], dim=1)

        # --- Step 2: Prepare Conditioning Signals (The "What" and "When") ---
        # Project the strategic command from the Sequencer.
        projected_subgoal_emb = self.subgoal_proj(subgoal_embedding)

        # Encode the diffusion timestep.
        time_emb = self.time_mlp(diffusion_timestep)

        # Create the combined embedding for AdaLN modulation.
        combined_emb = time_emb + projected_subgoal_emb

        # --- Step 3: Prepare Input for Denoising ---
        # Project the noisy actions and add positional embeddings.
        action_tokens = self.action_proj(noisy_action_sequence)
        pos_indices = torch.arange(self.action_horizon, device=action_tokens.device)
        action_tokens += self.action_pos_emb(pos_indices)

        # --- Step 4: Run the Denoising Process ---
        x = action_tokens
        for block in self.denoiser_blocks:
            # Each block is conditioned on the local context (cross-attention)
            # and modulated by the global command (AdaLN).
            x = block(x=x, unified_context=unified_context, combined_emb=combined_emb)

        # --- Step 5: Project Back to Action Space ---
        predicted_noise = self.out_proj(x)

        return predicted_noise
    



class UHP_Orchestrator(nn.Module):
    """
    [SOTA, DEFINITIVE VERSION]
    The Unified Hierarchical Policy (UHP) Orchestrator.

    This module acts as the top-level container for the UHP architecture, managing
    the interaction between the `Sequencer` (high-level planner) and the `Executor`
    (low-level diffusion policy). It is designed with two distinct modes of operation:

    1.  **Training (`forward`):** Implements a single, end-to-end computational
        graph. Gradients from the final action loss flow all the way back through
        the Executor and into the Sequencer, forcing the two modules to learn a
        robust, shared communication protocol via the `subgoal_embedding`.

    2.  **Inference (`plan`, `act`):** Provides separate, clean methods to run the
        planning and action generation steps. This decoupling is essential for
        real-world control loops. The `act` method is engineered for safety,
        incorporating data normalization and kinematic clamping.
    """

    def __init__(self, sequencer_cfg: Dict[str, Any], executor_cfg: Dict[str, Any]):
        super().__init__()
        logger.info("Initializing the UHP Orchestrator...")
        self.sequencer = Sequencer(**sequencer_cfg)
        self.executor = Executor(**executor_cfg)
        logger.info("UHP Orchestrator initialized successfully.")

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        The unified training forward pass for the UHP architecture.

        Args:
            batch (Dict[str, Any]): A dictionary containing all necessary data, including
                                    planner inputs, controller history, and diffusion targets.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the final predictions
                                     required for the hybrid loss calculation.
        """
        # --- 1. Sequencer Forward Pass (High-Level Planning) ---
        subgoal_embedding = self.sequencer(
            current_image=batch['planner_current_image'],
            goal_image=batch['planner_goal_image'],
            task_phase=batch['planner_task_phase']
        )

        # --- 2. Executor Forward Pass (Low-Level Action Generation) ---
        # The subgoal_embedding from the Sequencer is passed directly to the Executor,
        # creating the end-to-end computational graph.
        predicted_noise = self.executor(
            observation_history=batch['controller_observation_history'],
            subgoal_embedding=subgoal_embedding,
            noisy_action_sequence=batch['noisy_actions'],
            diffusion_timestep=batch['timesteps']
        )

        # --- 3. Return All Predictions for Loss Calculation ---
        return {
            "predicted_noise": predicted_noise
        }

    @torch.no_grad()
    def plan(self,
             current_image: torch.Tensor,
             goal_image: torch.Tensor,
             task_phase: torch.Tensor
             ) -> torch.Tensor:
        """
        Inference-only method to run the Sequencer and generate a plan.

        Returns both the command vector for the `act` method and a visualizable
        heatmap for debugging and interpretation.

        Args:
            current_image (torch.Tensor): The current visual observation [B, 3, 224, 224].
            goal_image (torch.Tensor): The final goal image [B, 3, 224, 224].
            task_phase (torch.Tensor): The current task phase enum [B].

        Returns:
            A tuple containing:
            - subgoal_embedding (torch.Tensor): The dense command vector [B, D_subgoal].
            - sigmoid_heatmap (torch.Tensor): The heatmap probabilities [B, 1, 56, 56].
        """
        self.eval()
        subgoal_embedding = self.sequencer(
            current_image, goal_image, task_phase
        )

        return subgoal_embedding

    @torch.no_grad()
    def act(self,
            observation_history: Dict[str, torch.Tensor],
            subgoal_embedding: torch.Tensor,
            noise_scheduler,
            num_inference_steps: int,
            action_normalizer: LinearNormalizer,
            proprio_normalizer: LinearNormalizer,
            joint_limits_low: torch.Tensor,
            joint_limits_high: torch.Tensor
            ) -> torch.Tensor:
        """
        [DEFINITIVE, KINEMATICS-AWARE, NORMALIZATION-AWARE VERSION]
        Inference-only method to generate a physically plausible action sequence.

        This method performs the full diffusion sampling loop and then applies all
        necessary post-processing (un-normalization and kinematic clamping) to
        produce actions that can be directly executed by a real or simulated robot.

        Returns:
            torch.Tensor: The final, safe, physical-scale action sequence.
        """
        self.eval()

        # --- 1. Prepare Inputs (Normalization and Device Management) ---
        # Create a safe copy to avoid modifying the original data.
        inference_obs_history = observation_history.copy()
        raw_proprio = inference_obs_history['proprio']
        # Normalize proprioceptive data before feeding it to the model.
        normalized_proprio = proprio_normalizer.normalize(raw_proprio)
        inference_obs_history['proprio'] = normalized_proprio

        B = raw_proprio.shape[0]
        device = raw_proprio.device

        # Ensure kinematic limits are on the correct device for the final clamp.
        joint_limits_low = joint_limits_low.to(device)
        joint_limits_high = joint_limits_high.to(device)

        # --- 2. Initialize Diffusion Latents ---
        latents = torch.randn(
            (B, self.executor.action_horizon, self.executor.action_dim),
            device=device,
            dtype=torch.float32
        )

        # --- 3. The Denoising Loop ---
        noise_scheduler.set_timesteps(num_inference_steps)
        for t in noise_scheduler.timesteps:
            timesteps = t.expand(B).to(device)

            predicted_noise = self.executor(
                observation_history=inference_obs_history, # Use the NORMALIZED dict
                subgoal_embedding=subgoal_embedding,
                noisy_action_sequence=latents,
                diffusion_timestep=timesteps
            )

            # Scheduler step to compute the previous, less noisy sample.
            latents = noise_scheduler.step(
                model_output=predicted_noise,
                timestep=t,
                sample=latents
            ).prev_sample

        # At this point, `latents` is the sequence of PREDICTED NORMALIZED ACTIONS.

        # --- 4. Post-Processing (Un-normalization and Safety Clamping) ---
        # Un-normalize the actions back to their original, physical scale.
        unnormalized_actions = action_normalizer.unnormalize(latents)

        # Apply hard kinematic constraints as a final safety net.
        clamped_actions = torch.clamp(
            unnormalized_actions,
            min=joint_limits_low,
            max=joint_limits_high
        )

        return clamped_actions