# To be placed in models/vip_c.py

import torch
import torch.nn as nn
from transformers import AutoModel, PretrainedConfig
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple
import logging
import math
from transformers import SiglipVisionModel # Import the CORRECT model class


logger = logging.getLogger(__name__)
# --- I. Component Deep Dive: The Planner ("Foreman") ---
from models.ego_planner import ResNetEncoder, EgoPlannerBlock

class Planner(nn.Module):
    """
    The SOTA, state-aware, high-level policy for the ViP-C framework.

    The Planner's objective is to analyze the current world state in the context
    of the final goal and produce a visually grounded, spatially precise subgoal
    heatmap for the low-level Controller to execute.

    It uses a frozen vision backbone, a trainable fusion transformer to incorporate
    the task phase, and a convolutional decoder head to generate the heatmap.
    """
    def __init__(self,
                 vision_backbone_model: str = "google/siglip-base-patch16-224",
                 num_task_phases: int = 5,
                 fusion_transformer_layers: int = 4,
                 fusion_transformer_heads: int = 12):
        super().__init__()
        
        logger.info(f"Initializing Planner with backbone: {vision_backbone_model}")
        self.vision_backbone = SiglipVisionModel.from_pretrained(vision_backbone_model)
        self.vision_backbone.requires_grad_(False)

        
        # Dynamically get the hidden dimension from the backbone's config
        config = self.vision_backbone.config
        if hasattr(config, 'vision_config'):
            # This handles models like Siglip, CLIP, etc.
            self.hidden_dim = config.vision_config.hidden_size
        elif hasattr(config, 'hidden_size'):
            # This handles standalone vision models like ViT, DeiT.
            self.hidden_dim = config.hidden_size
        else:
            raise AttributeError(
                "Could not determine hidden dimension from the vision backbone's config. "
                "The config does not have 'vision_config.hidden_size' or a top-level 'hidden_size'."
                f"Please inspect the config: {config}"
            )
        
        
        # --- 2. Learnable Embeddings ---
        self.task_phase_embedding = nn.Embedding(num_task_phases, self.hidden_dim)
        # We need embeddings to distinguish the three types of input tokens
        self.token_type_embeddings = nn.Embedding(3, self.hidden_dim) # 0: current, 1: goal, 2: phase

        # --- 3. Fusion Transformer (Trainable) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=fusion_transformer_heads,
            dim_feedforward=self.hidden_dim * 4,
            activation='gelu',
            batch_first=True,
            norm_first=True  # SOTA practice for better stability
        )
        self.fusion_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=fusion_transformer_layers
        )

        # --- 4. Heatmap Generation Head (Trainable) ---
        # This head takes the 14x14 grid of patch embeddings and upsamples it to 56x56
        self.heatmap_head = nn.Sequential(
            # Input: [B, hidden_dim, 14, 14]
            nn.ConvTranspose2d(self.hidden_dim, 256, kernel_size=2, stride=2),
            nn.GELU(),
            nn.LayerNorm([256, 28, 28]), # Add normalization for stability
            # State: [B, 256, 28, 28]
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.GELU(),
            nn.LayerNorm([128, 56, 56]),
            # State: [B, 128, 56, 56]
            nn.Conv2d(128, 1, kernel_size=1),
            # Output: [B, 1, 56, 56]
            nn.Sigmoid() # Ensure output is in [0, 1] range
        )

    def forward(self,
                current_image: torch.Tensor,
                goal_image: torch.Tensor,
                task_phase: torch.Tensor) -> torch.Tensor:
        """
        Performs the full, end-to-end planning pass.

        Args:
            current_image: The current visual observation [B, 3, 224, 224].
            goal_image: The final goal image [B, 3, 224, 224].
            task_phase: The current task phase enum [B].

        Returns:
            The predicted subgoal heatmap [B, 1, 56, 56].
        """
        B = current_image.shape[0]
        device = current_image.device

        # --- Step 1: Encode Images with Frozen Backbone ---
        with torch.no_grad():
            current_img_outputs = self.vision_backbone(pixel_values=current_image)
            goal_img_outputs = self.vision_backbone(pixel_values=goal_image)
        
        # Extract the full sequence of token embeddings (including [CLS])
        current_tokens = current_img_outputs.last_hidden_state
        goal_tokens = goal_img_outputs.last_hidden_state
        
        # --- Step 2: Prepare Tokens for Fusion ---
        
        # Get the task phase embedding token
        phase_token = self.task_phase_embedding(task_phase).unsqueeze(1) # [B, 1, D]

        # Add token type embeddings for differentiation
        current_tokens += self.token_type_embeddings(torch.zeros(1, 1, dtype=torch.long, device=device))
        goal_tokens += self.token_type_embeddings(torch.ones(1, 1, dtype=torch.long, device=device))
        phase_token += self.token_type_embeddings(torch.full((1, 1), 2, dtype=torch.long, device=device))
        
        # --- Step 3: Fuse Information in the Transformer ---

        # Concatenate all tokens into a single long sequence
        # Shape: [B, 197 + 197 + 1, hidden_dim]
        full_sequence = torch.cat([current_tokens, goal_tokens, phase_token], dim=1)
        
        # Process the full sequence. The transformer will create a rich,
        # contextualized representation for every token.
        fused_sequence = self.fusion_transformer(full_sequence)
        
        # --- Step 4: Decode to a Heatmap ---

        # Extract the hidden states corresponding to the current image's *patch* tokens.
        # We take the first 197 tokens (the ones for current_image) and discard the [CLS] token (at index 0).
        contextualized_patch_tokens = fused_sequence[:, 1:197, :] # Shape: [B, 196, D]
        
        # Reshape the sequence of patch tokens back into a spatial grid.
        # The ViT patch order is row-major, so this is a valid operation.
        # [B, 196, D] -> [B, 14, 14, D] -> [B, D, 14, 14]
        patch_grid = contextualized_patch_tokens.permute(0, 2, 1).reshape(B, self.hidden_dim, 14, 14)
        
        # Pass the grid through the convolutional decoder head to upsample to the final heatmap.
        predicted_heatmap = self.heatmap_head(patch_grid)
        
        return predicted_heatmap
    



# --- Helper Module for Positional Embeddings ---

class SinusoidalPosEmb(nn.Module):
    """
    A module for creating sinusoidal positional embeddings, a standard technique
    for encoding continuous values (like time or coordinates) for Transformers.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- [DELETE THE ENTIRE OLD forward METHOD] ---

        # --- [REPLACE WITH THIS DEFINITIVE, CORRECTED METHOD] ---
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        # This is the critical fix. We ensure `x` has a trailing dimension
        # of 1 so that it can be correctly broadcast with the embedding.
        # x (shape [B]) -> x.unsqueeze(-1) (shape [B, 1])
        # x (shape [B, 1]) -> x (shape [B, 1])
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        emb = x * emb.unsqueeze(0) # emb [D/2] -> [1, D/2]
        
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Controller(nn.Module):
    """
    The SOTA, low-level, goal-conditioned diffusion policy for the ViP-C framework.

    The Controller's objective is to receive a precise (u, v) subgoal coordinate
    from the Planner and generate a short, dynamically feasible sequence of
    low-level motor commands to reach it, conditioned on the robot's immediate
    sensor readings.
    """
    def __init__(self,
                 action_dim: int,
                 proprio_dim: int,
                 action_horizon: int,
                 obs_horizon: int,
                 controller_hidden_dim: int = 512,
                 resnet_feature_dim: int = 256,
                 denoiser_layers: int = 6,
                 denoiser_heads: int = 8):
        super().__init__()
        logger.info(f"Initializing Controller with hidden_dim: {controller_hidden_dim}")

        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

        # --- 1. Real-Time Observation Encoders ---
        # We reuse the battle-tested ResNet encoders and fusion from EgoPlanner.
        self.primary_obs_encoder = ResNetEncoder(out_features=resnet_feature_dim)
        self.wrist_obs_encoder = ResNetEncoder(out_features=resnet_feature_dim)
        
        # A simple cross-attention to fuse the two visual streams
        self.vision_obs_fusion = nn.MultiheadAttention(
            embed_dim=resnet_feature_dim,
            num_heads=4,
            batch_first=True
        )
        self.vision_obs_norm = nn.LayerNorm(resnet_feature_dim)

        # Final projections to the controller's main hidden dimension
        self.vision_obs_proj = nn.Linear(resnet_feature_dim, controller_hidden_dim)
        self.proprio_proj = nn.Linear(proprio_dim, controller_hidden_dim)
        
        # --- 2. Subgoal Encoder ---
        # This encoder transforms the [B, 2] coordinate into a powerful token.
        subgoal_embed_dim = 128
        self.subgoal_pos_emb = SinusoidalPosEmb(subgoal_embed_dim)
        self.subgoal_encoder = nn.Sequential(
            nn.Linear(subgoal_embed_dim * 2, controller_hidden_dim), # *2 for u and v
            nn.GELU(),
            nn.Linear(controller_hidden_dim, controller_hidden_dim)
        )

        # --- 3. Denoising Core (Diffusion Transformer) ---
        
        # Timestep embedding (standard for diffusion models)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(controller_hidden_dim),
            nn.Linear(controller_hidden_dim, controller_hidden_dim * 4),
            nn.Mish(),
            nn.Linear(controller_hidden_dim * 4, controller_hidden_dim),
        )

        # Projection for the noisy action sequence
        self.action_proj = nn.Linear(action_dim, controller_hidden_dim)
        # Learnable positional embeddings for the action horizon
        self.action_pos_emb = nn.Embedding(action_horizon, controller_hidden_dim)

        # The stack of Transformer blocks that performs the denoising.
        self.denoiser_blocks = nn.ModuleList([
            EgoPlannerBlock(d_model=controller_hidden_dim, n_heads=denoiser_heads)
            for _ in range(denoiser_layers)
        ])

        # The final projection layer back to the action dimension
        self.out_proj = nn.Linear(controller_hidden_dim, action_dim)

    def forward(self,
                observation_history: Dict[str, torch.Tensor],
                subgoal_coordinate: torch.Tensor,
                noisy_action_sequence: torch.Tensor,
                diffusion_timestep: torch.Tensor) -> torch.Tensor:
        """
        Performs the full, end-to-end denoising pass for the Controller.
        """
        # --- Step 1: Encode all conditioning information ---

        # A. Encode real-time observations
        primary_tokens = self.primary_obs_encoder(observation_history['image_primary'])
        wrist_tokens = self.wrist_obs_encoder(observation_history['image_wrist'])
        
        # Fuse visual tokens (Q=wrist, K=V=primary)
        fused_vision, _ = self.vision_obs_fusion(query=wrist_tokens, key=primary_tokens, value=primary_tokens)
        fused_vision = self.vision_obs_norm(fused_vision + wrist_tokens) # Residual connection
        
        vision_tokens = self.vision_obs_proj(fused_vision)
        
        # Proprioception tokens are processed per-timestep and then flattened
        proprio_flat = observation_history['proprio'].flatten(0, 1) # [B*H_o, D_proprio]
        proprio_tokens_flat = self.proprio_proj(proprio_flat)
        proprio_tokens = proprio_tokens_flat.unflatten(0, (vision_tokens.shape[0], self.obs_horizon))

        # B. Encode the subgoal coordinate
        # Input shape: [B, 2] -> (u, v)
        u_emb = self.subgoal_pos_emb(subgoal_coordinate[:, 0:1]) # [B, 128]
        v_emb = self.subgoal_pos_emb(subgoal_coordinate[:, 1:2]) # [B, 128]
        subgoal_emb = torch.cat([u_emb, v_emb], dim=-1) # [B, 256]
        subgoal_token = self.subgoal_encoder(subgoal_emb).unsqueeze(1) # [B, 1, D_ctrl]
        
        # C. Assemble the full conditioning context for the Transformer
        # Shape: [B, L_vis + H_o + 1, D_ctrl]
        unified_context = torch.cat([vision_tokens, proprio_tokens, subgoal_token], dim=1)
        
        # --- Step 2: Prepare inputs for the Denoising Transformer ---
        
        # Project the noisy actions and add positional embeddings
        action_tokens = self.action_proj(noisy_action_sequence)
        pos_indices = torch.arange(self.action_horizon, device=action_tokens.device)
        action_tokens += self.action_pos_emb(pos_indices)
        
        # Get the diffusion timestep embedding
        time_emb = self.time_mlp(diffusion_timestep)
        
        # --- Step 3: Run the Denoising Process ---
        x = action_tokens
        for block in self.denoiser_blocks:
            # Each block is conditioned on the unified context and the time embedding
            x = block(x=x, unified_context=unified_context, combined_emb=time_emb)
            
        # --- Step 4: Project back to action space ---
        predicted_noise = self.out_proj(x)
        
        return predicted_noise
    

class ViPC(nn.Module):
    """
    The Definitive, SOTA, Unified Hierarchical Policy.

    The ViPC module acts as a high-level orchestrator for the Planner and
    Controller components. It implements the two critical modes of operation:
    
    1.  **Training (`forward`):** Implements a "teacher-forced" forward pass where
        both the Planner and Controller are trained simultaneously using their
        respective ground-truth supervisory signals. This decouples their
        learning objectives for maximum stability.

    2.  **Inference (`plan`, `act`):** Provides separate, clean methods to run
        the Planner to generate a subgoal and to run the Controller to generate
        actions based on that subgoal. This allows an external control loop to
        implement the "Plan -> Act -> Re-plan" dialogue.
    """
    def __init__(self, planner_cfg: Dict[str, Any], controller_cfg: Dict[str, Any]):
        super().__init__()
        
        logger.info("Initializing the ViPC Orchestrator...")
        self.planner = Planner(**planner_cfg)
        self.controller = Controller(**controller_cfg)
        logger.info("ViPC Orchestrator initialized successfully.")

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        The unified training forward pass with teacher forcing.

        Args:
            batch: A dictionary from the ViPCDataset containing supervisory
                   signals for both the Planner and Controller.

        Returns:
            A dictionary containing the predictions from both components,
            ready for the combined loss calculation.
        """
        # --- 1. Planner Forward Pass ---
        # The Planner is trained on its own ground-truth data.
        predicted_heatmap = self.planner(
            current_image=batch['planner_current_image'],
            goal_image=batch['planner_goal_image'],
            task_phase=batch['planner_task_phase']
        )
        
        # --- 2. Controller Forward Pass (with Teacher Forcing) ---
        
        # We need to extract the ground-truth subgoal coordinate to feed
        # to the controller. We use a "soft argmax" for a differentiable
        # and stable way to find the peak of the ground-truth heatmap.
        gt_heatmap = batch['ground_truth_subgoal_heatmap']
        ground_truth_subgoal_coord = self.soft_argmax_2d(gt_heatmap)
        
        # The Controller is trained using the ground-truth subgoal.
        predicted_noise = self.controller(
            observation_history=batch['controller_observation_history'],
            subgoal_coordinate=ground_truth_subgoal_coord, # Teacher-forced
            noisy_action_sequence=batch['noisy_actions'],
            diffusion_timestep=batch['timesteps']
        )
        
        return {
            "predicted_heatmap": predicted_heatmap,
            "predicted_noise": predicted_noise
        }

    @torch.no_grad()
    def plan(self,
             current_image: torch.Tensor,
             goal_image: torch.Tensor,
             task_phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference-only method to run the Planner and extract the subgoal.
        """
        # Set to evaluation mode
        self.eval()
        
        predicted_heatmap = self.planner(current_image, goal_image, task_phase)
        
        # Use the same soft argmax to get the normalized coordinate
        predicted_subgoal_coord = self.soft_argmax_2d(predicted_heatmap)
        
        return predicted_heatmap, predicted_subgoal_coord

    @torch.no_grad()
    def act(self,
            observation_history: Dict[str, torch.Tensor],
            predicted_subgoal_coord: torch.Tensor,
            noise_scheduler,
            num_inference_steps: int) -> torch.Tensor:
        """
        Inference-only method to generate an action sequence from the Controller
        using DDPM/DDIM sampling.
        """
        self.eval()
        
        B = observation_history['proprio'].shape[0]
        device = observation_history['proprio'].device
        
        # 1. Initialize random noise for the action sequence
        latents = torch.randn(
            (B, self.controller.action_horizon, self.controller.action_dim),
            device=device,
            dtype=torch.float32
        )
        
        # 2. Set up the timesteps for the diffusion sampling loop
        noise_scheduler.set_timesteps(num_inference_steps)
        
        # 3. The denoising loop
        for t in noise_scheduler.timesteps:
            # Expand the timestep for batch compatibility
            timesteps = t.expand(B).to(device)
            
            # Predict the noise for the current latents
            predicted_noise = self.controller(
                observation_history=observation_history,
                subgoal_coordinate=predicted_subgoal_coord,
                noisy_action_sequence=latents,
                diffusion_timestep=timesteps
            )
            
            # Use the scheduler to compute the previous noisy sample
            latents = noise_scheduler.step(
                model_output=predicted_noise,
                timestep=t,
                sample=latents
            ).prev_sample
            
        return latents

    @staticmethod
    def soft_argmax_2d(heatmaps: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        SOTA method for extracting a differentiable coordinate from a heatmap.

        This function computes the center of mass of the heatmap after applying a
        softmax, which makes the argmax operation differentiable and more robust
        to noisy or flat-peaked heatmaps than a hard `argmax`.

        Args:
            heatmaps: A tensor of heatmaps with shape [B, 1, H, W].
            temperature: A scaling factor for the softmax. Higher values lead
                         to a sharper, more argmax-like distribution.

        Returns:
            A tensor of normalized coordinates [B, 2] in the range [-1, 1].
        """
        B, C, H, W = heatmaps.shape
        assert C == 1, "soft_argmax_2d expects a single-channel heatmap."

        # Apply temperature scaling and flatten
        heatmaps = heatmaps.reshape(B, -1)
        heatmaps = F.softmax(heatmaps / temperature, dim=1)
        heatmaps = heatmaps.reshape(B, 1, H, W)

        # Create coordinate grids
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, H, device=heatmaps.device),
            torch.linspace(-1, 1, W, device=heatmaps.device),
            indexing='ij'
        )
        
        # Compute the center of mass (expected value) for x and y coordinates
        # The heatmap acts as the probability distribution over the grid.
        expected_x = torch.sum(x_coords * heatmaps, dim=[1, 2, 3])
        expected_y = torch.sum(y_coords * heatmaps, dim=[1, 2, 3])
        
        return torch.stack([expected_x, expected_y], dim=1)