# FILE: models/semantic_planner.py
# (Definitive, SOTA, Disentangled Architecture v9.0 - Action Chunking)

"""
The Advantage-Weighted Semantic Planner (AWSP Strategist v9.0).

This module implements the **Semantic Planner**, a deterministic, goal-conditioned
trajectory generator.

Architectural Revolution (v9.0 - Action Chunking & Self-Awareness):
1.  **Action Chunking**: The model predicts a trajectory of `k` future steps
    (Pose + Gripper) rather than a single step. This enforces temporal consistency
    and solves the "drift" problem inherent in BC.
2.  **Phase Self-Supervision**: The model PREDICTS the phase. This acts as an
    auxiliary loss to force semantic understanding in the vision encoder.
    CRITICAL: Phase is NOT an input. This removes "Causal Confusion".
3.  **Temporal Vision**: Fuses (t-1), (t), and (Goal) frames to infer velocity
    and progress.
4.  **Triple-Query Mechanism**: Decouples gradients for Trajectory (Pose),
    Actuation (Gripper), and Semantics (Phase).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipVisionModel

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class SemanticPlannerConfig:
    """
    Hyperparameter configuration for the Semantic Planner v9.0.
    """
    # Architecture Dimensions
    proprio_dim: int = 22
    vision_backbone_model: str = "google/siglip-base-patch16-224"
    vision_feature_dim: int = 768
    
    # Transformer Config
    fusion_transformer_layers: int = 6  # Increased depth for temporal reasoning
    fusion_transformer_heads: int = 8
    dim_feedforward_ratio: int = 4
    dropout: float = 0.1
    
    # v9.0 Specifics
    chunk_size: int = 10        # Number of future steps to predict (k)
    num_task_phases: int = 5    # For classification head output
    
    # Note: phase_dropout_prob is removed as Phase is no longer an input


def _init_weights(module: nn.Module):
    """SOTA Weight Initialization Protocol."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
        if module.weight is not None:
            torch.nn.init.ones_(module.weight)


class ResidualMLPBlock(nn.Module):
    """
    A generic Residual Block for MLP heads.
    Improves gradient flow and capacity for regression tasks.
    Structure: x + Dropout(Linear(GELU(LN(x))))
    """
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.ln(x))


class SemanticPlanner(nn.Module):
    """
    The Disentangled Advantage-Weighted Semantic Planner (v9.0).
    
    Inputs:
        - Prev Image (t-1), Curr Image (t), Goal Image (T)
        - Proprioception (t)
    
    Outputs:
        - Pose Trajectory Chunk (k steps)
        - Gripper Trajectory Chunk (k steps)
        - Predicted Phase Class
    """

    def __init__(self, cfg: SemanticPlannerConfig):
        super().__init__()
        self.cfg = cfg

        self.chunk_size = cfg.chunk_size


        logger.info(f"[SemanticPlanner] Initializing v9.0 (Strategist) with config: {cfg}")

        # --- 1. Vision Backbone (Frozen) ---
        # logger.info(f"Loading Vision Backbone: {cfg.vision_backbone_model}")
        # self.vision_backbone = SiglipVisionModel.from_pretrained(cfg.vision_backbone_model)
        # self.vision_backbone.requires_grad_(False)
        # self.vision_backbone.eval()
        # logger.info(f"[SemanticPlanner] Initializing v9.0 (Fine-Tuning) with config: {cfg}")

        # --- 1. Vision Backbone (Partial Unfreeze) ---
        logger.info(f"Loading Vision Backbone: {cfg.vision_backbone_model}")
        self.vision_backbone = SiglipVisionModel.from_pretrained(cfg.vision_backbone_model)
        
        # A. Freeze EVERYTHING first
        for param in self.vision_backbone.parameters():
            param.requires_grad = False
            
        # B. Unfreeze the Last Encoder Layer
        # This allows the model to learn "Geometry" without forgetting "Objects"
        last_layers = self.vision_backbone.vision_model.encoder.layers[-3:]
        for layer in last_layers:
            for param in layer.parameters():
                param.requires_grad = True
            
        # C. Unfreeze the Final LayerNorm (Crucial for feature scaling)
        if hasattr(self.vision_backbone.vision_model, 'post_layernorm'):
             for param in self.vision_backbone.vision_model.post_layernorm.parameters():
                param.requires_grad = True

        # Ensure the model stays in train mode (for the unfrozen parts)
        # while we will manually handle BatchNorm freezing if needed (SigLIP usually uses LayerNorm, which is fine)
        
        logger.info("Backbone Status: Bottom layers FROZEN. Top layer UNFROZEN for geometric adaptation.")



        backbone_cfg = self.vision_backbone.config
        
        # Infer patch count
        self.num_patches = 196
        if hasattr(backbone_cfg, "image_size") and hasattr(backbone_cfg, "patch_size"):
            self.num_patches = (backbone_cfg.image_size // backbone_cfg.patch_size) ** 2
        
        if backbone_cfg.hidden_size != cfg.vision_feature_dim:
             raise ValueError(f"Config mismatch: Model dim {cfg.vision_feature_dim} != Backbone dim {backbone_cfg.hidden_size}")

        # --- 2. Context Encoders ---
        
        # Proprio Encoder: Projects raw physics state -> Model Dim
        self.proprio_encoder = nn.Sequential(
            nn.LayerNorm(cfg.proprio_dim),
            nn.Linear(cfg.proprio_dim, cfg.vision_feature_dim),
            nn.GELU(),
            nn.Linear(cfg.vision_feature_dim, cfg.vision_feature_dim),
        )
        
        # NOTE: task_phase_embedding removed. Phase is now an Output, not Input.
        
        # --- 3. Embeddings ---
        
        # Spatial Positional Embedding: Shared across all images
        self.spatial_pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, cfg.vision_feature_dim))
        
        # Token Types (Temporal/Modal Distinction)
        # 0:PrevImg, 1:CurrImg, 2:GoalImg, 3:Proprio, 4:TrajQ, 5:GripQ, 6:PhaseQ
        self.token_type_embeddings = nn.Embedding(7, cfg.vision_feature_dim)

        # --- 4. Triple-Query Mechanism (Learned Latents) ---
        self.traj_query_token = nn.Parameter(torch.randn(1, 1, cfg.vision_feature_dim))
        self.grip_query_token = nn.Parameter(torch.randn(1, 1, cfg.vision_feature_dim))
        self.phase_query_token = nn.Parameter(torch.randn(1, 1, cfg.vision_feature_dim))

        # --- 5. Fusion Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.vision_feature_dim,
            nhead=cfg.fusion_transformer_heads,
            dim_feedforward=cfg.vision_feature_dim * cfg.dim_feedforward_ratio,
            dropout=cfg.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=cfg.fusion_transformer_layers
        )

        # --- 6. Output Heads (Action Chunking) ---
        
        # A. Trajectory Head (Pose)
        # Output: chunk_size * 7 (3 Pos + 4 Quat)
        # A. Trajectory Head (Pose)
        # Output: chunk_size * 7 (3 Pos + 4 Quat)
        self.traj_head = nn.Sequential(
            nn.LayerNorm(cfg.vision_feature_dim),
            nn.Linear(cfg.vision_feature_dim, cfg.vision_feature_dim),
            nn.GELU(),
            # Existing blocks
            ResidualMLPBlock(cfg.vision_feature_dim, cfg.dropout),
            ResidualMLPBlock(cfg.vision_feature_dim, cfg.dropout),
            # [NEW] Added extra capacity for fine-grained coordinate geometry
            ResidualMLPBlock(cfg.vision_feature_dim, cfg.dropout),
            ResidualMLPBlock(cfg.vision_feature_dim, cfg.dropout),
            # Output projection
            nn.LayerNorm(cfg.vision_feature_dim),
            nn.Linear(cfg.vision_feature_dim, self.chunk_size * 7) 
        )

        # B. Gripper Head
        # Deepened to match pose capacity
        self.gripper_head = nn.Sequential(
            nn.LayerNorm(cfg.vision_feature_dim),
            nn.Linear(cfg.vision_feature_dim, cfg.vision_feature_dim // 2),
            nn.GELU(),
            ResidualMLPBlock(cfg.vision_feature_dim // 2, cfg.dropout),
            # [NEW] Added extra capacity
            ResidualMLPBlock(cfg.vision_feature_dim // 2, cfg.dropout),
            nn.Linear(cfg.vision_feature_dim // 2, self.chunk_size * 1)
        )
        
        
        # C. Phase Classification Head (Auxiliary)
        # Output: num_task_phases (Logits)
        self.phase_head = nn.Sequential(
            nn.LayerNorm(cfg.vision_feature_dim),
            ResidualMLPBlock(cfg.vision_feature_dim, cfg.dropout),
            nn.Linear(cfg.vision_feature_dim, cfg.num_task_phases)
        )

        # --- 7. Initialization ---
        logger.info("Initializing custom modules (Skipping Vision Backbone)...")
        for name, module in self.named_children():
            if "vision_backbone" in name: continue
            module.apply(_init_weights)

        # Initialize orphans
        nn.init.trunc_normal_(self.spatial_pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.traj_query_token, std=0.02)
        nn.init.trunc_normal_(self.grip_query_token, std=0.02)
        nn.init.trunc_normal_(self.phase_query_token, std=0.02)
        
        logger.info("[SemanticPlanner v9.0] Initialization Complete.")

    def train(self, mode: bool = True):
        """
        Hybrid Train Mode (v9.0 - Fine-Tuning):
        - Frozen layers stay in Eval mode (to freeze stats/dropout).
        - The last 3 unfrozen layers + PostLN are set to Train mode.
        """
        super().train(mode)
        if mode:
            # 1. Force entire backbone to eval first (default safety)
            self.vision_backbone.eval()
            
            # 2. Set the specifically unfrozen layers back to train mode
            last_layers = self.vision_backbone.vision_model.encoder.layers[-3:]
            for layer in last_layers:
                layer.train()
            
            if hasattr(self.vision_backbone.vision_model, 'post_layernorm'):
                self.vision_backbone.vision_model.post_layernorm.train()
        return self

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass generating action chunks and phase predictions.
        """
        # Unpack Inputs (Strict v9.0 keys)
        prev_image = batch['prev_image']
        curr_image = batch['curr_image']
        goal_image = batch['goal_image']
        curr_proprio = batch['curr_proprio']
        
        B, device = curr_image.shape[0], curr_image.device

        # --- 1. Vision Encoding (Temporal Batching) ---
        # To save compute, we stack images along batch dim: (3*B, C, H, W)
        stacked_images = torch.cat([prev_image, curr_image, goal_image], dim=0)
        
        # [FIX] Do NOT use no_grad() here. 
        # The flags set in __init__ (requires_grad=True for top layer) handle the selective training.
        # AMP will handle the mixed precision automatically.
        backbone_out = self.vision_backbone(stacked_images.float())
        
        visual_tokens = backbone_out.last_hidden_state

        
        seq_len = visual_tokens.shape[1]
        
        # Robust slicing (in case of CLS/Registers)
        if seq_len != self.num_patches:
            if seq_len > self.num_patches:
                visual_tokens = visual_tokens[:, -self.num_patches:, :]
            else:
                raise ValueError(f"Backbone output {seq_len} < expected {self.num_patches}")

        # Apply Spatial Positional Embedding (Shared)
        visual_tokens = visual_tokens + self.spatial_pos_embedding

        # Split back into (B, N, D)
        prev_tokens, curr_tokens, goal_tokens = torch.chunk(visual_tokens, 3, dim=0)

        # Apply Token Type Embeddings
        prev_tokens = prev_tokens + self.token_type_embeddings(torch.tensor(0, device=device))
        curr_tokens = curr_tokens + self.token_type_embeddings(torch.tensor(1, device=device))
        goal_tokens = goal_tokens + self.token_type_embeddings(torch.tensor(2, device=device))

        # --- 2. Proprio Encoding ---
        # Normalize proprio to [B, 1, D] token
        proprio_embed = self.proprio_encoder(curr_proprio).unsqueeze(1) 
        proprio_embed = proprio_embed + self.token_type_embeddings(torch.tensor(3, device=device))

        # --- 3. Query Initialization ---
        traj_q = self.traj_query_token.expand(B, -1, -1) + self.token_type_embeddings(torch.tensor(4, device=device))
        grip_q = self.grip_query_token.expand(B, -1, -1) + self.token_type_embeddings(torch.tensor(5, device=device))
        phase_q = self.phase_query_token.expand(B, -1, -1) + self.token_type_embeddings(torch.tensor(6, device=device))

        # --- 4. Fusion ---
        # Sequence: [TrajQ, GripQ, PhaseQ, Proprio, Prev, Curr, Goal]
        # This ordering allows queries to attend to all context
        fused_input = torch.cat([
            traj_q, grip_q, phase_q, 
            proprio_embed, 
            prev_tokens, curr_tokens, goal_tokens
        ], dim=1)

        # Transformer Pass
        fused_output = self.fusion_transformer(fused_input)

        # Extract Query Outputs (First 3 tokens)
        z_traj = fused_output[:, 0, :]
        z_grip = fused_output[:, 1, :]
        z_phase = fused_output[:, 2, :]

        # --- 5. Decode Heads (Action Chunking) ---
        
        # A. Trajectory Chunk
        raw_traj = self.traj_head(z_traj) # (B, K*7)
        # Reshape to (B, K, 7)
        pred_traj = raw_traj.view(B, self.cfg.chunk_size, 7)
        
        # Normalize Quaternions within the chunk
        pos_xyz = pred_traj[..., :3]
        quat_raw = pred_traj[..., 3:]
        quat_norm = F.normalize(quat_raw, p=2, dim=-1, eps=1e-6)
        
        final_traj_chunk = torch.cat([pos_xyz, quat_norm], dim=-1)

        # B. Gripper Chunk
        raw_grip = self.gripper_head(z_grip) # (B, K*1)
        final_grip_chunk = raw_grip.view(B, self.cfg.chunk_size, 1)

        # C. Phase Classification
        phase_logits = self.phase_head(z_phase) # (B, Num_Phases)

        return {
            'pose_chunk': final_traj_chunk,      # (B, K, 7)
            'gripper_chunk': final_grip_chunk,   # (B, K, 1)
            'phase_logits': phase_logits         # (B, N_Phases)
        }