# FILE: models/semantic_planner.py
# (Definitive, SOTA, Disentangled Architecture v8.0)

"""
The Advantage-Weighted Semantic Planner (AWSP Strategist).

This module implements the **Semantic Planner**, a deterministic, goal-conditioned
regression model.

Architectural Revolution (v8.0 - Disentangled Queries):
1.  **Dual-Query Mechanism**: Instead of a single [CLS] token, we initialize
    TWO distinct query tokens: `Pose_Query` and `Grip_Query`.
2.  **Gradient Decoupling**: By forcing the transformer to output two separate
    latent vectors, we ensure that the heavy gradients from the Pose loss do not
    wash out the delicate gradients from the Gripper loss.
3.  **Specialized Attention**: The `Grip_Query` is free to learn to attend specifically
    to the gripper fingers/object contact points, while `Pose_Query` attends to
    the global object geometry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipVisionModel

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class SemanticPlannerConfig:
    """
    Hyperparameter configuration for the Semantic Planner.
    """
    proprio_dim: int = 22
    vision_backbone_model: str = "google/siglip-base-patch16-224"
    vision_feature_dim: int = 768
    fusion_transformer_layers: int = 4
    fusion_transformer_heads: int = 8
    dim_feedforward_ratio: int = 4
    num_task_phases: int = 5
    dropout: float = 0.1
    phase_dropout_prob: float = 0.0


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
    The Disentangled Advantage-Weighted Semantic Planner.
    
    Uses a Dual-Query Transformer architecture with safe initialization protocols
    to preserve pre-trained vision backbone weights.
    """

    def __init__(self, cfg: SemanticPlannerConfig):
        super().__init__()
        self.cfg = cfg
        logger.info(f"[SemanticPlanner] Initializing v8.0 (Disentangled) with config: {cfg}")

        # --- 1. Vision Backbone (Frozen) ---
        logger.info(f"Loading Vision Backbone: {cfg.vision_backbone_model}")
        self.vision_backbone = SiglipVisionModel.from_pretrained(cfg.vision_backbone_model)
        self.vision_backbone.requires_grad_(False)
        self.vision_backbone.eval()

        backbone_cfg = self.vision_backbone.config
        
        # Infer patch count (safely default to 196 if config missing)
        self.num_patches = 196
        if hasattr(backbone_cfg, "image_size") and hasattr(backbone_cfg, "patch_size"):
            self.num_patches = (backbone_cfg.image_size // backbone_cfg.patch_size) ** 2
        
        # Sanity check dimension
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
        
        # Task Phase Embedding
        self.task_phase_embedding = nn.Embedding(cfg.num_task_phases, cfg.vision_feature_dim)
        
        # --- 3. Embeddings (Updated for Dual Queries) ---
        
        # Spatial Positional Embedding: Shared between Start/Goal
        self.spatial_pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, cfg.vision_feature_dim))
        
        # Token Types: 0:Start, 1:Goal, 2:Context, 4:PoseQ, 5:GripQ
        self.token_type_embeddings = nn.Embedding(6, cfg.vision_feature_dim)

        # [SOTA CHANGE] Split the Plan Token into Two Learnable Queries
        self.pose_query_token = nn.Parameter(torch.randn(1, 1, cfg.vision_feature_dim))
        self.grip_query_token = nn.Parameter(torch.randn(1, 1, cfg.vision_feature_dim))

        # --- 4. Fusion Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.vision_feature_dim,
            nhead=cfg.fusion_transformer_heads,
            dim_feedforward=cfg.vision_feature_dim * cfg.dim_feedforward_ratio,
            dropout=cfg.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True # Pre-LN is critical for stability
        )
        self.fusion_transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=cfg.fusion_transformer_layers
        )

        # --- 5. Deep Output Heads (Decoupled) ---
        
        # Pose Head: Regresses from the Pose Latent Vector
        self.pose_head = nn.Sequential(
            nn.LayerNorm(cfg.vision_feature_dim),
            nn.Linear(cfg.vision_feature_dim, cfg.vision_feature_dim),
            nn.GELU(),
            ResidualMLPBlock(cfg.vision_feature_dim, cfg.dropout),
            ResidualMLPBlock(cfg.vision_feature_dim, cfg.dropout),
            nn.LayerNorm(cfg.vision_feature_dim),
            nn.Linear(cfg.vision_feature_dim, 7) 
        )

        # Gripper Head: Regresses from the Grip Latent Vector
        self.gripper_head = nn.Sequential(
            nn.LayerNorm(cfg.vision_feature_dim),
            nn.Linear(cfg.vision_feature_dim, cfg.vision_feature_dim // 2),
            nn.GELU(),
            ResidualMLPBlock(cfg.vision_feature_dim // 2, cfg.dropout), 
            nn.Linear(cfg.vision_feature_dim // 2, 1)
        )

        # --- 6. Initialization (CRITICAL FIX) ---
        # We must NOT use self.apply() globally because it would re-init 
        # the Vision Backbone (wiping pre-trained weights).
        
        logger.info("Initializing custom modules (Skipping Vision Backbone)...")
        
        # Iterate over direct children and skip the backbone
        for name, module in self.named_children():
            if "vision_backbone" in name:
                logger.info(f"Skipped initialization for: {name}")
                continue
            module.apply(_init_weights)

        # Explicitly initialize Top-Level Parameters (Orphans)
        nn.init.trunc_normal_(self.spatial_pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.pose_query_token, std=0.02)
        nn.init.trunc_normal_(self.grip_query_token, std=0.02)
        
        logger.info("[SemanticPlanner] Initialization Complete.")

    def train(self, mode: bool = True):
        """
        [SOTA FIX] Override train mode to ensure Backbone stays FROZEN.
        PyTorch Lightning calls model.train() every epoch, which recursively
        activates Dropout in the backbone even if requires_grad=False.
        This override forces the backbone to stay in eval mode.
        """
        super().train(mode)
        if mode:
            # Revert backbone to eval to disable dropout/batchnorm updates
            self.vision_backbone.eval()
        return self

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass with robust shape handling.
        """
        initial_image = batch['initial_image']
        goal_image = batch['goal_image']
        task_phase = batch['task_phase']
        current_proprio = batch['current_proprio']

        B, device = initial_image.shape[0], initial_image.device

        # --- 1. Construct Context (Proprio + Phase) ---
        
        # Get Phase Embeddings
        phase_embed = self.task_phase_embedding(task_phase)

        # [IMPLEMENTATION] Phase Dropout
        if self.training and self.cfg.phase_dropout_prob > 0.0:
            keep_prob = 1.0 - self.cfg.phase_dropout_prob
            mask = torch.bernoulli(torch.full((B, 1), keep_prob, device=device))
            phase_embed = phase_embed * mask

        proprio_embed = self.proprio_encoder(current_proprio)

        # Fuse Phase + Proprio
        context_embedding = phase_embed + proprio_embed
        context_embedding = context_embedding + self.token_type_embeddings(torch.tensor(2, device=device))
        
        # --- 2. Create Disentangled Queries ---
        
        # Expand learnable tokens: (1, 1, D) -> (B, 1, D)
        pose_q = self.pose_query_token.expand(B, -1, -1)
        grip_q = self.grip_query_token.expand(B, -1, -1)
        
        # Fuse Context into Queries
        context_expanded = context_embedding.unsqueeze(1)
        pose_q = pose_q + context_expanded
        grip_q = grip_q + context_expanded
        
        # Add Distinct Token Types
        pose_q = pose_q + self.token_type_embeddings(torch.tensor(4, device=device))
        grip_q = grip_q + self.token_type_embeddings(torch.tensor(5, device=device))

        # --- 3. Visual Encoding ---
        with torch.no_grad():
            start_out = self.vision_backbone(initial_image.float())
            goal_out = self.vision_backbone(goal_image.float())
        
        start_tokens = start_out.last_hidden_state
        goal_tokens = goal_out.last_hidden_state

        # --- [CRITICAL FIX] Dynamic Shape Handling ---
        # Some ViTs return (197) tokens (CLS + Patches), others (196).
        # We must align strictly with self.spatial_pos_embedding (196).
        seq_len = start_tokens.shape[1]
        
        if seq_len != self.num_patches:
            if seq_len > self.num_patches:
                # If backbone adds CLS/Register tokens, take the LAST N tokens 
                # (Standard ViT: CLS is index 0, patches [1:])
                # Taking negative slice is safe regardless of where extra tokens are, 
                # assuming patches are the majority block.
                start_tokens = start_tokens[:, -self.num_patches:, :]
                goal_tokens = goal_tokens[:, -self.num_patches:, :]
            else:
                raise ValueError(
                    f"Vision Backbone output ({seq_len}) < Expected Patches ({self.num_patches}). "
                    "Check image size/patch size config."
                )

        # --- 4. Inject Geometry & Modality Types ---
        # Add Spatial Embeddings (Shared geometry)
        start_tokens = start_tokens + self.spatial_pos_embedding
        goal_tokens = goal_tokens + self.spatial_pos_embedding

        # Add Token Types (0 & 1)
        start_tokens = start_tokens + self.token_type_embeddings(torch.tensor(0, device=device))
        goal_tokens = goal_tokens + self.token_type_embeddings(torch.tensor(1, device=device))

        # --- 5. Fusion ---
        # Sequence: [Pose_Query, Grip_Query, Start_Patches..., Goal_Patches...]
        fused_input = torch.cat([pose_q, grip_q, start_tokens, goal_tokens], dim=1)

        # Transformer Output
        fused_output = self.fusion_transformer(fused_input)

        # --- 6. Decoupled Latent Extraction ---
        # Index 0 = Pose Latent, Index 1 = Grip Latent
        pose_vector = fused_output[:, 0, :] 
        grip_vector = fused_output[:, 1, :] 

        # --- 7. Prediction Heads ---
        
        # A. Pose Regression
        raw_pose = self.pose_head(pose_vector)
        pos_xyz = raw_pose[:, :3]
        
        quat_raw = raw_pose[:, 3:]
        quat_norm = F.normalize(quat_raw, p=2, dim=-1, eps=1e-6)
        
        predicted_pose = torch.cat([pos_xyz, quat_norm], dim=1)

        # B. Gripper Classification
        predicted_gripper_logit = self.gripper_head(grip_vector)


        return {
            'pose': predicted_pose,
            'gripper_logit': predicted_gripper_logit
        }