# FILE: models/semantic_planner.py
# (Definitive, SOTA, Production-Grade, Robust Version 7.0 - Perceiver Upgrade)

"""
The Advantage-Weighted Semantic Planner (AWSP Strategist).

This module implements the **Semantic Planner**, a deterministic, goal-conditioned
regression model. It acts as the high-level "Strategist" in the hierarchical
control stack.

Architectural Evolution (v7.0 - Perceiver-Lite):
1.  **Dynamic Query Injection**: Unlike BERT-style models that use a static [CLS]
    token, this model initializes the primary plan query directly from the
    semantic context (Task Phase + Proprioception). This forces the Transformer
    to act as a conditional cross-learner, attending to visual evidence *based on*
    the current agent state.
2.  **Shared Spatial Geometry**: Explicitly injects shared spatial encodings into
    both Start and Goal image tokens to induce geometric correspondence (optical flow)
    learning within the self-attention layers.
3.  **Deep Residual Heads**: The regression heads are upgraded to multi-layer
    residual MLPs (ResMLP) to allow for fine-grained coordinate refinement
    after semantic decoding.
4.  **Manifold Constraints**: Unit-norm quaternion enforcement remains strictly applied.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

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


def _init_weights(module: nn.Module):
    """
    SOTA Weight Initialization Protocol.
    """
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
    The Advantage-Weighted Semantic Planner.
    """

    def __init__(self, cfg: SemanticPlannerConfig):
        super().__init__()
        self.cfg = cfg
        logger.info(f"[SemanticPlanner] Initializing with config: {cfg}")

        # --- 1. Vision Backbone (Frozen) ---
        logger.info(f"Loading Vision Backbone: {cfg.vision_backbone_model}")
        self.vision_backbone = SiglipVisionModel.from_pretrained(cfg.vision_backbone_model)
        self.vision_backbone.requires_grad_(False)
        self.vision_backbone.eval()

        backbone_cfg = self.vision_backbone.config
        if backbone_cfg.hidden_size != cfg.vision_feature_dim:
            raise ValueError(
                f"Config mismatch: Model dim {cfg.vision_feature_dim} != "
                f"Backbone dim {backbone_cfg.hidden_size}"
            )
        
        # Infer patch count for spatial embeddings
        # Default for SigLIP 224 / Patch 16 is 196 patches (14x14)
        self.num_patches = 196
        if hasattr(backbone_cfg, "image_size") and hasattr(backbone_cfg, "patch_size"):
            self.num_patches = (backbone_cfg.image_size // backbone_cfg.patch_size) ** 2

        # --- 2. Context Encoders (The "Query" Generators) ---
        
        # Proprio Encoder: Projects raw physics state -> Model Dim
        self.proprio_encoder = nn.Sequential(
            nn.LayerNorm(cfg.proprio_dim),
            nn.Linear(cfg.proprio_dim, cfg.vision_feature_dim),
            nn.GELU(),
            nn.Linear(cfg.vision_feature_dim, cfg.vision_feature_dim),
        )
        
        # Task Phase Embedding
        self.task_phase_embedding = nn.Embedding(cfg.num_task_phases, cfg.vision_feature_dim)
        
        # --- 3. Visual Inductive Biases ---
        
        # Spatial Positional Embedding: Shared between Start/Goal to align geometry
        self.spatial_pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, cfg.vision_feature_dim))
        
        # Token Type Embeddings: Distinguish Start vs Goal tokens
        # 0: Start Image Patches, 1: Goal Image Patches, 2: Plan Query (Context)
        self.token_type_embeddings = nn.Embedding(3, cfg.vision_feature_dim)

        # --- 4. Fusion Transformer ---
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

        # --- 5. Deep Output Heads (ResMLP) ---
        
        # Pose Head: 3 layers deep with residuals for precise coordinate regression
        self.pose_head = nn.Sequential(
            nn.LayerNorm(cfg.vision_feature_dim),
            nn.Linear(cfg.vision_feature_dim, cfg.vision_feature_dim),
            nn.GELU(),
            ResidualMLPBlock(cfg.vision_feature_dim, cfg.dropout),
            ResidualMLPBlock(cfg.vision_feature_dim, cfg.dropout),
            nn.LayerNorm(cfg.vision_feature_dim),
            nn.Linear(cfg.vision_feature_dim, 7) 
        )

        # Gripper Head: Simpler head is sufficient for binary classification
        self.gripper_head = nn.Sequential(
            nn.LayerNorm(cfg.vision_feature_dim),
            nn.Linear(cfg.vision_feature_dim, cfg.vision_feature_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.vision_feature_dim // 2, 1)
        )

        # --- 6. Initialization ---
        self.apply(self._init_module_weights)
        # Special init for spatial embeddings
        nn.init.trunc_normal_(self.spatial_pos_embedding, std=0.02)
        logger.info("[SemanticPlanner] Initialization Complete.")

    def _init_module_weights(self, module):
        _init_weights(module)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Key Change in v7.0: The "Plan Query" token is not a fixed parameter.
        It is dynamically computed as: Query = Embed(Phase) + Encode(Proprio).
        This fused query is then prepended to the visual tokens.
        """
        initial_image = batch['initial_image']
        goal_image = batch['goal_image']
        task_phase = batch['task_phase']
        current_proprio = batch['current_proprio']

        B, device = initial_image.shape[0], initial_image.device

        # --- 1. Construct the Plan Query (The Context) ---
        # Unlike v6, we don't treat proprio/phase as separate sequence tokens.
        # We fuse them into the [CLS] token directly. This condenses the context.
        # (B, D)
        context_embedding = self.task_phase_embedding(task_phase) + self.proprio_encoder(current_proprio)
        
        # Add Type 2 embedding (Plan Context)
        context_embedding = context_embedding + self.token_type_embeddings(torch.tensor(2, device=device))
        
        # Reshape for sequence: (B, 1, D)
        plan_query_token = context_embedding.unsqueeze(1)

        # --- 2. Visual Encoding ---
        with torch.no_grad():
            start_out = self.vision_backbone(initial_image.float())
            goal_out = self.vision_backbone(goal_image.float())
        
        # (B, N, D)
        start_tokens = start_out.last_hidden_state
        goal_tokens = goal_out.last_hidden_state

        # --- 3. Inject Geometry & Modality Types ---
        # Add Spatial Embeddings (Shared geometry)
        start_tokens = start_tokens + self.spatial_pos_embedding
        goal_tokens = goal_tokens + self.spatial_pos_embedding

        # Add Token Types
        start_tokens = start_tokens + self.token_type_embeddings(torch.tensor(0, device=device))
        goal_tokens = goal_tokens + self.token_type_embeddings(torch.tensor(1, device=device))

        # --- 4. Fusion ---
        # Sequence: [Plan_Query (Proprio+Phase), Start_Patches..., Goal_Patches...]
        # The Plan_Query effectively attends to the visual patches to update its state.
        fused_input = torch.cat([plan_query_token, start_tokens, goal_tokens], dim=1)

        # Transformer Output
        fused_output = self.fusion_transformer(fused_input)

        # Extract updated Plan Vector (Index 0)
        # This vector now holds the "answer": Where to go, based on where I am and what I see.
        plan_vector = fused_output[:, 0, :] 

        # --- 5. Prediction Heads ---
        
        # Pose
        raw_pose = self.pose_head(plan_vector)
        pos_xyz = raw_pose[:, :3]
        quat_raw = raw_pose[:, 3:]
        quat_norm = F.normalize(quat_raw, p=2, dim=-1, eps=1e-8)
        predicted_pose = torch.cat([pos_xyz, quat_norm], dim=1)

        # Gripper
        predicted_gripper_logit = self.gripper_head(plan_vector)

        return {
            'pose': predicted_pose,
            'gripper_logit': predicted_gripper_logit
        }