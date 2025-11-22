"""
Residual History Adapter (The "Sidecar").
POST-TRAINING Module.

Purpose:
1. Loads a PRE-TRAINED SemanticPlanner (v8.0).
2. FREEZES it (no gradient updates to the base).
3. Adds a lightweight Transformer to process History (Proprio + Actions).
4. Outputs a DELTA correction to the base model's prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.semantic_planner import SemanticPlanner

class ResidualHistoryAdapter(nn.Module):
    def __init__(self, 
                 base_planner: SemanticPlanner, 
                 history_len: int = 10, 
                 embed_dim: int = 256):
        super().__init__()
        
        # 1. The Base Model (FROZEN)
        # We store it, set it to eval mode, and turn off gradients.
        self.base_planner = base_planner
        for param in self.base_planner.parameters():
            param.requires_grad = False
        self.base_planner.eval()

        # 2. The History Encoder
        # Input: Proprio (22) + Prev Action (8) = 30 dim
        input_dim = 22 + 8 
        self.adapter_embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # 3. Temporal Transformer (The "Memory")
        # Very lightweight (2 layers) because it only needs to learn
        # corrections, not full spatial reasoning.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=4, 
            dim_feedforward=512, 
            batch_first=True, 
            norm_first=True,
            dropout=0.1
        )
        self.adapter_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Learnable Positional Embedding for the history window
        self.pos_embed = nn.Parameter(torch.randn(1, history_len, embed_dim) * 0.02)

        # 4. Correction Head
        # Outputs: Delta Pose (7) + Delta Gripper Logit (1)
        self.correction_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 8) 
        )
        
        # CRITICAL: Zero Initialization
        # We start with weights/bias at zero.
        # This ensures that at Step 0 of post-training, the model acts 
        # EXACTLY like the base model. It learns to diverge only as needed.
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)

    def train(self, mode: bool = True):
        """
        Overridden to ensure the Base Planner NEVER enters train mode
        (which would mess up BatchNorm/Dropout statistics).
        """
        super().train(mode)
        self.base_planner.eval()
        return self

    def forward(self, batch):
        # 1. Get Base Prediction (No Gradients computed for this part)
        with torch.no_grad():
            base_out = self.base_planner(batch)
            base_pose = base_out['pose']
            base_grip = base_out['gripper_logit']

        # 2. Prepare History Data
        # batch['proprio_hist']: (B, T, 22)
        # batch['action_hist']:  (B, T, 8)
        proprio_hist = batch['proprio_hist']
        
        # Handle missing action history (fallback for initial inference)
        action_hist = batch.get('action_hist')
        if action_hist is None:
             action_hist = torch.zeros(
                 proprio_hist.size(0), proprio_hist.size(1), 8, 
                 device=proprio_hist.device
             )

        # Combine inputs
        history_seq = torch.cat([proprio_hist, action_hist], dim=-1) # (B, T, 30)
        
        # 3. Encode History
        x = self.adapter_embedding(history_seq)
        
        # Add Positional Embeddings (Handle variable lengths if needed)
        seq_len = x.size(1)
        x = x + self.pos_embed[:, :seq_len, :]
        
        # Causal Mask (Standard triangular mask so t can't see t+1)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), 
            diagonal=1
        )
        
        # Pass through small Transformer
        x = self.adapter_transformer(x, mask=mask)
        
        # Take the embedding of the LAST timestep (current time)
        current_context = x[:, -1, :]
        
        # 4. Predict Correction (Delta)
        correction = self.correction_head(current_context)
        delta_pose = correction[:, :7]
        delta_grip = correction[:, 7:]

        # 5. Apply Correction
        # Final = Base + Delta
        final_pose = base_pose + delta_pose
        final_grip = base_grip + delta_grip

        # Re-normalize quaternion to be valid
        final_pose_xyz = final_pose[:, :3]
        final_pose_quat = F.normalize(final_pose[:, 3:], p=2, dim=-1)

        return {
            'pose': torch.cat([final_pose_xyz, final_pose_quat], dim=1),
            'gripper_logit': final_grip,
            'delta_magnitude': correction.abs().mean() # Useful for logging
        }