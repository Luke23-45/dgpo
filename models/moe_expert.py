# FILE: models/moe_expert.py
# (Definitive, SOTA, Production-Grade Implementation)

r"""
Phase-Locked Mixture-of-Experts: Expert Network Module.

This module defines the **PhaseExpert** class and the **ExpertArray** container.
Each PhaseExpert is a specialized policy head trained on a specific task phase
(e.g., Approach, Grasp, Lift, Place, Retract).

Architecture Design Principles:
1.  **Cloning Initialization**: Experts are initialized by cloning the pre-trained
    trajectory and gripper heads from the Monolithic SemanticPlanner. This avoids
    the cold-start problem and ensures experts begin with a strong generalist baseline.
2.  **FiLM Conditioning (Optional)**: Experts can receive a context vector
    ($\alpha_t$) from the Router for parametric skill modulation (e.g., adjusting
    grip force based on object weight).
3.  **Residual MLPs**: Uses pre-activation LayerNorm and GELU for stable gradients.

Usage:
    # Initialize from pre-trained planner
    expert_array = ExpertArray.from_planner(planner_model, num_experts=5)
    
    # Forward pass (during training)
    z_traj, z_grip = router.get_embeddings(batch)
    phase_id = batch['gt_phase_label']
    pose_chunk, grip_chunk = expert_array(z_traj, z_grip, phase_id)
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Setup logger
logger = logging.getLogger(__name__)


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

@dataclass
class ExpertConfig:
    """Configuration for a single PhaseExpert."""
    input_dim: int = 768          # Dimension of visual embedding from Router
    chunk_size: int = 10          # Action prediction horizon
    dropout: float = 0.1          # Dropout probability
    num_residual_blocks: int = 4  # Number of ResidualMLPBlocks in trajectory head
    context_dim: int = 0          # Dimension of context vector (0 = no FiLM)


# =============================================================================
# 2. BUILDING BLOCKS
# =============================================================================

def _init_weights(module: nn.Module):
    """SOTA Weight Initialization Protocol (Matches SemanticPlanner)."""
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

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(self.ln(x))


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Layer.
    
    Given a context vector, produces scale (gamma) and shift (beta) parameters
    to modulate the main feature vector. This enables parametric skill adaptation.
    
    Reference: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer"
    """
    def __init__(self, context_dim: int, feature_dim: int):
        super().__init__()
        self.context_to_gamma = nn.Linear(context_dim, feature_dim)
        self.context_to_beta = nn.Linear(context_dim, feature_dim)
        
        # Initialize to identity transform (gamma=1, beta=0)
        nn.init.zeros_(self.context_to_gamma.weight)
        nn.init.ones_(self.context_to_gamma.bias)
        nn.init.zeros_(self.context_to_beta.weight)
        nn.init.zeros_(self.context_to_beta.bias)
    
    def forward(self, features: Tensor, context: Tensor) -> Tensor:
        """
        Args:
            features: (B, D) main feature vector
            context: (B, C) context vector from Router
        Returns:
            Modulated features: gamma * features + beta
        """
        gamma = self.context_to_gamma(context)  # (B, D)
        beta = self.context_to_beta(context)    # (B, D)
        return gamma * features + beta


# =============================================================================
# 3. PHASE EXPERT
# =============================================================================

class PhaseExpert(nn.Module):
    """
    A single phase-specialized expert policy head.
    
    Takes the visual embedding from the frozen Router and predicts action chunks.
    Optionally modulated by a context vector (FiLM) for parametric adaptation.
    
    Args:
        cfg: ExpertConfig with architecture hyperparameters.
        phase_id: Integer ID of this expert's designated phase (for logging).
    """
    
    def __init__(self, cfg: ExpertConfig, phase_id: int = -1):
        super().__init__()
        self.cfg = cfg
        self.phase_id = phase_id
        self.chunk_size = cfg.chunk_size
        
        logger.info(f"[PhaseExpert {phase_id}] Initializing with dim={cfg.input_dim}, "
                    f"chunk={cfg.chunk_size}, context_dim={cfg.context_dim}")
        
        # --- A. Optional FiLM Conditioning ---
        self.use_film = cfg.context_dim > 0
        if self.use_film:
            self.film_layer = FiLMLayer(cfg.context_dim, cfg.input_dim)
            logger.info(f"[PhaseExpert {phase_id}] FiLM conditioning ENABLED")
        
        # --- B. Trajectory Head (Pose Prediction) ---
        # Matches the architecture of SemanticPlanner.traj_head
        traj_layers = [
            nn.LayerNorm(cfg.input_dim),
            nn.Linear(cfg.input_dim, cfg.input_dim),
            nn.GELU(),
        ]
        for _ in range(cfg.num_residual_blocks):
            traj_layers.append(ResidualMLPBlock(cfg.input_dim, cfg.dropout))
        traj_layers.extend([
            nn.LayerNorm(cfg.input_dim),
            nn.Linear(cfg.input_dim, cfg.chunk_size * 7)  # 3 Pos + 4 Quat
        ])
        self.traj_head = nn.Sequential(*traj_layers)
        
        # --- C. Gripper Head ---
        # Matches the architecture of SemanticPlanner.gripper_head
        self.gripper_head = nn.Sequential(
            nn.LayerNorm(cfg.input_dim),
            nn.Linear(cfg.input_dim, cfg.input_dim // 2),
            nn.GELU(),
            ResidualMLPBlock(cfg.input_dim // 2, cfg.dropout),
            ResidualMLPBlock(cfg.input_dim // 2, cfg.dropout),
            nn.Linear(cfg.input_dim // 2, cfg.chunk_size * 1)
        )
        
        # --- D. Weight Initialization ---
        self.apply(_init_weights)
        
        logger.info(f"[PhaseExpert {phase_id}] Initialization Complete.")
    
    def forward(
        self,
        z_traj: Tensor,
        z_grip: Optional[Tensor] = None,
        context: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass predicting action chunks.
        
        [FIX v2.0] FiLM modulation is now applied AFTER the first LayerNorm
        to prevent cancellation. LayerNorm normalizes (subtracts mean, divides
        by std), which would erase any scaling/shifting applied before it.
        
        Args:
            z_traj: (B, D) visual embedding for trajectory prediction.
            z_grip: (B, D) visual embedding for gripper prediction.
                    If None, uses z_traj.
            context: (B, C) optional context vector for FiLM modulation.
        
        Returns:
            pose_chunk: (B, K, 7) predicted trajectory.
            grip_chunk: (B, K, 1) predicted gripper states.
        """
        B = z_traj.shape[0]
        
        # Use same embedding for both heads if z_grip not provided
        # Use same embedding for both heads if z_grip not provided
        if z_grip is None:
            z_grip = z_traj
        
        # --- Trajectory Prediction ---
        # [FIX v3.0] Robust LayerNorm finding to prevent FiLM cancellation
        # We need to apply LN(x) -> FiLM(x) -> MLP(x)
        # But traj_head is a Sequential. We must find the first LN layer index.
        
        # Dynamic Search for First LayerNorm
        ln_idx = -1
        for i, layer in enumerate(self.traj_head):
            if isinstance(layer, nn.LayerNorm):
                ln_idx = i
                break
        
        if ln_idx == -1:
            # Fallback: No LN found (unlikely for SOTA), just apply FiLM first
            z_traj_norm = z_traj
            start_layer = 0
            logger.warning(f"[PhaseExpert {self.phase_id}] No LayerNorm found in traj_head! FiLM applied directly.")
        else:
            # Apply up to and including the first LayerNorm
            x = z_traj
            for i in range(ln_idx + 1):
                x = self.traj_head[i](x)
            z_traj_norm = x
            start_layer = ln_idx + 1
        
        # Apply FiLM modulation to normalized features
        if self.use_film and context is not None:
            z_traj_norm = self.film_layer(z_traj_norm, context)
        
        # Pass through rest of trajectory head
        raw_traj = z_traj_norm
        for i in range(start_layer, len(self.traj_head)):
            raw_traj = self.traj_head[i](raw_traj)
        
        pred_traj = raw_traj.view(B, self.chunk_size, 7)
        
        # Normalize Quaternions
        pos_xyz = pred_traj[..., :3]
        quat_raw = pred_traj[..., 3:]
        quat_norm = F.normalize(quat_raw, p=2, dim=-1, eps=1e-6)
        pose_chunk = torch.cat([pos_xyz, quat_norm], dim=-1)
        
        # --- Gripper Prediction ---
        # Same robust logic for gripper head
        ln_idx_g = -1
        for i, layer in enumerate(self.gripper_head):
            if isinstance(layer, nn.LayerNorm):
                ln_idx_g = i
                break
                
        if ln_idx_g == -1:
            z_grip_norm = z_grip
            start_layer_g = 0
        else:
            x = z_grip
            for i in range(ln_idx_g + 1):
                x = self.gripper_head[i](x)
            z_grip_norm = x
            start_layer_g = ln_idx_g + 1

        if self.use_film and context is not None:
            z_grip_norm = self.film_layer(z_grip_norm, context)
        
        raw_grip = z_grip_norm
        for i in range(start_layer_g, len(self.gripper_head)):
            raw_grip = self.gripper_head[i](raw_grip)
        
        grip_chunk = raw_grip.view(B, self.chunk_size, 1)
        
        return pose_chunk, grip_chunk
    
    @torch.no_grad()
    def forward_with_uncertainty(
        self,
        z_traj: Tensor,
        z_grip: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        n_samples: int = 5
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        [ENHANCEMENT A] Monte Carlo Dropout for Uncertainty Estimation.
        
        Runs forward pass multiple times with dropout enabled to estimate
        predictive uncertainty. High uncertainty indicates out-of-distribution
        inputs where the model is not confident.
        
        Use cases:
        - Safety: if pose_std > threshold, ask human for help
        - Debugging: find phases where model struggles
        
        Args:
            z_traj: (B, D) visual embedding for trajectory prediction.
            z_grip: (B, D) visual embedding for gripper prediction.
            context: (B, C) optional context vector.
            n_samples: Number of forward passes for MC estimate.
        
        Returns:
            pose_mean: (B, K, 7) mean prediction.
            pose_std: (B, K, 7) standard deviation (uncertainty).
            grip_mean: (B, K, 1) mean gripper prediction.
            grip_std: (B, K, 1) gripper uncertainty.
        """
        # Store original training state
        was_training = self.training
        
        # Enable dropout by setting to train mode
        self.train()
        
        pose_samples = []
        grip_samples = []
        
        for _ in range(n_samples):
            pose, grip = self.forward(z_traj, z_grip, context)
            pose_samples.append(pose)
            grip_samples.append(grip)
        
        # Restore original training state
        self.train(was_training)
        
        # Stack and compute statistics
        pose_stack = torch.stack(pose_samples, dim=0)  # (N, B, K, 7)
        grip_stack = torch.stack(grip_samples, dim=0)  # (N, B, K, 1)
        
        pose_mean = pose_stack.mean(dim=0)
        pose_std = pose_stack.std(dim=0)
        grip_mean = grip_stack.mean(dim=0)
        grip_std = grip_stack.std(dim=0)
        
        return pose_mean, pose_std, grip_mean, grip_std
    
    @classmethod
    def from_planner_heads(
        cls,
        traj_head: nn.Module,
        gripper_head: nn.Module,
        cfg: ExpertConfig,
        phase_id: int = -1
    ) -> "PhaseExpert":
        """
        Factory method to create an Expert by cloning heads from a pre-trained planner.
        
        This is the core of the "Bootstrapping" strategy: instead of random init,
        each expert starts with the full generalist capability.
        
        [FIX v2.0] Now updates cfg to reflect actual cloned architecture.
        
        Args:
            traj_head: nn.Sequential from SemanticPlanner.traj_head
            gripper_head: nn.Sequential from SemanticPlanner.gripper_head
            cfg: ExpertConfig (will be updated to reflect cloned structure)
            phase_id: Phase ID for this expert
        
        Returns:
            PhaseExpert with cloned weights.
        """
        # Create expert with config (builds default heads)
        expert = cls(cfg, phase_id)
        
        # Deep copy the pre-trained heads (overwrites default)
        expert.traj_head = copy.deepcopy(traj_head)
        expert.gripper_head = copy.deepcopy(gripper_head)
        
        # [FIX vFinal] Force-Unfreeze Bootstrapped Weights
        # Since the Router is already frozen by this point, we MUST explicitly
        # enable gradients for the new experts.
        expert.traj_head.requires_grad_(True)
        expert.gripper_head.requires_grad_(True)
        
        # [FIX v2.1] Auto-detect chunk_size from output dimensions
        # Last layer of traj_head is Linear(dim, chunk_size * 7)
        last_traj_layer = list(expert.traj_head.modules())[-1]
        if isinstance(last_traj_layer, nn.Linear):
            detected_chunk = last_traj_layer.out_features // 7
            if detected_chunk != cfg.chunk_size:
                logger.warning(f"[Expert {phase_id}] Config mismatch! Config chunk={cfg.chunk_size}, "
                               f"Model chunk={detected_chunk}. Auto-correcting.")
                expert.chunk_size = detected_chunk
                expert.cfg.chunk_size = detected_chunk
        
        # [FIX v3.0] Auto-detect num_residual_blocks
        # Count ResidualMLPBlock instances
        res_blocks = sum(1 for m in expert.traj_head if isinstance(m, ResidualMLPBlock))
        if res_blocks != cfg.num_residual_blocks:
             logger.warning(f"[Expert {phase_id}] Config mismatch (Residual)! Config={cfg.num_residual_blocks}, "
                            f"Model={res_blocks}. Auto-correcting.")
             expert.cfg.num_residual_blocks = res_blocks

        expert.phase_id = phase_id
        
        logger.info(f"[PhaseExpert {phase_id}] Cloned weights from pre-trained planner.")
        
        return expert


# =============================================================================
# 4. EXPERT ARRAY (MoE Container)
# =============================================================================

class ExpertArray(nn.Module):
    """
    Container for multiple PhaseExperts.
    
    Handles routing during training (using ground truth phase labels) and
    inference (using predicted phase from Router).
    
    Key Features:
    - **Training Mode**: Routes to specific expert based on GT phase label.
    - **Inference Mode**: Routes based on argmax of phase logits from Router.
    - **Efficient Batching**: Groups samples by phase for parallel processing.
    """
    
    PHASE_NAMES = {
        0: "Approach",
        1: "Grasp",
        2: "Lift",
        3: "Place",
        4: "Retract"
    }
    
    def __init__(self, experts: List[PhaseExpert], num_phases: int = 5):
        super().__init__()
        self.num_phases = num_phases
        
        # Register experts as ModuleList for proper parameter tracking
        self.experts = nn.ModuleList(experts)
        
        if len(self.experts) != num_phases:
            raise ValueError(f"Expected {num_phases} experts, got {len(self.experts)}")
        
        logger.info(f"[ExpertArray] Initialized with {num_phases} experts: "
                    f"{[self.PHASE_NAMES.get(i, f'Phase_{i}') for i in range(num_phases)]}")
    
    def forward(
        self,
        z_traj: Tensor,
        z_grip: Tensor,
        phase_ids: Tensor,
        context: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Routes inputs to appropriate experts based on phase IDs.
        
        This implementation uses a loop over phases for clarity.
        For production, consider scatter/gather operations for efficiency.
        
        Args:
            z_traj: (B, D) trajectory embeddings from Router.
            z_grip: (B, D) gripper embeddings from Router.
            phase_ids: (B,) integer phase IDs (ground truth or predicted).
            context: (B, C) optional context vector.
        
        Returns:
            pose_chunks: (B, K, 7) aggregated predictions.
            grip_chunks: (B, K, 1) aggregated predictions.
        """
        B = z_traj.shape[0]
        K = self.experts[0].chunk_size
        device = z_traj.device
        
        # [FIX vFinal] Dynamic Dtype Matching for Mixed Precision (AMP)
        # Instead of z_traj.dtype (which might be float32 from the Router),
        # we initialize buffers with None and let them inherit the dtype 
        # produced by the Experts (which might be float16 in AMP).
        pose_chunks = None
        grip_chunks = None
        
        # Route to each expert
        for phase_id in range(self.num_phases):
            # Find samples belonging to this phase
            mask = (phase_ids == phase_id)
            if not mask.any():
                continue
            
            # Extract relevant inputs
            z_t = z_traj[mask]
            z_g = z_grip[mask]
            ctx = context[mask] if context is not None else None
            
            # Forward through expert
            pose, grip = self.experts[phase_id](z_t, z_g, ctx)
            
            # [FIX vFinal] Initialize buffers on first successful expert call
            if pose_chunks is None:
                pose_chunks = torch.zeros(B, K, 7, device=pose.device, dtype=pose.dtype)
                grip_chunks = torch.zeros(B, K, 1, device=grip.device, dtype=grip.dtype)
            
            # Scatter results back to output tensors
            pose_chunks[mask] = pose
            grip_chunks[mask] = grip
        
        # Fallback if no experts were active for this batch
        if pose_chunks is None:
            pose_chunks = torch.zeros(B, K, 7, device=device, dtype=z_traj.dtype)
            grip_chunks = torch.zeros(B, K, 1, device=device, dtype=z_traj.dtype)

        return pose_chunks, grip_chunks
    
    def forward_single_expert(
        self,
        z_traj: Tensor,
        z_grip: Tensor,
        phase_id: int,
        context: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Convenience method for inference with a single expert.
        
        Args:
            z_traj: (B, D) trajectory embeddings.
            z_grip: (B, D) gripper embeddings.
            phase_id: Integer phase ID (same for entire batch).
            context: (B, C) optional context vector.
        
        Returns:
            pose_chunks: (B, K, 7)
            grip_chunks: (B, K, 1)
        """
        return self.experts[phase_id](z_traj, z_grip, context)
    
    def forward_soft(
        self,
        z_traj: Tensor,
        z_grip: Tensor,
        router_probs: Tensor,
        context: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        [STAGE 3: END-TO-END RL] Differentiable soft routing.
        
        Unlike forward() which uses hard indexing (non-differentiable), this method
        computes a weighted sum over ALL expert outputs. This allows gradients to
        flow back through the router's decision, enabling end-to-end optimization.
        
        Output = Sum_i( router_probs[i] * Expert_i(input) )
        
        Args:
            z_traj: (B, D) trajectory embeddings from Router.
            z_grip: (B, D) gripper embeddings from Router.
            router_probs: (B, num_phases) probability distribution from Router.
                          Should be softmax(phase_logits), not argmax.
            context: (B, C) optional context vector.
        
        Returns:
            pose_chunks: (B, K, 7) weighted sum of expert predictions.
            grip_chunks: (B, K, 1) weighted sum of expert predictions.
        
        Note:
            This is computationally more expensive than hard routing (runs ALL experts).
            Use only during Stage 3 training when you need gradient flow to Router.
            For inference, use forward() with argmax(phase_logits).
        """
        B = z_traj.shape[0]
        K = self.experts[0].chunk_size
        device = z_traj.device
        dtype = z_traj.dtype
        
        # [FIX vFinal] Dynamic Dtype Matching for AMP
        pose_chunks = None
        grip_chunks = None
        # Weighted sum over all experts
        for phase_id in range(self.num_phases):
            # Get expert output for ALL samples
            pose, grip = self.experts[phase_id](z_traj, z_grip, context)  # (B, K, 7), (B, K, 1)
            
            if pose_chunks is None:
                pose_chunks = torch.zeros(B, K, 7, device=pose.device, dtype=pose.dtype)
                grip_chunks = torch.zeros(B, K, 1, device=grip.device, dtype=grip.dtype)
            
            # Get probability weight for this expert: (B,) -> (B, 1, 1) for broadcasting
            prob_weight = router_probs[:, phase_id].view(B, 1, 1)
            
            # Accumulate weighted output
            pose_chunks = pose_chunks + prob_weight * pose
            grip_chunks = grip_chunks + prob_weight * grip
        
        return pose_chunks, grip_chunks

    @classmethod
    def from_planner(
        cls,
        planner: nn.Module,
        num_phases: int = 5,
        context_dim: int = 0
    ) -> "ExpertArray":
        """
        Factory method to create an ExpertArray by cloning from a SemanticPlanner.
        
        This is the main entry point for bootstrapping MoE from a monolithic policy.
        
        Args:
            planner: Pre-trained SemanticPlanner model.
            num_phases: Number of experts to create.
            context_dim: Dimension of context vector for FiLM (0 = disabled).
        
        Returns:
            ExpertArray with all experts initialized from planner weights.
        """
        # Extract configuration from planner
        cfg = ExpertConfig(
            input_dim=planner.cfg.vision_feature_dim,
            chunk_size=planner.cfg.chunk_size,
            dropout=planner.cfg.dropout,
            num_residual_blocks=4,  # Match planner architecture
            context_dim=context_dim
        )
        
        logger.info(f"[ExpertArray] Creating {num_phases} experts from pre-trained planner...")
        
        experts = []
        for phase_id in range(num_phases):
            expert = PhaseExpert.from_planner_heads(
                traj_head=planner.traj_head,
                gripper_head=planner.gripper_head,
                cfg=cfg,
                phase_id=phase_id
            )
            experts.append(expert)
        
        logger.info(f"[ExpertArray] Bootstrapping complete. All experts initialized.")
        
        return cls(experts, num_phases)
    
    def get_expert_parameters(self) -> List[Dict]:
        """
        Returns parameter groups for optimizer setup.
        
        Useful for differential learning rates per expert.
        """
        param_groups = []
        for i, expert in enumerate(self.experts):
            param_groups.append({
                "name": f"expert_{i}_{self.PHASE_NAMES.get(i, 'Unknown')}",
                "params": list(expert.parameters())
            })
        return param_groups
    
    def freeze_expert(self, phase_id: int):
        """Freezes a specific expert's parameters."""
        for param in self.experts[phase_id].parameters():
            param.requires_grad = False
        logger.info(f"[ExpertArray] Expert {phase_id} ({self.PHASE_NAMES.get(phase_id)}) FROZEN")
    
    def unfreeze_expert(self, phase_id: int):
        """Unfreezes a specific expert's parameters."""
        for param in self.experts[phase_id].parameters():
            param.requires_grad = True
        logger.info(f"[ExpertArray] Expert {phase_id} ({self.PHASE_NAMES.get(phase_id)}) UNFROZEN")


# =============================================================================
# 5. UTILITY FUNCTIONS
# =============================================================================

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Counts the number of parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def log_expert_stats(expert_array: ExpertArray):
    """Logs parameter counts for each expert."""
    for i, expert in enumerate(expert_array.experts):
        total = count_parameters(expert, trainable_only=False)
        trainable = count_parameters(expert, trainable_only=True)
        logger.info(f"[Expert {i}] Total: {total:,} | Trainable: {trainable:,}")
