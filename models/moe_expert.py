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
    
    This module serves as the 'Specialist' in the Mixture-of-Experts architecture.
    It inherits generalist capabilities from a pre-trained Router via bootstrapping
    and refines them for a specific task phase (e.g., Approach, Grasp).
    
    Architecture:
        Input (z) -> [LayerNorm] -> [FiLM Modulation] -> [MLP + Residuals] -> Action Chunk
    
    Robustness Features:
    - Decoupled FiLM: Independent modulation for Trajectory and Gripper heads.
    - Memory Safety: Uses reshape() instead of view() to handle non-contiguous tensors.
    - Post-Norm Injection: Applies conditioning after normalization to prevent signal washout.
    - Uncertainty: Built-in Monte Carlo Dropout support.
    
    Args:
        cfg: ExpertConfig with architecture hyperparameters.
        phase_id: Integer ID of this expert's designated phase (for logging).
    """
    
    def __init__(self, cfg: ExpertConfig, phase_id: int = -1):
        super().__init__()
        self.cfg = cfg
        self.phase_id = phase_id
        self.chunk_size = cfg.chunk_size
        
        logger.info(f"[PhaseExpert {phase_id}] Initializing | Dim: {cfg.input_dim} | Chunk: {cfg.chunk_size}")
        
        # --- A. Contextual Modulation (FiLM) ---
        # CRITICAL: We use separate layers for Trajectory and Gripper.
        # This allows the context to scale/shift the arm and hand INDEPENDENTLY.
        # (e.g., "Heavy Object" -> Slow Arm, Strong Grip)
        self.use_film = cfg.context_dim > 0
        if self.use_film:
            self.traj_film_layer = FiLMLayer(cfg.context_dim, cfg.input_dim)
            self.grip_film_layer = FiLMLayer(cfg.context_dim, cfg.input_dim)
            logger.info(f"[PhaseExpert {phase_id}] FiLM Conditioning: ENABLED (Decoupled)")
        else:
            self.traj_film_layer = None
            self.grip_film_layer = None
        
        # --- B. Trajectory Head (Pose Prediction) ---
        # Standard SOTA Residual MLP architecture
        # Note: When bootstrapping, this is overwritten by the cloned Router weights
        traj_layers = [
            nn.LayerNorm(cfg.input_dim),
            nn.Linear(cfg.input_dim, cfg.input_dim),
            nn.GELU(),
        ]
        for _ in range(cfg.num_residual_blocks):
            traj_layers.append(ResidualMLPBlock(cfg.input_dim, cfg.dropout))
            
        # Output: 3 Position + 4 Quaternion (per timestep in chunk)
        traj_layers.extend([
            nn.LayerNorm(cfg.input_dim),
            nn.Linear(cfg.input_dim, cfg.chunk_size * 7)
        ])
        self.traj_head = nn.Sequential(*traj_layers)
        
        # --- C. Gripper Head (State Prediction) ---
        # Smaller capacity is usually sufficient for binary/scalar gripper state
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
    
    def _forward_head_with_film(
        self, 
        head: nn.Sequential, 
        x: Tensor, 
        film_layer: Optional[nn.Module], 
        context: Optional[Tensor]
    ) -> Tensor:
        """
        Helper method to inject FiLM modulation safely into a Sequential block.
        
        Strategy:
        1. Identify the first LayerNorm in the sequence.
        2. Run input through layers up to and including LayerNorm.
        3. Apply FiLM modulation (Gamma * x + Beta).
           * This must happen AFTER Norm, otherwise Norm cancels the shift/scale.
        4. Run the result through the rest of the layers.
        """
        # 1. Find the injection point (Post-Norm)
        ln_idx = -1
        for i, layer in enumerate(head):
            if isinstance(layer, nn.LayerNorm):
                ln_idx = i
                break
        
        # 2. Pre-processing (Up to Norm)
        if ln_idx == -1:
            # Fallback: No Norm found (Unlikely in SOTA). Apply FiLM directly.
            # This handles cases where architectures might differ.
            norm_x = x
            start_idx = 0
        else:
            norm_x = x
            for i in range(ln_idx + 1):
                norm_x = head[i](norm_x)
            start_idx = ln_idx + 1
            
        # 3. Apply Modulation
        if self.use_film and film_layer is not None and context is not None:
            norm_x = film_layer(norm_x, context)
            
        # 4. Post-processing (Rest of Network)
        out = norm_x
        for i in range(start_idx, len(head)):
            out = head[i](out)
            
        return out

    def forward(
        self,
        z_traj: Tensor,
        z_grip: Optional[Tensor] = None,
        context: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass predicting action chunks.
        
        Args:
            z_traj: (B, D) visual embedding for trajectory prediction.
            z_grip: (B, D) visual embedding for gripper prediction.
                    If None, defaults to using z_traj (Robustness check).
            context: (B, C) optional context vector for FiLM modulation.
        
        Returns:
            pose_chunk: (B, K, 7) [x, y, z, qx, qy, qz, qw] normalized quaternion.
            grip_chunk: (B, K, 1) logits (unscaled).
        """
        B = z_traj.shape[0]
        
        # Robust Fallback: If no separate gripper embedding, share the trajectory one
        if z_grip is None:
            z_grip = z_traj
        
        # --- 1. Trajectory Prediction ---
        # Execute Trajectory Head with specific Trajectory Modulation
        raw_traj = self._forward_head_with_film(
            self.traj_head, z_traj, self.traj_film_layer, context
        )
        
        # [ROBUSTNESS] Use reshape instead of view.
        # View crashes on non-contiguous tensors; reshape handles memory copy automatically.
        pred_traj = raw_traj.reshape(B, self.chunk_size, 7)
        
        # Geometric Safety: Normalize Quaternions
        # Invalid quaternions (norm != 1) cause physics explosions in simulation.
        pos_xyz = pred_traj[..., :3]
        quat_raw = pred_traj[..., 3:]
        # Epsilon ensures no division by zero
        quat_norm = F.normalize(quat_raw, p=2, dim=-1, eps=1e-6) 
        pose_chunk = torch.cat([pos_xyz, quat_norm], dim=-1)
        
        # --- 2. Gripper Prediction ---
        # Execute Gripper Head with specific Gripper Modulation
        raw_grip = self._forward_head_with_film(
            self.gripper_head, z_grip, self.grip_film_layer, context
        )
        
        # Reshape safely
        grip_chunk = raw_grip.reshape(B, self.chunk_size, 1)
        
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
        Monte Carlo Dropout for Uncertainty Estimation.
        
        Used to detect Out-of-Distribution (OOD) states during inference.
        If uncertainty > threshold, the system can trigger a safety stop.
        """
        # Preserve original training state
        was_training = self.training
        
        # Force training mode to activate Dropout layers (even during inference)
        self.train()
        
        pose_samples = []
        grip_samples = []
        
        for _ in range(n_samples):
            pose, grip = self.forward(z_traj, z_grip, context)
            pose_samples.append(pose)
            grip_samples.append(grip)
        
        # Restore original state
        self.train(was_training)
        
        # Stack and compute statistics
        pose_stack = torch.stack(pose_samples, dim=0)
        grip_stack = torch.stack(grip_samples, dim=0)
        
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
        Bootstrapping Factory: Creates an Expert by cloning pre-trained heads.
        
        Includes aggressive validation to ensure the requested config matches
        the actual architecture of the source weights.
        """
        # 1. Initialize empty expert with configuration
        expert = cls(cfg, phase_id)
        
        # 2. Clone Weights (Deep Copy to ensure independence from Router)
        expert.traj_head = copy.deepcopy(traj_head)
        expert.gripper_head = copy.deepcopy(gripper_head)
        
        # 3. [CRITICAL] Force Unfreeze
        # The Router is frozen during this phase, so cloned weights might inherit 
        # 'requires_grad=False'. We explicitly enable gradients for the experts.
        expert.traj_head.requires_grad_(True)
        expert.gripper_head.requires_grad_(True)
        
        # 4. Architecture Auto-Correction (Sanity Checks)
        
        # Check A: Chunk Size via Output Layer Dimensions
        try:
            # Check if it's a Sequential or similar iterable container
            if isinstance(expert.traj_head, nn.Sequential):
                last_traj_layer = expert.traj_head[-1]
            else:
                # Fallback for raw Module (rare but safe)
                last_traj_layer = list(expert.traj_head.children())[-1]

            if isinstance(last_traj_layer, nn.Linear):
                # 7 is dim of (x,y,z,qx,qy,qz,qw)
                detected_chunk = last_traj_layer.out_features // 7
                if detected_chunk != cfg.chunk_size:
                    logger.warning(f"[Expert {phase_id}] ⚠️ Config mismatch! Config Chunk={cfg.chunk_size}, "
                                   f"Detected Model Chunk={detected_chunk}. Auto-correcting expert.")
                    expert.chunk_size = detected_chunk
                    expert.cfg.chunk_size = detected_chunk
        except Exception as e:
            logger.warning(f"[Expert {phase_id}] Failed to validate chunk size: {e}")
        
        # Check B: Depth via Residual Blocks Count
        # (Checks how many ResidualMLPBlocks are in the sequential)
        try:
            # We still search recursively for blocks as they are nested
            res_blocks = sum(1 for m in expert.traj_head.modules() if isinstance(m, ResidualMLPBlock))
            if res_blocks != cfg.num_residual_blocks:
                 logger.warning(f"[Expert {phase_id}] ⚠️ Config mismatch! Config Blocks={cfg.num_residual_blocks}, "
                                f"Detected Model Blocks={res_blocks}. Auto-correcting expert.")
                 expert.cfg.num_residual_blocks = res_blocks
        except NameError:
            pass
        
        logger.info(f"[PhaseExpert {phase_id}] Bootstrapped successfully from SemanticPlanner.")
        
        return expert


# =============================================================================
# 4. EXPERT ARRAY (MoE Container)
# =============================================================================



class ExpertArray(nn.Module):
    """
    Container for multiple PhaseExperts in a Phase-Locked Mixture-of-Experts architecture.
    
    This module manages the routing of inputs to specific experts. It supports two modes:
    1. **Hard Routing (forward)**: Sparse execution. Routes samples to exactly one expert 
       based on an integer phase ID. Used for standard training and inference.
    2. **Soft Routing (forward_soft)**: Dense execution. Computes a weighted sum of ALL 
       experts based on router probabilities. Used for End-to-End differentiable training.
    
    Attributes:
        experts (nn.ModuleList): The collection of specialized PhaseExpert modules.
        num_phases (int): Total number of distinct task phases.
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
        
        # Validation: Ensure strict alignment between phase counts and expert modules
        if len(experts) != num_phases:
            raise ValueError(f"[ExpertArray] Critical Config Error: Expected {num_phases} experts, "
                             f"but received {len(experts)}.")
        
        # Register as ModuleList to ensure parameters are registered with the optimizer
        self.experts = nn.ModuleList(experts)
        
        # Log initialization for audit trail
        expert_names = [self.PHASE_NAMES.get(i, f'Phase_{i}') for i in range(num_phases)]
        logger.info(f"[ExpertArray] Initialized with {num_phases} experts: {expert_names}")
    
    def forward(
        self,
        z_traj: Tensor,
        z_grip: Tensor,
        phase_ids: Tensor,
        context: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Hard Routing (Sparse): Routes inputs to appropriate experts based on phase IDs.
        
        Optimized for memory efficiency: Only runs the expert required for each sample.
        
        Args:
            z_traj: (B, D) trajectory embeddings from Router.
            z_grip: (B, D) gripper embeddings from Router.
            phase_ids: (B,) integer phase IDs (ground truth or predicted).
            context: (B, C) optional context vector.
        
        Returns:
            pose_chunks: (B, K, 7) aggregated pose predictions.
            grip_chunks: (B, K, 1) aggregated gripper predictions.
        """
        B = z_traj.shape[0]
        # Retrieve output chunk size from the first expert (assumed uniform)
        K = self.experts[0].chunk_size
        device = z_traj.device
        
        # [ROBUSTNESS] Lazy Buffer Initialization
        # We start with None and initialize buffers upon the first successful expert execution.
        # Why? In Mixed Precision (AMP), experts might output float16 while z_traj is float32.
        # Initializing zeros(..., dtype=z_traj.dtype) would cause a dtype mismatch crash.
        pose_chunks = None
        grip_chunks = None
        
        # Iterate through each phase index
        for phase_id in range(self.num_phases):
            # Efficient Masking: Find samples belonging to this phase
            mask = (phase_ids == phase_id)
            
            # Optimization: Skip expert entirely if no samples match
            if not mask.any():
                continue
            
            # Gather inputs for this specific expert
            z_t = z_traj[mask]
            z_g = z_grip[mask]
            ctx = context[mask] if context is not None else None
            
            # Execute Expert
            # pose: (Batch_Subset, K, 7), grip: (Batch_Subset, K, 1)
            pose, grip = self.experts[phase_id](z_t, z_g, ctx)
            
            # [CRITICAL] Initialize buffers using the EXPERT'S output dtype/device
            if pose_chunks is None:
                pose_chunks = torch.zeros(B, K, 7, device=pose.device, dtype=pose.dtype)
                grip_chunks = torch.zeros(B, K, 1, device=grip.device, dtype=grip.dtype)
            
            # Scatter results back into the global batch buffer
            pose_chunks[mask] = pose
            grip_chunks[mask] = grip
        
        # [ROBUSTNESS] Fallback for Empty/Invalid Batch
        # If pose_chunks is still None, it means NO experts were executed (e.g., all phase_ids were -1).
        # We must return a valid zero tensor to prevent downstream crashes.
        if pose_chunks is None:
            pose_chunks = torch.zeros(B, K, 7, device=device, dtype=z_traj.dtype)
            grip_chunks = torch.zeros(B, K, 1, device=device, dtype=z_traj.dtype)

        return pose_chunks, grip_chunks
    
    def forward_soft(
        self,
        z_traj: Tensor,
        z_grip: Tensor,
        router_probs: Tensor,
        context: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Soft Routing (Dense): Differentiable weighted sum of all experts.
        
        Used for End-to-End training (Stage 3) to allow gradients to flow from
        expert performance back to the Router.
        
        Args:
            z_traj: (B, D) trajectory embeddings.
            z_grip: (B, D) gripper embeddings.
            router_probs: (B, num_phases) normalized probabilities.
            context: (B, C) optional context vector.
        
        Returns:
            pose_chunks: (B, K, 7) weighted output.
            grip_chunks: (B, K, 1) weighted output.
        """
        B = z_traj.shape[0]
        K = self.experts[0].chunk_size
        device = z_traj.device
        
        # [SAFETY] Gradient Explosion Guard
        # If logits are passed instead of probabilities, the weighted sum will explode.
        # We check if sums are close to 1.0. If not, we force Softmax.
        prob_sums = router_probs.sum(dim=-1)
        if not torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-3):
            # Log warning only once per run usually, but here we prioritize safety
            # logger.warning("[ExpertArray] Raw logits detected in forward_soft. Applying Softmax.")
            router_probs = F.softmax(router_probs, dim=-1)
        
        pose_chunks = None
        grip_chunks = None
        
        # Iterate over all experts (Dense Execution)
        for phase_id in range(self.num_phases):
            # Run expert on the FULL batch
            pose, grip = self.experts[phase_id](z_traj, z_grip, context)
            
            # Lazy Init for Dtype Safety (AMP)
            if pose_chunks is None:
                pose_chunks = torch.zeros(B, K, 7, device=pose.device, dtype=pose.dtype)
                grip_chunks = torch.zeros(B, K, 1, device=grip.device, dtype=grip.dtype)
            
            # Get gating weight for this expert: (B,) -> (B, 1, 1) for broadcasting
            # [SAFETY] Use reshape instead of view to handle non-contiguous probability tensors
            prob_weight = router_probs[:, phase_id].reshape(B, 1, 1)
            
            # Accumulate: Output += Weight * Expert_Output
            pose_chunks = pose_chunks + (pose * prob_weight)
            grip_chunks = grip_chunks + (grip * prob_weight)
            
        return pose_chunks, grip_chunks

    def forward_single_expert(
        self,
        z_traj: Tensor,
        z_grip: Tensor,
        phase_id: int,
        context: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Inference Utility: Force execution of a specific expert for the entire batch.
        Useful for debugging or analyzing specific expert behaviors.
        """
        return self.experts[phase_id](z_traj, z_grip, context)

    @classmethod
    def from_planner(
        cls,
        planner: nn.Module,
        num_phases: int = 5,
        context_dim: int = 0
    ) -> "ExpertArray":
        """
        Factory Method: Bootstraps an ExpertArray from a pre-trained SemanticPlanner.
        
        This initializes 'num_phases' experts, each starting as a clone of the 
        planner's generalist heads. This avoids the "cold start" problem.
        
        Args:
            planner: Pre-trained SemanticPlanner model (source of weights).
            num_phases: Number of experts to create.
            context_dim: Dimension of context vector for FiLM (0 = disabled).
        
        Returns:
            ExpertArray instance.
        """
        # 1. Extract Architecture Configuration from Planner
        # This ensures experts match the router's embedding space exactly
        cfg = ExpertConfig(
            input_dim=planner.cfg.vision_feature_dim,
            chunk_size=planner.cfg.chunk_size,
            dropout=planner.cfg.dropout,
            num_residual_blocks=4,  # Standard default, will be auto-corrected if needed
            context_dim=context_dim
        )
        
        logger.info(f"[ExpertArray] Creating {num_phases} experts from pre-trained planner...")
        
        experts = []
        for phase_id in range(num_phases):
            # 2. Clone weights using the robust PhaseExpert factory
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
        Returns parameter groups for optimizers.
        Useful for applying different learning rates to different experts if needed.
        """
        param_groups = []
        for i, expert in enumerate(self.experts):
            param_groups.append({
                "name": f"expert_{i}_{self.PHASE_NAMES.get(i, 'Unknown')}",
                "params": list(expert.parameters())
            })
        return param_groups
    
    def freeze_expert(self, phase_id: int):
        """Freezes parameters for a specific expert."""
        for param in self.experts[phase_id].parameters():
            param.requires_grad = False
        logger.info(f"[ExpertArray] Expert {phase_id} ({self.PHASE_NAMES.get(phase_id)}) FROZEN")
    
    def unfreeze_expert(self, phase_id: int):
        """Unfreezes parameters for a specific expert."""
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
