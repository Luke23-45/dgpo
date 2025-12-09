# FILE: models/residual_policy.py
# (Residual RL Fine-tuning for Semantic Planner)
#
# PURPOSE:
#   This module wraps a frozen BC-trained SemanticPlanner and adds a trainable
#   residual correction network. The residual learns small adjustments to the
#   BC policy's outputs, enabling closed-loop performance without destabilizing
#   the base policy.
#
# ARCHITECTURE:
#   action = frozen_bc_policy(obs) + alpha * residual_net(obs)
#
# REFERENCE:
#   - "Residual Policy Learning" (Silver et al., 2018)
#   - "HumanPlus: Humanoid Shadowing and Imitation" (2024)

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

logger = logging.getLogger(__name__)


@dataclass
class ResidualPolicyConfig:
    """Configuration for the Residual Policy."""
    
    # Residual network architecture
    hidden_dim: int = 256
    num_hidden_layers: int = 2
    
    # Residual scaling (controls how much correction is added)
    # Start small and can be annealed up during training
    residual_scale: float = 0.1
    
    # Action bounds for safety
    max_pos_residual: float = 0.05      # Max 5cm correction per step
    max_rot_residual: float = 0.1       # Max rotation correction (quaternion delta)
    max_grip_residual: float = 0.5      # Max gripper logit adjustment
    
    # Whether to learn a stochastic policy (for RL exploration)
    stochastic: bool = True
    log_std_init: float = -1.0          # Initial log std for stochastic policy
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    
    # Input dimensions (will be set from base policy)
    proprio_dim: int = 22
    chunk_size: int = 10


class ResidualMLP(nn.Module):
    """
    A simple MLP that outputs residual corrections.
    
    For stochastic policy, outputs both mean and log_std.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        stochastic: bool = True,
        log_std_init: float = -1.0
    ):
        super().__init__()
        self.stochastic = stochastic
        self.output_dim = output_dim
        
        # Build MLP layers
        layers = []
        in_dim = input_dim
        
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        
        self.trunk = nn.Sequential(*layers)
        
        # Output head for mean
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        
        # Output head for log_std (stochastic only)
        if stochastic:
            self.log_std = nn.Parameter(torch.ones(output_dim) * log_std_init)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            mean: (B, output_dim) residual mean
            log_std: (B, output_dim) or None if deterministic
        """
        features = self.trunk(x)
        mean = self.mean_head(features)
        
        if self.stochastic:
            log_std = self.log_std.expand_as(mean)
            return mean, log_std
        else:
            return mean, None


class ResidualPolicy(nn.Module):
    """
    Residual Policy wrapping a frozen BC policy.
    
    The residual policy learns small corrections on top of the base policy:
        action = frozen_base(obs) + alpha * tanh(residual(obs))
    
    For RL training, this provides a stable initialization (BC policy)
    while allowing exploration and improvement (residual).
    """
    
    def __init__(
        self,
        base_policy: nn.Module,
        cfg: ResidualPolicyConfig = ResidualPolicyConfig()
    ):
        super().__init__()
        self.cfg = cfg
        
        # Store and FREEZE the base BC policy
        self.base_policy = base_policy
        for param in self.base_policy.parameters():
            param.requires_grad = False
        self.base_policy.eval()
        
        logger.info("Base BC policy frozen with %d parameters", 
                   sum(p.numel() for p in self.base_policy.parameters()))
        
        # Get dimensions from base policy config
        base_cfg = getattr(base_policy, 'cfg', None)
        if base_cfg:
            self.proprio_dim = base_cfg.proprio_dim
            self.chunk_size = base_cfg.chunk_size
        else:
            self.proprio_dim = cfg.proprio_dim
            self.chunk_size = cfg.chunk_size
        
        # Residual network input: proprio (for state awareness)
        # Output: corrections for pose (K*7) and gripper (K*1)
        self.pose_residual_dim = self.chunk_size * 7  # Position + quaternion per step
        self.grip_residual_dim = self.chunk_size * 1  # Gripper logit per step
        
        # Separate residual networks for pose and gripper
        self.pose_residual_net = ResidualMLP(
            input_dim=self.proprio_dim,
            output_dim=self.pose_residual_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_hidden_layers,
            stochastic=cfg.stochastic,
            log_std_init=cfg.log_std_init
        )
        
        self.grip_residual_net = ResidualMLP(
            input_dim=self.proprio_dim,
            output_dim=self.grip_residual_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_hidden_layers,
            stochastic=cfg.stochastic,
            log_std_init=cfg.log_std_init
        )
        
        # Track residual scale (can be annealed during training)
        self.register_buffer('residual_scale', torch.tensor(cfg.residual_scale))
        
        logger.info("Residual networks initialized: pose=%d, grip=%d params",
                   sum(p.numel() for p in self.pose_residual_net.parameters()),
                   sum(p.numel() for p in self.grip_residual_net.parameters()))
    
    def set_residual_scale(self, scale: float):
        """Anneal the residual scale during training."""
        self.residual_scale.fill_(scale)
    
    def _clip_residual(
        self,
        residual: torch.Tensor,
        is_pose: bool
    ) -> torch.Tensor:
        """
        Clip residual to safety bounds.
        
        For pose: separate limits for position and rotation components.
        For gripper: single limit.
        """
        if is_pose:
            # residual shape: (B, K, 7) - split into pos and rot
            B = residual.shape[0]
            residual = residual.view(B, self.chunk_size, 7)
            
            pos_res = residual[..., :3]
            rot_res = residual[..., 3:]
            
            # Clip position
            pos_res = torch.clamp(pos_res, 
                                  -self.cfg.max_pos_residual, 
                                  self.cfg.max_pos_residual)
            
            # Clip rotation (quaternion delta components)
            rot_res = torch.clamp(rot_res,
                                  -self.cfg.max_rot_residual,
                                  self.cfg.max_rot_residual)
            
            return torch.cat([pos_res, rot_res], dim=-1)
        else:
            # Grip residual: simple clamp
            return torch.clamp(residual.view(-1, self.chunk_size, 1),
                              -self.cfg.max_grip_residual,
                              self.cfg.max_grip_residual)
    
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        deterministic: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass with residual correction.
        
        Args:
            batch: Dictionary containing model inputs
                - prev_image, curr_image, goal_image, curr_proprio
            deterministic: If True, use mean action instead of sampling
            
        Returns:
            Dictionary containing:
                - pose_chunk: BC pose + residual correction
                - gripper_chunk: BC gripper + residual correction
                - phase_logits: From BC policy (unchanged)
                - log_prob: Log probability if stochastic (for RL)
                - entropy: Policy entropy if stochastic (for RL)
        """
        # 1. Get base policy predictions (no gradients)
        with torch.no_grad():
            base_outputs = self.base_policy(batch)
        
        base_pose = base_outputs['pose_chunk']      # (B, K, 7)
        base_grip = base_outputs['gripper_chunk']   # (B, K, 1)
        phase_logits = base_outputs['phase_logits'] # (B, N_Phases)
        
        # 2. Compute residual corrections
        # Use normalized proprio for residual nets if available, otherwise fallback to raw
        proprio_for_residual = batch.get('proprio_norm', batch['curr_proprio'])  # (B, proprio_dim)
        
        pose_res_mean, pose_res_log_std = self.pose_residual_net(proprio_for_residual)
        grip_res_mean, grip_res_log_std = self.grip_residual_net(proprio_for_residual)
        
        B = proprio_for_residual.shape[0]
        
        # 3. Sample or use mean
        if self.cfg.stochastic and not deterministic:
            # Clamp log_std for numerical stability
            pose_log_std = torch.clamp(pose_res_log_std, 
                                       self.cfg.log_std_min, 
                                       self.cfg.log_std_max)
            grip_log_std = torch.clamp(grip_res_log_std,
                                       self.cfg.log_std_min,
                                       self.cfg.log_std_max)
            
            # Sample from Gaussian
            pose_dist = Normal(pose_res_mean, pose_log_std.exp())
            grip_dist = Normal(grip_res_mean, grip_log_std.exp())
            
            pose_residual_raw = pose_dist.rsample()
            grip_residual_raw = grip_dist.rsample()
            
            # Compute log probabilities
            pose_log_prob = pose_dist.log_prob(pose_residual_raw).sum(dim=-1)
            grip_log_prob = grip_dist.log_prob(grip_residual_raw).sum(dim=-1)
            log_prob = pose_log_prob + grip_log_prob
            
            # Compute entropy
            pose_entropy = pose_dist.entropy().sum(dim=-1)
            grip_entropy = grip_dist.entropy().sum(dim=-1)
            entropy = pose_entropy + grip_entropy
        else:
            pose_residual_raw = pose_res_mean
            grip_residual_raw = grip_res_mean
            log_prob = None
            entropy = None
        
        # 4. Apply tanh squashing and clip
        pose_residual = torch.tanh(pose_residual_raw)
        grip_residual = torch.tanh(grip_residual_raw)
        
        pose_residual = self._clip_residual(pose_residual, is_pose=True)
        grip_residual = self._clip_residual(grip_residual, is_pose=False)
        
        # 5. Combine base + scaled residual
        final_pose = base_pose + self.residual_scale * pose_residual
        final_grip = base_grip + self.residual_scale * grip_residual
        
        # 6. Renormalize quaternions after adding residual
        pos_xyz = final_pose[..., :3]
        quat_raw = final_pose[..., 3:]
        quat_norm = F.normalize(quat_raw, p=2, dim=-1, eps=1e-6)
        final_pose = torch.cat([pos_xyz, quat_norm], dim=-1)
        
        return {
            'pose_chunk': final_pose,
            'gripper_chunk': final_grip,
            'phase_logits': phase_logits,
            'log_prob': log_prob,
            'entropy': entropy,
            # Also return base outputs for analysis
            'base_pose_chunk': base_pose,
            'base_gripper_chunk': base_grip,
        }
    
    def get_action(
        self,
        batch: Dict[str, torch.Tensor],
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method for RL training.
        
        Returns first action of chunk (receding horizon) as [pose, gripper].
        """
        outputs = self.forward(batch, deterministic=deterministic)
        
        # Get first step of chunk (receding horizon)
        pose = outputs['pose_chunk'][:, 0, :]    # (B, 7)
        grip = outputs['gripper_chunk'][:, 0, :] # (B, 1)
        
        action = torch.cat([pose, grip], dim=-1)  # (B, 8)
        log_prob = outputs.get('log_prob', None)
        
        return action, log_prob
    
    def evaluate_actions(
        self,
        batch: Dict[str, torch.Tensor],
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy for given actions.
        Used during PPO update to recompute log probs with new policy params.
        
        Note: This is an approximation - we compute log_prob of the residual,
        not the full action, since the base policy is frozen.
        """
        outputs = self.forward(batch, deterministic=False)
        return outputs.get('log_prob', torch.zeros(actions.shape[0])), \
               outputs.get('entropy', torch.zeros(actions.shape[0]))
    
    def get_trainable_parameters(self):
        """Returns only the trainable residual network parameters."""
        params = list(self.pose_residual_net.parameters()) + \
                 list(self.grip_residual_net.parameters())
        return params
    
    def trainable_param_count(self) -> int:
        """Count of trainable parameters."""
        return sum(p.numel() for p in self.get_trainable_parameters())


def create_residual_policy(
    checkpoint_path: str,
    device: torch.device = torch.device('cuda'),
    cfg: Optional[ResidualPolicyConfig] = None
) -> ResidualPolicy:
    """
    Factory function to create ResidualPolicy from a BC checkpoint.
    
    Args:
        checkpoint_path: Path to trained SemanticPlanner checkpoint
        device: Device to load model on
        cfg: Optional ResidualPolicyConfig
        
    Returns:
        ResidualPolicy with frozen base and trainable residual
    """
    from train.train_semantic_planner import SemanticPlannerLightningModule
    
    logger.info(f"Loading base policy from: {checkpoint_path}")
    
    # Load the Lightning module
    pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device
    )
    base_policy = pl_module.model.eval().to(device)
    
    # Create residual policy
    if cfg is None:
        cfg = ResidualPolicyConfig(
            proprio_dim=base_policy.cfg.proprio_dim,
            chunk_size=base_policy.cfg.chunk_size
        )
    
    residual_policy = ResidualPolicy(base_policy, cfg).to(device)
    
    logger.info(f"ResidualPolicy created with {residual_policy.trainable_param_count()} trainable params")
    
    return residual_policy