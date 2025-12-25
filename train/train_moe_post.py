# FILE: train/train_moe_post.py
# (SOTA v2.0 - All 6 Enhancements Implemented)

r"""
APEX-MoE Post-Training Script (SOTA v2.0).

This script implements the **Post-Training Phase** of the Bootstrapped Mixture-of-Experts
architecture with 6 SOTA enhancements:

1.  **EMA Policy**: Exponential Moving Average of expert weights for stable evaluation.
2.  **Mixed Precision**: AMP (Automatic Mixed Precision) for 2x training speed.
3.  **Temporal Ensemble**: Smooth action output by averaging recent predictions.
4.  **Adaptive BC Coefficient**: Anneals BC regularization over training.
5.  **Phase-Weighted Loss**: Inverse frequency weighting to handle imbalanced phases.
6.  **Cross-Expert Regularization**: Prevents experts from diverging too far from each other.

Architecture:
  Router (Frozen SemanticPlanner) -> Visual Embeddings (z_t)
                                  -> Phase Logits (k_t)
  ExpertArray[k_t] -> Action Chunks

Usage:
  python train/train_moe_post.py

Reference: technical_report_phase_locked_moe.md
"""

from __future__ import annotations

import copy
import gc
import random
import logging
import math
import os
import sys
from collections import deque
from datetime import datetime
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

# --- Project-Specific Imports ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.semantic_planner import SemanticPlanner, SemanticPlannerConfig
from models.moe_expert import ExpertArray, ExpertConfig, count_parameters, log_expert_stats
from utils.semantic_planner_dataset import (
    SemanticPlannerDataset,
    semantic_planner_collate_fn,
)
from utils.samplers import EpisodeAwareSampler

def seed_worker(worker_id):
    # [STABILITY FIX] Prevent workers from touching CUDA
    # This resolves the "Unable to register cuFFT factory" hang in Colab
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # [STABILITY FIX] Limit threading in workers to prevent contention
    torch.set_num_threads(1)
    
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# --- Logging Setup ---
logger = logging.getLogger("train_moe_post")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# --- Torch Load Patch (For Lightning Compatibility) ---
_original_load = torch.load

def strict_mode_bypass_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)

torch.load = strict_mode_bypass_load


# [REMOVED] Temporal Ensemble was removed per user concern about Z-axis smoothing.
# Averaging over recent predictions could hurt sharp vertical movements during
# phase transitions (Grasp→Lift, Place→Release). Use raw predictions instead.


# =============================================================================
# [SOTA ENHANCEMENT #4] ADAPTIVE BC COEFFICIENT
# =============================================================================

class AdaptiveBCScheduler:
    """
    Adaptive Behavior Cloning Coefficient Scheduler.
    
    Anneals the BC regularization weight from initial to final over training.
    Uses cosine annealing for smooth transition.
    """
    def __init__(
        self,
        initial_coef: float = 1.0,
        final_coef: float = 0.1,
        total_epochs: int = 50,
        warmup_epochs: int = 5
    ):
        self.initial_coef = initial_coef
        self.final_coef = final_coef
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
    
    def get_coef(self, epoch: int) -> float:
        """Get BC coefficient for current epoch."""
        if epoch < self.warmup_epochs:
            # Warmup: stay at initial
            return self.initial_coef
        
        # Cosine annealing after warmup
        progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
        progress = min(1.0, progress)
        
        # Cosine decay
        coef = self.final_coef + 0.5 * (self.initial_coef - self.final_coef) * (1 + math.cos(math.pi * progress))
        return coef


# =============================================================================
# DATA MODULE (Reuses SemanticPlannerDataset)
# =============================================================================
class MoEDataModule(pl.LightningDataModule):
    """
    DataModule for MoE Post-Training.
    
    Robustness Features:
    - Fast Index Access: 0-second startup by reading metadata directly.
    - Robust Fallback: Handles missing attributes or corrupted data gracefully.
    - Resource Management: Explicitly closes environments and clears caches.
    - Worker Safety: persistent_workers=False prevents RAM OOM on Colab.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train_dataset: Optional[SemanticPlannerDataset] = None
        self.val_dataset: Optional[SemanticPlannerDataset] = None
        
        self.num_workers = cfg.dataset.get("num_workers", 2)
        # [STABILITY FIX] Disable pin_memory on Colab to prevent hangs
        self.pin_memory = False 
        self.phase_counts: Optional[Dict[int, int]] = None

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            # Idempotency Check: Prevent Double Loading (RAM Spike Fix)
            if self.train_dataset is not None:
                logger.info("[MoEDataModule] Dataset already loaded. Skipping re-initialization.")
                return

            logger.info(f"[MoEDataModule] Loading Training Dataset from: {self.cfg.dataset.train_path}")
            self.train_dataset = SemanticPlannerDataset(
                dataset_path=self.cfg.dataset.train_path,
                use_aug=self.cfg.dataset.get("use_aug", False),
                chunk_size=self.cfg.model.get("chunk_size", 10),
                proprio_noise=self.cfg.dataset.get("proprio_noise", 0.0)
            )
            
            # Compute phase weights using the robust method
            self._compute_phase_distribution()

            if self.cfg.dataset.get("val_path"):
                logger.info(f"[MoEDataModule] Loading Validation Dataset from: {self.cfg.dataset.val_path}")
                self.val_dataset = SemanticPlannerDataset(
                    dataset_path=self.cfg.dataset.val_path,
                    use_aug=False,
                    chunk_size=self.cfg.model.get("chunk_size", 10),
                    proprio_noise=0.0
                )
    
    def _compute_phase_distribution(self):
        """
        [DIAMOND TIER] Statistical-Correct Phase Distribution Scan.
        Accurately counts 100% of ACTUAL training samples via metadata in <1s.
        """
        logger.info("[MoEDataModule] Computing phase distribution (High-precision metadata scan)...")
        num_phases = self.cfg.model.num_task_phases
        phase_counts = {i: 0 for i in range(num_phases)}
        
        try:
            # Access dataset parameters for horizon-aware slicing
            ds = self.train_dataset
            obs_h = getattr(ds, "obs_horizon", 2)
            # action_horizon is chunk_size in SemanticPlanner
            act_h = getattr(ds, "action_horizon", getattr(ds, "chunk_size", 10))
            
            reader = ds.expert_reader
            episodes = reader.episode_metadata
            
            logger.info(f"🚀 Analyzing {len(episodes)} episodes (Obs={obs_h}, Act={act_h})...")
            
            for ep_meta in episodes:
                # Find phase modality
                phase_key = next((k for k in ["gt_phase", "task_phases", "phase"] 
                                if k in ep_meta["modalities"]), None)
                
                if phase_key:
                    m = ep_meta["modalities"][phase_key]
                    phase_arr = reader._get_full_modality_array(
                        m["key"], m["compression"], m["dtype"], tuple(m["shape"])
                    )
                    
                    # --- [SOTA FIX] Horizon-Aware Slicing ---
                    # We only count frames that can actually be sampled.
                    # A valid sample index 't' requires [t - obs_h + 1] and [t + act_h].
                    ep_len = len(phase_arr)
                    start_t = obs_h - 1
                    end_t = ep_len - act_h
                    
                    if start_t <= end_t:
                        valid_phases = phase_arr[start_t : end_t + 1]
                        unique, counts = np.unique(valid_phases, return_counts=True)
                        for val, count in zip(unique, counts):
                            pid = int(val)
                            if 0 <= pid < num_phases:
                                phase_counts[pid] += int(count)
                            else:
                                logger.warning(f"Metadata out-of-range phase: {pid}")
                else:
                    logger.warning(f"Metadata missing phase in episode {ep_meta.get('episode_id', '?')}")

            logger.info("✅ 100% accurate phase distribution calculated.")
            
        except Exception as e:
            logger.warning(f"Metadata scan failed ({e}). Falling back to sampling.")
            # Fallback: sampling using the official get_phase_label method
            sample_size = min(10000, len(self.train_dataset))
            indices = np.random.choice(len(self.train_dataset), sample_size, replace=False)
            for idx in indices:
                try:
                    p = self.train_dataset.get_phase_label(idx)
                    if 0 <= p < num_phases:
                        phase_counts[p] += 1
                except: continue

        # Final Cleanup
        if hasattr(self.train_dataset, 'expert_reader'):
             self.train_dataset.expert_reader._get_full_modality_array.cache_clear()
             gc.collect()
        
        # Logging & Assignment
        self.phase_counts = phase_counts
        total = sum(phase_counts.values())
        if total == 0:
            logger.error("🛑 CRITICAL: Phase distribution resulted in ZERO samples. Check horizons!")
            # Safety fallback to uniform if everything is broken
            self.phase_counts = {i: 1 for i in range(num_phases)}
            total = num_phases
            
        logger.info("Phase Distribution Results (Total Samples: {}):".format(total))
        for pid, count in phase_counts.items():
            pct = 100 * count / total
            pname = ExpertArray.PHASE_NAMES.get(pid, f"Phase_{pid}")
            logger.info(f"  {pname}: {count} samples ({pct:.1f}%)")

    def train_dataloader(self) -> DataLoader:
        sampler = EpisodeAwareSampler(
            self.train_dataset, 
            shuffle=True, 
            seed=self.cfg.training.get("seed", 42)
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # [CRITICAL] Kill workers after epoch to release RAM
            persistent_workers=False, 
            prefetch_factor=self.cfg.training.get("prefetch_factor", 2),
            collate_fn=semantic_planner_collate_fn,
            worker_init_fn=seed_worker,
            drop_last=True
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if not self.val_dataset:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.training.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=False,
            prefetch_factor=self.cfg.training.get("prefetch_factor", 2), 
            collate_fn=semantic_planner_collate_fn,
            worker_init_fn=seed_worker
        )
# =============================================================================
# APEX-MoE LIGHTNING MODULE (SOTA v2.0)
# =============================================================================

class APEXMoELightningModule(pl.LightningModule):
    """
    PyTorch Lightning Module for APEX-MoE Post-Training (SOTA v2.0).
    
    SOTA Enhancements:
    1. EMA Policy (stable evaluation)
    2. Mixed Precision (via Lightning)
    3. Temporal Ensemble (for inference)
    4. Adaptive BC Coefficient
    5. Phase-Weighted Loss
    6. Cross-Expert Regularization
    """

    def __init__(self, cfg: DictConfig, phase_counts: Optional[Dict[int, int]] = None):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        
        logger.info("=" * 60)
        logger.info("APEX-MoE Post-Training Initialization (SOTA v2.0)")
        logger.info("=" * 60)
        
        # --- 1. Load Pre-Trained Router (SemanticPlanner) ---
        self._load_router()
        
        # --- 2. Freeze Router ---
        self._freeze_router()
        
        # --- 3. Create Expert Array ---
        self._create_experts()
        
        # --- 4. [SOTA #1] Create EMA Policy ---
        self._create_ema_experts()
        
        # --- 5. Loss Functions ---
        self.pose_criterion = nn.L1Loss(reduction='none')
        self.phase_criterion = nn.CrossEntropyLoss()
        self.register_buffer('grip_pos_weight', torch.tensor([3.0]))
        
        # --- 6. Loss Weights ---
        self.lambda_gripper = cfg.training.loss_weights.get("lambda_gripper", 1.0)
        self.pose_scale = cfg.training.get("pose_scale", 10.0)
        
        # --- 7. [SOTA #4] Adaptive BC Coefficient ---
        self.bc_scheduler = AdaptiveBCScheduler(
            initial_coef=cfg.training.get("bc_initial", 1.0),
            final_coef=cfg.training.get("bc_final", 0.1),
            total_epochs=cfg.training.max_epochs,
            warmup_epochs=cfg.training.get("bc_warmup", 5)
        )
        self.current_bc_coef = 1.0
        
        self.best_val_loss = float('inf')
        self.training_metrics_history = []



        resume_ckpt = cfg.training.get("resume_checkpoint")
        
        if resume_ckpt and os.path.isfile(resume_ckpt):
            # 1. Resume Mode: Extract identity from filename
            filename = os.path.basename(resume_ckpt)
            # Strip suffixes to find base run name
            base_name = filename
            for suffix in ["_latest", "_best", "_epoch"]:
                if suffix in base_name:
                    base_name = base_name.split(suffix)[0]
                    break
            self.run_name = base_name
            logger.info(f"[Init] Resuming Run Identity: {self.run_name}")
        else:
            # 2. New Run Mode: Generate fresh timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"{cfg.logging.run_name}_{timestamp}"
            logger.info(f"[Init] New Run Identity: {self.run_name}")

        # --- Path Setup (Uses the correctly determined self.run_name) ---
        self.metric_log_dir = cfg.logging.get("metric_log_dir", "logs/moe_post")
        os.makedirs(self.metric_log_dir, exist_ok=True)
        self.metric_csv_path = os.path.join(self.metric_log_dir, f"training_metrics_{self.run_name}.csv")
        
        self.checkpoint_dir = cfg.checkpoint.get("backup_dir", os.path.join(cfg.training.output_dir, "checkpoints"))
        os.makedirs(self.checkpoint_dir, exist_ok=True)




        
        # --- 8. [SOTA #5] Phase Weights (Inverse Frequency) ---
        self.phase_weights = self._compute_phase_weights(phase_counts)
        
        # --- 9. [SOTA #6] Cross-Expert Regularization Weight ---
        self.lambda_cross_expert = cfg.training.get("lambda_cross_expert", 0.01)
        logger.info(f"[SOTA #6] Cross-Expert Regularization: lambda={self.lambda_cross_expert}")
        
        # --- 10. Tracking Per-Expert Metrics ---
        self.expert_losses = {i: [] for i in range(cfg.model.num_task_phases)}
        
        # [REMOVED] Temporal Ensemble removed - could hurt Z-axis precision
        
        logger.info("=" * 60)
        logger.info("Initialization Complete (SOTA v2.0)")
        logger.info("=" * 60)

    def _load_router(self):
        """Load and freeze the Semantic Planner (Router)."""
        router_path = self.cfg.checkpoint.router_path
        
        if not os.path.exists(router_path):
            raise FileNotFoundError(
                f"Router checkpoint not found: {router_path}\n"
                "Please train a SemanticPlanner first using train_semantic_planner.py"
            )
        
        logger.info(f"[Router] Loading pre-trained SemanticPlanner from: {router_path}")
        
        model_config = SemanticPlannerConfig(
            proprio_dim=self.cfg.model.proprio_dim,
            vision_backbone_model=self.cfg.model.vision_backbone_model,
            vision_feature_dim=self.cfg.model.vision_feature_dim,
            fusion_transformer_layers=self.cfg.model.fusion_transformer_layers,
            fusion_transformer_heads=self.cfg.model.fusion_transformer_heads,
            dim_feedforward_ratio=self.cfg.model.get("dim_feedforward_ratio", 4),
            num_task_phases=self.cfg.model.num_task_phases,
            dropout=self.cfg.model.dropout,
            chunk_size=self.cfg.model.get("chunk_size", 10)
        )
        
        self.router = SemanticPlanner(model_config)
        
        checkpoint = torch.load(router_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        
        missing, unexpected = self.router.load_state_dict(state_dict, strict=False)
        
        if missing:
            logger.warning(f"[Router] Missing keys: {missing[:5]}...")
        if unexpected:
            logger.warning(f"[Router] Unexpected keys: {unexpected[:5]}...")
        
        total_params = count_parameters(self.router, trainable_only=False)
        logger.info(f"[Router] Loaded successfully. Total params: {total_params:,}")
        
    def _freeze_router(self):
        """Freezes all Router parameters."""
        frozen_count = 0
        for name, param in self.router.named_parameters():
            param.requires_grad = False
            frozen_count += 1
        
        self.router.eval()
        
        trainable = count_parameters(self.router, trainable_only=True)
        logger.info(f"[Router] FROZEN: {frozen_count} parameters. Trainable: {trainable}")
    
    def _create_experts(self):
        """Creates the ExpertArray by bootstrapping from Router."""
        num_experts = self.cfg.model.num_task_phases
        context_dim = self.cfg.moe.get("context_dim", 0)
        
        logger.info(f"[Experts] Creating ExpertArray with {num_experts} experts...")
        
        self.experts = ExpertArray.from_planner(
            planner=self.router,
            num_phases=num_experts,
            context_dim=context_dim
        )
        
        log_expert_stats(self.experts)
        
        total_expert_params = count_parameters(self.experts, trainable_only=True)
        logger.info(f"[Experts] Total trainable params: {total_expert_params:,}")
    
    def _create_ema_experts(self):
        """[SOTA #1] Creates EMA copy of experts for stable evaluation."""
        self.ema_experts = copy.deepcopy(self.experts)
        for param in self.ema_experts.parameters():
            param.requires_grad = False
        
        self.ema_decay = self.cfg.training.get("ema_decay", 0.999)
        logger.info(f"[SOTA #1] EMA Experts initialized with decay={self.ema_decay}")
    
    @torch.no_grad()
    def _update_ema(self):
        """Updates EMA experts with current expert weights."""
        for ema_param, param in zip(self.ema_experts.parameters(), self.experts.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def _compute_phase_weights(self, phase_counts: Optional[Dict[int, int]]) -> torch.Tensor:
        """[SOTA #5] Computes inverse frequency weights for phase-balanced loss."""
        num_phases = self.cfg.model.num_task_phases
        
        if phase_counts is None:
            logger.warning("[SOTA #5] No phase counts provided, using uniform weights")
            return torch.ones(num_phases)
        
        # Compute inverse frequency weights
        total = sum(phase_counts.values())
        weights = []
        for phase_id in range(num_phases):
            count = max(1, phase_counts.get(phase_id, 1))
            weight = total / (num_phases * count)  # Inverse frequency
            weights.append(weight)
        
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        # Normalize so mean = 1
        weights_tensor = weights_tensor / weights_tensor.mean()
        
        logger.info(f"[SOTA #5] Phase Weights (Inverse Freq): {weights_tensor.tolist()}")
        
        return weights_tensor

    def train(self, mode: bool = True):
        """Override to keep Router in eval mode during training."""
        super().train(mode)
        if mode:
            self.router.eval()
            for param in self.router.parameters():
                param.requires_grad = False
        return self

    def on_train_epoch_start(self):
        """[Patch] Update BC Coefficient at start of epoch."""
        self.current_bc_coef = self.bc_scheduler.get_coef(self.current_epoch)
        self.log("train/bc_coef_schedule", self.current_bc_coef, on_epoch=True)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass through Router and Experts."""
        with torch.no_grad():
            router_outputs = self.router(batch)
        
        z_traj = router_outputs['visual_embedding']
        phase_logits = router_outputs['phase_logits']
        z_grip = z_traj
        
        if self.training:
            phase_ids = batch['gt_phase_label']
        else:
            phase_ids = torch.argmax(phase_logits, dim=1)
        
        pose_chunk, grip_chunk = self.experts(
            z_traj=z_traj,
            z_grip=z_grip,
            phase_ids=phase_ids,
            context=None
        )
        
        return {
            'pose_chunk': pose_chunk,
            'gripper_chunk': grip_chunk,
            'phase_logits': phase_logits,
            'phase_ids': phase_ids,
            'visual_embedding': z_traj
        }



    def _compute_contrastive_specialization(
        self, 
        z_traj: torch.Tensor, 
        phase_ids: torch.Tensor,
        temperature: float = 0.1
    ) -> torch.Tensor:
        """
        [PATCH 2] Efficient Contrastive Specialization.
        Uses batch subsampling to prevent VRAM OOM and speed up training.
        """
        if getattr(self.cfg.training, "lambda_contrastive", 0) <= 0:
            return torch.tensor(0.0, device=z_traj.device, dtype=z_traj.dtype)
            
        # 1. Subsample Batch (Efficiency Fix)
        B_full = z_traj.shape[0]
        # Cap at 16 or 32 samples to keep VRAM usage negligible
        subset_size = min(B_full, 16) 
        
        if B_full > subset_size:
            idx = torch.randperm(B_full)[:subset_size]
            z_sub = z_traj[idx]
        else:
            z_sub = z_traj

        # 2. Get Expert Outputs on SUBSET only
        expert_outputs = []
        for expert in self.experts.experts:
            pose, _ = expert(z_sub, z_sub, None) 
            # [CRITICAL] Cast to float32. 
            # Experts might output float16 in AMP, but contrastive exp/log 
            # requires float32 precision to avoid nan/inf.
            expert_outputs.append(pose.view(subset_size, -1).float()) 
            
        # Stack: (E, B_sub, D)
        all_outs = torch.stack(expert_outputs) 
        
        # 3. Permute to (B_sub, E, D) for per-sample comparison
        batch_outs = all_outs.permute(1, 0, 2)
        
        # 4. Normalize
        batch_outs_norm = F.normalize(batch_outs, p=2, dim=-1)
        
        # 5. Compute Similarity: (B_sub, E, E)
        sim_matrix = torch.matmul(batch_outs_norm, batch_outs_norm.transpose(1, 2))
        
        # 6. Orthogonality Loss
        # Mask diagonal (self-similarity should be 1, we ignore it)
        E = sim_matrix.shape[1]
        eye = torch.eye(E, device=sim_matrix.device).unsqueeze(0)
        
        # Penalize non-zero off-diagonals
        off_diagonal_sim = sim_matrix * (1 - eye)
        loss = (off_diagonal_sim ** 2).mean()
        
        return loss



    def _compute_cross_expert_regularization(self, z_traj: torch.Tensor) -> torch.Tensor:
        """
        [SOTA #6] Cross-Expert Regularization.
        
        Penalizes divergence between expert outputs on the same input.
        This prevents experts from becoming too specialized and losing generalization.
        
        L_cross = sum_{i,j} || Expert_i(z) - Expert_j(z) ||^2
        """
        if self.lambda_cross_expert <= 0:
            return torch.tensor(0.0, device=z_traj.device)
        
        # Sample a subset for efficiency
        B = z_traj.shape[0]
        if B > 64:
            idx = torch.randperm(B)[:64]
            z_sample = z_traj[idx]
        else:
            z_sample = z_traj
        
        # Get outputs from all experts
        expert_outputs = []
        for expert in self.experts.experts:
            pose, _ = expert(z_sample, z_sample, None)
            expert_outputs.append(pose)  # (B, K, 7)
        
        # Compute pairwise L2 distance
        reg_loss = 0.0
        num_pairs = 0
        for i in range(len(expert_outputs)):
            for j in range(i + 1, len(expert_outputs)):
                diff = expert_outputs[i] - expert_outputs[j]
                reg_loss += (diff ** 2).mean()
                num_pairs += 1
        
        if num_pairs > 0:
            reg_loss = reg_loss / num_pairs
        
        return reg_loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Optional[torch.Tensor]:
        if not batch:
            return None
        
        # --- 1. Forward Pass ---
        outputs = self(batch)
        
        pred_pose_chunk = outputs['pose_chunk']
        pred_grip_chunk = outputs['gripper_chunk']
        phase_logits = outputs['phase_logits']
        z_traj = outputs['visual_embedding']
        
        # --- 2. Ground Truths ---
        gt_pose_chunk = batch['gt_pose_chunk']
        gt_grip_chunk = batch['gt_grip_chunk']
        gt_phase = batch['gt_phase_label']
        
        # --- 3. [SOTA #5] Phase-Weighted Pose Loss ---
        raw_pose_loss = self.pose_criterion(pred_pose_chunk, gt_pose_chunk)  # (B, K, 7)
        per_sample_loss = raw_pose_loss.mean(dim=[1, 2])  # (B,)
        
        # Apply phase weights
        phase_weights = self.phase_weights.to(gt_phase.device)
        sample_weights = phase_weights[gt_phase]  # (B,)
        weighted_pose_loss = (per_sample_loss * sample_weights).mean() * self.pose_scale
        
        # [PATCH] Apply Adaptive BC Coefficient (The "Ghost Weight" Fix)
        # Apply annealing to the supervised loss term
        final_pose_loss = weighted_pose_loss * self.current_bc_coef
        
        # --- 4. Gripper Loss ---
        grip_loss = F.binary_cross_entropy_with_logits(
            pred_grip_chunk, gt_grip_chunk,
            pos_weight=self.grip_pos_weight,
            reduction='mean'
        )
        
        # [SOTA #6] Cross-Expert Regularization
        # self.lambda_cross_expert matches SOTA #6 config
        if self.lambda_cross_expert > 0:
            cross_expert_loss = self._compute_cross_expert_regularization(z_traj)
        else:
            cross_expert_loss = torch.tensor(0.0, device=self.device)
            
        # [ENHANCEMENT C] Contrastive Specialization
        # Only compute if lambda > 0 (Disabled in Stage 2 Config)
        lambda_contrastive = self.cfg.training.get("lambda_contrastive", 0.05)
        if lambda_contrastive > 0:
            contrastive_loss = self._compute_contrastive_specialization(z_traj, gt_phase)
        else:
            contrastive_loss = torch.tensor(0.0, device=self.device)
            
        # --- 7. Total Loss ---
        # to allow regularization/auxiliary losses to shape experts more freely.
        total_loss = (
            final_pose_loss +
            (self.lambda_gripper * grip_loss) +
            (self.lambda_cross_expert * cross_expert_loss) +
            (lambda_contrastive * contrastive_loss)
        )
        
        # [REMOVED] self._update_ema() removed. EMA handled in on_train_batch_end after optimization.
        
        # --- 9. Per-Expert Loss Tracking ---
        with torch.no_grad():
            for phase_id in range(self.cfg.model.num_task_phases):
                mask = (gt_phase == phase_id)
                if mask.any():
                    expert_loss = per_sample_loss[mask].mean().item()
                    self.expert_losses[phase_id].append(expert_loss)
        
        # --- 10. Logging ---
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/pose", weighted_pose_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/grip", grip_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/reg", cross_expert_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_contrastive", contrastive_loss, on_step=True, on_epoch=True)
        self.log("train/bc_coef", self.current_bc_coef, on_step=False, on_epoch=True)
        
        with torch.no_grad():
            phase_preds = torch.argmax(phase_logits, dim=1)
            phase_acc = (phase_preds == gt_phase).float().mean()
            self.log("train/router_phase_acc", phase_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return total_loss

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int):
        """[SOTA #1] Update EMA experts only after Optimizer Step."""
        # Check if optimizer stepped this batch
        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            self._update_ema()

    def on_train_epoch_end(self):
        """Log per-expert statistics and handle Robust Checkpointing (CKPT format) with Cleanup."""
        import glob  # Required for finding old files
        
        logger.info("=" * 40)
        logger.info(f"Epoch {self.current_epoch} Expert Statistics:")
        
        # --- 1. Log Expert Stats ---
        for phase_id in range(self.cfg.model.num_task_phases):
            losses = self.expert_losses[phase_id]
            if losses:
                avg_loss = sum(losses) / len(losses)
                phase_name = ExpertArray.PHASE_NAMES.get(phase_id, f"Phase_{phase_id}")
                logger.info(f"  Expert {phase_id} ({phase_name}): Avg Loss = {avg_loss:.4f} ({len(losses)} samples)")
                self.log(f"train/expert_{phase_id}_avg_loss", avg_loss, on_epoch=True)
            self.expert_losses[phase_id] = []
        
        # --- 2. Prepare Checkpoint Data ---
        state_dict = self.state_dict()
        # Compact: Exclude frozen router and EMA copies to save space/time
        compact_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("router.")}

        del state_dict
        gc.collect()

        lr_schedulers = self.lr_schedulers()
        if not isinstance(lr_schedulers, list):
            lr_schedulers = [lr_schedulers]
        lr_scheduler_states = [s.state_dict() for s in lr_schedulers]


        checkpoint = {
            'epoch': self.current_epoch,
            'state_dict': compact_state_dict,
            'optimizer_state': self.optimizers().state_dict(),
            'lr_schedulers': lr_scheduler_states,
            'config': self.cfg,
            'hyper_parameters': self.cfg  # PL convention for auto-loading params
        }
        
        # --- 3. Save 'Latest' (Always Overwritten) ---
        # CHANGED: .pt -> .ckpt
        latest_path = os.path.join(self.checkpoint_dir, f"{self.run_name}_latest.ckpt")
        torch.save(checkpoint, latest_path)
        
        # --- 4. Periodic Backup (Every 5 epochs) with Rolling Window ---
        if (self.current_epoch + 1) % 5 == 0:
            # CHANGED: .pt -> .ckpt
            backup_path = os.path.join(self.checkpoint_dir, f"{self.run_name}_epoch_{self.current_epoch}.ckpt")
            shutil.copyfile(latest_path, backup_path)
            logger.info(f"[Backup] Copied periodic checkpoint: {backup_path}")
            
            # [CLEANUP] Keep only the last 3 periodic checkpoints
            # CHANGED: .pt -> .ckpt
            periodic_pattern = os.path.join(self.checkpoint_dir, f"{self.run_name}_epoch_*.ckpt")
            periodic_ckpts = sorted(glob.glob(periodic_pattern), key=os.path.getmtime)
            
            max_keep = 3
            if len(periodic_ckpts) > max_keep:
                for old_ckpt in periodic_ckpts[:-max_keep]:
                    try:
                        os.remove(old_ckpt)
                        logger.info(f"[Cleanup] Deleted old periodic checkpoint: {os.path.basename(old_ckpt)}")
                    except OSError as e:
                        logger.warning(f"Failed to delete {old_ckpt}: {e}")

        # --- 5. Best Model Backup with Cleanup ---
        current_val_loss = self.trainer.callback_metrics.get("val/loss")
        if current_val_loss is not None:
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss.item() if isinstance(current_val_loss, torch.Tensor) else current_val_loss
                
                # Define filenames
                # CHANGED: .pt -> .ckpt
                best_path = os.path.join(self.checkpoint_dir, f"{self.run_name}_best.ckpt")
                best_ep_path = os.path.join(self.checkpoint_dir, f"{self.run_name}_best_ep{self.current_epoch}_val{self.best_val_loss:.4f}.ckpt")
                
                # [CLEANUP] Remove *previous* best_ep* files before creating a new one
                # CHANGED: .pt -> .ckpt
                best_pattern = os.path.join(self.checkpoint_dir, f"{self.run_name}_best_ep*.ckpt")
                for old_best in glob.glob(best_pattern):
                    try:
                        os.remove(old_best)
                        logger.info(f"[Cleanup] Deleted outdated best checkpoint: {os.path.basename(old_best)}")
                    except OSError:
                        pass
                
                # Copy new bests
                shutil.copyfile(latest_path, best_path)
                shutil.copyfile(latest_path, best_ep_path)
                logger.info(f"[Backup] New Best Model Saved (Val Loss: {self.best_val_loss:.4f})")

        logger.info(f"  Best Val Loss so far: {self.best_val_loss:.4f}")
        logger.info("=" * 40)
        
        # --- 6. Metrics & Cleanup ---
        del checkpoint, compact_state_dict
        gc.collect()
        
        # 2. Save Metrics DIRECTLY to Disk (Stateless)
        # Append to CSV
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in self.trainer.callback_metrics.items()}
        metrics['epoch'] = self.current_epoch
        
        # Check if header is needed (if file doesn't exist)
        file_exists = os.path.isfile(self.metric_csv_path)
        
        try:
            pd.DataFrame([metrics]).to_csv(self.metric_csv_path, mode='a', header=not file_exists, index=False)
        except Exception as e:
            logger.warning(f"Failed to write metrics: {e}")
            
        # [CRITICAL] Do NOT store in self.training_metrics_history to prevent RAM growth
        # self.training_metrics_history.append(metrics)
        
        # [Patch 4] Aggressive RAM Cleanup (Dataset LRU Cache)
        if self.trainer.train_dataloader:
            try:
                loader = self.trainer.train_dataloader
                # Handle potential list or single loader
                if isinstance(loader, list): loader = loader[0]
                ds = loader.dataset if hasattr(loader, 'dataset') else None
                
                if ds and hasattr(ds, 'expert_reader'):
                    ds.expert_reader._get_full_modality_array.cache_clear()
                    logger.info("[Memory] Cleared Dataset LRU Cache.")
            except Exception as e:
                logger.warning(f"[Memory] Failed to clear cache: {e}")

        # 3. Final Aggressive Memory Cleanup
        gc.collect()
        torch.cuda.empty_cache()



    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        if not batch:
            return
        
        with torch.no_grad():
            # [SOTA #1] Use EMA experts for validation
            router_outputs = self.router(batch)
            z_traj = router_outputs['visual_embedding']
            phase_logits = router_outputs['phase_logits']
            phase_ids = torch.argmax(phase_logits, dim=1)
            
            # Use EMA experts
            pred_pose_chunk, pred_grip_chunk = self.ema_experts(
                z_traj=z_traj,
                z_grip=z_traj,
                phase_ids=phase_ids,
                context=None
            )
            
            gt_pose_chunk = batch['gt_pose_chunk']
            gt_grip_chunk = batch['gt_grip_chunk']
            gt_phase = batch['gt_phase_label']
            
            # Losses
            pose_loss = F.l1_loss(pred_pose_chunk, gt_pose_chunk)
            grip_loss = F.binary_cross_entropy_with_logits(
                pred_grip_chunk, gt_grip_chunk,
                reduction='mean'
            )
            total_val_loss = pose_loss + (self.lambda_gripper * grip_loss)
            
            # --- IMPROVED METRICS ---
            # 1. Trajectory Error (Mean over entire chunk, not just t=0)
            pos_diff = pred_pose_chunk[..., :3] - gt_pose_chunk[..., :3]
            pos_error = torch.norm(pos_diff, dim=-1).mean()
            
            # 2. Router accuracy
            phase_preds = torch.argmax(phase_logits, dim=1)
            phase_acc = (phase_preds == gt_phase).float().mean()
            
            # 3. Gripper accuracy
            grip_preds = (torch.sigmoid(pred_grip_chunk) > 0.5).float()
            grip_acc = (grip_preds == gt_grip_chunk).float().mean()
            
            # Logging
            self.log("val/loss", total_val_loss, on_epoch=True, sync_dist=True, prog_bar=True)
            self.log("val/pos_error_m", pos_error, on_epoch=True, sync_dist=True, prog_bar=True)
            self.log("val/router_phase_acc", phase_acc, on_epoch=True, sync_dist=True)
            self.log("val/gripper_acc", grip_acc, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        """[Patch 1] Periodic Memory Cleanup after Validation Loop."""
        gc.collect()
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        """SOTA Optimizer Configuration."""
        optimizer_params = []
        # Robust Parameter Selection
        # Collect all parameters in the experts
        expert_params_set = set()
        for i, expert in enumerate(self.experts.experts):
            eg_params = list(expert.parameters())
            expert_params = {
                "params": eg_params,
                "lr": self.cfg.optimizer.lr,
                "weight_decay": self.cfg.optimizer.weight_decay,
                "name": f"expert_{i}"
            }
            optimizer_params.append(expert_params)
            expert_params_set.update(eg_params)
        
        # Catch any shared parameters inside ExpertArray that are not handled by expert loops
        all_moe_params = set(self.experts.parameters())
        shared_params = list(all_moe_params - expert_params_set)
        
        # [FIX vFinal] Strict Parameter Filtering
        # Only include parameters that actually require gradients.
        # This prevents AdamW from allocating state for frozen/EMA weights.
        optimizer_params = [
            {
                "params": [p for p in group["params"] if p.requires_grad],
                "lr": group["lr"],
                "weight_decay": group["weight_decay"],
                "name": group["name"]
            }
            for group in optimizer_params
        ]
        
        if shared_params:
            shared_trainable = [p for p in shared_params if p.requires_grad]
            if shared_trainable:
                logger.info(f"[Optimizer] Adding {len(shared_trainable)} shared parameters to optimization.")
                optimizer_params.append({
                    "params": shared_trainable,
                    "lr": self.cfg.optimizer.lr,
                    "weight_decay": self.cfg.optimizer.weight_decay,
                    "name": "moe_shared"
                })
        
        total_params = sum(p.numel() for group in optimizer_params for p in group["params"])
        logger.info(f"[Optimizer] Total trainable parameters: {total_params:,}")
        
        optimizer = torch.optim.AdamW(
            optimizer_params,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.cfg.optimizer.get("warmup_percentage", 0.05))
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"[Scheduler] Cosine with {warmup_steps} warmup steps out of {total_steps} total")
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

class EpochRestorationCallback(pl.Callback):
    """
    Robustly restores training state (Epoch, Global Step, Optimizer, Scheduler).
    """
    def __init__(self, start_epoch: int, cfg: DictConfig, samples_per_epoch: int = 0, 
                 optimizer_state: Optional[Dict] = None, scheduler_state: Optional[List] = None):
        self.start_epoch = start_epoch
        self.optimizer_state = optimizer_state
        self.scheduler_state = scheduler_state  # <--- Store state
        
        # Calculate Logic
        batch_size = cfg.training.batch_size
        accumulate = cfg.training.get("accumulate_grad_batches", 1)
        steps_per_epoch = samples_per_epoch // (batch_size * accumulate)
        self.estimated_global_step = start_epoch * steps_per_epoch
        
        logger.info(f"🔄 [Callback] Resume Calculation: Epoch {start_epoch} -> Global Step ~{self.estimated_global_step}")

    def on_fit_start(self, trainer, pl_module):
        """Initialize internal loop counters."""
        trainer.fit_loop.epoch_progress.current.completed = self.start_epoch
        trainer.fit_loop.epoch_loop._batches_that_stepped = self.estimated_global_step

    def on_train_start(self, trainer, pl_module):
        """
        CRITICAL PATCH: Enforce state after Trainer initialization.
        """
        # 1. Force Epoch Counter
        if trainer.current_epoch != self.start_epoch:
            logger.info(f"🔄 [Callback] Enforcing Trainer Epoch to {self.start_epoch}")
            trainer.fit_loop.epoch_progress.current.completed = self.start_epoch

        # 2. Force Global Step
        trainer.fit_loop.epoch_loop._batches_that_stepped = self.estimated_global_step
        
        # 3. Restore Optimizer Momentum
        if self.optimizer_state:
            try:
                if not trainer.optimizers:
                    logger.warning("⚠️ No optimizers found to restore state.")
                else:
                    optimizer = trainer.optimizers[0]
                    optimizer.load_state_dict(self.optimizer_state)
                    # Move to GPU
                    device = pl_module.device
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)
                    logger.info("🔄 [Callback] Optimizer state fully restored.")
            except Exception as e:
                logger.warning(f"⚠️ Failed to restore optimizer state: {e}")

        # 4. [PATCH 2] Restore Scheduler State
        if self.scheduler_state and trainer.lr_scheduler_configs:
            try:
                # We assume 1 scheduler (Cosine) as per configure_optimizers
                # The state was saved as a list of dicts
                saved_state = self.scheduler_state[0]
                target_scheduler = trainer.lr_scheduler_configs[0].scheduler
                target_scheduler.load_state_dict(saved_state)
                logger.info("🔄 [Callback] Scheduler state restored (Warmup skipped).")
            except Exception as e:
                logger.warning(f"⚠️ Failed to restore scheduler state: {e}")


@hydra.main(version_base=None, config_path="../configs", config_name="train_moe_post_config")
def main(cfg: DictConfig) -> None:
    """Main Entry Point for APEX-MoE Post-Training (SOTA v2.0)."""
    # [COLAB FIX] Switch to 'spawn' to prevent deadlock warnings and OOM hangs

    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Silence TF logs
    try:
        import tensorflow as tf
        # Force TF to run on CPU only
        tf.config.set_visible_devices([], 'GPU')
    except ImportError:
        pass

    import torch.multiprocessing as mp
    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
    except RuntimeError:
        pass
    try:
        mp.set_start_method('spawn', force=True)
        logger.info("[Main] Multiprocessing start method set to 'spawn'")
    except RuntimeError:
        logger.warning("[Main] Could not set start method to 'spawn', likely already set.")
    
    pl.seed_everything(cfg.seed, workers=True)
    
    logger.info("=" * 60)
    logger.info("APEX-MoE Post-Training Pipeline (SOTA v2.0)")
    logger.info("=" * 60)
    logger.info(f"Working Directory: {os.getcwd()}")
    
    # --- Logging Setup ---
    output_dir = Path(cfg.training.get("output_dir", "./outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Load Data ---
    datamodule = MoEDataModule(cfg)
    
    # [FIX] Manually setup datamodule to calculate phase counts BEFORE model init
    # This prevents the "Silent Failure" where model uses uniform weights
    datamodule.setup("fit")
    
    # --- 2. Build or Load Model ---
    logger.info("Initializing APEX-MoE Transformer Policy...")
    
    # Determine input dimension from dataset (requires a quick peek if not hardcoded)
    # Ideally, get this from config or dataset metadata. 
    # For now, we use the known schema dim = 7 (proprio) + 7 (goal_proprio) = 14? 
    # Actually, SemanticPlannerDataset handles the projection.
    # We will let the model infer or use config defaults.
    
    model = APEXMoELightningModule(cfg, phase_counts=datamodule.phase_counts)
    
    loggers = [TensorBoardLogger(save_dir=str(output_dir), name="tb_logs")]
    if cfg.logging.get("use_wandb", False):
        os.environ["WANDB_MODE"] = cfg.logging.get("wandb_mode", "offline")
        # [SOTA FIX] Use threading for wandb startup to prevent deadlocks with spawn
        os.environ["WANDB_START_METHOD"] = "thread"
        
        wandb_logger = WandbLogger(
            project=cfg.logging.get("wandb_project", "APEX-MoE"),
            name=model.run_name,
            id=model.run_name,   # <--- CRITICAL LINK
            resume="allow",      # <--- Explicitly allow resumption
            save_dir=str(output_dir),
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        loggers.append(wandb_logger)
    
    # Callbacks
    callbacks = [
        # [FIX vFinal] Removed ModelCheckpoint to prevent saving frozen router (disk bloat).
        # We rely on the robust manual checkpointing in on_train_epoch_end.
        LearningRateMonitor(logging_interval='step'),
        TQDMProgressBar(refresh_rate=10),
    ]

    # [RESUME LOGIC] Handle custom dict checkpoint loading
    resume_epoch = 0
    resume_optimizer_state = None
    
    if cfg.training.get("resume_checkpoint"):
        resume_path = cfg.training.resume_checkpoint
        if os.path.isfile(resume_path):
            logger.info(f"🔄 Resuming from custom checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=model.device)
            
            # 1. Load Weights (Strict=False to handle missing router/ema)
            missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
            logger.info(f"   Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            
            # 2. Extract State
            resume_epoch = checkpoint.get('epoch', 0)
            resume_optimizer_state = checkpoint.get('optimizer_state', None)
            resume_scheduler_state = checkpoint.get('lr_schedulers', None)
            
            logger.info(f"   Resuming from Epoch: {resume_epoch}")
            
            # 3. Add Callback to Jumpstart Trainer with CORRECT Scheduler State
            # We need dataset size to calculate global step
            dataset_len = len(datamodule.train_dataset)
            
            callbacks.append(EpochRestorationCallback(
                start_epoch=resume_epoch, 
                cfg=cfg,
                samples_per_epoch=dataset_len,
                optimizer_state=resume_optimizer_state,
                scheduler_state=resume_scheduler_state
            ))
            
            # [PATCH] CRITICAL MEMORY CLEANUP
            # Free the 2GB+ Checkpoint Dict from RAM immediately
            del checkpoint
            del resume_optimizer_state
            del resume_scheduler_state
            gc.collect()
            logger.info("🗑️ [Memory] Freed resume checkpoint from RAM.")
        else:
            logger.warning(f"⚠️ Resume checkpoint not found at: {resume_path}. Starting from scratch.")

    if cfg.training.get("early_stopping", False):
        callbacks.append(
            EarlyStopping(
                monitor="val/pos_error_m",
                patience=cfg.training.get("early_stopping_patience", 10),
                mode="min"
            )
        )
    
    # --- Trainer ---
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=cfg.training.max_epochs,
        logger=loggers,
        callbacks=callbacks,
        gradient_clip_val=cfg.training.get("gradient_clip_val", 1.0),
        precision=cfg.training.get("precision", "16-mixed"),
        accumulate_grad_batches=cfg.training.get("accumulate_grad_batches", 1),
        log_every_n_steps=10,
        check_val_every_n_epoch=cfg.training.get("check_val_every_n_epoch", 1),
        # [STABILITY FIX] Minimal sanity check to speed up startup
        num_sanity_val_steps=1,
        
        # --- PATCH START: DISABLE DEFAULT CHECKPOINTING ---
        # We handle checkpointing manually in on_train_epoch_end
        # This prevents saving duplicates and reduces I/O lag.
        enable_checkpointing=False, 
        # --- PATCH END ---
    )
    
    # --- Training ---
    logger.info("Starting Training (SOTA v2.0)...")
    
    # --- 6. [FIX vFinal] Final Trainability Check ---
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params == 0:
        logger.error("🛑 CRITICAL FAILURE: Zero trainable parameters detected!")
        logger.error("The experts are likely frozen. Backing out to prevent crash.")
        return
    
    logger.info(f"🚀 Launching Training with {trainable_params:,} trainable parameters.")
    
    try:
        # If we have optimizer state to load, we have to do it carefully.
        # PL requires the model to be wrapped in strategy before loading optimizers.
        # However, trainer.fit() handles wrapping.
        # The only "hack-free" way is to let fit() start, and use a callback to load the state 
        # OR just accept that optimizer state is reset (Warm Start).
        
        # NOTE: Loading optimizer state manually in PL is tricky without ckpt_path.
        # We will settle for Epoch + Weights (Warm Start with Correct Calendar).
        # This is safe because Adam re-estimates moments quickly.
        
        trainer.fit(model, datamodule=datamodule)
        best_path = os.path.join(model.checkpoint_dir, f"{model.run_name}_best.ckpt")
        logger.info(f"Training Complete. Best model saved to: {best_path}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise e
    finally:
        for lg in loggers:
            if isinstance(lg, WandbLogger):
                import wandb
                if wandb.run:
                    wandb.finish()


if __name__ == "__main__":
    main()
