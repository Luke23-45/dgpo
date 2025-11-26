# FILE: train/train_semantic_planner.py
# (Definitive, SOTA, Production-Grade Implementation)

r""" Training Script for the Advantage-Weighted Semantic Planner (AWSP).

This script implements the **Advantage-Weighted Regression (AWR)** training loop
for the hierarchical 'Strategist' model. Unlike standard Behavior Cloning, this
pipeline weights gradients by the pre-calculated **Advantage** ($A(s,a)$),
steering the planner towards subgoals that lead to higher long-term value.

Methodology:
1.  **Goal-Conditioned Regression**: The model predicts $(s_{next}, g_{action})$ given $(s_{current}, s_{goal}, \text{phase})$.
2.  **AWR Loss**: The loss for each sample is scaled by $w = \exp(A / \tau)$.
    This mathematically aligns the policy with the Boltzmann distribution of the
    optimal value function (Peters & Schaal, 2007; Peng et al., 2019).
3.  **Robust Optimization**: Uses AdamW with Cosine Annealing and strict
    parameter grouping (weight decay exclusion for norms/biases).
4.  **Comprehensive Metrics**: Tracks Euclidean Error, Geodesic Rotation Error,
    and Gripper Accuracy for holistic performance monitoring.

Architecture:
- **DataModule**: Manages `SemanticPlannerDataset` with robust caching.
- **LightningModule**: Encapsulates AWR logic, metrics, and optimization.
- **Hydra**: Manages hierarchical configuration.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import math
import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from utils.samplers import EpisodeAwareSampler
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

# --- Project-Specific Imports ---
# Robust path handling to ensure utils/models are importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.semantic_planner import SemanticPlanner, SemanticPlannerConfig
from utils.semantic_planner_dataset import (
    SemanticPlannerDataset,
    semantic_planner_collate_fn,
)

# Initialize logger
logger = logging.getLogger("train_semantic_planner")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


# ==============================================================================
# 1. SOTA DATA MODULE
# ==============================================================================

class SemanticPlannerDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the Semantic Planner.
    
    Handles dataset instantiation and DataLoader configuration with performance
    optimizations for high-throughput training (pin_memory, persistent_workers).
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train_dataset: Optional[SemanticPlannerDataset] = None
        self.val_dataset: Optional[SemanticPlannerDataset] = None
        
        # Loader optimizations
        self.num_workers = cfg.dataset.get("num_workers", 4)
        self.pin_memory = torch.cuda.is_available()
        # Only use persistent workers if we have actual workers to avoid obscure DataLoader errors
        self.persistent_workers = self.num_workers > 0



    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            logger.info(f"Loading Training Dataset from: {self.cfg.dataset.train_path}")
            self.train_dataset = SemanticPlannerDataset(
                dataset_path=self.cfg.dataset.train_path,
                use_aug=self.cfg.dataset.get("use_aug", True),
                # --- NEW: Pass v9.0 Params ---
                chunk_size=self.cfg.model.get("chunk_size", 10),
                proprio_noise=self.cfg.dataset.get("proprio_noise", 0.005)
            )

            if self.cfg.dataset.get("val_path"):
                logger.info(f"Loading Validation Dataset from: {self.cfg.dataset.val_path}")
                self.val_dataset = SemanticPlannerDataset(
                    dataset_path=self.cfg.dataset.val_path,
                    use_aug=False,
                    chunk_size=self.cfg.model.get("chunk_size", 10),
                    proprio_noise=0.0 # No noise for validation
                )

    def train_dataloader(self) -> DataLoader:
        sampler = EpisodeAwareSampler(
            self.train_dataset, 
            shuffle=True, 
            seed=self.cfg.seed
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,  # <--- MUST be False when using a custom sampler
            sampler=sampler, # <--- Inject the SOTA sampler
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=semantic_planner_collate_fn,
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
            persistent_workers=self.persistent_workers,
            collate_fn=semantic_planner_collate_fn
        )


# ==============================================================================
# 2. LIGHTNING MODULE (AWR Logic)
# ==============================================================================

class SemanticPlannerLightningModule(pl.LightningModule):
    """
    LightningModule implementing Advantage-Weighted Regression (AWR).

    Metrics:
        - Loss (Weighted vs Unweighted)
        - Position Error (L2 meters)
        - Rotation Error (Geodesic degrees)
        - Gripper Accuracy
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        # 1. Strict Configuration Construction (v9.0 Update)
        model_config = SemanticPlannerConfig(
            proprio_dim=cfg.model.proprio_dim,
            vision_backbone_model=cfg.model.vision_backbone_model,
            vision_feature_dim=cfg.model.vision_feature_dim,
            fusion_transformer_layers=cfg.model.fusion_transformer_layers,
            fusion_transformer_heads=cfg.model.fusion_transformer_heads,
            dim_feedforward_ratio=cfg.model.get("dim_feedforward_ratio", 4),
            num_task_phases=cfg.model.num_task_phases,
            dropout=cfg.model.dropout,
            # --- FIX: Removed phase_dropout_prob line to prevent TypeError ---
            chunk_size=cfg.model.get("chunk_size", 10)
        )
        
        # 2. Model Instantiation
        self.model = SemanticPlanner(model_config)

        # 3. Training Hyperparameters
        self.awr_temperature = cfg.training.awr_temperature
        self.awr_max_weight = cfg.training.get("awr_max_weight", 20.0)
        self.lambda_gripper = cfg.training.loss_weights.lambda_gripper
        # New Weight for Auxiliary Phase Loss
        self.lambda_phase = cfg.training.loss_weights.get("lambda_phase", 0.1)

        # 4. Loss Functions
        self.pose_criterion = nn.L1Loss(reduction='none') 
        self.phase_criterion = nn.CrossEntropyLoss()

        self.register_buffer('grip_pos_weight', torch.tensor([3.0]))

    def _compute_awr_weights(self, advantages: torch.Tensor) -> torch.Tensor:
        """
        Computes normalized importance weights based on advantages.
        Formula: w = clamp( exp(A / tau), max=max_weight )
        """
        with torch.no_grad():
            # Scale advantages
            scaled_adv = advantages / self.awr_temperature
            
            # SOTA OPTIMIZATION: Use pure Python math for scalar constants to avoid 
            # unnecessary Tensor creation and CPU-GPU synchronization overhead.
            # ln(20) is approx 3.0. 
            max_exponent = math.log(self.awr_max_weight) + 2.0 
            
            # Clamp using float values (safe and fast for GPU tensors)
            scaled_adv = torch.clamp(scaled_adv, max=max_exponent)
            
            weights = torch.exp(scaled_adv)
            weights = torch.clamp(weights, max=self.awr_max_weight)
            return weights

    def _compute_geodesic_loss(self, pred_quat: torch.Tensor, gt_quat: torch.Tensor) -> torch.Tensor:
        """
        Calculates the angular distance between two unit quaternions in degrees.
        Formula: theta = 2 * arccos(|<q1, q2>|)
        """
        # Dot product of quaternions
        dot_product = torch.sum(pred_quat * gt_quat, dim=-1).abs()
        # Numerical stability clamp (arccos requires [-1, 1])
        dot_product = torch.clamp(dot_product, -1.0 + 1e-6, 1.0 - 1e-6)
        angle_rad = 2 * torch.acos(dot_product)
        return torch.rad2deg(angle_rad).mean()



    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Optional[torch.Tensor]:
        if not batch: return None

        # 1. Forward Pass
        # Inputs: prev_image, curr_image, goal_image, curr_proprio
        outputs = self.model(batch)
        
        pred_pose_chunk = outputs['pose_chunk']       # (B, K, 7)
        pred_grip_chunk = outputs['gripper_chunk']    # (B, K, 1)
        pred_phase_logits = outputs['phase_logits']   # (B, N_Phases)

        # 2. Ground Truths
        gt_pose_chunk = batch['gt_pose_chunk']        # (B, K, 7)
        gt_grip_chunk = batch['gt_grip_chunk']        # (B, K, 1)
        gt_phase = batch['gt_phase_label']            # (B,)
        advantage = batch['advantage']                # (B, 1)

        # 3. AWR Weights (Based on Advantage)
        # Broadcast weights to match chunk dimension: (B, 1, 1)
        weights = self._compute_awr_weights(advantage.squeeze(-1))
        weights_expanded = weights.unsqueeze(1) 

        # 4. Compute Trajectory Losses (Advantage Weighted)
        
        # A. Pose Loss (L1)
        # Calculate raw L1 per element, then mean over (K, 7) dimensions
        raw_pose_loss = self.pose_criterion(pred_pose_chunk, gt_pose_chunk) # (B, K, 7)
        # Weighted mean over batch, standard mean over trajectory

        POSE_SCALE = 10.0 
        
        pose_loss = (raw_pose_loss.mean(dim=[1, 2]) * weights.squeeze()).mean() * POSE_SCALE

        # B. Gripper Loss (BCE)
        # Reshape for BCE: (B*K, 1)
        B, K, _ = pred_grip_chunk.shape
        raw_grip_loss = F.binary_cross_entropy_with_logits(
            pred_grip_chunk, gt_grip_chunk, 
            pos_weight=self.grip_pos_weight, 
            reduction='none'
        ) # (B, K, 1)
        # Average over chunk, weight by advantage
        grip_loss = (raw_grip_loss.mean(dim=[1, 2]) * weights.squeeze()).mean()

        # 5. Compute Phase Loss (Auxiliary / Supervised)
        # NOTE: Phase loss is NOT weighted by advantage. It is ground truth.
        phase_loss = self.phase_criterion(pred_phase_logits, gt_phase)

        # 6. Total Loss
        total_loss = pose_loss + (self.lambda_gripper * grip_loss) + (self.lambda_phase * phase_loss)

        # 7. Logging
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)

        self.log("train/loss_pose", pose_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_grip", grip_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_phase", phase_loss, on_step=True, on_epoch=True, prog_bar=True)

        self.log("train/weights_mean", weights.mean(), on_step=False, on_epoch=True)
        
        # Calculate Phase Accuracy
        phase_preds = torch.argmax(pred_phase_logits, dim=1)
        phase_acc = (phase_preds == gt_phase).float().mean()
        self.log("train/phase_acc", phase_acc, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss




# [IN CLASS SemanticPlannerLightningModule]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        if not batch: return

        with torch.no_grad():
            outputs = self.model(batch)
            pred_pose_chunk = outputs['pose_chunk']
            pred_grip_chunk = outputs['gripper_chunk']
            pred_phase_logits = outputs['phase_logits']

            gt_pose_chunk = batch['gt_pose_chunk']
            gt_grip_chunk = batch['gt_grip_chunk']
            gt_phase = batch['gt_phase_label']

            # 1. Unweighted Losses (Standard Validation)
            loss_pose = F.l1_loss(pred_pose_chunk, gt_pose_chunk)
            loss_grip = F.binary_cross_entropy_with_logits(pred_grip_chunk, gt_grip_chunk)
            loss_phase = self.phase_criterion(pred_phase_logits, gt_phase)
            
            total_val_loss = loss_pose + (self.lambda_gripper * loss_grip) + (self.lambda_phase * loss_phase)

            # 2. Physical Metrics (Evaluate first step of chunk for immediate accuracy)
            # We evaluate the immediate next action (t+1)
            pred_next_pose = pred_pose_chunk[:, 0, :]
            gt_next_pose = gt_pose_chunk[:, 0, :]
            
            pos_error_m = torch.norm(pred_next_pose[:, :3] - gt_next_pose[:, :3], dim=-1).mean()
            rot_error_deg = self._compute_geodesic_loss(pred_next_pose[:, 3:], gt_next_pose[:, 3:])
            
            # Gripper Accuracy (Whole Chunk)
            pred_cls = (torch.sigmoid(pred_grip_chunk) > 0.5).float()
            grip_acc = (pred_cls == gt_grip_chunk).float().mean()

            # Phase Accuracy
            phase_preds = torch.argmax(pred_phase_logits, dim=1)
            phase_acc = (phase_preds == gt_phase).float().mean()

            # 3. Logging
            self.log("val/loss", total_val_loss, on_epoch=True, sync_dist=True, prog_bar=True)
            self.log("val/pos_error_m", pos_error_m, on_epoch=True, sync_dist=True, prog_bar=True)
            self.log("val/rot_error_deg", rot_error_deg, on_epoch=True, sync_dist=True)
            self.log("val/gripper_acc", grip_acc, on_epoch=True, sync_dist=True)
            self.log("val/phase_acc", phase_acc, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        """
        SOTA Optimizer Configuration: Differential Learning Rates & Intelligent Weight Decay.
        
        Architecture-Aware Logic:
        1. **Parameter Groups**:
           - **Backbone** (SigLIP): Low LR (0.1x) to preserve pre-trained features.
           - **Head** (Planner): Base LR (1.0x) for rapid task adaptation.
        2. **Regularization Hygiene**:
           - **Decay**: applied ONLY to matrix multiplications (Linear, Conv, Attention weights).
           - **No Decay**: applied to Biases, LayerNorms, Embeddings, and 1D Vectors.
        """
        # 1. Define Targets
        # Use a lower learning rate for the pre-trained backbone to prevent catastrophic forgetting.
        base_lr = self.cfg.optimizer.lr
        backbone_lr = base_lr * 0.1
        weight_decay = self.cfg.optimizer.weight_decay

        # 2. Initialize Parameter Buckets
        # Format: (Backbone/Head) x (Decay/NoDecay)
        backbone_decay = []
        backbone_no_decay = []
        head_decay = []
        head_no_decay = []

        # 3. Setup Module Analysis
        # We identify modules that MUST NOT have weight decay (Norms, Embeddings)
        blacklist_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        whitelist_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        
        # Create a lookup map from parameter name -> owning module
        # allowing precise isinstance checks rather than string heuristics.
        name_to_module = {n: m for n, m in self.named_modules()}

        for pn, p in self.named_parameters():
            if not p.requires_grad:
                continue  # Skip frozen parameters (e.g. frozen parts of SigLIP)

            # --- A. Determine Group (Backbone vs. Head) ---
            is_backbone = "vision_backbone" in pn

            # --- B. Determine Regularization (Decay vs. No Decay) ---
            # Default: Apply decay
            apply_decay = True

            # Rule 1: Never decay biases
            if pn.endswith('bias'):
                apply_decay = False
            
            # Rule 2: Never decay 1D parameters (Scale/Shift params, raw vectors)
            # This catches LayerNorm weights, scalar gates, etc.
            elif p.ndim < 2:
                apply_decay = False

            # Rule 3: Special handling for known structural embeddings
            elif "spatial_pos_embedding" in pn or "query_token" in pn or "token_type" in pn:
                apply_decay = False

            # Rule 4: Module-based inspection (The Gold Standard)
            # If it's a weight, check the module type.
            elif pn.endswith('weight'):
                # Get parent module name (e.g. "model.transformer.layers.0.norm1")
                parent_name = pn.rpartition('.')[0]
                
                # Check the module type if we can find it
                if parent_name in name_to_module:
                    module = name_to_module[parent_name]
                    if isinstance(module, blacklist_modules):
                        apply_decay = False
                    elif isinstance(module, whitelist_modules):
                        apply_decay = True
                else:
                    # Fallback: If we can't find the module, but it's a weight > 1D, decay it.
                    apply_decay = True

            # --- C. Bucket Assignment ---
            if is_backbone:
                if apply_decay:
                    backbone_decay.append(p)
                else:
                    backbone_no_decay.append(p)
            else:
                if apply_decay:
                    head_decay.append(p)
                else:
                    head_no_decay.append(p)

        # 4. Logging Verification (Ensure no params are lost)
        if self.trainer.is_global_zero:
            logger.info(
                f"Optimizer Groups | "
                f"Head Decay: {len(head_decay)} | Head No-Decay: {len(head_no_decay)} | "
                f"Backbone Decay: {len(backbone_decay)} | Backbone No-Decay: {len(backbone_no_decay)}"
            )

        # 5. Construct Optimizer with Differential LRs
        optimizer = torch.optim.AdamW(
            [
                {"params": head_decay,       "lr": base_lr,     "weight_decay": weight_decay},
                {"params": head_no_decay,    "lr": base_lr,     "weight_decay": 0.0},
                {"params": backbone_decay,   "lr": backbone_lr, "weight_decay": weight_decay},
                {"params": backbone_no_decay,"lr": backbone_lr, "weight_decay": 0.0},
            ],
            betas=(0.9, 0.999)
        )

        # 6. Scheduler
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.trainer.estimated_stepping_batches * self.cfg.optimizer.warmup_percentage),
            num_training_steps=self.trainer.estimated_stepping_batches
        )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def on_train_batch_end(self, outputs, batch: Dict[str, Any], batch_idx: int) -> None:
        """
        [SOTA, ATOMICALLY SAFE, PRODUCTION-GRADE VERSION]
        Hook for Failsafe Backups. This version is engineered to be atomic,
        ensuring that the old backup is only deleted *after* the new one has
        been successfully saved. This prevents data loss during a crash.
        """
        if self.trainer.global_rank != 0: return

        try:
            total_batches = len(self.trainer.train_dataloader)
        except:
            total_batches = self.trainer.num_training_batches
        
        is_last_batch = (batch_idx + 1) == total_batches
        if not is_last_batch: return

        epoch = self.trainer.current_epoch
        backup_freq = self.cfg.training.get("backup_every_n_epochs", 1)
        
        if backup_freq <= 0 or (epoch + 1) % backup_freq != 0:
            return
        
        # --- [START OF THE DEFINITIVE PATCH 2] ---
        
        logger.info(f"End of epoch {epoch}: Triggering atomic failsafe backup...")
        
        # 1. Define the path for the NEW backup.
        backup_dir = Path(self.cfg.training.get("backup_dir", "checkpoints/backup"))
        backup_dir.mkdir(parents=True, exist_ok=True)
        new_backup_path = backup_dir / f"backup_epoch_{epoch:03d}.ckpt"

        try:
            # 2. SAVE THE NEW CHECKPOINT FIRST. This is the critical step.
            self.trainer.save_checkpoint(new_backup_path)
            logger.info(f"Failsafe backup for epoch {epoch} saved successfully to {new_backup_path}.")

            # 3. ONLY AFTER the save is successful, find and delete older backups.
            # This is safer than relying on a state variable. We scan the directory.
            all_backups = sorted(list(backup_dir.glob("backup_epoch_*.ckpt")))
            
            # Keep the most recent N backups (e.g., keep the last 2)
            backups_to_keep = self.cfg.training.get("backups_to_keep", 2)
            
            if len(all_backups) > backups_to_keep:
                backups_to_delete = all_backups[:-backups_to_keep]
                for old_backup in backups_to_delete:
                    try:
                        old_backup.unlink()
                        logger.info(f"Cleaned up old failsafe backup: {old_backup.name}")
                    except OSError as e:
                        logger.warning(f"Could not delete old backup {old_backup}: {e}")

        except Exception as e:
            logger.error(f"CRITICAL: Failed to save per-epoch failsafe backup: {e}", exc_info=True)

# ==============================================================================
# 3. MAIN EXECUTION ENTRY POINT
# ==============================================================================
@hydra.main(version_base=None, config_path="../configs", config_name="train_semantic_planner_config")
def main(cfg: DictConfig) -> None:
    """
    Main Entry Point. Sets up environment, loggers, and starts training.
    """
    # 1. Reproducibility
    pl.seed_everything(cfg.seed, workers=True)
    
    logger.info("--- Starting AWSP Training Pipeline (v8.0 SOTA) ---")
    logger.info(f"Working Dir: {os.getcwd()}")
    
    # 2. Logging Setup
    # Hydra sets the working directory, so '.' is the output directory
    output_dir = Path("/content/drive/MyDrive/pda/logs_awsp_newer/")
    
    loggers = [TensorBoardLogger(save_dir=".", name="tb_logs")]
    
    if cfg.logging.get("use_wandb", False):
        os.environ["WANDB_MODE"] = cfg.logging.get("wandb_mode", "online")
        if "WANDB_API_KEY" not in os.environ and cfg.logging.get("wandb_mode") != "offline":
            logger.warning("WandB enabled but API Key not found in env. Switching to offline mode.")
            os.environ["WANDB_MODE"] = "offline"
            
        wandb_logger = WandbLogger(
            project=cfg.logging.wandb_project,
            name=cfg.logging.run_name,
            save_dir=str(output_dir),
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        loggers.append(wandb_logger)

    # 3. Data & Model
    datamodule = SemanticPlannerDataModule(cfg)
    model = SemanticPlannerLightningModule(cfg)

    # 4. Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="awsp-{epoch:02d}-{val/pos_error_m:.4f}",
        monitor="val/pos_error_m",
        mode="min",
        save_top_k=3,
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    progress_bar = TQDMProgressBar(refresh_rate=10)

    # 5. Trainer
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=cfg.training.max_epochs,
        logger=loggers,
        callbacks=[checkpoint_callback, lr_monitor, progress_bar],
        gradient_clip_val=cfg.training.get("gradient_clip_val", 1.0),
        precision=cfg.training.get("precision", "16-mixed"),
        accumulate_grad_batches=cfg.trainer.get("accumulate_grad_batches", 1), # Ensure this is picked up
        log_every_n_steps=10,
        check_val_every_n_epoch=cfg.training.get("check_val_every_n_epoch", 1),
    )


    resume_path = cfg.training.get("resume_from_checkpoint")
    ckpt_arg = None 

    if resume_path and os.path.exists(resume_path):
        logger.info(f"Resuming training from checkpoint: {resume_path}")
        ckpt_arg = resume_path
    else:
        logger.info("No resume checkpoint found. Starting fresh.")


    # resume_path = cfg.training.get("resume_from_checkpoint")
    # ckpt_arg = None # Default: Start fresh

    # if resume_path and os.path.exists(resume_path):
    #     logger.info(f"--- DETECTED CHECKPOINT: {resume_path} ---")
    #     logger.info("Performing Surgical Weight Injection (v8 -> v9 Action Chunking)...")
        
    #     try:
    #         # 1. Load Raw Checkpoint
    #         checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    #         state_dict = checkpoint['state_dict']
            
    #         # 2. Get New Model Structure
    #         model_state = model.state_dict()
    #         filtered_state_dict = {}
            
    #         # 3. The Brain Transplant
    #         for k, v in state_dict.items():
    #             # --- A. Handle Query Token Split ---
    #             # Old model had 'plan_cls_token'. New model has 3 distinct queries.
    #             # We clone the old general knowledge into all 3 new specialists.
    #             if "plan_cls_token" in k:
    #                 logger.info("Migrating: Cloning 'plan_cls_token' -> 'traj', 'grip', & 'phase' queries")
    #                 filtered_state_dict["model.traj_query_token"] = v
    #                 filtered_state_dict["model.grip_query_token"] = v
    #                 filtered_state_dict["model.phase_query_token"] = v
    #                 continue

    #             # --- B. Handle Standard Keys ---
    #             if k in model_state:
    #                 # Shape Check: If shape changed (e.g. Heads, Embeddings), skip loading
    #                 if v.shape != model_state[k].shape:
    #                     logger.warning(f"Resetting Layer (Shape Mismatch): {k} | Old: {v.shape} -> New: {model_state[k].shape}")
    #                     continue
                    
    #                 # Exact Match: Keep it
    #                 filtered_state_dict[k] = v
            
    #         # 4. Inject Weights
    #         # strict=False is mandatory (we expect to miss heads and new embeddings)
    #         keys = model.load_state_dict(filtered_state_dict, strict=False)
            
    #         logger.info("Migration Successful.")
    #         logger.info(f"Layers Initialized from Scratch: {len(keys.missing_keys)}")
    #         # Expect missing: *.traj_head.*, *.token_type_embeddings*, etc.
            
    #         # 5. Force Optimizer Reset
    #         ckpt_arg = None 
            
    #     except Exception as e:
    #         logger.error(f"Surgical migration failed: {e}")
    #         raise e
    # else:
    #     logger.info("No checkpoint found. Starting fresh.")

    try:
        logger.info("Starting trainer.fit()...")
        trainer.fit(
            model, 
            datamodule=datamodule,
            ckpt_path=ckpt_arg 
        )
        logger.info(f"Training complete. Best model: {checkpoint_callback.best_model_path}")
    except Exception as e:
        logger.exception(f"Training failed with exception: {e}")
        raise e
    


    
    finally:
        for lg in loggers:
            if isinstance(lg, WandbLogger):
                import wandb
                if wandb.run:
                    wandb.finish()
                logger.info("W&B run finalized.")

if __name__ == "__main__":
    main()