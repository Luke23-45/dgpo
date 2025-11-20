# FILE: train/train_semantic_planner.py
# (Definitive, SOTA, Production-Grade Implementation)

"""
Training Script for the Advantage-Weighted Semantic Planner (AWSP).

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
                use_aug=self.cfg.dataset.get("use_aug", True)
            )

            if self.cfg.dataset.get("val_path"):
                logger.info(f"Loading Validation Dataset from: {self.cfg.dataset.val_path}")
                self.val_dataset = SemanticPlannerDataset(
                    dataset_path=self.cfg.dataset.val_path,
                    use_aug=False  # Strict validation (no augmentation)
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

        # 1. Strict Configuration Construction
        # We explicitly map Hydra config to the Dataclass to ensure type safety.
        model_config = SemanticPlannerConfig(
            proprio_dim=cfg.model.proprio_dim,
            vision_backbone_model=cfg.model.vision_backbone_model,
            vision_feature_dim=cfg.model.vision_feature_dim,
            fusion_transformer_layers=cfg.model.fusion_transformer_layers,
            fusion_transformer_heads=cfg.model.fusion_transformer_heads,
            dim_feedforward_ratio=cfg.model.get("dim_feedforward_ratio", 4),
            num_task_phases=cfg.model.num_task_phases,
            dropout=cfg.model.dropout,
            phase_dropout_prob=cfg.model.get("phase_dropout_prob", 0.0) 
        )
        
        # 2. Model Instantiation
        self.model = SemanticPlanner(model_config)

        # 3. Training Hyperparameters
        self.awr_temperature = cfg.training.awr_temperature
        self.awr_max_weight = cfg.training.get("awr_max_weight", 20.0)
        self.lambda_gripper = cfg.training.loss_weights.lambda_gripper

        # 4. Loss Functions (reduction='none' allows per-sample weighting)
        self.pose_criterion = nn.L1Loss(reduction='none') 

        self.register_buffer('grip_pos_weight', torch.tensor([3.0]))
        self.gripper_criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=self.grip_pos_weight)


    def _compute_awr_weights(self, advantages: torch.Tensor) -> torch.Tensor:
        """
        Computes normalized importance weights based on advantages.
        Formula: w = clamp( exp(A / tau), max=max_weight )
        """
        with torch.no_grad():
            # Scale advantages
            scaled_adv = advantages / self.awr_temperature
            # Exponentiate
            weights = torch.exp(scaled_adv)
            # Clamp for numerical stability (prevent gradients from exploding on outliers)
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
        # Gracefully handle dropped batches
        if not batch:
            return None

        # 1. Forward Pass
        outputs = self.model(batch)
        pred_pose = outputs['pose']      # (B, 7)
        pred_grip = outputs['gripper_logit'] # (B, 1)

        # 2. Ground Truths
        gt_pose = batch['ground_truth_subgoal_pose']
        gt_grip = batch['ground_truth_gripper_state']
        advantage = batch['advantage']

        # 3. Compute Raw Losses (Per Sample)
        # Pose: (B, 7) -> mean dim=1 -> (B,)
        loss_pose_sample = self.pose_criterion(pred_pose, gt_pose).mean(dim=-1)
        # Gripper: (B, 1) -> (B,)
        loss_grip_sample = self.gripper_criterion(pred_grip, gt_grip).squeeze(-1)

        # 4. Compute Weights
        # Advantage shape (B, 1) -> squeeze -> (B,)
        weights = self._compute_awr_weights(advantage.squeeze(-1))

        # 5. Weighted Aggregation
        # Total raw loss per sample
        loss_total_sample = loss_pose_sample + (self.lambda_gripper * loss_grip_sample)
        # Apply weights and mean
        weighted_loss = (loss_total_sample * weights).mean()

        # 6. Diagnostic Logging
        self.log("train/loss", weighted_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/weights_mean", weights.mean(), on_step=False, on_epoch=True)
        self.log("train/weights_max", weights.max(), on_step=False, on_epoch=True)
        self.log("train/pose_loss_raw", loss_pose_sample.mean(), on_step=False, on_epoch=True)
        self.log("train/gripper_loss_raw", loss_grip_sample.mean(), on_step=False, on_epoch=True)
        
        return weighted_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        if not batch: return

        with torch.no_grad():
            outputs = self.model(batch)
            pred_pose = outputs['pose']
            pred_grip = outputs['gripper_logit']

            gt_pose = batch['ground_truth_subgoal_pose']
            gt_grip = batch['ground_truth_gripper_state']

            # 1. Standard Unweighted Loss
            loss_pose = F.l1_loss(pred_pose, gt_pose)
            loss_grip = F.binary_cross_entropy_with_logits(pred_grip, gt_grip)
            total_val_loss = loss_pose + (self.lambda_gripper * loss_grip)

            # 2. Physical Metrics
            # Position Error (Euclidean distance in meters)
            pos_error_m = torch.norm(pred_pose[:, :3] - gt_pose[:, :3], dim=-1).mean()
            
            # Rotation Error (Degrees)
            rot_error_deg = self._compute_geodesic_loss(pred_pose[:, 3:], gt_pose[:, 3:])
            
            # Gripper Accuracy
            pred_cls = (torch.sigmoid(pred_grip) > 0.5).float()
            grip_acc = (pred_cls == gt_grip).float().mean()

            # 3. Logging
            self.log("val/loss", total_val_loss, on_epoch=True, sync_dist=True, prog_bar=True)
            self.log("val/pos_error_m", pos_error_m, on_epoch=True, sync_dist=True)
            self.log("val/rot_error_deg", rot_error_deg, on_epoch=True, sync_dist=True)
            self.log("val/gripper_acc", grip_acc, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        """
        Sets up AdamW with correct weight decay handling and Cosine Scheduler.
        
        SOTA Practice: 
        - Weight decay applied to MatMul weights.
        - Weight decay DISABLED for LayerNorms, Biases, and Embeddings.
        """
        # Separation of parameters
        decay = set()
        no_decay = set()
        
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                # Skip frozen backbone parameters (handled by requires_grad check later, but cleaner to skip here)
                if "vision_backbone" in fpn:
                    continue

                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)

        # Dictionary of all trainable parameters
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Intersect with requires_grad parameters
        decay_params = [param_dict[pn] for pn in sorted(list(decay)) if pn in param_dict]
        no_decay_params = [param_dict[pn] for pn in sorted(list(no_decay)) if pn in param_dict]

        optim_groups = [
            {"params": decay_params, "weight_decay": self.cfg.optimizer.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.cfg.optimizer.lr,
            betas=(0.9, 0.999)
        )

        # Cosine Schedule with Warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.trainer.estimated_stepping_batches * self.cfg.optimizer.warmup_percentage),
            num_training_steps=self.trainer.estimated_stepping_batches
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }


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
    
    logger.info("--- Starting AWSP Training Pipeline ---")
    logger.info(f"Working Dir: {os.getcwd()}")
    
    # 2. Logging Setup
    # Hydra sets the working directory, so '.' is the output directory
    output_dir = Path("/content/drive/MyDrive/pda/logs_ego/")
    
    loggers = [TensorBoardLogger(save_dir=".", name="tb_logs")]
    
    # Conditional WandB
    if cfg.logging.get("use_wandb", False):
        # Ensure mode is set
        os.environ["WANDB_MODE"] = cfg.logging.get("wandb_mode", "online")
        
        # Check for API key in environment, warn if missing but mode is not offline
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
        monitor="val/pos_error_m", # Monitoring physical error is more intuitive than combined loss
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
        precision=cfg.training.get("precision", "16-mixed"), # AMP
        log_every_n_steps=10,
        check_val_every_n_epoch=cfg.training.get("check_val_every_n_epoch", 1),
    )

    # 6. Execute
    try:
        logger.info("Starting trainer.fit()...")
        trainer.fit(
            model, 
            datamodule=datamodule,
            ckpt_path=cfg.training.get("resume_from_checkpoint")
        )
        logger.info(f"Training complete. Best model: {checkpoint_callback.best_model_path}")
    except Exception as e:
        logger.exception(f"Training failed with exception: {e}")
        raise e
    finally:
        # Ensure WandB run is closed properly
        # Check if any logger is a WandbLogger
        for lg in loggers:
            if isinstance(lg, WandbLogger):
                import wandb
                if wandb.run:
                    wandb.finish()
                logger.info("W&B run finalized.")

if __name__ == "__main__":
    main()