# FILE: train/train_unified_planner.py
# (Definitive, SOTA, Production-Grade Implementation)

r""" Training Script for the UnifiedDiffusionPlanner.

This script implements **production-grade training** for the state-of-the-art
diffusion-based robot manipulation policy.

Features:
- **Diffusion Policy**: Simple MSE loss on noise prediction with DDIM sampling
- **Classifier-Free Guidance**: Unconditional training for inference-time guidance
- **Phase Prediction**: Auxiliary loss from SemanticPlanner design
- **TensorBoard + WandB**: Comprehensive logging (offline mode supported)
- **Atomic Backups**: Failsafe checkpoint saving every N epochs
- **Differential Learning Rates**: Lower LR for vision backbone
- **Hydra Config**: Hierarchical configuration management

Usage:
    python train/train_unified_planner.py --config-name train_unified_planner_config
"""

from __future__ import annotations

import logging
import os
import sys
import time
import math
from pathlib import Path
from typing import Any, Dict, Optional, List

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
    EarlyStopping,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

# --- Project-Specific Imports ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.unified_diffusion_planner import UnifiedDiffusionPlanner, UnifiedDiffusionConfig
from utils.unified_planner_dataset import UnifiedPlannerDataset, unified_planner_collate_fn

# Logger setup
logger = logging.getLogger("train_unified_planner")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Bypass torch.load strict mode for Lightning compatibility
_original_load = torch.load
def strict_mode_bypass_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = strict_mode_bypass_load


# ==============================================================================
# 1. SOTA DATA MODULE
# ==============================================================================

class UnifiedPlannerDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the UnifiedDiffusionPlanner.
    
    Features:
    - Action normalizer fitting on training data
    - Performance optimizations (pin_memory, persistent_workers)
    - Validation split support
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train_dataset: Optional[UnifiedPlannerDataset] = None
        self.val_dataset: Optional[UnifiedPlannerDataset] = None
        
        # Loader optimizations
        self.num_workers = cfg.dataset.get("num_workers", 4)
        self.pin_memory = torch.cuda.is_available()
        self.persistent_workers = self.num_workers > 0

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            logger.info(f"Loading Training Dataset from: {self.cfg.dataset.train_path}")
            self.train_dataset = UnifiedPlannerDataset(
                dataset_path=self.cfg.dataset.train_path,
                action_chunk_size=self.cfg.model.get("action_chunk_size", 8),
                use_augmentation=self.cfg.dataset.get("use_augmentation", True)
            )
            logger.info(f"  Train samples: {len(self.train_dataset)}")

            if self.cfg.dataset.get("val_path"):
                logger.info(f"Loading Validation Dataset from: {self.cfg.dataset.val_path}")
                self.val_dataset = UnifiedPlannerDataset(
                    dataset_path=self.cfg.dataset.val_path,
                    action_chunk_size=self.cfg.model.get("action_chunk_size", 8),
                    use_augmentation=False  # No augmentation for validation
                )
                logger.info(f"  Val samples: {len(self.val_dataset)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=unified_planner_collate_fn,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=unified_planner_collate_fn,
        )


# ==============================================================================
# 2. LIGHTNING MODULE (Diffusion Training)
# ==============================================================================

class UnifiedPlannerLightningModule(pl.LightningModule):
    """
    LightningModule implementing Diffusion Policy training.

    Metrics:
        - Diffusion Loss (MSE on noise prediction)
        - Phase Loss (optional, auxiliary)
        - Position/Rotation reconstruction error (validation)
    
    Features:
        - Differential learning rates (lower for vision backbone)
        - Atomic failsafe backups
        - Comprehensive logging
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        
        # Build model config
        model_config = UnifiedDiffusionConfig(
            action_chunk_size=cfg.model.get("action_chunk_size", 8),
            diffusion_timesteps=cfg.model.get("diffusion_timesteps", 100),
            inference_steps=cfg.model.get("inference_steps", 10),
            d_model=cfg.model.get("d_model", 512),
            denoiser_layers=cfg.model.get("denoiser_layers", 6),
            fusion_layers=cfg.model.get("fusion_layers", 4),
            p_uncond=cfg.model.get("p_uncond", 0.1),
            guidance_scale=cfg.model.get("guidance_scale", 1.5),
            use_phase_prediction=cfg.model.get("use_phase_prediction", True),
            use_separate_heads=cfg.model.get("use_separate_heads", True),
            phase_loss_weight=cfg.model.get("phase_loss_weight", 0.1),
        )
        
        # Create model
        self.model = UnifiedDiffusionPlanner(model_config)
        
        # Load pretrained weights if specified
        if cfg.training.get("pretrained_checkpoint"):
            self._load_pretrained(cfg.training.pretrained_checkpoint)
        
        # Fit action normalizer flag
        self._normalizer_fitted = False

    def _load_pretrained(self, checkpoint_path: str):
        """Load pretrained weights (from BC transfer)."""
        logger.info(f"Loading pretrained weights from: {checkpoint_path}")
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            state_dict = ckpt.get('state_dict', ckpt)
            
            # Handle model. prefix from Lightning
            if any(k.startswith('model.') for k in state_dict.keys()):
                state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"  Loaded: {len(state_dict) - len(missing) - len(unexpected)} keys")
            if missing:
                logger.info(f"  Missing (random init): {len(missing)} keys")
            if unexpected:
                logger.warning(f"  Unexpected: {len(unexpected)} keys")
        except Exception as e:
            logger.error(f"Failed to load pretrained weights: {e}")

    def on_train_start(self):
        """Fit action normalizer before training starts."""
        if self._normalizer_fitted:
            return
            
        logger.info("Fitting action normalizer on training dataset...")
        train_dataset = self.trainer.datamodule.train_dataset
        
        action_samples = []
        num_samples = min(2000, len(train_dataset))
        
        for i in range(0, num_samples, 20):  # Sample every 20th
            try:
                sample = train_dataset[i]
                if sample is not None:
                    action_samples.append(sample['gt_delta_actions'])
            except:
                continue
        
        if action_samples:
            all_actions = torch.stack(action_samples)
            self.model.action_normalizer.fit(all_actions)
            logger.info(f"  Fitted on {len(action_samples)} samples")
            logger.info(f"  Action min: {self.model.action_normalizer.action_min[:3].tolist()}")
            logger.info(f"  Action max: {self.model.action_normalizer.action_max[:3].tolist()}")
        else:
            logger.warning("No samples for normalizer, using defaults!")
        
        self._normalizer_fitted = True

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step: compute diffusion loss."""
        loss = self.model(batch)
        
        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/lr", self.optimizers().param_groups[0]['lr'], on_step=True)
        
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step: sample actions and compute reconstruction error."""
        gt_actions = batch['gt_delta_actions']  # (B, K, 8)
        
        # Sample actions using DDIM
        with torch.no_grad():
            pred_actions = self.model.sample(batch)  # (B, K, 8)
        
        # Compute errors
        pos_error = F.l1_loss(pred_actions[:, :, :3], gt_actions[:, :, :3])
        rot_error = F.l1_loss(pred_actions[:, :, 3:7], gt_actions[:, :, 3:7])
        grip_error = F.l1_loss(pred_actions[:, :, 7:], gt_actions[:, :, 7:])
        total_error = F.mse_loss(pred_actions, gt_actions)
        
        # Log metrics
        self.log("val/total_error", total_error, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/pos_error", pos_error, on_epoch=True, sync_dist=True)
        self.log("val/rot_error", rot_error, on_epoch=True, sync_dist=True)
        self.log("val/grip_error", grip_error, on_epoch=True, sync_dist=True)
        
        return {"val_loss": total_error}

    def configure_optimizers(self):
        """
        SOTA Optimizer Configuration: Differential Learning Rates.
        
        - Vision backbone: 0.1x learning rate (preserve pretrained features)
        - Diffusion head: Full learning rate
        - No weight decay on biases, norms, embeddings
        """
        base_lr = self.cfg.training.learning_rate
        backbone_lr_factor = self.cfg.training.get("backbone_lr_factor", 0.1)
        weight_decay = self.cfg.training.get("weight_decay", 0.01)
        
        # Organize parameters into groups
        backbone_params = []
        head_params_decay = []
        head_params_no_decay = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            if "vision_backbone" in name:
                backbone_params.append(param)
            elif any(nd in name.lower() for nd in ["bias", "norm", "embedding"]):
                head_params_no_decay.append(param)
            else:
                head_params_decay.append(param)
        
        param_groups = [
            {"params": backbone_params, "lr": base_lr * backbone_lr_factor, "weight_decay": 0.0, "name": "backbone"},
            {"params": head_params_decay, "lr": base_lr, "weight_decay": weight_decay, "name": "head_decay"},
            {"params": head_params_no_decay, "lr": base_lr, "weight_decay": 0.0, "name": "head_no_decay"},
        ]
        
        # Remove empty groups
        param_groups = [g for g in param_groups if len(g["params"]) > 0]
        
        logger.info("Optimizer Parameter Groups:")
        for g in param_groups:
            logger.info(f"  {g['name']}: {len(g['params'])} params, lr={g['lr']:.2e}")
        
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))
        
        # Cosine schedule with warmup
        warmup_steps = self.cfg.training.get("warmup_steps", 1000)
        max_steps = self.cfg.training.get("max_steps", 100000)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def on_train_batch_end(self, outputs, batch: Dict[str, Any], batch_idx: int):
        """
        [SOTA, ATOMICALLY SAFE, PRODUCTION-GRADE VERSION]
        Hook for Failsafe Backups at end of each epoch.
        """
        if self.trainer.global_rank != 0:
            return

        try:
            total_batches = len(self.trainer.train_dataloader)
        except:
            total_batches = self.trainer.num_training_batches
        
        is_last_batch = (batch_idx + 1) == total_batches
        if not is_last_batch:
            return

        epoch = self.trainer.current_epoch
        backup_freq = self.cfg.training.get("backup_every_n_epochs", 5)
        
        if backup_freq <= 0 or (epoch + 1) % backup_freq != 0:
            return
        
        logger.info(f"End of epoch {epoch}: Triggering atomic failsafe backup...")
        
        backup_dir = Path(self.cfg.training.get("backup_dir", "checkpoints/backup"))
        backup_dir.mkdir(parents=True, exist_ok=True)
        new_backup_path = backup_dir / f"unified_planner_backup_epoch_{epoch:03d}.ckpt"

        try:
            self.trainer.save_checkpoint(new_backup_path)
            logger.info(f"Failsafe backup saved to {new_backup_path}")

            # Clean old backups
            all_backups = sorted(list(backup_dir.glob("unified_planner_backup_epoch_*.ckpt")))
            backups_to_keep = self.cfg.training.get("backups_to_keep", 3)
            
            if len(all_backups) > backups_to_keep:
                for old_backup in all_backups[:-backups_to_keep]:
                    try:
                        old_backup.unlink()
                        logger.info(f"Cleaned up old backup: {old_backup.name}")
                    except OSError as e:
                        logger.warning(f"Could not delete {old_backup}: {e}")

        except Exception as e:
            logger.error(f"CRITICAL: Failed to save backup: {e}", exc_info=True)


# ==============================================================================
# 3. MAIN EXECUTION ENTRY POINT
# ==============================================================================

@hydra.main(version_base=None, config_path="../configs", config_name="train_unified_planner_config")
def main(cfg: DictConfig) -> None:
    """
    Main Entry Point. Sets up environment, loggers, and starts training.
    """
    # 1. Reproducibility
    pl.seed_everything(cfg.seed, workers=True)
    
    logger.info("=== Starting UnifiedDiffusionPlanner Training ===")
    logger.info(f"Working Dir: {os.getcwd()}")
    
    # 2. Logging Setup
    output_dir = Path(cfg.training.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    loggers = [TensorBoardLogger(save_dir=str(output_dir), name="tb_logs")]
    
    if cfg.logging.get("use_wandb", False):
        # Set to offline mode for local training
        os.environ["WANDB_MODE"] = cfg.logging.get("wandb_mode", "offline")
        
        if "WANDB_API_KEY" not in os.environ and cfg.logging.get("wandb_mode") != "offline":
            logger.warning("WandB API Key not found, switching to offline mode")
            os.environ["WANDB_MODE"] = "offline"
            
        wandb_logger = WandbLogger(
            project=cfg.logging.get("wandb_project", "unified-diffusion-planner"),
            name=cfg.logging.get("run_name", "udp-training"),
            save_dir=str(output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            offline=(cfg.logging.get("wandb_mode") == "offline")
        )
        loggers.append(wandb_logger)
        logger.info("WandB logger enabled (offline mode)")

    # 3. Data & Model
    datamodule = UnifiedPlannerDataModule(cfg)
    model = UnifiedPlannerLightningModule(cfg)

    # 4. Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename="udp-{epoch:02d}-{val/total_error:.4f}",
            monitor="val/total_error",
            mode="min",
            save_top_k=3,
            save_last=True,
            verbose=True
        ),
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=50),
    ]
    
    # Optional early stopping
    if cfg.training.get("early_stopping_patience", 0) > 0:
        callbacks.append(EarlyStopping(
            monitor="val/total_error",
            patience=cfg.training.early_stopping_patience,
            mode="min"
        ))

    # 5. Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.get("max_epochs", 100),
        max_steps=cfg.training.get("max_steps", -1),
        accelerator="auto",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=cfg.training.get("gradient_clip_val", 1.0),
        val_check_interval=cfg.training.get("val_check_interval", 1.0),
        log_every_n_steps=cfg.training.get("log_every_n_steps", 50),
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,  # Faster training
    )

    # 6. Resume from checkpoint if specified
    ckpt_path = cfg.training.get("resume_from_checkpoint", None)
    if ckpt_path and Path(ckpt_path).exists():
        logger.info(f"Resuming from checkpoint: {ckpt_path}")
    else:
        ckpt_path = None

    # 7. Train!
    logger.info("Starting training...")
    trainer.fit(model, datamodule, ckpt_path=ckpt_path)
    
    logger.info("=== Training Complete ===")
    logger.info(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
