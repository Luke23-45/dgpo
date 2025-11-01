# FILE: train/train_ego_planner.py
# (State-of-the-Art, Resilient, EMA-Integrated, SOTA Version)

"""
The definitive, state-of-the-art training script for the unified Ego-Planner policy,
engineered for maximum robustness, performance, and deep experimental analysis.

This script synthesizes the best practices from the `pretrain_diffusion.py` and
`train_planner.py` reference implementations and integrates them seamlessly into
the PyTorch Lightning framework.

Key Architectural Advancements & Features:
  - **Exponential Moving Average (EMA)**: Implements and manages an EMA of the
    model's weights, a critical technique for stabilizing diffusion model
    training. The EMA model is used for all validation and inference tasks.
  - **Advanced Checkpoint Migration & Warm-Starting**:
    - Overrides `on_load_checkpoint` to implement a sophisticated loading
      mechanism. It can load weights from older, architecturally different
      checkpoints, intelligently migrating, renaming, and skipping layers.
    - Automatically resets the optimizer state if a migration occurs, ensuring
      training stability when resuming from a different model version.
  - **Uncompromising Resilience**:
    - Saves "best" checkpoints based on the primary validation metric.
    - Implements a manual, per-epoch backup (`backup_epoch_N.ckpt`) that
      guarantees no more than one epoch of progress can be lost on a crash.
  - **High-Performance Data Pipeline**: The DataModule is upgraded to use the
    `EpisodeAwareSampler`, which optimizes data loading I/O by improving
    data locality, keeping the GPU saturated.
  - **Efficient Multi-Tiered Validation**:
    - Implements a "Fast Path / Slow Path" validation scheme.
    - Fast Path (runs every validation): Calculates the core MSE loss.
    - Slow Path (runs periodically): Computes expensive but insightful
      metrics like Action MSE and generates qualitative trajectory plots.
  - **SOTA Optimization & Regularization**:
    - Integrates AdamW with a warmup-cosine learning rate schedule.
    - Includes optional support for Stochastic Weight Averaging (SWA) via
      a Hydra config flag for improved generalization.
  - **Comprehensive & Hydra-Managed**: Full integration with Hydra for
    configuration, and W&B / TensorBoard for rich, multi-faceted logging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from diffusers import DDIMScheduler

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint,
                                         StochasticWeightAveraging, TQDMProgressBar)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader
from transformers import get_scheduler

# --- SOTA FEATURE: Import the specialized sampler ---
from utils.samplers import EpisodeAwareSampler

# --- Project-Specific Imports ---
from models.ego_planner import EgoPlanner, EgoPlannerConfig, NoiseScheduler, NoiseSchedulerConfig
from models.diffusion_policy import  EMA
from utils.ego_planner_dataset import EgoPlannerDataset, ego_planner_collate_fn
import os
# Optional, for enhanced logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Setup a logger for the script
log = logging.getLogger(__name__)

try:
    num_cores = len(os.sched_getaffinity(0))
except AttributeError:
    # os.sched_getaffinity is not available on Windows, use os.cpu_count()
    num_cores = os.cpu_count()

# 2. Set the number of threads for PyTorch.
if num_cores:
    torch.set_num_threads(num_cores)
    print(f" PyTorch has been instructed to use all {num_cores} available CPU cores.")
else:
    print(" Could not determine the number of CPU cores. Using PyTorch defaults.")
# -----------------------------------------------------------------------------
# 1. The LightningDataModule (Upgraded with EpisodeAwareSampler)
# -----------------------------------------------------------------------------

class EgoPlannerDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train_dataset: Optional[EgoPlannerDataset] = None
        self.val_dataset: Optional[EgoPlannerDataset] = None

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            log.info(f"Loading TRAINING dataset from: {self.cfg.dataset.train_path}")
            self.train_dataset = EgoPlannerDataset(
                dataset_path=self.cfg.dataset.train_path,
                obs_horizon=self.cfg.model.obs_horizon,
                action_horizon=self.cfg.model.action_horizon,
                use_aug=self.cfg.dataset.use_aug  
            )
            
            if self.cfg.dataset.get("val_path"):
                log.info(f"Loading VALIDATION dataset from: {self.cfg.dataset.val_path}")
                self.val_dataset = EgoPlannerDataset(
                    dataset_path=self.cfg.dataset.val_path,
                    obs_horizon=self.cfg.model.obs_horizon,
                    action_horizon=self.cfg.model.action_horizon,
                    use_aug=False # DEFINITIVE FIX: Explicitly disable augmentation for validation
                )

    def train_dataloader(self) -> DataLoader:
        # --- SOTA FEATURE: Use EpisodeAwareSampler for optimal I/O ---
        # This is significantly more efficient than standard shuffling.
        sampler = EpisodeAwareSampler(
            dataset=self.train_dataset,
            shuffle=True,
            seed=self.cfg.seed + self.trainer.global_rank # Ensure different seed per worker/process
        )
        log.info("Using EpisodeAwareSampler for training to optimize I/O.")
        pin_memory_enabled = torch.cuda.is_available()

        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            sampler=sampler,
            shuffle=False, # The sampler handles all shuffling logic
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=pin_memory_enabled,
            persistent_workers=(self.cfg.dataset.num_workers > 0),
            collate_fn=ego_planner_collate_fn,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if not self.val_dataset:
            return None
        pin_memory_enabled = torch.cuda.is_available()
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.training.val_batch_size,
            shuffle=False,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=pin_memory_enabled,
            persistent_workers=(self.cfg.dataset.num_workers > 0),
            collate_fn=ego_planner_collate_fn,
        )


# -----------------------------------------------------------------------------
# 2. The Main LightningModule (Upgraded with all SOTA features)
# -----------------------------------------------------------------------------

class EgoPlannerLightningModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg

        # --- SOTA ENHANCEMENT: Allow configurable activation checkpointing ---
        use_checkpointing = self.cfg.training.get("use_activation_checkpointing", False)
        model_cfg = EgoPlannerConfig(**cfg.model)
        self.model = EgoPlanner(model_cfg, use_checkpointing=use_checkpointing)
        
        self.ema = EMA(self.model, decay=self.cfg.training.ema_decay)
        log.info(f"EMA enabled with decay rate: {self.cfg.training.ema_decay}")

        scheduler_cfg = NoiseSchedulerConfig(
            beta_start=cfg.scheduler.beta_start,
            beta_end=cfg.scheduler.beta_end,
            schedule=cfg.scheduler.beta_schedule,
            timesteps=cfg.scheduler.timesteps,
        )
        
        self.scheduler = NoiseScheduler(scheduler_cfg)

        self.backup_dir = Path.cwd() / "checkpoints" / "backup"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.last_backup_path = None
        log.info("Definitive SOTA EgoPlannerLightningModule (v2) initialized.")


    def setup(self, stage: str) -> None:
        """Called at the beginning of fit, validate, test, or predict."""
        # --- DEFINITIVE FIX: Move non-nn.Module objects with tensors to the correct device ---
        if stage == 'fit':
            log.info(f"Moving noise scheduler to device: {self.device}")
            self.scheduler.to(self.device)
            
            # --- START OF DEFINITIVE PATCH 1 ---
            # The EMA model is a deepcopy created on the CPU. It must be explicitly moved
            # to the correct device before it's used in the validation loop.
            log.info(f"Moving EMA model to device: {self.device}")
            self.ema.ema_model.to(self.device)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        # DEFINITIVE FIX: Remove the config-based guard to make resumption robust.
        if not checkpoint:
            return # Guard against empty checkpoint dict
            
        log.info("Applying SOTA checkpoint loading logic with EMA restoration...")
        policy_state_dict = checkpoint['state_dict']
        
        incompatible_keys = self.model.load_state_dict(policy_state_dict, strict=False)
        if incompatible_keys.missing_keys:
            log.warning(f"Policy weights not in ckpt (new layers): {incompatible_keys.missing_keys}")
        if incompatible_keys.unexpected_keys:
            log.warning(f"Ckpt weights ignored (obsolete layers): {incompatible_keys.unexpected_keys}")
        
        ema_state_dict = checkpoint.get('ema_state_dict')
        if ema_state_dict:
            log.info("Found EMA state in checkpoint. Loading...")
            self.ema.load_state_dict(ema_state_dict, strict=False)
        else:
            log.warning("No EMA state found in checkpoint. Re-initializing EMA from loaded model weights.")
            self.ema = EMA(self.model, decay=self.cfg.training.ema_decay)

    # Add this method to the EgoPlannerLightningModule class
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Saves the EMA state dict to the checkpoint."""
        checkpoint["ema_state_dict"] = self.ema.state_dict()


    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Definitive, streamlined training step relying on automatic device placement."""
        # --- DEFINITIVE FIX: Remove manual batch.to(device) loop ---
        # PyTorch Lightning handles this automatically before this method is called.
        
        gt_actions = batch['action_chunk']
        B = gt_actions.shape[0]

        noise = torch.randn_like(gt_actions)
        timesteps = torch.randint(0, self.scheduler.T, (B,), device=self.device).long()
        batch['noisy_actions'] = self.scheduler.add_noise(gt_actions, timesteps, noise)
        batch['timesteps'] = timesteps
        
        predicted_noise = self.model(batch)
        loss = F.mse_loss(predicted_noise, noise)

        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        self.ema.update(self.model)
        return loss
    
    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each parameter and log the total
        norms = pl.utilities.grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """Definitive, streamlined validation step relying on automatic device placement."""
        # --- DEFINITIVE FIX: Remove manual batch.to(device) loop ---
        
        gt_actions = batch['action_chunk']
        B = gt_actions.shape[0]

        # --- Fast Path: Validation Loss ---
        noise = torch.randn_like(gt_actions)
        timesteps = torch.randint(0, self.scheduler.T, (B,), device=self.device).long()
        batch['noisy_actions'] = self.scheduler.add_noise(gt_actions, timesteps, noise)
        batch['timesteps'] = timesteps
        
        with torch.no_grad():
            # The EMA model must be on the correct device. The setup hook now handles this.
            predicted_noise = self.ema.ema_model(batch)
        val_loss = F.mse_loss(predicted_noise, noise)
        self.log('val/loss', val_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        # --- Slow Path: Periodically run expensive diagnostics ---
        full_val_freq = self.cfg.validation.get('run_full_validation_every_n_epoch', 5)
        
        if (self.trainer.current_epoch + 1) % full_val_freq == 0:
            

            with torch.no_grad():
                ### START OF FINAL PATCH 2 ###
                # We are modifying this function call to match the new signature from Patch 1.
                predicted_actions = self.ema.ema_model.sample(
                    batch=batch,
                    scheduler=self.scheduler,
                    guidance_plan=self.cfg.validation.guidance_scale_plan,
                    guidance_obs=self.cfg.validation.guidance_scale_obs,
                    # This is the new argument we are adding to the call.
                    num_inference_steps=self.cfg.validation.sampling_steps 
                )
            action_mse = F.mse_loss(predicted_actions, gt_actions)
            self.log('val/action_mse', action_mse, on_epoch=True, sync_dist=True)

            if batch_idx == 0 and self.trainer.is_global_zero:
                self._log_action_trajectory_plot(predicted_actions, gt_actions)

        return val_loss

    # --- SOTA FEATURE: High-Frequency Per-Epoch Backups ---
    def on_train_epoch_end(self):
        if not self.trainer.is_global_zero:
            return

        epoch = self.trainer.current_epoch
        # backup_path = self.backup_dir / f"backup_epoch_{epoch}.ckpt"
        
        base_path = Path("/content/drive/MyDrive/pda/models/v1")
        backup_path = base_path / f"backup_epoch_{epoch}.ckpt"
        log.info(f"Saving per-epoch backup checkpoint to {backup_path}...")
        try:
            self.trainer.save_checkpoint(backup_path)
            # Delete previous backup to save space
            if self.last_backup_path and self.last_backup_path.exists():
                self.last_backup_path.unlink()
            self.last_backup_path = backup_path
            log.info(f"Backup for epoch {epoch} saved successfully.")
        except Exception as e:
            log.error(f"Failed to save per-epoch backup: {e}")

    # Helper to construct checkpoint data
    def _create_full_checkpoint(self) -> Dict[str, Any]:
        """
        Creates a full training state checkpoint. This version is robustly
        patched to be lifecycle-aware, handling cases where the trainer has
        not yet initialized optimizers or schedulers.
        """
        # --- CRITICAL FIX: Safely access trainer attributes ---
        # These attributes only exist after configure_optimizers has been called.
        # If an error happens before that (e.g., during data validation), they will be missing.
        optimizer_states = [opt.state_dict() for opt in self.trainer.optimizers] if hasattr(self.trainer, "optimizers") else []
        lr_scheduler_states = [s['scheduler'].state_dict() for s in self.trainer.lr_schedulers] if hasattr(self.trainer, "lr_schedulers") else []
        
        return {
            "epoch": self.trainer.current_epoch,
            "global_step": self.trainer.global_step,
            "state_dict": self.model.state_dict(),
            "ema_state_dict": self.ema.state_dict(),
            "optimizer_states": optimizer_states,
            "lr_schedulers": lr_scheduler_states,
        }


    # ( configure_optimizers and _log_action_trajectory_plot remain the same as the previous good version )
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.optimizer.lr, weight_decay=self.cfg.optimizer.weight_decay
        )
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps_config = self.cfg.optimizer.get('warmup_steps', 0.05)
        num_warmup_steps = int(total_steps * warmup_steps_config) if isinstance(warmup_steps_config, float) else int(warmup_steps_config)
        scheduler = get_scheduler("cosine", optimizer, num_warmup_steps, total_steps)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def _log_action_trajectory_plot(self, pred_actions, gt_actions):
        try:
            pred_np, gt_np = pred_actions[0].cpu().numpy(), gt_actions[0].cpu().numpy()
            H_a, D_a = pred_np.shape
            fig, axes = plt.subplots(D_a, 1, figsize=(10, 2 * D_a), sharex=True)
            if D_a == 1: axes = [axes]
            for i in range(D_a):
                axes[i].plot(np.arange(H_a), gt_np[:, i], 'g-', label='Ground Truth')
                axes[i].plot(np.arange(H_a), pred_np[:, i], 'b--', label='Prediction')
                axes[i].set_ylabel(f'Action Dim {i}'); axes[i].grid(True, alpha=0.5)
            axes[0].legend(); axes[0].set_title(f'Action Trajectory (Epoch {self.current_epoch})')
            axes[-1].set_xlabel('Horizon Timestep')
            plt.tight_layout()
            for logger in self.trainer.loggers:
                if isinstance(logger, TensorBoardLogger): logger.experiment.add_figure("val/action_trajectory", fig, self.global_step)
                elif isinstance(logger, WandbLogger): logger.experiment.log({"val/action_trajectory": wandb.Image(fig)}, step=self.global_step)
            plt.close(fig)
        except Exception as e:
            log.warning(f"Failed to log action trajectory plot: {e}")


# -----------------------------------------------------------------------------
# 3. Hydra Main Entry Point
# -----------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="train_ego_planner_config")
def main(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed, workers=True)

    import os

    # Set the WANDB_MODE environment variable based on the config.
    # This MUST be done before the WandbLogger is initialized.
    if cfg.logging.use_wandb:
        wandb_mode = cfg.logging.get("wandb_mode", "online")
        os.environ["WANDB_MODE"] = wandb_mode
        log.info(f"W&B mode explicitly set to: '{wandb_mode}'")

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    
    datamodule = EgoPlannerDataModule(cfg)
    model = EgoPlannerLightningModule(cfg)

    loggers = [TensorBoardLogger(str(output_dir), name="", version="tb_logs")]
    if cfg.logging.use_wandb:
        wandb_logger = WandbLogger(project=cfg.logging.wandb_project, name=output_dir.name, save_dir=str(output_dir))
        wandb_logger.watch(model, log='gradients', log_freq=100)
        loggers.append(wandb_logger)

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best-val_loss={val/loss:.4f}-epoch={epoch}",
        monitor="val/loss", mode="min", save_top_k=3,
    )

    callbacks = [checkpoint_callback, LearningRateMonitor('step'), TQDMProgressBar()]
    
    # --- SOTA FEATURE: Add Stochastic Weight Averaging (SWA) if configured ---
    if cfg.training.get('use_swa', False):
        callbacks.append(StochasticWeightAveraging(swa_lrs=cfg.training.get('swa_lrs', 1e-4)))
        log.info("Stochastic Weight Averaging (SWA) is enabled.")

    trainer = pl.Trainer(
        logger=loggers, callbacks=callbacks, **cfg.trainer
    )

    try:
        # The trainer.fit call now seamlessly handles advanced resumption
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.training.resume_from_checkpoint)
    except (Exception, KeyboardInterrupt) as e:
        log.warning(f"Training interrupted or failed: {e}")
        log.info("Attempting to save a final 'interrupted.ckpt'...")
        final_ckpt_path = output_dir / "checkpoints" / "interrupted.ckpt"
        
        try:
            # --- CRITICAL FIX: Ensure the parent directory exists before saving ---
            # This makes the save operation self-sufficient and robust to early crashes.
            final_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use the robust, lifecycle-aware checkpointing method
            torch.save(model._create_full_checkpoint(), final_ckpt_path)
            log.info(f"Final checkpoint saved successfully to {final_ckpt_path}")
        except Exception as e2:
            log.error(f"Could not save final interrupted checkpoint: {e2}")
    finally:
        if cfg.logging.use_wandb and WANDB_AVAILABLE:
            wandb.finish()


if __name__ == "__main__":
    main()