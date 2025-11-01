# FILE: train/train_ego_planner.py
# (State-of-the-Art, Resilient, EMA-Integrated, SOTA Version)

"""
The definitive, state-of-the-art training script for the unified Ego-Planner policy,
engineered for maximum robustness, performance, and deep experimental analysis.

This script synthesizes the best practices from reference implementations and integrates
them seamlessly into the PyTorch Lightning framework for a stable and transparent
training experience.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional

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

# --- SOTA FEATURE: Import the specialized sampler for I/O optimization ---
from utils.samplers import EpisodeAwareSampler

# --- Project-Specific Imports ---
from models.ego_planner import EgoPlanner, EgoPlannerConfig
from models.diffusion_policy import NoiseScheduler, NoiseSchedulerConfig, EMA
from utils.ego_planner_dataset import EgoPlannerDataset, ego_planner_collate_fn

# Optional, for enhanced logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Setup a logger for the script
log = logging.getLogger(__name__)


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
        sampler = EpisodeAwareSampler(
            dataset=self.train_dataset,
            shuffle=True,
            seed=self.cfg.seed + self.trainer.global_rank
        )
        log.info("Using EpisodeAwareSampler for training to optimize I/O.")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            sampler=sampler,
            shuffle=False, # The sampler handles all shuffling logic
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(self.cfg.dataset.num_workers > 0),
            collate_fn=ego_planner_collate_fn,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if not self.val_dataset:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.training.val_batch_size,
            shuffle=False,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(self.cfg.dataset.num_workers > 0),
            collate_fn=ego_planner_collate_fn,
        )


# -----------------------------------------------------------------------------
# 2. The Main LightningModule (Rewritten for Robustness)
# -----------------------------------------------------------------------------

class EgoPlannerLightningModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg

        use_checkpointing = self.cfg.training.get("use_activation_checkpointing", False)
        model_cfg = EgoPlannerConfig(**OmegaConf.to_container(cfg.model, resolve=True))
        self.model = EgoPlanner(model_cfg, use_checkpointing=use_checkpointing)
        
        self.ema = EMA(self.model, decay=self.cfg.training.ema_decay)
        
        scheduler_cfg = NoiseSchedulerConfig()
        self.scheduler = NoiseScheduler(scheduler_cfg)

        self.last_backup_path: Optional[Path] = None
        log.info("Definitive SOTA EgoPlannerLightningModule initialized.")

    def setup(self, stage: str) -> None:
        """
        Called at the beginning of fit/validate/test. This is the CORRECT place
        to move non-parameter tensors like the noise schedule to the device.
        """
        if stage == 'fit':
            log.info(f"Moving noise scheduler to device: {self.device}")
            self.scheduler.to(self.device)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        """Robustly loads model and EMA state, ignoring mismatches."""
        if not checkpoint: return
        
        log.info("Applying robust checkpoint loading logic...")
        policy_state_dict = checkpoint['state_dict']
        
        incompatible_keys = self.model.load_state_dict(policy_state_dict, strict=False)
        if incompatible_keys.missing_keys:
            log.warning(f"Policy weights not in ckpt (new layers): {incompatible_keys.missing_keys}")
        if incompatible_keys.unexpected_keys:
            log.warning(f"Ckpt weights ignored (obsolete layers): {incompatible_keys.unexpected_keys}")
        
        ema_state_dict = checkpoint.get('ema_state_dict')
        if ema_state_dict:
            self.ema.load_state_dict(ema_state_dict, strict=False)
        else:
            log.warning("No EMA state found in checkpoint. Re-initializing EMA from loaded model weights.")
            self.ema = EMA(self.model, decay=self.cfg.training.ema_decay)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Injects the EMA state into the checkpoint file."""
        checkpoint["ema_state_dict"] = self.ema.state_dict()

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Streamlined training step. PyTorch Lightning automatically handles
        device placement and AMP context before this method is called.
        """
        gt_actions = batch['action_chunk']
        B = gt_actions.shape[0]

        noise = torch.randn_like(gt_actions)
        timesteps = torch.randint(0, self.scheduler.T, (B,), device=self.device).long()
        
        # Add noisy actions and timesteps to the batch for the model
        batch['noisy_actions'] = self.scheduler.add_noise(gt_actions, timesteps, noise)
        batch['timesteps'] = timesteps
        
        predicted_noise = self.model(batch)
        loss = F.mse_loss(predicted_noise, noise)

        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.ema.update(self.model) # Update EMA weights after the train step
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        gt_actions = batch['action_chunk']
        B = gt_actions.shape[0]

        # Fast Path: Validation Loss (always runs)
        noise = torch.randn_like(gt_actions)
        timesteps = torch.randint(0, self.scheduler.T, (B,), device=self.device).long()
        batch['noisy_actions'] = self.scheduler.add_noise(gt_actions, timesteps, noise)
        batch['timesteps'] = timesteps
        
        # Use the EMA model for all validation
        with torch.no_grad():
            predicted_noise = self.ema.ema_model(batch)
        val_loss = F.mse_loss(predicted_noise, noise)
        self.log('val/loss', val_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        # Slow Path: Periodically run expensive inference and metrics
        run_full_val = (self.trainer.current_epoch + 1) % self.cfg.validation.run_full_validation_every_n_epoch == 0
        if run_full_val:
            with torch.no_grad():
                predicted_actions = self.ema.ema_model.sample(
                    batch=batch,
                    scheduler=self.scheduler,
                    guidance_plan=self.cfg.validation.guidance_scale_plan,
                    guidance_obs=self.cfg.validation.guidance_scale_obs
                )
            action_mse = F.mse_loss(predicted_actions, gt_actions)
            self.log('val/action_mse', action_mse, on_epoch=True, sync_dist=True)

            if batch_idx == 0 and self.trainer.is_global_zero:
                self._log_action_trajectory_plot(predicted_actions, gt_actions)

    def on_train_epoch_end(self):
        """Saves a high-frequency backup checkpoint after every training epoch."""
        if not self.trainer.is_global_zero: return

        epoch = self.trainer.current_epoch
        backup_path = Path("/content/drive/MyDrive/pda/models/v1") / f"backup_epoch_{epoch}.ckpt"
        log.info(f"Saving per-epoch backup checkpoint to {backup_path}...")
        try:
            self.trainer.save_checkpoint(backup_path)
            # Clean up previous backup to save space
            if self.last_backup_path and self.last_backup_path.exists():
                self.last_backup_path.unlink()
            self.last_backup_path = backup_path
        except Exception as e:
            log.error(f"Failed to save per-epoch backup: {e}")

    def _create_full_checkpoint(self) -> Dict[str, Any]:
        """
        Creates a full training state checkpoint, robust to early crashes
        before optimizers are initialized.
        """
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

@hydra.main(version_base=None, config_path="./configs", config_name="train_ego_planner_config")
def main(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed, workers=True)

    # Set WANDB_MODE from config before logger initialization
    if cfg.logging.use_wandb:
        import os
        os.environ["WANDB_MODE"] = cfg.logging.get("wandb_mode", "online")

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    
    datamodule = EgoPlannerDataModule(cfg)
    model = EgoPlannerLightningModule(cfg)

    loggers = [TensorBoardLogger(str(output_dir), name="", version="tb_logs")]
    if cfg.logging.use_wandb:
        wandb_logger = WandbLogger(project=cfg.logging.wandb_project, name=output_dir.name, save_dir=str(output_dir))
        wandb_logger.watch(model, log='gradients', log_freq=100)
        loggers.append(wandb_logger)

    callbacks = [
        ModelCheckpoint(dirpath=output_dir / "checkpoints", filename="best-val_loss={val/loss:.4f}-epoch={epoch}", monitor="val/loss", mode="min", save_top_k=3),
        LearningRateMonitor('step'),
        TQDMProgressBar()
    ]
    if cfg.training.get('use_swa', False):
        callbacks.append(StochasticWeightAveraging(swa_lrs=cfg.training.swa_lrs))

    trainer = pl.Trainer(logger=loggers, callbacks=callbacks, **OmegaConf.to_container(cfg.trainer, resolve=True))

    try:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.training.resume_from_checkpoint)
    except (Exception, KeyboardInterrupt) as e:
        log.warning(f"Training interrupted or failed: {e}", exc_info=True) # Log stack trace
        log.info("Attempting to save a final 'interrupted.ckpt'...")
        final_ckpt_path = output_dir / "checkpoints" / "interrupted.ckpt"
        try:
            final_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model._create_full_checkpoint(), final_ckpt_path)
            log.info(f"Final checkpoint saved successfully to {final_ckpt_path}")
        except Exception as e2:
            log.error(f"Could not save final interrupted checkpoint: {e2}")
    finally:
        if cfg.logging.use_wandb and WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    main()