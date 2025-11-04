# To be placed in train/train_vip_c.py

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from models.vip_c import LinearNormalizer
from tqdm import tqdm
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint,
                                         StochasticWeightAveraging,
                                         TQDMProgressBar)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader
from transformers import get_scheduler
import numpy as np
# --- Project-Specific Imports ---
# Ensure the project root is in the Python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.diffusion_policy import EMA
from models.vip_c import ViPC
from utils.vip_c_dataset import ViPCDataset, vip_c_collate_fn

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_vip_c")


class ViPCDataModule(pl.LightningDataModule):
    """
    [DEFINITIVE, CORRECTED VERSION]
    The SOTA DataModule for the ViP-C Framework. Implements a robust
    two-stage setup to compute normalization statistics before training.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train_dataset: Optional[ViPCDataset] = None
        self.val_dataset: Optional[ViPCDataset] = None
        self._has_setup = False
        self.action_normalizer = LinearNormalizer()
        self.proprio_normalizer = LinearNormalizer()

    def setup(self, stage: Optional[str] = None):
        if self._has_setup:
            return
        logger.info(f"Setting up ViPCDataModule for stage: {stage}")
        dataset_cfg = self.cfg.dataset
        model_cfg = self.cfg.model

        # This logic is now robust to resuming from a checkpoint.
        is_resuming = hasattr(self.trainer, 'ckpt_path') and self.trainer.ckpt_path is not None

        if stage == 'fit' and not is_resuming:
            logger.info("Fresh run detected. Computing normalization stats...")
            
            # Stage 1: Create a temporary dataset that returns RAW, unnormalized data.
            temp_train_dataset = ViPCDataset(
                enhanced_dataset_path=dataset_cfg.train_path,
                obs_horizon=model_cfg.controller_cfg.obs_horizon,
                action_horizon=model_cfg.controller_cfg.action_horizon,
                action_normalizer=LinearNormalizer(), # Pass empty normalizers
                proprio_normalizer=LinearNormalizer()
            )

            temp_loader = DataLoader(
                temp_train_dataset, batch_size=self.cfg.training.batch_size,
                num_workers=0, collate_fn=vip_c_collate_fn
            )
            all_actions, all_proprios = [], []
            for batch in tqdm(temp_loader, desc="Computing Normalization Stats"):
                if batch.get("batch_failed"): continue
                all_actions.append(batch['ground_truth_action_chunk_raw'].numpy())
                all_proprios.append(batch['controller_observation_history']['proprio_raw'].numpy())

            # Stage 2: Fit the normalizers that are attributes of THIS class.
            self.action_normalizer.fit(np.concatenate(all_actions).reshape(-1, self.cfg.model.controller_cfg.action_dim))
            self.proprio_normalizer.fit(np.concatenate(all_proprios).reshape(-1, self.cfg.model.controller_cfg.proprio_dim))
            logger.info("Normalization stats computed and fitted.")
        
        # Stage 3: Create the final datasets using the (now possibly fitted) normalizers.
        logger.info("Creating final datasets for training and validation...")
        self.train_dataset = ViPCDataset(
            enhanced_dataset_path=dataset_cfg.train_path,
            obs_horizon=model_cfg.controller_cfg.obs_horizon,
            action_horizon=model_cfg.controller_cfg.action_horizon,
            action_normalizer=self.action_normalizer,
            proprio_normalizer=self.proprio_normalizer
        )
        if dataset_cfg.val_path:
            self.val_dataset = ViPCDataset(
                enhanced_dataset_path=dataset_cfg.val_path,
                obs_horizon=model_cfg.controller_cfg.obs_horizon,
                action_horizon=model_cfg.controller_cfg.action_horizon,
                action_normalizer=self.action_normalizer,
                proprio_normalizer=self.proprio_normalizer
            )
        self._has_setup = True

    def train_dataloader(self) -> DataLoader:
        """
        Creates the DataLoader for the training set.
        """
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Did you call setup()?")

        # Use CUDA-aware pin_memory for faster host-to-device transfers.
        pin_memory_enabled = self.trainer.strategy.root_device.type == "cuda"
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True, # Standard shuffling for training
            num_workers=self.cfg.dataset.num_workers,
            collate_fn=vip_c_collate_fn, # The crucial custom collate function
            pin_memory=pin_memory_enabled,
            persistent_workers=(self.cfg.dataset.num_workers > 0) # The SOTA performance feature
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """
        Creates the DataLoader for the validation set.
        """
        if self.val_dataset is None:
            return None # PyTorch Lightning handles this gracefully.

        pin_memory_enabled = self.trainer.strategy.root_device.type == "cuda"

        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.training.val_batch_size,
            shuffle=False, # No need to shuffle validation data
            num_workers=self.cfg.dataset.num_workers,
            collate_fn=vip_c_collate_fn,
            pin_memory=pin_memory_enabled,
            persistent_workers=(self.cfg.dataset.num_workers > 0)
        )
    



# --- II. Component Deep Dive: The ViPCLightningModule (The Training Engine) ---

class ViPCLightningModule(pl.LightningModule):
    """
    The Definitive, SOTA Training Engine for the ViP-C Framework.

    This LightningModule orchestrates the end-to-end training of the ViPC model.
    It is engineered for resilience, deep diagnostics, and high performance.

    Key SOTA Features:
    - **Dual-Objective Loss:** Manages and weights the separate loss functions for
      the Planner and the Controller.
    - **Granular Diagnostic Logging:** Logs raw and weighted losses for both
      components, providing deep insight into the training dynamics.
    - **Qualitative Visual Validation:** Periodically generates and logs images
      comparing the predicted heatmaps to the ground truth, offering an
      intuitive check on the Planner's learning progress.
    - **Exponential Moving Average (EMA):** Maintains an EMA of the model weights
      for improved stability and evaluation performance, a best practice for
      training diffusion models.
    - **Principled Optimization:** Implements the AdamW optimizer with a
      warmup-cosine learning rate schedule, the standard for SOTA models.
    - **Resilient Checkpointing:** Includes logic to save and correctly load both
      the model and EMA states for seamless resumption of training.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        # Save the config as hyperparameters, making it accessible in logs and checkpoints
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        # 1. Instantiate the model, EMA, and noise scheduler
        self.model = ViPC(
            planner_cfg=cfg.model.planner_cfg,
            controller_cfg=cfg.model.controller_cfg
        )
        self.last_backup_path = None

        self.joint_limits_low = None
        self.joint_limits_high = None

        # EMA is critical for stabilizing diffusion model training
        self.ema = EMA(self.model, decay=cfg.training.ema_decay)
        
        # The noise scheduler is used to add noise during training
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.scheduler.timesteps,
            beta_schedule=cfg.scheduler.beta_schedule,
            beta_start=cfg.scheduler.beta_start,
            beta_end=cfg.scheduler.beta_end,
            clip_sample=False # We handle clipping in our model/data if needed
        )

        self.action_normalizer = LinearNormalizer()
        self.proprio_normalizer = LinearNormalizer()

        logger.info("ViPCLightningModule initialized successfully.")



    def setup(self, stage: str) -> None:
        """
        [DEFINITIVE, CORRECTED VERSION]
        This hook is the single source of truth for linking normalizers between
        the DataModule and the LightningModule, handling both fresh and resumed runs.
        """
        if stage == 'fit':
            # This hook runs AFTER the datamodule's setup has completed.
            is_resuming = hasattr(self.trainer, 'ckpt_path') and self.trainer.ckpt_path is not None


            if is_resuming:
                # If we are resuming, the normalizers were loaded in `on_load_checkpoint`.
                # We now push them to the already-created datamodule.
                self.trainer.datamodule.action_normalizer = self.action_normalizer
                self.trainer.datamodule.proprio_normalizer = self.proprio_normalizer
                self.trainer.datamodule.train_dataset.action_normalizer = self.action_normalizer
                self.trainer.datamodule.train_dataset.proprio_normalizer = self.proprio_normalizer
                if self.trainer.datamodule.val_dataset:
                    self.trainer.datamodule.val_dataset.action_normalizer = self.action_normalizer
                    self.trainer.datamodule.val_dataset.proprio_normalizer = self.proprio_normalizer
                logger.info("Normalizers from checkpoint have been restored and linked to the DataModule.")
            else:
                # On a fresh run, the datamodule has just fitted the normalizers.
                # We pull them into the LightningModule so they can be checkpointed.
                self.action_normalizer = self.trainer.datamodule.action_normalizer
                self.proprio_normalizer = self.trainer.datamodule.proprio_normalizer
                logger.info("Fitted normalizers have been linked from DataModule to LightningModule.")
            
            self.ema.ema_model.to(self.device)

            if self.joint_limits_low is None:
                logger.info("Dynamically extracting kinematic limits from the environment...")
                from envs.panda_env import PandaEnv # Local import
                temp_env = PandaEnv(xml_path=self.cfg.env.xml_path)
                low, high = temp_env.get_action_space_limits()
                self.joint_limits_low = torch.tensor(low, dtype=torch.float32)
                self.joint_limits_high = torch.tensor(high, dtype=torch.float32)
                temp_env.close()
                logger.info("Kinematic limits successfully extracted and stored.")


    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Saves the EMA state and normalizers alongside the model state."""
        checkpoint["ema_state_dict"] = self.ema.state_dict()
        checkpoint["action_normalizer"] = self.action_normalizer
        checkpoint["proprio_normalizer"] = self.proprio_normalizer
        checkpoint["joint_limits_low"] = self.joint_limits_low
        checkpoint["joint_limits_high"] = self.joint_limits_high

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Loads the EMA state and normalizers from a checkpoint."""
        if "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])
            logger.info("Successfully loaded EMA weights from checkpoint.")

        
        if "action_normalizer" in checkpoint and "proprio_normalizer" in checkpoint:
            self.action_normalizer = checkpoint["action_normalizer"]
            self.proprio_normalizer = checkpoint["proprio_normalizer"]
            logger.info("Successfully loaded normalizers from checkpoint into LightningModule.")
        else:
            logger.warning("No normalizers found in checkpoint. A fresh fit is required.")


        if "joint_limits_low" in checkpoint and "joint_limits_high" in checkpoint:
            self.joint_limits_low = checkpoint["joint_limits_low"]
            self.joint_limits_high = checkpoint["joint_limits_high"]
            logger.info("Successfully loaded kinematic limits from checkpoint.")
        else:
            logger.warning("No kinematic limits found in checkpoint. They will be re-derived if possible.")


    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Optional[torch.Tensor]:
        """
        The core dual-objective training loop.
        """
        # --- 1. Failsafe Check for Data Loading Errors ---
        if batch.get("batch_failed", False):
            logger.warning(f"Skipping training step for batch {batch_idx} due to data loading failure.")
            return None # Gracefully skip the update

        # --- 2. Prepare Diffusion Inputs ---
        B = batch['ground_truth_action_chunk'].shape[0]
        gt_actions = batch['ground_truth_action_chunk']
        
        noise = torch.randn_like(gt_actions)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=self.device
        ).long()
        
        noisy_actions = self.noise_scheduler.add_noise(gt_actions, noise, timesteps)
        
        # Add the diffusion-specific inputs to the batch for the model
        batch['noisy_actions'] = noisy_actions
        batch['timesteps'] = timesteps

        # --- 3. Unified Forward Pass ---
        predictions = self.model(batch)

        # --- 4. Compute and Log Losses ---
        raw_loss_planner = F.binary_cross_entropy(
            predictions['predicted_heatmap'],
            batch['ground_truth_subgoal_heatmap']
        )


        raw_loss_controller = F.mse_loss(predictions['predicted_noise'], noise)
        
        # Apply configured weights
        loss_weights = self.cfg.training.loss_weights
        weighted_loss_planner = loss_weights.planner * raw_loss_planner
        weighted_loss_controller = loss_weights.controller * raw_loss_controller
        
        combined_loss = weighted_loss_planner + weighted_loss_controller

        # Extensive diagnostic logging (SOTA feature)
        self.log_dict({
            'train/loss_combined': combined_loss,
            'train/loss_planner_raw': raw_loss_planner,
            'train/loss_controller_raw': raw_loss_controller,
            'train/loss_planner_weighted': weighted_loss_planner,
            'train/loss_controller_weighted': weighted_loss_controller,
        }, on_step=True, on_epoch=True, prog_bar=True)
        
        return combined_loss




    def on_before_optimizer_step(self, optimizer) -> None:
        # This is a hook to update EMA weights *after* the forward pass
        # but *before* the optimizer updates the model weights.
        self.ema.update(self.model)



    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """
        The deep diagnostic validation loop.
        """
        if batch.get("batch_failed", False):
            return # Skip validation on a failed batch

        # --- 1. Fast Path: Quantitative Loss Calculation ---
        # Same logic as training_step, but using the EMA model for evaluation
        B = batch['ground_truth_action_chunk'].shape[0]
        gt_actions = batch['ground_truth_action_chunk']
        noise = torch.randn_like(gt_actions)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=self.device
        ).long()
        noisy_actions = self.noise_scheduler.add_noise(gt_actions, noise, timesteps)
        batch['noisy_actions'] = noisy_actions
        batch['timesteps'] = timesteps

        with torch.no_grad():
            # Use the EMA-averaged model for all validation
            predictions = self.ema.ema_model(batch)

        raw_loss_planner = F.binary_cross_entropy(
            predictions['predicted_heatmap'],
            batch['ground_truth_subgoal_heatmap']
        )

        raw_loss_controller = F.mse_loss(predictions['predicted_noise'], noise)
        
        self.log_dict({
            'val/loss_planner_raw': raw_loss_planner,
            'val/loss_controller_raw': raw_loss_controller,
        }, on_step=False, on_epoch=True, sync_dist=True)

        # --- 2. Slow Path: Qualitative Visual Validation (SOTA feature) ---
        run_qualitative_val = (self.trainer.current_epoch + 1) % self.cfg.validation.run_qualitative_every_n_epoch == 0
        
        if batch_idx == 0 and self.trainer.is_global_zero and run_qualitative_val:
            self._log_qualitative_validation(batch)
            
    def _log_qualitative_validation(self, batch: Dict[str, Any]):
        """Helper function to generate and log diagnostic images."""
        try:
            with torch.no_grad():
                # Run the Planner in inference mode on the validation data
                predicted_heatmap, _, _ = self.ema.ema_model.plan(
                    current_image=batch['planner_current_image'],
                    goal_image=batch['planner_goal_image'],
                    task_phase=batch['planner_task_phase']
                )

            # Select the first item in the batch for visualization
            img = batch['planner_current_image'][0].cpu().numpy()
            gt_h = batch['ground_truth_subgoal_heatmap'][0].cpu().numpy()
            pred_h = predicted_heatmap[0].cpu().numpy()
            
            # Un-normalize image for visualization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = (img.transpose(1, 2, 0) * std + mean).clip(0, 1)

            # Create the composite image
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f"Planner Validation - Epoch {self.current_epoch}", fontsize=16)

            # Original Image
            axes[0].imshow(img)
            axes[0].set_title("Current Image")
            axes[0].axis('off')

            # Ground Truth Heatmap
            axes[1].imshow(img)
            axes[1].imshow(gt_h[0], cmap='jet', alpha=0.5)
            axes[1].set_title("Ground Truth Subgoal")
            axes[1].axis('off')
            
            # Predicted Heatmap
            axes[2].imshow(img)
            axes[2].imshow(pred_h[0], cmap='jet', alpha=0.5)
            axes[2].set_title("Predicted Subgoal")
            axes[2].axis('off')

            plt.tight_layout()

            # Log to wandb
            self.logger.experiment.log({
                "val/planner_visualization": wandb.Image(fig)
            }, step=self.global_step)

            plt.close(fig)
            logger.info("Logged qualitative validation image to W&B.")

        except Exception as e:
            logger.error(f"Failed to log qualitative validation image: {e}", exc_info=True)

    def on_train_batch_end(self, outputs, batch: Dict[str, Any], batch_idx: int) -> None:
        """
        [SOTA, REFINED VERSION]
        SOTA Hook: Called after every training batch.
        We use this to manually save a backup checkpoint on the very last batch of an epoch.
        This is more robust than `on_train_epoch_end`.
        """
        # --- 1. Check if the backup feature is enabled ---
        if not self.cfg.resilience.use_per_epoch_backup:
            return

        # --- 2. Run only on the main process in a distributed setup ---
        if not self.trainer.is_global_zero:
            return

        # --- 3. Check if we are at a backup-worthy epoch ---
        epoch = self.trainer.current_epoch
        if (epoch + 1) % self.cfg.resilience.backup_every_n_epochs != 0:
            return

        # --- 4. Check if this is the last batch of the training epoch ---
        is_last_batch = (batch_idx + 1) == self.trainer.num_training_batches
        if not is_last_batch:
            return
            
        # --- 5. If all conditions are met, perform the backup ---
        logger.info(
            f"End of epoch {epoch}: Triggering periodic failsafe backup..."
        )
        
        # Determine the backup directory from the config
        if self.cfg.resilience.backup_dir is not None:
            base_path = Path(self.cfg.resilience.backup_dir)
        else:
            # Default to a 'backups' folder inside the main checkpoint directory
            base_path = Path(self.trainer.callbacks[-1].dirpath) / "backups"
        
        base_path.mkdir(parents=True, exist_ok=True)
        backup_path = base_path / f"backup_epoch_{epoch}.ckpt"
        
        try:
            # SOTA REFINEMENT: Delete the previous backup *before* saving the new one.
            # This is more disk-space efficient.
            if self.last_backup_path and self.last_backup_path.exists():
                self.last_backup_path.unlink()
                logger.info(f"Deleted previous backup: {self.last_backup_path}")

            # Save the new checkpoint
            self.trainer.save_checkpoint(backup_path)
            
            # Store the path of the backup we just created
            self.last_backup_path = backup_path
            logger.info(f"Failsafe backup for epoch {epoch} saved successfully to {backup_path}.")

        except Exception as e:
            logger.error(f"Failed to save per-epoch failsafe backup: {e}", exc_info=True)

    def configure_optimizers(self):
        """
        [DEFINITIVE, REVERTED VERSION]
        This version uses all parameters, which is necessary for correctly
        resuming from a checkpoint saved with the same configuration. The
        robustness is now handled by the correct EMA and device placement hooks.
        """
        # --- START OF THE DEFINITIVE, FINAL PATCH ---

        logger.info("Configuring optimizer for all model parameters.")

        optimizer = torch.optim.AdamW(
            self.parameters(), # Use all parameters to match the checkpoint's optimizer
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay
        )
        
        # The scheduler logic remains correct.
        num_training_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(num_training_steps * self.cfg.optimizer.warmup_percentage)

        scheduler = get_scheduler(
            "cosine",
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        # --- END OF THE DEFINITIVE, FINAL PATCH ---
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }




@hydra.main(version_base=None, config_path="../configs", config_name="train_vip_c_config")
def main(cfg: DictConfig) -> None:
    """
    The SOTA, Hydra-powered main entry point for training the ViP-C model.

    This function orchestrates the entire training pipeline, including:
    - Seeding for reproducibility.
    - Setting up advanced logging with W&B.
    - Instantiating the DataModule and LightningModule.
    - Configuring a suite of SOTA callbacks for resilience and performance.
    - Initializing and launching the PyTorch Lightning Trainer.
    - Ensuring graceful shutdown and finalization.
    """
    logger.info("=" * 80)
    logger.info("Initializing ViP-C Training Pipeline")
    logger.info("=" * 80)
    logger.info(f"Full configuration:\n{OmegaConf.to_yaml(cfg)}")

    # --- 1. Seeding and Environment Setup ---
    pl.seed_everything(cfg.seed, workers=True)

    # Set W&B mode (e.g., 'online', 'offline', 'disabled')
    # This must be done *before* the WandbLogger is initialized.
    os.environ["WANDB_MODE"] = cfg.logging.wandb_mode
    logger.info(f"W&B mode set to: '{cfg.logging.wandb_mode}'")

    # --- 2. Instantiate Core Components ---
    datamodule = ViPCDataModule(cfg)
    model = ViPCLightningModule(cfg)

    # --- 3. Configure Loggers ---
    loggers = []
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    
    # TensorBoard Logger (always enabled)
    tb_logger = TensorBoardLogger(str(output_dir), name="tb_logs", version="")
    loggers.append(tb_logger)
    logger.info(f"TensorBoard logs will be saved to: {output_dir}/tb_logs")

    # Weights & Biases Logger
    if cfg.logging.use_wandb:
        wandb_logger = WandbLogger(
            project=cfg.logging.wandb_project,
            name=cfg.logging.wandb_run_name or output_dir.name, # Use specified name or Hydra's run name
            save_dir=str(output_dir),
            log_model=True,  # Automatically log model checkpoints to W&B
            checkpoint_name="best-{epoch}-{val/loss_planner_raw:.4f}"
        )
        # SOTA Feature: Watch the model for gradient logging
        wandb_logger.watch(model, log='gradients', log_freq=cfg.logging.log_grad_freq)
        loggers.append(wandb_logger)
        logger.info(f"W&B logging enabled for project '{cfg.logging.wandb_project}'.")

    # --- 4. Configure Callbacks (The Automation Engine) ---
    callbacks = []

    # ModelCheckpoint: Saves the best models based on a validation metric.
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best-epoch={epoch}-planner_loss={val/loss_planner_raw:.4f}",
        monitor=cfg.training.checkpoint_monitor,
        mode="min",
        save_top_k=cfg.training.save_top_k,
        save_last=True, # Always save the latest model for easy resumption
    )
    callbacks.append(checkpoint_callback)
    logger.info(f"ModelCheckpoint enabled. Monitoring '{cfg.training.checkpoint_monitor}'.")

    # LearningRateMonitor: Logs the learning rate at each step.
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # TQDMProgressBar: For a clean progress bar in the console.
    progress_bar = TQDMProgressBar(refresh_rate=cfg.logging.progress_bar_refresh_rate)
    callbacks.append(progress_bar)
    
    # StochasticWeightAveraging (SWA): Optional SOTA technique for generalization.
    if cfg.training.use_swa:
        swa_callback = StochasticWeightAveraging(swa_lrs=cfg.training.swa_lrs)
        callbacks.append(swa_callback)
        logger.info(f"Stochastic Weight Averaging (SWA) enabled with LR: {cfg.training.swa_lrs}.")

    # --- 5. Initialize the PyTorch Lightning Trainer ---
    trainer = pl.Trainer(
        logger=loggers,
        callbacks=callbacks,
        **cfg.trainer  # Unpack all trainer settings from the config (accelerator, devices, etc.)
    )

    # --- 6. Launch Training with Graceful Shutdown ---
    try:
        logger.info("Starting trainer.fit()...")
        # The trainer.fit call now seamlessly handles resumption if ckpt_path is not None
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.training.resume_from_checkpoint)
        logger.info("Training finished successfully.")
    except Exception as e:
        logger.error(f"Training was interrupted by an exception: {e}", exc_info=True)
    finally:
        # SOTA Feature: Ensure W&B is finalized correctly on exit or interrupt.
        if cfg.logging.use_wandb and wandb.run is not None:
            wandb.finish()
            logger.info("W&B run finalized.")

if __name__ == "__main__":
    main()


