# To be placed in train/train_uhp.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
import os
import sys
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig,OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint,  TQDMProgressBar, StochasticWeightAveraging
from transformers import get_scheduler
# --- Project-Specific Imports ---
# Ensure the project root is in the Python path for these imports to work.
from models.uhp import LinearNormalizer
from utils.vip_c_dataset import ViPCDataset, vip_c_collate_fn
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from typing import Any, Dict
import hydra

# Initialize logger for this module
logger = logging.getLogger(__name__)

# ==============================================================================
# SECTION 1: THE DATA PIPELINE (UHPDataModule)
#
# This section defines the DataModule, which is responsible for all aspects of
# data loading, processing, and robust normalization.
# ==============================================================================


class UHPDataModule(pl.LightningDataModule):
    """
    [SOTA, RESUMPTION-AWARE VERSION]
    The DataModule for the UHP Framework.

    Implements a robust two-stage setup to compute normalization statistics on a
    fresh run while intelligently skipping this expensive step when resuming
    from a checkpoint. It is engineered for high performance using persistent
    workers and pinned memory.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train_dataset: Optional[ViPCDataset] = None
        self.val_dataset: Optional[ViPCDataset] = None
        self._has_setup = False

        # These normalizers will be fitted on a fresh run or loaded from a
        # checkpoint via the LightningModule's "handshake" during setup.
        self.action_normalizer = LinearNormalizer()
        self.proprio_normalizer = LinearNormalizer()

    def setup(self, stage: Optional[str] = None):
        """
        The core logic for data preparation. Handles the critical two-stage
        normalization process and distinguishes between fresh and resumed runs.
        """
        if self._has_setup:
            return  # Prevent redundant setup.

        logger.info(f"Setting up UHPDataModule for stage: {stage}")
        dataset_cfg = self.cfg.dataset
        model_cfg = self.cfg.model

        # This is the SOTA method to detect if we are resuming training.
        is_resuming = hasattr(self.trainer, 'ckpt_path') and self.trainer.ckpt_path is not None

        # --- The Two-Stage Normalization Process (run only if `stage` is 'fit' and it's a fresh run) ---
        if stage == 'fit' and not is_resuming:
            logger.info("Fresh run detected. Computing normalization stats...")

            # Stage 1: Create a temporary dataset that returns RAW, unnormalized data.
            # We pass in empty normalizers to trigger the dataset's raw data mode.
            temp_train_dataset = ViPCDataset(
                enhanced_dataset_path=dataset_cfg.train_path,
                obs_horizon=model_cfg.executor_cfg.obs_horizon,
                action_horizon=model_cfg.executor_cfg.action_horizon,
                action_normalizer=LinearNormalizer(),
                proprio_normalizer=LinearNormalizer()
            )

            # Use a simple DataLoader to iterate and collect raw data.
            temp_loader = DataLoader(
                temp_train_dataset,
                batch_size=self.cfg.training.batch_size,
                num_workers=self.cfg.dataset.num_workers, # Can use multiple workers
                collate_fn=vip_c_collate_fn
            )
            all_actions_raw, all_proprios_raw = [], []
            for batch in tqdm(temp_loader, desc="[DataModule] Computing Normalization Stats"):
                # The collate_fn will produce this key if a batch fails.
                if batch.get("batch_failed"):
                    continue
                all_actions_raw.append(batch['ground_truth_action_chunk_raw'].numpy())
                all_proprios_raw.append(batch['controller_observation_history']['proprio_raw'].numpy())

            # Stage 2: Fit the normalizers that are attributes of THIS class.
            action_data = np.concatenate(all_actions_raw).reshape(-1, self.cfg.model.executor_cfg.action_dim)
            proprio_data = np.concatenate(all_proprios_raw).reshape(-1, self.cfg.model.executor_cfg.proprio_dim)
            self.action_normalizer.fit(action_data)
            self.proprio_normalizer.fit(proprio_data)
            logger.info("Normalization stats computed and fitted successfully.")
        
        elif stage == 'fit' and is_resuming:
            logger.info("Resumed run detected. Normalization stats will be loaded from checkpoint.")

        # --- Final Dataset Creation ---
        # This runs on both fresh and resumed runs. On a resumed run, the normalizers
        # will be empty here, but they will be replaced by the LightningModule's
        # setup hook before training begins.
        logger.info("Creating final datasets for training and validation...")
        self.train_dataset = ViPCDataset(
            enhanced_dataset_path=dataset_cfg.train_path,
            obs_horizon=model_cfg.executor_cfg.obs_horizon,
            action_horizon=model_cfg.executor_cfg.action_horizon,
            action_normalizer=self.action_normalizer,
            proprio_normalizer=self.proprio_normalizer
        )
        if dataset_cfg.val_path:
            self.val_dataset = ViPCDataset(
                enhanced_dataset_path=dataset_cfg.val_path,
                obs_horizon=model_cfg.executor_cfg.obs_horizon,
                action_horizon=model_cfg.executor_cfg.action_horizon,
                action_normalizer=self.action_normalizer,
                proprio_normalizer=self.proprio_normalizer
            )
        self._has_setup = True

    def train_dataloader(self) -> DataLoader:
        """Creates the high-performance DataLoader for the training set."""
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Please call setup() first.")

        # Conditionally enable pinned memory for faster CPU-to-GPU transfers.
        pin_memory_enabled = self.trainer.strategy.root_device.type == "cuda"

        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            # --- [START OF FINAL PATCH 1] ---
            # Correctness Patch: When iterating directly over a shufflable dataset,
            # this should be False. True is only for map-style datasets without a sampler.
            # This also makes it compatible with custom samplers in the future.
            shuffle=False,
            # --- [END OF FINAL PATCH 1] ---
            num_workers=self.cfg.dataset.num_workers,
            collate_fn=vip_c_collate_fn,
            pin_memory=pin_memory_enabled,
            persistent_workers=(self.cfg.dataset.num_workers > 0)
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        """Creates the DataLoader for the validation set."""
        if self.val_dataset is None:
            return None  # PyTorch Lightning handles this gracefully.

        pin_memory_enabled = self.trainer.strategy.root_device.type == "cuda"

        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.training.val_batch_size,
            shuffle=False,  # No need to shuffle validation data.
            num_workers=self.cfg.dataset.num_workers,
            collate_fn=vip_c_collate_fn,
            pin_memory=pin_memory_enabled,
            persistent_workers=(self.cfg.dataset.num_workers > 0)
        )
    

# To be placed in train/train_uhp.py (append after Section 1 code)

import os
import sys


import torch






# --- Project-Specific Imports ---
# This ensures that we can find the `models` and `envs` directories.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.diffusion_policy import EMA
from models.uhp import UHP_Orchestrator

# ==============================================================================
# SECTION 2: THE TRAINING ENGINE (UHPLightningModule)
#
# This section defines the LightningModule, which is the heart of the training
# process. It encapsulates the model, EMA, losses, optimization, and the
# logic for robust state management and validation.
# ==============================================================================


class UHPLightningModule(pl.LightningModule):
    """
    [SOTA, RESILIENT, HYBRID-LOSS VERSION]
    The definitive Training Engine for the UHP Framework.

    Orchestrates the end-to-end training of the UHP_Orchestrator, engineered for
    resilience, deep diagnostics, and high performance.

    Key Features:
    - **Hybrid-Objective Loss:** Manages and weights the primary action loss
      (MSE on noise) and the auxiliary heatmap loss (BCE on logits).
    - **Robust State Management:** Implements a sophisticated "handshake" protocol
      to save and restore all critical training state, including model weights,
      EMA weights, and data normalization statistics, enabling seamless resumption.
    - **Multi-Tiered Validation:** Uses a fast path/slow path approach to provide
      both efficient per-epoch health checks and deep, periodic qualitative analysis.
    - **Exponential Moving Average (EMA):** Maintains an EMA of model weights for
      improved stability and evaluation performance, a best practice for diffusion models.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        # Save the config, making it accessible in logs and checkpoints.
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        # --- 1. Instantiate the Model, EMA, and Noise Scheduler ---
        self.model = UHP_Orchestrator(
            sequencer_cfg=cfg.model.sequencer_cfg,
            executor_cfg=cfg.model.executor_cfg
        )

        # EMA is critical for stabilizing diffusion model training.
        self.ema = EMA(self.model, decay=cfg.training.ema_decay)
        supported_schedules = ["linear", "squaredcos_cap_v2"]
        beta_schedule = cfg.scheduler.beta_schedule
        if beta_schedule not in supported_schedules:
            raise ValueError(
                f"Unsupported beta_schedule: '{beta_schedule}'. "
                f"Please use one of the following supported by diffusers.DDPMScheduler: {supported_schedules}"
            )
        
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.scheduler.timesteps,
            beta_schedule=beta_schedule,
            beta_start=cfg.scheduler.beta_start,
            beta_end=cfg.scheduler.beta_end,
            clip_sample=False
        )

        # --- 2. Initialize Master Copies of Normalizers and Kinematic Limits ---
        # These will be populated either by the DataModule (fresh run) or from a checkpoint.
        self.action_normalizer = LinearNormalizer()
        self.proprio_normalizer = LinearNormalizer()
        self.joint_limits_low = None
        self.joint_limits_high = None

        logger.info("UHPLightningModule initialized successfully.")

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Saves all critical state alongside the model weights."""
        checkpoint["ema_state_dict"] = self.ema.state_dict()
        checkpoint["action_normalizer"] = self.action_normalizer
        checkpoint["proprio_normalizer"] = self.proprio_normalizer
        checkpoint["joint_limits_low"] = self.joint_limits_low
        checkpoint["joint_limits_high"] = self.joint_limits_high

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Loads all critical state from a checkpoint."""
        if "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])
            logger.info("Successfully loaded EMA weights from checkpoint.")
        
        if "action_normalizer" in checkpoint and "proprio_normalizer" in checkpoint:
            self.action_normalizer = checkpoint["action_normalizer"]
            self.proprio_normalizer = checkpoint["proprio_normalizer"]
            logger.info("Successfully loaded normalizers from checkpoint into LightningModule.")

        if "joint_limits_low" in checkpoint and "joint_limits_high" in checkpoint:
            self.joint_limits_low = checkpoint["joint_limits_low"]
            self.joint_limits_high = checkpoint["joint_limits_high"]
            logger.info("Successfully loaded kinematic limits from checkpoint.")

    def setup(self, stage: str) -> None:
        """
        [SOTA "HANDSHAKE" PROTOCOL]
        This hook is the single source of truth for synchronizing normalizers
        between the DataModule and the LightningModule. It also handles device
        placement and one-time environment setup.
        """
        if stage == 'fit':
            # This hook runs AFTER on_load_checkpoint and AFTER the datamodule's setup.
            is_resuming = hasattr(self.trainer, 'ckpt_path') and self.trainer.ckpt_path is not None

            if is_resuming:
                # PUSH stats from self (loaded from checkpoint) to the datamodule.
                self.trainer.datamodule.action_normalizer = self.action_normalizer
                self.trainer.datamodule.proprio_normalizer = self.proprio_normalizer
                # Also update the datasets that were already created.
                self.trainer.datamodule.train_dataset.action_normalizer = self.action_normalizer
                self.trainer.datamodule.train_dataset.proprio_normalizer = self.proprio_normalizer
                if self.trainer.datamodule.val_dataset:
                    self.trainer.datamodule.val_dataset.action_normalizer = self.action_normalizer
                    self.trainer.datamodule.val_dataset.proprio_normalizer = self.proprio_normalizer
                logger.info("[Handshake] PUSHED normalizers from checkpoint to DataModule.")
            else:
                # PULL stats from the datamodule (which just fitted them) into self.
                self.action_normalizer = self.trainer.datamodule.action_normalizer
                self.proprio_normalizer = self.trainer.datamodule.proprio_normalizer
                logger.info("[Handshake] PULLED fitted normalizers from DataModule to LightningModule.")

            # The EMA model is a deepcopy on CPU; it must be moved to the correct device.
            self.ema.ema_model.to(self.device)

            # Programmatically extract kinematic limits if not loaded from a checkpoint.
            if self.joint_limits_low is None:
                logger.info("Dynamically extracting kinematic limits from the environment...")
                from envs.panda_env import PandaEnv # Local import for safety
                temp_env = PandaEnv(xml_path=self.cfg.env.xml_path)
                low, high = temp_env.get_action_space_limits()
                self.joint_limits_low = torch.tensor(low, dtype=torch.float32)
                self.joint_limits_high = torch.tensor(high, dtype=torch.float32)
                temp_env.close()
                logger.info("Kinematic limits extracted and stored.")

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Optional[torch.Tensor]:
        """The core hybrid-objective training loop."""
        if batch.get("batch_failed", False):
            logger.warning(f"Skipping training step for batch {batch_idx} due to data loading failure.")
            return None

        # --- Prepare Diffusion Inputs ---
        gt_actions = batch['ground_truth_action_chunk']
        B = gt_actions.shape[0]
        noise = torch.randn_like(gt_actions)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device
        ).long()
        batch['noisy_actions'] = self.noise_scheduler.add_noise(gt_actions, noise, timesteps)
        batch['timesteps'] = timesteps

        # --- Unified Forward Pass ---
        predictions = self.model(batch)

        # --- Compute and Log Hybrid Losses ---
        raw_loss_action = F.mse_loss(predictions['predicted_noise'], noise)
        raw_loss_heatmap = F.binary_cross_entropy_with_logits(
            predictions['predicted_heatmap_logits'],
            batch['ground_truth_subgoal_heatmap']
        )

        # Apply configured weights
        lambda_heatmap = self.cfg.training.loss_weights.lambda_heatmap
        weighted_loss_heatmap = lambda_heatmap * raw_loss_heatmap
        
        combined_loss = raw_loss_action + weighted_loss_heatmap

        # Extensive diagnostic logging for deep analysis.
        self.log_dict({
            'train/loss_combined': combined_loss,
            'train/loss_action_raw': raw_loss_action,
            'train/loss_heatmap_raw': raw_loss_heatmap,
            'train/loss_heatmap_weighted': weighted_loss_heatmap,
        }, on_step=True, on_epoch=True, prog_bar=True)

        return combined_loss

    def on_before_optimizer_step(self, optimizer) -> None:
        """Hook to update EMA weights before the optimizer step."""
        self.ema.update(self.model)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """The multi-tiered diagnostic validation loop."""
        if batch.get("batch_failed", False):
            return

        # --- Fast Path: Quantitative Loss Calculation (runs every epoch) ---
        with torch.no_grad():
            # Use the EMA-averaged model for all validation.
            # We still need to create diffusion inputs for a comparable loss value.
            gt_actions = batch['ground_truth_action_chunk']
            B = gt_actions.shape[0]
            noise = torch.randn_like(gt_actions)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device
            ).long()
            batch['noisy_actions'] = self.noise_scheduler.add_noise(gt_actions, noise, timesteps)
            batch['timesteps'] = timesteps
            
            predictions = self.ema.ema_model(batch)
            
            val_loss_action = F.mse_loss(predictions['predicted_noise'], noise)
            val_loss_heatmap = F.binary_cross_entropy_with_logits(
                predictions['predicted_heatmap_logits'], batch['ground_truth_subgoal_heatmap']
            )

        self.log_dict({
            'val/loss_action_raw': val_loss_action,
            'val/loss_heatmap_raw': val_loss_heatmap,
        }, on_step=False, on_epoch=True, sync_dist=True)

        # --- Slow Path: Qualitative Visual Validation (runs periodically) ---
        run_qual_freq = self.cfg.validation.run_qualitative_every_n_epoch
        run_qualitative_val = (self.trainer.current_epoch + 1) % run_qual_freq == 0
        
        if batch_idx == 0 and self.trainer.is_global_zero and run_qualitative_val:
            # --- [START OF PATCH 2] ---
            logger.info(f"Epoch {self.trainer.current_epoch}: Running full validation.")
            
            # 1. Plan using the EMA model to get the subgoal embedding.
            subgoal_embedding, pred_heatmap_viz = self.ema.ema_model.plan(
                current_image=batch['planner_current_image'],
                goal_image=batch['planner_goal_image'],
                task_phase=batch['planner_task_phase']
            )

            # 2. Act using the EMA model to generate the action sequence.
            predicted_actions = self.ema.ema_model.act(
                observation_history=batch['controller_observation_history'],
                subgoal_embedding=subgoal_embedding,
                noise_scheduler=self.noise_scheduler,
                num_inference_steps=self.cfg.validation.num_inference_steps,
                action_normalizer=self.action_normalizer,
                proprio_normalizer=self.proprio_normalizer,
                joint_limits_low=self.joint_limits_low,
                joint_limits_high=self.joint_limits_high
            )

            # 3. Calculate and log the true end-to-end action MSE.
            action_mse = F.mse_loss(predicted_actions, gt_actions)
            self.log('val/action_mse', action_mse, on_epoch=True, sync_dist=True)

            # 4. Log the qualitative heatmap visualization.
            self._log_qualitative_validation(batch, pred_heatmap_viz)

    # In UHPLightningModule:

    def _log_qualitative_validation(self, batch: Dict[str, Any], pred_heatmap_viz: torch.Tensor):
        """Helper function to generate and log diagnostic images."""
        try:


            # The logic now begins directly with selecting the items for visualization.
            img = batch['planner_current_image'][0].cpu().numpy()
            gt_h = batch['ground_truth_subgoal_heatmap'][0].cpu().numpy()
            pred_h = pred_heatmap_viz[0].cpu().numpy()
            
            # Un-normalize image for visualization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = (img.transpose(1, 2, 0) * std + mean).clip(0, 1)

            # Create a composite image (reusing code from ViPC's logger)
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f"Sequencer Validation - Epoch {self.current_epoch}", fontsize=16)
            axes[0].imshow(img); axes[0].set_title("Current Image"); axes[0].axis('off')
            axes[1].imshow(img); axes[1].imshow(gt_h[0], cmap='jet', alpha=0.5); axes[1].set_title("Ground Truth Subgoal"); axes[1].axis('off')
            axes[2].imshow(img); axes[2].imshow(pred_h[0], cmap='jet', alpha=0.5); axes[2].set_title("Predicted Subgoal"); axes[2].axis('off')
            plt.tight_layout()

            # Log to active loggers (W&B, TensorBoard, etc.)
            self.logger.experiment.log({
                "val/sequencer_visualization": wandb.Image(fig)
            }, step=self.global_step)

            plt.close(fig)
            logger.info("Logged qualitative validation image to W&B.")

        except Exception as e:
            logger.error(f"Failed to log qualitative validation image: {e}", exc_info=True)

    def configure_optimizers(self):
        """Configures the AdamW optimizer and a cosine learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay
        )
        
        num_training_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(num_training_steps * self.cfg.optimizer.warmup_percentage)

        scheduler = get_scheduler(
            "cosine",
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": { "scheduler": scheduler, "interval": "step" }
        }
    





@hydra.main(version_base=None, config_path="../configs", config_name="train_uhp_config")
def main(cfg: DictConfig) -> None:
    """
    The SOTA, Hydra-powered main entry point for training the UHP model.

    This function orchestrates the entire training pipeline, including:
    - Seeding for reproducibility.
    - Setting up advanced logging with Weights & Biases.
    - Instantiating the UHPDataModule and UHPLightningModule.
    - Configuring a suite of SOTA callbacks for resilience and performance.
    - Initializing and launching the PyTorch Lightning Trainer with graceful shutdown.
    """
    logger.info("=" * 80)
    logger.info("Initializing UHP v2.0 Training Pipeline")
    logger.info("=" * 80)
    logger.info(f"Full configuration:\n{OmegaConf.to_yaml(cfg)}")

    # --- 1. Seeding and Environment Setup ---
    pl.seed_everything(cfg.seed, workers=True)

    # Set W&B mode (e.g., 'online', 'offline', 'disabled') before logger init.
    os.environ["WANDB_MODE"] = cfg.logging.wandb_mode
    logger.info(f"W&B mode set to: '{cfg.logging.wandb_mode}'")

    # --- 2. Instantiate Core Components ---
    datamodule = UHPDataModule(cfg)
    model = UHPLightningModule(cfg)

    # --- 3. Configure Loggers ---
    loggers = []
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    
    # TensorBoard Logger (always enabled for local logs).
    tb_logger = TensorBoardLogger(str(output_dir), name="tb_logs", version="")
    loggers.append(tb_logger)
    logger.info(f"TensorBoard logs will be saved to: {output_dir}/tb_logs")

    # Weights & Biases Logger (optional).
    if cfg.logging.use_wandb:
        wandb_logger = WandbLogger(
            project=cfg.logging.wandb_project,
            name=cfg.logging.wandb_run_name or output_dir.name,
            save_dir=str(output_dir),
            log_model=True,  # Automatically log model checkpoints to W&B.
        )
        # SOTA Feature: Watch the model for gradient logging.
        wandb_logger.watch(model, log='gradients', log_freq=cfg.logging.log_grad_freq)
        loggers.append(wandb_logger)
        logger.info(f"W&B logging enabled for project '{cfg.logging.wandb_project}'.")

    # --- 4. Configure Callbacks (The Automation Engine) ---
    callbacks = []

    # ModelCheckpoint: Saves the best models and the last model for resumption.
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best-epoch={epoch}-loss_act={val/loss_action_raw:.4f}",
        monitor=cfg.training.checkpoint_monitor,
        mode="min",
        save_top_k=cfg.training.save_top_k,
        save_last=True,  # Critical for easy resumption.
    )
    callbacks.append(checkpoint_callback)
    logger.info(f"ModelCheckpoint enabled. Monitoring '{cfg.training.checkpoint_monitor}'.")

    # LearningRateMonitor: Logs the learning rate at each step.
    callbacks.append(LearningRateMonitor(logging_interval='step'))

    # TQDMProgressBar: For a clean progress bar in the console.
    callbacks.append(TQDMProgressBar(refresh_rate=cfg.logging.progress_bar_refresh_rate))
    
    # StochasticWeightAveraging (SWA): Optional SOTA technique for generalization.
    if cfg.training.use_swa:
        callbacks.append(StochasticWeightAveraging(swa_lrs=cfg.training.swa_lrs))
        logger.info(f"Stochastic Weight Averaging (SWA) enabled with LR: {cfg.training.swa_lrs}.")

    # --- 5. Initialize the PyTorch Lightning Trainer ---
    trainer = pl.Trainer(
        logger=loggers,
        callbacks=callbacks,
        **cfg.trainer  # Unpack all trainer settings from the config.
    )

    # --- 6. Launch Training with Graceful Shutdown ---
    try:
        logger.info("Starting trainer.fit()...")
        # The trainer.fit call seamlessly handles resumption if ckpt_path is not None.
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
    # This allows the script to be run directly.
    # e.g., python -m train.train_uhp
    main()