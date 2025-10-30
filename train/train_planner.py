# FILE: train/train_planner.py
# SOTA Training Script for the SOTA Visual Planner (PyTorch Lightning)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar, StochasticWeightAveraging
import hydra
from omegaconf import DictConfig, OmegaConf
import lpips  # For perceptual loss
import os
import sys
from pathlib import Path
import logging
from typing import Dict, Any, Tuple, Optional


try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# SOTA: Use transformers scheduler
try:
    from transformers import get_scheduler
except ImportError:
    raise ImportError("Please install transformers: pip install transformers")

# Add project root for imports if necessary
try:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    # Import the SOTA model
    from models.planner import VisualPlannerDiffusion
    from utils.planner_dataset import HierarchicalPlannerDataset
    from utils.samplers import EpisodeAwareSampler
except ImportError as e:
    print(f"Error importing project modules: {e}. Ensure script is run from project root or PYTHONPATH is set.")
    sys.exit(1)

log = logging.getLogger(__name__)

# --- PyTorch Lightning Module ---
class EpochBackupCallback(pl.Callback):
    """
    SOTA Callback to save a high-frequency backup checkpoint after every training epoch.
    This replicates the robust backup strategy from the original pretraining script.
    """
    def __init__(self, backup_dir: str, delete_previous: bool = True):
        super().__init__()
        self.backup_dir = Path(backup_dir)
        self.delete_previous = delete_previous
        self.last_backup_path = None
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"EpochBackupCallback enabled. Backups will be saved to: {self.backup_dir}")

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        """Hook that runs at the very end of a training epoch."""
        epoch = trainer.current_epoch
        
        # Define the path for the current backup
        current_backup_path = self.backup_dir / f"backup_epoch_{epoch}.ckpt"
        
        # Save the checkpoint using Lightning's built-in method
        trainer.save_checkpoint(current_backup_path)
        log.info(f"Saved epoch backup to {current_backup_path}")

        # Delete the previous backup to save disk space
        if self.delete_previous and self.last_backup_path and self.last_backup_path.exists():
            try:
                self.last_backup_path.unlink()
            except OSError as e:
                log.warning(f"Could not delete previous backup {self.last_backup_path}: {e}")

        # Update the path of the last saved backup
        self.last_backup_path = current_backup_path


class PlannerLightningModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        # Save config to checkpoint, resolving any interpolations
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        # Store config for easy access
        self.cfg = cfg

        # --- SOTA: Instantiate the SOTA Model ---
        # This now passes all the new SOTA parameters from the config
        self.model = VisualPlannerDiffusion(
            image_size=cfg.model.image_size,
            vit_model_name=cfg.model.vit_model_name,
            vit_feature_dim=cfg.model.vit_feature_dim,
            freeze_vit=cfg.model.freeze_vit,
            progress_embed_dim=cfg.model.progress_embed_dim,
            unet_block_out_channels=tuple(cfg.model.unet_block_out_channels),
            unet_down_block_types=tuple(cfg.model.unet_down_block_types),
            unet_up_block_types=tuple(cfg.model.unet_up_block_types),
            unet_attention_head_dim=cfg.model.unet_attention_head_dim,
            condition_drop_prob=cfg.model.condition_drop_prob, # For CFG
            num_diffusion_timesteps=cfg.scheduler.timesteps
        )

        # --- SOTA: Loss Functions ---
        self.mse_loss = F.mse_loss
        # LPIPS is a great perceptual metric
        if cfg.training.use_lpips_loss:
            try:
                # Initialize LPIPS model and freeze it
                self.lpips_loss = lpips.LPIPS(net='alex').to(self.device)
                for param in self.lpips_loss.parameters():
                    param.requires_grad = False
            except Exception as e:
                log.warning(f"Could not initialize LPIPS loss: {e}. Disabling LPIPS.")
                self.cfg.training.use_lpips_loss = False
                self.lpips_loss = None
        else:
            self.lpips_loss = None

    def forward(self,
                current_image: torch.Tensor,
                goal_image: torch.Tensor,
                progress: torch.Tensor,
                guidance_scale: float = 7.5,
                num_inference_steps: Optional[int] = None
               ) -> torch.Tensor:
        """
        SOTA: Forward pass for inference.
        Updated to pass guidance_scale to the model's sample method.
        """
        steps = num_inference_steps if num_inference_steps is not None else self.cfg.training.eval_inference_steps
        
        # Call the SOTA model's sample method
        return self.model.sample(
            current_image,
            goal_image,
            progress,
            num_inference_steps=steps,
            guidance_scale=guidance_scale
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        # Unpack batch
        current_img = batch['current_image']
        goal_img = batch['goal_image']
        progress = batch['progress']
        gt_subgoal_img = batch['gt_subgoal_image']

        # Model forward pass for training (SOTA model's forward handles CFG dropout)
        predicted_noise, target_noise = self.model(
            gt_subgoal_image=gt_subgoal_img,
            current_image=current_img,
            goal_image=goal_img,
            progress=progress
        )

        # Calculate MSE loss on noise (standard diffusion objective)
        loss = self.mse_loss(predicted_noise, target_noise)

        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        # Unpack batch
        current_img = batch['current_image']
        goal_img = batch['goal_image']
        progress = batch['progress']
        gt_subgoal_img = batch['gt_subgoal_image']

        # --- SOTA: 1. Calculate Validation MSE Loss ---
        # This gives a stable, non-perceptual metric
        predicted_noise, target_noise = self.model(
            gt_subgoal_image=gt_subgoal_img,
            current_image=current_img,
            goal_image=goal_img,
            progress=progress
        )
        val_mse_loss = self.mse_loss(predicted_noise, target_noise)
        self.log('val_mse_loss', val_mse_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # --- SOTA: 2. Generate Guided Subgoal Image ---
        # Use the `forward` method (which calls `sample`) with CFG
        generated_subgoal_guided = self(
            current_img,
            goal_img,
            progress,
            guidance_scale=self.cfg.training.guidance_scale
        )

        # --- 3. Calculate LPIPS Loss (Perceptual Similarity) ---
        if self.cfg.training.use_lpips_loss and self.lpips_loss is not None:
             # LPIPS expects images in range [-1, 1].
             # Clamp generated image to be safe
             generated_clamped = generated_subgoal_guided.clamp(-1, 1)
             gt_rescaled = gt_subgoal_img # Assume already [-1, 1]

             val_lpips_loss = self.lpips_loss(generated_clamped, gt_rescaled).mean()
             self.log('val_lpips_loss', val_lpips_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        else:
             # Log 0 if LPIPS is not used, so checkpointing doesn't fail
             self.log('val_lpips_loss', 0.0, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)


        # --- SOTA: 4. Log Sample Images (first batch only) ---
        if batch_idx == 0 and self.trainer.is_global_zero:
            # SOTA: Generate an *unguided* sample for comparison
            generated_subgoal_unguided = self(
                current_img,
                goal_img,
                progress,
                guidance_scale=1.0 # 1.0 = no guidance
            )
            
            self._log_image_samples(
                current_img,
                goal_img,
                gt_subgoal_img,
                generated_subgoal_guided,
                generated_subgoal_unguided # Pass unguided image for logging
            )

    def _log_image_samples(self, current, goal, gt_subgoal, generated_guided, generated_unguided):
        """SOTA: Logs a 5-image comparison grid to configured loggers."""
        try:
            # Select first image from batch and move to CPU
            images = [
                current[0].cpu(),
                goal[0].cpu(),
                gt_subgoal[0].cpu(),
                generated_guided[0].cpu().clamp(-1, 1),
                generated_unguided[0].cpu().clamp(-1, 1)
            ]

            # Denormalize images
            mean = torch.tensor(self.cfg.dataset.img_mean, device='cpu').view(3, 1, 1)
            std = torch.tensor(self.cfg.dataset.img_std, device='cpu').view(3, 1, 1)
            
            def denorm(img):
                return torch.clamp(img * std + mean, 0, 1)

            denorm_images = [denorm(img) for img in images]
            
            # SOTA: Concatenate horizontally: Current | Goal | GT | Guided | Unguided
            grid = torch.cat(denorm_images, dim=2) 
            caption = "Current | Goal | GT Subgoal | Generated (Guided) | Generated (Unguided)"

            # Log to TensorBoard
            if isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_image("val_samples", grid, self.global_step)

            # Log to W&B
            if isinstance(self.logger, WandbLogger):
                self.logger.experiment.log({
                    "val_samples": [wandb.Image(grid, caption=caption)]
                }, step=self.global_step)
        except Exception as e:
            log.warning(f"Failed to log validation images: {e}")

    def configure_optimizers(self):
        # SOTA: AdamW is the standard for Transformer-based models
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay
        )
        
        # --- SOTA: Scheduler with Warmup ---
        # Get total number of training steps
        try:
            train_loader = self.trainer.datamodule.train_dataloader()
            if self.trainer.max_steps and self.trainer.max_steps > 0:
                total_steps = self.trainer.max_steps
            else:
                total_steps = (len(train_loader) // self.trainer.accumulate_grad_batches) * self.trainer.max_epochs
        except Exception:
            log.warning("Could not determine total steps from trainer. Estimating.")
            total_steps = self.cfg.training.max_epochs * 1000 # Fallback estimate

        # Get warmup steps (can be int or float percentage)
        warmup_config = self.cfg.optimizer.get('warmup_steps', 0)
        if isinstance(warmup_config, float):
            num_warmup_steps = int(total_steps * warmup_config)
        else:
            num_warmup_steps = int(warmup_config)

        log.info(f"Configuring scheduler: Total steps={total_steps}, Warmup steps={num_warmup_steps}")

        # SOTA: Use transformers `get_scheduler` for linear warmup + cosine decay
        scheduler = get_scheduler(
            name="cosine", # "linear" or "cosine"
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # Step scheduler every optimizer step
                "frequency": 1,
            },
        }

# --- PyTorch Lightning DataModule ---

class PlannerDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        """SOTA: Initializes separate datasets for training and validation."""
        if stage == 'fit' or stage is None:
            # --- SOTA PATCH: LOAD SEPARATE DATASETS ---
            
            # 1. Instantiate the Training Dataset
            log.info(f"Loading TRAINING dataset from: {self.cfg.dataset.train_path}")
            self.train_dataset = HierarchicalPlannerDataset(
                dataset_path=self.cfg.dataset.train_path,
                subgoal_horizon_k=self.cfg.dataset.subgoal_horizon_k,
                image_size=tuple(self.cfg.dataset.image_size),
                img_mean=tuple(self.cfg.dataset.img_mean),
                img_std=tuple(self.cfg.dataset.img_std),
                use_random_aug=self.cfg.dataset.use_random_aug,
            )
            
            # 2. Instantiate the Validation Dataset (if path is provided)
            if self.cfg.dataset.get("val_path"):
                log.info(f"Loading VALIDATION dataset from: {self.cfg.dataset.val_path}")
                self.val_dataset = HierarchicalPlannerDataset(
                    dataset_path=self.cfg.dataset.val_path,
                    subgoal_horizon_k=self.cfg.dataset.subgoal_horizon_k,
                    image_size=tuple(self.cfg.dataset.image_size),
                    img_mean=tuple(self.cfg.dataset.img_mean),
                    img_std=tuple(self.cfg.dataset.img_std),
                    use_random_aug=False,
                )
            else:
                log.warning("No `dataset.val_path` provided. Validation will not be run.")

    def train_dataloader(self):
        # The sampler logic we developed is still CRITICAL for training performance.
        sampler = EpisodeAwareSampler(self.train_dataset, shuffle=True, seed=self.cfg.seed)
        log.info("Using SOTA EpisodeAwareSampler for training to improve performance.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False, # Sampler handles shuffling
            sampler=sampler,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=(self.cfg.trainer.accelerator == 'gpu'),
            persistent_workers=(self.cfg.dataset.num_workers > 0),
            drop_last=True
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None # PyTorch Lightning handles this gracefully

        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.training.val_batch_size,
            shuffle=False, # No shuffling needed for validation
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=(self.cfg.trainer.accelerator == 'gpu'),
            persistent_workers=(self.cfg.dataset.num_workers > 0),
            drop_last=False
        )


# --- Hydra Main Entry Point ---

@hydra.main(version_base=None, config_path="../configs", config_name="train_planner_config")
def main(cfg: DictConfig):
    log.info("----------- SOTA Planner Training Configuration -----------")
    log.info(OmegaConf.to_yaml(cfg))
    log.info("-----------------------------------------------------------")

    # --- Seed ---
    pl.seed_everything(cfg.seed, workers=True)

    # --- DataModule ---
    datamodule = PlannerDataModule(cfg)

    # --- LightningModule ---
    model = PlannerLightningModule(cfg)

    # --- Loggers ---
    loggers = []
    if cfg.logging.use_tensorboard:
        loggers.append(TensorBoardLogger(save_dir=str(Path.cwd()), name="", version="tb_logs"))
    if cfg.logging.use_wandb and WANDB_AVAILABLE:
        wandb_logger = WandbLogger(
            project=cfg.logging.wandb_project,
            name=cfg.logging.wandb_run_name or Path.cwd().name,
            save_dir=str(Path.cwd()),
            log_model=cfg.logging.wandb_log_model,
        )
        loggers.append(wandb_logger)
        # SOTA: Watch the model (log='all' is very verbose, 'gradients' is good)
        wandb_logger.watch(model, log='gradients', log_freq=cfg.logging.wandb_watch_log_freq)

    # --- SOTA Callbacks ---
    # SOTA: Checkpoint based on LPIPS (perceptual) and MSE (stable)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(Path.cwd() / "checkpoints"),
        filename="planner-{epoch:02d}-lpips{val_lpips_loss:.4f}-mse{val_mse_loss:.4f}",
        monitor="val_lpips_loss", # SOTA: Monitor perceptual loss
        mode="min",
        save_top_k=cfg.training.save_top_k_checkpoints,
        save_last=True,
        auto_insert_metric_name=False, # Filename is already custom
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    progress_bar = TQDMProgressBar(refresh_rate=cfg.logging.progress_bar_refresh_rate)
    backup_callback = EpochBackupCallback(
        backup_dir=str(Path.cwd() / "checkpoints" / "backup")
    )
    callbacks = [checkpoint_callback, lr_monitor, progress_bar, backup_callback]

    # SOTA: Add Stochastic Weight Averaging (SWA) if configured
    if cfg.training.get('use_swa', False):
        swa_lrs = cfg.training.get('swa_lrs', 1e-3)
        callbacks.append(StochasticWeightAveraging(swa_lrs=swa_lrs))
        log.info(f"Using Stochastic Weight Averaging (SWA) with LRs: {swa_lrs}")

    # --- Trainer ---
    trainer = pl.Trainer(
        logger=loggers,
        callbacks=callbacks,
        max_epochs=cfg.training.max_epochs,
        max_steps=cfg.training.get('max_steps', -1), # Allow setting max_steps
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision, # SOTA: "bf16-mixed" or "16-mixed"
        accumulate_grad_batches=cfg.training.gradient_accumulation_batches,
        gradient_clip_val=cfg.optimizer.gradient_clip_val,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        # SOTA: Use deterministic mode for reproducibility if configured
        deterministic=cfg.trainer.get('deterministic', False),
    )

    # --- Find LR (Optional) ---
    if cfg.optimizer.find_lr:
         log.info("Starting Learning Rate Finder...")
         tuner = pl.tuner.Tuner(trainer)
         lr_finder = tuner.lr_find(model, datamodule=datamodule)
         fig = lr_finder.plot(suggest=True)
         
         if isinstance(trainer.logger, WandbLogger):
             trainer.logger.experiment.log({"lr_finder_plot": wandb.Image(fig)})
         else:
             fig.savefig(str(Path.cwd() / "lr_finder_plot.png"))
             
         log.info(f"Suggested LR: {lr_finder.suggestion()}")
         log.info("LR Finder finished. Exiting.")
         return # Exit after finding LR

    # --- Training ---
    log.info("Starting SOTA training...")
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=cfg.training.resume_from_checkpoint
    )

    log.info("Training finished.")

if __name__ == "__main__":
    main()
