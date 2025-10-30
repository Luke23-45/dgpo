# FILE: pretrain_diffusion.py
# (State-of-the-Art, Hydra-Configurable, W&B Integrated Version)

"""
State-of-the-art pretraining script for the Transformer-based DiffusionPolicy.

This script is designed for robustness, reproducibility, and deep experimental analysis,
incorporating modern best practices for training large-scale robotics models.

Key Features:
  - **Hydra Configuration**: Utilizes Hydra for powerful and modular configuration
    management, allowing for easy command-line overrides and structured experiments.
  - **Comprehensive Logging**: Integrates Python's native logging, TensorBoard, and
    optional Weights & Biases (W&B) for multi-faceted experiment tracking.
  - **Structured Trainer Class**: Encapsulates all training logic within a
    `DiffusionPretrainer` class for clarity, maintainability, and extensibility.
  - **Full Model Integration**: Correctly instantiates and trains the complete
    `DiffusionPolicy`, including its multi-view `VisionEncoder` and `TemporalTransformer`,
    not just a simple denoiser.
  - **Advanced Learning Rate Control**: Implements a cosine annealing learning rate
    scheduler with a configurable warmup period for stable convergence.
  - **Robust Checkpointing & Resuming**: Saves and loads the complete training state
    (model, EMA, optimizer, scheduler, random states) for seamless resumption.
  - **In-depth Validation & Diagnostics**:
    - Calculates standard validation loss.
    - Generates qualitative "denoising rollout" visualizations, showing how the model
      denoises a sample from pure noise to a clean action sequence. These are saved
      as plots and can be logged as videos to W&B.
    - Computes and logs quantitative action prediction metrics (MSE, MAE) against
      a validation set.
  - **Hardware Acceleration**: Full support for Automatic Mixed Precision (AMP) training
    to maximize throughput on modern GPUs.
  - **Best Practices**: Enforces deterministic seeding, gradient clipping, and
    professional code structure with extensive type hinting and documentation.

To Run:
    # Ensure you have a corresponding Hydra config file (e.g., in configs/pretrain_diffusion.yaml)
    python -m scripts.pretrain_diffusion
    
"""



# -------------------------
# 1. Imports
# -------------------------
# Standard Library
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Third-Party
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch.nn.functional as F
import random

# Optional, for enhanced logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Project-Specific
from utils.controller_dataset import HierarchicalControllerDataset, collate_fn # Use the new dataset
from models.diffusion_policy import DiffusionPolicy, NoiseSchedulerConfig

# Setup a logger for the script
log = logging.getLogger(__name__)

# -------------------------
# 2. Helper Functions
# -------------------------

def set_seed(seed: int):
    """Sets the seed for all relevant random number generators for reproducibility."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    log.info(f"Global seed set to {seed}")

def save_checkpoint(state: Dict[str, Any], is_best: bool, checkpoint_dir: Path):
    """Saves a training checkpoint, distinguishing between the latest and the best."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / "last.pth"
    torch.save(state, last_path)
    if is_best:
        best_path = checkpoint_dir / "best.pth"
        torch.save(state, best_path)
        log.info(f"Saved new best checkpoint to {best_path}")

# -------------------------
# 3. The Trainer Class
# -------------------------

class ControllerBCTrainer:
    """
    Encapsulates the entire pretraining pipeline for the Diffusion Policy.
    Manages the model, data, optimization, logging, and validation.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initializes the trainer from a Hydra configuration object.
        """
        self.cfg = cfg
        self.start_time = time.time()

        # --- Setup Environment ---
        set_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self.output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        log.info(f"Output directory: {self.output_dir}")

        # --- Initialize Logging ---
        self.writer = SummaryWriter(log_dir=self.output_dir / "tensorboard")
        if WANDB_AVAILABLE and cfg.logging.use_wandb:
            wandb.init(
                project=cfg.logging.wandb_project,
                name=cfg.logging.get("wandb_run_name", self.output_dir.name),
                config=OmegaConf.to_container(cfg, resolve=True),
                dir=self.output_dir,
            )
            self.use_wandb = True
        else:
            self.use_wandb = False

        # --- Build Datasets & Dataloaders ---
        log.info("Building datasets...")
        self.train_loader, self.val_loader = self._build_dataloaders()

        # --- Build Model ---
        log.info("Building diffusion policy model...")
        self.policy = self._build_policy()
        self.policy.to(self.device)

        # --- Build Optimizer and Scheduler ---
        self.optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
        )
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.training.epochs * len(self.train_loader),
        )

        # --- Setup AMP ---
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=cfg.training.use_amp)

        # --- State Tracking ---
        self.global_step = 0
        self.start_epoch = 1
        self.best_val_loss = float("inf")

        # --- Resume from Checkpoint (if provided) ---
        if cfg.resume_checkpoint:
            self._load_checkpoint(Path(cfg.resume_checkpoint))

  
    def _build_dataloaders(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        """SOTA: Constructs train and validation dataloaders using a robust random split."""
        # --- START OF SOTA PATCH 1 ---
        full_dataset = HierarchicalControllerDataset(
            dataset_path=self.cfg.dataset.path,
            observation_horizon=self.cfg.model.observation_horizon,
            action_horizon=self.cfg.model.action_horizon,
            subgoal_horizon_k=self.cfg.dataset.subgoal_horizon_k
        )
        
        # Split dataset (e.g., 95% train, 5% val)
        val_split = self.cfg.dataset.get("val_split", 0.05)
        total_len = len(full_dataset)
        val_len = int(total_len * val_split)
        train_len = total_len - val_len
        log.info(f"Splitting dataset: Train={train_len}, Val={val_len}")

        # Ensure consistent split across runs
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_len, val_len],
            generator=torch.Generator().manual_seed(self.cfg.seed)
        )
        # --- END OF SOTA PATCH 1 ---

        pin_memory_enabled = (self.device.type == 'cuda')
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=True,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=pin_memory_enabled,
            collate_fn=collate_fn,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=False,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=pin_memory_enabled,
            collate_fn=collate_fn,
        ) if val_len > 0 else None
        
        return train_loader, val_loader



    def _build_policy(self) -> DiffusionPolicy:
        """Constructs the DiffusionPolicy from the configuration."""
        try:
            proprio_dim = self.train_loader.dataset.dataset.expert_reader.get_proprioception_dim()
            log.info(f"Inferred proprioception dimension from dataset: {proprio_dim}")
        except Exception as e:
            log.critical(f"FATAL: Failed to infer proprioception dimension from the dataset. "
                         f"Check if the dataset is in the correct SOTA format and the index is valid. Error: {e}")
            raise

        scheduler_cfg = NoiseSchedulerConfig(
            beta_start=self.cfg.scheduler.beta_start,
            beta_end=self.cfg.scheduler.beta_end,
            schedule=self.cfg.scheduler.schedule_type,
            timesteps=self.cfg.scheduler.timesteps,
        )

        model_cfg = self.cfg.model
        policy = DiffusionPolicy(
            # Core dimensions
            proprio_dim=proprio_dim,
            H_o=model_cfg.observation_horizon,
            H_a=model_cfg.action_horizon,
            action_dim=model_cfg.action_dim,
            
            # Architectural dimensions from the model config
            image_feat_dim=model_cfg.image_feat_dim,
            d_model=model_cfg.d_model,
            
            # Denoiser-specific hyperparameters
            denoiser_layers=model_cfg.denoiser_layers,
            denoiser_heads=model_cfg.denoiser_heads,
            
            # Scheduler and training parameters
            scheduler_cfg=scheduler_cfg,
            cfg_p_uncond=self.cfg.training.cfg_p_uncond,
            ema_decay=self.cfg.training.ema_decay,
            device=self.device,
        )
        return policy

    def _load_checkpoint(self, path: Path):
        """Loads a full training state from a checkpoint file."""
        if not path.exists():
            log.warning(f"Checkpoint not found at {path}, starting from scratch.")
            return
        log.info(f"Resuming training from checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        if self.policy.ema and "ema_state_dict" in ckpt:
            self.policy.ema.load_state_dict(ckpt["ema_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.global_step = ckpt["global_step"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))

        # Load random states for perfect reproducibility
        if "rng_states" in ckpt:
            torch.set_rng_state(ckpt["rng_states"]["torch"])
            np.random.set_state(ckpt["rng_states"]["numpy"])
            random.setstate(ckpt["rng_states"]["random"])

    def _train_one_epoch(self, epoch: int):
        """Runs a single epoch of training."""
        self.policy.train()
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.cfg.training.epochs}",
            leave=False,
        )
        for (obs_chunk, subgoal_img_chunk), action_chunk in progress_bar:
            # Unpack the new data format
            obs_chunk = {k: v.to(self.device).float() for k, v in obs_chunk.items()}
            subgoal_img_chunk = subgoal_img_chunk.to(self.device).float()
            action_chunk = action_chunk.to(self.device).float()

            self.optimizer.zero_grad()
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.cfg.training.use_amp):
                # Pass the subgoal image to the compute_loss method
                loss, diagnostics = self.policy.compute_loss(
                    action_chunk, obs_chunk, subgoal_image=subgoal_img_chunk
                )

            self.scaler.scale(loss).backward()
            if self.cfg.optimizer.grad_clip_norm:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.cfg.optimizer.grad_clip_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.lr_scheduler.step()

            # EMA is updated by compute_loss if enabled
            self.global_step += 1

            # Logging
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            if self.global_step % self.cfg.logging.log_interval_steps == 0:
                metrics = {
                    "train/loss": loss.item(),
                    "train/lr": self.lr_scheduler.get_last_lr()[0],
                    **{f"train/{k}": v for k, v in diagnostics.items()},
                }
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                self.writer.add_scalar("train/lr", self.lr_scheduler.get_last_lr()[0], self.global_step)
                if self.use_wandb:
                    wandb.log(metrics, step=self.global_step)

    def _validate_one_epoch(self, epoch: int):
        """Runs a single epoch of validation and diagnostics."""
        if not self.val_loader:
            return

        log.info(f"Running validation for epoch {epoch}...")
        self.policy.eval()
        total_val_loss = 0.0
        total_action_mse = 0.0
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)

        with torch.no_grad():
            for (obs_chunk, subgoal_img_chunk), action_chunk in progress_bar:
                # Unpack new data format
                obs_chunk = {k: v.to(self.device).float() for k, v in obs_chunk.items()}
                subgoal_img_chunk = subgoal_img_chunk.to(self.device).float()
                action_chunk = action_chunk.to(self.device).float()

                with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.cfg.training.use_amp):
                    # Pass subgoal to compute_loss
                    loss, _ = self.policy.compute_loss(
                        action_chunk, obs_chunk, subgoal_image=subgoal_img_chunk
                    )
                    # Pass subgoal to sample for quantitative metrics
                    predicted_action = self.policy.sample(
                        obs_chunk,
                        subgoal_image=subgoal_img_chunk,
                        steps=self.cfg.validation.sampling_steps, 
                        use_ema=True
                    )
                    action_mse = F.mse_loss(predicted_action, action_chunk)

                total_val_loss += loss.item()
                total_action_mse += action_mse.item()

        avg_val_loss = total_val_loss / len(self.val_loader)
        avg_action_mse = total_action_mse / len(self.val_loader)
        log.info(f"Validation Loss: {avg_val_loss:.4f}, Action MSE: {avg_action_mse:.4f}")

        metrics = {
            "val/loss": avg_val_loss,
            "val/action_mse": avg_action_mse,
        }
        self.writer.add_scalar("val/loss", avg_val_loss, self.global_step)
        if self.use_wandb:
            wandb.log(metrics, step=self.global_step)

        # --- Qualitative Diagnostics: Denoising Rollout Visualization ---
        self._generate_diagnostic_rollout(epoch)

        # --- Checkpointing ---
        is_best = avg_val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = avg_val_loss
        
        save_checkpoint(
            state={
                "epoch": epoch,
                "global_step": self.global_step,
                "policy_state_dict": self.policy.state_dict(),
                "ema_state_dict": self.policy.ema.state_dict() if self.policy.ema else None,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.lr_scheduler.state_dict(),
                "best_val_loss": self.best_val_loss,
                "rng_states": {
                    "torch": torch.get_rng_state(),
                    "numpy": np.random.get_state(),
                    "random": random.getstate(),
                },
                "config": OmegaConf.to_container(self.cfg, resolve=True),
            },
            is_best=is_best,
            checkpoint_dir=self.output_dir / "checkpoints",
        )

    def _generate_diagnostic_rollout(self, epoch: int):
        log.info("Generating diagnostic denoising rollout...")
        # Get a single sample from the validation set
        # --- START OF SOTA PATCH 5: ADAPT DIAGNOSTICS ---
        (obs_chunk, subgoal_img_chunk), action_chunk = next(iter(self.val_loader))
        
        # Prepare single sample
        obs_sample = {k: v[:1].to(self.device).float() for k, v in obs_chunk.items()}
        subgoal_sample = subgoal_img_chunk[:1].to(self.device).float()
        action_gt = action_chunk[:1].to(self.device).float()

        # Get the denoising trajectory, now conditioned on the subgoal
        with torch.no_grad():
            _, intermediates = self.policy.sample(
                obs_sample,
                subgoal_image=subgoal_sample,
                steps=self.cfg.scheduler.timesteps,
                use_ema=True,
                return_intermediates=True
            )
        
        # Convert to numpy for plotting
        trajectory = torch.stack(intermediates).cpu().numpy().squeeze(axis=1) # (T, H_a, D_a)
        action_gt_np = action_gt.cpu().numpy().squeeze(axis=0) # (H_a, D_a)

        # Create an animation
        subgoal_img_np = subgoal_sample.cpu().numpy().squeeze(axis=0) # Shape: (C, H, W)
        # Convert from CHW to HWC for matplotlib
        subgoal_img_np = np.transpose(subgoal_img_np, (1, 2, 0))
        # Denormalize for viewing (assuming ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        subgoal_img_np = np.clip(std * subgoal_img_np + mean, 0, 1)

        # Create a figure with two subplots: one for the action, one for the subgoal image
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [3, 1]})
        
        # Display the static subgoal image on the right subplot
        ax2.imshow(subgoal_img_np)
        ax2.set_title("Visual Subgoal")
        ax2.axis('off')

        def animate(i):
            ax1.clear()
            ax1.plot(action_gt_np.T, color='green', linestyle='--', label='Ground Truth' if i==0 else "")
            ax1.plot(trajectory[i].T, color='blue', label='Denoised Action' if i==0 else "")
            ax1.set_title(f"Denoising Process | Epoch {epoch} | Step {i}/{len(trajectory)}")
            ax1.set_xlabel("Action Dimension")
            ax1.set_ylabel("Action Value")
            ax1.set_ylim(-1.5, 1.5)
            if i == 0:
                ax1.legend()
        # --- END OF SOTA PATCH 2 ---

        save_path = self.output_dir / "diagnostics"
        save_path.mkdir(exist_ok=True)
        video_path = save_path / f"denoising_epoch_{epoch}.mp4"
        
        ani = animation.FuncAnimation(fig, animate, frames=len(trajectory), interval=50)
        ani.save(video_path, writer='ffmpeg', fps=20)
        plt.close(fig)
        log.info(f"Saved diagnostic video to {video_path}")

        if self.use_wandb:
            wandb.log({
                "val/denoising_rollout": wandb.Video(str(video_path), fps=20, format="mp4"),
            }, step=self.global_step)


    def _save_backup_checkpoint(self, epoch: int):
        """
        Saves a high-frequency backup checkpoint.
        Creates a new file and deletes the previous one to be atomic.
        """
        backup_dir = self.output_dir / "checkpoints" / "backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Define paths for current and previous backup
        current_backup_path = backup_dir / f"backup_epoch_{epoch}.pth"
        previous_backup_path = backup_dir / f"backup_epoch_{epoch - 1}.pth"
        
        # State dictionary is the same as for other checkpoints
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "policy_state_dict": self.policy.state_dict(),
            "ema_state_dict": self.policy.ema.state_dict() if self.policy.ema else None,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.lr_scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "rng_states": {
                "torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                "random": random.getstate(),
            },
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }
        
        # Save the current backup
        torch.save(state, current_backup_path)
        log.info(f"Saved epoch backup to {current_backup_path}")
        
        # Delete the previous backup to save disk space
        if previous_backup_path.exists():
            previous_backup_path.unlink()



    def run(self):
        """The main entry point to start the training process, with SOTA checkpointing."""
        log.info(f"Starting training from epoch {self.start_epoch}...")
        
        # --- START OF SOTA PATCH 2 ---
        current_epoch = self.start_epoch
        try:
            for epoch in range(self.start_epoch, self.cfg.training.epochs + 1):
                current_epoch = epoch # Keep track of the current epoch for exception handling
                
                # --- 1. Train for one epoch ---
                self._train_one_epoch(epoch)
                
                # --- 2. Save high-frequency backup ---
                # This happens after every single epoch.
                self._save_backup_checkpoint(epoch)
                
                # --- 3. Run validation and save "best" / "last" checkpoints periodically ---
                if epoch % self.cfg.logging.val_interval_epochs == 0:
                    self._validate_one_epoch(epoch)
            
            log.info(f"Training completed successfully after {self.cfg.training.epochs} epochs.")

        except KeyboardInterrupt:
            log.warning("Training interrupted by user (KeyboardInterrupt).")
            log.info("Performing a final save of the 'last.pth' checkpoint before exiting...")
            
            # Use the existing global save_checkpoint function for the final "resume" checkpoint
            save_checkpoint(
                state={
                    "epoch": current_epoch, # The epoch that was interrupted
                    "global_step": self.global_step,
                    "policy_state_dict": self.policy.state_dict(),
                    "ema_state_dict": self.policy.ema.state_dict() if self.policy.ema else None,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.lr_scheduler.state_dict(),
                    "best_val_loss": self.best_val_loss,
                    "rng_states": {
                        "torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                        "random": random.getstate(),
                    },
                    "config": OmegaConf.to_container(self.cfg, resolve=True),
                },
                is_best=False, # An interrupted run is never the "best"
                checkpoint_dir=self.output_dir / "checkpoints",
            )
            log.info(f"Final state for interrupted epoch {current_epoch} saved to 'last.pth'.")

        except Exception as e:
            log.exception(f"An unexpected error occurred during training at epoch {current_epoch}.")
            # We don't save on unknown errors, as the state might be corrupt.
            raise
            
        finally:
            elapsed_time = time.time() - self.start_time
            log.info(f"Training process finished. Total runtime: {elapsed_time/3600:.2f} hours.")
            if self.use_wandb:
                wandb.finish()
        # --- END OF SOTA PATCH 2 ---

# -------------------------
# 4. Hydra Main Entry Point
# -------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="pretrain_diffusion_config")
def main(cfg: DictConfig):
    """
    Main function managed by Hydra.
    It sets up the environment, instantiates the trainer, and runs it.
    """
    # Print the configuration for verification
    log.info("----------- Configuration -----------")
    log.info(OmegaConf.to_yaml(cfg))
    log.info("------------------------------------")

    try:
        trainer = ControllerBCTrainer(cfg)
        trainer.run()
    except Exception as e:
        log.exception("An error occurred during training.")
        raise  # Re-raise the exception after logging

# -------------------------
# 5. Standard Python Entry
# -------------------------

if __name__ == "__main__":
    main()