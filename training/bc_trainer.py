# bc_trainer.py
"""
BCTrainer: A pure, robust, and reusable training engine for Behavioral Cloning.

This engine is project-agnostic and accepts a pre-built model, optimizer, 
and loss function. It handles the complete training lifecycle, including:
- AMP with NaN/Inf guards and optional gradient clipping.
- A robust training loop with detailed progress logging.
- Checkpointing for best model, final model, and safe resume on interruption.
- Deterministic seeding and device management.
"""

from __future__ import annotations
import os
import logging
import time
from typing import Any, Dict, Union, Mapping, Optional

import numpy as np
import torch
from torch import autocast 
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.loggers import BaseLogger 
from pathlib import Path

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("bc_trainer")


# -----------------------------------------------------------------------------
# Generic Utilities (Safe to keep here)
# -----------------------------------------------------------------------------
def set_seeds(seed: int):
    """Best-effort determinism."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

def _move_optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device):
    """Move optimizer state tensors to device after loading state_dict()."""
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device)

def _to_device(obj: Any, device: torch.device):
    """Recursively move tensors (or containers of tensors) to device."""
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, Mapping):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(x, device) for x in obj)
    return obj

def _extract_pred_actions(model_out: Any) -> torch.Tensor:
    """Robustly extracts predicted actions from a model's output."""
    if isinstance(model_out, torch.Tensor):
        pred = model_out
    elif isinstance(model_out, dict):
        pred = None
        for key in ("action", "mu", "mean", "logits", "out"):
            if key in model_out and isinstance(model_out[key], torch.Tensor):
                pred = model_out[key]
                break
        if pred is None:
            raise RuntimeError(f"Could not find a valid action tensor in model output. Keys: {list(model_out.keys())}")
    else:
        raise TypeError(f"Unsupported model output type: {type(model_out)}")

    if pred.ndim == 3 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    if pred.ndim != 2:
        raise ValueError(f"Expected a 2D action tensor (B, A), but got shape {pred.shape} after processing.")
    return pred


# -----------------------------------------------------------------------------
# BC Trainer Engine
# -----------------------------------------------------------------------------
class BCTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        is_classification_loss: bool,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, 
        metrics_logger: Optional[BaseLogger] = None,
        *,
        device: Union[str, torch.device] = "auto",
        use_amp: bool = True,
        grad_clip_norm: float = 0.0,
        early_stop_patience: int = 0,
        seed: int = 42,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.metrics_logger = metrics_logger

        
        self.is_classification = is_classification_loss
        if isinstance(device, torch.device): self.device = device
        else:
            dev_str = str(device).lower()
            self.device = torch.device("cuda" if (dev_str == "auto" and torch.cuda.is_available()) else dev_str if dev_str != "auto" else "cpu")
        
        self.model.to(self.device)
        self.seed = seed
        set_seeds(self.seed)

        self.use_amp = use_amp and self.device.type == "cuda"
        self.autocast_device_type = "cuda" if self.device.type == "cuda" else "cpu"
        self.scaler = torch.amp.GradScaler(device=self.device, enabled=self.use_amp)

        self.grad_clip_norm = grad_clip_norm
        self.early_stop_patience = early_stop_patience
        self._epochs_since_improve = 0
        self.start_epoch = 0
        self.best_loss = float("inf")
        
        logger.info("BCTrainer initialized as a pure engine")
        logger.info(f"  Device: {self.device} | AMP enabled: {self.use_amp}")
        logger.info(f"  Model params: {sum(p.numel() for p in self.model.parameters()):,}")
    def save_checkpoint(self, path: Union[str, Path], epoch_completed: int, loss: float):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ckpt = {
            "epoch": int(epoch_completed),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": float(loss),
        }
        if self.scheduler:
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(ckpt, path)
        logger.info(f"Checkpoint saved to {path} | epoch_completed={epoch_completed} | best_loss={loss:.6f}")

    def load_checkpoint(self, path: str):
        if not os.path.exists(path):
            logger.warning(f"No checkpoint at {path}, starting fresh.")
            return
        
        # Load the checkpoint dictionary FIRST
        ckpt = torch.load(path, map_location=self.device)
        
        # Now safely access the contents of ckpt
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)

        if "optimizer_state_dict" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                _move_optimizer_to_device(self.optimizer, self.device)
            except Exception as e:
                logger.warning(f"Optimizer state not fully restored: {e}")

        if "scheduler_state_dict" in ckpt and self.scheduler:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception as e:
                logger.warning(f"Scheduler state not fully restored: {e}")

        self.start_epoch = int(ckpt.get("epoch", 0))
        self.best_loss = float(ckpt.get("best_loss", float('inf')))
        logger.info(f"Loaded checkpoint {path} | epochs_completed={self.start_epoch} | best_loss={self.best_loss:.6f}")

    def _validate_one_epoch(self, data_loader: DataLoader, pbar_desc: str) -> float:
        """Runs a validation loop for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        pbar = tqdm(data_loader, desc=pbar_desc, leave=False)

        with torch.no_grad():
            for obs_batch, action_batch in pbar:
                obs_batch = _to_device(obs_batch, self.device)
                action_batch = _to_device(action_batch, self.device)
                current_bsz = action_batch.shape[0]

                with autocast(device_type=self.autocast_device_type, enabled=self.use_amp):
                    model_out = self.model(obs_batch)
                    preds = _extract_pred_actions(model_out)
                    
                    if not self.is_classification and action_batch.dtype != preds.dtype:
                        action_batch = action_batch.to(preds.dtype)

                    loss = self.loss_fn(preds, action_batch)
                
                if torch.isfinite(loss):
                    total_loss += loss.item() * current_bsz
                    total_samples += current_bsz

        if total_samples == 0:
            logger.warning("No samples were processed during validation.")
            return float("inf")
        
        avg_loss = total_loss / total_samples
        logger.info(f"    Validation avg_loss={avg_loss:.6f}")
        return avg_loss
    def _train_one_epoch(self, data_loader: DataLoader, pbar_desc: str) -> float:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        start_time = time.time()
        pbar = tqdm(data_loader, desc=pbar_desc, leave=False)

        for obs_batch, action_batch in pbar:
            # CORRECT: The trainer is always responsible for moving data to its device
            obs_batch = _to_device(obs_batch, self.device)
            action_batch = _to_device(action_batch, self.device)
            
            current_bsz = action_batch.shape[0]

            with autocast(device_type=self.autocast_device_type, enabled=self.use_amp):
                model_out = self.model(obs_batch)
                preds = _extract_pred_actions(model_out)

                if not self.is_classification:
                    if preds.shape != action_batch.shape:
                        raise RuntimeError(f"Shape mismatch: preds {preds.shape} vs targets {action_batch.shape}")
                    if action_batch.dtype != preds.dtype:
                        action_batch = action_batch.to(preds.dtype)
                
                loss = self.loss_fn(preds, action_batch)

            if not torch.isfinite(loss):
                logger.warning("Non-finite loss detected; skipping batch.")
                continue

            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            total_loss += loss.item() * current_bsz
            total_samples += current_bsz

        elapsed = time.time() - start_time
        if total_samples == 0:
            logger.warning("No samples were processed this epoch.")
            return float("inf")

        avg_loss = total_loss / total_samples
        sps = total_samples / max(elapsed, 1e-6)
        logger.info(f"    avg_loss={avg_loss:.6f} | samples={total_samples} | time={elapsed:.2f}s | {sps:.1f} samples/s")
        return avg_loss

# In your BCTrainer class, replace the old fit method with this one:

    def fit(
        self,
        build_dataloaders_fn,
        dataloader_args: dict,
        epochs: int,
        run_dir: Path,
    ):
        """
        Main training loop that dynamically generates fresh data for each epoch.
        Saves artifacts to a standardized directory structure within the provided run_dir.
        """
        # --- 1. Create the standardized directory structure ---
        checkpoints_dir = run_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        # Note: The logs/ directory is created automatically by the TensorBoardLogger.

        # --- 2. Define artifact paths within the new structure ---
        final_path = checkpoints_dir / "final_model.pth"
        best_path = checkpoints_dir / "best_model.pth"
        resume_path = checkpoints_dir / "resume_checkpoint.pth"
        error_ckpt_path = checkpoints_dir / "resume_on_error.pth"

        try:
            # The loop correctly starts from the last completed epoch
            for epoch in range(self.start_epoch, epochs):
                
                # --- DYNAMIC DATALOADER CREATION FOR THIS EPOCH ---
                # This is the core of the fix. New, fresh data is generated every epoch.
                logger.info(f"--- Epoch {epoch + 1}/{epochs} ---")
                logger.info("Building new dataloaders with fresh, randomized data...")
                
                # Create a copy of the base arguments and inject the current epoch number
                # to ensure a unique seed for the ExpertDataset.
                current_epoch_dataloader_args = dataloader_args.copy()
                current_epoch_dataloader_args['epoch'] = epoch + 1
                
                # Call the provided function to build the dataloaders for this specific epoch
                train_loader, val_loader = build_dataloaders_fn(**current_epoch_dataloader_args)
                
                # --- Train Step (using the new train_loader) ---
                train_pbar_desc = f"Train Epoch {epoch + 1}/{epochs}"
                train_loss = self._train_one_epoch(train_loader, train_pbar_desc)

                # --- Validation Step (using the new val_loader) ---
                val_loss = None
                if val_loader:
                    val_pbar_desc = f"Val Epoch   {epoch + 1}/{epochs}"
                    val_loss = self._validate_one_epoch(val_loader, val_pbar_desc)
                
                # Use validation loss for saving best model if available, otherwise use train loss
                current_metric = val_loss if val_loader else train_loss
                
                # --- Step the scheduler ---
                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(current_metric)
                    else:
                        self.scheduler.step()

                # --- Checkpointing and Early Stopping ---
                # This logic remains the same, but it's now acting on results from fresh data.
                if np.isfinite(current_metric) and current_metric < self.best_loss:
                    self.best_loss = current_metric
                    self._epochs_since_improve = 0
                    self.save_checkpoint(best_path, epoch_completed=epoch + 1, loss=self.best_loss)
                    logger.info(f"  ✓ New best model saved (metric={self.best_loss:.6f})")
                else:
                    self._epochs_since_improve += 1
                
                # The regular end-of-epoch resume checkpoint
                self.save_checkpoint(resume_path, epoch_completed=epoch + 1, loss=self.best_loss)

                if val_loader and self.early_stop_patience > 0 and self._epochs_since_improve >= self.early_stop_patience:
                    logger.info(f"Early stopping triggered after {self._epochs_since_improve} epochs with no validation improvement.")
                    break
                
                if self.metrics_logger:
                    metrics = {"train/loss": train_loss}
                    if val_loss is not None:
                        metrics["val/loss"] = val_loss
                    self.metrics_logger.log_metrics(metrics, step=epoch + 1) # Use epoch+1 for correct step
        
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user. Saving emergency resume checkpoint...")
            if self.metrics_logger: 
                self.metrics_logger.finish()
            
            # Save state from the last FULLY completed epoch
            epochs_completed = (epoch) if 'epoch' in locals() and epoch >= self.start_epoch else self.start_epoch
            self.save_checkpoint(error_ckpt_path, epoch_completed=epochs_completed, loss=self.best_loss)
            logger.info(f"Emergency checkpoint for epoch {epochs_completed} saved to {error_ckpt_path}.")
            raise

        except Exception:
            logger.exception("Fatal error during training! Saving emergency checkpoint...")
            if self.metrics_logger: 
                self.metrics_logger.finish()
            
            # Save state from the last FULLY completed epoch
            epochs_completed = (epoch) if 'epoch' in locals() and epoch >= self.start_epoch else self.start_epoch
            try:
                self.save_checkpoint(error_ckpt_path, epoch_completed=epochs_completed, loss=self.best_loss)
                logger.info(f"Emergency checkpoint for epoch {epochs_completed} saved to {error_ckpt_path}.")
            except Exception:
                logger.error("Failed to save checkpoint after fatal error.")
            raise
        
        else: # This block runs only if the loop completes without a break or exception
            logger.info(f"Training complete. Final model state is from epoch {epochs}.")
            # The 'final_model.pth' is just a copy of the last 'resume_checkpoint.pth'
            if resume_path.is_file():
                import shutil
                shutil.copy(resume_path, final_path)
                logger.info(f"Final model saved to {final_path}")