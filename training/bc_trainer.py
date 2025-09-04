# bc_trainer.py
"""
BCTrainer: Robust Behavioral Cloning trainer

Key robustness features:
- Flexible batch parsing + safe custom collate for dict-of-tensors observations
- Device-safe optimizer state restore; optional dataset-to-device flow
- AMP with NaN/Inf guards; optional grad clipping
- Configurable loss: mse|l1|ce (auto dtype handling)
- Deterministic seeding (best-effort), clean resume semantics
- Helpful shape/dtype checks and throughput logging
"""

from __future__ import annotations
import os
import argparse
import logging
import time
from typing import Any, Dict, List, Tuple, Union, Sequence, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.bc_policy import BCNet
from utils.expert_dataset import ExpertDataset

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("bc_trainer")


# -----------------------------------------------------------------------------
# Utilities
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


def _to_device(obj: Any, device: torch.device):
    """Recursively move tensors (or containers of tensors) to device."""
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, Mapping):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(x, device) for x in obj)
    return obj


def _move_optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device):
    """Move optimizer state tensors to device after loading state_dict()."""
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device)


def _stack_tensors(items: Sequence[torch.Tensor]) -> torch.Tensor:
    """Stack a sequence of tensors along dim 0, with sanity checks."""
    if not items:
        raise ValueError("Empty list passed to _stack_tensors.")
    ref = items[0]
    for i, t in enumerate(items):
        if not torch.is_tensor(t):
            raise TypeError(f"Non-tensor in batch at index {i}: {type(t)}")
        if t.shape != ref.shape:
            raise ValueError(f"Tensor shape mismatch in batch: {t.shape} vs {ref.shape}")
        if t.dtype != ref.dtype:
            # Allow safe upcast to float32 for mixed float types
            if not (t.is_floating_point() and ref.is_floating_point()):
                raise TypeError(f"Dtype mismatch in batch: {t.dtype} vs {ref.dtype}")
    try:
        return torch.stack(items, dim=0)
    except Exception as e:
        raise RuntimeError(f"Failed to stack tensors: {e}")


def _collate_obs(batch_obs: List[Union[Dict[str, Any], Tuple[Any, Any], torch.Tensor]]) -> Any:
    """
    Recursively collate a list of observation entries. Supports nested dictionaries.
    """
    first = batch_obs[0]
    # Base case: if the items are tensors or numpy arrays, stack them.
    if torch.is_tensor(first):
        return _stack_tensors(batch_obs)
    elif isinstance(first, np.ndarray):
        return torch.from_numpy(np.stack(batch_obs, axis=0))

    # Recursive case: if the items are dictionaries, collate each key's values.
    if isinstance(first, Mapping):
        return {k: _collate_obs([d[k] for d in batch_obs]) for k in first}
    
    # Handle other sequence types like tuples or lists
    if isinstance(first, (list, tuple)):
        # Transpose the list of tuples/lists
        transposed = zip(*batch_obs)
        return type(first)(_collate_obs(samples) for samples in transposed)

    raise TypeError(f"Unsupported observation type in batch: {type(first)}")

def _default_collate(batch: List[Any]) -> Tuple[Any, torch.Tensor]:
    """
    Robust collate_fn for ExpertDataset.
    Supports dataset __getitem__ returning either:
      - (obs_dict, action_tensor)
      - {"obs": obs_dict_or_tuple_or_tensor, "action": action_tensor}
    """
    if not isinstance(batch, list) or len(batch) == 0:
        raise RuntimeError("DataLoader batch is empty or invalid.")

    sample = batch[0]

    # Case 1: tuple/list (obs, action)
    if isinstance(sample, (list, tuple)) and len(sample) == 2:
        obs_list = [b[0] for b in batch]
        act_list = [b[1] for b in batch]
        obs = _collate_obs(obs_list)
        # stack actions
        if torch.is_tensor(act_list[0]):
            actions = _stack_tensors(act_list)
        elif isinstance(act_list[0], np.ndarray):
            actions = torch.from_numpy(np.stack(act_list, axis=0))
        else:
            raise TypeError(f"Unsupported action type: {type(act_list[0])}")
        return obs, actions

    # Case 2: dict with keys "obs" and "action"
    if isinstance(sample, Mapping):
        if "obs" in sample and "action" in sample:
            obs_list = [b["obs"] for b in batch]
            act_list = [b["action"] for b in batch]
            obs = _collate_obs(obs_list)
            if torch.is_tensor(act_list[0]):
                actions = _stack_tensors(act_list)
            elif isinstance(act_list[0], np.ndarray):
                actions = torch.from_numpy(np.stack(act_list, axis=0))
            else:
                raise TypeError(f"Unsupported action type: {type(act_list[0])}")
            return obs, actions

    raise RuntimeError("Unexpected batch structure. Expected (obs, action) tuple or {'obs': ..., 'action': ...} dict.")


def _infer_and_fix_action_dtype(actions: torch.Tensor, loss_type: str, preds_dtype: torch.dtype) -> torch.Tensor:
    """
    Ensure action tensor has the right dtype for the chosen loss.
    - 'ce': targets must be Long
    - regression losses: match prediction dtype (float32)
    """
    if loss_type == "ce":
        if actions.dtype != torch.long:
            actions = actions.long()
        return actions
    # regression
    if actions.dtype != preds_dtype:
        # safe cast for floating point regression
        if actions.is_floating_point():
            actions = actions.to(preds_dtype)
        else:
            actions = actions.float().to(preds_dtype)
    return actions


# -----------------------------------------------------------------------------
# BC Trainer
# -----------------------------------------------------------------------------
class BCTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config.copy()

        # Device
        dev_arg = self.config.get("device", "cpu")
        if isinstance(dev_arg, torch.device):
            self.device = dev_arg
        else:
            dev_str = str(dev_arg).lower()
            self.device = torch.device("cuda" if (dev_str == "auto" and torch.cuda.is_available()) else dev_str if dev_str != "auto" else "cpu")

        # Optional: improve matmul perf on Ampere+ (safe on CPU too)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        # Seed
        self.seed = int(self.config.get("seed", 42))
        set_seeds(self.seed)

        # Model (allow passing extra BCNet kwargs via 'bc_kwargs')
        n_actions = int(self.config.get("n_actions", 8))
        proprio_dim = int(self.config.get("proprio_dim", 14))
        bc_kwargs = dict(self.config.get("bc_kwargs", {}))
        self.model = BCNet(n_actions=n_actions, proprio_dim=proprio_dim, **bc_kwargs).to(self.device)

        # Optimizer + optional weight decay
        lr = float(self.config.get("learning_rate", self.config.get("lr", 1e-4)))
        weight_decay = float(self.config.get("weight_decay", 0.0))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Loss
        self.loss_type = str(self.config.get("loss_type", "mse")).lower()
        if self.loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif self.loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif self.loss_type == "ce":
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss_type={self.loss_type}")

        # AMP
        self.use_amp = (self.device.type == "cuda")
        self.autocast_kwargs = dict(enabled=self.use_amp)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Optional grad clipping
        self.grad_clip_norm = float(self.config.get("grad_clip_norm", 0.0))  # 0.0 disables

        # Early stopping
        self.early_stop_patience = int(self.config.get("early_stop_patience", 0))  # 0 disables
        self._epochs_since_improve = 0

        # Checkpointing state
        self.start_epoch = 0
        self.best_loss = float("inf")

        # Dataset/device policy
        self.move_dataset_to_device = bool(self.config.get("move_dataset_to_device", False))

        logger.info("BCTrainer initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Loss: {self.loss_type.upper()}")
        logger.info(f"  AMP enabled: {self.use_amp}")
        logger.info(f"  Grad clip L2 norm: {self.grad_clip_norm if self.grad_clip_norm > 0 else 'disabled'}")
        logger.info(f"  Model params: {sum(p.numel() for p in self.model.parameters()):,}")

    # -------------------------------------------------------------------------
    def _get_dataloader(self) -> DataLoader:
        dataset = ExpertDataset(
            urdf_path=self.config["urdf_path"],
            instruction=self.config.get("instruction", "pick up the red block"),
            device=self.device if self.move_dataset_to_device else None,
            max_samples_per_epoch=int(self.config.get("steps_per_epoch", 1000)),
        )
        try:
            ds_len = len(dataset)
        except Exception:
            ds_len = -1
        if ds_len >= 0:
            logger.info(f"Dataset size (samples/epoch): {ds_len}")
        else:
            logger.info("Dataset size unknown (IterableDataset?).")

        batch_size = int(self.config.get("batch_size", 64))
        num_workers = int(self.config.get("num_workers", 0))  # 0 = deterministic & simplest
        pin_memory = (self.device.type == "cuda") and not self.move_dataset_to_device
        drop_last = bool(self.config.get("drop_last", False))
        prefetch_factor = None if num_workers == 0 else int(self.config.get("prefetch_factor", 2))

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=_default_collate,
            prefetch_factor=prefetch_factor,
        )

    # -------------------------------------------------------------------------
    def save_checkpoint(self, path: str, epoch_completed: int, loss: float):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ckpt = {
            "epoch": int(epoch_completed),  # number of epochs fully completed
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": float(loss),
            "config": self.config,
        }
        torch.save(ckpt, path)
        logger.info(f"Checkpoint saved to {path} | epoch_completed={epoch_completed} | best_loss={loss:.6f}")

    def load_checkpoint(self, path: str):
        if not os.path.exists(path):
            logger.warning(f"No checkpoint at {path}, starting fresh.")
            return
        ckpt = torch.load(path, map_location=self.device)
        model_sd = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(model_sd, strict=False)

        if "optimizer_state_dict" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                _move_optimizer_to_device(self.optimizer, self.device)
            except Exception as e:
                logger.warning(f"Optimizer state not fully restored: {e}")

        # epoch is "completed", so we resume starting at that index
        self.start_epoch = int(ckpt.get("epoch", 0))
        self.best_loss = float(ckpt.get("best_loss", float('inf')))
        logger.info(f"Loaded checkpoint {path} | epochs_completed={self.start_epoch} | best_loss={self.best_loss:.6f}")

    # -------------------------------------------------------------------------
    def _train_one_epoch(self, data_loader: DataLoader) -> float:
        self.model.train()
        epoch_loss = 0.0
        batch_count = 0
        start_time = time.time()

        for obs_batch, action_batch in tqdm(data_loader, desc="  Training", leave=False):
            # Device transfer (if dataset didn't already)
            if not self.move_dataset_to_device:
                obs_batch = _to_device(obs_batch, self.device)
                action_batch = _to_device(action_batch, self.device)

            # Forward pass
            with torch.cuda.amp.autocast(**self.autocast_kwargs):
                preds = self.model(obs_batch)

                # Validate shapes: preds should match actions for regression
                if self.loss_type in ("mse", "l1"):
                    if preds.shape != action_batch.shape:
                        raise RuntimeError(
                            f"Prediction/target shape mismatch: preds {tuple(preds.shape)} vs actions {tuple(action_batch.shape)}"
                        )

                # Fix dtype for loss
                action_batch = _infer_and_fix_action_dtype(action_batch, self.loss_type, preds.dtype)
                loss = self.loss_fn(preds, action_batch)

            # Guard against NaN/Inf
            if not torch.isfinite(loss):
                logger.warning("Non-finite loss detected; skipping this batch.")
                continue

            # Backprop
            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                self.scaler.scale(loss).backward()
                # Optional grad clipping
                if self.grad_clip_norm and self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip_norm and self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            epoch_loss += float(loss.item())
            batch_count += 1

        elapsed = time.time() - start_time
        if batch_count == 0:
            logger.warning("No batches were processed this epoch. Check dataset length / steps_per_epoch / collate.")
            return float("inf")

        avg = epoch_loss / batch_count
        samples = batch_count * data_loader.batch_size
        sps = samples / max(elapsed, 1e-6)
        logger.info(f"    avg_loss={avg:.6f} | samples={samples} | time={elapsed:.2f}s | {sps:.1f} samples/s")
        return avg

    # -------------------------------------------------------------------------
    def train(self):
        data_loader = self._get_dataloader()

        num_epochs = int(self.config.get("num_epochs", 50))
        save_path = self.config.get("save_path", "trained_models/policy_pretrained_bc.pth")
        save_dir = os.path.dirname(save_path) or "."
        os.makedirs(save_dir, exist_ok=True)
        best_path = os.path.join(save_dir, "best_model.pth")

        # Auto-resume if configured path exists (optional convenience)
        auto_resume = bool(self.config.get("auto_resume", True))
        resume_path = os.path.join(save_dir, "resume_checkpoint.pth")
        if auto_resume and os.path.exists(resume_path) and self.start_epoch == 0:
            logger.info(f"Auto-resuming from {resume_path}")
            self.load_checkpoint(resume_path)

        logger.info(f"Starting training for {num_epochs} epochs (resuming at epoch index {self.start_epoch})")

        try:
            for epoch in range(self.start_epoch, num_epochs):
                logger.info(f"Epoch {epoch+1}/{num_epochs}")
                avg_loss = self._train_one_epoch(data_loader)

                improved = np.isfinite(avg_loss) and avg_loss < self.best_loss
                if improved:
                    self.best_loss = avg_loss
                    self._epochs_since_improve = 0
                    self.save_checkpoint(best_path, epoch_completed=epoch + 1, loss=avg_loss)
                    logger.info(f"  ✓ New best model saved to {best_path}")
                else:
                    self._epochs_since_improve += 1

                # Early stopping
                if self.early_stop_patience > 0 and self._epochs_since_improve >= self.early_stop_patience:
                    logger.info(f"Early stopping triggered (no improvement for {self.early_stop_patience} epochs).")
                    break

            # final checkpoint (epochs completed may be < num_epochs if early-stopped)
            completed = min(num_epochs, self.start_epoch + (epoch - self.start_epoch + 1))
            self.save_checkpoint(save_path, epoch_completed=completed, loss=self.best_loss)
            logger.info(f"Training complete. Final model saved to {save_path}")

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user. Saving resume checkpoint...")
            # Save with number of fully completed epochs so far
            epochs_completed = (epoch if 'epoch' in locals() else self.start_epoch)
            self.save_checkpoint(resume_path, epoch_completed=epochs_completed, loss=self.best_loss)
            raise
        except Exception:
            logger.exception("Fatal error during training!")
            try:
                err_path = os.path.join(save_dir, "resume_on_error.pth")
                epochs_completed = (epoch if 'epoch' in locals() else self.start_epoch)
                self.save_checkpoint(err_path, epoch_completed=epochs_completed, loss=self.best_loss)
            except Exception:
                logger.error("Failed to save checkpoint after error.")
            raise


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("Behavioral Cloning trainer")
    p.add_argument("--urdf_path", type=str, default="urdf/panda.urdf")
    p.add_argument("--save_path", type=str, default="trained_models/policy_pretrained_bc.pth")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--steps_per_epoch", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--loss_type", type=str, default="mse", help="mse|l1|ce")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip_norm", type=float, default=0.0, help="0 disables")
    p.add_argument("--early_stop_patience", type=int, default=0, help="0 disables")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--drop_last", action="store_true")
    p.add_argument("--move_dataset_to_device", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # --- Auto-infer model dimensions from a dummy environment ---
    # This is more robust than hard-coding dimensions.
    logger.info("Creating a temporary environment to infer model dimensions...")
    from envs.panda_env import PandaEnv # Assuming this is your env
    xml_path_for_inference = "envs/panda_pick_place.xml"
    if not os.path.exists(xml_path_for_inference):
        raise FileNotFoundError(
            f"Could not find '{xml_path_for_inference}' for model dimension inference. "
            f"Please run this script from the project root."
        )
    temp_env = PandaEnv(xml_path=xml_path_for_inference)
    obs, _ = temp_env.reset()
    
    action_dim = temp_env.action_space.shape[0]
    proprio_dim = obs["proprio"].shape[0]
    
    logger.info(f"Inferred action_dim={action_dim}, proprio_dim={proprio_dim}")
    temp_env.close()
    # --- End of inference ---

    config = {
        "urdf_path": args.urdf_path,
        "instruction": "pick up the red block",
        "save_path": args.save_path,

        # Model I/O dims (now inferred automatically)
        "n_actions": action_dim,
        "proprio_dim": proprio_dim,

        # Training hyperparams
        "num_epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip_norm": args.grad_clip_norm,
        "early_stop_patience": args.early_stop_patience,

        # System
        "device": args.device,
        "seed": args.seed,
        "loss_type": args.loss_type,
        "num_workers": args.num_workers,
        "drop_last": args.drop_last,
        "move_dataset_to_device": args.move_dataset_to_device,
    }

    trainer = BCTrainer(config)
    trainer.train()