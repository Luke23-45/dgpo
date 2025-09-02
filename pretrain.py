#!/usr/bin/env python3
#!/pretrain_policy
"""
Unified pretraining entry point (ExpertDataset + BCTrainer + optional SB3 export)

This script consolidates the project’s Phase-1 components into a single, robust CLI:

  1) Builds an ExpertDataset that generates OCTO/IK expert actions on-the-fly.
  2) Trains a BCNet policy via BCTrainer (preferred) or a safe built-in fallback loop.
  3) Saves a .pth checkpoint of BCNet (supervised BC weights).
  4) Optionally instantiates an SB3 PPO agent, transfers compatible weights, and saves .zip.

Design goals:
 - One canonical pretraining path (no parallel/duplicate pipelines).
 - Deterministic seeding + strong runtime validations (shapes/devices).
 - Gentle failure modes: if BCTrainer is missing/incompatible, use a local trainer.
 - Clear, explicit logging at every critical step.

Outputs:
 - BC weights:   --save-bc "path/to/model.pth"
 - SB3 PPO .zip: --save-sb3 "path/to/agent.zip" (requires Stable Baselines3 + PandaEnv)
"""

from __future__ import annotations

import os
import sys
import gc
import math
import time
import json
import random
import logging
import argparse
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# ---------------------------
# Project imports (Phase-1)
# ---------------------------
from utils.expert_dataset import ExpertDataset
from models.bc_policy import BCNet

# Prefer your BCTrainer, but keep a robust fallback if not present or incompatible
_BCTrainer = None
try:
    # Expect bc_trainer.py to define a BCTrainer class with .fit(...) or similar
    from bc_trainer import BCTrainer as _BCTrainer  # type: ignore
except Exception as _e:
    _BCTrainer = None

# Optional imports only needed if exporting to SB3
def _maybe_import_sb3() -> Tuple[Optional[object], Optional[object], Optional[object]]:
    try:
        from stable_baselines3 import PPO
        from envs.panda_env import PandaEnv
        from utils.transfer_bc_weights import transfer_bc_weights
        return PPO, PandaEnv, transfer_bc_weights
    except Exception:
        return None, None, None


logger = logging.getLogger("pretrain_policy")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

# ---------------------------
# Utilities
# ---------------------------

def set_global_seed(seed: int) -> None:
    """Deterministic-ish setup. (Note: Full determinism depends on env/backends.)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # cuDNN flags
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _infer_action_dim_from_batch(model: torch.nn.Module, batch_obs: Dict[str, torch.Tensor]) -> int:
    """
    Run a no-grad forward to infer the action dimension, regardless of BCNet's return format.
    Supports:
      - returning a Tensor of shape (B, A)
      - returning a dict with keys like 'action', 'mu', 'mean'
    """
    model.eval()
    with torch.no_grad():
        out = model(batch_obs)

    if isinstance(out, torch.Tensor):
        if out.ndim != 2:
            raise RuntimeError(f"BCNet forward returned tensor with shape {tuple(out.shape)}, expected (B, A).")
        return int(out.shape[1])

    if isinstance(out, dict):
        for k in ("action", "mu", "mean", "logits", "out"):
            if k in out and isinstance(out[k], torch.Tensor) and out[k].ndim == 2:
                return int(out[k].shape[1])
        raise RuntimeError(
            "BCNet forward returned a dict but no (B, A) tensor was found under "
            "keys ('action','mu','mean','logits','out')."
        )

    raise RuntimeError(f"BCNet forward returned unsupported type: {type(out)}")


def _extract_pred_actions(model_out: Any) -> torch.Tensor:
    """Normalize BCNet outputs into a (B, A) float tensor for loss computation."""
    if isinstance(model_out, torch.Tensor):
        return model_out
    if isinstance(model_out, dict):
        for k in ("action", "mu", "mean", "logits", "out"):
            v = model_out.get(k, None)
            if isinstance(v, torch.Tensor) and v.ndim == 2:
                return v
    raise RuntimeError("Unable to extract predicted actions from model output.")


class _FallbackBCTrainer:
    """
    Minimal, safe BC trainer used only if your bc_trainer.BCTrainer is missing/incompatible.
    - Supervised MSE on actions (common + robust).
    - Gradient clipping + optional weight decay.
    - Logs running loss.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = 1.0,
        amp: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.optim = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.grad_clip_norm = grad_clip_norm
        self.amp = bool(amp)
        self._scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))
        self._loss_fn = torch.nn.MSELoss()

    def fit(
        self,
        loader: DataLoader,
        epochs: int,
        log_interval: int = 50,
    ) -> None:
        self.model.train()
        for ep in range(1, epochs + 1):
            running = 0.0
            count = 0
            t0 = time.time()
            for it, (obs_dict, expert_action) in enumerate(loader, start=1):
                obs = {k: v.to(self.device, non_blocking=True) for k, v in obs_dict.items()}
                target = expert_action.to(self.device, non_blocking=True)

                self.optim.zero_grad(set_to_none=True)
                if self.amp:
                    with torch.cuda.amp.autocast():
                        out = self.model(obs)
                        pred = _extract_pred_actions(out)
                        loss = self._loss_fn(pred, target)
                    self._scaler.scale(loss).backward()
                    if self.grad_clip_norm is not None:
                        self._scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self._scaler.step(self.optim)
                    self._scaler.update()
                else:
                    out = self.model(obs)
                    pred = _extract_pred_actions(out)
                    loss = self._loss_fn(pred, target)
                    loss.backward()
                    if self.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.optim.step()

                running += float(loss.item())
                count += 1
                if it % log_interval == 0:
                    logger.info(f"[Epoch {ep}] iter {it} | loss {running / max(1, count):.6f}")

            logger.info(f"Epoch {ep} done in {time.time() - t0:.1f}s | avg loss {running / max(1, count):.6f}")


def _build_dataloader(
    *,
    urdf_path: str,
    instruction: str,
    octo_model_name: str,
    env_xml_path: Optional[str],
    base_seed: Optional[int],
    warmup: bool,
    samples_per_epoch: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device_push_inside_ds: bool,
    device: torch.device,
) -> DataLoader:
    """
    Construct ExpertDataset + DataLoader with a finite epoch size via max_samples_per_epoch.
    """
    ds = ExpertDataset(
        urdf_path=urdf_path,
        instruction=instruction,
        octo_model_name=octo_model_name,
        env_xml_path=env_xml_path,
        base_seed=base_seed,
        move_to_device=device_push_inside_ds,
        device=device if device_push_inside_ds else None,
        octo_pad_mask=np.array([[False, True]], dtype=bool),
        max_samples_per_epoch=int(samples_per_epoch),
        skip_on_error=True,
        warmup=warmup,
    )
    # Recommended: num_workers=0 for Mujoco/OCTO stability. Expose argument for experts.
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,              # IterableDataset; shuffling happens at the source
        num_workers=num_workers,
        pin_memory=pin_memory and (device.type == "cuda"),
        drop_last=True,             # stabilize batch shapes if samples_per_epoch not divisible
    )
    return loader


def _instantiate_bcnet(
    device: torch.device,
    model_ctor_overrides: Optional[str] = None,
) -> BCNet:
    """
    Instantiate BCNet with optional JSON overrides for constructor kwargs.
    This avoids hardcoding architecture details while remaining explicit.
    """
    ctor_kwargs: Dict[str, Any] = {}
    if model_ctor_overrides:
        try:
            ctor_kwargs = json.loads(model_ctor_overrides)
            assert isinstance(ctor_kwargs, dict)
        except Exception as e:
            raise ValueError(f"--model-ctor-overrides must be a JSON object; got: {model_ctor_overrides}") from e

    model = BCNet(**ctor_kwargs)  # your BCNet should handle defaults
    model.to(device)
    logger.info("BCNet created with kwargs=%s | device=%s", ctor_kwargs, device)
    return model


def _train_with_preferred_or_fallback_trainer(
    model: BCNet,
    device: torch.device,
    loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip_norm: Optional[float],
    amp: bool,
    log_interval: int,
) -> None:
    """
    Try your BCTrainer first; if missing/incompatible, use local safe fallback.
    We duck-type the interface to avoid brittle assumptions.
    """
    if _BCTrainer is not None:
        try:
            logger.info("Using project BCTrainer.")
            # Try common constructor signatures:
            # 1) BCTrainer(model, device=..., lr=..., ...)
            # 2) BCTrainer(model=model, ...)
            # 3) BCTrainer(...) where model is set later -> then call .set_model(model)
            try:
                trainer = _BCTrainer(model=model, device=device, lr=lr, weight_decay=weight_decay,
                                     grad_clip_norm=grad_clip_norm, amp=amp)
            except TypeError:
                # Try a simpler signature
                trainer = _BCTrainer(model, device=device, lr=lr)
                # If available, set extra attrs
                if hasattr(trainer, "weight_decay"):
                    setattr(trainer, "weight_decay", weight_decay)
                if hasattr(trainer, "grad_clip_norm"):
                    setattr(trainer, "grad_clip_norm", grad_clip_norm)
                if hasattr(trainer, "amp"):
                    setattr(trainer, "amp", amp)

            # Prefer a 'fit' method
            if hasattr(trainer, "fit"):
                trainer.fit(loader=loader, epochs=epochs, log_interval=log_interval)
                return
            # Or 'train' method
            if hasattr(trainer, "train"):
                trainer.train(loader=loader, epochs=epochs, log_interval=log_interval)
                return

            logger.warning("BCTrainer found but has no .fit/.train; falling back to local trainer.")

        except Exception as e:
            logger.exception("BCTrainer path failed; using fallback trainer. Reason: %s", e)

    # Fallback path
    logger.info("Using built-in fallback BC trainer (MSE).")
    fb = _FallbackBCTrainer(
        model=model, device=device, lr=lr, weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm, amp=amp
    )
    fb.fit(loader=loader, epochs=epochs, log_interval=log_interval)


def _save_bc_weights(model: BCNet, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info("Saved BC weights to %s", path)


def _export_to_sb3_zip(
    *,
    bc_state_dict: Dict[str, torch.Tensor],
    save_path_zip: str,
    ppo_kwargs_json: Optional[str],
) -> None:
    """
    Create an SB3 PPO agent, transfer BC weights, and save a .zip archive.
    """
    PPO, PandaEnv, transfer_bc_weights = _maybe_import_sb3()
    if PPO is None or PandaEnv is None or transfer_bc_weights is None:
        raise RuntimeError("SB3 export requested but stable_baselines3/PandaEnv/transfer function are unavailable.")

    # Instantiate env + agent
    env = PandaEnv()
    ppo_kwargs = {}
    if ppo_kwargs_json:
        try:
            ppo_kwargs = json.loads(ppo_kwargs_json)
            assert isinstance(ppo_kwargs, dict)
        except Exception as e:
            raise ValueError(f"--ppo-kwargs must be a JSON object; got: {ppo_kwargs_json}") from e

    logger.info("Creating PPO agent with kwargs=%s", ppo_kwargs)
    agent = PPO("MultiInputPolicy", env, verbose=0, **ppo_kwargs)

    # Transfer compatible weights
    report = transfer_bc_weights(bc_state_dict, agent, allow_shape_only_fallback=False, verbose=True)
    if not report.get("loaded_ok", False):
        logger.warning("transfer_bc_weights did not report loaded_ok=True. Review the transfer report above.")

    # Save .zip
    os.makedirs(os.path.dirname(save_path_zip) or ".", exist_ok=True)
    agent.save(save_path_zip)
    logger.info("Saved SB3 PPO agent to %s", save_path_zip)

    # Cleanup
    try:
        env.close()
    except Exception:
        pass
    del agent
    gc.collect()


def _sanity_check_batch(
    model: BCNet,
    device: torch.device,
    loader: DataLoader,
) -> Tuple[int, int]:
    """
    Pull one batch, ensure forward compat, and return (B, action_dim).
    """
    it = iter(loader)
    obs_dict, expert_action = next(it)  # raises if loader misconfigured

    # Move a small batch to device to probe shapes
    obs_small = {k: v[:2].to(device) for k, v in obs_dict.items()}
    target_small = expert_action[:2].to(device)

    # Run forward to detect action dims
    action_dim_pred = _infer_action_dim_from_batch(model, obs_small)
    if target_small.ndim != 2:
        raise RuntimeError(f"Expert action batch must be (B, A). Got shape: {tuple(target_small.shape)}")
    action_dim_target = int(target_small.shape[1])

    if action_dim_pred != action_dim_target:
        raise RuntimeError(
            f"Action dimension mismatch between BCNet({action_dim_pred}) and dataset({action_dim_target})."
        )

    logger.info("Sanity check OK | batch=%d | action_dim=%d", int(target_small.shape[0]), action_dim_pred)
    return int(target_small.shape[0]), action_dim_pred


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified BC pretraining (ExpertDataset + BCTrainer) with optional SB3 export.")
    # Dataset / OCTO / IK
    p.add_argument("--urdf_path", type=str, required=True, help="URDF path for IKSolver.")
    p.add_argument("--instruction", type=str, default="pick up the red block", help="Natural language instruction for OCTO.")
    p.add_argument("--octo_model_name", type=str, default="hf://rail-berkeley/octo-small-1.5", help="Octo model identifier.")
    p.add_argument("--env_xml_path", type=str, default=None, help="Optional MuJoCo XML for PandaEnv.")
    p.add_argument("--base_seed", type=int, default=1234, help="Base RNG seed for dataset/model.")
    p.add_argument("--warmup", action="store_true", help="Run one warmup OCTO inference per worker to reduce first-sample latency.")
    p.add_argument("--samples_per_epoch", type=int, default=4096, help="Finite samples per epoch (ExpertDataset.max_samples_per_epoch).")

    # Training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0, help="Prefer 0 for Mujoco stability.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", help="Enable mixed precision (CUDA only).")
    p.add_argument("--log_interval", type=int, default=50)

    # Model / PPO knobs
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model-ctor-overrides", type=str, default=None,
                   help='JSON dict with kwargs for BCNet constructor. Example: \'{"image_channels":3,"proprio_dim":14}\'')

    # Artifacts
    p.add_argument("--artifact", type=str, default="both", choices=("pth", "zip", "both"),
                   help="Which artifacts to produce.")
    p.add_argument("--save-bc", type=str, default="artifacts/bc_model.pth", help="Path to save BC weights.")
    p.add_argument("--save-sb3", type=str, default="artifacts/ppo_pretrained.zip", help="Path to save PPO agent.")
    p.add_argument("--ppo-kwargs", type=str, default=None,
                   help='JSON dict of kwargs to pass to PPO(...). Example: \'{"n_steps":2048,"batch_size":256}\'')

    return p.parse_args()


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    args = parse_args()
    set_global_seed(args.base_seed)

    device = torch.device(args.device)
    logger.info("Device: %s | CUDA available: %s", device, torch.cuda.is_available())

    # Build DataLoader with an ExpertDataset that yields a finite epoch
    loader = _build_dataloader(
        urdf_path=args.urdf_path,
        instruction=args.instruction,
        octo_model_name=args.octo_model_name,
        env_xml_path=args.env_xml_path,
        base_seed=args.base_seed,
        warmup=args.warmup,
        samples_per_epoch=args.samples_per_epoch,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        device_push_inside_ds=False,  # keep loader CPU, move in trainer for clarity
        device=device,
    )

    # Create model
    model = _instantiate_bcnet(device=device, model_ctor_overrides=args.model_ctor_overrides)

    # Sanity check on a real batch (detect action dims & basic forward compat)
    _sanity_check_batch(model, device, loader)

    # Train (prefer project BCTrainer; fallback if necessary)
    _train_with_preferred_or_fallback_trainer(
        model=model,
        device=device,
        loader=loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=float(args.grad_clip_norm) if args.grad_clip_norm is not None else None,
        amp=args.amp,
        log_interval=args.log_interval,
    )

    # Save BC weights (.pth) if requested
    if args.artifact in ("pth", "both"):
        _save_bc_weights(model, args.save_bc)

    # Export to SB3 .zip if requested
    if args.artifact in ("zip", "both"):
        # We transfer from the *state dict* to avoid device/dtype mismatches
        bc_sd = model.state_dict()
        _export_to_sb3_zip(
            bc_state_dict=bc_sd,
            save_path_zip=args.save_sb3,
            ppo_kwargs_json=args.ppo_kwargs,
        )

    # Cleanup
    del model, loader
    gc.collect()
    logger.info("Pretraining complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
    except Exception as e:
        logger.exception("Fatal error in pretraining: %s", e)
        sys.exit(1)
