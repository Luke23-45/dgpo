#!/usr/bin/env python3
# pretrain_policy.py
"""
Unified pretraining entry point (ExpertDataset + BCTrainer + optional SB3 export)

This script consolidates Phase-1 components into a single, robust CLI:

  1) Builds an ExpertDataset that generates OCTO/IK expert actions on-the-fly.
  2) Trains a BCNet policy via BCTrainer (preferred) or a safe built-in fallback loop.
  3) Saves a .pth checkpoint of BCNet (supervised BC weights).
  4) Optionally instantiates an SB3 PPO agent, transfers compatible weights, and saves .zip.

Hardened points:
 - Removes dataset args your ExpertDataset doesn’t support.
 - Float32 coercion for obs/actions to avoid uint8/float64 drift.
 - Infers action_dim directly from dataset batch before model instantiation.
 - Asserts SB3 env action_dim == BC action_dim for safe export.
"""

from __future__ import annotations

import os
import sys
import gc
import time
import json
import random
import logging
import argparse
from typing import Dict, Any, Optional, Tuple
from torch.utils.data._utils.collate import default_collate
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------
# Project imports (Phase-1)
# ---------------------------
from utils.expert_dataset import ExpertDataset
from models.bc_policy import BCNet
from models.custom_sb3_extractor import BCFeaturesExtractor
from utils.paths import resolve_path
# Prefer your BCTrainer, but keep a robust fallback if not present or incompatible
_BCTrainer = None
try:
    from bc_trainer import BCTrainer as _BCTrainer  # type: ignore
except Exception:
    _BCTrainer = None

# Optional imports only needed if exporting to SB3
def _maybe_import_sb3():
    try:
        from stable_baselines3 import PPO
        from envs.panda_env import PandaEnv
        from utils.transfer_bc_to_ppo import transfer_bc_weights
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
    """Deterministic-ish setup."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _to_float32_tree(x):
    """Recursively convert all torch.Tensors to float32, except for uint8 images."""
    if isinstance(x, torch.Tensor):
        # Keep uint8 images as-is, convert everything else
        return x if x.dtype == torch.uint8 else x.float()
    if isinstance(x, dict):
        return {k: _to_float32_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_float32_tree(v) for v in x)
    return x

def _float32_collate(batch):
    """
    A custom collate_fn that takes a batch of (obs, action) from the dataset,
    converts them to tensors, coerces dtypes to float32, and then batches them.
    """
    # The dataset can yield (numpy_obs_dict, numpy_action)
    obs_list, act_list = zip(*batch)

    # First, ensure everything is a torch.Tensor
    torch_obs_list = []
    for obs_np in obs_list:
        torch_obs = {}
        for key, val in obs_np.items():
            if isinstance(val, dict):
                torch_obs[key] = {k: torch.from_numpy(np.asarray(v)) for k, v in val.items()}
            else:
                torch_obs[key] = torch.from_numpy(np.asarray(val))
        torch_obs_list.append(torch_obs)
    torch_act_list = [torch.from_numpy(np.asarray(a)) for a in act_list]

    # Now, apply the float32 conversion and default batching
    obs_batch = default_collate([_to_float32_tree(o) for o in torch_obs_list])
    act_batch = default_collate([_to_float32_tree(a) for a in torch_act_list])

    return obs_batch, act_batch


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


# --- Replace the entire existing _FallbackBCTrainer class with this version ---

class _FallbackBCTrainer:
    """
    Minimal, safe BC trainer using modern torch.amp API and self-contained helpers.
    - Supervised MSE on actions.
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
        # CORRECT: Safely enable AMP only on CUDA devices
        self.amp = bool(amp and self.device.type == "cuda")
        # CORRECT: Use the modern torch.amp API
        self._scaler = torch.amp.GradScaler(device=self.device, enabled=self.amp)
        self._loss_fn = torch.nn.MSELoss()

    def fit(self, loader: DataLoader, epochs: int, log_interval: int = 50) -> None:
        self.model.train()
        for ep in range(1, epochs + 1):
            running_loss = 0.0
            num_batches = 0
            t0 = time.time()
            pbar = tqdm(loader, desc=f"Fallback Trainer Epoch {ep}/{epochs}", leave=False)
            for obs_dict, expert_action in pbar:
                # CORRECT: Helper function is self-contained inside the method
                def to_device_recursive(item, device):
                    if isinstance(item, torch.Tensor):
                        # This logic correctly preserves uint8 for images while moving to device
                        return item.to(device)
                    if isinstance(item, dict):
                        return {k: to_device_recursive(v, device) for k, v in item.items()}
                    if isinstance(item, (list, tuple)):
                        return type(item)(to_device_recursive(v, device) for v in item)
                    return item

                obs = to_device_recursive(obs_dict, self.device)
                target = expert_action.to(self.device).float()

                self.optim.zero_grad(set_to_none=True)
                # CORRECT: Use the modern torch.amp API
                with torch.amp.autocast(device_type=self.device.type, enabled=self.amp):
                    out = self.model(obs)
                    pred = _extract_pred_actions(out)
                    loss = self._loss_fn(pred, target)

                if not torch.isfinite(loss):
                    logger.warning("Non-finite loss detected, skipping batch.")
                    continue

                if self.amp:
                    self._scaler.scale(loss).backward()
                    if self.grad_clip_norm:
                        self._scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self._scaler.step(self.optim)
                    self._scaler.update()
                else:
                    loss.backward()
                    if self.grad_clip_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.optim.step()

                running_loss += loss.item()
                num_batches += 1
                if num_batches % log_interval == 0:
                    pbar.set_postfix(loss=f"{running_loss / num_batches:.6f}")

            avg_loss = running_loss / num_batches if num_batches > 0 else float("nan")
            logger.info(
                f"Epoch {ep} done in {time.time() - t0:.1f}s | avg loss {avg_loss:.6f}"
            )

def _to_device_tree(x, device):
    """
    Recursively move all torch.Tensors to the specified device,
    intelligently handling dtypes for the observation pipeline.
    """
    if isinstance(x, torch.Tensor):
        # Critical step: Keep image data as uint8 to ensure correct
        # normalization inside the model. Convert all other tensors to float32.
        if x.dtype == torch.uint8:
            return x.to(device)
        return x.to(device).float()
    
    if isinstance(x, dict):
        return {k: _to_device_tree(v, device) for k, v in x.items()}
    
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device_tree(v, device) for v in x)
    
    return x
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
) -> DataLoader:
    """
    Construct ExpertDataset + DataLoader with a finite epoch size via max_samples_per_epoch.
    NOTE: we intentionally DO NOT pass move_to_device/device/octo_pad_mask (not supported by your dataset).
    """
    ds = ExpertDataset(
        urdf_path=urdf_path,
        instruction=instruction,
        octo_model_name=octo_model_name,
        env_xml_path=env_xml_path,
        base_seed=base_seed,
        max_samples_per_epoch=int(samples_per_epoch),
        skip_on_error=True,
        warmup=warmup,
    )
    # Recommended: num_workers=0 for Mujoco/OCTO stability.
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,              # IterableDataset; shuffling happens at the source
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,             # stabilize batch shapes if samples_per_epoch not divisible
        collate_fn=_float32_collate,
    )
    return loader


def _peek_action_dim_and_small_batch(
    loader: DataLoader, device: torch.device
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, int]:
    """
    Pull one batch, coerce float32, and return (obs_small, target_small, action_dim).
    This lets us discover action_dim BEFORE instantiating BCNet.
    """
    it = iter(loader)
    obs_dict, expert_action = next(it)  # raises if loader misconfigured

    # Recursively process the observation dictionary to handle nested structures
    def process_and_slice_recursive(item):
        if isinstance(item, torch.Tensor):
            # Also handle uint8 images correctly, don't convert them to float
            if item.dtype == torch.uint8:
                return item[:2].to(device)
            return item[:2].to(device).float()
        if isinstance(item, dict):
            return {k: process_and_slice_recursive(v) for k, v in item.items()}
        # For any non-tensor, non-dict values, just return them
        return item
    
    obs_small = process_and_slice_recursive(obs_dict)
    target_small = expert_action[:2].to(device).float()

    if target_small.ndim != 2:
        raise RuntimeError(f"Expert action batch must be (B, A). Got shape: {tuple(target_small.shape)}")
    action_dim = int(target_small.shape[1])
    return obs_small, target_small, action_dim

def _instantiate_bcnet(
    *,
    device: torch.device,
    action_dim: int,
    model_ctor_overrides: Optional[str] = None,
) -> BCNet:
    """
    Instantiate BCNet with required action_dim and optional JSON overrides for constructor kwargs.
    This avoids hardcoding architecture details while remaining explicit.
    """
    ctor_kwargs: Dict[str, Any] = {}
    if model_ctor_overrides:
        try:
            ctor_kwargs = json.loads(model_ctor_overrides)
            assert isinstance(ctor_kwargs, dict)
        except Exception as e:
            raise ValueError(f"--model-ctor-overrides must be a JSON object; got: {model_ctor_overrides}") from e

    # If user also set action_dim in overrides, ensure consistency
    if "n_actions" in ctor_kwargs and int(ctor_kwargs["n_actions"]) != action_dim:
        raise ValueError(f"n_actions mismatch: overrides={ctor_kwargs['n_actions']} vs dataset={action_dim}")

    ctor_kwargs.setdefault("n_actions", action_dim)
    model = BCNet(**ctor_kwargs)  # your BCNet should handle sensible defaults for encoders
    model.to(device)
    logger.info("BCNet created with kwargs=%s | device=%s", ctor_kwargs, device)
    return model


def _sanity_forward_shape(
    model: BCNet,
    obs_small: Dict[str, torch.Tensor],
    action_dim: int,
) -> None:
    """
    Run a small forward and ensure BCNet produces a (B,A) tensor (or dict w/ such tensor).
    """
    model.eval()
    with torch.no_grad():
        out = model(obs_small)
    pred = _extract_pred_actions(out).float()
    if pred.ndim != 2 or int(pred.shape[1]) != action_dim:
        raise RuntimeError(
            f"BCNet forward produced shape {tuple(pred.shape)} but expected (B, {action_dim})."
        )
    logger.info("Sanity forward OK | batch=%d | action_dim=%d", int(pred.shape[0]), action_dim)


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
            try:
                trainer = _BCTrainer(model=model, device=device, lr=lr, weight_decay=weight_decay,
                                     grad_clip_norm=grad_clip_norm, amp=amp)
            except TypeError:
                # Simpler signature support
                trainer = _BCTrainer(model, device=device, lr=lr)
                if hasattr(trainer, "weight_decay"):
                    setattr(trainer, "weight_decay", weight_decay)
                if hasattr(trainer, "grad_clip_norm"):
                    setattr(trainer, "grad_clip_norm", grad_clip_norm)
                if hasattr(trainer, "amp"):
                    setattr(trainer, "amp", amp)

            if hasattr(trainer, "fit"):
                trainer.fit(loader=loader, epochs=epochs, log_interval=log_interval)
                return
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
    expected_action_dim: int,
    env_xml_path: Optional[str],
) -> None:
    """
    Create an SB3 PPO agent, transfer BC weights, and save a .zip archive.
    Asserts env action_dim == expected_action_dim before saving.
    """
    PPO, PandaEnv, transfer_bc_weights = _maybe_import_sb3()
    if PPO is None or PandaEnv is None or transfer_bc_weights is None:
        raise RuntimeError("SB3 export requested but stable_baselines3/PandaEnv/transfer function are unavailable.")

    # Instantiate env (try both signatures)
    try:
        env = PandaEnv(env_xml_path=env_xml_path, for_sb3=True)
    except TypeError:
        env = PandaEnv(for_sb3=True)

    # Guard: action dim must match
    env_act_dim = int(env.action_space.shape[0])
    if env_act_dim != expected_action_dim:
        raise RuntimeError(
            f"Action dim mismatch: env {env_act_dim} != BC {expected_action_dim}. "
            "Ensure dataset/BCNet/env use the same joint ordering and dimension."
        )

    policy_kwargs = {
        "features_extractor_class": BCFeaturesExtractor,
        "net_arch": {
            "pi": [512, 256],  # Policy network hidden layers
            "vf": [512, 256],  # Value network hidden layers
        }
    }
    logger.info(f"Forcing PPO policy_kwargs to match BCNet head: {policy_kwargs}")

    # --- 3. Parse OPTIONAL User-Provided PPO Arguments from command line ---
    user_ppo_kwargs = {}
    if ppo_kwargs_json:
        try:
            user_ppo_kwargs = json.loads(ppo_kwargs_json)
            assert isinstance(user_ppo_kwargs, dict)
        except Exception as e:
            raise ValueError(f"--ppo-kwargs must be a valid JSON object string; got: {ppo_kwargs_json}") from e

    # --- 4. Create the PPO Agent with the Correct, Merged Configuration ---
    agent = PPO(
        "MultiInputPolicy",
        env,
        verbose=0,
        policy_kwargs=policy_kwargs,  # Pass the required architecture here
        **user_ppo_kwargs              # Pass any other user args (like n_steps) here
    )



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
                   help='JSON dict with kwargs for BCNet constructor. Example: \'{"image_channels":3}\'')
    p.add_argument("--expect_action_dim", type=int, default=None,
                   help="Optional guard: assert dataset action_dim equals this value.")

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
    args.urdf_path = resolve_path(args.urdf_path)
    device = torch.device(args.device)
    logger.info("Device: %s | CUDA available: %s", device, torch.cuda.is_available())
    print("\n\n >>>>>>>>>> RUNNING THE NEW, CORRECTED MAIN FUNCTION <<<<<<<<<< \n\n")

    # --- FIX: Create a temporary loader for shape inference to avoid resource conflicts ---
    logger.info("Creating a temporary dataloader to infer action dimension...")
    peek_loader = _build_dataloader(
        urdf_path=args.urdf_path,
        instruction=args.instruction,
        octo_model_name=args.octo_model_name,
        env_xml_path=args.env_xml_path,
        base_seed=args.base_seed,
        warmup=False,  # No need to warmup for just one batch
        samples_per_epoch=args.batch_size,
        batch_size=args.batch_size,
        num_workers=0, # Must use 0 workers for clean resource management
        pin_memory=False,
    )
    obs_small, target_small, action_dim = _peek_action_dim_and_small_batch(peek_loader, device)
    del peek_loader
    gc.collect()
    logger.info("Action dimension inferred successfully. Proceeding with main dataloader.")

    if args.expect_action_dim is not None and int(args.expect_action_dim) != action_dim:
        raise RuntimeError(f"--expect_action_dim={args.expect_action_dim} but dataset has {action_dim}")

    # --- Now create the main loader for training ---
    logger.info("Creating main dataloader for training...")
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
    )

    # Create model with correct action_dim
    model = _instantiate_bcnet(device=device, action_dim=action_dim, model_ctor_overrides=args.model_ctor_overrides)

    # Sanity check forward shape on tiny batch
    _sanity_forward_shape(model, obs_small, action_dim)

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
    # Export to SB3 .zip if requested
    if args.artifact in ("zip", "both"):
        logger.info("Attempting to export model to Stable-Baselines3 format...")
        try:
            bc_sd = model.state_dict()
            _export_to_sb3_zip(
                bc_state_dict=bc_sd,
                save_path_zip=args.save_sb3,
                ppo_kwargs_json=args.ppo_kwargs,
                expected_action_dim=action_dim,
                env_xml_path=args.env_xml_path,
            )
        except Exception as e:
            logger.warning(f"Could not export to SB3 format. This is non-fatal.")
            logger.warning(f"Reason: {e}")
            logger.warning("Please ensure 'stable-baselines3' is installed (`pip install stable-baselines3[extra]`) and all dependencies are met.")

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
