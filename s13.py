# FILE: scripts/pretrain_diffusion.py
"""
State-of-the-art pretraining script for the DiffusionPolicy model.

This script is the definitive, production-ready implementation for training
the advanced, vision-aware Transformer-based diffusion policy. It correctly
instantiates the self-contained DiffusionPolicy class from `models/diffusion_policy.py`
and trains it on the multi-modal, temporally-chunked data provided by the
`ExpertTrajectoryDataset`.

Highlights:
 - YAML-driven configuration for complete and reproducible experiments.
 - Full support for multi-view, multi-horizon datasets.
 - Robust training loop with gradient clipping, mixed-precision (AMP), and EMA updates.
 - Comprehensive checkpointing with support for resuming training.
 - Validation loop for identifying and saving the best-performing model.
 - Qualitative diagnostics by periodically sampling and saving action trajectories.
 - Integrated logging with TensorBoard and optional Weights & Biases support.
"""

from __future__ import annotations
import argparse
import os
import random
import time
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

# Import the definitive, state-of-the-art model and dataset components
from models.diffusion_policy import DiffusionPolicy, NoiseSchedulerConfig
from utils.expert_dataset import ExpertTrajectoryDataset, collate_fn


# =========================================================
# Utility Functions
# =========================================================

def set_seed(seed: int):
    """Ensures full reproducibility by seeding all relevant libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # The following two lines are essential for fully reproducible GPU training
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def print_stage(title: str):
    """Prints a formatted stage header for clear console output."""
    print("\n" + "=" * 80)
    print(f"✅ {title}")
    print("=" * 80 + "\n")


# =========================================================
# Core Training and Validation Logic
# =========================================================

def train_epoch(policy: DiffusionPolicy, loader: DataLoader, optimizer, scaler, device, use_amp: bool, grad_clip: float, log_interval: int, epoch: int, writer, global_step: int) -> tuple[float, int]:
    """Runs a single training epoch."""
    policy.train()
    running_loss = 0.0
    start_time = time.time()

    for i, (obs_batch, actions) in enumerate(loader):
        # Data is already on device from collate_fn if using GPU workers, but good practice to ensure
        actions = actions.to(device, non_blocking=True).float()
        
        optimizer.zero_grad(set_to_none=True)

        # The policy's internal logic handles the observation dictionary
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss, diagnostics = policy.compute_loss(actions, obs_batch)

        scaler.scale(loss).backward()
        
        # Unscale gradients before clipping to avoid scaling the clip value
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()

        # NOTE: EMA update is handled inside policy.compute_loss() for encapsulation
        
        running_loss += float(loss.item())
        global_step += 1

        if i > 0 and i % log_interval == 0:
            avg_loss = running_loss / (i + 1)
            print(f"[Epoch {epoch:03d} | Step {i:05d}]  Train Loss = {avg_loss:.6f}")
            if writer:
                writer.add_scalar("train/loss_step", avg_loss, global_step)

    avg_loss = running_loss / max(1, len(loader))
    duration = time.time() - start_time
    print(f"Epoch {epoch:03d} completed in {duration:.1f}s — Average Train Loss = {avg_loss:.6f}")
    return avg_loss, global_step

@torch.no_grad()
def validate_epoch(policy: DiffusionPolicy, loader: DataLoader, device, use_amp: bool, writer, epoch: int) -> float:
    """Runs a single validation epoch."""
    policy.eval()
    total_loss = 0.0
    count = 0

    for obs_batch, actions in loader:
        actions = actions.to(device, non_blocking=True).float()
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss, _ = policy.compute_loss(actions, obs_batch)
        total_loss += float(loss.item())
        count += 1

    avg_val_loss = total_loss / max(1, count)
    print(f"Validation — Average Loss = {avg_val_loss:.6f}")
    if writer:
        writer.add_scalar("val/loss", avg_val_loss, epoch)
    return avg_val_loss

@torch.no_grad()
def generate_diagnostics(policy: DiffusionPolicy, loader: DataLoader, out_dir: Path, epoch: int):
    """Saves a few sampled action trajectories for qualitative sanity checks."""
    policy.eval()
    try:
        obs_batch, _ = next(iter(loader))
    except StopIteration:
        print("Diagnostics skipped: DataLoader is empty.")
        return

    # Use a small, consistent subset of the first batch for diagnostics
    obs_batch_subset = {k: v[:4] for k, v in obs_batch.items()}
    
    # Sample using the stable EMA model
    sampled_actions = policy.sample(obs_batch_subset, steps=50, eta=0.0, use_ema=True)
    
    file_path = out_dir / f"diagnostics_epoch{epoch:03d}.json"
    data = {"sampled_actions": sampled_actions.cpu().numpy().tolist()}
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved diagnostic samples to {file_path}")


# =========================================================
# Main Training Orchestrator
# =========================================================

def train(cfg: Dict[str, Any], train_dataset, val_dataset=None):
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(int(cfg.get("seed", 42)))

    print_stage("Initializing Training")
    print(f"Using device: {device}")
    print(f"Output directory: {out_dir}")

    # --- Logging ---
    writer = SummaryWriter(log_dir=str(out_dir / "tensorboard"))
    try:
        import wandb
        if cfg.get("use_wandb", False):
            wandb.init(project=cfg.get("wandb_project", "diffusion_pretrain"), config=cfg, dir=str(out_dir))
    except ImportError:
        cfg["use_wandb"] = False
        print("W&B not found, disabling.")

    # --- Model Instantiation ---
    mcfg = cfg["model"]
    scfg = cfg["scheduler"]
    tcfg = cfg["training"]

    scheduler_cfg_obj = NoiseSchedulerConfig(
        beta_start=float(scfg["beta_start"]),
        beta_end=float(scfg["beta_end"]),
        schedule=str(scfg["type"]),
        timesteps=int(scfg["T"])
    )

    policy = DiffusionPolicy(
        image_channels=3,
        image_feat_dim=int(mcfg["vision_features"]),
        proprio_dim=int(mcfg["proprio_dim"]),
        H_o=int(mcfg["observation_horizon"]),
        H_a=int(mcfg["action_horizon"]),
        action_dim=int(mcfg["action_dim"]),
        scheduler_cfg=scheduler_cfg_obj,
        d_model=int(mcfg["d_model"]),
        denoiser_layers=int(mcfg["denoiser_layers"]),
        denoiser_heads=int(mcfg["denoiser_heads"]),
        denoiser_dropout=float(mcfg["denoiser_dropout"]),
        ema_decay=float(tcfg["ema_decay"]),
        device=device
    )

    optimizer = optim.AdamW(
        policy.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"])
    )

    use_amp = bool(tcfg.get("amp", False)) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # --- Data Loaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["dataset"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["dataset"]["num_workers"]),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=int(cfg["dataset"]["batch_size"]), shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn)

    # --- Checkpoint Resumption ---
    start_epoch, global_step, best_val_loss = 1, 0, float("inf")
    if cfg.get("resume_checkpoint"):
        print_stage(f"Resuming from Checkpoint: {cfg['resume_checkpoint']}")
        ckpt = torch.load(cfg["resume_checkpoint"], map_location=device)
        policy.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 1) + 1
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resuming training from Epoch {start_epoch}")

    # --- Main Training Loop ---
    print_stage("Starting Training")
    for epoch in range(start_epoch, int(tcfg["epochs"]) + 1):
        avg_train_loss, global_step = train_epoch(
            policy, train_loader, optimizer, scaler, device,
            use_amp, tcfg.get("grad_clip", 1.0),
            tcfg["log_interval_steps"], epoch, writer, global_step
        )
        if cfg.get("use_wandb", False):
            wandb.log({"train/epoch_loss": avg_train_loss, "epoch": epoch})

        # --- Validation and Checkpointing ---
        if val_loader:
            avg_val_loss = validate_epoch(policy, val_loader, device, use_amp, writer, epoch)
            if cfg.get("use_wandb", False):
                wandb.log({"val/loss": avg_val_loss, "epoch": epoch})
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                policy.save(str(out_dir / "best_model.pt"))
                print(f"✅ New best checkpoint saved (val_loss={best_val_loss:.6f})")

        if epoch % int(tcfg["save_interval_epochs"]) == 0:
            policy.save(str(out_dir / f"ckpt_epoch_{epoch:03d}.pt"))

        if epoch % max(1, int(tcfg["save_interval_epochs"]) // 2) == 0:
            generate_diagnostics(policy, train_loader, out_dir / "diagnostics", epoch)

    # --- Finalization ---
    writer.close()
    if cfg.get("use_wandb", False):
        wandb.finish()
    print_stage("Training Complete")


# =========================================================
# CLI Entrypoint
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Pretrain a State-of-the-Art Diffusion Policy")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--resume", type=str, default=None, help="Optional path to a checkpoint to resume training from.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.resume:
        cfg["resume_checkpoint"] = args.resume

    # --- Dataset Instantiation ---
    print_stage("Loading Dataset")
    train_dataset = ExpertTrajectoryDataset(
        cfg["dataset"]["path"],
        observation_horizon=int(cfg["model"]["observation_horizon"]),
        action_horizon=int(cfg["model"]["action_horizon"])
    )
    val_dataset = None
    if cfg["dataset"].get("val_path"):
        val_dataset = ExpertTrajectoryDataset(cfg["dataset"]["val_path"], observation_horizon=int(cfg["model"]["observation_horizon"]), action_horizon=int(cfg["model"]["action_horizon"]))

    print(f"Training dataset loaded with {len(train_dataset)} samples.")
    
    train(cfg, train_dataset, val_dataset)


if __name__ == "__main__":
    main()