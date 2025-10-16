"""
pretrain_diffusion.py

Configuration-driven pretraining script for DiffusionPolicy denoiser.

Features:
- YAML configuration (paths, hyperparams, scheduler selection, EMA, AMP)
- Reproducible seeding (numpy, random, torch)
- TensorBoard logging and optional Weights & Biases (if installed)
- Checkpointing (model + optimizer + scheduler) and resume capability
- Validation loop and diagnostics: numeric logs + a small "denoising trajectory" dump for quick visual checks
- Designed to accept a PyTorch Dataset that yields (obs_tensor, action_tensor)
  where `action_tensor` is normalized to the policy action range (e.g., [-1,1])

Usage:
    python pretrain_diffusion.py --config configs/pretrain.yaml
    python -m scripts.pretrain_diffusion --config configs/pretrain_config.yaml


YAML config example (minimal):
    seed: 1234
    device: cuda
    output_dir: runs/diffusion_pretrain
    dataset:
      path: data/demos/expert_2025_...
      batch_size: 64
      num_workers: 4
    model:
      action_dim: 8
      cond_dim: 128
      hidden: 512
      denoiser_layers: 4
    scheduler:
      type: linear
      T: 1000
      beta_start: 1e-4
      beta_end: 0.02
    training:
      epochs: 50
      lr: 1e-4
      weight_decay: 0.0
      ema_decay: 0.9999
      amp: true
      log_interval_steps: 50
      save_interval_epochs: 5
"""

from __future__ import annotations
from utils.expert_dataset import collate_fn
import argparse
import os
import random
import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

# Import the diffusion module components
from models.diffusion_policy import (
    SimpleMLPDenoiser,
    NoiseScheduler,
    DiffusionPolicy,
    EMA,
)

# ----------------------------
# Util functions
# ----------------------------
def set_seed(seed: int):
    import random as py_random
    py_random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_cond_embed_fn(cond_dim: int):
    """
    Returns a cond_embed_fn that maps a conditioning dict or tensor to a tensor of shape (B, cond_dim).
    This example assumes cond is a dict with keys 'proprio' (B, P) and optionally 'image_feat' (B, F).
    You should adapt to your data format.
    """
    def embed_fn(cond):
        # strip to tensor(s)
        if isinstance(cond, dict):
            if "proprio" in cond:
                p = cond["proprio"].float()
            else:
                raise ValueError("cond dict must contain 'proprio' key for this embed_fn")
            # flatten if needed
            if p.ndim > 2:
                p = p.view(p.shape[0], -1)
            # simple linear projection to cond_dim
            if p.shape[-1] != cond_dim:
                # simple zero-pad or linear layer could be used, but keep this simple:
                pad = cond_dim - p.shape[-1]
                if pad > 0:
                    p = torch.cat([p, torch.zeros(p.shape[0], pad, device=p.device)], dim=-1)
                else:
                    p = p[:, :cond_dim]
            return p
        elif torch.is_tensor(cond):
            x = cond.float()
            if x.ndim > 2:
                x = x.view(x.shape[0], -1)
            if x.shape[-1] != cond_dim:
                pad = cond_dim - x.shape[-1]
                if pad > 0:
                    x = torch.cat([x, torch.zeros(x.shape[0], pad, device=x.device)], dim=-1)
                else:
                    x = x[:, :cond_dim]
            return x
        else:
            raise ValueError("cond format not supported by embed_fn")
    return embed_fn


# ----------------------------
# Training loop
# ----------------------------
def train(
    cfg: Dict[str, Any],
    train_dataset,
    val_dataset = None,
):
    # config unpacking
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(int(cfg.get("seed", 1234)))

    # writer
    writer = SummaryWriter(log_dir=str(out_dir/"tensorboard"))
    try:
        import wandb
        use_wandb = cfg.get("use_wandb", False)
        if use_wandb:
            wandb.init(project=cfg.get("wandb_project", "diffusion_pretrain"), config=cfg)
    except Exception:
        use_wandb = False

    # model scaffolding
    action_dim = int(cfg["model"]["action_dim"])
    cond_dim = int(cfg["model"].get("cond_dim", 128))
    denoiser = SimpleMLPDenoiser(action_dim=action_dim, cond_dim=cond_dim, hidden=int(cfg["model"].get("hidden", 512)), n_layers=int(cfg["model"].get("denoiser_layers", 3)))
    # scheduler
    T = int(cfg["scheduler"]["T"])
    sched_type = cfg["scheduler"].get("type", "linear")
    if sched_type == "linear":
        scheduler = NoiseScheduler.linear(float(cfg["scheduler"].get("beta_start", 1e-4)), float(cfg["scheduler"].get("beta_end", 0.02)), T, device=device)
    elif sched_type == "cosine":
        scheduler = NoiseScheduler.cosine(T=T, device=device)
    else:
        raise ValueError("Unknown scheduler type")

    cond_embed_fn = build_cond_embed_fn(cond_dim)
    policy = DiffusionPolicy(denoiser=denoiser, action_dim=action_dim, cond_embed_fn=cond_embed_fn, scheduler=scheduler, device=device, ema_decay=float(cfg["training"].get("ema_decay", 0.0)))
    policy.to(device)

    # optimizer
    optimizer = optim.AdamW(policy.parameters(), lr=float(cfg["training"].get("lr", 1e-4)), weight_decay=float(cfg["training"].get("weight_decay", 0.0)))

    # datasets & loaders
    train_loader = DataLoader(train_dataset, batch_size=int(cfg["dataset"]["batch_size"]), shuffle=True, num_workers=int(cfg["dataset"].get("num_workers", 4)), pin_memory=True, drop_last=True, collate_fn=collate_fn,)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=int(cfg["dataset"]["batch_size"]), shuffle=False, num_workers=2, pin_memory=True)

    # AMP
    use_amp = bool(cfg["training"].get("amp", False))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    epochs = int(cfg["training"].get("epochs", 50))
    log_interval = int(cfg["training"].get("log_interval_steps", 50))
    save_interval = int(cfg["training"].get("save_interval_epochs", 5))
    best_val = float("inf")
    global_step = 0

    # checkpoint resume
    ckpt_path = cfg.get("resume_checkpoint", None)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        policy.load_state_dict(ckpt["policy"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = ckpt.get("global_step", 0)
        best_val = ckpt.get("best_val", best_val)
        print(f"Resumed from {ckpt_path} at step {global_step}")

    # training loop
    for epoch in range(1, epochs + 1):
        policy.train()
        epoch_loss = 0.0
        epoch_count = 0
        t0 = time.time()
        for i, batch in enumerate(train_loader):
            # batch expected as (obs, action) or (dict_obs, action_tensor)
            obs_batch, actions = batch
            # normalize or cast actions to float32 in [-1,1] domain: assume dataset already normalized
            actions = actions.float().to(device)
            cond = obs_batch  # cond_embed_fn expects dict or tensor depending on implementation
            B = actions.shape[0]

            # Flatten the action horizon
            # Shape changes from (B, H_action, action_dim) -> (B, H_action * action_dim)
            actions = actions.reshape(B, -1)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, per_sample = policy(actions, cond)
            scaler.scale(loss).backward()
            # gradient clip if configured
            if "grad_clip" in cfg["training"]:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), float(cfg["training"]["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()

            # EMA
            policy.maybe_update_ema()

            epoch_loss += float(loss.item())
            epoch_count += 1
            global_step += 1

            if global_step % log_interval == 0:
                avg_loss = epoch_loss / max(1, epoch_count)
                writer.add_scalar("train/loss", avg_loss, global_step)
                if use_wandb:
                    wandb.log({"train/loss": avg_loss, "global_step": global_step})
                print(f"[Epoch {epoch}] step {global_step} train loss {avg_loss:.6f}")

        # end epoch
        dt = time.time() - t0
        avg_epoch_loss = epoch_loss / max(1, epoch_count)
        print(f"Epoch {epoch} finished. avg_loss={avg_epoch_loss:.6f}, time={dt:.1f}s")
        writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch)

        # validation
        if val_loader is not None:
            policy.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for j, vbatch in enumerate(val_loader):
                    vobs, vactions = vbatch
                    vactions = vactions.float().to(device)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        loss_v, _ = policy(vactions, vobs)
                    val_loss += float(loss_v.item())
                    val_count += 1
            val_loss /= max(1, val_count)
            writer.add_scalar("val/loss", val_loss, epoch)
            print(f"Validation loss: {val_loss:.6f}")
            if use_wandb:
                wandb.log({"val/loss": val_loss, "epoch": epoch})

            # checkpoint best
            if val_loss < best_val:
                best_val = val_loss
                save_path = out_dir / f"best_ckpt_epoch{epoch}.pt"
                torch.save({
                    "policy": policy.state_dict_for_save(),
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step,
                    "best_val": best_val
                }, save_path)
                print(f"Saved best checkpoint to {save_path}")

        # periodic save
        if epoch % save_interval == 0:
            save_path = out_dir / f"ckpt_epoch{epoch}.pt"
            torch.save({
                "policy": policy.state_dict_for_save(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "best_val": best_val
            }, save_path)
            print(f"Saved checkpoint to {save_path}")

        # diagnostics: dump a few denoising trajectories
        if epoch % max(1, save_interval // 2) == 0:
            policy.eval()
            with torch.no_grad():
                # pick a small batch from train_loader
                try:
                    sample_batch = next(iter(train_loader))
                except StopIteration:
                    sample_batch = None
                if sample_batch is not None:
                    obs_s, acts_s = sample_batch
                    acts_s = acts_s.float().to(device)[:8]
                    # get noisy versions for a handful of samples and run denoising steps
                    for idx in range(min(8, acts_s.shape[0])):
                        a0 = acts_s[idx:idx+1]
                        # produce noisy trajectory at some timesteps
                        noise = torch.randn_like(a0).to(device)
                        tlist = [policy.T - 1, max(0, policy.T // 2), 0]
                        denoised = {}
                        for t in tlist:
                            alpha_bar = policy._alpha_bars[t].to(device)
                            x_t = torch.sqrt(alpha_bar).unsqueeze(-1) * a0 + torch.sqrt(1.0 - alpha_bar).unsqueeze(-1) * noise
                            mu = policy.sample_deterministic({"proprio": obs_s["proprio"][:1]}, steps=1)
                            denoised[int(t)] = {
                                "noisy": x_t.cpu().numpy().tolist(),
                                "denoised": mu.cpu().numpy().tolist(),
                                "clean": a0.cpu().numpy().tolist()
                            }
                        diag_file = out_dir / f"diag_epoch{epoch}_sample{idx}.json"
                        with open(diag_file, "w") as f:
                            json.dump(denoised, f)
    # end training loop
    writer.close()
    if use_wandb:
        wandb.finish()


# ----------------------------
# CLI / config
# ----------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint to resume from (optional)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.resume:
        cfg["resume_checkpoint"] = args.resume

    # Create datasets from path or factory -- user must adapt this block to their dataset class
    # Expect: train_dataset yields (obs_dict, action_tensor)
    import importlib
    # Example: dataset module should expose `build_datasets(cfg)` that returns (train_dataset, val_dataset)
    if "dataset_factory" in cfg:
        modname = cfg["dataset_factory"].split(":")[0]
        fnname = cfg["dataset_factory"].split(":")[1]
        module = importlib.import_module(modname)
        train_dataset, val_dataset = getattr(module, fnname)(cfg)
    else:
        # Default: user-supplied ExpertTrajectoryDataset usage
        from utils.expert_dataset import ExpertTrajectoryDataset,collate_fn 

        demo_path = cfg["dataset"]["path"]
        print(f"demo path diffusion - {demo_path}")
        train_dataset = ExpertTrajectoryDataset(
        demo_path,
        observation_horizon=cfg["model"]["observation_horizon"],
        action_horizon=cfg["model"]["action_horizon"]
        )
        val_dataset = None

    train(cfg, train_dataset, val_dataset)


if __name__ == "__main__":
    main()
