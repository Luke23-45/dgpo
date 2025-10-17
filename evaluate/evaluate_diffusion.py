# FILE: scripts/evaluate_policy.py
# ==============================================================
# State-of-the-Art Evaluation Script for Diffusion Policy
# ==============================================================
"""
Performs rigorous offline (dataset) and online (environment) evaluation
for a trained Diffusion Policy checkpoint.

Features:
- Hydra-based configuration (consistent with pretraining).
- Canonical checkpoint loading (with architecture reconstruction).
- Offline metrics (MSE, MAE) against expert actions.
- Online rollouts with environment wrappers and video saving.
- EMA weight handling, reproducible seeding, and structured logging.
"""

from __future__ import annotations
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import imageio
import hydra
from omegaconf import DictConfig, OmegaConf

# Project Imports
from scripts.pretrain_diffusion import set_seed
from utils.expert_dataset import ExpertTrajectoryDataset, collate_fn
from models.diffusion_policy import DiffusionPolicy
from models.diffusion_policy import NoiseSchedulerConfig

# Optional: environment modules for online rollout evaluation
try:
    from envs.panda_env import PandaEnv
    from utils.rl_reward_wrapper import AdvancedRewardWrapper
    from envs.panda_env_wrapper import HERGoalEnvWrapper
except ImportError:
    PandaEnv = None
    AdvancedRewardWrapper = None
    HERGoalEnvWrapper = None

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# ==============================================================
# 1. Policy Loading Utilities
# ==============================================================

def build_policy_from_cfg(cfg: DictConfig, device: torch.device) -> DiffusionPolicy:
    """
    Builds a DiffusionPolicy exactly as done in pretrain_diffusion.py.
    Ensures consistency between training and evaluation.
    """
    from scripts.pretrain_diffusion import DiffusionPretrainer
    # Use same builder used during training to avoid argument mismatch
    trainer = DiffusionPretrainer(cfg)
    policy = trainer._build_policy()
    policy.to(device)
    return policy


def load_policy_from_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[DiffusionPolicy, DictConfig]:
    """
    Loads a DiffusionPolicy model from a training checkpoint.
    Returns (policy, config)
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    log.info(f"Loading checkpoint from: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=device)

    # Retrieve saved Hydra config if available
    if "config" in payload:
        cfg = DictConfig(payload["config"])
    else:
        raise RuntimeError("Checkpoint missing 'config' key — cannot rebuild model.")

    # Build model from saved config
    policy = build_policy_from_cfg(cfg, device)
    policy.load_state_dict(payload["policy_state_dict"])

    # Load EMA if available
    if hasattr(policy, "ema") and "ema_state_dict" in payload:
        try:
            policy.ema.load_state_dict(payload["ema_state_dict"])
            log.info("EMA weights loaded successfully.")
        except Exception as e:
            log.warning(f"Failed to load EMA weights: {e}")

    # Attach meta info if available
    meta = payload.get("meta", {})
    if meta:
        log.info(f"Checkpoint meta info: {json.dumps(meta, indent=2)}")

    log.info("Policy successfully reconstructed and loaded.")
    return policy, cfg


# ==============================================================
# 2. Offline Evaluation
# ==============================================================

@torch.no_grad()
def evaluate_offline(policy: DiffusionPolicy, dataloader: DataLoader, cfg: DictConfig) -> Dict[str, float]:
    """
    Computes offline quantitative metrics:
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)
    """
    policy.eval()

    if hasattr(policy, "ema") and cfg.training.get("ema_decay", 0.0) > 0:
        log.info("Using EMA weights for evaluation.")
        policy.ema.store(policy.parameters())
        policy.ema.copy_to(policy.parameters())

    mses, maes = [], []
    pbar = tqdm(dataloader, desc="[Offline Eval]")

    for obs_chunk, gt_action_chunk in pbar:
        obs_chunk = {k: v.to(policy.device).float() for k, v in obs_chunk.items()}
        gt_action_chunk = gt_action_chunk.to(policy.device).float()

        # Sample predicted action sequence
        try:
            pred_actions = policy.sample(
                obs_chunk,
                steps=cfg.validation.sampling_steps,
                guidance_scale=cfg.validation.guidance_scale,
                use_ema=True
            )
        except TypeError:
            pred_actions = policy.sample(obs_chunk)

        mse = torch.nn.functional.mse_loss(pred_actions, gt_action_chunk)
        mae = torch.nn.functional.l1_loss(pred_actions, gt_action_chunk)

        mses.append(mse.item())
        maes.append(mae.item())
        pbar.set_postfix({"MSE": f"{mse.item():.4f}", "MAE": f"{mae.item():.4f}"})

    if hasattr(policy, "ema") and cfg.training.get("ema_decay", 0.0) > 0:
        policy.ema.restore(policy.parameters())

    return {
        "mean_mse": float(np.mean(mses)),
        "mean_mae": float(np.mean(maes)),
        "std_mse": float(np.std(mses)),
    }


# ==============================================================
# 3. Online Rollout Evaluation
# ==============================================================

@torch.no_grad()
def evaluate_online_rollouts(policy: DiffusionPolicy, cfg: DictConfig, output_dir: Path) -> Dict[str, float]:
    """
    Runs policy rollouts in a simulated environment and computes success rate.
    Saves rollout videos for qualitative inspection.
    """
    if PandaEnv is None:
        log.warning("Environment modules not found; skipping online evaluation.")
        return {}

    log.info("Initializing evaluation environment...")
    env_cfg = cfg.get("environment", {})
    base_env = PandaEnv(xml_path=env_cfg.get("xml_path", None), control_mode="delta")
    reward_env = AdvancedRewardWrapper(base_env)
    env = HERGoalEnvWrapper(reward_env)

    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    success_flags = []

    for rollout_idx in tqdm(range(cfg.eval.n_rollouts), desc="[Online Eval]"):
        obs, _ = env.reset()
        frames = [env.render()]
        obs_queue = []

        done = False
        step = 0
        while not done and step < cfg.eval.max_rollout_steps:
            # Maintain observation queue
            obs_queue.append(obs)
            if len(obs_queue) > cfg.model.observation_horizon:
                obs_queue.pop(0)

            # Convert to policy input format
            obs_dict = {k: torch.from_numpy(v).unsqueeze(0).to(policy.device) for k, v in obs["observation"].items()}
            obs_for_policy = {k: v.repeat(1, cfg.model.observation_horizon, 1) for k, v in obs_dict.items()}

            # Predict action sequence
            try:
                actions_pred = policy.sample(
                    obs_for_policy,
                    steps=cfg.validation.sampling_steps,
                    guidance_scale=cfg.validation.guidance_scale,
                    use_ema=True
                )
            except TypeError:
                actions_pred = policy.sample(obs_for_policy)

            # Execute first action
            action = actions_pred[0, 0, :].cpu().numpy()
            obs, _, _, _, info = env.step(action)
            frames.append(env.render())

            done = bool(info.get("is_success", False))
            step += 1

        success_flags.append(float(done))

        # Save video
        video_path = video_dir / f"rollout_{rollout_idx:03d}_success_{int(done)}.mp4"
        imageio.mimsave(video_path, frames, fps=20)
        log.info(f"Saved rollout video: {video_path}")

    return {
        "mean_success_rate": float(np.mean(success_flags)),
        "std_success_rate": float(np.std(success_flags)),
    }


# ==============================================================
# 4. Hydra Main Entrypoint
# ==============================================================

@hydra.main(version_base=None, config_path="../configs", config_name="evaluate_policy_config")
def main(cfg: DictConfig):
    # Hydra automatically sets the working directory to the output dir
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    fh = logging.FileHandler(output_dir / "eval.log", mode="w")
    fh.setLevel(logging.INFO)
    log.addHandler(fh)
    log.info("===================================================")
    log.info("           Diffusion Policy Evaluation             ")
    log.info("===================================================")
    log.info(OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # --- Load trained policy ---
    ckpt_path = Path(cfg.ckpt_path)
    policy, train_cfg = load_policy_from_checkpoint(ckpt_path, device)
    policy.to(device)

    # --- Offline evaluation ---
    if cfg.get("do_offline_eval", True):
        log.info("Starting offline evaluation...")
        val_path = cfg.dataset.get("val_path", None) or cfg.dataset["path"]
        val_dataset = ExpertTrajectoryDataset(
            demo_path=val_path,
            observation_horizon=policy.H_o,
            action_horizon=policy.H_a,
        )
        val_loader = DataLoader(val_dataset, batch_size=cfg.validation.get("batch_size", 16),
                                shuffle=False, collate_fn=collate_fn, num_workers=0)
        offline_metrics = evaluate_offline(policy, val_loader, cfg)
        log.info(f"Offline Metrics:\n{json.dumps(offline_metrics, indent=2)}")
        with open(output_dir / "offline_metrics.json", "w") as f:
            json.dump(offline_metrics, f, indent=2)

    # --- Online rollouts ---
    if cfg.get("do_online_eval", False):
        log.info("Starting online evaluation...")
        online_metrics = evaluate_online_rollouts(policy, cfg, output_dir)
        log.info(f"Online Metrics:\n{json.dumps(online_metrics, indent=2)}")
        with open(output_dir / "online_metrics.json", "w") as f:
            json.dump(online_metrics, f, indent=2)

    log.info("Evaluation complete.")
    log.info(f"Results saved in: {output_dir}")


# ==============================================================
# 5. Standard Entrypoint
# ==============================================================

if __name__ == "__main__":
    main()
