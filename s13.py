#!/usr/bin/env python3
"""
scripts/evaluate_ego_planner.py

State-of-the-art evaluation and visualization script for Ego-Planner / Diffusion policy.

Features:
- Load a checkpoint safely (model state + optional EMA state).
- Offline evaluation (MSE, MAE) on an LMDB validation dataset (uses index json).
- Online closed-loop rollouts with a user-supplied environment factory; renders and saves videos.
- Utility to merge two LMDB datasets (rewrites index and rekeys episodes).
- Outputs metrics JSON, example samples, and videos.
- Robust checks & verbose logging.

Usage examples:
  # Offline eval only:
  python scripts/evaluate_ego_planner.py --ckpt path/to/best.ckpt --val-lmdb path/to/val.lmdb --val-index path/to/val_index.json --offline

  # Online rollouts (requires env factory callable string like "envs.panda_env:make_env"):
  python scripts/evaluate_ego_planner.py --ckpt best.ckpt --env-factory "envs.panda_env:make_env" --n-rollouts 10 --outdir runs/eval_run

  # Merge two LMDBs:
  python scripts/evaluate_ego_planner.py --merge src1.lmdb src2.lmdb --merge-index src1_index.json src2_index.json --out-merged merged.lmdb --out-index merged_index.json
"""

from __future__ import annotations
import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Optional, Tuple, Dict, List, Callable

import lmdb
import numpy as np
import torch
from tqdm import tqdm
import imageio.v2 as imageio  # imageio for writing mp4
from torch.utils.data import DataLoader

# Try to import project modules defensively
try:
    from models.ego_planner import EgoPlanner  # or your DiffusionPolicy
except Exception:
    EgoPlanner = None

try:
    # Expect the project's ExpertTrajectoryDataset that returns (obs_dict, action_tensor)
    from utils.expert_dataset import ExpertTrajectoryDataset, collate_fn
except Exception:
    ExpertTrajectoryDataset = None
    collate_fn = None

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)7s | %(name)s | %(message)s"
)
log = logging.getLogger("evaluate_ego_planner")


# -------------------------
# Helpers
# -------------------------
def safe_load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def open_lmdb_read(path: Path) -> lmdb.Environment:
    if not path.exists():
        raise FileNotFoundError(f"LMDB path not found: {path}")
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=True, max_readers=126)
    log.info(f"Opened LMDB (RO): {path}")
    return env


def open_lmdb_write(path: Path, map_size: int = 1 << 30) -> lmdb.Environment:
    # map_size default = 1GB; caller should set appropriately for large datasets
    path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(path), map_size=map_size, writemap=True, lock=True)
    log.info(f"Opened LMDB (RW): {path} with map_size={map_size}")
    return env


# -------------------------
# Checkpoint / Model loader
# -------------------------
def load_model_from_checkpoint(ckpt_path: Path, device: torch.device, model_factory: Optional[Callable] = None):
    """
    Loads a model instance from checkpoint.
    - ckpt is expected to be a dict containing at least 'model_state' or 'state_dict' and optionally 'config' and 'ema_state'.
    - model_factory: a callable that accepts checkpoint['config'] (or nothing) and returns an initialized model.
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = torch.load(str(ckpt_path), map_location=device)
    log.info(f"Loaded checkpoint keys: {list(payload.keys())}")

    # Common checkpoint key guesses
    state_key = None
    for k in ("model_state", "model_state_dict", "policy_state_dict", "state_dict", "policy"):
        if k in payload:
            state_key = k
            break
    if state_key is None:
        raise KeyError("No recognized model state in checkpoint. Expected one of model_state / state_dict / policy_state_dict / policy")

    config = payload.get("config", None)

    # Create model via factory if given; otherwise try to instantiate EgoPlanner.
    if model_factory is not None:
        model = model_factory(config)
    else:
        if EgoPlanner is None:
            raise RuntimeError("EgoPlanner import failed; provide model_factory to construct the model.")
        # If config provided and matches dataclass, user should adapt model creation accordingly
        model = EgoPlanner(config) if config is not None else EgoPlanner()

    model.to(device)
    sd = payload[state_key]
    # Carefully attempt to load
    missing, unexpected = model.load_state_dict(sd, strict=False) if isinstance(sd, dict) else (None, None)
    log.info(f"Model loaded. Missing keys? {getattr(missing, 'missing_keys', missing)}; Unexpected? {getattr(missing, 'unexpected_keys', unexpected)}")

    # If EMA present, store it on the model if attribute exists
    ema_state = None
    for k in ("ema_state", "ema_state_dict", "ema_state_dict"):
        if k in payload:
            ema_state = payload[k]
            break
    if ema_state is not None and hasattr(model, "ema"):
        try:
            model.ema.load_state_dict(ema_state)
            log.info("Loaded EMA state.")
        except Exception as e:
            log.warning(f"Failed to load EMA state: {e}")

    model.eval()
    return model, config


# -------------------------
# Offline evaluation
# -------------------------
def evaluate_offline(
    model,
    val_lmdb_path: Path,
    val_index_path: Path,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 0,
    sampling_steps: int = 50,
    guidance_scale: float = 1.0,
    use_ema: bool = True,
    max_batches: Optional[int] = None,
):
    """
    Evaluate policy offline by comparing sampled actions with ground truth actions.
    Expects ExpertTrajectoryDataset or similar that yields (obs_dict, actions_tensor) where actions shape = (B, H_a, action_dim).
    """
    if ExpertTrajectoryDataset is None or collate_fn is None:
        raise RuntimeError("ExpertTrajectoryDataset not importable from project utils. Install or adjust imports.")

    if not val_lmdb_path.exists():
        raise FileNotFoundError(f"Validation LMDB not found: {val_lmdb_path}")
    if not val_index_path.exists():
        raise FileNotFoundError(f"Validation index JSON not found: {val_index_path}")

    log.info("Building validation dataset...")
    dataset = ExpertTrajectoryDataset(
        val_lmdb_path,
        index_path=str(val_index_path),
        observation_horizon=getattr(model, "H_o", getattr(model, "obs_horizon", 2)),
        action_horizon=getattr(model, "H_a", getattr(model, "action_horizon", 8))
    )

    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=num_workers, pin_memory=True)

    mses = []
    maes = []
    nb = 0
    for batch_idx, (obs, actions) in enumerate(tqdm(loader, desc="Offline eval")):
        if max_batches is not None and batch_idx >= max_batches:
            break
        # Move to device
        obs = {k: v.to(device) for k, v in obs.items()}
        actions = actions.to(device)

        # sample from model
        with torch.no_grad():
            # Use model's sample method if present
            if hasattr(model, "sample"):
                pred = model.sample(
                    obs,
                    scheduler=getattr(model, "scheduler", None),
                    guidance_plan=guidance_scale,
                    guidance_obs=1.0,
                    num_inference_steps=sampling_steps
                )
            else:
                # fallback: if model callable directly returns predicted action sequences
                pred = model(obs)
        # pred should be shape (B, H_a, action_dim)
        # If model returns flattened shape, attempt to reshape
        if pred.dim() == 2:
            # assume (B, H_a * action_dim)
            B = actions.shape[0]
            Ha = actions.shape[1]
            D = actions.shape[2]
            pred = pred.view(B, Ha, D)

        mse = torch.nn.functional.mse_loss(pred, actions, reduction="mean").item()
        mae = torch.nn.functional.l1_loss(pred, actions, reduction="mean").item()
        mses.append(mse)
        maes.append(mae)
        nb += 1

    metrics = {"mean_mse": float(np.mean(mses)) if mses else None, "mean_mae": float(np.mean(maes)) if maes else None, "num_batches": nb}
    return metrics


# -------------------------
# Online rollouts + video
# -------------------------
def import_env_factory(factory_spec: str) -> Callable:
    """
    Import a callable env factory using 'module:callable' string.
    The factory should return an environment instance supporting reset() -> obs, step(action) -> (obs, reward, done, info), and render(mode='rgb_array') or render() -> ndarray.
    """
    modname, funcname = factory_spec.split(":")
    import importlib
    mod = importlib.import_module(modname)
    if not hasattr(mod, funcname):
        raise AttributeError(f"{modname} does not expose {funcname}")
    return getattr(mod, funcname)


def rollout_and_record(
    model,
    env_factory: Callable,
    cfg: dict,
    device: torch.device,
    out_dir: Path,
    n_rollouts: int = 10,
    max_steps: int = 200,
    sampling_steps: int = 50,
    guidance_scale: float = 1.0
) -> Dict:
    """
    Run closed-loop rollouts: at every step, get observation, run model.sample, apply first action, step env, record frames.
    Saves videos (mp4) for each rollout and returns per-rollout success flags and metadata.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    results = []
    factory = env_factory

    for i in range(n_rollouts):
        env = factory(cfg.get("env", {}))
        obs = env.reset()
        frames = []
        is_success = False
        for step in range(max_steps):
            # Build model-compatible obs dict
            # The exact obs formatting depends on dataset; we attempt common keys
            model_obs = {}
            # If env returns dict with 'observation' and 'achieved_goal' etc.
            if isinstance(obs, dict) and "observation" in obs:
                for k, v in obs["observation"].items():
                    if isinstance(v, np.ndarray):
                        model_obs[k] = torch.from_numpy(v).unsqueeze(0).to(device).float()
                    else:
                        # fallback
                        model_obs[k] = torch.tensor(v).unsqueeze(0).to(device).float()
            elif isinstance(obs, dict):
                # flatten dict values that are arrays
                for k, v in obs.items():
                    if isinstance(v, np.ndarray):
                        model_obs[k] = torch.from_numpy(v).unsqueeze(0).to(device).float()
            else:
                raise RuntimeError("Unsupported env observation format. Provide env factory that returns dict observations.")

            # sample action sequence
            with torch.no_grad():
                if hasattr(model, "sample"):
                    sampled_actions = model.sample(
                        model_obs,
                        scheduler=getattr(model, "scheduler", None),
                        guidance_plan=guidance_scale,
                        guidance_obs=1.0,
                        num_inference_steps=sampling_steps
                    )
                else:
                    sampled_actions = model(model_obs)
            # pick first action of sequence
            if isinstance(sampled_actions, torch.Tensor):
                action = sampled_actions.cpu().numpy()
                if action.ndim == 3:
                    action = action[0, 0]  # first batch, first step
                elif action.ndim == 2:
                    action = action[0]
                else:
                    # unexpected shape
                    action = action.ravel()
            else:
                # fallback: assume numpy array
                action = np.asarray(sampled_actions)
                if action.ndim == 3:
                    action = action[0, 0]

            # step environment
            next_obs, rew, done, info = env.step(action)
            # render frame (prefer rgb array)
            try:
                frame = env.render(mode="rgb_array")
            except TypeError:
                try:
                    frame = env.render()
                except Exception:
                    frame = None
            if frame is not None:
                frames.append(frame)
            obs = next_obs
            if isinstance(info, dict) and info.get("is_success"):
                is_success = True
            if done:
                break

        # Save video
        vid_path = video_dir / f"rollout_{i:03d}_success_{int(is_success)}.mp4"
        if frames:
            # imageio expects uint8 frames
            try:
                imageio.mimwrite(str(vid_path), [np.asarray(f).astype(np.uint8) for f in frames], fps=25, macro_block_size=None)
            except Exception as e:
                log.warning(f"Failed to write video via imageio: {e}. Trying fallback with cv2.")
                try:
                    import cv2
                    h, w, _ = frames[0].shape
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    vw = cv2.VideoWriter(str(vid_path), fourcc, 25.0, (w, h))
                    for f in frames:
                        vw.write(cv2.cvtColor(np.asarray(f).astype(np.uint8), cv2.COLOR_RGB2BGR))
                    vw.release()
                except Exception as e2:
                    log.error(f"Fallback video writer failed: {e2}")
        else:
            log.info("No frames captured for rollout; skipping video write.")

        env.close()
        results.append({"rollout": i, "success": bool(is_success), "steps": len(frames), "video": str(vid_path) if frames else None})
        log.info(f"Rollout {i} finished: success={is_success}, steps={len(frames)}, video={vid_path if frames else 'none'}")
    # return aggregated metrics
    successes = [r["success"] for r in results]
    metrics = {"mean_success_rate": float(np.mean(successes)), "std_success_rate": float(np.std(successes)), "n_rollouts": len(results)}
    return {"metrics": metrics, "per_rollout": results, "video_dir": str(video_dir)}


# -------------------------
# LMDB merging utility
# -------------------------
def merge_lmdbs(
    src_paths: List[Path],
    src_index_paths: List[Path],
    dst_lmdb: Path,
    dst_index: Path,
    map_size_gb: int = 8
) -> Tuple[int, Path]:
    """
    Merge/concatenate multiple LMDBs that follow the project's index format.
    - Expects each source index JSON to have top-level "episodes": list of episode metadata with "episode_id" and per-modality keys referencing LMDB keys.
    - Re-keys episodes in the dst LMDB as ep_00000000 ... ep_NNNNNN to avoid conflicts.
    - Copies raw blobs from each source LMDB and writes new index JSON.
    Returns (num_episodes_written, dst_index_path)
    """
    assert len(src_paths) == len(src_index_paths), "src_paths and src_index_paths length mismatch"
    log.info(f"Merging {len(src_paths)} LMDBs into {dst_lmdb}")
    # prepare dst
    map_size = int(map_size_gb * (1 << 30))
    dst_env = open_lmdb_write(dst_lmdb, map_size=map_size)
    merged_index = {"episodes": [], "metadata": {}}

    total_written = 0
    try:
        with dst_env.begin(write=True) as dst_txn:
            for src_path, src_index in zip(src_paths, src_index_paths):
                src_idx = safe_load_json(Path(src_index))
                if "episodes" not in src_idx:
                    log.warning(f"No episodes key in index {src_index}. Skipping.")
                    continue
                # open src env read-only:
                src_env = open_lmdb_read(Path(src_path))
                with src_env.begin() as src_txn:
                    for ep_meta in src_idx["episodes"]:
                        old_ep_id = ep_meta["episode_id"]
                        new_ep_id = f"ep_{total_written:08d}"
                        new_ep = dict(ep_meta)
                        new_ep["episode_id"] = new_ep_id
                        # re-key every modality referencing an old key
                        for mod_key, mod_meta in new_ep.get("modalities", {}).items():
                            old_key_str = mod_meta.get("key")
                            if old_key_str is None:
                                continue
                            old_key = old_key_str.encode("ascii")
                            val = src_txn.get(old_key)
                            if val is None:
                                log.warning(f"Missing key {old_key_str} in source {src_path}. Skipping episode {old_ep_id}.")
                                new_ep = None
                                break
                            # new key string replace old ep id substring if present, otherwise prefix with new ep id
                            if old_ep_id in old_key_str:
                                new_key_str = old_key_str.replace(old_ep_id, new_ep_id)
                            else:
                                new_key_str = f"{new_ep_id}_{mod_key}"
                            new_key = new_key_str.encode("ascii")
                            dst_txn.put(new_key, val)
                            # update index
                            mod_meta["key"] = new_key_str
                        if new_ep is None:
                            continue
                        merged_index["episodes"].append(new_ep)
                        total_written += 1
                src_env.close()
        # persist index JSON
        dst_index.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_index, "w") as f:
            json.dump(merged_index, f, indent=2)
        log.info(f"Merging complete. Wrote {total_written} episodes and index -> {dst_index}")
    finally:
        dst_env.sync()
        dst_env.close()
    return total_written, dst_index


# -------------------------
# CLI main
# -------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Evaluate and visualize Ego-Planner / Diffusion policy, and merge LMDBs.")
    grp = p.add_mutually_exclusive_group(required=False)
    grp.add_argument("--offline", action="store_true", help="Run offline eval on validation dataset")
    grp.add_argument("--online", action="store_true", help="Run online rollouts in env")
    p.add_argument("--ckpt", type=str, help="Model checkpoint path (required for offline or online eval).")
    p.add_argument("--device", type=str, default="cpu", help="Device to run evaluation on (cpu / cuda).")
    # offline args
    p.add_argument("--val-lmdb", type=str, help="Validation LMDB path")
    p.add_argument("--val-index", type=str, help="Validation index JSON path")
    p.add_argument("--batch-size", type=int, default=8)
    # online args
    p.add_argument("--env-factory", type=str, help="Env factory in module:callable format returning a fresh env when called with config dict")
    p.add_argument("--env-config", type=str, default=None, help="Optional JSON file with environment configuration passed to factory")
    p.add_argument("--n-rollouts", type=int, default=6)
    p.add_argument("--max-steps", type=int, default=200)
    # merging args
    p.add_argument("--merge", nargs=2, metavar=("SRC1", "SRC2"), help="Merge two LMDB files: provide two lmdb paths")
    p.add_argument("--merge-index", nargs=2, metavar=("IDX1", "IDX2"), help="Corresponding index jsons for the two LMDBs")
    p.add_argument("--out-merged", type=str, help="Destination merged LMDB path")
    p.add_argument("--out-index", type=str, help="Destination merged index JSON path")
    # sampling params
    p.add_argument("--sampling-steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--outdir", type=str, default="runs/eval")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    log.info(f"Evaluation outdir: {outdir}; device: {device}")

    # 1) LMDB merging if requested
    if args.merge:
        if not args.merge_index or not args.out_merged or not args.out_index:
            raise RuntimeError("When using --merge you must also supply --merge-index SRC_IDX1 SRC_IDX2 --out-merged PATH --out-index PATH")
        src_paths = [Path(x) for x in args.merge]
        src_index_paths = [Path(x) for x in args.merge_index]
        dst_lmdb = Path(args.out_merged)
        dst_index = Path(args.out_index)
        num_written, idx_path = merge_lmdbs(src_paths, src_index_paths, dst_lmdb, dst_index)
        log.info(f"Merged LMDBs -> wrote {num_written} episodes. Index at {idx_path}")
        # If only merging, exit
        if not (args.offline or args.online):
            return

    # 2) Load model if needed
    model = None
    config = None
    if args.ckpt:
        model, config = load_model_from_checkpoint(Path(args.ckpt), device)
        log.info("Model loaded successfully.")
    else:
        if args.offline or args.online:
            raise RuntimeError("Checkpoint (--ckpt) is required for offline or online evaluation.")

    # 3) offline evaluation
    results = {"meta": {"time": time.time(), "device": str(device)}}
    if args.offline:
        if not args.val_lmdb or not args.val_index:
            raise RuntimeError("--val-lmdb and --val-index must be provided for offline evaluation.")
        metrics = evaluate_offline(
            model,
            Path(args.val_lmdb),
            Path(args.val_index),
            device,
            batch_size=args.batch_size,
            sampling_steps=args.sampling_steps,
            guidance_scale=args.guidance,
        )
        results["offline"] = metrics
        with open(outdir / "offline_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        log.info(f"Offline metrics saved to {outdir / 'offline_metrics.json'}")

    # 4) online evaluation
    if args.online:
        if not args.env_factory:
            raise RuntimeError("--env-factory must be provided for online evaluation.")
        env_factory = import_env_factory(args.env_factory)
        env_cfg = {}
        if args.env_config:
            env_cfg = safe_load_json(Path(args.env_config))
        rollout_info = rollout_and_record(
            model,
            env_factory,
            {"env": env_cfg},
            device,
            outdir,
            n_rollouts=args.n_rollouts,
            max_steps=args.max_steps,
            sampling_steps=args.sampling_steps,
            guidance_scale=args.guidance
        )
        results["online"] = rollout_info
        with open(outdir / "online_metrics.json", "w") as f:
            json.dump(rollout_info, f, indent=2)
        log.info(f"Online metrics & videos saved to {outdir}")

    # 5) finalize
    with open(outdir / "eval_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("Evaluation finished. Summary saved to eval_summary.json")


if __name__ == "__main__":
    main()
