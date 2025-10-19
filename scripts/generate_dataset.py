#!/usr/bin/env python3
"""
generate_dataset.py - Robust LMDB-sharded dataset generator.

This definitive version ensures consistency by using single-file LMDBs (`subdir=False`),
which is more robust on Windows and aligns with the data loading pipeline.
"""

from __future__ import annotations
import os
import sys
import time
import yaml
import logging
import hashlib
import json
from pathlib import Path
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import pickle
import argparse
# project imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.expert_dataset import ExpertDataset, replay_validate_episode
from utils.scripted_expert import ExpertConfig # Ensure this import is correct

try:
    import lmdb
except ImportError:
    lmdb = None

logger = logging.getLogger("generate_dataset")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def open_lmdb_writer(db_path: Path, map_size: int):
    """Opens a SINGLE-FILE LMDB environment for writing."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if lmdb is None:
        raise RuntimeError("lmdb not available; please install python-lmdb.")
    return lmdb.open(str(db_path), map_size=map_size, subdir=False, readonly=False, lock=True)

def worker_stream_write_lmdb(env, base_key_idx: int, episodes_iter):
    idx = int(base_key_idx)
    count = 0
    with env.begin(write=True) as txn:
        for ep in episodes_iter:
            key = f"{idx:08d}".encode("ascii")
            val = pickle.dumps(ep, protocol=pickle.HIGHEST_PROTOCOL)
            txn.put(key, val)
            idx += 1
            count += 1
    return idx, count

def worker_loop_fn(worker_id: int, cfg: dict, shard_file_path_str: str, shard_map_size: int, samples_per_worker: int, summary_path_str: str):
    """Worker entrypoint that streams episodes to a single LMDB file."""
    shard_file_path = Path(shard_file_path_str)
    summary_path = Path(summary_path_str)
    log_prefix = f"[worker {worker_id}]"
    
    try:
        logging.info(f"{log_prefix} starting. seed base={cfg.get('seed',0)} target={samples_per_worker} shard={shard_file_path}")

        expert_config_dict = cfg.get("expert_config", {})
        expert_config_instance = ExpertConfig(**expert_config_dict)

        base_seed = int(cfg.get("seed", 0)) if cfg.get("seed") is not None else int(time.time())
        seed_for_worker = base_seed + worker_id * cfg.get("worker_seed_offset", 10000)
        
        ds = ExpertDataset(
            urdf_path=cfg["urdf_path"],
            env_xml_path=cfg.get("xml_path"),
            base_seed=seed_for_worker,
            max_samples_per_epoch=samples_per_worker,
            skip_on_error=cfg.get("skip_on_error", True),
            scripted_cfg=expert_config_instance,
            object_size=tuple(np.array(cfg.get("object_size", [0.04,0.04,0.04])).tolist()),
            object_grasp_width=float(cfg.get("grasp_width", 0.6)),
            action_scaling_factor=float(cfg.get("action_scaling_factor", 0.5)),
            warmup=bool(cfg.get("warmup", True)),
            yield_full_obs=True,
        )

        env = open_lmdb_writer(shard_file_path, map_size=shard_map_size)
        written_episodes = 0
        key_idx = 0
        last_saved_episode_count = 0

        pbar = tqdm(total=samples_per_worker, desc=f"Worker {worker_id}", leave=True)
        samples_yielded_by_ds = 0

        for _ in ds:
            samples_yielded_by_ds += 1
            pbar.update(1)
            
            if len(ds.episodes) > last_saved_episode_count:
                new_eps = ds.episodes[last_saved_episode_count:]
                _, wrote = worker_stream_write_lmdb(env, key_idx, new_eps)
                written_episodes += wrote
                key_idx += wrote
                last_saved_episode_count = len(ds.episodes)
        
        pbar.close()

        if len(ds.episodes) > last_saved_episode_count:
            new_eps = ds.episodes[last_saved_episode_count:]
            _, wrote = worker_stream_write_lmdb(env, key_idx, new_eps)
            written_episodes += wrote
        
        env.sync()
        env.close()

        summary = { "worker_id": worker_id, "status": "ok", "written_episodes": written_episodes, "samples_yielded": samples_yielded_by_ds, "shard_path": str(shard_file_path), "seed_used": int(seed_for_worker) }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logging.info(f"{log_prefix} finished. wrote {written_episodes} episodes to {shard_file_path}")

    except Exception as e:
        logging.exception(f"{log_prefix} failed: {e}")
        summary = { "worker_id": worker_id, "status": "error", "error": str(e) }
        try:
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
        except Exception: pass
        raise

def merge_shards_to_lmdb(shard_files: list[Path], out_file: Path, map_size: int):
    out_env = open_lmdb_writer(out_file, map_size=map_size)
    nxt = 0
    with out_env.begin(write=True) as out_txn:
        for shard_file in shard_files:
            if not shard_file.exists():
                logger.warning(f"Shard file not found, skipping: {shard_file}")
                continue
            
            env = lmdb.open(str(shard_file), subdir=False, readonly=True, lock=False)
            with env.begin() as txn:
                cursor = txn.cursor()
                for k, v in tqdm(cursor, desc=f"Merging {shard_file.name}", leave=False):
                    key = f"{nxt:08d}".encode("ascii")
                    out_txn.put(key, v)
                    nxt += 1
            env.close()
    out_env.sync()
    out_env.close()
    return nxt

def main():
    parser = argparse.ArgumentParser(description="Robust dataset generation (LMDB-sharded)")
    parser.add_argument("--config", required=True, help="YAML config")
    parser.add_argument("--out_dir", default=None, help="Final output directory (overrides config)")
    parser.add_argument("--resume", action="store_true", help="Resume mode (do not clobber existing shards)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    run_name = time.strftime("%Y%m%d_%H%M%S") # Generate a run name

    num_workers = int(cfg.get("num_workers", 0))
    total_samples = int(cfg["num_samples"])

    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    
    samples_per_worker = (total_samples + num_workers - 1) // num_workers if num_workers > 0 else total_samples

    shard_map_size = int(cfg.get("shard_map_size_bytes", 13 * 1024**3))
    final_map_size = int(cfg.get("final_map_size_bytes", 13 * 1024**3))

    ctx = mp.get_context("spawn")
    worker_processes, summary_paths, shard_paths = [], [], []

    if num_workers > 0:
        for w in range(num_workers):
            shard_file = shards_dir / f"shard_w{w}.lmdb"
            summary_file = shards_dir / f"summary_w{w}.json"
            if args.resume and shard_file.exists():
                logger.info(f"Shard {shard_file} exists and resume=True — skipping worker {w}")
                shard_paths.append(shard_file)
                continue

            p = ctx.Process(target=worker_loop_fn, args=(w, cfg, str(shard_file), shard_map_size, samples_per_worker, str(summary_file)), daemon=False)
            p.start()
            worker_processes.append(p)
            shard_paths.append(shard_file)
        
        for p in worker_processes:
            p.join()
    else:
        shard_file = shards_dir / "shard_single.lmdb"
        summary_file = shards_dir / "summary_single.json"
        if args.resume and shard_file.exists():
            logger.info(f"Shard {shard_file} exists and resume=True — using existing shard")
            shard_paths.append(shard_file)
        else:
            try:
                worker_loop_fn(0, cfg, str(shard_file), shard_map_size, samples_per_worker, str(summary_file))
                shard_paths.append(shard_file)
            except Exception:
                logger.exception("Single-threaded generation failed")
                return

    logger.info(f"All worker shards written (shard dir: {shards_dir})")

    final_file = out_dir / f"expert_{run_name}_{total_samples}_samples.lmdb"
    logger.info(f"Merging {len(shard_paths)} shards into final LMDB at {final_file}")
    merged_count = merge_shards_to_lmdb([p for p in shard_paths if p.exists()], final_file, map_size=final_map_size)
    logger.info(f"Merged episodes count: {merged_count}")

    logger.info("Dataset generation complete.")

if __name__ == "__main__":
    main()