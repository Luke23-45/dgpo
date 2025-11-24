#!/usr/bin/env python3
"""
generate_dataset.py - Robust LMDB-sharded dataset generator.

This definitive version includes episode-based generation control, SOTA
SoA formatting, and robust multiprocessing orchestration with AUTO-CLEANUP.
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
import shutil
import copy
from typing import Optional

# project imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.expert_dataset import ExpertDataset
from utils.scripted_expert import ExpertConfig
from utils.expert_dataset import ExpertDatasetWriter

try:
    import lmdb
except ImportError:
    lmdb = None

logger = logging.getLogger("generate_dataset")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def merge_sota_shards(shard_dirs: list[Path], out_dir: Path, run_name: str, total_target: int, use_episode_target: bool):
    """
    Merges SOTA-formatted shards by combining their JSON indexes and copying
    the LMDB files into a single, unified dataset directory.
    """
    if use_episode_target:
        final_dataset_name = f"expert_{run_name}_{total_target}_episodes"
    else:
        final_dataset_name = f"expert_{run_name}_{total_target}_samples"

    final_dataset_path = out_dir / final_dataset_name
    final_dataset_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Creating final merged dataset at: {final_dataset_path}")

    merged_index = {"episodes": [], "metadata": {}}
    total_episodes = 0
    all_keys_in_use = set()

    # We need a central LMDB writer for the merged data
    final_lmdb_path = final_dataset_path / f"{final_dataset_name}.lmdb"
    map_size = int(30 * 1024**3)  # 100 GB, adjust as needed
    final_env = lmdb.open(str(final_lmdb_path), map_size=map_size, subdir=False, readonly=False, lock=True)

    try:
        with final_env.begin(write=True) as final_txn:
            for shard_dir in tqdm(shard_dirs, desc="Merging Shards"):
                shard_index_files = list(shard_dir.glob("*_index.json"))
                shard_lmdb_files = list(shard_dir.glob("*.lmdb"))

                if not shard_index_files or not shard_lmdb_files:
                    logger.warning(f"Shard at {shard_dir} is incomplete, skipping.")
                    continue

                shard_index_path = shard_index_files[0]
                shard_lmdb_path = shard_lmdb_files[0]

                with open(shard_index_path, "r") as f:
                    shard_index_data = json.load(f)

                if not merged_index["metadata"] and "metadata" in shard_index_data:
                    merged_index["metadata"] = shard_index_data["metadata"]

                shard_env = lmdb.open(str(shard_lmdb_path), subdir=False, readonly=True, lock=False)
                with shard_env.begin() as shard_txn:
                    for ep_meta in shard_index_data["episodes"]:
                        new_ep_id = f"ep_{total_episodes:06d}"
                        new_ep_meta = copy.deepcopy(ep_meta)
                        new_ep_meta["episode_id"] = new_ep_id

                        for modality_name, modality_meta in ep_meta["modalities"].items():
                            old_modality_key = modality_meta["key"]
                            new_modality_key = old_modality_key.replace(ep_meta["episode_id"], new_ep_id)
                            new_ep_meta["modalities"][modality_name]["key"] = new_modality_key

                            data_blob = shard_txn.get(old_modality_key.encode('ascii'))
                            if data_blob:
                                final_txn.put(new_modality_key.encode('ascii'), data_blob)
                                all_keys_in_use.add(new_modality_key)
                            else:
                                logger.warning(f"Key {old_modality_key} not found in shard {shard_lmdb_path}")

                        merged_index["episodes"].append(new_ep_meta)
                        total_episodes += 1
                shard_env.close()

    finally:
        final_env.sync()
        final_env.close()

    final_index_path = final_dataset_path / f"{final_dataset_name}_index.json"
    with open(final_index_path, "w") as f:
        json.dump(merged_index, f)

    logger.info(f"Merge complete. Total episodes: {total_episodes}")
    return total_episodes

def worker_loop_fn_SOTA(worker_id: int, cfg: dict, shard_dir_path_str: str,
                        summary_path_str: str, samples_per_worker: Optional[int] = None,
                        episodes_per_worker: Optional[int] = None):
    """
    SOTA Worker entrypoint.
    """
    if samples_per_worker is None and episodes_per_worker is None:
        raise ValueError("Must provide either samples_per_worker or episodes_per_worker target.")

    shard_dir_path = Path(shard_dir_path_str)
    summary_path = Path(summary_path_str)
    log_prefix = f"[worker {worker_id}]"

    try:
        run_name = f"shard_w{worker_id}"
        target_str = f"{episodes_per_worker} episodes" if episodes_per_worker is not None else f"{samples_per_worker} samples"
        logging.info(f"{log_prefix} starting. seed_base={cfg.get('seed',0)} target={target_str} shard_dir={shard_dir_path}")

        writer = ExpertDatasetWriter(
            out_dir=str(shard_dir_path),
            run_name=run_name,
            image_compression=cfg.get("image_compression", "jpeg"),
            jpeg_quality=cfg.get("jpeg_quality", 90)
        )

        expert_config_dict = cfg.get("expert_config", {})
        expert_config_instance = ExpertConfig(**expert_config_dict)
        base_seed = int(cfg.get("seed", 0)) if cfg.get("seed") is not None else int(time.time())
        seed_for_worker = base_seed + worker_id * cfg.get("worker_seed_offset", 10000)

        ds = ExpertDataset(
            urdf_path=cfg["urdf_path"],
            env_xml_path=cfg.get("xml_path"),
            base_seed=seed_for_worker,
            max_samples_per_epoch=samples_per_worker,
            max_episodes_per_epoch=episodes_per_worker,
            skip_on_error=cfg.get("skip_on_error", True),
            scripted_cfg=expert_config_instance,
            object_size=tuple(np.array(cfg.get("object_size", [0.04,0.04,0.04])).tolist()),
            object_grasp_width=float(cfg.get("grasp_width", 0.6)),
            action_scaling_factor=float(cfg.get("action_scaling_factor", 0.5)),
            warmup=bool(cfg.get("warmup", True)),
            yield_full_obs=True,
        )

        last_saved_episode_count = 0
        use_episode_target = episodes_per_worker is not None
        pbar_total = episodes_per_worker if use_episode_target else samples_per_worker
        pbar_desc = f"Worker {worker_id} (Episodes)" if use_episode_target else f"Worker {worker_id} (Samples)"
        pbar = tqdm(total=pbar_total, desc=pbar_desc, leave=True)

        for _ in ds: 
            if use_episode_target:
                if ds._episode_id_counter > pbar.n:
                    pbar.update(ds._episode_id_counter - pbar.n)
            else:
                pbar.update(1) 

          
            if len(ds.episodes) > 0:
                writer.save_batch(ds.episodes)
                ds.episodes.clear()

      
        pbar.close()
        writer.save() 
        
        final_stats = ds.get_stats()
        summary = {
            "worker_id": worker_id,
            "status": "ok",
            "written_episodes": final_stats["episodes_collected"],
            "samples_yielded": final_stats["samples_yielded"],
            "shard_dir_path": str(shard_dir_path),
            "seed_used": int(seed_for_worker)
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logging.info(f"{log_prefix} finished. Wrote {final_stats['episodes_collected']} episodes to {shard_dir_path}")

    except Exception as e:
        logging.exception(f"{log_prefix} failed: {e}")
        summary = { "worker_id": worker_id, "status": "error", "error": str(e) }
        try:
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
        except Exception: pass
        raise

def main():
    parser = argparse.ArgumentParser(description="Robust dataset generation (LMDB-sharded)")
    parser.add_argument("--config", required=True, help="YAML config")
    parser.add_argument("--out_dir", default=None, help="Final output directory (overrides config)")
    parser.add_argument("--resume", action="store_true", help="Resume mode (do not clobber existing shards)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    
    if args.resume and "seed" in cfg:
        logger.info(f"Resume mode: Shifting base seed {cfg['seed']} by +999999 to avoid duplicates.")
        cfg["seed"] = int(cfg["seed"]) + 999999

    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    run_name = cfg.get("run_name", time.strftime("%Y%m%d_%H%M%S"))
    num_workers = int(cfg.get("num_workers", 0))

    shards_base_dir = out_dir / "shards"
    shards_base_dir.mkdir(parents=True, exist_ok=True)
    
    if "num_episodes" in cfg and cfg["num_episodes"] is not None:
        use_episode_target = True
        total_target = int(cfg["num_episodes"])
        target_per_worker = (total_target + num_workers - 1) // num_workers if num_workers > 0 else total_target
        logger.info(f"TARGET MODE: EPISODES. Total: {total_target}, Per Worker: {target_per_worker}")
    else:
        use_episode_target = False
        total_target = int(cfg["num_samples"])
        target_per_worker = (total_target + num_workers - 1) // num_workers if num_workers > 0 else total_target
        logger.info(f"TARGET MODE: SAMPLES. Total: {total_target}, Per Worker: {target_per_worker}")

    ctx = mp.get_context("spawn")
    worker_processes = []
    shard_dirs_to_merge = []

    # [HELPER] Robustly delete directory (handles Windows locking)
    def robust_rmtree(path, retries=3):
        for i in range(retries):
            try:
                if path.exists():
                    shutil.rmtree(path)
                return
            except Exception:
                if i < retries - 1:
                    time.sleep(1.0)
                else:
                    logger.warning(f"Could not fully delete {path} after retries. Proceeding anyway.")

    if num_workers > 0:
        for w in range(num_workers):
            shard_dir = shards_base_dir / f"worker_{w}"
            summary_file = shard_dir / "summary.json"
            shard_dirs_to_merge.append(shard_dir)

            # [PATCH] Aggressive Directory Cleaning
            if args.resume:
                if summary_file.exists():
                    try:
                        with open(summary_file, 'r') as f:
                            summary = json.load(f)
                            if summary.get("status") == "ok":
                                logger.info(f"Shard for worker {w} already complete. Skipping.")
                                continue
                    except Exception:
                        pass # File corrupted, will re-run
            else:
                # Force Clean if not resuming
                if shard_dir.exists():
                    logger.info(f"Cleaning existing shard directory: {shard_dir}")
                    robust_rmtree(shard_dir)

            shard_dir.mkdir(parents=True, exist_ok=True)

            worker_kwargs = {
                "worker_id": w, "cfg": cfg, "shard_dir_path_str": str(shard_dir),
                "summary_path_str": str(summary_file),
                "samples_per_worker": None if use_episode_target else target_per_worker,
                "episodes_per_worker": target_per_worker if use_episode_target else None
            }
            p = ctx.Process(target=worker_loop_fn_SOTA, kwargs=worker_kwargs, daemon=False)
            p.start()
            worker_processes.append(p)
        
        for p in worker_processes:
            p.join()
            
    else: # Single-threaded
        shard_dir = shards_base_dir / "worker_0"
        summary_file = shard_dir / "summary.json"
        shard_dirs_to_merge.append(shard_dir)
        
        if not args.resume and shard_dir.exists():
            logger.info(f"Cleaning existing shard directory: {shard_dir}")
            robust_rmtree(shard_dir)
        
        shard_dir.mkdir(parents=True, exist_ok=True)
        try:
            worker_kwargs = {
                "worker_id": 0, "cfg": cfg, "shard_dir_path_str": str(shard_dir),
                "summary_path_str": str(summary_file),
                "samples_per_worker": None if use_episode_target else target_per_worker,
                "episodes_per_worker": target_per_worker if use_episode_target else None
            }
            worker_loop_fn_SOTA(**worker_kwargs)
        except Exception:
            logger.exception("Single-threaded generation failed")
            return

    logger.info(f"All worker shards written to subdirectories in: {shards_base_dir}")

    # Final merge step
    all_workers_succeeded = True
    for shard_dir in shard_dirs_to_merge:
        summary_file = shard_dir / "summary.json"
        if not summary_file.exists():
            all_workers_succeeded = False
            logger.error(f"Worker shard at {shard_dir} is missing a summary file. Merge might be incomplete.")
            continue
        with open(summary_file, 'r') as f:
            try:
                summary = json.load(f)
                if summary.get("status") != "ok":
                    all_workers_succeeded = False
                    logger.error(f"Worker {summary.get('worker_id')} at {shard_dir} reported an error.")
            except Exception:
                all_workers_succeeded = False

    if all_workers_succeeded or args.resume:
        if not all_workers_succeeded:
            logger.warning("Resuming and merging despite some workers failing. The dataset may be smaller than targeted.")
        merged_count = merge_sota_shards(shard_dirs_to_merge, out_dir, run_name, total_target, use_episode_target)
        logger.info(f"Merged episodes count: {merged_count}")
        # Optional: Clean up shard directories after successful merge
        try:
            robust_rmtree(shards_base_dir)
        except:
            pass
    else:
        logger.error("One or more workers failed and not in resume mode. Skipping final merge.")

    logger.info("Dataset generation complete.")

if __name__ == "__main__":
    main()


# python -m scripts.generate_dataset --config "configs\gen_dataset_config.yaml" 