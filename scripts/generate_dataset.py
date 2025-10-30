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
import shutil
import copy
# project imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.expert_dataset import ExpertDataset, replay_validate_episode
from utils.scripted_expert import ExpertConfig # Ensure this import is correct
from utils.expert_dataset import ExpertDatasetWriter # Add this import at the top

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


# ...

# DELETE the old merge_shards_to_lmdb function.

# --- START OF SOTA PATCH 2 ---

def merge_sota_shards(shard_dirs: list[Path], out_dir: Path, run_name: str, total_samples: int):
    """
    Merges SOTA-formatted shards by combining their JSON indexes and copying
    the LMDB files into a single, unified dataset directory.
    """
    final_dataset_name = f"expert_{run_name}_{total_samples}_samples"
    final_dataset_path = out_dir / final_dataset_name
    final_dataset_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Creating final merged dataset at: {final_dataset_path}")

    merged_index = {"episodes": [], "metadata": {}}
    total_episodes = 0

    for shard_dir in tqdm(shard_dirs, desc="Merging Shards"):
        # Find the index and lmdb file in the shard directory
        shard_index_files = list(shard_dir.glob("*_index.json"))
        shard_lmdb_files = list(shard_dir.glob("*.lmdb"))

        if not shard_index_files or not shard_lmdb_files:
            logger.warning(f"Shard at {shard_dir} is incomplete, skipping.")
            continue
        
        shard_index_path = shard_index_files[0]
        shard_lmdb_path = shard_lmdb_files[0]

        # 1. Load the shard's index
        with open(shard_index_path, "r") as f:
            shard_index_data = json.load(f)
        
        if not merged_index["metadata"]:
            merged_index["metadata"] = shard_index_data.get("metadata", {})

        # 2. Re-key and append episode metadata
        for ep_meta in shard_index_data["episodes"]:
            new_ep_id = f"ep_{total_episodes:06d}"
            
            # Create a deep copy to modify
            new_ep_meta = copy.deepcopy(ep_meta)
            new_ep_meta["episode_id"] = new_ep_id
            
            # IMPORTANT: Update the internal keys for each modality
            for modality_name in new_ep_meta["modalities"]:
                old_modality_key = new_ep_meta["modalities"][modality_name]["key"]
                # The key was like "ep_000001_actions", based on local episode ID
                # We need to find the old local ID to replace it
                old_ep_id = ep_meta["episode_id"]
                new_modality_key = old_modality_key.replace(old_ep_id, new_ep_id)
                new_ep_meta["modalities"][modality_name]["key"] = new_modality_key

            merged_index["episodes"].append(new_ep_meta)
            total_episodes += 1
            
        # 3. Copy the LMDB file to the final destination with a new name
        final_lmdb_path = final_dataset_path / f"{shard_lmdb_path.stem}.lmdb"
        shutil.copy(shard_lmdb_path, final_lmdb_path)

    # 4. Write the final, merged JSON index
    final_index_path = final_dataset_path / f"{final_dataset_name}_index.json"
    with open(final_index_path, "w") as f:
        json.dump(merged_index, f)
    
    logger.info(f"Merge complete. Total episodes: {total_episodes}")
    return total_episodes

# --- END OF SOTA PATCH 2 ---

def worker_loop_fn_SOTA(worker_id: int, cfg: dict, shard_dir_path_str: str, samples_per_worker: int, summary_path_str: str):
    """
    SOTA Worker entrypoint that uses ExpertDatasetWriter to generate an
    optimized, SoA-formatted, self-contained dataset shard.
    """
    shard_dir_path = Path(shard_dir_path_str)
    summary_path = Path(summary_path_str)
    log_prefix = f"[worker {worker_id}]"

    try:
        run_name = f"shard_w{worker_id}"
        logging.info(f"{log_prefix} starting. seed_base={cfg.get('seed',0)} target={samples_per_worker} shard_dir={shard_dir_path}")

        # --- 1. Initialize the SOTA Writer for this specific shard ---
        writer = ExpertDatasetWriter(
            out_dir=str(shard_dir_path),
            run_name=run_name,
            image_compression=cfg.get("image_compression", "jpeg"),
            jpeg_quality=cfg.get("jpeg_quality", 90)
        )

        # --- 2. Initialize the Online Data Generator (ExpertDataset) ---
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

        # --- 3. Run the Generation Loop and Stream to the Writer ---
        last_saved_episode_count = 0
        pbar = tqdm(total=samples_per_worker, desc=f"Worker {worker_id}", leave=True)
        samples_yielded_by_ds = 0

        for _ in ds:
            samples_yielded_by_ds += 1
            pbar.update(1)

            # Check if new episodes have been collected by the generator
            if len(ds.episodes) > last_saved_episode_count:
                new_eps_to_write = ds.episodes[last_saved_episode_count:]
                
                # Use the writer's batch saving mechanism
                writer.save_batch(new_eps_to_write)
                
                last_saved_episode_count = len(ds.episodes)
        
        pbar.close()

        # Save any remaining episodes that didn't form a full batch
        if len(ds.episodes) > last_saved_episode_count:
            writer.save_batch(ds.episodes[last_saved_episode_count:])

        # Finalize the writer (this saves the final index.json for the shard)
        # The save() method is now idempotent if save_batch was used, it will just finalize.
        writer.save() 
        
        # --- 4. Write Worker Summary ---
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

    run_name = time.strftime("%Y%m%d_%H%M%S")

    num_workers = int(cfg.get("num_workers", 0))
    total_samples = int(cfg["num_samples"])

    # --- START OF PATCH 3 ---

    # Each worker now creates a directory, not a single file
    shards_base_dir = out_dir / "shards"
    shards_base_dir.mkdir(parents=True, exist_ok=True)
    
    samples_per_worker = (total_samples + num_workers - 1) // num_workers if num_workers > 0 else total_samples

    ctx = mp.get_context("spawn")
    worker_processes = []
    shard_dirs_to_merge = []

    if num_workers > 0:
        for w in range(num_workers):
            shard_dir = shards_base_dir / f"worker_{w}"
            summary_file = shard_dir / "summary.json"
            shard_dirs_to_merge.append(shard_dir)

            if args.resume and summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    if summary.get("status") == "ok":
                        logger.info(f"Shard for worker {w} at {shard_dir} already complete. Skipping.")
                        continue

            # Ensure the directory exists for the worker
            shard_dir.mkdir(exist_ok=True)

            # Call the new SOTA worker function
            p = ctx.Process(target=worker_loop_fn_SOTA, args=(w, cfg, str(shard_dir), samples_per_worker, str(summary_file)), daemon=False)
            p.start()
            worker_processes.append(p)
        
        for p in worker_processes:
            p.join()
    else: # Single-threaded case
        shard_dir = shards_base_dir / "worker_0"
        summary_file = shard_dir / "summary.json"
        shard_dirs_to_merge.append(shard_dir)
        
        # ... (resume logic for single worker) ...
        
        shard_dir.mkdir(exist_ok=True)
        try:
            worker_loop_fn_SOTA(0, cfg, str(shard_dir), samples_per_worker, str(summary_file))
        except Exception:
            logger.exception("Single-threaded generation failed")
            return

    logger.info(f"All worker shards written to subdirectories in: {shards_base_dir}")

    # Call the new merge function
    final_dataset_name = f"expert_{run_name}_{total_samples}_samples"
    merged_count = merge_sota_shards(shard_dirs_to_merge, out_dir, run_name, total_samples)
    
    # --- END OF PATCH 3 ---

    logger.info(f"Merged episodes count: {merged_count}")
    logger.info("Dataset generation complete.")

if __name__ == "__main__":
    main()