#!/usr/bin/env python3
"""
generate_dataset.py - Robust LMDB-sharded dataset generator.

Key features:
 - Per-worker LMDB shard files (workers stream episodes to disk; we never send big objects via pipes)
 - Single-process mode also supported
 - Resume support (will not overwrite existing shard files unless forced)
 - Deterministic per-worker seeding
 - Replay validation optional (runs post-merge, using replay_validate_episode)
 - Lightweight per-worker JSON status summary written on completion
 - Strong logging and error handling (Windows-friendly)
 Usage:
    python generate_dataset.py --config configs/gen_dataset.yaml
    python scripts/generate_dataset.py --config configs/gen_dataset_config.yaml
    python -m scripts.generate_dataset --config configs/gen_dataset_config.yaml
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
from typing import List, Dict, Any, Optional, Tuple
import multiprocessing as mp
from utils.scripted_expert import ExpertConfig

import numpy as np
from tqdm import tqdm

# project imports (assume project root is parent of scripts/)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.expert_dataset import ExpertDataset, replay_validate_episode

# Optional import: lmdb. If not present we fallback to per-worker pickle shards (less ideal).
try:
    import lmdb
except Exception:
    lmdb = None

logger = logging.getLogger("generate_dataset")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def compute_file_hash(filepath: str, algo: str = "sha1") -> Optional[str]:
    p = Path(filepath)
    if not p.exists():
        return None
    h = hashlib.new(algo)
    with open(p, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------
# LMDB helper for streaming writes
# ---------------------------
def open_lmdb_writer(shard_dir: Path, map_size: int = 12 * 1024**3, subdir: bool = True):
    """
    Opens an LMDB environment for writing into a directory (shard_dir).
    Returns the env and a write transaction helper function: put_episode(txn, key_idx, episode_bytes).
    """
    # Ensure directory exists (LMDB expects a directory when subdir=True).
    shard_dir.mkdir(parents=True, exist_ok=True)

    if lmdb is None:
        raise RuntimeError("lmdb not available; please install python-lmdb for shard writing.")

    env = lmdb.open(str(shard_dir), map_size=map_size, subdir=subdir, readonly=False, lock=True)
    return env


def worker_stream_write_lmdb(env, base_key_idx: int, episodes_iter):
    """
    Stream writes episodes_iter (an iterable of episode dicts) into env starting at base_key_idx.
    We expect episodes_iter yields serializable (pickle) episodes; we will pickle them here.
    Returns final key index (next free index) and count written.
    """
    import pickle

    idx = int(base_key_idx)
    count = 0
    try:
        with env.begin(write=True) as txn:
            for ep in episodes_iter:
                key = f"{idx:08d}".encode("ascii")
                val = pickle.dumps(ep, protocol=pickle.HIGHEST_PROTOCOL)
                txn.put(key, val)
                idx += 1
                count += 1
                # Keep transaction small enough: commit periodically if many writes
                if (count % 256) == 0:
                    txn.commit()
                    txn = env.begin(write=True)
            # final commit is done by context manager exit
    except Exception:
        # Try to close env cleanly
        try:
            env.close()
        except Exception:
            pass
        raise
    return idx, count


# ---------------------------
# Worker function (runs in separate process)
# ---------------------------
def worker_loop_fn(
    worker_id: int,
    cfg: Dict[str, Any],
    shard_dir: str,
    shard_map_size: int,
    samples_per_worker: int,
    summary_path: str,
):
    """
    Worker entrypoint.
    Streams episodes produced by ExpertDataset into an LMDB shard (shard_dir).
    Writes a small JSON summary to summary_path on completion (status, counts, error).
    """
    try:
        log_prefix = f"[worker {worker_id}]"
        logging.info(f"{log_prefix} starting. seed base={cfg.get('seed',0)} target={samples_per_worker} shard={shard_dir}")

        # Build dataset instance for this worker
        expert_cfg = cfg.get("expert_config", {})
        base_seed = int(cfg.get("seed", 0)) if cfg.get("seed") is not None else int(time.time())
        seed_for_worker = base_seed + worker_id * cfg.get("worker_seed_offset", 10000) + cfg.get("shard_idx", 0) * cfg.get("shard_seed_offset", 1000000)
        expert_config_dict = cfg.get("expert_config", {})
        expert_config_instance = ExpertConfig(**expert_config_dict)

        ds = ExpertDataset(
            urdf_path=cfg["urdf_path"],
            env_xml_path=cfg.get("xml_path"),
            base_seed=seed_for_worker,
            max_samples_per_epoch=samples_per_worker,
            skip_on_error=cfg.get("skip_on_error", True),
            
            # 3. Pass the INSTANCE, not the dictionary.
            scripted_cfg=expert_config_instance,
            
            object_size=tuple(np.array(cfg.get("object_size", [0.04,0.04,0.04])).tolist()),
            object_grasp_width=float(cfg.get("grasp_width", 0.6)),
            action_scaling_factor=float(cfg.get("action_scaling_factor", 0.5)),
            warmup=bool(cfg.get("warmup", True)),
            yield_full_obs=True,
        )

        # open lmdb env for writing
        shard_path = Path(shard_dir)
        if lmdb is None:
            # fallback: write per-episode pickles (one file), but this is less ideal.
            raise RuntimeError("LMDB is required for worker streaming. Install lmdb.")
        env = open_lmdb_writer(shard_path, map_size=shard_map_size, subdir=True)

        # We'll stream episodes as they appear: ds yields observation/action pairs *per sample*
        # But episodes appear in ds.episodes only after a full episode is generated.
        # So we iterate over the dataset to trigger episode collection, and whenever ds.episodes
        # has new episodes, we write them and clear ds.episodes to bound memory usage.
        written = 0
        key_idx = 0
        last_saved_count = 0

        # Provide a safety generator: we will periodically flush episodes if ds.episodes grows.
        # iterate dataset (it yields individual samples) but episodes are stored to ds.episodes list.
        # We monitor ds.episodes and write any new episodes to LMDB immediately.
        pbar = tqdm(total=samples_per_worker, desc=f"Worker {worker_id}", leave=False)
        samples_seen = 0

        for _ in ds:
            samples_seen += 1
            pbar.update(1)
            # whenever an episode is added to ds.episodes, ds._episode_id_counter increments.
            # We'll detect extra episodes beyond last_saved_count
            if len(ds.episodes) > last_saved_count:
                # new episodes to flush
                new_eps = ds.episodes[last_saved_count:]
                # write new_eps to LMDB
                key_idx, wrote = worker_stream_write_lmdb(env, key_idx, new_eps)
                written += wrote
                last_saved_count += wrote
                # To limit memory, zero-out the flushed episodes indexes
                # Keep only episodes that are not yet flushed (none)
                # We can clear the entire list safely if all were flushed
                # but to be safe, keep only any remaining (should be none)
                ds.episodes = ds.episodes[last_saved_count:]
        pbar.close()

        # After iteration, there may be remaining episodes to flush (rare)
        if len(ds.episodes) > 0:
            key_idx, wrote = worker_stream_write_lmdb(env, key_idx, ds.episodes)
            written += wrote

        env.sync()
        env.close()

        summary = {
            "worker_id": worker_id,
            "status": "ok",
            "written_episodes": written,
            "samples_seen": samples_seen,
            "shard_path": str(shard_path),
            "seed_used": int(seed_for_worker),
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logging.info(f"{log_prefix} finished. wrote {written} episodes to {shard_path}")

    except Exception as e:
        logging.exception(f"[worker {worker_id}] failed: {e}")
        summary = {
            "worker_id": worker_id,
            "status": "error",
            "error": str(e),
        }
        try:
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
        except Exception:
            pass
        # Reraise to make sure the process exits non-zero for debugging
        raise


# ---------------------------
# Merge shards into a final dataset (single LMDB)
# ---------------------------
def merge_shards_to_lmdb(shard_dirs: List[Path], out_dir: Path, map_size: int = 16 * 1024**3):
    """
    Read each shard LMDB and merge its episodes into a single LMDB in out_dir.
    Keys are reindexed sequentially.
    """
    import pickle
    if lmdb is None:
        raise RuntimeError("lmdb not available; merging requires python-lmdb.")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_env = lmdb.open(str(out_dir), map_size=map_size, subdir=True, readonly=False, lock=True)
    nxt = 0
    try:
        with out_env.begin(write=True) as out_txn:
            for sd in shard_dirs:
                # open shard in readonly mode
                env = lmdb.open(str(sd), subdir=True, readonly=True, lock=False)
                with env.begin() as txn:
                    cursor = txn.cursor()
                    for k, v in cursor:
                        key = f"{nxt:08d}".encode("ascii")
                        out_txn.put(key, v)
                        nxt += 1
                        # commit periodically to limit transaction size
                        if (nxt % 512) == 0:
                            out_txn.commit()
                            out_txn = out_env.begin(write=True)
                env.close()
            # final commit by context manager
    finally:
        out_env.sync()
        out_env.close()
    return nxt


# ---------------------------
# Replay filter helper (unchanged semantics)
# ---------------------------
def replay_filter(episodes_shard_dirs: List[Path], cfg: Dict[str, Any], tmp_extract_limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Optionally replay-validate each episode by streaming episodes from shards.
    Returns a list of validated episodes (as Python dicts) — note this may be large.
    Use with caution; better to do filtering during writing if you need to save memory.
    """
    import pickle
    validated = []
    tol = cfg.get("replay_tol", 0.03)
    for sd in episodes_shard_dirs:
        if lmdb is None:
            continue
        env = lmdb.open(str(sd), subdir=True, readonly=True, lock=False)
        with env.begin() as txn:
            cursor = txn.cursor()
            for k, v in cursor:
                ep = pickle.loads(v)
                ok = replay_validate_episode(ep, cfg["urdf_path"], cfg.get("xml_path", None))
                if ok:
                    validated.append(ep)
                else:
                    logger.warning(f"Dropping episode id={ep.get('episode_id')} from shard={sd} due to replay mismatch")
                if tmp_extract_limit is not None and len(validated) >= tmp_extract_limit:
                    break
            env.close()
    return validated


# ---------------------------
# CLI main
# ---------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Robust dataset generation (LMDB-sharded)")
    parser.add_argument("--config", required=True, help="YAML config")
    parser.add_argument("--out_dir", default=None, help="Final output directory (overrides config)")
    parser.add_argument("--resume", action="store_true", help="Resume mode (do not clobber existing shards)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # metadata
    metadata = {
        "config": cfg,
        "urdf_hash": compute_file_hash(cfg.get("urdf_path", "")),
        "xml_hash": compute_file_hash(cfg.get("xml_path", "")),
        "timestamp_start": time.time(),
    }

    num_workers = int(cfg.get("num_workers", 0))
    total_samples = int(cfg["num_samples"])
    shard_idx = int(cfg.get("shard_idx", 0))
    num_shards = int(cfg.get("num_shards", 1))
    base_seed = int(cfg.get("seed", 0)) if cfg.get("seed") is not None else 0

    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    # Per-worker samples (ceil division)
    if num_workers > 0:
        samples_per_worker = (total_samples + num_workers - 1) // num_workers
    else:
        samples_per_worker = total_samples

    # LMDB map sizes (configurable)
    shard_map_size = int(cfg.get("shard_map_size_bytes", 12 * 1024**3))
    final_map_size = int(cfg.get("final_map_size_bytes", 12 * 1024**3))

    # For Windows and safety, prefer 'spawn'
    ctx = mp.get_context("spawn")

    worker_processes = []
    summary_paths = []
    shard_paths = []

    if num_workers > 0:
        # Launch worker processes (each writes its own shard directory)
        for w in range(num_workers):
            shard_subdir = shards_dir / f"shard_w{w}"
            summary_path = shards_dir / f"summary_w{w}.json"
            # Avoid clobbering existing shards unless not resume
            if args.resume and shard_subdir.exists():
                logger.info(f"Shard {shard_subdir} exists and resume=True — skipping worker {w}")
                summary_paths.append(str(summary_path))
                shard_paths.append(shard_subdir)
                continue

            p = ctx.Process(
                target=worker_loop_fn,
                args=(w, cfg, str(shard_subdir), shard_map_size, samples_per_worker, str(summary_path)),
                daemon=False
            )
            p.start()
            worker_processes.append((p, shard_subdir, summary_path))
            summary_paths.append(str(summary_path))
            shard_paths.append(shard_subdir)

        # Wait for processes to finish
        for p, shard_subdir, summary_path in worker_processes:
            p.join()

        # Check summaries
        total_written = 0
        for _, shard_subdir, summary_path in worker_processes:
            try:
                if Path(summary_path).exists():
                    s = json.load(open(summary_path, "r"))
                    if s.get("status") == "ok":
                        total_written += int(s.get("written_episodes", 0))
                    else:
                        logger.warning(f"Worker summary error: {s}")
                else:
                    logger.warning(f"No summary for worker shard {shard_subdir}")
            except Exception as e:
                logger.exception(f"Error reading summary {summary_path}: {e}")

    else:
        # Single process mode: run worker_loop_fn inline to avoid spawn overhead
        shard_subdir = shards_dir / "shard_single"
        summary_path = shards_dir / "summary_single.json"
        if args.resume and shard_subdir.exists():
            logger.info(f"Shard {shard_subdir} exists and resume=True — using existing shard")
            shard_paths.append(shard_subdir)
        else:
            # call inline
            try:
                worker_loop_fn(0, cfg, str(shard_subdir), shard_map_size, samples_per_worker, str(summary_path))
            except Exception as e:
                logger.exception("Single-threaded generation failed")
                return
            shard_paths.append(shard_subdir)

    logger.info(f"All worker shards written (shard dir: {shards_dir})")

    # Merge shards into final LMDB
    final_dir = out_dir / "dataset.lmdb"
    if args.resume and final_dir.exists():
        logger.info(f"Final dataset exists and resume=True: skipping merge (use --resume=False to overwrite)")
    else:
        logger.info(f"Merging {len(shard_paths)} shards into final LMDB at {final_dir}")
        merged_count = merge_shards_to_lmdb([p for p in shard_paths if p.exists()], final_dir, map_size=final_map_size)
        logger.info(f"Merged episodes count: {merged_count}")

    # (Optional) Replay validation on merged dataset (expensive)
    if cfg.get("do_replay_validate", True):
        logger.info("Running replay validation on merged dataset (this is optional and may take time).")
        # We'll stream episodes from final_dir and validate; do not load all episodes to memory.
        env = lmdb.open(str(final_dir), subdir=True, readonly=True, lock=False)
        import pickle
        ok_count = 0
        bad_count = 0
        with env.begin() as txn:
            cursor = txn.cursor()
            for k, v in tqdm(cursor, desc="replay-validate"):
                ep = pickle.loads(v)
                ok = replay_validate_episode(ep, cfg["urdf_path"], cfg.get("xml_path", None))
                if ok:
                    ok_count += 1
                else:
                    bad_count += 1
        env.close()
        logger.info(f"Replay validation done: ok={ok_count} bad={bad_count}")

    # Save metadata file
    metadata.update({
        "timestamp_end": time.time(),
        "num_workers": num_workers,
        "total_samples_target": total_samples,
        "shards": [str(p) for p in shard_paths],
    })
    with open(out_dir / "gen_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Dataset generation complete.")


if __name__ == "__main__":
    main()
