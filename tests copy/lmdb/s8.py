# FILE: transform_legacy_dataset.py
# (State-of-the-Art, Multiprocess, Memory-Managed Legacy-to-SOTA Dataset Converter)


import argparse
import logging
import pickle
import time
import json
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

# local project utils (must exist in PYTHONPATH)
from utils.expert_dataset import ExpertDatasetWriter
from utils.lmdb_utils import open_lmdb_env, close_lmdb_env

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("transform_legacy_dataset")


def read_legacy_episode_chunk(keys_chunk: List[bytes], db_path: str) -> List[Dict[str, Any]]:
    """
    Worker function to read and unpickle a list of keys from the legacy LMDB.

    Each worker opens its own read-only LMDB environment (safe for multiprocessing).
    Returns a list of episode dicts (the same structure as in the legacy dataset).
    """
    episodes = []
    env = None
    try:
        env = open_lmdb_env(db_path, readonly=True, lock=False, readahead=False, subdir=False)
        with env.begin(write=False) as txn:
            for key in keys_chunk:
                try:
                    blob = txn.get(key)
                    if not blob:
                        logger.debug("Missing blob for key: %s", key)
                        continue
                    # The legacy DB stored pickled episode dictionaries
                    ep = pickle.loads(blob)
                    episodes.append(ep)
                except pickle.UnpicklingError:
                    logger.warning("Could not unpickle data for key: %s (skipping)", key)
                except Exception as e:
                    logger.error("Error reading key %s: %s", key, e)
    except Exception as e:
        logger.exception("Worker failed to open or read LMDB: %s", e)
    finally:
        if env:
            try:
                close_lmdb_env(env)
            except Exception:
                logger.debug("Error closing LMDB env in worker", exc_info=True)
    return episodes


def _discover_existing_index(output_dir: Path):
    """
    Look for existing index JSON files created by previous runs.
    Returns the most recently modified index file path or None.
    """
    idx_files = list(output_dir.glob("*_index.json"))
    if not idx_files:
        return None
    # pick the most recently modified index file
    idx_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return idx_files[0]


def _init_writer_episode_counter_from_index(writer: ExpertDatasetWriter, index_path: Path):
    """
    Inspect an existing index JSON and set writer._episode_id_counter to
    (max_existing_index + 1) so new episodes append safely.
    If the index file format differs, we fallback to leaving the counter unchanged.
    """
    try:
        with open(index_path, "r") as f:
            idx = json.load(f)
        episodes = idx.get("episodes") if isinstance(idx, dict) else None
        if not episodes:
            logger.warning("Index file %s contains no 'episodes' list; cannot infer counter.", index_path)
            return
        # episodes expected to be list of meta objects containing "episode_id" like "ep_000123"
        max_idx = -1
        for meta in episodes:
            epid = meta.get("episode_id") or meta.get("id") or ""
            if isinstance(epid, str) and epid.startswith("ep_"):
                try:
                    n = int(epid.split("_")[1])
                    if n > max_idx:
                        max_idx = n
                except Exception:
                    continue
        if max_idx >= 0:
            next_idx = max_idx + 1
            logger.info("Initializing writer._episode_id_counter = %d (based on index %s)", next_idx, index_path.name)
            try:
                setattr(writer, "_episode_id_counter", next_idx)
            except Exception:
                logger.warning("Writer does not support setting _episode_id_counter; continuing without init.")
    except Exception as e:
        logger.exception("Failed to read/parse existing index file %s: %s", index_path, e)


def main(args):
    legacy_db_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legacy_db_path.exists():
        logger.error("Input legacy LMDB not found: %s", legacy_db_path)
        return

    logger.info("Legacy DB: %s", legacy_db_path)
    logger.info("Output dir: %s", output_dir)

    # --- scan legacy database keys ---
    logger.info("Scanning legacy database for keys...")
    env_legacy = None
    all_keys = []
    try:
        env_legacy = open_lmdb_env(str(legacy_db_path), readonly=True, lock=False, subdir=False)
        with env_legacy.begin(write=False) as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                all_keys.append(key)
    except Exception as e:
        logger.exception("Failed to scan legacy DB: %s", e)
        return
    finally:
        if env_legacy:
            try:
                close_lmdb_env(env_legacy)
            except Exception:
                logger.debug("Error closing legacy env", exc_info=True)

    if args.limit:
        all_keys = all_keys[: args.limit]
        logger.info("Processing limited subset: %d keys", len(all_keys))

    if not all_keys:
        logger.error("No keys found in legacy DB.")
        return

    total_episodes = len(all_keys)
    logger.info("Found %d keys to process.", total_episodes)

    # --- initialize writer ---
    run_name = args.run_name or legacy_db_path.stem.replace("expert_", "")
    writer = ExpertDatasetWriter(
        out_dir=str(output_dir),
        run_name=run_name,
        image_compression="jpeg",
        jpeg_quality=args.jpeg_quality
    )
    logger.info("Initialized ExpertDatasetWriter (run_name=%s)", run_name)

    # If append requested, try to discover existing index and initialize writer counter
    if args.append:
        idx_path = _discover_existing_index(output_dir)
        if idx_path:
            logger.info("Append requested; discovered existing index: %s", idx_path.name)
            _init_writer_episode_counter_from_index(writer, idx_path)
        else:
            logger.info("Append requested but no existing index file found; starting fresh.")

    # --- process in batches using a process pool ---
    batch_size = args.batch_size
    num_workers = max(1, args.num_workers)
    logger.info("Starting transformation: batch_size=%d num_workers=%d", batch_size, num_workers)

    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # iterate over batches
            for batch_idx, start in enumerate(range(0, total_episodes, batch_size), start=1):
                end = min(start + batch_size, total_episodes)
                batch_keys = all_keys[start:end]
                logger.info("Batch %d: keys %d..%d (count=%d)", batch_idx, start, end - 1, len(batch_keys))

                # split batch among workers in balanced fashion
                # make sure chunks are contiguous-ish to improve IO locality
                chunk_size = max(1, (len(batch_keys) + num_workers - 1) // num_workers)
                key_chunks = [batch_keys[i : i + chunk_size] for i in range(0, len(batch_keys), chunk_size)]

                # submit worker jobs
                futures = [executor.submit(read_legacy_episode_chunk, chunk, str(legacy_db_path)) for chunk in key_chunks]

                batch_episodes: List[Dict[str, Any]] = []
                # collect results as they finish
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"  - Reading batch {batch_idx}", leave=False):
                    try:
                        res = fut.result()
                        if res:
                            batch_episodes.extend(res)
                    except Exception as e:
                        logger.exception("A worker failed while processing a chunk: %s", e)

                logger.info("Batch %d: gathered %d episodes (will save now)", batch_idx, len(batch_episodes))

                if not batch_episodes:
                    logger.warning("Batch %d produced no episodes; skipping save.", batch_idx)
                    continue

                # Save this batch with the writer's streaming API
                save_start = time.time()
                try:
                    writer.save_batch(batch_episodes)
                    save_dur = time.time() - save_start
                    logger.info("Batch %d saved: %d episodes (%.2fs)", batch_idx, len(batch_episodes), save_dur)
                except Exception as e:
                    logger.exception("Failed to save batch %d: %s", batch_idx, e)
                    # depending on policy, we can either abort or continue; here we abort to avoid data inconsistency
                    logger.error("Aborting transformation due to save failure on batch %d.", batch_idx)
                    return

    except KeyboardInterrupt:
        logger.warning("Interrupted by user (KeyboardInterrupt). Exiting early.")
        return
    except Exception as e:
        logger.exception("Unexpected error during transformation: %s", e)
        return

    logger.info("All batches processed and saved.")
    logger.info("Transformation finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transform a legacy expert dataset to the new SOTA format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_path", type=str, help="Path to legacy LMDB file (pickled episodes).")
    parser.add_argument("output_dir", type=str, help="Output directory for converted dataset.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name for output files.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker processes for reading.")
    parser.add_argument("--batch-size", type=int, default=256, help="Episodes per overall batch saved in one save_batch() call.")
    parser.add_argument("--jpeg-quality", type=int, default=90, help="JPEG quality (1-100) for compressed images.")
    parser.add_argument("--append", action="store_true", help="If set, attempt to append to any existing converted dataset in output_dir.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of episodes processed (for tests).")
    args = parser.parse_args()
    main(args)


