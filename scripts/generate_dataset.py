# FILE: generate_dataset.py
# (State-of-the-Art, Hydra-Configurable, Resumable, Validated, Sharded LMDB Generator)

"""
State-of-the-art script for generating large-scale, high-quality expert demonstration
datasets for robotic manipulation tasks using multiprocessing, LMDB, and Hydra.

This version is engineered for maximum robustness, fault tolerance, and data quality.

Key SOTA Features:
  - **Hydra Configuration**: Uses Hydra for standardized and flexible configuration
    management, consistent with the pretraining and RL scripts[cite: 747, 804].
  - **Sharded Parallelism**: Distributes data generation across multiple worker
    processes, each writing to an isolated shard directory to prevent conflicts.
  - **Granular Checkpointing & Resumability**: Workers periodically save their state
    (episodes saved, seeds used, LMDB index). The script can be interrupted and
    resumed, continuing generation exactly where each worker left off, minimizing
    lost work due to crashes or interruptions.
  - **Integrated Data Validation**: Ensures data quality by:
    1. Checking the expert's own success flag (`ScriptedExpert.was_successful()`).
    2. Performing physics-based replay validation (`replay_validate_episode`)
       to verify trajectory plausibility .
    Only episodes passing *both* checks are saved.
  - **Atomic File Operations**: Uses atomic writes (write-to-temp then rename) for
    checkpoints and summaries to prevent corruption if interrupted during saving.
  - **Comprehensive Metadata**: Saves the full Hydra config, run details, worker
    statistics (attempted, validated, saved episodes), and validation settings.
  - **Efficient LMDB Storage**: Leverages LMDB for fast read access during training,
    writing episodes in batches for performance [cite: 362-366].
  - **Robust Error Handling**: Workers log errors and report failure status, allowing
    the main process to identify and report issues.
"""

from __future__ import annotations
import os
import sys
import time
import yaml # Still needed if loading non-hydra configs within ExpertDataset maybe
import logging
import hashlib
import json
from pathlib import Path
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import pickle
import random
import shutil # For atomic rename
from typing import Optional, Any, List, Dict, Tuple
# Third-party
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from dataclasses import dataclass
# Project imports
# Ensure ROOT points correctly relative to this script's location if moved
try:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from utils.expert_dataset import ExpertDataset, replay_validate_episode
    from utils.scripted_expert import ExpertConfig
    from utils.lmdb_utils import open_lmdb_env, close_lmdb_env # Use robust LMDB helpers
except ImportError as e:
    print(f"Error importing project modules. Ensure PYTHONPATH is set correctly or script is run from the project root. {e}")
    sys.exit(1)

try:
    import lmdb
except ImportError:
    print("Error: python-lmdb is required. Please install it: pip install lmdb")
    lmdb = None
    sys.exit(1)

# Setup logger
log = logging.getLogger(__name__) # Hydra typically configures logging handlers


# === Atomic File Operations ===

def atomic_write_json(data: dict, path: Path):
    """Writes JSON data atomically to avoid corruption."""
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(temp_path, path) # Atomic rename
    except Exception:
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise

def atomic_write_pickle(data: Any, path: Path):
    """Writes pickled data atomically."""
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(temp_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_path, path)
    except Exception:
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise

# === Worker Checkpoint Management ===

CHECKPOINT_FILENAME = "worker_checkpoint.pkl"
SUMMARY_FILENAME = "worker_summary.json"

@dataclass
class WorkerState:
    """State saved by workers for resuming."""
    worker_id: int
    episodes_generated_attempted: int = 0
    episodes_validated_saved: int = 0
    lmdb_key_index: int = 0
    last_expert_dataset_seed: Optional[int] = None
    numpy_rng_state: Optional[Any] = None
    random_rng_state: Optional[Any] = None
    # Note: torch RNG state not needed if not using torch directly in worker

def save_worker_checkpoint(state: WorkerState, shard_dir: Path):
    """Saves the worker state atomically."""
    atomic_write_pickle(state, shard_dir / CHECKPOINT_FILENAME)

def load_worker_checkpoint(shard_dir: Path) -> Optional[WorkerState]:
    """Loads worker state if checkpoint exists."""
    ckpt_path = shard_dir / CHECKPOINT_FILENAME
    if ckpt_path.exists():
        try:
            with open(ckpt_path, "rb") as f:
                state = pickle.load(f)
            if isinstance(state, WorkerState):
                log.info(f"Loaded checkpoint from {ckpt_path}")
                return state
            else:
                log.warning(f"Invalid checkpoint format found at {ckpt_path}. Ignoring.")
        except Exception as e:
            log.warning(f"Could not load checkpoint from {ckpt_path}: {e}. Ignoring.")
    return None

def save_worker_summary(summary: dict, shard_dir: Path):
    """Saves the final worker summary atomically."""
    atomic_write_json(summary, shard_dir / SUMMARY_FILENAME)

def load_worker_summary(shard_dir: Path) -> Optional[dict]:
    """Loads worker summary if it exists."""
    summary_path = shard_dir / SUMMARY_FILENAME
    if summary_path.exists():
        try:
            with open(summary_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Could not load summary from {summary_path}: {e}. Ignoring.")
    return None


# === Worker Function ===

def worker_process_fn(worker_id: int, cfg: DictConfig, shard_dir: Path, target_samples: int):
    """
    The main function executed by each worker process.
    Generates episodes, validates them, saves valid ones to a shard LMDB,
    and periodically checkpoints its state for resumability.
    """
    log_prefix = f"[Worker {worker_id}]"
    start_time = time.time()
    worker_state = WorkerState(worker_id=worker_id)
    rng = None # Initialize later based on checkpoint/seed

    try:
        log.info(f"{log_prefix} Starting. Target Samples={target_samples}. Shard Dir={shard_dir}")
        shard_dir.mkdir(parents=True, exist_ok=True)

        # --- Resume Logic ---
        loaded_checkpoint = load_worker_checkpoint(shard_dir)
        if loaded_checkpoint:
            worker_state = loaded_checkpoint
            log.info(f"{log_prefix} Resuming from checkpoint: {worker_state.episodes_validated_saved} episodes saved.")
            # Restore RNG states
            if worker_state.numpy_rng_state:
                np.random.set_state(worker_state.numpy_rng_state)
            if worker_state.random_rng_state:
                random.setstate(worker_state.random_rng_state)
            # Create RNG *after* potential state restoration
            rng = np.random.default_rng(np.random.randint(0, 2**32 -1))

        # --- Initialize RNG if not resuming ---
        if rng is None:
            base_seed = int(cfg.seed)
            worker_base_seed = base_seed + worker_id * cfg.get("worker_seed_offset", 10000)
            log.info(f"{log_prefix} Initializing with base seed {worker_base_seed}")
            np.random.seed(worker_base_seed)
            random.seed(worker_base_seed + 1)
            rng = np.random.default_rng(worker_base_seed + 2)
            # The ExpertDataset will use its own internal seeding derived from `expert_start_seed`

        # --- LMDB Initialization ---
        # Use robust helper with subdir=True (creates data.mdb and lock.mdb in shard_dir)
        lmdb_env = open_lmdb_env(
            str(shard_dir),
            map_size_gb=cfg.lmdb.shard_map_size_gb,
            readonly=False,
            lock=True,
            subdir=True # Critical for isolated shards
        )

        # --- ExpertDataset Initialization ---
        # Determine the seed for ExpertDataset
        expert_start_seed = worker_state.last_expert_dataset_seed
        if expert_start_seed is None: # If not resuming or first run
             # Calculate the initial seed deterministically
             expert_start_seed = base_seed + worker_id * cfg.get("worker_seed_offset", 10000)

        # Calculate how many more samples this worker needs to generate *from scratch*
        # Note: ExpertDataset uses max_samples_per_epoch as a *stopping condition* for generation attempts
        # We need to track validated/saved episodes separately.
        # Let ExpertDataset run until it thinks it has generated enough attempts.
        # We rely on the outer loop check `worker_state.episodes_validated_saved < target_samples`
        dataset_target_attempts = target_samples * cfg.generation.get("attempt_oversampling_factor", 1.5)
        expert_config_dict = {}
        expert_config_instance = ExpertConfig(**expert_config_dict)

        ds = ExpertDataset(
            urdf_path=cfg.env.urdf_path,
            env_xml_path=cfg.env.xml_path,
            base_seed=expert_start_seed, # Seed for the generator's internal RNG manager
            max_samples_per_epoch=int(dataset_target_attempts), # Target generation *attempts*
            skip_on_error=True,
            scripted_cfg=expert_config_instance,
            warmup=True,
            yield_full_obs=True, 
        )

        episodes_to_write_buf: List[Dict] = []
        save_interval = cfg.generation.save_interval_episodes
        checkpoint_interval = cfg.generation.checkpoint_interval_episodes

        pbar = tqdm(
            initial=worker_state.episodes_validated_saved,
            total=target_samples,
            desc=f"Worker {worker_id} (Validated)",
            leave=False,
            position=worker_id
        )

        dataset_iterator = iter(ds)

        # --- Main Generation Loop ---
        while worker_state.episodes_validated_saved < target_samples:
            try:
                # ================================================================= #
                # FIX 1 (cont.): Call next() on the PERSISTENT iterator.
                # This correctly advances the generator to produce the next sample.
                # ================================================================= #
                next(dataset_iterator)
            except StopIteration:
                log.info(f"{log_prefix} ExpertDataset iterator finished.")
                break # ExpertDataset reached its internal target

            num_generated_in_ds = len(ds.episodes)
            if num_generated_in_ds > 0: # Check if there are any episodes to process
                newly_generated_eps = ds.episodes
                
                for ep in newly_generated_eps:
                    worker_state.episodes_generated_attempted += 1
                    ep_success = ep.get("success", False)

                    # --- Validation ---
                    validation_passed = False
                    if ep_success:
                        if cfg.generation.enable_replay_validation:
                            try:
                                replay_ok = replay_validate_episode(
                                    ep,
                                    urdf_path=cfg.env.urdf_path,
                                    env_xml_path=cfg.env.xml_path
                                )
                                if replay_ok:
                                    validation_passed = True
                                else:
                                    log.debug(f"{log_prefix} Episode failed replay validation.")
                            except Exception as val_err:
                                log.warning(f"{log_prefix} Replay validation failed: {val_err}")
                        else:
                            validation_passed = True
                    
                    # --- Add to Write Buffer ---
                    if validation_passed:
                        episodes_to_write_buf.append(ep)
                        pbar.update(1)
                        if worker_state.episodes_validated_saved + len(episodes_to_write_buf) >= target_samples:
                             break

                # ================================================================= #
                # FIX 2: Explicitly clear the ExpertDataset's internal episode list
                # to prevent unbounded memory growth.
                # ================================================================= #
                ds.episodes.clear()

                # --- Periodic Saving & Checkpointing ---
                if len(episodes_to_write_buf) >= save_interval:
                    with lmdb_env.begin(write=True) as txn:
                        for ep_to_save in episodes_to_write_buf:
                            key = f"{worker_state.lmdb_key_index:08d}".encode("ascii")
                            val = pickle.dumps(ep_to_save, protocol=pickle.HIGHEST_PROTOCOL)
                            txn.put(key, val)
                            worker_state.lmdb_key_index += 1
                    
                    num_written = len(episodes_to_write_buf)
                    worker_state.episodes_validated_saved += num_written
                    episodes_to_write_buf.clear()
                    
                    if worker_state.episodes_validated_saved % checkpoint_interval < num_written:
                        worker_state.last_expert_dataset_seed = ds.get_last_seed()
                        worker_state.numpy_rng_state = np.random.get_state()
                        worker_state.random_rng_state = random.getstate()
                        save_worker_checkpoint(worker_state, shard_dir)
                        log.debug(f"{log_prefix} Saved checkpoint.")
            
            if worker_state.episodes_validated_saved >= target_samples:
                break

        # --- Final Write & Checkpoint ---
        if episodes_to_write_buf:
            log.info(f"{log_prefix} Writing final {len(episodes_to_write_buf)} episodes...")
            with lmdb_env.begin(write=True) as txn:
                for ep_to_save in episodes_to_write_buf:
                    key = f"{worker_state.lmdb_key_index:08d}".encode("ascii")
                    val = pickle.dumps(ep_to_save, protocol=pickle.HIGHEST_PROTOCOL)
                    txn.put(key, val)
                    worker_state.lmdb_key_index += 1
            worker_state.episodes_validated_saved += len(episodes_to_write_buf)
            episodes_to_write_buf.clear()

        # Update and save final state in checkpoint before closing
        worker_state.last_expert_dataset_seed = ds.get_last_seed()
        worker_state.numpy_rng_state = np.random.get_state()
        worker_state.random_rng_state = random.getstate()
        save_worker_checkpoint(worker_state, shard_dir)

        # --- Cleanup ---
        pbar.close()
        close_lmdb_env(lmdb_env) # Use robust helper

        end_time = time.time()
        summary = {
            "worker_id": worker_id,
            "status": "ok",
            "episodes_attempted": worker_state.episodes_generated_attempted,
            "episodes_validated_saved": worker_state.episodes_validated_saved,
            "lmdb_keys_written": worker_state.lmdb_key_index,
            "duration_seconds": round(end_time - start_time, 2),
            "shard_path": str(shard_dir),
            "last_seed": worker_state.last_expert_dataset_seed,
        }
        save_worker_summary(summary, shard_dir)
        log.info(f"{log_prefix} Finished successfully. Saved {worker_state.episodes_validated_saved} valid episodes.")

    except Exception as e:
        end_time = time.time()
        log.exception(f"{log_prefix} FAILED with error: {e}")
        summary = {
            "worker_id": worker_id,
            "status": "error",
            "error": str(e),
            "episodes_attempted": worker_state.episodes_generated_attempted,
            "episodes_validated_saved": worker_state.episodes_validated_saved,
            "duration_seconds": round(end_time - start_time, 2),
            "shard_path": str(shard_dir),
        }
        # Attempt to save error summary
        try:
            save_worker_summary(summary, shard_dir)
        except Exception as summary_e:
            log.error(f"{log_prefix} CRITICAL: Failed to save error summary: {summary_e}")
        # Ensure LMDB is closed if it was opened
        if 'lmdb_env' in locals() and lmdb_env is not None:
             close_lmdb_env(lmdb_env)
        raise # Re-raise exception to signal failure to the main process

# === Merge Function ===

def merge_shards_fn(shard_dirs: List[Path], out_file: Path, final_map_size_gb: float):
    """Merges LMDB shards into a single final LMDB file."""
    log.info(f"Starting merge of {len(shard_dirs)} shards into {out_file}...")
    
    # Ensure parent directory exists
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Open output LMDB (use robust helper, subdir=False for single file)
    out_env = open_lmdb_env(
        str(out_file),
        map_size_gb=final_map_size_gb,
        readonly=False,
        lock=True,
        subdir=False # Final dataset is a single file
    )
    
    total_merged_count = 0
    try:
        with out_env.begin(write=True) as out_txn:
            for shard_dir in tqdm(shard_dirs, desc="Merging Shards"):
                shard_summary = load_worker_summary(shard_dir)
                if not shard_summary or shard_summary.get("status") != "ok":
                    log.warning(f"Skipping invalid or incomplete shard: {shard_dir}")
                    continue

                log.debug(f"Merging shard: {shard_dir}")
                shard_env = open_lmdb_env(str(shard_dir), readonly=True, lock=False, subdir=True)
                shard_merged_count = 0
                try:
                    with shard_env.begin() as shard_txn:
                        cursor = shard_txn.cursor()
                        for _key, value in cursor:
                            # Generate new sequential key for the merged DB
                            merged_key = f"{total_merged_count:08d}".encode("ascii")
                            out_txn.put(merged_key, value)
                            total_merged_count += 1
                            shard_merged_count += 1
                finally:
                    close_lmdb_env(shard_env) # Close shard env
                
                # Verify count matches summary
                expected_count = shard_summary.get("lmdb_keys_written", -1)
                if shard_merged_count != expected_count:
                     log.warning(f"Shard {shard_dir.name} merge count mismatch! Expected {expected_count}, got {shard_merged_count}.")

    finally:
        # Ensure final LMDB is synced and closed
        log.info("Syncing final LMDB...")
        out_env.sync()
        close_lmdb_env(out_env)
        
    log.info(f"Merge complete. Total episodes merged: {total_merged_count}")
    return total_merged_count

# === Metadata Function ===

def save_metadata_fn(cfg: DictConfig, output_dir: Path, run_name: str, worker_summaries: List[Dict], total_merged_count: Optional[int]):
    """Saves final metadata for the dataset generation run."""
    metadata = {
        "run_name": run_name,
        "timestamp_start": time.strftime("%Y-%m-%d_%H:%M:%S"), # Approximate start
        "config": OmegaConf.to_container(cfg, resolve=True),
        "total_episodes_merged": total_merged_count,
        "workers": worker_summaries,
        # Add git hash?
    }
    metadata_path = output_dir / f"expert_{run_name}_metadata.json"
    atomic_write_json(metadata, metadata_path)
    log.info(f"Saved final metadata to {metadata_path}")


# === Main Orchestrator ===

@hydra.main(version_base=None, config_path="../configs", config_name="gen_dataset_config")
def main(cfg: DictConfig):
    """Main function orchestrated by Hydra."""
    if lmdb is None:
        log.error("python-lmdb is not installed. Cannot generate dataset. Exiting.")
        sys.exit(1)

    start_time = time.time()
    # Hydra automatically creates and manages the output directory
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    log.info(f"Starting dataset generation run. Output Dir: {output_dir}")
    log.info("----------- Configuration -----------")
    log.info(OmegaConf.to_yaml(cfg))
    log.info("------------------------------------")

    run_name = cfg.run_name if cfg.run_name else output_dir.name
    num_workers = int(cfg.generation.parallel.num_workers)
    total_samples_target = int(cfg.generation.num_samples)
    resume = cfg.resume

    shards_base_dir = output_dir / "shards"
    shards_base_dir.mkdir(parents=True, exist_ok=True)

    if num_workers <= 0:
        log.error("num_workers must be positive.")
        sys.exit(1)

    samples_per_worker = (total_samples_target + num_workers - 1) // num_workers

    ctx = mp.get_context("spawn") # Use spawn for better isolation
    processes: List[mp.Process] = []
    shard_dirs: List[Path] = []
    active_workers = 0

    log.info(f"Targeting {total_samples_target} total validated samples across {num_workers} workers (~{samples_per_worker} per worker).")

    # --- Worker Spawning & Resume Check ---
    for w_idx in range(num_workers):
        shard_dir = shards_base_dir / f"shard_w{w_idx}"
        shard_dirs.append(shard_dir)
        worker_summary = load_worker_summary(shard_dir)

        should_spawn = True
        if resume:
            if worker_summary and worker_summary.get("status") == "ok":
                # Check if target is met
                saved_count = worker_summary.get("episodes_validated_saved", 0)
                if saved_count >= samples_per_worker:
                    log.info(f"Shard {shard_dir.name} already complete ({saved_count}/{samples_per_worker} episodes). Skipping worker {w_idx}.")
                    should_spawn = False
                else:
                    log.info(f"Shard {shard_dir.name} found completed summary but needs more samples ({saved_count}/{samples_per_worker}). Will resume.")
                    # Worker will resume based on its checkpoint
            elif load_worker_checkpoint(shard_dir):
                log.info(f"Found checkpoint for worker {w_idx}. Will resume.")
            else:
                 log.info(f"No valid summary or checkpoint for worker {w_idx}. Starting from scratch.")
        else:
             # Not resuming, always start
             # Clean up previous shard if it exists? Optional, maybe safer not to.
             pass

        if should_spawn:
            log.info(f"Spawning worker {w_idx}...")
            # Pass Hydra config directly - Omegaconf handles pickling
            p = ctx.Process(
                target=worker_process_fn,
                args=(w_idx, cfg, shard_dir, samples_per_worker),
                daemon=False # Ensure cleanup happens
            )
            p.start()
            processes.append(p)
            active_workers += 1

    log.info(f"Launched {active_workers} worker processes.")

    # --- Wait for Workers ---
    successful_workers = 0
    failed_workers = 0
    for p in processes:
        p.join() # Wait for the process to finish
        if p.exitcode == 0:
            successful_workers += 1
        else:
            log.error(f"Worker process (PID {p.pid}) exited with non-zero code {p.exitcode}.")
            failed_workers += 1

    log.info(f"All worker processes finished. Success: {successful_workers}, Failed: {failed_workers}")

    # --- Collect Summaries & Check Status ---
    worker_summaries = []
    valid_shard_dirs = []
    overall_success = (failed_workers == 0)

    for shard_dir in shard_dirs:
        summary = load_worker_summary(shard_dir)
        if summary:
            worker_summaries.append(summary)
            if summary.get("status") == "ok":
                valid_shard_dirs.append(shard_dir)
            else:
                log.error(f"Worker {summary.get('worker_id', 'N/A')} reported error: {summary.get('error', 'Unknown')}")
                overall_success = False
        elif any(p.exitcode != 0 for p in processes if f"shard_w{shard_dirs.index(shard_dir)}" in p.name): # Rough check if corresponding process failed silently
             log.error(f"Worker for shard {shard_dir.name} seems to have failed without writing a summary.")
             overall_success = False


    if not overall_success:
        log.error("One or more workers failed. Merge step will be skipped. Check worker logs and summaries in the shards directory.")
        total_merged_count = None
    elif not valid_shard_dirs:
         log.warning("No valid shards were generated by any worker. Nothing to merge.")
         total_merged_count = 0
    else:
        # --- Merge Shards ---
        final_lmdb_file = output_dir / f"expert_{run_name}.lmdb"
        try:
            total_merged_count = merge_shards_fn(
                valid_shard_dirs,
                final_lmdb_file,
                cfg.lmdb.final_map_size_gb
            )
            log.info(f"Successfully merged {total_merged_count} episodes into {final_lmdb_file}")
            # Optional: Clean up shard directories after successful merge?
            if cfg.generation.cleanup_shards_after_merge:
                 log.info("Cleaning up shard directories...")
                 for shard_dir in valid_shard_dirs:
                     shutil.rmtree(shard_dir)
        except Exception as merge_err:
            log.exception(f"Failed to merge shards: {merge_err}")
            total_merged_count = None
            overall_success = False # Mark run as failed if merge fails

    # --- Save Final Metadata ---
    save_metadata_fn(cfg, output_dir, run_name, worker_summaries, total_merged_count)

    end_time = time.time()
    log.info(f"Dataset generation finished in {(end_time - start_time) / 60:.2f} minutes.")

    if not overall_success:
         log.error("Dataset generation finished with errors.")
         # sys.exit(1) # Optional: exit with error code

if __name__ == "__main__":
    # Add a check for needing ExpertDataset method
    if not hasattr(ExpertDataset, 'get_last_seed'):
         print("\nERROR: Your `ExpertDataset` class is missing the `get_last_seed` method required for checkpointing.")
         print("Please add the following method to `utils/expert_dataset.py` inside the `ExpertDataset` class:")
         print("""
    def get_last_seed(self) -> Optional[int]:
        # Returns the seed used for the *last completed or currently running* episode generation attempt.
        # Assumes _init_worker_state sets _worker_master_seed and _episode_attempt_counter
        if not hasattr(self, '_worker_master_seed') or not hasattr(self, '_episode_attempt_counter'):
             # Should not happen if worker is initialized correctly
             return None
        # The seed for the *next* episode would be master + attempts.
        # The seed for the *current or last* attempt is master + attempts - 1.
        if self._episode_attempt_counter > 0:
            return (self._worker_master_seed + self._episode_attempt_counter - 1) & 0x7FFFFFFF
        else:
             # If no attempts made yet, return the initial seed planned
             return self._worker_master_seed & 0x7FFFFFFF
""")
         sys.exit(1)

    main()


