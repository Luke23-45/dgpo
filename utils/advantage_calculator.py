# FILE: utils/advantage_calculator.py
# (Definitive, SOTA, Production-Grade Implementation)

"""
Advantage Calculator for Advantage-Weighted Regression (AWR).

This script performs offline pre-processing of an existing expert dataset to
calculate and inject 'advantage' values for every timestep.

Pipeline:
1.  **Safe Replication**: Clones the source dataset to a destination to prevent data corruption.
2.  **Reward Re-evaluation**: Uses the SOTA `reward_functions.py` to re-compute rewards
    based on the physical state in the dataset. This allows for reward shaping iteration
    without re-running expensive simulations.
3.  **Return Calculation**: Computes Discounted Returns (G_t) using a backwards pass.
4.  **Baseline Estimation**: Computes a global value baseline (V) to center the returns.
5.  **Advantage Injection**: Calculates Advantage (A_t = G_t - V) and efficiently writes
    it back to the LMDB store as a new modality `advantages`.

Usage:
    python -m utils.advantage_calculator \
        --source-db path/to/expert.lmdb \
        --dest-db path/to/enhanced.lmdb \
        --gamma 0.99
"""

#python -m utils.advantage_calculator --source-db "C:\Users\Hellx\Documents\Programming\python\Project\redhot\data\training\expert_expert_run_validation_dataset_6_episodes\expert_expert_run_validation_dataset_6_episodes.lmdb" --dest-db "C:\Users\Hellx\Documents\Programming\python\Project\redhot\data\training_ready\expert_expert_run_validation_dataset_6_episodes\expert_expert_run_validation_dataset_6_episodes.lmdb" --gamma 0.99

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import lmdb
import numpy as np
from tqdm import tqdm

# Ensure project root is in path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.expert_dataset import ExpertTrajectoryDataset
from utils.reward_functions import calculate_rewards_for_episode, RewardConfig

# Configure SOTA logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AdvantageCalculator")


def compute_discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """
    Computes the discounted return G_t for each timestep t.
    G_t = r_t + gamma * G_{t+1}
    
    Args:
        rewards: Array of rewards [r_0, ..., r_T].
        gamma: Discount factor.

    Returns:
        Array of discounted returns [G_0, ..., G_T].
    """
    T = len(rewards)
    returns = np.zeros_like(rewards, dtype=np.float32)
    running_return = 0.0
    
    # Iterate backwards
    for t in reversed(range(T)):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
        
    return returns


def load_episode_data_for_rewards(reader: ExpertTrajectoryDataset, ep_idx: int) -> List[Dict[str, Any]]:
    """
    Efficiently extracts only the modalities required for reward calculation
    from the dataset reader and structures them into a list of observation dicts.
    
    This avoids loading heavy image data, ensuring high throughput.
    """
    ep_meta = reader.episode_metadata[ep_idx]
    length = ep_meta['length']
    
    # Helper to get full array from Reader's LRU cache / LMDB loader
    def get_mod(name):
        if name not in ep_meta['modalities']:
            # Graceful fallback for optional keys if reward function is robust
            return None
        meta = ep_meta['modalities'][name]
        return reader._get_full_modality_array(
            meta['key'], meta['compression'], meta['dtype'], tuple(meta['shape'])
        )

    # Load physical states (fast, low memory)
    ee_poses = get_mod('ee_pose_world')
    obj_pos = get_mod('object_pos_world')
    goal_pos = get_mod('goal_pos_world')
    is_grasped = get_mod('is_grasped')
    proprio = get_mod('proprio')

    episode_obs_list = []
    for t in range(length):
        obs = {
            'ee_pose_world': ee_poses[t] if ee_poses is not None else None,
            'object_pos_world': obj_pos[t] if obj_pos is not None else None,
            'goal_pos_world': goal_pos[t] if goal_pos is not None else None,
            'is_grasped': is_grasped[t] if is_grasped is not None else None,
            'proprio': proprio[t] if proprio is not None else None,
        }
        episode_obs_list.append(obs)
        
    return episode_obs_list


def main():
    parser = argparse.ArgumentParser(description="SOTA Advantage Calculator for AWR")
    parser.add_argument("--source-db", type=str, required=True, help="Path to input .lmdb file")
    parser.add_argument("--dest-db", type=str, required=True, help="Path to output .lmdb file")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for returns")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination if exists")
    args = parser.parse_args()

    source_path = Path(args.source_db)
    dest_path = Path(args.dest_db)
    
    # --- 1. Validation & Replication ---
    if not source_path.exists():
        raise FileNotFoundError(f"Source DB not found: {source_path}")
    
    source_index_path = source_path.parent / f"{source_path.stem}_index.json"
    if not source_index_path.exists():
        raise FileNotFoundError(f"Source Index not found: {source_index_path}")

    if dest_path.exists():
        if args.overwrite:
            logger.warning(f"Destination {dest_path} exists. Overwriting...")
            if dest_path.is_dir():
                shutil.rmtree(dest_path)
            else:
                dest_path.unlink()
            # Also clean up index
            dest_index_pre = dest_path.parent / f"{dest_path.stem}_index.json"
            if dest_index_pre.exists():
                dest_index_pre.unlink()
        else:
            raise FileExistsError(f"Destination {dest_path} exists. Use --overwrite.")

    # Ensure parent dir
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Cloning dataset: {source_path} -> {dest_path}")
    shutil.copy(source_path, dest_path)
    
    dest_index_path = dest_path.parent / f"{dest_path.stem}_index.json"
    shutil.copy(source_index_path, dest_index_path)
    
    # --- 2. Initialization ---
    # We use the ExpertTrajectoryDataset to read the *Destination* copy.
    # This ensures we are working on the file we will eventually modify.
    # Note: The Reader opens LMDB in Read-Only mode. We will open a separate Write handle later.
    logger.info("Initializing Dataset Reader for computation...")
    reader = ExpertTrajectoryDataset(
        demo_path=str(dest_path),
        observation_horizon=1, # Horizon doesn't matter for full-sequence reading
        action_horizon=1
    )
    
    num_episodes = reader.get_num_episodes()
    logger.info(f"Processing {num_episodes} episodes.")
    
    # Helper config for rewards
    reward_config = RewardConfig()
    
    # Storage for Pass 2
    episode_returns_cache = {} # ep_idx -> np.ndarray
    global_return_sum = 0.0
    global_return_count = 0

    # --- 3. Pass 1: Calculate Returns & Global Stats ---
    logger.info("--- Pass 1: Calculating Returns & Global Baseline ---")
    
    for ep_idx in tqdm(range(num_episodes), desc="Calculating Returns"):
        # A. Reconstruct Episode Context
        obs_list = load_episode_data_for_rewards(reader, ep_idx)
        
        # B. Calculate Rewards using SOTA Heuristic
        rewards = calculate_rewards_for_episode(obs_list, config=reward_config)
        
        # C. Calculate Discounted Returns
        returns = compute_discounted_returns(rewards, args.gamma)
        
        # D. Cache and Accumulate Stats
        episode_returns_cache[ep_idx] = returns
        global_return_sum += np.sum(returns)
        global_return_count += len(returns)

    # Calculate Global Baseline (V)
    if global_return_count == 0:
        raise ValueError("Dataset appears empty or invalid (0 timesteps).")
        
    global_baseline = global_return_sum / global_return_count
    logger.info(f"Global Value Baseline (V): {global_baseline:.4f}")
    
    # Close the reader's LMDB handle to free resources/locks before writing
    del reader

    # --- 4. Pass 2: Calculate Advantage & Write to LMDB ---
    logger.info("--- Pass 2: Injecting Advantages into LMDB ---")
    
    # Load JSON index to update metadata
    with open(dest_index_path, 'r') as f:
        index_data = json.load(f)
        
    # Open LMDB for writing (map_size set generously for additions)
    # Standard 1TB map size for safety, actual file size grows as needed.
    env = lmdb.open(str(dest_path), map_size=int(32 * 1024**3), subdir=False, readonly=False, lock=True)
    
    advantages_stats = []
    
    try:
        with env.begin(write=True) as txn:
            for ep_idx in tqdm(range(num_episodes), desc="Writing Advantages"):
                # Get cached returns
                G_t = episode_returns_cache[ep_idx]
                
                # A. Calculate Advantage
                # A_t = G_t - V
                # Note: We cast to float32 for storage efficiency/DL standard
                advantages = (G_t - global_baseline).astype(np.float32)
                
                # Stats for logging
                advantages_stats.append(advantages)
                
                # B. Prepare Metadata
                ep_meta = index_data['episodes'][ep_idx]
                ep_id = ep_meta['episode_id'] # e.g. "ep_000000"
                
                key_name = f"{ep_id}_advantages"
                
                # C. Write Data (Raw Binary)
                txn.put(key_name.encode('ascii'), advantages.tobytes())
                
                # D. Update Index Metadata
                # Follows the SOTA "Struct of Arrays" schema
                ep_meta['modalities']['advantages'] = {
                    "key": key_name,
                    "compression": "raw", # No compression for 1D float arrays
                    "dtype": "float32",
                    "shape": list(advantages.shape)
                }
                
        logger.info("LMDB Write Committed.")
        
    finally:
        env.close()
        
    # --- 5. Finalize & Save Index ---
    logger.info(f"Updating Index JSON at {dest_index_path}...")
    with open(dest_index_path, 'w') as f:
        json.dump(index_data, f, indent=None) # Minimal JSON size
        
    # --- 6. Report Distribution Stats ---
    all_advs = np.concatenate(advantages_stats)
    logger.info(f"Processing Complete.")
    logger.info(f"Advantage Stats | Mean: {np.mean(all_advs):.4f} | Std: {np.std(all_advs):.4f}")
    logger.info(f"                | Min:  {np.min(all_advs):.4f} | Max: {np.max(all_advs):.4f}")
    logger.info(f"Output saved to: {dest_path}")

if __name__ == "__main__":
    main()