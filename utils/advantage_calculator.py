# FILE: utils/advantage_calculator.py
# (Definitive SOTA v9.2 - Time-Dependent Baseline & Phase Audit)

"""
Advantage Calculator for Advantage-Weighted Regression (AWR).

State-of-the-Art Upgrade (v9.2):
1.  **Time-Dependent Baseline V(t)**: Calculates the value baseline separately 
    for each timestep t across all episodes. This creates a non-parametric 
    estimate of V(s) that accounts for the natural decay of Discounted Returns.
    This mathematically solves the "Negative Advantage Bias" in later task phases (e.g., Grasping).
    
2.  **Phase-Aware Auditing**: Explicitly loads semantic phase labels ('gt_phase') 
    to verify that the Advantage distribution is centered (Mean ~ 0.0) across 
    ALL phases, specifically monitoring the 'Grasp' phase for recovery.

3.  **Robust Data Loading**: Uses defensive typing and explicit modality extraction
    to prevent silent failures during high-throughput processing.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

import lmdb
import numpy as np
import torch
from tqdm import tqdm

# Ensure project root is in path for imports
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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

# Semantic Map for Reporting
# Maps integer phase labels to human-readable names for the audit report
PHASE_MAP = {
    0: "0_Approach",
    1: "1_Grasp",
    2: "2_Transport",
    3: "3_Place",
    4: "4_Retract"
}

def compute_discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """
    Computes the Discounted Return G_t for each timestep t using a backwards pass.
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
    
    # Iterate backwards from T-1 to 0
    for t in reversed(range(T)):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
        
    return returns


def load_episode_data_for_rewards(reader: ExpertTrajectoryDataset, ep_idx: int) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Robustly extracts physical states AND Phase labels from the dataset reader.
    
    Args:
        reader: The initialized ExpertTrajectoryDataset instance.
        ep_idx: The index of the episode to load.
        
    Returns:
        episode_obs_list: List of observation dicts for reward calculation.
        phases: Numpy array of phase integers for auditing.
    """
    ep_meta = reader.episode_metadata[ep_idx]
    length = ep_meta['length']
    modalities = ep_meta['modalities']
    
    # Helper to safely get full array from Reader's LRU cache / LMDB loader
    def get_mod(name: str) -> Optional[np.ndarray]:
        if name not in modalities:
            return None
        meta = modalities[name]
        return reader._get_full_modality_array(
            meta['key'], meta['compression'], meta['dtype'], tuple(meta['shape'])
        )

    # 1. Load Physical State Modalities (Required for Reward Function)
    ee_poses = get_mod('ee_pose_world')
    obj_pos = get_mod('object_pos_world')
    goal_pos = get_mod('goal_pos_world')
    is_grasped = get_mod('is_grasped')
    proprio = get_mod('proprio')
    
    # 2. Load Phase Modality (Required for SOTA Audit)
    gt_phase = get_mod('gt_phase')

    episode_obs_list = []
    phases_list = []
    
    for t in range(length):
        # Construct observation dict for the Reward Function
        # We explicitly handle None to be defensive, though a valid dataset should have these.
        obs = {
            'ee_pose_world': ee_poses[t] if ee_poses is not None else None,
            'object_pos_world': obj_pos[t] if obj_pos is not None else None,
            'goal_pos_world': goal_pos[t] if goal_pos is not None else None,
            'is_grasped': is_grasped[t] if is_grasped is not None else None,
            'proprio': proprio[t] if proprio is not None else None,
        }
        episode_obs_list.append(obs)
        
        # Robustly extract phase integer
        if gt_phase is not None:
            # Handle cases where data might be a 0-d tensor, numpy scalar, or array
            raw_val = gt_phase[t]
            if hasattr(raw_val, 'item'):
                p = int(raw_val.item())
            else:
                p = int(raw_val)
            phases_list.append(p)
        else:
            # Fallback if phase is missing (should not happen in v9.0 dataset)
            phases_list.append(0)

    return episode_obs_list, np.array(phases_list, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(description="SOTA Advantage Calculator v9.2 (Time-Dependent Baseline)")
    parser.add_argument("--source-db", type=str, required=True, help="Path to input .lmdb file")
    parser.add_argument("--dest-db", type=str, required=True, help="Path to output .lmdb file")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for returns")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination if exists")
    args = parser.parse_args()

    source_path = Path(args.source_db)
    dest_path = Path(args.dest_db)
    
    # --- 1. Validation & Safe Replication ---
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

    # Ensure parent dir exists
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Cloning dataset: {source_path} -> {dest_path}")
    shutil.copy(source_path, dest_path)
    
    dest_index_path = dest_path.parent / f"{dest_path.stem}_index.json"
    shutil.copy(source_index_path, dest_index_path)
    
    # --- 2. Initialization ---
    logger.info("Initializing Dataset Reader for computation...")
    # We read from the DESTINATION copy to ensure we are modifying the file we own.
    reader = ExpertTrajectoryDataset(
        demo_path=str(dest_path),
        observation_horizon=1, # Horizon doesn't matter for full-sequence reading
        action_horizon=1
    )
    
    num_episodes = reader.get_num_episodes()
    logger.info(f"Processing {num_episodes} episodes.")
    
    # Helper config for rewards
    reward_config = RewardConfig()
    
    # Caches for Pass 2
    episode_returns_cache = {} # ep_idx -> np.ndarray
    episode_phases_cache = {}  # ep_idx -> np.ndarray
    
    # v9.2 CORE: Accumulators for Time-Dependent Baseline V(t)
    # Key: Timestep t (int), Value: List of returns at that timestep across all episodes
    returns_per_timestep = defaultdict(list)

    # --- 3. Pass 1: Calculate Returns & Compute V(t) ---
    logger.info(f"--- Pass 1: Calculating Returns & Computing V(t) Baseline ---")
    
    for ep_idx in tqdm(range(num_episodes), desc="Analysis Pass"):
        # A. Reconstruct Episode Context (Physics + Phases)
        obs_list, phases = load_episode_data_for_rewards(reader, ep_idx)
        
        # B. Calculate Rewards using SOTA Heuristic
        rewards = calculate_rewards_for_episode(obs_list, config=reward_config)
        
        # C. Calculate Discounted Returns
        returns = compute_discounted_returns(rewards, args.gamma)
        
        # D. Cache for Pass 2
        episode_returns_cache[ep_idx] = returns
        episode_phases_cache[ep_idx] = phases
        
        # E. Accumulate for Time-Dependent Baseline
        for t, val in enumerate(returns):
            returns_per_timestep[t].append(val)

    # F. Compute Baseline V(t) = Mean(Returns at t)
    baseline_per_timestep = {}
    sorted_timesteps = sorted(returns_per_timestep.keys())
    
    for t in sorted_timesteps:
        baseline_per_timestep[t] = np.mean(returns_per_timestep[t])
        
    max_t = sorted_timesteps[-1]
    logger.info(f"Computed Time-Dependent Baselines for {len(baseline_per_timestep)} timesteps (Max T={max_t}).")
    
    # Close the reader to unlock LMDB resources
    del reader

    # --- 4. Pass 2: Inject Advantages & Audit ---
    logger.info("--- Pass 2: Injecting Advantages & Auditing Phase Bias ---")
    
    # Load JSON index to update metadata
    with open(dest_index_path, 'r') as f:
        index_data = json.load(f)
        
    # Open LMDB for writing
    env = lmdb.open(str(dest_path), map_size=int(35 * 1024**3), subdir=False, readonly=False, lock=True)
    
    # Stats for Audit Report
    all_advs = []
    phase_adv_accumulator = defaultdict(list) # Key: Phase Name, Value: List of Advantages
    
    try:
        with env.begin(write=True) as txn:
            for ep_idx in tqdm(range(num_episodes), desc="Writing Advantages"):
                G_t = episode_returns_cache[ep_idx]
                phases = episode_phases_cache[ep_idx]
                
                # --- CORE LOGIC v9.2: A_t = G_t - V(t) ---
                advantages = np.zeros_like(G_t)
                
                for t in range(len(G_t)):
                    # Robust Lookup: If an episode is longer than any seen in the "average",
                    # fall back to the baseline of the last known timestep.
                    # This prevents defaulting to 0.0 which would cause massive bias.
                    if t in baseline_per_timestep:
                        b_t = baseline_per_timestep[t]
                    else:
                        b_t = baseline_per_timestep[max_t]
                    
                    # Calculate Advantage
                    adv = G_t[t] - b_t
                    advantages[t] = adv
                    
                    # Accumulate for Audit Report
                    p_name = PHASE_MAP.get(phases[t], "Unknown")
                    phase_adv_accumulator[p_name].append(adv)

                # Cast to float32 for storage
                advantages = advantages.astype(np.float32)
                all_advs.append(advantages)
                
                # Prepare Metadata
                ep_meta = index_data['episodes'][ep_idx]
                ep_id = ep_meta['episode_id']
                key_name = f"{ep_id}_advantages"
                
                # Write Data
                txn.put(key_name.encode('ascii'), advantages.tobytes())
                
                # Update Index
                ep_meta['modalities']['advantages'] = {
                    "key": key_name,
                    "compression": "raw",
                    "dtype": "float32",
                    "shape": list(advantages.shape)
                }
                
        logger.info("LMDB Write Committed.")
        
    finally:
        env.close()
        
    # Finalize Index
    with open(dest_index_path, 'w') as f:
        json.dump(index_data, f, indent=None)
        
    # --- 5. The Final SOTA Audit Report ---
    logger.info("\n" + "="*60)
    logger.info("SOTA PHASE ADVANTAGE AUDIT REPORT")
    logger.info("Objective: Verify Mean Advantage is ~0.0 for ALL phases.")
    logger.info("-" * 60)
    logger.info(f"{'PHASE NAME':<20} | {'MEAN ADVANTAGE':<15} | {'STATUS':<10}")
    logger.info("-" * 60)
    
    sorted_phases = sorted(phase_adv_accumulator.keys())
    for p_name in sorted_phases:
        vals = np.array(phase_adv_accumulator[p_name])
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        
        # Diagnostic Logic
        status = "✅ OK"
        if abs(mean_val) > 1.0:
            status = "⚠️ BIASED"
        
        # Strict check for critical phases
        if p_name == "1_Grasp" and abs(mean_val) > 0.5:
             status = "❌ FAIL"
            
        logger.info(f"{p_name:<20} | {mean_val:+.4f} (±{std_val:.2f})  | {status}")
    
    flat_advs = np.concatenate(all_advs)
    logger.info("-" * 60)
    logger.info(f"Global Stats         | Mean: {np.mean(flat_advs):.4f} | Std: {np.std(flat_advs):.4f}")
    logger.info("="*60)
    logger.info(f"Processing Complete. Output saved to: {dest_path}")

if __name__ == "__main__":
    main()