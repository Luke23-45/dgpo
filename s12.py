"""
Script: tune_awr_parameters.py
Mathematically derives the optimal AWR Temperature (tau) and Max Weight
based on the actual distribution of Advantages in the dataset.
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

# --- Project Imports ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from utils.expert_dataset import ExpertTrajectoryDataset

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("AWR_TUNER")

def calculate_optimal_parameters(dataset_path: str):
    log.info(f"--- AWR Parameter Auto-Tuner ---")
    log.info(f"Loading dataset: {dataset_path}")

    # 1. Load Dataset (Lightweight Mode)
    try:
        dataset = ExpertTrajectoryDataset(
            demo_path=dataset_path,
            observation_horizon=1, # Minimal load
            action_horizon=1
        )
    except Exception as e:
        log.error(f"Failed to load dataset: {e}")
        return

    num_episodes = dataset.get_num_episodes()
    log.info(f"Found {num_episodes} episodes.")

    # 2. Extract All Advantages
    all_advantages = []
    
    log.info("Scanning Advantages...")
    for i in tqdm(range(num_episodes)):
        ep_meta = dataset.episode_metadata[i]
        
        if "advantages" not in ep_meta["modalities"]:
            log.error("Dataset does not contain 'advantages'. Run advantage_calculator.py first!")
            return

        adv_meta = ep_meta["modalities"]["advantages"]
        
        # Direct LMDB extraction for speed
        adv_data = dataset._get_full_modality_array(
            adv_meta["key"], 
            adv_meta["compression"], 
            adv_meta["dtype"], 
            tuple(adv_meta["shape"])
        )
        all_advantages.append(adv_data)

    # Concatenate into one giant array
    advantages = np.concatenate(all_advantages)
    
    # 3. Analyze Statistics
    mean_adv = np.mean(advantages)
    std_adv = np.std(advantages)
    min_adv = np.min(advantages)
    max_adv = np.max(advantages)
    
    print("\n" + "="*40)
    print("DATASET STATISTICS")
    print("="*40)
    print(f"Count: {len(advantages)}")
    print(f"Mean:  {mean_adv:.4f} (Should be close to 0.0)")
    print(f"Std:   {std_adv:.4f}")
    print(f"Min:   {min_adv:.4f}")
    print(f"Max:   {max_adv:.4f}")
    print("="*40)

    # 4. Derive Optimal Temperature (Tau)
    # Theory: We want weights w = exp(A/tau).
    # A common heuristic is to set tau s.t. the 'Best' actions get a weight of ~5.0 to 10.0.
    # Usually, 'Best' actions are around +2 Standard Deviations.
    
    # Let's target that an Advantage of +1.5 StdDev results in a weight of ~5.0
    # w = exp(A / tau) -> ln(w) = A / tau -> tau = A / ln(w)
    
    target_weight = 5.0
    target_sigma = 1.5 # We want the top ~7% of data to have strong weights
    
    advantage_at_target = std_adv * target_sigma
    
    optimal_tau = advantage_at_target / np.log(target_weight)
    
    # 5. Determine Max Weight Clipping
    # We calculate what the weight WOULD be for the absolute max advantage
    max_theoretical_weight = np.exp(max_adv / optimal_tau)
    
    # We clip to avoid exploding gradients, usually to 10.0 or 20.0
    recommended_clip = min(20.0, max(5.0, max_theoretical_weight))

    print("\n" + "="*40)
    print("🏆 RECOMMENDED CONFIGURATION")
    print("="*40)
    print(f"awr_temperature: {optimal_tau:.4f}")
    print(f"awr_max_weight:  {recommended_clip:.1f}")
    print("="*40 + "\n")
    
    print("Logic:")
    print(f"1. Std Dev is {std_adv:.2f}.")
    print(f"2. We want actions at +{target_sigma} sigma ({advantage_at_target:.2f}) to have weight {target_weight}.")
    print(f"3. Max Advantage ({max_adv:.2f}) would yield weight {max_theoretical_weight:.2f} (Clipped to {recommended_clip}).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to training_set.lmdb")
    args = parser.parse_args()
    
    calculate_optimal_parameters(args.dataset)