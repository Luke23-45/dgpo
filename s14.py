# FILE: scripts/verify_dataset_integrity.py
# (SOTA Diagnostic Tool)

"""
Dataset Integrity Validator.

Run this BEFORE training to ensure:
1. The LMDB data is uncorrupted.
2. The Advantage-Weighted Regression (AWR) statistics are valid.
3. The Semantic Phase labels are present and logical.
4. The Gripper interaction labels are balanced.
"""

import argparse
import logging
import sys
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.semantic_planner_dataset import SemanticPlannerDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("DataCheck")

def check_tensor_stats(name, tensor):
    """Helper to print tensor stats."""
    if isinstance(tensor, torch.Tensor):
        data = tensor.float().numpy()
    else:
        data = np.array(tensor)
    
    return {
        "shape": data.shape,
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data))
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to training.lmdb")
    parser.add_argument("--samples", type=int, default=2000, help="Number of random samples to check")
    args = parser.parse_args()

    # 1. Load Dataset
    log.info(f"Loading Dataset from: {args.dataset}")
    try:
        # Disable augmentation for pure data inspection
        ds = SemanticPlannerDataset(args.dataset, use_aug=False)
    except Exception as e:
        log.error(f"CRITICAL: Failed to initialize dataset class. Error: {e}")
        sys.exit(1)

    total_len = len(ds)
    log.info(f"Dataset Size: {total_len} frames")

    # 2. Random Sampling Loop
    log.info(f"Inspecting {args.samples} random samples...")
    indices = np.random.choice(total_len, args.samples, replace=False)
    
    advantages = []
    gripper_states = []
    phases = []
    pose_deltas = []

    for idx in tqdm(indices):
        try:
            sample = ds[idx]
            if sample is None:
                log.error(f"Sample {idx} returned None! Data corruption detected.")
                continue

            # Collect Stats
            advantages.append(sample['advantage'].item())
            gripper_states.append(sample['ground_truth_gripper_state'].item())
            phases.append(sample['task_phase'].item())

            # Check Geometric Logic
            # Calculate distance between Current Proprio (State) and Ground Truth Subgoal (Target)
            # Note: Proprio is [qpos, qvel...], we need EE pose comparison.
            # Since dataset gives us tensors, we'll roughly estimate activity by checking if target != 0
            
            target_pose = sample['ground_truth_subgoal_pose']
            # Sanity check: Target should not be all zeros
            if torch.sum(torch.abs(target_pose)) < 1e-3:
                log.warning(f"Sample {idx}: Zero-Vector Subgoal Pose detected.")

        except Exception as e:
            log.error(f"Error processing sample {idx}: {e}")

    # 3. Report Findings

    # --- A. Advantage Analysis ---
    adv_np = np.array(advantages)
    log.info("-" * 40)
    log.info("1. ADVANTAGE DISTRIBUTION (AWR)")
    log.info(f"   Mean: {np.mean(adv_np):.4f} (Should be approx 0.0)")
    log.info(f"   Std:  {np.std(adv_np):.4f}")
    log.info(f"   Min:  {np.min(adv_np):.4f}")
    log.info(f"   Max:  {np.max(adv_np):.4f}")
    
    # Warning if Advantages are broken
    if np.max(adv_np) < 1.0:
        log.warning("   ⚠️  Max Advantage is very low. Model might not differentiate good vs bad actions.")
    else:
        log.info("   ✅ Advantage spread looks healthy.")

    # --- B. Gripper Analysis ---
    grip_np = np.array(gripper_states)
    closed_ratio = np.sum(grip_np > 0.5) / len(grip_np)
    log.info("-" * 40)
    log.info("2. GRIPPER CLASS BALANCE")
    log.info(f"   % Closed (1.0): {closed_ratio*100:.2f}%")
    log.info(f"   % Open   (0.0): {(1-closed_ratio)*100:.2f}%")
    
    if closed_ratio < 0.05:
        log.error("   ❌ CRITICAL: < 5% Positive Gripper Labels. The model will collapse to 'Always Open'.")
        log.error("      Fix: Re-run 'preprocess_advantages.py' or check expert logic.")
    elif closed_ratio > 0.40:
         log.warning("   ⚠️  Unusually high grasp ratio. Ensure 'is_grasped' logic isn't inverted.")
    else:
        log.info("   ✅ Class balance is within expected range for Pick & Place (10-30%).")

    # --- C. Phase Analysis ---
    phase_counts = np.bincount(phases, minlength=5)
    log.info("-" * 40)
    log.info("3. TASK PHASES")
    for i, count in enumerate(phase_counts):
        log.info(f"   Phase {i}: {count} samples ({count/len(phases)*100:.1f}%)")
    
    if phase_counts[1] == 0: # Grasp Phase
         log.error("   ❌ CRITICAL: No 'Grasp' (Phase 1) samples found. The Phase transition logic is broken.")

    log.info("-" * 40)
    log.info("VERIFICATION COMPLETE.")

if __name__ == "__main__":
    main()