# FILE: debug/debug_z_heights.py
# (Patched v1.0 - Correct Arguments for SemanticPlannerDataset v9.0)

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Path Setup ---
# Assumes this script is located in /project_root/ or /project_root/debug/
# We search for the root by looking for 'utils'
current_path = Path(__file__).resolve()
ROOT = current_path.parent
if (ROOT / "utils").exists():
    pass # We are in root
elif (ROOT.parent / "utils").exists():
    ROOT = ROOT.parent # We are in a subdirectory
else:
    # Fallback to standard 2-level depth
    ROOT = current_path.parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

print(f"Project Root detected at: {ROOT}")

from utils.semantic_planner_dataset import SemanticPlannerDataset

def analyze_z_heights():
    """Analyzes Z heights in gt_pose_chunk from training data."""
    
    print("=" * 60)
    print("Z HEIGHT ANALYSIS FOR TRAINING DATA")
    print("=" * 60)
    
    # --- CONFIG ---
    # Update this path to where your LMDB actually is
    lmdb_path = "/content/drive/MyDrive/pda/data/validation/training_set.lmdb" 
    
    if not os.path.exists(lmdb_path):
        print(f"❌ ERROR: Dataset not found at: {lmdb_path}")
        print("Please edit the 'lmdb_path' variable in the script.")
        return

    try:
        # [PATCH] Corrected arguments to match SemanticPlannerDataset.__init__
        dataset = SemanticPlannerDataset(
            dataset_path=lmdb_path,  # Was 'lmdb_path'
            chunk_size=10,
            use_aug=False            # Was 'use_augmentation'
        )
        print(f"✅ Loaded dataset with {len(dataset)} chunks")
    except Exception as e:
        print(f"❌ ERROR loading dataset class: {e}")
        return
    
    # Collect Z heights
    all_z_heights = []
    gripper_when_low_z = []  
    
    # Analyze a subset for speed
    num_samples = min(2000, len(dataset)) 
    print(f"Analyzing {num_samples} samples randomly...")
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i in indices:
        try:
            sample = dataset[i]
            if sample is None:
                continue
                
            # Convert tensors to numpy
            gt_pose_chunk = sample['gt_pose_chunk'].numpy()  # (K, 7)
            gt_grip_chunk = sample['gt_grip_chunk'].numpy()  # (K, 1)
            
            # Analyze every step in the chunk
            for k in range(len(gt_pose_chunk)):
                z = gt_pose_chunk[k, 2]  # Z coordinate (Index 2 is Z)
                grip = gt_grip_chunk[k, 0]
                all_z_heights.append(z)
                
                # Check gripper state when Z is in "Grasp Zone" (< 0.45m)
                if z < 0.45:
                    gripper_when_low_z.append(grip)
                    
        except Exception as e:
            print(f"Error reading index {i}: {e}")
            continue
    
    if not all_z_heights:
        print("❌ ERROR: No valid samples found!")
        return
    
    all_z_heights = np.array(all_z_heights)
    gripper_when_low_z = np.array(gripper_when_low_z)
    
    # --- REPORT ---
    print("\n" + "-" * 60)
    print("Z HEIGHT STATISTICS:")
    print("-" * 60)
    print(f"  Min Z:    {all_z_heights.min():.4f}m")
    print(f"  Max Z:    {all_z_heights.max():.4f}m")
    print(f"  Mean Z:   {all_z_heights.mean():.4f}m")
    print(f"  Std Z:    {all_z_heights.std():.4f}m")
    
    print("\n" + "-" * 60)
    print("Z HEIGHT DISTRIBUTION (The 'Retract Bias' Check):")
    print("-" * 60)
    
    total = len(all_z_heights)
    ranges = [
        ("Grasp Zone (< 0.45m)", all_z_heights < 0.45),
        ("Transition (0.45-0.50m)", (all_z_heights >= 0.45) & (all_z_heights < 0.50)),
        ("Transport Zone (> 0.50m)", all_z_heights >= 0.50),
    ]
    
    for name, mask in ranges:
        count = mask.sum()
        pct = 100 * count / total
        print(f"  {name:<25}: {count:6d} ({pct:5.1f}%)")
    
    # --- GRIPPER LOGIC CHECK ---
    if len(gripper_when_low_z) > 0:
        print("\n" + "-" * 60)
        print("WHAT IS THE ROBOT DOING LOW? (Z < 0.45m):")
        print("-" * 60)
        # Assuming > 0.5 is CLOSED (based on your training logic)
        closed_count = (gripper_when_low_z > 0.5).sum()
        open_count = (gripper_when_low_z <= 0.5).sum()
        
        print(f"  Gripper CLOSED: {closed_count:5d} ({100*closed_count/len(gripper_when_low_z):.1f}%)")
        print(f"  Gripper OPEN:   {open_count:5d} ({100*open_count/len(gripper_when_low_z):.1f}%)")
        
        if open_count == 0:
            print("  ⚠️ WARNING: No samples found where gripper is OPEN while low.")
            print("     The model might think it must CLOSE before descending!")
    else:
        print("\n  ⚠️ CRITICAL: Dataset contains ZERO samples in the grasp zone (< 0.45m).")
        print("     This explains why the robot hovers: it has no data teaching it to go low.")

    # Save simple histogram
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(all_z_heights, bins=100, color='blue', alpha=0.7)
        plt.axvline(x=0.42, color='red', linestyle='--', label='Table Surface (approx)')
        plt.title('Z-Height Distribution in Training Data')
        plt.xlabel('Z Height (meters)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig('z_height_debug.png')
        print(f"\n📊 Plot saved to 'z_height_debug.png'")
    except:
        pass

if __name__ == "__main__":
    analyze_z_heights()