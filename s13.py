import torch
import numpy as np
from utils.semantic_planner_dataset import SemanticPlannerDataset

# 1. Load Dataset (Point to your local path)
dataset_path = "C:/Users/Hellx/Documents/Programming/python/Project/redhot/data/training_ready/training.lmdb" # or your Drive path
ds = SemanticPlannerDataset(dataset_path, use_aug=False)

print(f"Total Samples: {len(ds)}")

# 2. Inspect Random Samples
indices = np.random.randint(0, len(ds), 5)

print("\n--- Data Integrity Check ---")
for idx in indices:
    sample = ds[idx]
    
    # A. Check Advantage
    adv = sample['advantage'].item()
    
    # B. Check Gripper (Should be exactly 0.0 or 1.0)
    grip = sample['ground_truth_gripper_state'].item()
    
    # C. Check Pose (Should not be all zeros or NaNs)
    pose = sample['ground_truth_subgoal_pose']
    
    status = "✅ OK"
    if np.isnan(adv) or adv == 0.0: status = "❌ SUSPICIOUS ADVANTAGE"
    if grip not in [0.0, 1.0]: status = "❌ INVALID GRIPPER"
    if torch.sum(torch.abs(pose)) < 1e-4: status = "❌ EMPTY POSE"
    
    print(f"Idx {idx}: Adv={adv:.4f}, Grip={grip}, PoseSum={pose.sum():.2f} -> {status}")

# 3. Check Global Distribution (First 1000)
print("\n--- Distribution Check (First 1000) ---")
advantages = []
grippers = []
for i in range(1000):
    s = ds[i]
    advantages.append(s['advantage'].item())
    grippers.append(s['ground_truth_gripper_state'].item())

print(f"Advantage Mean: {np.mean(advantages):.4f} (Should be near 0)")
print(f"Gripper Active %: {np.mean(grippers)*100:.1f}% (Should be >0 and <100)")