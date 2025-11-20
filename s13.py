# FILE: scripts/verify_new_data.py

import torch
import numpy as np
from utils.semantic_planner_dataset import SemanticPlannerDataset
import logging

# Setup
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("DataAudit")

# [IMPORTANT] Point this to your NEW dataset
dataset_path = r"C:\Users\Hellx\Documents\Programming\python\Project\redhot\data\training_ready\training_data.lmdb" 

log.info(f"Auditing Dataset: {dataset_path}")
ds = SemanticPlannerDataset(dataset_path, use_aug=False)

gt_grips = []
gt_phases = []

print("Scanning first 2000 samples...")
for i in range(2000):
    sample = ds[i]
    # Check Gripper Label
    grip = sample['ground_truth_gripper_state'].item()
    gt_grips.append(grip)
    
    # Check Phase Input
    phase = sample['task_phase'].item()
    gt_phases.append(phase)

# Statistics
grip_ratio = np.mean(gt_grips)
phase_counts = np.bincount(gt_phases, minlength=5)

print("-" * 30)
print(f"Gripper Closed Ratio: {grip_ratio*100:.2f}%")
print("-" * 30)
print(f"Phase Distribution:")
for i, c in enumerate(phase_counts):
    print(f"  Phase {i}: {c} samples ({c/len(gt_phases)*100:.1f}%)")
print("-" * 30)

if grip_ratio == 0.0:
    print("🚨 FAILURE: Dataset Gripper Labels are ALL ZERO.")
    print("   The Expert generation script failed to capture the grasp intent.")
elif phase_counts[1] == 0:
    print("🚨 FAILURE: Dataset Phase Labels are MISSING GRASP PHASE.")
else:
    print("✅ DATA LOOKS GOOD. The issue is in Training/Evaluation config.")