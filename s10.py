import sys
import torch
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

# --- Project Imports ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.semantic_planner import SemanticPlanner, SemanticPlannerConfig
from utils.semantic_planner_dataset import SemanticPlannerDataset, semantic_planner_collate_fn
from torch.utils.data import DataLoader

# --- CONFIGURATION ---
# UPDATE THESE PATHS TO MATCH YOUR SETUP
CHECKPOINT_PATH = r"C:/Users/Hellx/Documents/Programming/python/Project/redhot/checkpoints/backup/backup_epoch_029.ckpt"
VAL_DATASET_PATH = r"C:/Users/Hellx/Documents/Programming/python/Project/redhot/data/validation/training_set.lmdb"

# Map integers to human-readable names based on utils/scripted_expert.py
PHASE_MAP = {
    0: "0_Approach (Pre-Grasp)",
    1: "1_Grasp (Action)",
    2: "2_Lift & Move",
    3: "3_Place & Release",
    4: "4_Retract/Done"
}

def load_model_from_checkpoint(ckpt_path):
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu",weights_only=False)
    
    # Reconstruct Config from the checkpoint hyperparameters if available, 
    # otherwise manually match your training config
    hparams = checkpoint.get("hyper_parameters", {})
    
    # Reconstruct the config object
    # We default to v9.0 standard if params aren't found
    cfg = SemanticPlannerConfig(
        proprio_dim=hparams.get("model", {}).get("proprio_dim", 22),
        vision_feature_dim=hparams.get("model", {}).get("vision_feature_dim", 768),
        chunk_size=hparams.get("model", {}).get("chunk_size", 10),
        # Add other params if your config structure differs
    )
    
    model = SemanticPlanner(cfg)
    # Load state dict (removing 'model.' prefix if saved by Lightning)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("model.", "") if k.startswith("model.") else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model
    model = load_model_from_checkpoint(CHECKPOINT_PATH)
    model.to(device)

    # 2. Load Validation Dataset
    print("Loading Validation Dataset...")
    val_dataset = SemanticPlannerDataset(
        dataset_path=VAL_DATASET_PATH,
        use_aug=False,
        chunk_size=10,
        proprio_noise=0.0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        collate_fn=semantic_planner_collate_fn,
        shuffle=False,
        num_workers=2
    )

    # 3. Storage for Metrics
    phase_errors = defaultdict(list)

    print("Running Inference...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Inference
            outputs = model(batch)
            pred_pose_chunk = outputs['pose_chunk'] # (B, K, 7)
            gt_pose_chunk = batch['gt_pose_chunk']  # (B, K, 7)
            gt_phases = batch['gt_phase_label']     # (B,)

            # Evaluation Logic:
            # We compare the IMMEDIATE NEXT step (t+1), which is index 0 of the chunk
            pred_next_pos = pred_pose_chunk[:, 0, :3]
            gt_next_pos = gt_pose_chunk[:, 0, :3]

            # Calculate L2 Distance (Euclidean Error) in Meters
            # Shape: (B,)
            errors = torch.norm(pred_next_pos - gt_next_pos, dim=1).cpu().numpy()
            phases = gt_phases.cpu().numpy()

            # Bucketing
            for p, err in zip(phases, errors):
                phase_name = PHASE_MAP.get(p, "Unknown")
                phase_errors[phase_name].append(err)

    # 4. Report Generation
    print("\n" + "="*60)
    print(f"{'PHASE NAME':<25} | {'COUNT':<6} | {'MEAN ERR (cm)':<13} | {'STD (cm)':<10}")
    print("-" * 60)

    stats = []
    
    # Sort by phase index
    sorted_keys = sorted(phase_errors.keys())
    
    for phase_name in sorted_keys:
        errs = np.array(phase_errors[phase_name])
        count = len(errs)
        mean_cm = np.mean(errs) * 100
        std_cm = np.std(errs) * 100
        
        print(f"{phase_name:<25} | {count:<6} | {mean_cm:.4f} cm      | {std_cm:.4f}")
        stats.append({"Phase": phase_name, "Mean": mean_cm, "Std": std_cm})

    print("="*60)

    # 5. Diagnosis logic
    print("\nDIAGNOSIS:")
    best_phase = min(stats, key=lambda x: x['Mean'])
    worst_phase = max(stats, key=lambda x: x['Mean'])
    
    print(f"✅ Best Performance:  {best_phase['Phase']} ({best_phase['Mean']:.2f} cm)")
    print(f"❌ Worst Performance: {worst_phase['Phase']} ({worst_phase['Mean']:.2f} cm)")
    
    if worst_phase['Phase'].startswith("0"):
        print("-> Issue: APPROACH LAG. The robot is moving fast and the model is lagging.")
        print("   Fix: Increase Action Horizon (k) or train on velocity inputs.")
    elif worst_phase['Phase'].startswith("1"):
        print("-> Issue: FINE MOTOR PRECISION. The model fails at the critical grasp moment.")
        print("   Fix: Unfreeze more Vision Layers (3-4) and reduce Proprio Noise.")
    elif worst_phase['Phase'].startswith("3"):
        print("-> Issue: DRIFT ACCUMULATION. Errors pile up over the long trajectory.")
        print("   Fix: Generate longer episodes (1000+) to cover more drift scenarios.")

if __name__ == "__main__":
    main()