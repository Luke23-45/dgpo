# FILE: scripts/audit_dataset_visuals.py

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import cv2

# --- Project Imports ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.semantic_planner_dataset import SemanticPlannerDataset

def main():
    # PATH TO YOUR DATASET
    # Update this to match your actual LMDB path
    dataset_path = r"C:\Users\Hellx\Documents\Programming\python\Project\redhot\data\final_training_set\training_set.lmdb" 
    
    print(f"Loading dataset: {dataset_path}")
    
    try:
        dataset = SemanticPlannerDataset(
            dataset_path=dataset_path,
            use_aug=False, # Look at RAW data, not augmented
            chunk_size=1
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Dataset Size: {len(dataset)}")
    
    # Output Directory
    out_dir = Path("audit_images")
    out_dir.mkdir(exist_ok=True)
    
    # Sample 100 random frames
    indices = np.random.choice(len(dataset), 100, replace=False)
    
    print("Generating visual audit...")
    
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    axes = axes.flatten()
    
    for i, idx in enumerate(tqdm(indices)):
        sample = dataset[idx]
        
        # Get Image (Tensor C,H,W -> Numpy H,W,C)
        # Assuming values are 0-1 or normalized. 
        # If normalized, we need to un-normalize to see it clearly.
        img_tensor = sample['curr_image']
        
        # Simple un-normalization for visualization (assuming mean 0.5)
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np * 0.5) + 0.5
        img_np = np.clip(img_np, 0, 1)
        
        # Get Proprio (Z-Height)
        z_height = sample['curr_proprio'][2].item()
        phase = sample['gt_phase_label'].item()
        
        ax = axes[i]
        ax.imshow(img_np)
        ax.axis('off')
        
        # Color border based on phase (Green=Reach, Red=Grasp)
        title_color = 'green' if phase == 0 else 'red'
        ax.set_title(f"Z:{z_height:.2f}", fontsize=8, color=title_color)

    plt.tight_layout()
    plt.savefig(out_dir / "dataset_audit_grid.png", dpi=150)
    print(f"\nSaved audit grid to: {out_dir / 'dataset_audit_grid.png'}")
    print("OPEN THIS IMAGE. Look for:")
    print("1. Robot arm blocking the red cube.")
    print("2. Cube barely visible at the edge.")
    print("3. Extremely dark images.")

if __name__ == "__main__":
    main()