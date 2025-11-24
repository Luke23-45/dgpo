# FILE: scripts/validate_open_loop.py
# (Definitive Phase 1 Validation Tool)

"""
Phase 1: Open-Loop Sanity Check & SOTA Validation.

This script performs a rigorous statistical audit of the trained model 
against the validation dataset (Held-out Expert Data).

It computes:
1. Euclidean Position Error (cm)
2. Geodesic Rotation Error (degrees) - Handling Quaternion Double Cover
3. Gripper Precision/Recall (Action Classification)
4. Inference Latency (ms)

It generates a 'Report Card' to determine if the model is ready for simulation.
"""

import logging
import sys
import time
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# --- Project Imports ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.semantic_planner import SemanticPlanner
from train.train_semantic_planner import SemanticPlannerLightningModule
from utils.semantic_planner_dataset import SemanticPlannerDataset, semantic_planner_collate_fn

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [VALIDATOR] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# --- METRIC UTILITIES ---

def compute_geodesic_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Computes Geodesic Rotation Error in Degrees.
    Matches SOTA standard: theta = 2 * arccos(|<q1, q2>|)
    Handles double cover: q and -q represent the same rotation.
    """
    # Normalize inputs to be safe
    q1 = F.normalize(q1, dim=-1)
    q2 = F.normalize(q2, dim=-1)
    
    # Dot product
    dot = torch.sum(q1 * q2, dim=-1)
    
    # Absolute value handles double cover
    dot = torch.abs(dot)
    
    # Clamp for numerical stability (arccos domain is [-1, 1])
    dot = torch.clamp(dot, -1.0 + 1e-6, 1.0 - 1e-6)
    
    # Calculate angle
    theta_rad = 2 * torch.acos(dot)
    theta_deg = torch.rad2deg(theta_rad)
    
    return theta_deg

def visualize_batch_prediction(
    batch_idx: int, 
    images: torch.Tensor, 
    pred_pose: torch.Tensor, 
    gt_pose: torch.Tensor, 
    save_dir: Path
):
    """
    Saves a visual comparison of Pred vs GT for qualitative analysis.
    Overlays 2D projection of target (Just XY center) for sanity.
    """
    if batch_idx > 5: return # Only save first 5 batches

    # Unnormalize image for display
    # Assuming (0.5, 0.5) normalization
    img_tensor = images[0].cpu().permute(1, 2, 0)
    img_np = (img_tensor * 0.5 + 0.5).numpy()
    img_np = np.clip(img_np, 0, 1)

    pred_xyz = pred_pose[0, :3].cpu().numpy()
    gt_xyz = gt_pose[0, :3].cpu().numpy()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img_np)
    
    # Annotation
    info_text = (
        f"GT:  [{gt_xyz[0]:.2f}, {gt_xyz[1]:.2f}, {gt_xyz[2]:.2f}]\n"
        f"Pred:[{pred_xyz[0]:.2f}, {pred_xyz[1]:.2f}, {pred_xyz[2]:.2f}]"
    )
    ax.text(5, 20, info_text, color='yellow', fontsize=8, backgroundcolor='black')
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_dir / f"val_vis_batch_{batch_idx}.png")
    plt.close()

class Validator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        
        # 1. Load Model
        log.info(f"Loading Checkpoint: {cfg.checkpoint_path}")
        self.pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
            cfg.checkpoint_path, map_location=self.device
        )
        self.model = self.pl_module.model.eval().to(self.device)
        
        # 2. Load Dataset (Validation Split Only)
        log.info(f"Loading Validation Dataset: {cfg.dataset_path}")
        self.dataset = SemanticPlannerDataset(
            dataset_path=cfg.dataset_path,
            use_aug=False  # Strict validation, no noise
        )
        
        self.loader = DataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            shuffle=False, # Sequential makes debugging easier
            num_workers=4,
            pin_memory=True,
            collate_fn=semantic_planner_collate_fn
        )
        
        # Metric Storage
        self.pos_errors = []
        self.rot_errors = []
        self.gripper_preds = []
        self.gripper_gts = []
        self.latencies = []

    @torch.no_grad()
    def run(self):
        log.info("Starting Open-Loop Validation Loop...")
        
        for batch_idx, batch in enumerate(tqdm(self.loader, desc="Validating")):
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
            
            # --- Inference & Latency Check ---
            start_time = time.time()
            outputs = self.model(batch)
            end_time = time.time()
            self.latencies.append((end_time - start_time) * 1000) # ms

            # --- Extract Predictions ---
            # Handle chunked output (v8/v13) vs single output
            if 'pose_chunk' in outputs:
                # For Open-Loop BC, we compare the immediate next step (t=0)
                pred_pose = outputs['pose_chunk'][:, 0, :]
                pred_grip_logit = outputs['gripper_chunk'][:, 0, :]
            else:
                pred_pose = outputs['pose']
                pred_grip_logit = outputs['gripper_logit']

            # --- Extract Ground Truth ---
            # Note: Dataset returns gt_next_pose, which corresponds to the prediction target
            gt_pose = batch['ground_truth_subgoal_pose']
            gt_grip = batch['ground_truth_gripper_state'].float().view(-1, 1)

            # --- 1. Position Error (Euclidean in meters) ---
            pos_err = torch.norm(pred_pose[:, :3] - gt_pose[:, :3], dim=-1) # (B,)
            self.pos_errors.extend(pos_err.cpu().numpy())

            # --- 2. Rotation Error (Geodesic in degrees) ---
            rot_err = compute_geodesic_distance(pred_pose[:, 3:], gt_pose[:, 3:]) # (B,)
            self.rot_errors.extend(rot_err.cpu().numpy())

            # --- 3. Gripper Classification ---
            pred_cls = (torch.sigmoid(pred_grip_logit) > 0.5).float()
            self.gripper_preds.extend(pred_cls.cpu().numpy().flatten())
            self.gripper_gts.extend(gt_grip.cpu().numpy().flatten())

            # --- 4. Visualization (Sanity Check) ---
            if batch_idx < 5:
                visualize_batch_prediction(
                    batch_idx, batch['initial_image'], pred_pose, gt_pose, self.output_dir
                )

        self._generate_report()

    def _generate_report(self):
        log.info("--- Generating SOTA Validation Report ---")
        
        # Convert to numpy for stats
        pos_errors = np.array(self.pos_errors) * 100.0 # Convert m to cm
        rot_errors = np.array(self.rot_errors)
        latencies = np.array(self.latencies)
        
        # Gripper Metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.gripper_gts, self.gripper_preds, average='binary', zero_division=0
        )
        conf_mat = confusion_matrix(self.gripper_gts, self.gripper_preds)
        
        # --- The Data Sheet ---
        stats = {
            "Metric": [
                "Position Error (Mean)", "Position Error (Median)", "Position Error (99th %)",
                "Rotation Error (Mean)", "Rotation Error (Median)", "Rotation Error (99th %)",
                "Gripper F1-Score", "Gripper Precision", "Gripper Recall",
                "Inference Latency (Mean)"
            ],
            "Value": [
                f"{np.mean(pos_errors):.2f} cm", 
                f"{np.median(pos_errors):.2f} cm", 
                f"{np.percentile(pos_errors, 99):.2f} cm",
                f"{np.mean(rot_errors):.2f} deg", 
                f"{np.median(rot_errors):.2f} deg", 
                f"{np.percentile(rot_errors, 99):.2f} deg",
                f"{f1:.4f}", 
                f"{precision:.4f}", 
                f"{recall:.4f}",
                f"{np.mean(latencies):.2f} ms"
            ]
        }
        
        df = pd.DataFrame(stats)
        
        # Save to Disk
        csv_path = self.output_dir / "validation_report.csv"
        df.to_csv(csv_path, index=False)
        
        print("\n" + "="*40)
        print("PHASE 1: SANITY CHECK REPORT CARD")
        print("="*40)
        print(df.to_string(index=False))
        print("="*40)
        print(f"Gripper Confusion Matrix:\n{conf_mat}")
        print("="*40)
        
        # --- Interpretive Feedback ---
        mean_pos = np.mean(pos_errors)
        mean_rot = np.mean(rot_errors)
        
        print("\n>>> AUTOMATED DIAGNOSIS:")
        if mean_pos < 1.5 and mean_rot < 5.0 and f1 > 0.90:
            print("[PASS] Model is robust. Ready for Simulation (Phase 2).")
        else:
            print("[FAIL] Model is undertrained. DO NOT run simulation yet.")
            if mean_pos >= 1.5:
                print(f" - Position Error too high ({mean_pos:.2f}cm). Check Normalization or LR.")
            if mean_rot >= 5.0:
                print(f" - Rotation Error too high ({mean_rot:.2f}deg). Check Quaternion loss weights.")
            if f1 <= 0.90:
                print(f" - Gripper Logic weak (F1={f1:.2f}). Check class imbalance weights.")

@hydra.main(version_base=None, config_path="../configs", config_name="validate_open_loop_config")
def main(cfg: DictConfig):
    validator = Validator(cfg)
    validator.run()

if __name__ == "__main__":
    main()