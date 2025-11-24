# FILE: eval/validate_open_loop.py
# (Definitive Phase 1: Open-Loop Validation for Action Chunking Models)

"""
Phase 1: Open-Loop Sanity Check & SOTA Validation.

This script performs a rigorous statistical audit of the trained Action Chunking model 
against the validation dataset.

It computes metrics for the full Chunk (Trajectory) AND the Immediate Step (Next Action):
1. Trajectory Euclidean Error (cm) - Mean over K steps
2. Trajectory Geodesic Error (deg) - Mean over K steps
3. Next-Step Position Error (cm) - Critical for RHC
4. Gripper Precision/Recall (Action Classification)
5. Inference Latency (ms)

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
    format="%(asctime)s [%(levelname)s] [VAL-v9.0] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# --- METRIC UTILITIES ---

def compute_geodesic_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Computes Geodesic Rotation Error in Degrees.
    Input shapes: [B, K, 4] or [B, 4]
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
    pred_chunk: torch.Tensor, 
    gt_chunk: torch.Tensor, 
    save_dir: Path
):
    """
    Saves a visual comparison of Pred vs GT Trajectories.
    Projects the 3D points onto the 2D image (Orthographic approx for sanity).
    """
    if batch_idx > 5: return # Only save first 5 batches

    # Unnormalize image for display (Assuming 0.5 mean/std)
    img_tensor = images[0].cpu().permute(1, 2, 0)
    img_np = (img_tensor * 0.5 + 0.5).numpy()
    img_np = np.clip(img_np, 0, 1)

    # Extract trajectories (Batch 0)
    # Shape: [K, 3]
    pred_traj = pred_chunk[0, :, :3].cpu().numpy()
    gt_traj = gt_chunk[0, :, :3].cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_np)
    
    # Simple text annotation of the start and end points
    # Real 3D projection requires camera intrinsics which aren't always available here.
    # We print the deltas instead.
    info_text = (
        f"Step 0 GT:   {gt_traj[0]}\n"
        f"Step 0 Pred: {pred_traj[0]}\n"
        f"Step K GT:   {gt_traj[-1]}\n"
        f"Step K Pred: {pred_traj[-1]}"
    )
    ax.text(5, 30, info_text, color='yellow', fontsize=8, backgroundcolor='black', verticalalignment='top')
    
    plt.title(f"Batch {batch_idx} Trajectory Analysis")
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
        
        # Extract Architecture Params
        self.chunk_size = self.pl_module.cfg.model.get("chunk_size", 10)
        log.info(f"Model Chunk Size detected: {self.chunk_size}")

        # 2. Load Dataset (Validation Split Only)
        # CRITICAL: Must match the v9.0 Dataset API
        log.info(f"Loading Validation Dataset: {cfg.dataset_path}")
        self.dataset = SemanticPlannerDataset(
            dataset_path=cfg.dataset_path,
            use_aug=False,  # Strict validation, no noise
            chunk_size=self.chunk_size, # Sync chunk size
            proprio_noise=0.0
        )
        
        self.loader = DataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            shuffle=False, 
            num_workers=4,
            pin_memory=True,
            collate_fn=semantic_planner_collate_fn
        )
        
        # Metric Storage
        self.metrics = {
            "traj_pos_errors": [], # Mean over K
            "traj_rot_errors": [], # Mean over K
            "next_pos_errors": [], # Immediate step (Index 0)
            "next_rot_errors": [], # Immediate step (Index 0)
            "gripper_preds": [],
            "gripper_gts": [],
            "latencies": []
        }

    @torch.no_grad()
    def run(self):
        log.info("Starting Open-Loop Validation Loop...")
        
        for batch_idx, batch in enumerate(tqdm(self.loader, desc="Validating")):
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # --- Inference & Latency Check ---
            start_time = time.time()
            outputs = self.model(batch)
            end_time = time.time()
            self.metrics["latencies"].append((end_time - start_time) * 1000) # ms

            # --- Extract Predictions (Chunked) ---
            # Shape: [B, K, 7] and [B, K, 1]
            pred_pose_chunk = outputs['pose_chunk']
            pred_grip_chunk = outputs['gripper_chunk']

            # --- Extract Ground Truth (Chunked) ---
            # Shape: [B, K, 7] and [B, K, 1]
            # Fixed: Using the correct v9.0 keys
            gt_pose_chunk = batch['gt_pose_chunk']
            gt_grip_chunk = batch['gt_grip_chunk']

            # --- 1. Trajectory Errors (Mean over Chunk) ---
            # Position (Euclidean)
            traj_pos_err = torch.norm(pred_pose_chunk[..., :3] - gt_pose_chunk[..., :3], dim=-1) # (B, K)
            self.metrics["traj_pos_errors"].extend(traj_pos_err.mean(dim=1).cpu().numpy())

            # Rotation (Geodesic)
            traj_rot_err = compute_geodesic_distance(pred_pose_chunk[..., 3:], gt_pose_chunk[..., 3:]) # (B, K)
            self.metrics["traj_rot_errors"].extend(traj_rot_err.mean(dim=1).cpu().numpy())

            # --- 2. Next-Step Errors (Immediate Action - Critical for RHC) ---
            # Position (Index 0)
            next_pos_err = torch.norm(pred_pose_chunk[:, 0, :3] - gt_pose_chunk[:, 0, :3], dim=-1)
            self.metrics["next_pos_errors"].extend(next_pos_err.cpu().numpy())
            
            # Rotation (Index 0)
            next_rot_err = compute_geodesic_distance(pred_pose_chunk[:, 0, 3:], gt_pose_chunk[:, 0, 3:])
            self.metrics["next_rot_errors"].extend(next_rot_err.cpu().numpy())

            # --- 3. Gripper Classification (Flat) ---
            # Flatten B and K to evaluate every single timestep classification
            pred_cls = (torch.sigmoid(pred_grip_chunk) > 0.5).float().view(-1)
            gt_cls = gt_grip_chunk.float().view(-1)
            
            self.metrics["gripper_preds"].extend(pred_cls.cpu().numpy())
            self.metrics["gripper_gts"].extend(gt_cls.cpu().numpy())

            # --- 4. Visualization (Sanity Check) ---
            if batch_idx < 5:
                visualize_batch_prediction(
                    batch_idx, batch['curr_image'], pred_pose_chunk, gt_pose_chunk, self.output_dir
                )

        self._generate_report()

    def _generate_report(self):
        log.info("--- Generating SOTA Validation Report ---")
        
        # Convert to numpy for stats
        # Multiply by 100 for CM
        traj_pos = np.array(self.metrics["traj_pos_errors"]) * 100.0
        next_pos = np.array(self.metrics["next_pos_errors"]) * 100.0
        traj_rot = np.array(self.metrics["traj_rot_errors"])
        next_rot = np.array(self.metrics["next_rot_errors"])
        latencies = np.array(self.metrics["latencies"])
        
        # Gripper Metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.metrics["gripper_gts"], self.metrics["gripper_preds"], 
            average='binary', zero_division=0
        )
        conf_mat = confusion_matrix(self.metrics["gripper_gts"], self.metrics["gripper_preds"])
        
        # --- The Data Sheet ---
        stats = {
            "Metric": [
                "Trajectory Pos Error (Mean)", "Trajectory Rot Error (Mean)",
                "Next-Step Pos Error (Mean)", "Next-Step Rot Error (Mean)",
                "Next-Step Pos Error (99th%)", "Gripper F1-Score", 
                "Gripper Precision", "Gripper Recall", "Inference Latency"
            ],
            "Value": [
                f"{np.mean(traj_pos):.2f} cm", f"{np.mean(traj_rot):.2f} deg",
                f"{np.mean(next_pos):.2f} cm", f"{np.mean(next_rot):.2f} deg",
                f"{np.percentile(next_pos, 99):.2f} cm",
                f"{f1:.4f}", f"{precision:.4f}", f"{recall:.4f}",
                f"{np.mean(latencies):.2f} ms"
            ]
        }
        
        df = pd.DataFrame(stats)
        
        # Save to Disk
        csv_path = self.output_dir / "validation_report_v9.csv"
        df.to_csv(csv_path, index=False)
        
        print("\n" + "="*50)
        print("PHASE 1: ACTION CHUNKING SANITY REPORT")
        print("="*50)
        print(df.to_string(index=False))
        print("="*50)
        print(f"Gripper Confusion Matrix:\n{conf_mat}")
        print("="*50)
        
        # --- Interpretive Feedback ---
        mean_next_pos = np.mean(next_pos)
        
        print("\n>>> AUTOMATED DIAGNOSIS:")
        if mean_next_pos < 2.0 and f1 > 0.90:
            print(f"[PASS] Model is robust (Next Step Err: {mean_next_pos:.2f}cm). Ready for Simulation.")
        else:
            print("[FAIL] Model is undertrained. DO NOT run simulation yet.")
            if mean_next_pos >= 2.0:
                print(f" - Position Error too high ({mean_next_pos:.2f}cm). Need < 2.0cm for reliable grasping.")
            if f1 <= 0.90:
                print(f" - Gripper Logic weak (F1={f1:.2f}). Check class imbalance.")

@hydra.main(version_base=None, config_path="../configs", config_name="validate_open_loop_config")
def main(cfg: DictConfig):
    validator = Validator(cfg)
    validator.run()

if __name__ == "__main__":
    main()