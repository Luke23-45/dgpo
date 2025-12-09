"""
(Definitive Phase 1: Open-Loop Validation for Action Chunking Models)

This script performs a rigorous statistical audit of the trained Action Chunking model 
(v9.0) against the validation dataset.

FEATURES:
1. Fixes PyTorch 2.6+ Checkpoint Loading (weights_only=False).
2. Computes both TRAJECTORY (Mean over K) and NEXT-STEP (Immediate) errors.
3. Generates a "Go/No-Go" Report Card for simulation.

Usage:
    python eval/validate_open_loop.py checkpoint_path=path/to/model.ckpt dataset_path=path/to/val.lmdb
"""

import logging
import sys
import time
import csv
from pathlib import Path
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

# --- Project Imports ---
# Add root to path to ensure imports work
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

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
    Matches SOTA standard: theta = 2 * arccos(|<q1, q2>|)
    Handles double cover property (q == -q).
    """
    # Normalize inputs
    q1 = F.normalize(q1, dim=-1)
    q2 = F.normalize(q2, dim=-1)
    
    # Dot product
    dot = torch.sum(q1 * q2, dim=-1)
    
    # Absolute value handles double cover
    dot = torch.abs(dot)
    
    # Clamp for numerical stability
    dot = torch.clamp(dot, -1.0 + 1e-6, 1.0 - 1e-6)
    
    # Calculate angle
    theta_rad = 2 * torch.acos(dot)
    theta_deg = torch.rad2deg(theta_rad)
    
    return theta_deg

def visualize_prediction(batch_idx, image, pred_traj, gt_traj, save_dir):
    """
    Saves a quick sanity check visualization (projected trajectory).
    """
    if batch_idx > 5: return # Limit to first 5 batches

    # Unnormalize image (Assuming standard ImageNet or 0.5 stats)
    # Adjust based on your normalization strategy
    img_np = image.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * 0.5 + 0.5)
    img_np = np.clip(img_np, 0, 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_np)
    ax.set_title(f"Batch {batch_idx} Trajectory Check")
    
    # Text overlay
    info = (
        f"Step 0 (GT):   {gt_traj[0, :3].cpu().numpy().round(3)}\n"
        f"Step 0 (Pred): {pred_traj[0, :3].cpu().numpy().round(3)}"
    )
    ax.text(5, 20, info, color='white', fontsize=8, backgroundcolor='black')
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_dir / f"vis_val_{batch_idx}.png")
    plt.close()

class Validator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        
        # 1. Load Model (With Security Fix)
        log.info(f"Loading Checkpoint: {cfg.checkpoint_path}")
        
        # [CRITICAL FIX] weights_only=False allows loading the OmegaConf config stored in the checkpoint
        self.pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
            cfg.checkpoint_path, 
            map_location=self.device,
            weights_only=False 
        )
        self.model = self.pl_module.model.eval().to(self.device)
        
        # Extract Architecture Params
        self.chunk_size = self.pl_module.cfg.model.get("chunk_size", 10)
        log.info(f"Detected Chunk Size: {self.chunk_size}")

        # 2. Load Dataset (Validation Split)
        log.info(f"Loading Validation Dataset: {cfg.dataset_path}")
        self.dataset = SemanticPlannerDataset(
            dataset_path=cfg.dataset_path,
            use_aug=False,  # No noise for validation
            chunk_size=self.chunk_size,
            proprio_noise=0.0
        )
        
        self.loader = DataLoader(
            self.dataset,
            batch_size=cfg.get("batch_size", 32),
            shuffle=False, 
            num_workers=4,
            pin_memory=True,
            collate_fn=semantic_planner_collate_fn
        )
        
        # Storage
        self.stats = {
            "traj_pos": [], "traj_rot": [], # Mean over K
            "next_pos": [], "next_rot": [], # Immediate step (K=0)
            "grip_gt": [], "grip_pred": [],
            "latencies": []
        }

    @torch.no_grad()
    def run(self):
        log.info("Starting Validation Loop...")
        
        for batch_idx, batch in enumerate(tqdm(self.loader, desc="Validating")):
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # --- Inference ---
            t0 = time.time()
            out = self.model(batch)
            self.stats["latencies"].append((time.time() - t0) * 1000)

            # Predictions
            pred_pose = out['pose_chunk']   # (B, K, 7)
            pred_grip = out['gripper_chunk'] # (B, K, 1)

            # Ground Truths (Keys match your v9.0 Dataset)
            gt_pose = batch['gt_pose_chunk']
            gt_grip = batch['gt_grip_chunk']

            # --- 1. Position Error (L2 Euclidean) ---
            # Shape: (B, K)
            pos_err = torch.norm(pred_pose[..., :3] - gt_pose[..., :3], dim=-1)
            
            # Metric A: Trajectory Mean (How good is the plan?)
            self.stats["traj_pos"].extend(pos_err.mean(dim=1).cpu().numpy())
            
            # Metric B: Next Step (How good is the action?)
            self.stats["next_pos"].extend(pos_err[:, 0].cpu().numpy())

            # --- 2. Rotation Error (Geodesic Degrees) ---
            rot_err = compute_geodesic_distance(pred_pose[..., 3:], gt_pose[..., 3:])
            
            self.stats["traj_rot"].extend(rot_err.mean(dim=1).cpu().numpy())
            self.stats["next_rot"].extend(rot_err[:, 0].cpu().numpy())

            # --- 3. Gripper Classification ---
            # Evaluate every step in the chunk
            pred_cls = (torch.sigmoid(pred_grip) > 0.5).float().view(-1)
            gt_cls = gt_grip.float().view(-1)
            
            self.stats["grip_pred"].extend(pred_cls.cpu().numpy())
            self.stats["grip_gt"].extend(gt_cls.cpu().numpy())

            # --- 4. Visualization ---
            if batch_idx < 5:
                visualize_prediction(
                    batch_idx, batch['curr_image'][0], pred_pose[0], gt_pose[0], self.output_dir
                )

        self._finalize()

    def _finalize(self):
        log.info("Calculating Aggregate Statistics...")
        
        # Process Arrays (Multiply by 100 for CM)
        res = {}
        for k in ["traj_pos", "next_pos"]:
            arr = np.array(self.stats[k]) * 100.0 # m -> cm
            res[k] = {"mean": np.mean(arr), "p99": np.percentile(arr, 99)}
            
        for k in ["traj_rot", "next_rot"]:
            arr = np.array(self.stats[k])
            res[k] = {"mean": np.mean(arr)}
            
        # Gripper Stats
        p, r, f1, _ = precision_recall_fscore_support(
            self.stats["grip_gt"], self.stats["grip_pred"], average='binary', zero_division=0
        )
        
        # --- Report Generation ---
        report_data = {
            "Metric": [
                "Traj Pos Error (Mean)", "Next-Step Pos Error (Mean)", "Next-Step Pos Error (99%)",
                "Traj Rot Error (Mean)", "Next-Step Rot Error (Mean)",
                "Gripper F1", "Gripper Precision", "Gripper Recall", "Avg Latency (ms)"
            ],
            "Value": [
                f"{res['traj_pos']['mean']:.2f} cm", 
                f"{res['next_pos']['mean']:.2f} cm", 
                f"{res['next_pos']['p99']:.2f} cm",
                f"{res['traj_rot']['mean']:.2f} deg",
                f"{res['next_rot']['mean']:.2f} deg",
                f"{f1:.4f}", f"{p:.4f}", f"{r:.4f}",
                f"{np.mean(self.stats['latencies']):.2f} ms"
            ]
        }
        
        df = pd.DataFrame(report_data)
        print("\n" + "="*60)
        print("   AWSP v9.0 VALIDATION REPORT CARD   ")
        print("="*60)
        print(df.to_string(index=False))
        print("="*60)
        
        # Save
        df.to_csv(self.output_dir / "validation_results.csv", index=False)
        
        # --- Final Verdict ---
        mean_err = res['next_pos']['mean']
        print("\n>>> AUTOMATED DIAGNOSIS:")
        if mean_err < 2.0:
            print(f"✅ PASS. Precision ({mean_err:.2f}cm) is within 2cm tolerance.")
            print("   Recommendation: Proceed to Simulation / Real Robot deployment.")
        else:
            print(f"❌ CAUTION. Precision ({mean_err:.2f}cm) exceeds 2cm threshold.")
            print("   Recommendation: Train longer or check dataset quality.")

@hydra.main(version_base=None, config_path="../configs", config_name="validate_open_loop_config")
def main(cfg: DictConfig):
    # Ensure output dir exists
    Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir).mkdir(parents=True, exist_ok=True)
    
    val = Validator(cfg)
    val.run()

if __name__ == "__main__":
    main()