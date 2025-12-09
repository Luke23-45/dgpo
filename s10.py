# FILE: eval/validate_phase_analysis.py
# (Definitive SOTA Implementation for AWSP/BC Diagnostics - CLI Enabled)

import sys
import os
import torch
import numpy as np
import logging
import pandas as pd
import argparse  # <--- NEW: For command line arguments
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --- Project Imports ---
# robustly add root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.semantic_planner import SemanticPlanner, SemanticPlannerConfig
from utils.semantic_planner_dataset import SemanticPlannerDataset, semantic_planner_collate_fn

# --- DEFAULTS (Used if no arguments provided) ---
DEFAULT_CHECKPOINT = r"/content/drive/MyDrive/pda/data/awr/backup_epoch_021.ckpt"
DEFAULT_DATASET = r"/content/drive/MyDrive/pda/data/validation/training_set.lmdb"

# Map integers to human-readable names
PHASE_MAP = {
    0: "0_Approach (Pre-Grasp)",
    1: "1_Grasp (Action)",
    2: "2_Lift & Move",
    3: "3_Place & Release",
    4: "4_Retract/Done"
}

# Setup Professional Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("PhaseValidator")

def compute_geodesic_distance(q1: torch.Tensor, q2: torch.Tensor) -> np.ndarray:
    """
    Computes Geodesic Rotation Error in Degrees between two batches of quaternions.
    """
    q1 = F.normalize(q1, dim=-1)
    q2 = F.normalize(q2, dim=-1)
    dot = torch.sum(q1 * q2, dim=-1)
    dot = torch.abs(dot) # Handle double cover
    dot = torch.clamp(dot, -1.0 + 1e-6, 1.0 - 1e-6)
    theta_rad = 2 * torch.acos(dot)
    return torch.rad2deg(theta_rad).cpu().numpy()


def load_model_robust(ckpt_path, device):
    """
    SOTA Checkpoint Loader.
    Includes FIX for 'proprio_dim' multiple values error.
    """
    logger.info(f"Loading checkpoint from: {ckpt_path}")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    # 1. Load the raw checkpoint safely
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        logger.error(f"Failed to load checkpoint. Error: {e}")
        sys.exit(1)
    
    # 2. Reconstruct Configuration (Robust Method)
    hydra_cfg = checkpoint.get("cfg", {}).get("model", {})
    
    # Base configuration args
    config_args = {
        "proprio_dim": hydra_cfg.get("proprio_dim", 22),
        "vision_feature_dim": 768, 
        "chunk_size": 10,
    }

    # Update with Hyperparameters saved by Lightning (if they exist)
    hparams = checkpoint.get("hyper_parameters", {})
    model_params = hparams.get("model", {})
    
    for k, v in model_params.items():
        if hasattr(SemanticPlannerConfig, k):
            config_args[k] = v

    # Hard override standard params to prevent mismatch issues
    config_args["vision_feature_dim"] = 768 
    config_args["chunk_size"] = 10

    # Instantiate Config using **kwargs to avoid duplicate argument error
    cfg = SemanticPlannerConfig(**config_args)
    
    logger.info(f"Model Configuration Reconstructed: Chunk Size={cfg.chunk_size}")

    # 3. Initialize Model
    model = SemanticPlanner(cfg)
    
    # 4. Clean State Dict (Remove 'model.' prefix)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        # Strip 'model.' prefix if present
        name = k[6:] if k.startswith("model.") else k
        new_state_dict[name] = v
        
    # 5. Load Weights
    keys = model.load_state_dict(new_state_dict, strict=False)
    
    if keys.missing_keys:
        critical_missing = [k for k in keys.missing_keys if "head" in k or "encoder" in k]
        if critical_missing:
            logger.warning(f"Critical layers missing: {critical_missing}")
        else:
            logger.info("Weights loaded successfully (minor non-critical keys missing).")
            
    model.to(device)
    model.eval()
    return model

def main():
    # --- ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description="AWSP SOTA Phase-wise Diagnostic Tool")
    
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=DEFAULT_CHECKPOINT,
        help="Path to the model checkpoint (.ckpt)"
    )
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        default=DEFAULT_DATASET,
        help="Path to the validation dataset (.lmdb)"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="Inference batch size"
    )

    args = parser.parse_args()

    # --- SETUP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Accelerator: {device}")

    # 1. Load Model
    model = load_model_robust(args.checkpoint, device)

    # 2. Load Validation Dataset
    logger.info(f"Loading Dataset: {args.dataset}")
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset not found at {args.dataset}")
        sys.exit(1)

    val_dataset = SemanticPlannerDataset(
        dataset_path=args.dataset,
        use_aug=False,       # CRITICAL: No noise for validation
        chunk_size=10,       # Must match training
        proprio_noise=0.0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        collate_fn=semantic_planner_collate_fn,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 3. Storage
    metrics = {
        "pos_error": defaultdict(list),
        "rot_error": defaultdict(list),
        "grip_acc": defaultdict(list)
    }

    # 4. Inference Loop
    logger.info("Starting Phase-wise Inference...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            outputs = model(batch)
            
            # Use Index 0 (Immediate Next Step)
            pred_next_pos = outputs['pose_chunk'][:, 0, :3]
            pred_next_rot = outputs['pose_chunk'][:, 0, 3:]
            pred_next_grip = outputs['gripper_chunk'][:, 0, 0]

            gt_next_pos = batch['gt_pose_chunk'][:, 0, :3]
            gt_next_rot = batch['gt_pose_chunk'][:, 0, 3:]
            gt_next_grip = batch['gt_grip_chunk'][:, 0, 0]
            gt_phases = batch['gt_phase_label']

            # Metrics
            pos_errs_m = torch.norm(pred_next_pos - gt_next_pos, dim=1).cpu().numpy()
            rot_errs_deg = compute_geodesic_distance(pred_next_rot, gt_next_rot)
            pred_grip_binary = (torch.sigmoid(pred_next_grip) > 0.5).float()
            grip_correct = (pred_grip_binary == gt_next_grip).float().cpu().numpy()

            phases_np = gt_phases.cpu().numpy()
            
            for i in range(len(phases_np)):
                p_id = phases_np[i]
                metrics["pos_error"][p_id].append(pos_errs_m[i] * 100.0) # cm
                metrics["rot_error"][p_id].append(rot_errs_deg[i])
                metrics["grip_acc"][p_id].append(grip_correct[i])

    # 5. Report Generation
    logger.info("Aggregating Statistics...")
    
    report_rows = []
    all_phases = sorted(metrics["pos_error"].keys())
    
    for p_id in all_phases:
        p_name = PHASE_MAP.get(p_id, f"Phase {p_id}")
        pos_data = np.array(metrics["pos_error"][p_id])
        rot_data = np.array(metrics["rot_error"][p_id])
        grip_data = np.array(metrics["grip_acc"][p_id])
        
        row = {
            "Phase": p_name,
            "Count": len(pos_data),
            "Pos Mean (cm)": np.mean(pos_data),
            "Pos Std (cm)": np.std(pos_data),
            "Rot Mean (deg)": np.mean(rot_data),
            "Grip Acc (%)": np.mean(grip_data) * 100
        }
        report_rows.append(row)

    df = pd.DataFrame(report_rows)
    
    total_pos = np.concatenate(list(metrics["pos_error"].values()))
    total_rot = np.concatenate(list(metrics["rot_error"].values()))
    total_grip = np.concatenate(list(metrics["grip_acc"].values()))
    
    overall_row = {
        "Phase": "OVERALL",
        "Count": len(total_pos),
        "Pos Mean (cm)": np.mean(total_pos),
        "Pos Std (cm)": np.std(total_pos),
        "Rot Mean (deg)": np.mean(total_rot),
        "Grip Acc (%)": np.mean(total_grip) * 100
    }
    df = pd.concat([df, pd.DataFrame([overall_row])], ignore_index=True)

    # --- PRINT REPORT ---
    print("\n" + "="*80)
    print(f"   SOTA DIAGNOSTIC REPORT: {Path(args.checkpoint).name}")
    print("="*80)
    print(df.to_string(index=False, float_format=lambda x: "{:.2f}".format(x)))
    print("="*80)

    # --- AUTOMATED CONSULTANT ---
    print("\n>>> AUTOMATED RESEARCH CONSULTANT >>>")
    
    grasp_row = df[df['Phase'].str.contains("Grasp")]
    if not grasp_row.empty:
        grasp_err = grasp_row.iloc[0]['Pos Mean (cm)']
        if grasp_err > 2.0:
            print(f"❌ CRITICAL: Grasp Phase Error is {grasp_err:.2f}cm (> 2.0cm).")
            print("   -> Implication: The robot will miss small objects.")
        else:
            print(f"✅ SUCCESS: Grasp Phase Error is {grasp_err:.2f}cm. High precision confirmed.")

    retract_row = df[df['Phase'].str.contains("Retract")]
    if not retract_row.empty and not grasp_row.empty:
        retract_err = retract_row.iloc[0]['Pos Mean (cm)']
        bias_ratio = grasp_err / (retract_err + 1e-6)
        if bias_ratio > 3.0:
            print(f"⚠️ WARNING: Retract Bias Detected (Ratio: {bias_ratio:.1f}x).")
            print("   -> Model is significantly more accurate on Retract (Easy) than Grasp (Hard).")
    
    final_err = overall_row['Pos Mean (cm)']
    if final_err < 1.70:
        print(f"🏆 VERDICT: SOTA PERFORMANCE ({final_err:.2f}cm). Publish immediately.")
    elif final_err < 2.20:
        print(f"⚖️ VERDICT: COMPETITIVE ({final_err:.2f}cm). Strong baseline.")
    else:
        print(f"📉 VERDICT: NEEDS TUNING ({final_err:.2f}cm).")

    print("="*80 + "\n")

if __name__ == "__main__":
    main()