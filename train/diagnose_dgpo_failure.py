
"""
DGPO Deep Diagnostic Tool
=========================
This script performs a forensic analysis of the Pre-trained Policy and the DGPO Environment
to definitively confirm the root cause of the "0% Success / 16cm Error" failure mode.

It tests three core hypotheses:
1. "The Hovering Bug": Does the policy output the Current Pose (Identity Function)?
2. "The Phantom Expert": Is the Scripted Expert actually generating moving targets?
3. "The Scale Mismatch": Are inputs entering the network with the correct Order of Magnitude?

Usage:
    python train/diagnose_dgpo_failure.py
"""

import os
import sys
import logging
import numpy as np
import torch
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
# from termcolor import colored # REMOVED due to env issues

# Inject Root Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.panda_env import PandaEnv
from envs.dgpo_env_wrapper import DGPOEnvWrapper
from train.train_semantic_planner import SemanticPlannerLightningModule

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("DIAGNOSIS")

def calc_dist(a, b):
    return np.linalg.norm(a[:3] - b[:3])

@hydra.main(config_path="../configs", config_name="train_dgpo_config", version_base="1.2")
def diagnose(cfg: DictConfig):
    print("\n=== DGPO FORENSIC DIAGNOSTIC TOOL ===")
    print(f"Loading Configuration from: configs/train_dgpo_config.yaml")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Environment
    print("\n[1] Initializing Environment...")
    try:
        raw_env = PandaEnv(
            xml_path=cfg.environment.xml_path,
            control_mode="delta",
            render_mode="rgb_array" # Need visuals for policy
        )
        env = DGPOEnvWrapper(raw_env, cfg)
        print("    Success: DGPOEnvWrapper initialized.")
    except Exception as e:
        print(f"    FATAL: Environment failed to load: {e}")
        return

    # 2. Load Policy
    print("\n[2] Loading Policy Checkpoint...")
    ckpt_path = cfg.bc_checkpoint
    print(f"    Target: {ckpt_path}")
    
    if not os.path.exists(ckpt_path):
        print(f"    FATAL: Checkpoint not found at {ckpt_path}")
        return

    try:
        pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
            ckpt_path, map_location=device, strict=True
        )
        policy = pl_module.model.to(device)
        policy.eval()
        print("    Success: Policy loaded (Strict Mode).")
    except Exception as e:
        print(f"    FATAL: Policy load failed: {e}")
        return

    # 3. Diagnostic Loop
    print("\n[3] Running Forensic Rollout (1 Episode)...")
    
    obs, info = env.reset()
    
    # Trackers
    proprio_stats = []
    hovering_counts = 0
    expert_movement = 0.0
    policy_movement = 0.0
    ep_steps = 100 # Analyis duration
    
    print("\n    | Step | Phase | Current EE (World) | Expert Tgt (World) | Policy Pred (World) | Err(P-E) | Err(P-C) | Err(E-C) |")
    print("    |------|-------|--------------------|--------------------|---------------------|----------|----------|----------|")

    for step in range(ep_steps):
        # Prepare Batch
        # 1. Resize to 224x224 (SigLIP Requirement)
        curr_img_t = torch.from_numpy(obs['image_primary']).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0
        curr_img = torch.nn.functional.interpolate(curr_img_t, size=(224, 224), mode='bicubic', align_corners=False)
        # 2. Normalize [-1, 1]
        curr_img = (curr_img - 0.5) / 0.5
        
        # Proprio (RAW)
        proprio_raw = obs['proprio']
        proprio_stats.append(proprio_raw)
        proprio_t = torch.from_numpy(proprio_raw).float().unsqueeze(0).to(device)
        
        # Prev / Goal (Dummy for diag if not available, but logic usually requires them)
        # We reuse curr for prev on step 0, or cache it
        if step == 0:
            prev_img = curr_img
        
        # Goal Img check
        if 'goal_img' in info and getattr(info['goal_img'], 'shape', None) == (256, 256, 3):
             goal_img_t = torch.from_numpy(info['goal_img']).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0
             goal_img = torch.nn.functional.interpolate(goal_img_t, size=(224, 224), mode='bicubic', align_corners=False)
             goal_img = (goal_img - 0.5) / 0.5
        else:
             # Fallback to zero goal if missing (rare in wrapper)
             goal_img = torch.zeros_like(curr_img)

        # Batch Dict
        batch = {
            "prev_image": prev_img,
            "curr_image": curr_img,
            "goal_image": goal_img,
            "curr_proprio": proprio_t
        }

        # Inference
        with torch.no_grad():
            output = policy(batch)
            pred_chunk = output['pose_chunk'].cpu().numpy()[0] # (K, 7)
        
        # Analysis Data
        current_ee = env.get_ee_pose()
        expert_target = info['expert_pose']
        pred_step0 = pred_chunk[0] # Immediate action
        
        # Distances
        dist_pol_expert = calc_dist(pred_step0, expert_target)
        dist_pol_curr   = calc_dist(pred_step0, current_ee)
        dist_exp_curr   = calc_dist(expert_target, current_ee)
        
        # Logging Row
        phase = info.get('expert_phase', 'UNK')
        
        # Formatting
        row_str = f"    | {step:4d} | {phase:15s} | {np.array2string(current_ee[:3], precision=2)} | {np.array2string(expert_target[:3], precision=2)} | {np.array2string(pred_step0[:3], precision=2)} | {dist_pol_expert*100:5.1f}cm | {dist_pol_curr*100:5.1f}cm | {dist_exp_curr*100:5.1f}cm |"
        
        # Check Anomalies
        if dist_pol_curr < 0.01 and dist_exp_curr > 0.05:
            row_str += " << HOVERING"
        
        print(row_str)
        
        # Accumulate metrics
        expert_movement += dist_exp_curr
        policy_movement += dist_pol_curr
        
        # Update State (Shadow Mode - Use EXPERT to step)
        dummy_action = np.zeros(8)
        obs, reward, terminated, truncated, info = env.step(dummy_action)
        prev_img = curr_img
        
        if terminated or truncated:
            break

    # 4. Final Verdict
    print("\n=== DIAGNOSTIC VERDICT ===")
    
    # Check 1: Input Scaling
    proprio_stats = np.array(proprio_stats)
    p_mean = np.mean(proprio_stats)
    p_std = np.std(proprio_stats)
    print(f"Proprioception Statistics: Mean={p_mean:.4f}, Std={p_std:.4f}")
    if abs(p_mean) < 0.1 and abs(p_std - 1.0) < 0.2:
        print("    WARN: Inputs look Normalized (Mean~0, Std~1). If Model expects RAW, this is bad.") 
    else:
        print("    INFO: Inputs look RAW (Mean!=0, Std!=1).")

    # Check 2: Expert Vitality
    if expert_movement < 0.05: # Total movement over 30 steps
        print(f"    FAIL: Expert is DEAD. Total dist from EE: {expert_movement:.3f}m")
        print("          Root Cause: Expert Script or wrapping failure.")
    else:
        print(f"    PASS: Expert is ALIVE. Divergence from EE: {expert_movement:.3f}m")

    # Check 3: The Hovering Bug
    print(f"Hovering Frames Detected: {hovering_counts} / {step+1}")
    if hovering_counts > (step // 2):
        print("    CRITICAL FAIL: POLICY IS HOVERING (Identity Function).")
        print("    Evidence: Policy predicted 'Current EE' while Expert demanded movement.")
        print("    ROOT CAUSE: Training Dataset labeled 'expert_target_pose' as 'ee_pose'.")
        print("    ACTION: Regenerate Dataset with correct 'expert_target_pose' keys or logic.")
    elif dist_pol_expert < 0.05: # 5cm avg error
        print("    PASS: Policy is closely tracking Expert.")
    else:
        print("    FAIL: High Error (Policy != Expert), but NOT Hovering.")
        print("          Possible Causes: Visual Domain Gap, Coordinate Frame Mismatch, or Under-trained.")

if __name__ == "__main__":
    diagnose()
