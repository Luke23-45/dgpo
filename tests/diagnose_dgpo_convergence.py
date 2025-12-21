import os
import sys
import torch
import hydra
import logging
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

# Add project root to path
sys.path.append(os.getcwd())

from train.train_dgpo_robust import DGPOTrainer

# Setup simplistic logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("DIAGNOSE")

@hydra.main(config_path="./configs", config_name="train_dgpo_config", version_base="1.2")
def main(cfg):
    print("\n==================================================")
    print(" 🏥 DGPO CONVERGENCE DIAGNOSTIC TOOL")
    print("==================================================\n")

    # 1. Override Options for Diagnostics
    cfg.num_envs = 1 # Single thread for clarity
    cfg.training.use_policy_blending = True # We will force alpha=0 manually
    
    # 2. Determine Checkpoint
    # Allows passing 'checkpoint=path/to/ckpt.pt' via command line
    # Hydra handles this if passed as argument
    print(f"Loading Checkpoint: {cfg.checkpoint.resume_from}")
    print(f"Config: rampup={cfg.training.blend_rampup_iters}")
    
    # 3. Initialize Trainer
    try:
        trainer = DGPOTrainer(cfg)
    except Exception as e:
        print(f"Failed to init Trainer: {e}")
        return

    # 4. Load the Checkpoint
    ckpt_path = cfg.checkpoint.resume_from
    if ckpt_path is None or str(ckpt_path).lower() == "null" or str(ckpt_path).lower() == "none":
        print("ℹ️ No resume checkpoint provided. Using Initialized Policy (BC Baseline).")
    elif not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found at: {ckpt_path}")
        print("Please assume 'Fresh Start' baseline if not found.")
        # We can simulate a fresh policy check too
    else:
        trainer._load_checkpoint(ckpt_path)
        print("✅ Checkpoint Loaded Successfully.")

    # 5. FORCE SHADOW MODE
    # We want to see if the policy *knows* what to do when watching the expert.
    # Alpha = 0.0 -> Expert acts, Policy predicts.
    # We check 'shadow_pos_div' (Prediction Error).
    print("\n[TEST 1] SHADOW MODE EVALUATION (Alpha=0.0)")
    print("Running 5 episodes to measure 'Shadow Divergence'...")
    
    # Hack the blending schedule to return 0.0 always
    trainer.blending_schedule.get_alpha = lambda x: 0.0
    
    # Run Collection
    rollout_stats = trainer.collect_rollouts(n_steps=1000) # ~5 episodes
    
    pos_div = rollout_stats['shadow_pos_div'] * 100 # cm
    orn_div = rollout_stats['shadow_orn_div']
    
    print("\n--------------------------------------------------")
    print(f"📊 RESULTS: Shadow Divergence (Prediction Error)")
    print(f"   Position Error: {pos_div:.2f} cm")
    print(f"   Rotation Error: {orn_div:.4f} rad")
    print("--------------------------------------------------")
    
    # 6. Diagnosis
    if pos_div < 3.5:
        print("✅ PASS: The Policy is HEALTHY.")
        print("   Explanation: It can clone the expert within <3.5cm error.")
        print("   Root Cause of Failure: The 'Alpha Ramp' was too fast.")
        print("   Fix: The config tuning (rampup=300) WAS correct.")
    elif pos_div < 6.0:
        print("⚠️ WARNING: The Policy is WEAK.")
        print("   Explanation: 3.5cm - 6.0cm error is borderline.")
        print("   Recommendation: Resume training with slower rampup.")
    else:
        print("❌ FAIL: The Policy is BROKEN.")
        print(f"   Explanation: {pos_div:.2f}cm error is too high for Shadow Mode.")
        print("   Root Cause: Weights are corrupted, normalization is off, or the 'Ghost Penalty' ruined it.")
        print("   Recommendation: START FRESH. This checkpoint is unrecoverable.")

    # 7. Check Normalizer Logic
    mean = trainer.reward_normalizer.return_rms.mean
    var = trainer.reward_normalizer.return_rms.var
    print(f"\n[TEST 2] Normalizer Stats: Mean={mean:.4f}, Var={var:.4f}")
    if var < 1e-4:
        print("⚠️ Warning: Normalizer variance is tiny. Rewards might be scaling explosively.")

if __name__ == "__main__":
    main()
