#!/bin/bash
# run_dgpo_experiments.sh
# Automated Experiment Supervisor

# PATHS (Update these!)
BC_CKPT="/content/drive/MyDrive/pda/bc/bc_backup_epoch_088.ckpt"
AWR_CKPT="/content/drive/MyDrive/pda/awr/awr_best.ckpt" # Update if different
LOG_BASE="/content/drive/MyDrive/pda/experiments"

echo "=============================================="
echo "      DGPO EXPERIMENT SUITE v1.0"
echo "=============================================="

# 1. BASELINE EVALUATION
echo "[Stage A] Evaluating Baselines..."
# python evaluate_policy.py --ckpt "$BC_CKPT" --episodes 100 > "$LOG_BASE/eval_bc_result.txt"
# python evaluate_policy.py --ckpt "$AWR_CKPT" --episodes 100 > "$LOG_BASE/eval_awr_result.txt"

# 2. DGPO TRAINING (FROM AWR) - PRIORITY 1
echo "[Stage B1] Training DGPO (Prior: AWR)..."
# Override config on the fly using Hydra syntax or manual update
# python -m train.train_dgpo_robust \
#   bc_checkpoint="$AWR_CKPT" \
#   logging.csv_log_name="metrics_dgpo_awr.csv" \
#   checkpoint.save_dir="outputs/dgpo_awr"

# 3. DGPO TRAINING (FROM BC) - PRIORITY 2
echo "[Stage B2] Training DGPO (Prior: BC)..."
# python -m train.train_dgpo_robust \
#   bc_checkpoint="$BC_CKPT" \
#   logging.csv_log_name="metrics_dgpo_bc.csv" \
#   checkpoint.save_dir="outputs/dgpo_bc"

echo "=============================================="
echo "All experiments queued. Check logs in $LOG_BASE"
echo "=============================================="
