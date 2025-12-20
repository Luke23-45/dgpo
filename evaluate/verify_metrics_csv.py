import pandas as pd
import sys
from pathlib import Path

def verify_csv(file_path):
    print(f"Verifying {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"ERROR: File {file_path} not found.")
        sys.exit(1)
        
    required_columns = [
        "iteration", "timestamp", "mean_reward", "success_rate", "n_episodes",
        "policy_loss", "value_loss", "bc_loss", "shadow_pos_div_cm", 
        "shadow_orn_div_rad", "grip_agreement", "total_steps", "blend_alpha",
        "exec_pos_div", "lr_policy", "lr_value", "entropy", "kl_div", "kl_beta",
        "grad_norm_p", "grad_norm_v", "explained_var", "clip_frac", "adv_mean",
        "loss_ppo", "loss_value", "loss_bc", "loss_entropy", "loss_phase", "loss_smooth"
    ]
    
    missing_cols = [c for c in required_columns if c not in df.columns]
    
    if missing_cols:
        print(f"FAILED: Missing columns: {missing_cols}")
        sys.exit(1)
        
    print("SUCCESS: All required columns present.")
    print("Preview of data:")
    print(df.head())
    
    # Check for empty values
    if df.isnull().values.any():
        print("WARNING: Dataset contains NaN values.")
    else:
        print("SUCCESS: No NaN values found.")

if __name__ == "__main__":
    verify_csv("c:/Users/Hellx/Documents/Programming/python/Project/redhot/logs/metrics_test.csv")
