import numpy as np
import matplotlib.pyplot as plt

def test_awr_distribution():
    # 1. Simulate your specific data distribution
    # Based on your logs: Mean ~0, Std ~8.15
    N_SAMPLES = 100_000
    adv_mean = 0.0
    adv_std = 8.15
    
    print(f"--- Simulating {N_SAMPLES} samples (Mean={adv_mean}, Std={adv_std}) ---")
    advantages = np.random.normal(adv_mean, adv_std, N_SAMPLES)
    
    # 2. Define Configuration Candidates
    configs = [
        {"tau": 4.87, "max_w": 20.0, "name": "Old Config (Unstable)"},
        {"tau": 5.0,  "max_w": 5.0,  "name": "Proposed Fix (Stable)"},
        {"tau": 8.0,  "max_w": 5.0,  "name": "Alternative (Smoother)"},
    ]
    
    print(f"{'Name':<25} | {'Mean W':<8} | {'Max W':<8} | {'% Clipped':<10} | {'Impact'}")
    print("-" * 80)
    
    for cfg in configs:
        tau = cfg['tau']
        max_w = cfg['max_w']
        
        # Calculate Weights: w = min(exp(A/tau), max_w)
        # We clip the exponent first for numerical safety, then apply max_w
        unclipped_weights = np.exp(advantages / tau)
        weights = np.clip(unclipped_weights, 0, max_w)
        
        # Calculate Stats
        mean_w = np.mean(weights)
        actual_max_w = np.max(weights)
        
        # Count how many hit the ceiling (within small epsilon)
        n_clipped = np.sum(weights >= (max_w - 0.01))
        pct_clipped = (n_clipped / N_SAMPLES) * 100
        
        print(f"{cfg['name']:<25} | {mean_w:.2f}     | {actual_max_w:.2f}     | {pct_clipped:.2f}%     |", end="")
        
        if pct_clipped > 20.0:
            print(" ⚠️ Too Indiscriminate (Caps too early)")
        elif actual_max_w > 15.0:
             print(" ⚠️ High Variance (Risk of Explosion)")
        else:
             print(" ✅ Balanced")

if __name__ == "__main__":
    test_awr_distribution()