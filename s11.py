"""
AWR Calibration Tool.
Reads the computed advantages from your LMDB and suggests optimal 
Temperature (tau) and Max Weight parameters to prevent training collapse.
"""
import lmdb
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt # Optional, for histogram if you want

def calibrate(lmdb_path):
    print(f"--- Calibrating AWR for: {lmdb_path} ---")
    
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False)
    
    all_advantages = []
    
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in tqdm(cursor, desc="Reading Advantages"):
            key_str = key.decode('ascii')
            if key_str.endswith('_advantages'):
                # Read raw bytes -> float32 numpy
                adv = np.frombuffer(value, dtype=np.float32)
                all_advantages.append(adv)
                
    env.close()
    
    if not all_advantages:
        print("ERROR: No advantages found in LMDB. Did you run advantage_calculator.py?")
        return

    # Concatenate all to get global stats
    flat_adv = np.concatenate(all_advantages)
    
    mean_adv = np.mean(flat_adv)
    std_adv = np.std(flat_adv)
    min_adv = np.min(flat_adv)
    max_adv = np.max(flat_adv)
    
    print(f"\n[Data Statistics]")
    print(f"Count: {len(flat_adv)}")
    print(f"Mean:  {mean_adv:.4f}")
    print(f"Std:   {std_adv:.4f}")
    print(f"Min:   {min_adv:.4f}")
    print(f"Max:   {max_adv:.4f}")
    
    print(f"\n[Calibration Analysis]")
    print("-" * 60)
    print(f"{'Temp (tau)':<12} | {'Mean Weight':<12} | {'Max Weight (Raw)':<18} | {'Suggestion'}")
    print("-" * 60)

    # We test temperatures based on fractions of the Standard Deviation
    test_taus = [
        std_adv * 2.0,   # Very Conservative (BC-like)
        std_adv * 1.0,   # Standard AWR
        std_adv * 0.5,   # Aggressive
        std_adv * 0.2,   # Very Aggressive
        0.5,             # The default guess
        0.1              # The aggressive guess
    ]
    test_taus = sorted(list(set(test_taus)), reverse=True) # Remove duplicates

    recommended_config = None

    for tau in test_taus:
        if tau < 1e-6: continue
        
        # Calculate weights: w = exp(A / tau)
        # Note: We usually normalize A by mean for calculation if not already centered
        # But AWR formula is exp(A/tau). Let's see raw impact.
        
        # To avoid overflow in printing, we check exponents first
        max_exponent = max_adv / tau
        
        if max_exponent > 80: # exp(80) is huge
            weight_max_str = "EXPLODES (NaN)"
            weight_mean_str = "N/A"
            note = "Too Unstable"
        else:
            weights = np.exp(flat_adv / tau)
            w_mean = np.mean(weights)
            w_max = np.max(weights)
            weight_max_str = f"{w_max:.2f}"
            weight_mean_str = f"{w_mean:.2f}"
            
            if w_max < 5.0:
                note = "Too Flat (BC-like)"
            elif w_max > 1000.0:
                note = "Highly Selective"
            elif w_max > 1e6:
                note = "Unstable"
            else:
                note = "balanced"
                # Pick the most aggressive one that doesn't explode (>1000 is okay if clipped)
                if recommended_config is None or (w_max < 5000):
                     recommended_config = (tau, 20.0 if w_max > 20 else w_max)

        print(f"{tau:<12.4f} | {weight_mean_str:<12} | {weight_max_str:<18} | {note}")

    print("-" * 60)
    
    if recommended_config:
        rec_tau, rec_max = recommended_config
        print(f"\n>>> RECOMMENDATION:")
        print(f"awr_temperature: {rec_tau:.4f}")
        print(f"awr_max_weight:  {20.0}") 
        print(f"(This sets tau approx equal to your data's Std Dev)")
    else:
        print("\n>>> RECOMMENDATION: Data requires manual review. Returns vary too wildly.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, required=True, help="Path to LMDB")
    args = parser.parse_args()
    
    calibrate(args.db)