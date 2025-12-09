import argparse
import lmdb
import numpy as np
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Optimizer")

def evaluate_config(advantages, temp, max_w):
    """
    Returns metrics for a specific (Temp, MaxW) configuration.
    """
    # Calculate weights
    raw_weights = np.exp(advantages / temp)
    
    # Calculate Clipping stats
    clipped_weights = np.clip(raw_weights, 0, max_w)
    clip_mask = raw_weights > (max_w * 0.999) # Float tolerance
    clip_ratio = np.mean(clip_mask)
    
    # Statistics
    mean_w = np.mean(clipped_weights)
    std_w = np.std(clipped_weights)
    
    return {
        "temp": temp,
        "max_w": max_w,
        "clip_ratio": clip_ratio,
        "mean_weight": mean_w,
        "std_weight": std_w, # Proxy for "Signal Strength"
        "variance_coeff": std_w / mean_w # Coefficient of Variation
    }

def main():
    parser = argparse.ArgumentParser(description="Find Optimal AWR Parameters")
    parser.add_argument("--db", type=str, default="/content/fresh_data/final_training_set/training_set.lmdb")
    args = parser.parse_args()
    
    db_path = Path(args.db)
    if not db_path.exists():
        logger.error(f"❌ DB not found: {db_path}")
        return

    # 1. Load All Advantages
    logger.info(f"📥 Loading dataset: {db_path.name}...")
    env = lmdb.open(str(db_path), readonly=True, lock=False,subdir=False)
    all_advs = []
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            if b"advantages" in key:
                all_advs.append(np.frombuffer(value, dtype=np.float32))
    env.close()
    
    flat_advs = np.concatenate(all_advs)
    data_std = np.std(flat_advs)
    data_mean = np.mean(flat_advs)
    
    logger.info(f"📊 Dataset Stats: Mean={data_mean:.4f} | StdDev={data_std:.4f}")
    logger.info("-" * 60)

    # 2. Define Search Space
    # Search T around the StdDev (0.8x to 1.5x)
    temps = np.linspace(data_std * 0.8, data_std * 2.0, 20)
    # Search MaxW in reasonable steps
    max_weights = [2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    valid_configs = []

    # 3. Grid Search
    print(f"{'TEMP':<8} | {'MAX_W':<8} | {'CLIP %':<8} | {'MEAN':<8} | {'SIGNAL':<8} | {'STATUS'}")
    print("-" * 65)

    for T in temps:
        for M in max_weights:
            metrics = evaluate_config(flat_advs, T, M)
            
            # --- SELECTION CRITERIA ---
            # 1. Stability: We don't want to clip more than 5% of data (Hard Limit)
            #    exception: if T is very tight, we might clip more, but let's aim for <5%
            is_stable = metrics['clip_ratio'] < 0.05
            
            # 2. Effectiveness: We don't want to clip LESS than 0.1% (Useless cap)
            is_active = metrics['clip_ratio'] > 0.001
            
            # 3. Balance: Mean weight should be reasonable (0.8 - 1.5)
            is_balanced = 0.8 < metrics['mean_weight'] < 1.5
            
            status = ""
            if is_stable and is_active and is_balanced:
                valid_configs.append(metrics)
                status = "✅ Candidate"
            elif not is_stable:
                status = "❌ High Clip"
            elif not is_balanced:
                status = "❌ Bad Mean"
            
            # Only print candidates to keep log clean
            # print(f"{T:<8.2f} | {M:<8.1f} | {metrics['clip_ratio']*100:<7.2f}% | {metrics['mean_weight']:<8.2f} | {metrics['std_weight']:<8.3f} | {status}")

    # 4. Rank Candidates
    # We want the config with the HIGHEST Standard Deviation (Signal Strength)
    # that still passes the Stability checks.
    
    if not valid_configs:
        logger.error("No valid configurations found! Try widening search space.")
        return

    # Sort by std_weight descending
    best_config = sorted(valid_configs, key=lambda x: x['std_weight'], reverse=True)[0]
    
    logger.info("\n🏆 OPTIMAL CONFIGURATION FOUND")
    logger.info("=" * 60)
    logger.info(f"   AWR Temperature:  {best_config['temp']:.4f}")
    logger.info(f"   AWR Max Weight:   {best_config['max_w']:.1f}")
    logger.info("-" * 60)
    logger.info(f"   Expected Clip %:  {best_config['clip_ratio']*100:.2f}%")
    logger.info(f"   Expected Mean W:  {best_config['mean_weight']:.4f}")
    logger.info(f"   Signal Strength:  {best_config['std_weight']:.4f} (Higher is better)")
    logger.info("=" * 60)
    logger.info("Update your opal.txt with these exact values.")

if __name__ == "__main__":
    main()