# FILE: scripts/verify_awr_params.py
import argparse
import lmdb
import numpy as np
import logging
import sys
from collections import defaultdict
from pathlib import Path

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("AWR_Sim")

PHASE_MAP = {
    0: "0_Approach",
    1: "1_Grasp",
    2: "2_Transport",
    3: "3_Place",
    4: "4_Retract"
}

def get_weight_stats(advantages, temp, max_weight):
    """
    Simulates AWR weight calculation: w = clamp(exp(A / T), max=M)
    """
    # 1. Scale
    scaled_adv = advantages / temp
    
    # 2. Exponentiate
    weights = np.exp(scaled_adv)
    
    # 3. Stats BEFORE clipping (to see raw demand)
    raw_mean = np.mean(weights)
    raw_max = np.max(weights)
    
    # 4. Clip
    clipped_weights = np.clip(weights, 0, max_weight)
    
    # 5. Calculate "Clipping Ratio" (How much data hits the ceiling?)
    # We check how many samples are within 1% of the max weight
    n_clipped = np.sum(clipped_weights >= (max_weight * 0.99))
    clip_pct = (n_clipped / len(weights)) * 100
    
    return clipped_weights, raw_mean, raw_max, clip_pct

def main():
    parser = argparse.ArgumentParser(description="AWR Parameter Verification Simulation")
    parser.add_argument("--db", type=str, default=r"C:\Users\Hellx\Documents\Programming\python\Project\redhot\data\validation\training_set.lmdb")
    # PROPOSED PARAMETERS TO TEST
    parser.add_argument("--temp", type=float, default=6.0, help="Proposed Temperature")
    parser.add_argument("--max_w", type=float, default=2.5, help="Proposed Max Weight")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        logger.error(f"❌ Database not found: {db_path}")
        sys.exit(1)

    logger.info(f"🔍 ANALYZING DATASET: {db_path}")
    logger.info(f"⚙️  TESTING CONFIG: Temp={args.temp} | Max_Weight={args.max_w}")
    logger.info("-" * 60)

    # 1. Load Data
    env = lmdb.open(str(db_path), readonly=True, lock=False,subdir=False)
    
    all_advs = []
    all_phases = []
    
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            # We need both advantages and phases to check phase bias
            if b"advantages" in key:
                adv = np.frombuffer(value, dtype=np.float32)
                all_advs.append(adv)
            elif b"gt_phase" in key:
                # Phases are often stored as ints or floats
                ph = np.frombuffer(value, dtype=np.int32)
                # Fallback if stored as float
                if ph.size == 0 or ph.nbytes != len(value):
                     ph = np.frombuffer(value, dtype=np.float32).astype(np.int32)
                all_phases.append(ph)

    # Flatten
    flat_advs = np.concatenate(all_advs)
    flat_phases = np.concatenate(all_phases)
    
    if len(flat_advs) != len(flat_phases):
        logger.warning(f"⚠️ Length Mismatch! Adv: {len(flat_advs)}, Phase: {len(flat_phases)}")
        min_len = min(len(flat_advs), len(flat_phases))
        flat_advs = flat_advs[:min_len]
        flat_phases = flat_phases[:min_len]

    # 2. Global Statistics
    weights, raw_mean, raw_max, clip_pct = get_weight_stats(flat_advs, args.temp, args.max_w)
    
    logger.info(f"📊 GLOBAL STATISTICS")
    logger.info(f"   Mean Weight:      {np.mean(weights):.4f}  (Ideal: ~1.0)")
    logger.info(f"   Std Dev:          {np.std(weights):.4f}   (Lower is more stable)")
    logger.info(f"   Max Weight (Raw): {raw_max:.4f}   (If > 100, Temp is too low)")
    logger.info(f"   Clipping Ratio:   {clip_pct:.2f}%     (Should be < 5%)")
    
    if clip_pct > 10.0:
        logger.error("❌ CRITICAL: Over 10% of your data is being clipped. Raise Temperature or Max Weight.")
    elif clip_pct > 5.0:
        logger.warning("⚠️ WARNING: High clipping. AWR is acting like a step function.")
    else:
        logger.info("✅ Clipping is healthy.")

    logger.info("-" * 60)
    logger.info(f"⚖️  PHASE BALANCE REPORT (The 'Fighting' Check)")
    logger.info(f"{'PHASE':<15} | {'MEAN ADV':<10} | {'MEAN WEIGHT':<12} | {'MULTIPLIER'}")
    logger.info("-" * 60)

    # 3. Phase-wise Analysis
    phase_weights = defaultdict(list)
    phase_advs = defaultdict(list)
    
    for w, p, a in zip(weights, flat_phases, flat_advs):
        phase_weights[p].append(w)
        phase_advs[p].append(a)
        
    sorted_phases = sorted(phase_weights.keys())
    
    # We use Approach (Phase 0) as the baseline (1.0x)
    base_weight = np.mean(phase_weights.get(0, [1.0]))
    
    for p in sorted_phases:
        p_name = PHASE_MAP.get(p, f"Phase_{p}")
        w_mean = np.mean(phase_weights[p])
        a_mean = np.mean(phase_advs[p])
        
        # How much stronger is this phase compared to Approach?
        multiplier = w_mean / base_weight if base_weight > 0 else 0
        
        status = ""
        if multiplier > 2.0: status = "⚠️ DOMINATING"
        elif multiplier < 0.5: status = "⚠️ IGNORED"
        
        logger.info(f"{p_name:<15} | {a_mean:+.4f}     | {w_mean:.4f}       | {multiplier:.2f}x {status}")

    logger.info("-" * 60)
    
    # 4. Final Verdict
    if np.mean(weights) < 0.5 or np.mean(weights) > 2.0:
        logger.warning("❌ UNSTABLE: Global mean weight is far from 1.0. Adjust Temp.")
    else:
        logger.info("✅ STABLE: Global mean weight is healthy.")
        
    if "DOMINATING" in status:
        logger.warning("❌ BIASED: One phase is dominating gradients. Increase Temp.")
    else:
        logger.info("✅ BALANCED: No phase is overpowering the others.")

if __name__ == "__main__":
    main()