import argparse
import json
import logging
import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm


try:
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

try:
    from scipy.optimize import minimize_scalar
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Project imports
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.expert_dataset import ExpertTrajectoryDataset

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("AWROptimizerSOTA")

def calculate_ess(weights):
    """Effective Sample Size: (sum w)^2 / sum(w^2)"""
    sum_w = np.sum(weights)
    sum_w2 = np.sum(weights**2)
    if sum_w2 < 1e-9: return 0
    return (sum_w**2) / sum_w2

def calculate_entropy(weights):
    """Shannon Entropy of the normalized weight distribution."""
    sum_w = np.sum(weights) + 1e-9
    p = weights / sum_w
    p = p[p > 1e-9]  # Avoid log(0)
    return -np.sum(p * np.log2(p))

def calculate_average_weight(weights):
    """Average unnormalized weight."""
    return np.mean(weights)

def ess_objective(beta, advantages, target_ess, max_weight=None):
    """
    Objective for beta optimization: difference from target ESS.
    """
    N = len(advantages)
    A = advantages - np.mean(advantages)  # Center advantages
    exp_terms = np.clip(A / beta, -20, 20)
    w = np.exp(exp_terms)
    if max_weight is not None:
        w = np.clip(w, 0, max_weight)
    ess = calculate_ess(w)
    return abs(ess - target_ess)

def optimize_beta(advantages, target_ess_fraction=0.3, max_weight=20.0, method='scipy'):
    """
    Optimizes beta using scipy minimize_scalar to target ESS fraction.
    Falls back to grid search if needed.
    Matches and improves AWR (Peng et al., 2019) heuristics with numerical stability.
    Incorporates weight clipping as per original AWR implementation.
    """
    N = len(advantages)
    target_ess = N * target_ess_fraction
    
    # Center advantages
    A = advantages - np.mean(advantages)
    best_beta = 1.0
    
    # Try SciPy if available
    if method == 'scipy' and SCIPY_AVAILABLE:
        # Use scalar optimization for efficiency
        res = minimize_scalar(
            ess_objective,
            bounds=(1e-3, 1e2),  # Reasonable range for beta
            args=(advantages, target_ess, max_weight),
            method='bounded',
            options={'xatol': 1e-4}
        )
        if res.success:
            best_beta = res.x
        else:
            logger.warning("SciPy optimization failed, falling back to grid search.")
            method = 'grid'
    
    if method == 'grid':
        # Grid search as fallback
        betas = np.logspace(-3, 2, 100)  # Finer grid: 0.001 to 100
        min_diff = float('inf')
        best_beta = 1.0
        for beta in betas:
            exp_terms = np.clip(A / beta, -20, 20)
            w = np.exp(exp_terms)
            if max_weight is not None:
                w = np.clip(w, 0, max_weight)
            ess = calculate_ess(w)
            diff = abs(ess - target_ess)
            if diff < min_diff:
                min_diff = diff
                best_beta = beta
    
    # Compute sweep data for diagnostics
    betas = np.logspace(-3, 2, 50)
    results = []
    for beta in betas:
        exp_terms = np.clip(A / beta, -20, 20)
        w = np.exp(exp_terms)
        if max_weight is not None:
            w = np.clip(w, 0, max_weight)
        ess = calculate_ess(w)
        entropy = calculate_entropy(w)
        avg_w = calculate_average_weight(w)
        results.append({
            'beta': beta,
            'ess': ess,
            'ess_fraction': ess / N if N > 0 else 0,
            'entropy': entropy,
            'avg_weight': avg_w
        })
    
    return best_beta, results

def detect_corruption(advantages, weights, threshold=0.1):
    """
    Detect potential data corruption or poor explorations.
    Inspired by CAWR (Hu et al., 2025): Check fraction of near-zero or negative advantages.
    """
    neg_frac = np.mean(advantages < 0)
    low_w_frac = np.mean(weights < threshold)
    if neg_frac > 0.5 or low_w_frac > 0.7:
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description="AWR Hyperparameter Optimizer (SOTA v2.0)")
    parser.add_argument("--db-path", type=str, required=True, help="Path to LMDB with advantages")
    parser.add_argument("--target-ess", type=float, default=0.3, help="Target ESS fraction (0-1)")
    parser.add_argument("--max-weight", type=float, default=20.0, help="Max weight clipping (as in AWR paper)")
    parser.add_argument("--plot", action="store_true", help="Generate diagnostic plots")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save JSON report")
    args = parser.parse_args()

    try:
        # 1. Load Advantages
        logger.info(f"Loading advantages from {args.db_path}...")
        
        try:
            # Horizon 1,1 ensures we read all timesteps as valid chunks for statistical analysis
            reader = ExpertTrajectoryDataset(args.db_path, observation_horizon=1, action_horizon=1)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return

        all_advs = []
        phase_map = defaultdict(list)
        
        for ep_idx in tqdm(range(len(reader.episode_metadata)), desc="Reading Data"):
            meta = reader.episode_metadata[ep_idx]
            if 'advantages' not in meta['modalities']:
                continue
                
            key = meta['modalities']['advantages']['key']
            blob = reader._get_lmdb_blob(key)
            advs = np.frombuffer(blob, dtype=np.float32).copy()
            all_advs.append(advs)
            
            if 'gt_phase' in meta['modalities']:
                p_meta = meta['modalities']['gt_phase']
                p_blob = reader._get_lmdb_blob(p_meta['key'])
                phases = np.frombuffer(p_blob, dtype=np.int32)
                for t, a in enumerate(advs):
                    phase_map[int(phases[t])].append(a)

        if not all_advs:
            logger.error("No advantages found in the dataset index.")
            return

        flat_advs = np.concatenate(all_advs)
        N = len(flat_advs)
        logger.info(f"Analyzing {N} samples...")

        # 2. Optimize Beta
        best_beta, sweep_data = optimize_beta(
            flat_advs, 
            target_ess_fraction=args.target_ess,
            max_weight=args.max_weight
        )
        
        # Compute final weights for reporting
        A = flat_advs - np.mean(flat_advs)
        exp_terms = np.clip(A / best_beta, -20, 20)
        final_w = np.exp(exp_terms)
        final_w = np.clip(final_w, 0, args.max_weight)
        final_ess = calculate_ess(final_w)
        final_entropy = calculate_entropy(final_w)
        final_avg_w = calculate_average_weight(final_w)
        
        # 3. Report
        report = {
            "suggested_beta": best_beta,
            "ess_fraction": final_ess / N,
            "entropy": final_entropy,
            "avg_weight": final_avg_w,
            "adv_std": float(np.std(flat_advs)),
            "adv_mean": float(np.mean(flat_advs)),
            "effective_samples": int(final_ess),
            "total_samples": N
        }
        
        logger.info("\n" + "="*80)
        logger.info("AWR HYPERPARAMETER OPTIMIZATION REPORT (SOTA v2.0)")
        logger.info("="*80)
        logger.info(f"Advantage Mean / Std: {report['adv_mean']:.4f} / {report['adv_std']:.4f}")
        logger.info(f"Suggested Beta:       {best_beta:.4f} (ESS-Optimized with Clipping)")
        logger.info(f"Average Weight:       {final_avg_w:.4f} (Target ~0.5 for balance)")
        logger.info(f"Effective Samples:    {report['effective_samples']} / {N} (Fraction: {report['ess_fraction']:.3f})")
        logger.info(f"Weight Entropy:       {final_entropy:.4f} bits")
        logger.info("-" * 80)
        
        # 4. Phase Bias Check
        if phase_map:
            logger.info("Phase Weighting Audit (at Suggested Beta):")
            logger.info(f"{'PHASE':<15} | {'MEAN WEIGHT':<15} | {'MAX WEIGHT':<15} | {'ESS FRACTION':<15}")
            phase_report = {}
            for p_idx in sorted(phase_map.keys()):
                p_advs = np.array(phase_map[p_idx])
                p_A = p_advs - np.mean(flat_advs)  # Use global mean
                p_exp = np.clip(p_A / best_beta, -20, 20)
                p_w = np.exp(p_exp)
                p_w = np.clip(p_w, 0, args.max_weight)
                p_ess = calculate_ess(p_w)
                p_N = len(p_advs)
                logger.info(f"{p_idx:<15} | {np.mean(p_w):15.4f} | {np.max(p_w):15.4f} | {p_ess / p_N if p_N > 0 else 0:15.3f}")
                phase_report[p_idx] = {
                    "mean_weight": float(np.mean(p_w)),
                    "max_weight": float(np.max(p_w)),
                    "ess_fraction": float(p_ess / p_N) if p_N > 0 else 0
                }
            report["phase_audit"] = phase_report
        
        # 5. Corruption Detection (Inspired by CAWR)
        is_corrupted = detect_corruption(flat_advs, final_w)
        if is_corrupted:
            logger.warning("Potential data corruption detected: High fraction of negative advantages or low weights.")
            logger.info("Recommendation: Consider robust losses (e.g., Huber) or prioritized sampling as in CAWR.")
        else:
            logger.info("Data quality appears good: Low corruption risk.")
        
        # 6. Training Strategy Suggestions
        logger.info("-" * 80)
        logger.info("Training Strategy Suggestions:")
        if best_beta < 0.1:
            logger.info("Very Sharp Temperature: Extremely selective. Use tiny LR (e.g., 1e-5) and monitor for instability.")
        elif best_beta < 0.5:
            logger.info("Sharp Temperature: Highly selective. Use smaller LR (e.g., 5e-5) to prevent gradient explosions.")
        else:
            logger.info("Soft Temperature: Diverse contributions. Standard LR (1e-4 to 3e-4) should be stable.")
        
        if final_avg_w < 0.3 or final_avg_w > 0.7:
            logger.info(f"Note: Average weight {final_avg_w:.2f} deviates from ~0.5. Consider adjusting target-ess or max-weight.")
        
        if is_corrupted:
            logger.info("For corrupted data: Implement advantage-based prioritized replay and robust loss functions.")
        
        # 7. Save JSON Report
        if args.output_json:
            with open(args.output_json, 'w') as f:
                json.dump(report, f, indent=4)
            logger.info(f"JSON report saved to {args.output_json}")

        # 8. Plotting
        if args.plot:
            if not PLOT_AVAILABLE:
                logger.warning("matplotlib not found. Skipping plots.")
                return

            logger.info("Generating diagnostic plots...")
            betas = [d['beta'] for d in sweep_data]
            ess_fracs = [d['ess_fraction'] for d in sweep_data]
            entropies = [d['entropy'] for d in sweep_data]
            avg_ws = [d['avg_weight'] for d in sweep_data]
            
            plt.figure(figsize=(15, 10))
            
            # ESS Plot
            plt.subplot(2, 2, 1)
            plt.semilogx(betas, ess_fracs, 'b-o', label='ESS Fraction')
            plt.axhline(y=args.target_ess, color='r', linestyle='--', label='Target ESS')
            plt.axvline(x=best_beta, color='g', linestyle='--', label=f'Best Beta ({best_beta:.2f})')
            plt.xlabel("Beta (Temperature)")
            plt.ylabel("Effective Sample Fraction")
            plt.title("ESS Sweep")
            plt.grid(True, which="both", ls="-", alpha=0.5)
            plt.legend()
            
            # Entropy Plot
            plt.subplot(2, 2, 2)
            plt.semilogx(betas, entropies, 'm-s', label='Entropy')
            plt.xlabel("Beta (Temperature)")
            plt.ylabel("Shannon Entropy (bits)")
            plt.title("Information Density Sweep")
            plt.grid(True, which="both", ls="-", alpha=0.5)
            
            # Average Weight Plot
            plt.subplot(2, 2, 3)
            plt.semilogx(betas, avg_ws, 'c-^', label='Avg Weight')
            plt.axhline(y=0.5, color='r', linestyle='--', label='Target Avg ~0.5')
            plt.axvline(x=best_beta, color='g', linestyle='--')
            plt.xlabel("Beta (Temperature)")
            plt.ylabel("Average Weight")
            plt.title("Average Weight Sweep")
            plt.grid(True, which="both", ls="-", alpha=0.5)
            plt.legend()
            
            # Advantage Histogram
            plt.subplot(2, 2, 4)
            plt.hist(flat_advs, bins=50, color='skyblue', edgecolor='black')
            plt.title("Advantage Distribution")
            plt.xlabel("Advantage Value")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.5)
            
            plot_path = Path("awr_optimization_diagnostics_sota.png")
            plt.tight_layout()
            plt.savefig(plot_path)
            logger.info(f"Diagnostic plot saved to {plot_path.absolute()}")
    finally:
        if 'reader' in locals():
            reader.close_env()

if __name__ == "__main__":
    main()