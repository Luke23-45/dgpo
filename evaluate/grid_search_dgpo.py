# FILE: evaluate/grid_search_dgpo.py
"""
Grid Search for DGPO Policy Evaluation

Systematically searches over IK PID gains to find optimal values
for the DGPO-trained SemanticPlanner policy.

Usage:
    python evaluate/grid_search_dgpo.py --checkpoint outputs/dgpo_runs/dgpo_final.pt
"""

import argparse
import csv
import logging
import subprocess
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("GridSearchDGPO")


def run_evaluation(
    checkpoint: str,
    bc_checkpoint: str,
    ik_kp: float,
    ik_ki: float,
    ik_kd: float,
    n_episodes: int,
    max_steps: int,
    output_dir: str,
    seed: int = 42,
) -> Tuple[float, str]:
    """
    Run a single evaluation with given parameters.
    
    Returns: (success_rate, csv_path)
    """
    cmd = [
        sys.executable,
        "evaluate/evaluate_dgpo.py",
        "--checkpoint", checkpoint,
        "--bc_checkpoint", bc_checkpoint,
        "--ik_kp", str(ik_kp),
        "--ik_ki", str(ik_ki),
        "--ik_kd", str(ik_kd),
        "--n_episodes", str(n_episodes),
        "--max_steps", str(max_steps),
        "--output_dir", output_dir,
        "--seed", str(seed),
    ]
    
    log.info(f"Running: Kp={ik_kp}, Ki={ik_ki}, Kd={ik_kd}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 min timeout
        )
        
        # Parse success rate from output
        for line in result.stdout.split("\n"):
            if "Success Rate:" in line:
                # Extract percentage value
                rate_str = line.split("Success Rate:")[1].strip().replace("%", "")
                return float(rate_str), output_dir
        
        log.warning(f"Could not parse success rate from output")
        return 0.0, output_dir
        
    except subprocess.TimeoutExpired:
        log.error(f"Evaluation timed out")
        return 0.0, output_dir
    except Exception as e:
        log.error(f"Evaluation failed: {e}")
        return 0.0, output_dir


def main():
    parser = argparse.ArgumentParser(description="Grid Search for DGPO Evaluation")
    
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to DGPO checkpoint (.pt file)"
    )
    parser.add_argument(
        "--bc_checkpoint", type=str,
        default="/content/drive/MyDrive/pda/bc/bc_backup_epoch_088.ckpt",
        help="Path to BC checkpoint (for model architecture)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/dgpo_grid_search",
        help="Directory to save grid search results"
    )
    parser.add_argument(
        "--n_episodes", type=int, default=5,
        help="Episodes per configuration"
    )
    parser.add_argument(
        "--max_steps", type=int, default=400,
        help="Max steps per episode (shorter for faster search)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fixed list of parameter configurations (curated from prior experiments)
    combinations = [
        # =====================================================
        # STAGE 1: Baseline High Stiffness (from Unified Planner)
        # =====================================================
        {"ik_kp": 139.0, "ik_ki": 0.1, "ik_kd": 20.0},
        {"ik_kp": 150.0, "ik_ki": 0.1, "ik_kd": 20.0},
        {"ik_kp": 170.0, "ik_ki": 0.1, "ik_kd": 20.0},
        {"ik_kp": 250.0, "ik_ki": 0.1, "ik_kd": 20.0},
        {"ik_kp": 300.0, "ik_ki": 0.2, "ik_kd": 20.0},

        {"ik_kp": 250.0, "ik_ki": 0.1, "ik_kd": 20.0},
        {"ik_kp": 450.0, "ik_ki": 0.1, "ik_kd": 20.0},
        {"ik_kp": 500.0, "ik_ki": 0.2, "ik_kd": 20.0},
        
        # =====================================================
        # STAGE 2: Very High Stiffness
        # =====================================================
        {"ik_kp": 520.0, "ik_ki": 0.1, "ik_kd": 10.0},
        {"ik_kp": 550.0, "ik_ki": 0.1, "ik_kd": 20.0},
        
        # =====================================================
        # STAGE 3: Integral Variants (for steady-state error)
        # =====================================================
        {"ik_kp": 470.0, "ik_ki": 1.0, "ik_kd": 20.0},
        {"ik_kp": 470.0, "ik_ki": 2.0, "ik_kd": 20.0},
        
        # =====================================================
        # STAGE 4: Lower Damping (faster response)
        # =====================================================
        {"ik_kp": 500.0, "ik_ki": 0.2, "ik_kd": 10.0},
        {"ik_kp": 500.0, "ik_ki": 0.2, "ik_kd": 5.0},
    ]
    
    log.info(f"Grid search with {len(combinations)} configurations")
    
    # Results storage
    results = []
    
    # Run grid search
    for i, params in enumerate(combinations):
        log.info(f"[{i+1}/{len(combinations)}] Testing: {params}")
        
        # Create unique output dir for this run
        run_dir = output_dir / f"kp{params['ik_kp']}_ki{params['ik_ki']}_kd{params['ik_kd']}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        success_rate, _ = run_evaluation(
            checkpoint=args.checkpoint,
            bc_checkpoint=args.bc_checkpoint,
            ik_kp=params["ik_kp"],
            ik_ki=params["ik_ki"],
            ik_kd=params["ik_kd"],
            n_episodes=args.n_episodes,
            max_steps=args.max_steps,
            output_dir=str(run_dir),
        )
        
        results.append({
            **params,
            "success_rate": success_rate
        })
        
        log.info(f"Result: Success Rate = {success_rate:.1f}%")
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = output_dir / f"grid_search_results_{timestamp}.csv"
    
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ik_kp", "ik_ki", "ik_kd", "success_rate"])
        writer.writeheader()
        writer.writerows(results)
    
    log.info(f"Results saved to: {results_csv}")
    
    # Find best configuration
    if results:
        best = max(results, key=lambda x: x["success_rate"])
        log.info("=" * 60)
        log.info("GRID SEARCH COMPLETE")
        log.info("=" * 60)
        log.info(f"Best Configuration:")
        log.info(f"  Kp = {best['ik_kp']}")
        log.info(f"  Ki = {best['ik_ki']}")
        log.info(f"  Kd = {best['ik_kd']}")
        log.info(f"  Success Rate = {best['success_rate']:.1f}%")
        log.info("=" * 60)


if __name__ == "__main__":
    main()
