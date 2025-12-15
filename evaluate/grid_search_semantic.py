# FILE: evaluate/grid_search_semantic.py
# Grid Search for Semantic Planner IK PID Parameters

"""
Grid Search Script for Semantic Planner PID Tuning

This script systematically searches over IK PID parameters (Kp, Ki, Kd)
to find optimal values for the Semantic Planner evaluation.

Based on Unified Planner findings:
- High Kp values (400-550) are needed for responsive movement
- Ki should be low (0.1-2.0) to avoid instability
- Kd should be moderate (10-20) for damping

Usage:
    python evaluate/grid_search_semantic.py --checkpoint /path/to/model.ckpt --n_episodes 3
    
    # With custom max steps for faster iteration:
    python evaluate/grid_search_semantic.py --checkpoint /path/to/model.ckpt --n_episodes 2 --max_steps 200
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from omegaconf import OmegaConf

# Add root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate.evaluate_semantic_planner_v2 import SemanticPlannerEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("GridSearch_Semantic")


def main():
    parser = argparse.ArgumentParser(description="Grid Search for Semantic Planner IK PID Parameters")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to semantic planner checkpoint")
    parser.add_argument("--config", type=str, default="configs/eval_semantic_planner_v2_config.yaml",
                        help="Path to base config file")
    parser.add_argument("--n_episodes", type=int, default=3, 
                        help="Episodes per parameter configuration")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum steps per episode (override config)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/grid_search_semantic",
                        help="Directory to save results")
    args = parser.parse_args()
    
    # Load base config
    config_path = ROOT / args.config
    if not config_path.exists():
        log.error(f"Config file not found: {config_path}")
        log.info("Creating minimal config for grid search...")
        cfg = OmegaConf.create({
            "checkpoint": args.checkpoint,
            "output_dir": args.output_dir,
            "env": {
                "xml_path": "envs/panda_pick_place.xml",
                "urdf_path": "urdf/panda_mujoco_kinematics.urdf"
            },
            "n_episodes": args.n_episodes,
            "seed": args.seed,
            "max_steps": args.max_steps or 400,
            "success_threshold": 0.05,
            "success_duration_steps": 10,
            "ik_kp": 400.0,
            "ik_ki": 0.1,
            "ik_kd": 20.0
        })
    else:
        cfg = OmegaConf.load(config_path)
    
    # Override with command line args
    cfg.checkpoint = args.checkpoint
    cfg.n_episodes = args.n_episodes
    cfg.seed = args.seed
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.output_dir = f"{args.output_dir}_{timestamp}"
    
    # ===========================================================================
    # DEFINE PARAMETER GRID
    # ===========================================================================
    # Based on Unified Planner results, we start with high Kp values
    # that were shown to enable robot movement.
    
    param_grid = [
        # =====================================================
        # STAGE 1: Baseline High Stiffness (from Unified Planner)
        # =====================================================
        {"ik_kp": 400.0, "ik_ki": 0.1, "ik_kd": 20.0},
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
    
    results = []
    
    print("=" * 70)
    print(f"SEMANTIC PLANNER GRID SEARCH")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Configurations: {len(param_grid)}")
    print(f"Episodes per config: {args.n_episodes}")
    print(f"Max steps: {cfg.get('max_steps', 400)}")
    print("=" * 70)
    
    for i, params in enumerate(param_grid):
        print(f"\n[Config {i+1}/{len(param_grid)}] Testing: {params}")
        
        # Update config with current parameters
        cfg.ik_kp = params["ik_kp"]
        cfg.ik_ki = params["ik_ki"]
        cfg.ik_kd = params["ik_kd"]
        
        # Create unique output dir for this configuration
        cfg.output_dir = f"{args.output_dir}_{timestamp}/kp{params['ik_kp']:.0f}_ki{params['ik_ki']:.1f}_kd{params['ik_kd']:.0f}"
        
        try:
            # Run evaluation
            evaluator = SemanticPlannerEvaluator(cfg)
            success_rate = evaluator.run()
            
            # Record result
            result_entry = {
                "params": params.copy(),
                "success_rate": success_rate,
                "output_dir": cfg.output_dir
            }
            results.append(result_entry)
            
            print(f"-> Result: Success Rate = {success_rate:.1f}%")
            
        except Exception as e:
            log.error(f"Configuration failed: {e}")
            results.append({
                "params": params.copy(),
                "success_rate": -1.0,
                "error": str(e)
            })
    
    # ===========================================================================
    # PRINT SUMMARY
    # ===========================================================================
    print("\n" + "=" * 70)
    print("GRID SEARCH COMPLETE - RESULTS RANKING")
    print("=" * 70)
    
    # Sort by success rate (descending)
    results.sort(key=lambda x: x["success_rate"], reverse=True)
    
    for rank, res in enumerate(results):
        status = f"{res['success_rate']:>5.1f}%" if res['success_rate'] >= 0 else "ERROR"
        print(f"{rank+1}. Success: {status} | "
              f"Kp={res['params']['ik_kp']:.0f}, "
              f"Ki={res['params']['ik_ki']:.1f}, "
              f"Kd={res['params']['ik_kd']:.0f}")
    
    # Best parameters
    if results and results[0]['success_rate'] >= 0:
        best = results[0]
        print("\n" + "=" * 70)
        print("RECOMMENDED PARAMETERS:")
        print(f"  ik_kp: {best['params']['ik_kp']}")
        print(f"  ik_ki: {best['params']['ik_ki']}")
        print(f"  ik_kd: {best['params']['ik_kd']}")
        print(f"  Success Rate: {best['success_rate']:.1f}%")
        print("=" * 70)
    
    # Save results to file
    results_file = Path(f"{args.output_dir}_{timestamp}") / "grid_search_results.txt"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write("SEMANTIC PLANNER GRID SEARCH RESULTS\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Episodes per config: {args.n_episodes}\n")
        f.write(f"Max steps: {cfg.get('max_steps', 400)}\n")
        f.write("=" * 50 + "\n\n")
        
        for rank, res in enumerate(results):
            status = f"{res['success_rate']:.1f}%" if res['success_rate'] >= 0 else "ERROR"
            f.write(f"{rank+1}. Success: {status} | Params: {res['params']}\n")
        
        if results and results[0]['success_rate'] >= 0:
            f.write(f"\nBest: {results[0]['params']}\n")
    
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    main()
