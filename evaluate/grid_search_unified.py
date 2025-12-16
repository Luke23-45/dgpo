# FILE: evaluate/grid_search_unified.py
# Grid Search for Unified Planner Parameters (Including Handoff Phase Testing)

"""
Grid Search Script for Unified Planner Parameter Tuning.

This script systematically searches over:
1. IK PID parameters (Kp, Ki, Kd) - for control performance
2. Action scale - for model output magnitude
3. Handoff phases - to identify which phase the model struggles with

The handoff phase testing is particularly powerful:
- Running with handoff_phase=0 tests the model on the entire task
- Running with handoff_phase=1 tests the model after expert has approached and grasped
- Running with handoff_phase=2 tests the model after expert has lifted

Comparing success rates across handoff phases reveals which phase is the bottleneck.

Usage:
    # Standard PID tuning (model controls entire task):
    python evaluate/grid_search_unified.py --checkpoint /path/to/model.ckpt --n_episodes 3
    
    # Handoff phase comparison (diagnose which phase model fails at):
    python evaluate/grid_search_unified.py --checkpoint /path/to/model.ckpt --mode handoff --n_episodes 3
    
    # Fast iteration mode:
    python evaluate/grid_search_unified.py --checkpoint /path/to/model.ckpt --n_episodes 2 --max_steps 300
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
from omegaconf import OmegaConf

# Add root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate.evaluate_unified_planner_auto import HybridEvaluator, PHASE_NAMES

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("GridSearch_Unified")


def run_pid_grid_search(args) -> List[Dict]:
    """
    Grid search over IK PID parameters and action scale.
    Uses full model control (handoff_phase=0).
    """
    # Load base config
    config_path = ROOT / args.config
    if config_path.exists():
        cfg = OmegaConf.load(config_path)
    else:
        log.warning("Config not found, creating minimal config")
        cfg = OmegaConf.create({
            "env": {
                "xml_path": "envs/panda_pick_place.xml",
                "urdf_path": "urdf/panda_mujoco_kinematics.urdf"
            },
            "sampling": {
                "inference_steps": 10,
                "guidance_scale": 1.5
            },
            "success_threshold": 0.05,
            "success_duration_steps": 10
        })
    
    # Override with command line args
    cfg.checkpoint = args.checkpoint
    cfg.n_episodes = args.n_episodes
    cfg.max_steps = args.max_steps
    cfg.seed = args.seed
    BASE_SCALE = args.action_scale 
    
    # Define parameter grid
    param_grid = [
        # =====================================================
        # STAGE 1: Baseline High Stiffness (validated values)
        # =====================================================
        {"ik_kp": 400.0, "ik_ki": 0.5, "ik_kd": 10.0, "action_scale": BASE_SCALE},
        {"ik_kp": 450.0, "ik_ki": 0.5, "ik_kd": 10.0, "action_scale": BASE_SCALE},
        {"ik_kp": 470.0, "ik_ki": 0.5, "ik_kd": 15.0, "action_scale": BASE_SCALE},
        {"ik_kp": 420.0, "ik_ki": 0.5, "ik_kd": 15.0, "action_scale": BASE_SCALE},
        {"ik_kp": 160.0, "ik_ki": 0.5, "ik_kd": 15.0, "action_scale": BASE_SCALE},
        
        # =====================================================
        # STAGE 2: Action Scale Variants
        # =====================================================
        {"ik_kp": 500.0, "ik_ki": 0.5, "ik_kd": 15.0, "action_scale": BASE_SCALE},
        {"ik_kp": 500.0, "ik_ki": 0.5, "ik_kd": 15.0, "action_scale": BASE_SCALE},
        {"ik_kp": 500.0, "ik_ki": 0.5, "ik_kd": 15.0, "action_scale": BASE_SCALE},
        
        # =====================================================
        # STAGE 3: Damping Variants
        # =====================================================
        {"ik_kp": 500.0, "ik_ki": 0.5, "ik_kd": 10.0, "action_scale": BASE_SCALE},
        {"ik_kp": 500.0, "ik_ki": 0.5, "ik_kd": 20.0, "action_scale": BASE_SCALE},
        {"ik_kp": 500.0, "ik_ki": 0.5, "ik_kd": 25.0, "action_scale": BASE_SCALE},
        
        # =====================================================
        # STAGE 4: Integral Gain Variants
        # =====================================================
        {"ik_kp": 500.0, "ik_ki": 0.1, "ik_kd": 15.0, "action_scale": BASE_SCALE},
        {"ik_kp": 500.0, "ik_ki": 1.0, "ik_kd": 15.0, "action_scale": BASE_SCALE},
        {"ik_kp": 500.0, "ik_ki": 2.0, "ik_kd": 15.0, "action_scale": BASE_SCALE},
    ]
    
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 70)
    print("UNIFIED PLANNER GRID SEARCH - PID TUNING")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Configurations: {len(param_grid)}")
    print(f"Episodes per config: {args.n_episodes}")
    print(f"Max steps: {args.max_steps}")
    print("=" * 70)
    
    
    # Check for requested phases or default to 0
    target_phases = args.phases if args.phases and len(args.phases) > 0 else [0]
    
    for phase_idx, handoff_phase in enumerate(target_phases):
        phase_name = PHASE_NAMES[handoff_phase] if handoff_phase < len(PHASE_NAMES) else f"Phase{handoff_phase}"
        print(f"\n" + "-" * 50)
        print(f"TESTING HANDOFF PHASE {handoff_phase} ({phase_name})")
        print("-" * 50)
        
        for i, params in enumerate(param_grid):
            print(f"\n[Phase {handoff_phase} | Config {i+1}/{len(param_grid)}] Testing: {params}")
            
            # Update config
            cfg.ik_kp = params["ik_kp"]
            cfg.ik_ki = params["ik_ki"]
            cfg.ik_kd = params["ik_kd"]
            cfg.action_scale = params["action_scale"]
            
            # FLAT OUTPUT STRUCTURE with Phase info in filename
            cfg.output_dir = f"{args.output_dir}_{timestamp}"
            cfg.experiment_name = f"phase{handoff_phase}_config{i+1}_kp{params['ik_kp']:.0f}_as{params['action_scale']:.0f}"
            
            try:
                evaluator = HybridEvaluator(cfg, handoff_phase=handoff_phase)
                eval_results = evaluator.run()
                
                result_entry = {
                    "handoff_phase": handoff_phase,
                    "params": params.copy(),
                    "success_rate": eval_results["success_rate"],
                    "output_dir": cfg.output_dir
                }
                results.append(result_entry)
                
                print(f"-> Result: Phase {handoff_phase} Success = {eval_results['success_rate']:.1f}%")
                
            except Exception as e:
                log.error(f"Configuration failed: {e}")
                results.append({
                    "handoff_phase": handoff_phase,
                    "params": params.copy(),
                    "success_rate": -1.0,
                    "error": str(e)
                })
    
    return results


def run_handoff_grid_search(args) -> List[Dict]:
    """
    Grid search over handoff phases to diagnose at which phase the model fails.
    Uses fixed optimal PID parameters.
    """
    # Load base config
    config_path = ROOT / args.config
    if config_path.exists():
        cfg = OmegaConf.load(config_path)
    else:
        cfg = OmegaConf.create({
            "env": {
                "xml_path": "envs/panda_pick_place.xml",
                "urdf_path": "urdf/panda_mujoco_kinematics.urdf"
            },
            "sampling": {
                "inference_steps": 10,
                "guidance_scale": 1.5
            },
            "success_threshold": 0.05,
            "success_duration_steps": 10
        })
    
    # Override with command line args and use optimal PID
    cfg.checkpoint = args.checkpoint
    cfg.n_episodes = args.n_episodes
    cfg.max_steps = args.max_steps
    cfg.seed = args.seed
    cfg.ik_kp = args.ik_kp
    cfg.ik_ki = args.ik_ki
    cfg.ik_kd = args.ik_kd
    cfg.action_scale = args.action_scale
    
    # Handoff phases to test
    if args.phases:
        handoff_phases = args.phases
    else:
        handoff_phases = [0, 1, 2, 3]  # Default: Test all phases
    
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 70)
    print("UNIFIED PLANNER GRID SEARCH - HANDOFF PHASE DIAGNOSIS")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Handoff Phases: {handoff_phases}")
    print(f"Episodes per phase: {args.n_episodes}")
    print(f"PID: Kp={cfg.ik_kp}, Ki={cfg.ik_ki}, Kd={cfg.ik_kd}")
    print(f"Action Scale: {cfg.action_scale}")
    print("=" * 70)
    print("\nInterpretation Guide:")
    print("  - Phase 0 (REACH): Model controls entire task from start")
    print("  - Phase 1 (GRASP): Expert approaches & grasps, Model lifts & places")
    print("  - Phase 2 (LIFT):  Expert lifts object, Model transports & places")
    print("  - Phase 3 (PLACE): Expert does everything, Model only releases")
    print("\nIf success increases with higher handoff phase, model struggles early in task.")
    print("=" * 70)
    
    for handoff_phase in handoff_phases:
        print(f"\n[Handoff Phase {handoff_phase}] Testing: "
              f"Expert controls until {PHASE_NAMES[handoff_phase]}, then Model takes over")
        
        # FLAT OUTPUT STRUCTURE
        cfg.output_dir = f"{args.output_dir}_{timestamp}"
        cfg.experiment_name = f"handoff_{handoff_phase}_{PHASE_NAMES[handoff_phase]}"
        
        try:
            evaluator = HybridEvaluator(cfg, handoff_phase=handoff_phase)
            eval_results = evaluator.run()
            
            result_entry = {
                "handoff_phase": handoff_phase,
                "handoff_name": PHASE_NAMES[handoff_phase],
                "success_rate": eval_results["success_rate"],
                "output_dir": cfg.output_dir
            }
            results.append(result_entry)
            
            print(f"-> Result: Success Rate = {eval_results['success_rate']:.1f}%")
            
        except Exception as e:
            log.error(f"Handoff phase {handoff_phase} failed: {e}")
            results.append({
                "handoff_phase": handoff_phase,
                "handoff_name": PHASE_NAMES[handoff_phase],
                "success_rate": -1.0,
                "error": str(e)
            })
    
    return results


def print_summary(results: List[Dict], mode: str, args):
    """Print summary of grid search results."""
    print("\n" + "=" * 70)
    print("GRID SEARCH COMPLETE - RESULTS RANKING")
    print("=" * 70)
    
    if mode == "handoff":
        # Sort by handoff phase (ascending to show progression)
        results.sort(key=lambda x: x.get("handoff_phase", 0))
        
        print("\nHandoff Phase Analysis:")
        print("-" * 50)
        for res in results:
            status = f"{res['success_rate']:>5.1f}%" if res['success_rate'] >= 0 else "ERROR"
            print(f"  Phase {res['handoff_phase']} ({res['handoff_name']:<7}): Success = {status}")
        
        # Interpretation
        print("\n" + "-" * 50)
        print("Interpretation:")
        valid_results = [r for r in results if r['success_rate'] >= 0]
        if len(valid_results) >= 2:
            sr_p0 = next((r['success_rate'] for r in valid_results if r.get('handoff_phase') == 0), None)
            sr_p1 = next((r['success_rate'] for r in valid_results if r.get('handoff_phase') == 1), None)
            sr_p2 = next((r['success_rate'] for r in valid_results if r.get('handoff_phase') == 2), None)
            
            if sr_p0 is not None and sr_p1 is not None:
                if sr_p1 > sr_p0 + 20:
                    print("  ⚠️  Model struggles with APPROACH/GRASP phase")
                    print("     Consider retraining with focus on reaching behavior")
                elif sr_p2 is not None and sr_p2 > sr_p1 + 20:
                    print("  ⚠️  Model struggles with LIFT phase")
                    print("     Increase IK gains or check action_scale")
                elif sr_p0 > 50:
                    print("  ✓  Model performs well on the complete task!")
                else:
                    print("  ⚠️  Model struggles across all phases")
                    print("     Consider more training data or architecture changes")
    else:
        # PID tuning mode - sort by success rate
        results.sort(key=lambda x: x.get("success_rate", -1), reverse=True)
        
        for rank, res in enumerate(results):
            status = f"{res['success_rate']:>5.1f}%" if res['success_rate'] >= 0 else "ERROR"
            p = res['params']
            print(f"{rank+1}. Success: {status} | "
                  f"Kp={p['ik_kp']:.0f}, Ki={p['ik_ki']:.1f}, Kd={p['ik_kd']:.0f}, AS={p['action_scale']:.0f}")
        
        # Best parameters
        if results and results[0]['success_rate'] >= 0:
            best = results[0]
            print("\n" + "=" * 70)
            print("RECOMMENDED PARAMETERS:")
            print(f"  ik_kp: {best['params']['ik_kp']}")
            print(f"  ik_ki: {best['params']['ik_ki']}")
            print(f"  ik_kd: {best['params']['ik_kd']}")
            print(f"  action_scale: {best['params']['action_scale']}")
            print(f"  Success Rate: {best['success_rate']:.1f}%")
    
    print("=" * 70)
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(f"{args.output_dir}_{timestamp}") / "grid_search_results.txt"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write(f"UNIFIED PLANNER GRID SEARCH RESULTS ({mode.upper()} MODE)\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Episodes per config: {args.n_episodes}\n")
        f.write(f"Max steps: {args.max_steps}\n")
        f.write("=" * 50 + "\n\n")
        
        for rank, res in enumerate(results):
            if mode == "handoff":
                status = f"{res['success_rate']:.1f}%" if res['success_rate'] >= 0 else "ERROR"
                f.write(f"Phase {res['handoff_phase']} ({res['handoff_name']}): {status}\n")
            else:
                status = f"{res['success_rate']:.1f}%" if res['success_rate'] >= 0 else "ERROR"
                f.write(f"{rank+1}. Success: {status} | Params: {res['params']}\n")
    
    print(f"\nResults saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Grid Search for Unified Planner Parameters")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to unified planner checkpoint")
    parser.add_argument("--config", type=str, default="configs/eval_unified_planner_config.yaml",
                        help="Path to base config file")
    parser.add_argument("--mode", type=str, default="pid", choices=["pid", "handoff"],
                        help="Grid search mode: 'pid' for parameter tuning, 'handoff' for phase diagnosis")
    parser.add_argument("--n_episodes", type=int, default=3,
                        help="Episodes per configuration")
    parser.add_argument("--max_steps", type=int, default=800,
                        help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/grid_search_unified",
                        help="Directory to save results")
    
    # Default optimal PID values (for handoff mode)
    parser.add_argument("--ik_kp", type=float, default=500.0)
    parser.add_argument("--ik_ki", type=float, default=0.5)
    parser.add_argument("--ik_kd", type=float, default=15.0)
    parser.add_argument("--action_scale", type=float, default=50.0)
    parser.add_argument("--phases", type=int, nargs="+", default=None,
                        help="Specific handoff phases to test (e.g. --phases 1 2). Default: all [0, 1, 2, 3]")
    
    args = parser.parse_args()
    
    if args.mode == "pid":
        results = run_pid_grid_search(args)
    else:
        results = run_handoff_grid_search(args)
    
    print_summary(results, args.mode, args)
    
    return results


if __name__ == "__main__":
    main()
