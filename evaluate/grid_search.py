
import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

# Add root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate.evaluate_unified_planner import UnifiedPlannerEvaluator

# Configure simple logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("GridSearch")

def main():
    parser = argparse.ArgumentParser(description="Grid Search for Unified Planner Parameters")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/eval_unified_planner_config.yaml", help="Path to base config")
    parser.add_argument("--n_episodes", type=int, default=5, help="Episodes per parameter configuration")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum steps per episode (override config)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load base config
    cfg = OmegaConf.load(ROOT / args.config)
    cfg.checkpoint = args.checkpoint
    cfg.n_episodes = args.n_episodes
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    cfg.seed = args.seed
    
    # Define Parameter Grid (STAGE 1: PID CALIBRATION)
    # Context: We fix Action Scale = 1.0 (Physical Limit: 2.2 rad/s).
    # We tune PID to get maximum responsiveness from this "True" signal.
    
    # We test Higher Kp to see if we can get drive without "Fake Gain".
    BASE_SCALE = 1.0
    
    # param_grid = [
    #     # 1. Baseline Stiffness (High Damping? Low Damping?)
    #     {"action_scale": BASE_SCALE, "guidance_scale": 1.0, "ik_kp": 100.0, "ik_kd": 1.0, "ik_ki": 0.0},
    #     {"action_scale": BASE_SCALE, "guidance_scale": 1.0, "ik_kp": 100.0, "ik_kd": 5.0, "ik_ki": 0.1},
        
    #     # 2. High Stiffness (To compensate for lack of Scale boost)
    #     {"action_scale": BASE_SCALE, "guidance_scale": 1.0, "ik_kp": 200.0, "ik_kd": 5.0, "ik_ki": 0.1},
    #     {"action_scale": BASE_SCALE, "guidance_scale": 1.0, "ik_kp": 200.0, "ik_kd": 10.0, "ik_ki": 0.1},
        
    #     # 3. Very High Stiffness (Industrial Robot style)
    #     {"action_scale": BASE_SCALE, "guidance_scale": 1.0, "ik_kp": 300.0, "ik_kd": 10.0, "ik_ki": 0.1},
    #     {"action_scale": BASE_SCALE, "guidance_scale": 1.0, "ik_kp": 400.0, "ik_kd": 20.0, "ik_ki": 0.2},
        
    #     # 4. Integral Heavy (To fix undershoot/drift)
    #     {"action_scale": BASE_SCALE, "guidance_scale": 1.0, "ik_kp": 150.0, "ik_kd": 5.0, "ik_ki": 1.0},
    #     {"action_scale": BASE_SCALE, "guidance_scale": 1.0, "ik_kp": 150.0, "ik_kd": 5.0, "ik_ki": 2.0},
    # ]

    param_grid = [
        # 1. Baseline Stiffness (High Damping? Low Damping?)

        # {"action_scale": BASE_SCALE, "guidance_scale": 1.0, "ik_kp": 435.0, "ik_kd": 10.0, "ik_ki": 0.1},
        
        # # 3. Very High Stiffness (Industrial Robot style)
        # {"action_scale": BASE_SCALE, "guidance_scale": 1.0, "ik_kp": 445.0, "ik_kd": 20.0, "ik_ki": 0.1},
        # {"action_scale": BASE_SCALE, "guidance_scale": 1.0, "ik_kp": 450.0, "ik_kd": 20.0, "ik_ki": 0.2},
        # {"action_scale": BASE_SCALE, "guidance_scale": 1.0, "ik_kp": 400.0, "ik_kd": 20.0, "ik_ki": 0.2},
        # 4. Integral Heavy (To fix undershoot/drift)

        {"action_scale": BASE_SCALE, "guidance_scale": 1.0, "ik_kp": 800.0, "ik_kd": 20.0, "ik_ki": 2.0},
        {"action_scale": BASE_SCALE, "guidance_scale": 1.0, "ik_kp": 900.0, "ik_kd": 20.0, "ik_ki": 0.2},
        {"action_scale": BASE_SCALE, "guidance_scale": 1.0, "ik_kp": 700.0, "ik_kd": 20.0, "ik_ki": 1.0},
        {"action_scale": BASE_SCALE, "guidance_scale": 1.0, "ik_kp": 600.0, "ik_kd": 20.0, "ik_ki": 1.0},
    ]

    # {"action_scale": BASE_SCALE, "guidance_scale": 1.0, "ik_kp": 470.0, "ik_kd": 20.0, "ik_ki": 2.0}, and   {"action_scale": BASE_SCALE, "guidance_scale": 1.0, "ik_kp": 500.0, "ik_kd": 20.0, "ik_ki": 0.2},

    # param_grid = [
    #     # 1. Baseline Stiffness (High Damping? Low Damping?)

    #     {"action_scale": BASE_SCALE, "guidance_scale": 1.0, "ik_kp": 520.0, "ik_kd": 10.0, "ik_ki": 0.1},
        
    #     # 3. Very High Stiffness (Industrial Robot style)
    #     {"action_scale": BASE_SCALE, "guidance_scale": 1.0, "ik_kp": 550.0, "ik_kd": 20.0, "ik_ki": 0.1},
    #     {"action_scale": BASE_SCALE, "guidance_scale": 1.0, "ik_kp": 500.0, "ik_kd": 20.0, "ik_ki": 0.2},

    # ]
    results = []

    print("="*60)
    print(f"STARTING GRID SEARCH: {len(param_grid)} Configurations")
    print(f"Episodes per config: {args.n_episodes}")
    print("="*60)

    for i, params in enumerate(param_grid):
        print(f"\n[Config {i+1}/{len(param_grid)}] Testing: {params}")
        
        # update config
        cfg.action_scale = params["action_scale"]
        cfg.sampling.guidance_scale = params["guidance_scale"]
        cfg.ik_kp = params.get("ik_kp", 139.0)
        cfg.ik_kd = params.get("ik_kd", 3.0)
        cfg.ik_ki = params.get("ik_ki", 0.1)
        
        # Run Evaluation
        # Note: We re-instantiate because env.close() is called inside run()
        try:
            evaluator = UnifiedPlannerEvaluator(cfg)
            # Suppress excessive logging during search if desired, or keep it for debug
            success_rate = evaluator.run()
            
            # Record result
            result_entry = {
                "params": params,
                "success_rate": success_rate,
                "output_dir": str(evaluator.output_dir)
            }
            results.append(result_entry)
            
            print(f"-> Result: Success Rate = {success_rate}%")
            
        except Exception as e:
            print(f"-> Run FAILED with error: {e}")
            results.append({"params": params, "success_rate": -1.0, "error": str(e)})

    # Print Summary
    print("\n" + "="*60)
    print("GRID SEARCH COMPLETE - RESULTS RANKING")
    print("="*60)
    
    # Sort by success rate (descending)
    results.sort(key=lambda x: x["success_rate"], reverse=True)
    
    for rank, res in enumerate(results):
        print(f"{rank+1}. Success: {res['success_rate']:>5.1f}% | Params: {res['params']}")

    best_params = results[0]['params']
    print(f"\n*** RECOMMENDED PARAMETERS: {best_params} ***")

if __name__ == "__main__":
    main()
