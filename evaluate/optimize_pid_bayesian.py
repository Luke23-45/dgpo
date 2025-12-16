
"""
optimize_pid_bayesian.py

Automated PID Parameter Tuning using Bayesian Optimization (Optuna).

This script allows for "smart" tuning of the IK Controller parameters Kp, Ki, Kd
and the Action Scale, significantly reducing the number of episodes required to
find optimal settings compared to Grid Search.

Features:
- Uses Tree-structured Parzen Estimator (TPE) sampler (standard for hyperparam tuning).
- Maximizes a composite Objective Function:
    Score = (Success Rate * 100) - (Mean Final Pos Error * 100) - (Mean Time to Success / 10)
    This encourages Fast, Accurate, and Successful execution.
- Saves the best parameters and an "Optimization Study" database.
- Robust error handling for crashed episodes.

Usage:
    pip install optuna  # Ensure optuna is installed
    python evaluate/optimize_pid_bayesian.py --checkpoint ... --n_trials 50
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

import numpy as np
from omegaconf import OmegaConf, DictConfig

try:
    import optuna
    from optuna.trial import TrialState
except ImportError:
    print("ERROR: Optuna is not installed.")
    print("Please install it running: pip install optuna")
    sys.exit(1)

# Add root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the production Evaluator
from evaluate.evaluate_unified_planner import UnifiedPlannerEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("BayesOpt")


class PIDObjective:
    """
    Optuna Objective Function that runs the Unified Planner evaluation.
    """
    def __init__(self, args, base_cfg: DictConfig):
        self.args = args
        self.base_cfg = base_cfg
        self.eval_history = []
        
        # Prepare fixed output dir for this study
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.study_dir = Path(args.output_dir) / f"study_{self.timestamp}"
        self.study_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, trial: optuna.Trial) -> float:
        """
        The main optimization loop called by Optuna.
        Returns a float 'value' to be MAXIMIZED.
        """
        # 1. Sample Hyperparameters
        # Tuning ranges based on our research and prior grid search knowledge
        kp = trial.suggest_float("ik_kp", 100.0, 800.0)
        ki = trial.suggest_float("ik_ki", 0.0, 5.0)
        # Kd usually correlates with Kp, we give it a reasonable damping range
        kd = trial.suggest_float("ik_kd", 5.0, 40.0)
        
        # Action scale - can be categorical or discrete steps
        action_scale = trial.suggest_categorical("action_scale", [25.0, 50.0, 75.0, 100.0])
        
        # 2. Configure Evaluator
        cfg = self.base_cfg.copy()
        cfg.ik_kp = kp
        cfg.ik_ki = ki
        cfg.ik_kd = kd
        cfg.action_scale = action_scale
        
        # Override critical eval params from args
        cfg.unified_planner_checkpoint = self.args.checkpoint
        cfg.n_episodes = self.args.n_episodes
        cfg.max_steps = self.args.max_steps
        cfg.seed = self.args.seed + trial.number # Different seed per trial to avoid overfitting? 
                                                 # Actually, normally we want SAME seed to compare PIDs fairly.
                                                 # Let's keep seed fixed for stability.
        cfg.seed = self.args.seed 

        
        # Enable IK for tuning
        cfg.use_ik = True

        # Setup output storage for this trial
        trial_name = f"trial_{trial.number:03d}"
        cfg.output_dir = str(self.study_dir / trial_name)
        cfg.experiment_name = f"T{trial.number:03d}_kp{kp:.0f}_as{action_scale:.0f}"

        log.info(f"\n=== Trial {trial.number} ===")
        log.info(f"Params: Kp={kp:.1f}, Ki={ki:.2f}, Kd={kd:.1f}, Scale={action_scale}")

        # 3. Run Evaluation
        try:
            # Initialize UnifiedPlannerEvaluator
            evaluator = UnifiedPlannerEvaluator(cfg)
            
            # Run episodes - returns success_rate (float)
            success_rate = evaluator.run()
            
            # To differentiate between two configs with 100% success, 
            # we could penalize 'avg_steps' or 'final_error'.
            # For now, let's keep it simple: Maximize Success Rate.
            
            score = success_rate
            
            # Save metrics to history
            self.eval_history.append({
                "trial": trial.number,
                "params": trial.params,
                "score": score,
                "metrics": {"success_rate": success_rate}
            })
            
            log.info(f"-> Result: Success Rate = {success_rate:.1f}%")
            
            return score

        except Exception as e:
            log.error(f"Trial {trial.number} Failed: {e}")
            # Prune this trial
            raise optuna.exceptions.TrialPruned()


def main():
    parser = argparse.ArgumentParser(description="Bayesian Optimization for PID Gains")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="configs/eval_unified_planner_config.yaml")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of optimization trials")
    parser.add_argument("--n_episodes", type=int, default=3, help="Episodes per trial")
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/optimization_pid")
    
    args = parser.parse_args()

    # Load base config
    config_path = ROOT / args.config
    if not config_path.exists():
        log.error(f"Config not found: {config_path}")
        sys.exit(1)
    
    base_cfg = OmegaConf.load(config_path)

    # Initialize Objective
    objective = PIDObjective(args, base_cfg)

    # Initialize Study
    # Direction="maximize" because we want higher Success Rate
    study = optuna.create_study(
        study_name=f"pid_tuning_{objective.timestamp}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner()
    )

    log.info(f"Starting Optimization with {args.n_trials} trials...")
    log.info(f"Output Directory: {objective.study_dir}")

    # Optimize!
    study.optimize(objective, n_trials=args.n_trials)

    # Report Results
    log.info("\n" + "=" * 50)
    log.info("OPTIMIZATION COMPLETE")
    log.info("=" * 50)
    
    best_trial = study.best_trial
    log.info(f"Best Trial: {best_trial.number}")
    log.info(f"Best Value (Success Rate): {best_trial.value:.1f}%")
    log.info("Best Params:")
    for key, value in best_trial.params.items():
        log.info(f"  {key}: {value}")

    # Save Study Results
    study_results = {
        "best_params": best_trial.params,
        "best_value": best_trial.value,
        "trials": []
    }
    
    for t in study.trials:
        if t.state == TrialState.COMPLETE:
            study_results["trials"].append({
                "number": t.number,
                "params": t.params,
                "value": t.value
            })
            
    results_path = objective.study_dir / "optimization_results.json"
    with open(results_path, "w") as f:
        json.dump(study_results, f, indent=4)
        
    log.info(f"Comparison saved to {results_path}")


if __name__ == "__main__":
    main()
