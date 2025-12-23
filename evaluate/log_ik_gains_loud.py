
import logging
import csv
import sys
import argparse
import os
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf

# Force Headless MuJoCo rendering - use 'egl' on Linux/WSL if needed, but Windows should use default
if sys.platform != "win32":
    os.environ["MUJOCO_GL"] = "egl" 

# Add root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.ik_solver import IKSolver
from evaluate.evaluate_unified_planner_auto import HybridEvaluator, render_goal_image
from models.unified_diffusion_planner import UnifiedDiffusionPlanner, UnifiedDiffusionConfig

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("DistalLog")

# 1. Instrumented IK Solver
class InstrumentedIKSolver(IKSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_data = []
        log.info("InstrumentedIKSolver initialized.")

    def fuzzy_gain_schedule(self, pos_error, rot_error_vec, effective_dt, phase_hint=None):
        # Call original logic
        pos_boost, rot_boost = super().fuzzy_gain_schedule(
            pos_error, rot_error_vec, effective_dt, phase_hint
        )
        
        # Log inputs and outputs
        entry = {
            "step": len(self.log_data),
            "pos_error_cm": float(np.linalg.norm(pos_error) * 100),
            "rot_error_rad": float(np.linalg.norm(rot_error_vec)),
            "pos_boost": float(pos_boost),
            "rot_boost": float(rot_boost),
            "effective_kp": float(self.kp * pos_boost)
        }
        if len(self.log_data) < 5:
            log.info(f"Logging Step {len(self.log_data)}: Error={entry['pos_error_cm']:.2f}cm, Boost={pos_boost:.2f}x")
        
        self.log_data.append(entry)
        return pos_boost, rot_boost

    def save_analysis(self, output_dir: Path):
        log.info(f"Final data points collected: {len(self.log_data)}")
        if not self.log_data:
            log.warning("No data collected!")
            return

        csv_path = output_dir / "ik_gains_log.csv"
        keys = self.log_data[0].keys()
        with open(csv_path, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.log_data)
        log.info(f"Raw data saved to: {csv_path}")

# Mock Policy Loader
def mock_load_policy(self, checkpoint_path):
    log.info("Mock policy loading...")
    cfg = UnifiedDiffusionConfig()
    self.model = UnifiedDiffusionPlanner(cfg).to(self.device)
    self.model.eval()

HybridEvaluator._load_policy = mock_load_policy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ik_kp", type=float, default=100.0)
    args = parser.parse_args()

    cfg = OmegaConf.create({
        "checkpoint": "dummy",
        "env": { "xml_path": "envs/panda_pick_place.xml", "urdf_path": "urdf/panda_mujoco_kinematics.urdf" },
        "sampling": { "inference_steps": 1, "guidance_scale": 1.0 },
        "success_threshold": 0.05,
        "success_duration_steps": 2,
        "output_dir": "outputs/ik_verification",
        "n_episodes": 1,
        "max_steps": 200,
        "seed": 42,
        "ik_kp": args.ik_kp,
        "ik_ki": 0.5,
        "ik_kd": 15.0,
        "action_scale": 1.0
    })

    log.info("Starting Loud Instrumented Evaluation...")
    evaluator = HybridEvaluator(cfg, handoff_phase=3) 
    
    # Hot-swap IK Solver
    original = evaluator.ik_solver
    instrumented = InstrumentedIKSolver(
        urdf_path=cfg.env.urdf_path,
        kp=original.kp, ki=original.ki, kd=original.kd,
        lookahead_steps=original.lookahead_steps
    )
    evaluator.ik_solver = instrumented
    log.info(f"Evaluator IK solver swapped to: {type(evaluator.ik_solver)}")

    # Mock Video Writer and Rendering to avoid EGL crashes
    import unittest.mock as mock
    
    # Mock render to return dummy image
    def dummy_render(*args, **kwargs):
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Apply mocks
    evaluator.env.render = dummy_render
    
    # Also patch render_goal_image to avoid initial render
    import evaluate.evaluate_unified_planner_auto
    evaluate.evaluate_unified_planner_auto.render_goal_image = lambda env, ik, obs: np.zeros((480, 640, 3), dtype=np.uint8)

    # Overload run_episode to inject debug prints
    original_run_episode = evaluator.run_episode
    def loud_run_episode(*largs, **lkwargs):
        log.info("Loud run_episode entered.")
        # Check expert
        log.info(f"Expert state: {evaluator.expert.get_state()}")
        log.info(f"Expert is_done: {evaluator.expert.is_done()}")
        return original_run_episode(*largs, **lkwargs)
    
    evaluator.run_episode = loud_run_episode

    # Run
    log.info("Starting EVALUATION (Rendering Disabled)...")
    try:
        evaluator.run()
    except Exception as e:
        log.error(f"FATAL: {e}")
        import traceback
        traceback.print_exc()
        
    # Report
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    instrumented.save_analysis(Path(cfg.output_dir))

if __name__ == "__main__":
    main()
