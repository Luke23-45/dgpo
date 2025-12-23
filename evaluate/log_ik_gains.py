
import logging
import csv
import sys
import argparse
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf

# Add root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.ik_solver import IKSolver
from evaluate.evaluate_unified_planner_auto import HybridEvaluator
from models.unified_diffusion_planner import UnifiedDiffusionPlanner, UnifiedDiffusionConfig

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("DistalLog")

# Silence noisy loggers
logging.getLogger("models.unified_diffusion_planner").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# 1. Instrumented IK Solver
class InstrumentedIKSolver(IKSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_data = []

    def fuzzy_gain_schedule(self, pos_error, rot_error_vec, effective_dt, phase_hint=None):
        # Call original logic
        pos_boost, rot_boost = super().fuzzy_gain_schedule(
            pos_error, rot_error_vec, effective_dt, phase_hint
        )
        
        # Log inputs and outputs
        entry = {
            "step": len(self.log_data),
            "pos_error_cm": np.linalg.norm(pos_error) * 100, # Convert to cm for readability
            "rot_error_rad": np.linalg.norm(rot_error_vec),
            "pos_boost": pos_boost,
            "rot_boost": rot_boost,
            "effective_kp": self.kp * pos_boost
        }
        self.log_data.append(entry)
        return pos_boost, rot_boost

    def save_analysis(self, output_dir: Path):
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

        # --- ANALYSIS REPORT ---
        log.info("\n" + "="*40)
        log.info("       DYNAMIC GAIN ANALYSIS REPORT")
        log.info("="*40)
        log.info(f"Base KP: {self.kp}")
        log.info(f"Total Steps: {len(self.log_data)}")
        log.info("-" * 40)
        
        # 1. Stability Check (Large Errors)
        large_errors = [d['pos_boost'] for d in self.log_data if d['pos_error_cm'] > 5.0]
        if large_errors:
            avg_boost = sum(large_errors) / len(large_errors)
            log.info(f"[STABILITY] Avg Boost at Error > 5cm:  {avg_boost:.2f}x (Expected ~1.0-2.0)")
        else:
            log.info("[STABILITY] No large errors > 5cm observed.")

        # 2. Precision Check (Small Errors)
        small_errors = [d['pos_boost'] for d in self.log_data if d['pos_error_cm'] < 0.5]
        if small_errors:
            avg_boost = sum(small_errors) / len(small_errors)
            max_boost = max(small_errors)
            log.info(f"[PRECISION] Avg Boost at Error < 0.5cm: {avg_boost:.2f}x")
            log.info(f"[PRECISION] Max Boost at Error < 0.5cm: {max_boost:.2f}x (Capped at 4.0)")
        else:
            log.info("[PRECISION] No small errors < 0.5cm observed.")
            
        log.info("="*40 + "\n")


# Mock Policy Loader
def mock_load_policy(self, checkpoint_path):
    # log.info("Initializing fast mock policy (No checkpoint loading)...")
    cfg = UnifiedDiffusionConfig()
    self.model = UnifiedDiffusionPlanner(cfg).to(self.device)
    self.model.eval()

HybridEvaluator._load_policy = mock_load_policy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ik_kp", type=float, default=100.0)
    args = parser.parse_args()

    # Minimal Config
    cfg = OmegaConf.create({
        "checkpoint": "dummy",
        "env": { "xml_path": "envs/panda_pick_place.xml", "urdf_path": "urdf/panda_mujoco_kinematics.urdf" },
        "sampling": { "inference_steps": 1, "guidance_scale": 1.0 },
        "success_threshold": 0.05,
        "success_duration_steps": 2,
        "output_dir": "outputs/ik_verification",
        "n_episodes": 1,
        "max_steps": 300,
        "seed": 42,
        "ik_kp": args.ik_kp,
        "ik_ki": 0.5,
        "ik_kd": 15.0,
        "action_scale": 50.0
    })

    log.info(f"Starting Instrumented Evaluation (Base KP={args.ik_kp})...")
    
    # Handoff Phase 3: Expert does everything up to Placement.
    # This guarantees we see Reach (Fast) and Grasp (Precise).
    evaluator = HybridEvaluator(cfg, handoff_phase=3) 
    
    # Hot-swap IK Solver
    original = evaluator.ik_solver
    instrumented = InstrumentedIKSolver(
        urdf_path=cfg.env.urdf_path,
        kp=original.kp, ki=original.ki, kd=original.kd,
        lookahead_steps=original.lookahead_steps
    )
    evaluator.ik_solver = instrumented
    
    # Run
    log.info("Starting evaluator.run()...")
    try:
        evaluator.run()
    except Exception as e:
        log.error(f"FATAL ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
        
    # Report
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    instrumented.save_analysis(Path(cfg.output_dir))

if __name__ == "__main__":
    main()
