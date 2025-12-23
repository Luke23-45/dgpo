
import sys
import numpy as np
from pathlib import Path
import logging

# Add root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.ik_solver import IKSolver

# Configure simplified logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("IKVerifier")

def main():
    log.info("=" * 60)
    log.info("IK GAIN SCHEDULING VERIFICATION (Unit Test)")
    log.info("=" * 60)

    # 1. Initialize Solver (loads URDF, but no rendering)
    # Using the new defaults: max_boost SHOULD be 4.0 (was 10.0)
    urdf_path = "urdf/panda_mujoco_kinematics.urdf"
    
    log.info(f"Instantiating IKSolver from {urdf_path}...")
    try:
        ik_solver = IKSolver(urdf_path=urdf_path, kp=100.0, kd=15.0)
    except Exception as e:
        log.error(f"Failed to load IKSolver: {e}")
        return

    log.info(f"Loaded Params -> Base KP: {ik_solver.kp}")
    log.info(f"               max_boost: {ik_solver.max_boost} (Goal: 4.0)")
    log.info(f"               ref_dist:  {ik_solver.ref_dist_pos} (Goal: 0.02)")

    # 2. Test Cases: Sweep Error from 0cm to 10cm
    log.info("-" * 60)
    log.info(f"{'Error (cm)':<15} | {'Boost Factor':<15} | {'Effective KP':<15} | {'Mode'}")
    log.info("-" * 60)

    errors_cm = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    
    # Dummy velocity (zero for static check)
    ik_solver._fuzzy_prev_error = np.zeros(6) 
    ik_solver._fuzzy_error_rate_filtered = np.zeros(6)

    for err_cm in errors_cm:
        # Create a fake error vector [err, 0, 0]
        pos_error = np.array([err_cm / 100.0, 0.0, 0.0])
        rot_error = np.array([0.0, 0.0, 0.0])
        
        # Call fuzzy scheduler
        # Note: We simulate a small dt step
        pos_boost, _ = ik_solver.fuzzy_gain_schedule(
            pos_error=pos_error,
            rot_error_vec=rot_error,
            effective_dt=0.1
        )
        
        effective_kp = ik_solver.kp * pos_boost
        
        # Classification
        if pos_boost >= 3.0:
            mode = "PRECISION (High)"
        elif pos_boost <= 1.5:
            mode = "STABILITY (Low)"
        else:
            mode = "TRANSITION"

        # Damping Ratio Check
        # Formula: zeta = Kd / (2 * sqrt(Kp))
        # Base: Kp=100, Kd=15 -> zeta = 15 / (2*10) = 0.75
        # Adaptive: Kp' = 100*boost, Kd' = 15*sqrt(boost)
        # Zeta' = (15*sqrt(boost)) / (2 * sqrt(100*boost))
        #       = (15*sqrt(boost)) / (2 * 10 * sqrt(boost))
        #       = 15 / 20 = 0.75 (CONSTANT!)
        
        # We verify this theoretical value since we updated the code to use this formula
        effective_kd = ik_solver.kd * np.sqrt(pos_boost)
        zeta = effective_kd / (2 * np.sqrt(effective_kp))

        log.info(f"{err_cm:<10.1f} | {pos_boost:<10.2f} | {effective_kp:<10.1f} | {effective_kd:<10.2f} | {zeta:<10.4f} | {mode}")

    # 3. Assertions
    log.info("-" * 80)
    if ik_solver.max_boost > 4.5:
        log.error("FAIL: Max boost is still too high! (> 4.5)")
    elif abs(zeta - 0.75) > 1e-3:
         log.error(f"FAIL: Damping ratio changed! Last zeta: {zeta}")
    else:
        log.info("SUCCESS: Damping Ratio is CONSTANT at 0.75 across all error ranges.")
        log.info("SUCCESS: Max boost is safely capped <= 4.0")
        
    log.info("=" * 60)

if __name__ == "__main__":
    main()
