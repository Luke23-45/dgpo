import optuna
import numpy as np
import logging
from omegaconf import OmegaConf
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(".").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.panda_env import PandaEnv
from utils.ik_solver import IKSolver

# Config logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PID_Tuner")

def run_physics_test(env, ik_solver, target_delta, steps=50):
    """
    Moves the robot by a specific delta and measures tracking error.
    Returns: Average Position Error (lower is better)
    """
    # 1. Get current state
    obs = env.get_expert_obs()
    start_ee_pos = obs["ee_pose_world"][:3]
    
    # 2. Define Target (Global Frame)
    target_pos = start_ee_pos + target_delta
    target_quat = obs["ee_pose_world"][3:] # Keep orientation same
    target_pose_7d = np.concatenate([target_pos, target_quat])
    
    total_pos_error = 0.0
    
    # 3. Physics Loop (No AI, just Control)
    for _ in range(steps):
        # Compute Action using IK
        action = ik_solver.compute_delta_action(
            target_ee_pose=target_pose_7d,
            model=env.model,
            data=env.data,
            ee_site_id=env.ee_site_id,
            joint_qpos_indices=np.arange(7),
            effective_dt=env.model.opt.timestep * 20, # 20 substeps
            max_dq=env.ACTION_SCALING_FACTOR / (env.model.opt.timestep * 20)
        )
        
        # Compensate for Env Scaling (Critical for Delta Control)
        # We send the RAW action because the Env multiplies it by Scale
        # In your eval script you used `compensated = delta`, let's stick to raw consistency
        
        # Step Env
        full_action = np.zeros(8)
        full_action[:7] = action
        env.step(full_action)
        
        # Measure Error
        current_ee = env.get_ee_pose()[:3]
        dist = np.linalg.norm(current_ee - target_pos)
        total_pos_error += dist
        
    return total_pos_error / steps

def objective(trial):
    """
    Optuna Objective Function.
    Suggests Kp, Ki, Kd -> Returns Error Score.
    """
    # 1. Suggest Parameters (Bayesian Search Space)
    kp = trial.suggest_float("kp", 100.0, 1000.0) # Search range for Stiffness
    kd = trial.suggest_float("kd", 5.0, 50.0)     # Search range for Damping
    ki = trial.suggest_float("ki", 0.0, 10.0)     # Search range for Integral
    
    # 2. Setup Environment (Re-init to be safe/clean)
    env = PandaEnv(render_mode=None, control_mode="delta")
    
    # 3. Setup Solver with Trial Parameters
    ik_solver = IKSolver(urdf_path="urdf/panda.urdf")
    ik_solver.set_gains(kp=kp, ki=ki, kd=kd)
    
    try:
        env.reset()
        
        # --- TEST 1: Fast Move X (10cm) ---
        error_x = run_physics_test(env, ik_solver, np.array([0.1, 0.0, 0.0]))
        
        # --- TEST 2: Fast Move Z (10cm Up - Fights Gravity) ---
        error_z = run_physics_test(env, ik_solver, np.array([0.0, 0.0, 0.1]))
        
        # --- TEST 3: Holding Still (Steady State Error) ---
        error_hold = run_physics_test(env, ik_solver, np.array([0.0, 0.0, 0.0]))

        # Metric: Weighted sum of errors
        # We penalize Z error more because gravity makes it harder
        score = error_x + (1.5 * error_z) + (2.0 * error_hold)
        
        # Check for instability (Explosion)
        if score > 1.0 or np.isnan(score):
            return 1000.0 # Bad penalty
            
        return score

    finally:
        env.close()

if __name__ == "__main__":
    print("🚀 Starting Bayesian Optimization for PID Gains...")
    
    # Create Study
    study = optuna.create_study(direction="minimize")
    
    # Optimize (Runs 50 trials)
    # This will take ~2 minutes total (vs 5 hours for grid search)
    study.optimize(objective, n_trials=50) 
    
    print("\n✅ Optimization Complete!")
    print(f"Best Error Score: {study.best_value}")
    print("Best Parameters:")
    print(study.best_params)