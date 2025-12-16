
"""
tune_pid.py

Fast, Physics-Based PID Tuning for IKSolver.
Optimizes Kp, Ki, Kd to minimize position tracking error on deterministic reference moves.
Runs in seconds (vs hours for model evaluation).

Usage:
    pip install optuna
    python tune_pid.py
"""

import sys
import numpy as np
import logging
from pathlib import Path

try:
    import optuna
except ImportError:
    print("Please install optuna: pip install optuna")
    sys.exit(1)

# Add project root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.panda_env import PandaEnv
from utils.ik_solver import IKSolver

# Config logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("PID_Tuner")

# CONFIG
XML_PATH = str(ROOT / "envs/panda_pick_place.xml")
URDF_PATH = str(ROOT / "urdf/panda_mujoco_kinematics.urdf")
SIM_SUBSTEPS = 20  # Matches evaluate_unified_planner.py logic

def run_physics_test(env, ik_solver, target_delta, steps=50):
    """
    Moves the robot by a specific delta and measures tracking error.
    Returns: Average Position Error (lower is better)
    """
    # 1. Get current state
    obs = env.get_expert_obs()
    start_ee_pos = obs["ee_pose_world"][:3]
    start_ee_quat = obs["ee_pose_world"][3:]
    
    # 2. Define Target (Global Frame)
    target_pos = start_ee_pos + target_delta
    # Keep orientation same as start
    target_pose_7d = np.concatenate([target_pos, start_ee_quat])
    
    total_pos_error = 0.0
    effective_dt = env.model.opt.timestep * SIM_SUBSTEPS
    
    # Calculate max_dq based on environment scaling
    # We want the solver to output velocities that, when sealed by env, result in valid motion.
    # Env applies: q_vel = action * ACTION_SCALING_FACTOR
    # So max action should constitute max physical velocity.
    # But for IK, we typically pass max_dq in radians/sec.
    # Let's use a safe physical limit (e.g., 2.0 rad/s) and let the PID try to achieve it.
    max_dq = 2.0 
    
    # 3. Physics Loop (No AI, just Control)
    for _ in range(steps):
        # Update Solver Internal State (Current Joint Config)
        # Note: IKSolver usually reads from env.data in compute_delta_action
        
        # Compute Action using IK
        try:
            delta_joints = ik_solver.compute_delta_action(
                target_ee_pose=target_pose_7d,
                model=env.model,
                data=env.data,
                ee_site_id=env.ee_site_id,
                joint_qpos_indices=np.arange(7),
                effective_dt=effective_dt,
                max_dq=max_dq
            )
        except Exception as e:
            return 10.0 # Heavy penalty for IK failure
        
        # Step Env (Delta Control Mode)
        full_action = np.zeros(8)
        full_action[:7] = delta_joints
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
    kp = trial.suggest_float("kp", 100.0, 1000.0) # Stiffness
    ki = trial.suggest_float("ki", 0.0, 10.0)     # Integral (fixes steady state error)
    kd = trial.suggest_float("kd", 5.0, 100.0)    # Damping (stops oscillation)
    
    # 2. Setup Environment
    # We use a headless env for speed
    env = PandaEnv(
        xml_path=XML_PATH,
        render_mode=None, 
        control_mode="delta",
        action_scaling_factor=1.0 # Use 1.0 to behave like "Raw" physics for tuning
    )
    
    # 3. Setup Solver with Trial Parameters
    ik_solver = IKSolver(urdf_path=URDF_PATH)
    ik_solver.set_gains(kp=kp, ki=ki, kd=kd)
    
    try:
        env.reset()
        
        # Warmup
        for _ in range(10): env.step(np.zeros(8))
        
        # --- TEST 1: Fast Move X (10cm) ---
        # Tests transient response and rise time
        error_x = run_physics_test(env, ik_solver, np.array([0.1, 0.0, 0.0]), steps=40)
        
        # --- TEST 2: Fast Move Z (10cm Up - Fights Gravity) ---
        # Tests gravity compensation and integral term
        error_z = run_physics_test(env, ik_solver, np.array([0.0, 0.0, 0.1]), steps=40)
        
        # --- TEST 3: Holding Still (Steady State Error) ---
        # Tests stability and jitter
        error_hold = run_physics_test(env, ik_solver, np.array([0.0, 0.0, 0.0]), steps=20)

        # Metric: Weighted sum of errors
        # High penalty for Z error (sagging) and Hold error (jitter)
        score = error_x + (2.0 * error_z) + (3.0 * error_hold)
        
        # Check for numeric instability
        if np.isnan(score) or score > 10.0:
            return 1000.0 
            
        return score

    finally:
        env.close()

if __name__ == "__main__":
    print("🚀 Starting Physics-Based PID Tuning...")
    print(f"XML: {XML_PATH}")
    
    # Create Study
    study = optuna.create_study(
        study_name="physics_pid_tuning",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Optimize (Runs 50 trials)
    # This runs very fast (seconds per trial)
    study.optimize(objective, n_trials=50) 
    
    print("\n✅ Optimization Complete!")
    print(f"Best Error Score: {study.best_value:.4f} meters (cumulative)")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nUSE THESE VALUES IN YOUR CONFIG:")
    print(f"ik_kp: {study.best_params['kp']:.1f}")
    print(f"ik_ki: {study.best_params['ki']:.2f}")
    print(f"ik_kd: {study.best_params['kd']:.1f}")