# FILE: scripts/tune_pid_optimized.py (New, Superior Version)
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- NEW: Imports for Bayesian Optimization ---
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# --- Project Imports ---
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from envs.panda_env import PandaEnv
from utils.ik_solver import IKSolver


# NOTE: The helper functions calculate_metrics and run_single_test are assumed
# to be the same as the robust versions from the previous step. I will include
# them here for completeness.

def calculate_metrics(log_data: pd.DataFrame, target_z: float, settling_threshold: float = 0.02):
    """Calculates key performance metrics from a trajectory log."""
    max_z = log_data['ee_pos_z'].max()
    initial_z = log_data['ee_pos_z'].iloc[0]
    z_range = abs(target_z - initial_z)
    overshoot_abs = max(0, max_z - target_z)
    overshoot_pct = (overshoot_abs / z_range) * 100.0 if z_range > 1e-6 else 0.0

    settling_band_half_width = z_range * settling_threshold
    settled = np.abs(log_data['ee_pos_z'] - target_z) < settling_band_half_width
    unsettled_indices = np.where(settled == False)[0]
    if len(unsettled_indices) == 0:
        settling_time = log_data.loc[settled.idxmax(), 'time'] if settled.any() else log_data['time'].max()
    else:
        last_unsettled_idx = unsettled_indices[-1]
        settling_time = log_data['time'].iloc[last_unsettled_idx + 1] if last_unsettled_idx + 1 < len(log_data) else log_data['time'].max()

    last_10_percent_idx = int(len(log_data) * 0.9)
    steady_state_error = np.abs(target_z - log_data['ee_pos_z'].iloc[last_10_percent_idx:]).mean()
    control_effort = (log_data['action_norm']**2).mean()

    return {
        "overshoot_pct": overshoot_pct,
        "settling_time_s": settling_time,
        "steady_state_error_m": steady_state_error,
        "mean_control_effort": control_effort
    }

def run_single_test(env: PandaEnv, ik_solver: IKSolver, gains: dict, test_params: dict):
    """Runs one full simulation with a given set of PID gains."""
    obs, _ = env.reset(seed=test_params["seed"])
    ik_solver.reset_controller_state()
    ik_solver.set_gains(kp=gains["kp"], ki=gains["ki"], kd=gains["kd"])

    start_pose = obs["ee_pose_world"]
    target_pos = start_pose[:3] + np.array([0, 0, test_params["z_move_dist"]])
    target_ee_pose = np.concatenate([target_pos, start_pose[3:]])
    
    log_entries = []
    
    N_SUBSTEPS = 20
    effective_dt = env.model.opt.timestep * N_SUBSTEPS
    arm_joint_ids = np.arange(7)
    max_dq = env.ACTION_SCALING_FACTOR / effective_dt

    for step in range(test_params["num_steps"]):
        arm_action = ik_solver.compute_delta_action(
            target_ee_pose=target_ee_pose,
            model=env.model, data=env.data, ee_site_id=env.ee_site_id,
            joint_qpos_indices=arm_joint_ids,
            effective_dt=effective_dt, max_dq=max_dq
        )
        action = np.concatenate([arm_action, [1.0]])
        obs, _, _, _, _ = env.step(action)
        log_entries.append({
            "time": step * effective_dt,
            "ee_pos_z": obs["ee_pose_world"][2],
            "action_norm": np.linalg.norm(arm_action)
        })

    return pd.DataFrame(log_entries), target_pos[2]


# --- MAIN SCRIPT USING BAYESIAN OPTIMIZATION ---

def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger("PID_OPTIMIZER")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    env = PandaEnv(xml_path=args.xml_path, control_mode='delta')
    ik_solver = IKSolver(urdf_path=args.urdf_path)

    # --- 1. DEFINE THE PARAMETER SEARCH SPACE ---
    # We define a continuous range for each parameter.
    search_space = [
        Real(0.0, 40.0, name='kp'),
        Real(0.1, 15.0, name='kd'),
        Real(4.0, 15.0, name='ki')
    ]
    
    # --- 2. DEFINE THE OBJECTIVE FUNCTION ---
    # This function takes the gains and returns the cost. It's what the optimizer will call.
    @use_named_args(search_space)
    def objective_function(kp, kd, ki):
        gains = {"kp": kp, "kd": kd, "ki": ki}
        
        test_params = {"z_move_dist": 0.3, "num_steps": 400, "seed": 42}
        cost_weights = {"overshoot_pct": 2.0, "settling_time_s": 1.0, "steady_state_error_m": 100.0, "mean_control_effort": 0.5}

        log_df, target_z = run_single_test(env, ik_solver, gains, test_params)
        metrics = calculate_metrics(log_df, target_z)
        
        cost = (cost_weights["overshoot_pct"] * metrics["overshoot_pct"] +
                cost_weights["settling_time_s"] * metrics["settling_time_s"] +
                cost_weights["steady_state_error_m"] * metrics["steady_state_error_m"] +
                cost_weights["mean_control_effort"] * metrics["mean_control_effort"])
        
        log.info(f"Tested Kp={kp:.2f}, Kd={kd:.2f}, Ki={ki:.2f} -> Cost={cost:.4f}")
        return cost

    # --- 3. RUN THE OPTIMIZATION ---
    log.info(f"Starting Bayesian Optimization with {args.n_calls} evaluations.")
    result = gp_minimize(
        objective_function,
        search_space,
        n_calls=args.n_calls,
        n_initial_points=5, # Start with 5 random points to build the initial map
        random_state=args.seed
    )

    # --- 4. ANALYZE AND REPORT RESULTS ---
    best_gains = {dim.name: val for dim, val in zip(search_space, result.x)}
    best_cost = result.fun

    log.info("\n--- Bayesian Optimization Results ---")
    log.info(f"Optimization finished after {args.n_calls} evaluations.")
    log.info(f"Optimal Gains Found: Kp={best_gains['kp']:.3f}, Kd={best_gains['kd']:.3f}, Ki={best_gains['ki']:.3f}")
    log.info(f"Associated Minimum Cost: {best_cost:.4f}")
    
    # Save results to a file
    with open(output_dir / "optimal_gains.txt", "w") as f:
        f.write(f"Optimal Kp: {best_gains['kp']}\n")
        f.write(f"Optimal Kd: {best_gains['kd']}\n")
        f.write(f"Optimal Ki: {best_gains['ki']}\n")
        f.write(f"Best Cost: {best_cost}\n")

    log.info("Visualizing the response of the BEST controller found...")
    final_log_df, target_z = run_single_test(env, ik_solver, best_gains, {"z_move_dist": 0.3, "num_steps": 400, "seed": 42})
    
    plt.figure(figsize=(12, 6))
    plt.plot(final_log_df['time'], final_log_df['ee_pos_z'], label=f"Best Response (Kp={best_gains['kp']:.2f}, Ki={best_gains['ki']:.2f}, Kd={best_gains['kd']:.2f})")
    plt.axhline(y=target_z, color='r', linestyle='--', label="Target Position")
    plt.title("Step Response of Bayesian-Optimized PID Controller")
    plt.xlabel("Time (s)")
    plt.ylabel("End-Effector Z Position (m)")
    plt.grid(True)
    plt.legend()
    plot_path = output_dir / "best_response_optimized.png"
    plt.savefig(plot_path)
    log.info(f"Saved best response plot to {plot_path}")
    plt.show()

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Optimization PID Tuner for the Panda Arm")
    parser.add_argument("--urdf_path", type=str, default="urdf/panda_mujoco_kinematics.urdf")
    parser.add_argument("--xml_path", type=str, default="envs/panda_pick_place.xml")
    parser.add_argument("--output_dir", type=str, default="pid_optimization_output")
    parser.add_argument("--n_calls", type=int, default=30, help="Number of simulations to run for optimization.")
    parser.add_argument("--seed", type=int, default=123)
    
    args = parser.parse_args()
    main(args)