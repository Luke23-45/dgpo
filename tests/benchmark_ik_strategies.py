import os
import sys
import hydra
import logging
import numpy as np
import mujoco

# Fix Import Path
sys.path.append(os.getcwd())

from typing import Dict, List, Tuple
from envs.panda_env import PandaEnv
from envs.dgpo_env_wrapper import DGPOEnvWrapper
from utils.ik_solver import IKSolver

# Setup Logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("IK_BENCH")

class IKBenchmark:
    def __init__(self, cfg):
        self.cfg = cfg
        # Direct Instantiation for Whitebox testing
        self.env = PandaEnv()
        self.wrapper = DGPOEnvWrapper(self.env, cfg)
        self.wrapper.reset()
        
        # Strategies to Test
        self.strategies = {
            "Jacobian (Baseline)": self.run_jacobian,
            "Analytical Snap (Current)": self.run_analytical_snap,
            "Analytical Smooth (0.5)": lambda t, c: self.run_analytical_smooth(t, c, 0.5)
        }
        
    def run_jacobian(self, target_pose, q_current):
        """Standard Jacobian Delta Control"""
        # Note: We must use the correct singular argument
        try:
            delta_joints = self.wrapper.ik_solver.compute_delta_action(
                target_ee_pose=target_pose, 
                model=self.wrapper.env.model,
                data=self.wrapper.env.data,
                ee_site_id=self.wrapper.env.ee_site_id,
                joint_qpos_indices=np.arange(7),
                effective_dt=self.wrapper.effective_dt,
                max_dq=self.wrapper.env.ACTION_SCALING_FACTOR / self.wrapper.effective_dt
            )
        except Exception:
            delta_joints = np.zeros(7)
        return delta_joints

    def run_analytical_snap(self, target_pose, q_current):
        """Solve Global + Full Speed Command"""
        target_q = self.wrapper.ik_solver.compute_target_joint_positions(
            target_pose_7d=target_pose,
            current_joint_angles=q_current,
            solution_position_tolerance=0.01
        )
        scale = self.wrapper.env.ACTION_SCALING_FACTOR
        # Formula: action = (target - current) / scale
        delta_rads = target_q - q_current
        delta_joints = delta_rads / (scale + 1e-9)
        return np.clip(delta_joints, -1.0, 1.0)

    def run_analytical_smooth(self, target_pose, q_current, alpha=0.5):
        """Solve Global + Partial Speed Command (Smoothing)"""
        target_q = self.wrapper.ik_solver.compute_target_joint_positions(
            target_pose_7d=target_pose,
            current_joint_angles=q_current,
            solution_position_tolerance=0.01
        )
        scale = self.wrapper.env.ACTION_SCALING_FACTOR
        delta_rads = target_q - q_current
        
        # Smoothing: Only command a fraction of the distance
        smoothed_delta = delta_rads * alpha
        
        delta_joints = smoothed_delta / (scale + 1e-9)
        return np.clip(delta_joints, -1.0, 1.0)

    def evaluate_strategy(self, name, strategy_fn, steps=50):
        print(f"\n[BENCHMARK] Testing: {name}...")
        self.wrapper.reset()
        
        history = {
            "ik_err": [],
            "exec_err": [],
            "pos_err": [] # Cartesian distance to goal
        }
        
        for t in range(steps):
            q_t = self.wrapper.env.data.qpos[:7].copy()
            curr_pos = self.wrapper.env.data.site_xpos[self.wrapper.env.ee_site_id]
            
            # 1. Get Expert Goal
            expert_obs = self.wrapper.env.get_expert_obs()
            target_pose, grip, _ = self.wrapper.expert.get_target_pose(expert_obs)
            
            # 2. Compute Action using Strategy
            delta_joints = strategy_fn(target_pose, q_t)
            action = np.concatenate([delta_joints, [grip]])
            
            # 3. Measure Theoretical Velocity (Intent)
            # Desired Vel (Ideal)
            desired_vel = (target_pose[:3] - curr_pos) / self.wrapper.effective_dt
            
            # Realized Vel (Command)
            # We need Jacobian to map dq -> dx
            jac_pos = np.zeros((3, self.wrapper.env.model.nv))
            mujoco.mj_jac(self.wrapper.env.model, self.wrapper.env.data, jac_pos, None, curr_pos, self.wrapper.env.model.site_bodyid[self.wrapper.env.ee_site_id])
            J = jac_pos[:, :7]
            
            cmd_dq_rads = delta_joints * self.wrapper.env.ACTION_SCALING_FACTOR # Delta
            cmd_vel_rads = cmd_dq_rads / self.wrapper.effective_dt
            
            cmd_cart_vel = J @ cmd_vel_rads
            ik_err = np.linalg.norm(desired_vel - cmd_cart_vel)
            
            # 4. Step
            self.wrapper.step(action)
            q_next = self.wrapper.env.data.qpos[:7].copy()
            
            # 5. Measure Physics Execution
            expected_q_next = q_t + delta_joints * self.wrapper.env.ACTION_SCALING_FACTOR
            exec_err = np.linalg.norm(q_next - expected_q_next)
            
            # 6. Measure Cartesian Pos Error (Lag)
            new_pos = self.wrapper.env.data.site_xpos[self.wrapper.env.ee_site_id]
            pos_err = np.linalg.norm(target_pose[:3] - new_pos)
            
            history["ik_err"].append(ik_err)
            history["exec_err"].append(exec_err)
            history["pos_err"].append(pos_err)
            
        # Stats
        mean_ik = np.mean(history["ik_err"])
        mean_exec = np.mean(history["exec_err"])
        mean_pos = np.mean(history["pos_err"])
        
        print(f"  > Mean IK Error (Command Quality): {mean_ik:.4f} m/s")
        print(f"  > Mean Exec Error (Physics Lag):   {mean_exec:.4f} rad")
        print(f"  > Mean Pos Error (Goal Distance):  {mean_pos:.4f} m")
        return mean_ik, mean_exec, mean_pos

    def run(self):
        results = {}
        for name, fn in self.strategies.items():
            results[name] = self.evaluate_strategy(name, fn)
            
        print("\n" + "="*60)
        print(" 🏆 FINAL RESULTS TABLE")
        print("="*60)
        print(f"{'Strategy':<30} | {'Cmd Err (m/s)':<15} | {'Lag (m)':<10}")
        print("-" * 60)
        
        best_lag = float('inf')
        winner = ""
        
        for name, (ik, exc, pos) in results.items():
            print(f"{name:<30} | {ik:.4f}          | {pos:.4f}")
            if pos < best_lag:
                best_lag = pos
                winner = name
                
        print("="*60)
        print(f" WINNER: {winner} (Lowest Lag/Divergence)")

@hydra.main(config_path="../configs", config_name="train_dgpo_config", version_base="1.2")
def main(cfg):
    bench = IKBenchmark(cfg)
    bench.run()

if __name__ == "__main__":
    main()
