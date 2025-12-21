import os
import sys
import torch
import hydra
import logging
import numpy as np
import mujoco
from typing import Dict, List, Tuple
from scipy.spatial.transform import Rotation as R


# Add project root to path
sys.path.append(os.getcwd())

# Import Utils
from utils.ik_solver import IKSolver
from envs.dgpo_env_wrapper import make_dgpo_env, DGPOEnvWrapper

# Debug Imports
# Debug Imports
import inspect
print(f" [DEBUG] IKSolver loaded from: {inspect.getfile(IKSolver)}")
print(f" [DEBUG] compute_delta_action Sig: {inspect.signature(IKSolver.compute_delta_action)}")

# Setup Logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("SYS_ID")

class SystemIdentifier:
    def __init__(self, cfg):
        self.cfg = cfg
        # Force single environment for diagnostics
        self.cfg.num_envs = 1
        self.env = make_dgpo_env(self.cfg)
        self.env.reset()
        
        # Access the wrapper deeply
        # VectorEnv -> AsyncVectorEnv -> Call 'get_wrapper_attr' is hard.
        # We will assume we can get the local env if it's not async, but make_dgpo_env returns Async.
        # WORKAROUND: Create a direct local instance for white-box testing.
        from envs.dgpo_env_wrapper import DGPOEnvWrapper
        import gymnasium as gym
        
        # We need to manually instantiate the underlying Gym Env
        # This depends on how make_dgpo_env constructs it.
        # Let's trust the 'diagnose_ik_controller.py' approach was too fragile.
        # We will use the VEC ENV but extract info via the Info dict if possible, 
        # OR just instantiate the logic classes directly.
        
        # Best approach: Instantiate the Wrapper Logic Class directly without Gym interface 
        # to avoid Pickling/Async issues, provided we can mock the inputs.
        # Actually, let's just use the `env.envs[0]` trick if `num_envs=1` and `asynchronous=False`.
        # Config usually sets sync/async.
        
        # For Robustness: We will just run the loop and capture data from the 'info' dict 
        # which we will modify the wrapper to provide if needed.
        # But we want to inspect `qpos`.
        
        # Let's try to get the underlying MuJoCo model from the first env.
        # In SyncVectorEnv, env.envs[0] works. In Async, it hangs.
        # Let's assume we can create a local one.
        
        print(" [SysID] Creating Local Environment for Deep Inspection...")
        # Re-create internal env
        # self.wrapper = DGPOEnvWrapper(gym.make(cfg.environment.env_id), cfg)
        # Config doesn't have env_id, using hardcoded Panda-v0 or similar fallback 
        # based on context (make_dgpo_env uses Panda-v0 inside usually).
        # Checking dgpo_env_wrapper.py/make_dgpo_env... it creates AsyncVectorEnv of 'Panda-v0'.
        
        # Let's try minimal direct instantiation
        from envs.panda_env import PandaEnv
        self.wrapper = DGPOEnvWrapper(PandaEnv(), cfg)
        self.wrapper.reset()
        
        self.history = {
            "qpos": [],
            "action": [],
            "next_qpos": [],
            "target_pose": [],
            "jacobian_error": [],
            "execution_error": [],
            "timestamp": []
        }

    def run_diagnostics(self, steps=50):
        print(f" [SysID] Running {steps} steps of Expert Policy...")
        
        obs, _ = self.wrapper.reset()
        
        for t in range(steps):
            # 1. Capture State PRE-Step
            q_t = self.wrapper.env.data.qpos[:7].copy()
            
            # 2. Get Expert Action
            # Expert logic is internal to wrapper step, but we want to intercept it.
            # We can use the wrapper's expert directly.
            expert_obs = self.wrapper.env.get_expert_obs()
            target_pose, grip, _ = self.wrapper.expert.get_target_pose(expert_obs)
            
            # 3. Compute Action (Using the Wrapper's current controller)
            # We explicitly call the internal method to measure IT.
            try:
                # [TESTING FIX] Robust Analytical IK Logic
                # 1. Solve Global Target
                target_joint_angles = self.wrapper.ik_solver.compute_target_joint_positions(
                    target_pose_7d=target_pose,
                    current_joint_angles=q_t,
                    solution_position_tolerance=0.01
                )
                
                # 2. Reverse Engineer Delta
                scaling = self.wrapper.env.ACTION_SCALING_FACTOR
                delta_rads = target_joint_angles - q_t
                delta_joints = delta_rads / (scaling + 1e-9)
                delta_joints = np.clip(delta_joints, -1.0, 1.0)
                
            except Exception as e:
                print(f"IK Failed: {e}")
                delta_joints = np.zeros(7)
                
            action = np.concatenate([delta_joints, [grip]])
            
            # 4. Measure "IK Error" (Jacobian Effectiveness)
            # Does J @ delta_q approx V_target?
            # V_target approx (Target_Pose - Current_Pose) / dt
            curr_pos = self.wrapper.env.data.site_xpos[self.wrapper.env.ee_site_id]
            desired_vel = (target_pose[:3] - curr_pos) / self.wrapper.effective_dt
            
            # J @ dq
            jac_pos = np.zeros((3, self.wrapper.env.model.nv))
            mujoco.mj_jac(self.wrapper.env.model, self.wrapper.env.data, jac_pos, None, curr_pos, self.wrapper.env.model.site_bodyid[self.wrapper.env.ee_site_id])
            J = jac_pos[:, :7]
            # Convert normalized action to rad/s
            dq_rads = delta_joints * self.wrapper.env.ACTION_SCALING_FACTOR # This is Delta Pos
            d_vel_rads = dq_rads / self.wrapper.effective_dt # This is Velocity
            
            realized_vel = J @ d_vel_rads
            ik_err = np.linalg.norm(desired_vel - realized_vel)
            
            # 5. Step Environment
            self.wrapper.step(action)
            
            # 6. Capture State POST-Step
            q_next = self.wrapper.env.data.qpos[:7].copy()
            
            # 7. Measure "Execution Error" (Physics Feasibility)
            # Did the robot actually move by `delta_joints * Scaling`?
            # Expected: q_next = q_t + action * scaling
            # Actual: q_next
            expected_q_next = q_t + delta_joints * self.wrapper.env.ACTION_SCALING_FACTOR
            exec_err = np.linalg.norm(q_next - expected_q_next)
            
            # 8. Log
            self.history["qpos"].append(q_t)
            self.history["action"].append(delta_joints)
            self.history["next_qpos"].append(q_next)
            self.history["jacobian_error"].append(ik_err)
            self.history["execution_error"].append(exec_err)
            self.history["timestamp"].append(t * self.wrapper.effective_dt)
            
            if t % 10 == 0:
                print(f"   Step {t}: IK Err={ik_err:.4f}, Exec Err={exec_err:.4f}")
                
        self.analyze_results()

    def analyze_results(self):
        print("\n" + "="*50)
        print(" 📊 SYSTEM IDENTIFICATION REPORT")
        print("="*50)
        
        ik_errs = np.array(self.history["jacobian_error"])
        exec_errs = np.array(self.history["execution_error"])
        
        mean_ik = np.mean(ik_errs)
        max_ik = np.max(ik_errs)
        
        mean_exec = np.mean(exec_errs)
        max_exec = np.max(exec_errs)
        
        print(f" 1. CONTROLLER ACCURACY (Jacobian Linearity)")
        print(f"    Mean Error: {mean_ik:.4f} m/s")
        print(f"    Max Error:  {max_ik:.4f} m/s")
        print(f"    interpretation: High error means the Linear Jacobian approximation is invalid (Non-linear geometric jump).")
        
        print(f"\n 2. PHYSICS FIDELITY (Servo Tracking)")
        print(f"    Mean Error: {mean_exec:.4f} rad")
        print(f"    Max Error:  {max_exec:.4f} rad")
        print(f"    Interpretation: High error means the Low-Level Controller (PID/Impedance) cannot track the command (Gravity, Friction, Limits).")
        
        # Diagnosis Logic
        print("\n 🔍 DIAGNOSIS MATRIX:")
        if mean_ik > 0.1:
            print("    [CRITICAL] JACOBIAN FAILURE DETECTED.")
            print("    The Linear Controller cannot describe the required motion.")
            print("    -> Causes: Singularity, Manifold Jump (e.g. Elbow Flip), or Large Step Size.")
            print("    -> Fix: Use Global Analytical IK (as proposed).")
            
        if mean_exec > 0.05:
            print("    [CRITICAL] PHYSICS SATURATION DETECTED.")
            print("    The robot motors physically cannot execute the command.")
            print("    -> Causes: Action Scaling too high, PID gains too low, or Joint Limits.")
            print("    -> Fix: Tuning PID gains or Reducing Action Scale.")
            
        if mean_ik < 0.1 and mean_exec < 0.05:
            print("    [GREEN] SYSTEM NOMINAL.")
            print("    If Divergence is still high, the issue is likely LATENCY or OBSERVATION NOISE.")
            
        print("="*50 + "\n")

@hydra.main(config_path="../configs", config_name="train_dgpo_config", version_base="1.2")
def main(cfg):
    sys_id = SystemIdentifier(cfg)
    sys_id.run_diagnostics()

if __name__ == "__main__":
    main()
