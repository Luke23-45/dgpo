# FILE: scripts/tune_pid_gains.py

import argparse
import logging
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import itertools

# --- Project Imports ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from envs.panda_env import PandaEnv
from utils.scripted_expert import ScriptedExpert, ExpertConfig, ObjectProfile
from utils.ik_solver import IKSolver
from scipy.spatial.transform import Rotation as R

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("TUNER")

def evaluate_gains(
    env: PandaEnv, 
    expert: ScriptedExpert, 
    solver: IKSolver, 
    gains: tuple, 
    max_dq: float, 
    steps: int = 60 # Reduced: We only need to see Approach+Contact stability
) -> dict:
    kp, ki, kd = gains
    solver.set_gains(kp, ki, kd)
    solver.reset_controller_state()
    
    # Use fixed seed for fair comparison
    env.reset(seed=8888) 
    expert.reset()
    
    errors = []
    
    for _ in range(steps):
        obs = env.get_expert_obs()
        target_pose, grip_act, _ = expert.get_target_pose(obs)
        
        action_arm = solver.compute_delta_action(
            target_ee_pose=target_pose,
            model=env.model,
            data=env.data,
            ee_site_id=env.ee_site_id,
            joint_qpos_indices=np.arange(7),
            effective_dt=0.01, 
            max_dq=max_dq
        )
        
        env.step(np.concatenate([action_arm, [grip_act]]))
        
        # Metric: Distance to target (Accuracy)
        curr_ee = obs["ee_pose_world"]
        pos_err = np.linalg.norm(target_pose[:3] - curr_ee[:3])
        errors.append(pos_err)

        if expert.is_done(): break

    errors = np.array(errors)
    
    # Score = RMSE + (Penalty * Jitter)
    # We punish jitter (std dev) heavily because shaking ruins data
    rmse = np.sqrt(np.mean(errors**2))
    jitter = np.std(errors)
    score = rmse + (3.0 * jitter)
    
    return {"Kp": kp, "Ki": ki, "Kd": kd, "Score": score, "RMSE": rmse, "Jitter": jitter}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", default="urdf/panda_mujoco_kinematics.urdf")
    parser.add_argument("--xml", default="envs/panda_pick_place.xml")
    args = parser.parse_args()

    log.info("--- FAST PID TUNER ---")
    
    # 1. Setup Env (No Rendering = Fast)
    env = PandaEnv(xml_path=args.xml, control_mode='delta')
    env.set_rendering_enabled(False)  # <--- SPEED BOOST
    
    effective_dt = 0.01 # Fast Physics
    max_dq = env.ACTION_SCALING_FACTOR / effective_dt
    log.info(f"Physics: dt={effective_dt}s | max_dq={max_dq:.1f}")

    expert = ScriptedExpert(ObjectProfile(np.array([0.04, 0.04, 0.04]), 0.6), ExpertConfig())
    solver = IKSolver(urdf_path=args.urdf)

    # 2. Targeted Search Space
    # We search around Kp=60 (theoretical sweet spot)
    kp_range = np.arange(80, 200, 1).tolist()
    kd_range = [1.5, 2.0, 2.5, 3.0]
    ki_range = [0.0, 0.05, 0.1] 

    combinations = list(itertools.product(kp_range, ki_range, kd_range))
    log.info(f"Testing {len(combinations)} configs...")

    results = []
    pbar = tqdm(combinations, desc="Benchmarking")
    
    for kp, ki, kd in pbar:
        try:
            metrics = evaluate_gains(env, expert, solver, (kp, ki, kd), max_dq)
            results.append(metrics)
        except: pass

    # 3. Results
    df = pd.DataFrame(results).sort_values("Score", ascending=True)
    
    print("\n" + "="*60)
    print("🏆 WINNER CONFIGURATION")
    print("="*60)
    best = df.iloc[0]
    print(f"Kp: {best['Kp']} | Ki: {best['Ki']} | Kd: {best['Kd']}")
    print(f"Metrics: RMSE={best['RMSE']:.4f} | Jitter={best['Jitter']:.4f}")
    print("="*60)
    print("Top 5 Candidates:")
    print(df.head(5)[['Kp','Ki','Kd','Score','RMSE','Jitter']].to_string(index=False))

    env.close()

if __name__ == "__main__":
    main()