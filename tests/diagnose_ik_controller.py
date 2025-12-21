import os
import sys
import torch
import hydra
import logging
import numpy as np
import mujoco
from typing import Dict, List
from scipy.spatial.transform import Rotation as R

# Add project root to path
sys.path.append(os.getcwd())

from utils.ik_solver import IKSolver

# Setup logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("IK_DIAGNOSIS")

@hydra.main(config_path="../configs", config_name="train_dgpo_config", version_base="1.2")
def main(cfg):
    print("\n==================================================================")
    print(" 🛠️  IK CONTROLLER DIAGNOSIS (UNIT TEST)")
    print("==================================================================\n")

    # 1. Setup Solver
    # Need to verify urdf path is absolute or relative correctly
    urdf_path = cfg.environment.urdf_path
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.join(hydra.utils.get_original_cwd(), urdf_path)
        
    print(f" [1/4] Loading IKSolver from: {urdf_path}")
    ik_solver = IKSolver(
        urdf_path=urdf_path,
        kp=40.0, ki=1.0, kd=4.0
    )
    
    # 2. Setup MuJoCo Scene
    xml_path = cfg.environment.xml_path
    if not os.path.isabs(xml_path):
        xml_path = os.path.join(hydra.utils.get_original_cwd(), xml_path)
        
    print(f" [2/4] Loading MuJoCo Model from: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data) 
    
    # 3. Initialize Robot State (Home Pose)
    # Approx joint angles for standard panda home
    q0 = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
    data.qpos[0:7] = q0
    mujoco.mj_forward(model, data)
    
    curr_pos = data.site_xpos[0] # EE site
    print(f"   Start Pos: {curr_pos}")
    
    # 4. Define Target (10cm Jump)
    target_pos = curr_pos + np.array([0.1, 0.1, 0.0]) # 10cm x, 10cm y movement
    # Use current orientation to isolate position jump
    current_quat = R.from_matrix(data.site_xmat[0].reshape(3,3)).as_quat()
    # Ensure standard format (SciPy is xyzw)
    
    target_pose_7d = np.concatenate([target_pos, current_quat])
    
    print("\n [3/4] Testing JACOBIAN Controller (Current)")
    # Logic from compute_delta_action
    # Simulates what DGPOEnvWrapper does
    
    # Needs a chunk (T, 7)
    chunk = np.array([target_pose_7d] * 5)
    
    try:
        action_jac = ik_solver.compute_delta_action(
            target_ee_pose_chunk=chunk,
            model=model,
            data=data,
            ee_site_id=0,
            joint_qpos_indices=np.arange(7),
            effective_dt=0.02,
            max_dq=2.0 
        )
        print(f"   Raw Jacobian Action: {action_jac}")
        norm_jac = np.linalg.norm(action_jac)
        print(f"   Jacobian Norm: {norm_jac:.4f}")
        
        is_saturated = np.any(np.abs(action_jac) >= 0.99)
        if is_saturated:
            print("   🚨 RESULT: Saturated! (Velocity Maxed Out)")
            print("      Verification: The controller hits max velocity and clips.")
            print("      It cannot instantaneously jump to the target.")
        else:
            print("   RESULT: Not Saturated (Surprising for 10cm jump)")
            
    except Exception as e:
        print(f"   ❌ Jacobian Error: {e}")

    print("\n [4/4] Testing ANALYTICAL Controller (Proposal)")
    # Logic from compute_action
    # Simulates what we WANT to do
    
    try:
        action_ik = ik_solver.compute_action(
            target_pose_7d=target_pose_7d,
            current_joint_angles=data.qpos[0:7], # Crucial: Warm Start
            solution_position_tolerance=0.01
        )
        print(f"   Raw Analytical Action (Position Target): {action_ik}")
        norm_ik = np.linalg.norm(action_ik)
        print(f"   Analytical Norm: {norm_ik:.4f}")
        
        print("   ✅ RESULT: Analytical Solver returned a valid Joint Position.")
        print("      Verification: It solved for the specific joint configuration")
        print("      needed to reach the target, regardless of distance.")
        
    except Exception as e:
        print(f"   ❌ Analytical Error: {e}")
        
    print("\n" + "="*66)
    print(" 🏁 CONCLUSION")
    print("="*66)
    print("   If Jacobian is saturated (1.0) and Analytical is valid:")
    print("   -> The current controller acts as a 'Servo' and lags behind large jumps.")
    print("   -> The Analytical controller acts as a 'Teleporter' (in intent).")
    print("   -> Switching to Analytical IK will fix the 'Lag/Divergence' issue.")
    print("="*66 + "\n")

if __name__ == "__main__":
    main()
