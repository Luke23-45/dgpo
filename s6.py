# FILE: scripts/verify_expert_trajectory.py

"""
A focused, high-speed script to verify that the PandaEnv and ScriptedExpert
can physically complete a pick-and-place task using DELTA CONTROL.

This validates the entire pipeline:
1. Expert Logic (State Machine)
2. IK Solver (Tuned PID Controller)
3. Physics Integration (Delta Actions)

OPTIMIZATIONS:
- Smart Logging: Reduces I/O overhead by only printing state transitions.
- Direct Math: Bypasses unnecessary conversions.
"""

import argparse
import logging
from pathlib import Path
import mujoco
import cv2
import numpy as np
import sys

# --- Project Imports ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from envs.panda_env import PandaEnv
from utils.scripted_expert import ScriptedExpert, ObjectProfile, ExpertConfig
from utils.ik_solver import IKSolver

# Configure minimal, high-speed logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("VERIFY_DELTA")

def main(args: argparse.Namespace):
    log.info("--- Starting Optimized Delta Verification ---")

    # 1. Setup Output
    output_dir = Path("verification_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"delta_trajectory_seed{args.seed}.mp4"

    # 2. Config
    object_to_grasp = ObjectProfile(
        size=np.array([0.04, 0.04, 0.04]),
        grasp_width_normalized=0.6
    )
    
    # Expert Config (Patient timeouts)
    expert_config = ExpertConfig()

    # 3. Initialize Components (DELTA MODE)
    env = PandaEnv(xml_path=args.xml_path, control_mode='delta')
    
    # Physics Synchronization (Fast Physics Settings)
    # N_SUBSTEPS=5 (Standard) -> 0.002 * 5 = 0.01s
    effective_dt = 0.01 
    max_dq = env.ACTION_SCALING_FACTOR / effective_dt
    arm_joint_ids = np.arange(7)
    
    log.info(f"Physics Config: dt={effective_dt}s | Scale={env.ACTION_SCALING_FACTOR} | max_dq={max_dq:.1f}")

    expert = ScriptedExpert(object_profile=object_to_grasp, cfg=expert_config)
    ik_solver = IKSolver(urdf_path=args.urdf_path)
    
    log.info("Components initialized. Control Mode: DELTA")

    # 4. Diagnostic: Verify Joint Mapping
    ik_joints = list(ik_solver.joint_names())
    mj_joints = [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, env.model.actuator_trnid[i, 0]) for i in range(7)]
    
    if ik_joints != mj_joints:
        log.error("❌ CRITICAL JOINT MISMATCH")
        log.error(f"IK: {ik_joints}")
        log.error(f"MJ: {mj_joints}")
        return
    else:
        log.info("✅ Joint mapping verified.")

    # 5. Reset Episode
    env.set_object_size(object_to_grasp.size)
    expert.reset()
    ik_solver.reset_controller_state() # Important for PID history
    obs, _ = env.reset(seed=args.seed)
    
    # Video Setup
    frame0 = env.render()
    h, w, _ = frame0.shape
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    
    log.info(f"Running episode (Max steps: {env.max_episode_steps})...")
    
    last_state = None
    
    try:
        for step in range(env.max_episode_steps):
            # A. Expert Logic (Get Target Cartesian Pose)
            expert_obs = env.get_expert_obs()
            target_pose, gripper_act, _ = expert.get_target_pose(expert_obs)
            
            # SMART LOGGING: Only print when state changes to reduce lag
            current_state = expert.get_state()
            if current_state != last_state:
                log.info(f"Step {step:03d} | State Transition: {last_state} -> {current_state}")
                last_state = current_state

            # B. Compute Delta Action (Using Tuned PID Controller)
            arm_delta = ik_solver.compute_delta_action(
                target_ee_pose=target_pose,
                model=env.model,
                data=env.data,
                ee_site_id=env.ee_site_id,
                joint_qpos_indices=arm_joint_ids,
                effective_dt=effective_dt,
                max_dq=max_dq
            )
            
            # C. Step Environment
            final_action = np.concatenate([arm_delta, [gripper_act]])
            obs, _, terminated, truncated, _ = env.step(final_action)
            
            # D. Render (Overlay Info)
            frame = env.render()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            cv2.putText(frame, f"State: {current_state}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Step: {step}", (10, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            writer.write(frame)

            if expert.is_done() or terminated or truncated:
                log.info(f"Episode ended at step {step}. Final State: {current_state}")
                break
                
    finally:
        writer.release()
        env.close()
        
    if expert.was_successful():
        log.info(f"✅ VERIFICATION PASSED. Video: {video_path}")
    else:
        log.error(f"❌ VERIFICATION FAILED. Expert got stuck in: {expert.get_state()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf_path", type=str, default="urdf/panda_mujoco_kinematics.urdf")
    parser.add_argument("--xml_path", type=str, default="envs/panda_pick_place.xml")
    parser.add_argument("--seed", type=int, default=8888)
    args = parser.parse_args()
    main(args)