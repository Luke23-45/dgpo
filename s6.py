# FILE: scripts/verify_expert_trajectory.py

"""
A focused script to verify that the combination of PandaEnv and ScriptedExpert
can produce a complete, successful pick-and-place trajectory.

This test bypasses the complex ExpertDataset iterator and interacts directly
with the core components to isolate and validate their behavior.

It runs one full episode and:
1. Prints detailed step-by-step diagnostics to the console.
2. Saves a video of the expert's performance for visual confirmation.

If this script produces a video of a successful pick-and-place, it proves
that the environment and the expert are working correctly.
"""
import argparse
import logging
from pathlib import Path
import mujoco
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils.scripted_expert import ScriptedExpert, ObjectProfile
# --- Project Imports ---
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from envs.panda_env import PandaEnv
from utils.scripted_expert import ScriptedExpert
from utils.ik_solver import IKSolver


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | [%(name)s] | %(message)s"
)
log = logging.getLogger("VERIFY_TRAJECTORY")


def main(args: argparse.Namespace):
    log.info("--- Starting Expert Trajectory Verification Script ---")

    output_dir = Path("verification_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"expert_trajectory_seed{args.seed}.mp4"
    object_to_grasp = ObjectProfile(
        size=np.array([0.04, 0.04, 0.04]),
        grasp_width_normalized=0.6 # Close most of the way but not fully
    )
    # --- 1. Initialize Core Components ---
    log.info("Initializing components...")
    env = PandaEnv(xml_path=args.xml_path)
    model = env.model


    expert = ScriptedExpert(object_profile=object_to_grasp)
    ik_solver = IKSolver(urdf_path=args.urdf_path)
    log.info("Components initialized.")
    
    # === START: CORRECTED JOINT ORDER DIAGNOSTIC ===
    log.info("--- Verifying Joint Order Between IK Solver and MuJoCo ---")
    ik_joint_names = list(ik_solver.joint_names())
    mj_joint_names = []
    
    # Get the first 7 joint names from the MuJoCo model's actuators
    try:
        # The arm actuators are the first 7 in the model
        for i in range(7):
            actuator_id = i
            joint_id = env.model.actuator_trnid[actuator_id, 0]
            joint_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            # Modern mujoco bindings return str, so .decode() is not needed and will crash.
            mj_joint_names.append(joint_name)
    except Exception as e:
        log.error(f"Failed to extract joint names from MuJoCo model. Error: {e}", exc_info=True)
    
    if ik_joint_names == mj_joint_names:
        log.info(f"✅ Joint order matches: {ik_joint_names}")
    else:
        log.error("❌ JOINT ORDER MISMATCH!")
        log.error(f"  - IK Solver expects: {ik_joint_names}")
        log.error(f"  - MuJoCo model has: {mj_joint_names}")
    # === END: CORRECTED JOINT ORDER DIAGNOSTIC ===

    # --- 2. Setup Video Recording ---
    log.info(f"Setting up video recorder. Frame size will be determined by first render.")
    # Render once to get frame dimensions
    frame = env.render()
    h, w, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(video_path), fourcc, 30, (w, h))

    # --- 3. Run One Full Episode ---
    log.info(f"Starting episode generation. Video will be saved to: {video_path}")
    env.set_object_size(object_to_grasp.size)
    expert.reset()
    obs, _ = env.reset(seed=args.seed)

    try:
        for step_num in range(env.max_episode_steps):
            # Get the rich observation dictionary needed by the expert
            expert_obs = env.get_expert_obs()

            # Get the target pose and gripper command from our stateful expert
            target_pose, gripper_action = expert.get_target_pose(expert_obs)

            log.info(f"--- Step {step_num} | Expert State: {expert.get_state()} ---")
            # log.info(f"   Object Position (World): {np.round(expert_obs['object_pos_world'], 3)}")
            # log.info(f"   Object Orientation (World): {np.round(expert_obs['object_orn_world'], 3)}")
            # Convert the target pose into a joint action via IK
            base_pos, base_quat = env.get_base_pose()
            R_world_base = R.from_quat(base_quat)
            R_base_world = R_world_base.inv()
            pos_in_base = R_base_world.apply(target_pose[:3] - base_pos)
            rot_in_base = R_base_world * R.from_quat(target_pose[3:7])
            target_pose_base = np.concatenate([pos_in_base, rot_in_base.as_quat()]).astype(np.float32)

            current_joints = expert_obs["internal_full_proprio"][:7]
            arm_action = ik_solver.compute_action(target_pose_base, current_joints)
            
            # --- START OF FIX ---
            # The expert already outputs a normalized [-1, 1] gripper action.
            # The environment is designed to accept this directly.
            # The `gripper_action_to_ctrl` helper was incorrect and has been removed.
            final_action = np.concatenate([arm_action, [gripper_action]])
            # --- END OF FIX ---
            
            # Step the environment with the calculated expert action
            obs, _, terminated, truncated, _ = env.step(final_action)
            
            # Render and write frame
            frame_rgb = env.render()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            state_text = f"State: {expert.get_state()}"
            cv2.putText(
                img=frame_bgr,
                text=state_text,
                org=(10, 30),  # Position (bottom-left corner of text)
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(255, 205, 100),  # White color in BGR
                thickness=2,
                lineType=cv2.LINE_AA
            )
            video_writer.write(frame_bgr)

            # Check for episode completion
            if expert.is_done() or terminated or truncated:
                log.info(f"Episode finished at step {step_num}. Final expert state: {expert.get_state()}")
                break
        
        # Add a few final frames to the video for padding
        for _ in range(30): video_writer.write(frame_bgr)

    finally:
        log.info("Releasing resources...")
        video_writer.release()
        env.close()

    if expert.is_done():
        log.info("✅ SUCCESS: The expert successfully completed its state machine and the episode.")
    else:
        log.error("❌ FAILURE: The expert did not reach the 'DONE' state.")
    log.info("--- Verification complete. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify the PandaEnv+ScriptedExpert trajectory generation.")
    parser.add_argument("--urdf_path", type=str, default="urdf/panda_mujoco_kinematics.urdf")
    parser.add_argument("--xml_path", type=str, default="envs/panda_pick_place.xml")
    parser.add_argument("--seed", type=int, default=813)
    args = parser.parse_args()
    main(args)