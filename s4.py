# FILE: scripts/verify_delta_conversion.py
"""
A definitive script to verify the absolute-to-delta action conversion logic.

This test is critical for ensuring the expert data we generate is compatible
with the delta-controlled environment our RL agent will be trained in.

It works as follows:
1.  Instantiates a PandaEnv in 'delta' control mode.
2.  At each step, it runs the full expert pipeline (ScriptedExpert + IKSolver)
    to determine the ideal ABSOLUTE joint position action.
3.  It then performs the mathematical conversion from this absolute action to the
    equivalent DELTA action.
4.  It steps the 'delta'-controlled environment with this DELTA action.
5.  It saves a video of the resulting trajectory.

If the video shows a successful pick-and-place, it proves that our action
space translation layer is correct and the data in expert_demos.pkl is valid.
"""
import argparse
import logging
from pathlib import Path
import mujoco
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# --- Project Imports ---
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from envs.panda_env import PandaEnv
from utils.scripted_expert import ScriptedExpert, ObjectProfile
from utils.ik_solver import IKSolver

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | [%(name)s] | %(message)s"
)
log = logging.getLogger("VERIFY_DELTA_CONVERSION")

def convert_abs_to_delta(
    abs_arm_action: np.ndarray,
    current_qpos: np.ndarray,
    actuator_ctrlrange: np.ndarray,
    scaling_factor: float
) -> np.ndarray:
    """
    Converts a normalized absolute arm action into a normalized delta arm action.
    This function contains the critical translation logic.
    """
    # 1. De-normalize the absolute action to get the target physical joint positions
    arm_lo, arm_hi = actuator_ctrlrange[:, 0], actuator_ctrlrange[:, 1]
    physical_target_qpos = arm_lo + 0.5 * (abs_arm_action + 1.0) * (arm_hi - arm_lo)
    
    # 2. Calculate the required physical delta
    required_physical_delta = physical_target_qpos - current_qpos
    
    # 3. Normalize the delta to get the final delta action
    delta_arm_action = required_physical_delta / scaling_factor
    
    return np.clip(delta_arm_action, -1.0, 1.0)


def main(args: argparse.Namespace):
    log.info("--- Starting Absolute-to-Delta Action Conversion Verification ---")

    output_dir = Path("verification_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"delta_conversion_seed{args.seed}.mp4"
    object_to_grasp = ObjectProfile(
        size=np.array([0.04, 0.04, 0.04]),
        grasp_width_normalized=1.0
    )
    
    # --- 1. Initialize Components ---
    log.info("Initializing components for delta control test...")
    # The main environment we will step and record is in 'delta' mode.
    env_delta = PandaEnv(xml_path=args.xml_path, control_mode='delta')
    
    # We need the actuator ctrlrange for the conversion math. We can get this from the model.
    actuator_ctrlrange = env_delta.model.actuator_ctrlrange[:7]
    
    expert = ScriptedExpert(object_profile=object_to_grasp)
    ik_solver = IKSolver(urdf_path=args.urdf_path)
    log.info("Components initialized.")

    # --- 2. Setup Video Recording ---
    frame = env_delta.render()
    h, w, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(video_path), fourcc, 30, (w, h))
    log.info(f"Video will be saved to: {video_path}")

    # --- 3. Run One Full Episode ---
    env_delta.set_object_size(object_to_grasp.size)
    expert.reset()
    obs, _ = env_delta.reset(seed=args.seed)

    try:
        for step_num in range(env_delta.max_episode_steps):
            expert_obs = obs # In this script, the env obs is the expert obs
            
            # --- Get Expert Command (in Absolute Space) ---
            target_pose_world, gripper_action = expert.get_target_pose(expert_obs)

            # --- Convert to IK Target and get Absolute Action ---
            base_pos, base_quat = env_delta.get_base_pose()
            R_world_base_inv = R.from_quat(base_quat).inv()
            pos_in_base = R_world_base_inv.apply(target_pose_world[:3] - base_pos)
            rot_in_base = R_world_base_inv * R.from_quat(target_pose_world[3:7])
            target_pose_base = np.concatenate([pos_in_base, rot_in_base.as_quat()])
            
            current_joints = expert_obs["proprio"][:7]
            absolute_arm_action = ik_solver.compute_action(target_pose_base, current_joints)
            
            # --- CRITICAL STEP: Convert Absolute Action to Delta Action ---
            delta_arm_action = convert_abs_to_delta(
                absolute_arm_action,
                current_joints,
                actuator_ctrlrange,
                env_delta.ACTION_SCALING_FACTOR
            )
            
            final_delta_action = np.concatenate([delta_arm_action, [gripper_action]])

            # Step the DELTA environment with the calculated DELTA action
            obs, _, terminated, truncated, _ = env_delta.step(final_delta_action)
            
            # --- Logging and Rendering ---
            frame_rgb = env_delta.render()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            state_text = f"State: {expert.get_state()}"
            cv2.putText(
                img=frame_bgr, text=state_text, org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA
            )
            video_writer.write(frame_bgr)

            if expert.is_done() or terminated or truncated:
                log.info(f"Episode finished at step {step_num}. Final expert state: {expert.get_state()}")
                break
        
        for _ in range(30): video_writer.write(frame_bgr)

    finally:
        log.info("Releasing resources...")
        video_writer.release()
        env_delta.close()

    if expert.was_successful():
        log.info("✅ SUCCESS: The delta-controlled environment successfully followed the expert's trajectory.")
    else:
        log.error("❌ FAILURE: The expert did not complete its trajectory in the delta-controlled environment.")
    log.info("--- Conversion verification complete. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify the absolute-to-delta action conversion.")
    parser.add_argument("--urdf_path", type=str, default="urdf/panda_mujoco_kinematics.urdf")
    parser.add_argument("--xml_path", type=str, default="envs/panda_pick_place.xml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)