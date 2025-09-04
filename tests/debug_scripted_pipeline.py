# In file: scripts/debug_scripted_pipeline.py
import numpy as np
import logging
import os
import time
from PIL import Image
from scipy.spatial.transform import Rotation as R
import cv2  # Import OpenCV

# --- Project Imports ---
from envs.panda_env import PandaEnv
from utils.scripted_expert import ScriptedExpert, ExpertConfig
from utils.ik_solver import IKSolver
from utils.controls import gripper_action_to_ctrl


# --- Configuration ---
XML_PATH = "envs/panda_pick_place.xml"
URDF_PATH = "urdf/panda_mujoco_kinematics.urdf"
SEED = 42
MAX_EPISODE_STEPS = 200 # Increased to show a more complete pick-and-place
VIDEO_FPS = 30 # Frames per second for the output video
OUTPUT_DIR = "debug_output"
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "scripted_trajectory.mp4")

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("SCRIPTED_DEBUG_VIDEO")

def print_section_header(title):
    print("\n" + "="*80)
    print(f"🔬 {title.upper()} 🔬")
    print("="*80)

def world_to_base(env: PandaEnv, pose_world_7d: np.ndarray) -> np.ndarray:
    """Helper to convert a world-frame pose to the robot's base frame."""
    base_pos, base_quat_xyzw = env.get_base_pose_expert()
    R_world_base = R.from_quat(base_quat_xyzw)
    R_base_world = R_world_base.inv()
    pos_in_base = R_base_world.apply(pose_world_7d[:3] - base_pos)
    rot_in_base = R_base_world * R.from_quat(pose_world_7d[3:7])
    return np.concatenate([pos_in_base, rot_in_base.as_quat()]).astype(np.float32)

def main():
    log.info("--- Starting Scripted Expert Pipeline Video Generation ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log.info(f"Trajectory video will be saved to: {OUTPUT_VIDEO_PATH}")

    # 1. Initialize components
    env = PandaEnv(xml_path=XML_PATH)
    ik_solver = IKSolver(urdf_path=URDF_PATH)
    expert_cfg = ExpertConfig(pos_tolerance=0.01) # Use a tight tolerance
    expert = ScriptedExpert(cfg=expert_cfg)

    # 2. Reset the environment and expert
    obs, _ = env.reset(seed=SEED)
    expert.reset()
    log.info(f"Initial object position: {np.round(obs['object_pos_world'], 3)}")

    # 3. Setup Video Writer
    # Get frame dimensions from the first rendered image
    frame = env.render(camera_name="fixed_camera")
    height, width, layers = frame.shape
    
    # Define the codec and create VideoWriter object. MP4V is a good, standard choice.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, VIDEO_FPS, (width, height))
    
    if not video_writer.isOpened():
        raise IOError(f"Could not open video writer for path: {OUTPUT_VIDEO_PATH}")
    log.info(f"Video writer initialized for {width}x{height} @ {VIDEO_FPS} FPS.")

    # Use a try...finally block to ensure the video is always saved
    try:
        # 4. Run the main simulation loop
        for t in range(MAX_EPISODE_STEPS):
            print_section_header(f"Simulation Step {t}")

            # --- A. Get Current State ---
            ee_pose = obs["ee_pose_world"]
            # During the trajectory, the object might move, so we get its position fresh each step
            obj_pos = env.get_object_pos_expert() 
            goal_pos = obs["goal_pos_world"]
            current_joints = obs["internal_full_proprio"][:7]
            log.info(f"State | Expert is in state: '{expert._state}'")

            # --- B. Get Expert's Target Pose ---
            target_pose_world, gripper_action = expert.get_target_pose(ee_pose, obj_pos, goal_pos)

            # --- C. IK and Action Calculation ---
            target_pose_base = world_to_base(env, target_pose_world)
            arm_action_normalized = ik_solver.compute_action(
                target_pose_base, 
                current_joints,
                max_delta=0.02, # Use a small, stable gain
                solution_position_tolerance=0.01
            )[:7]

            # The full action includes the gripper state
            # Here we assume a simple gripper model or use a utility

            gripper_ctrl = gripper_action_to_ctrl(gripper_action)
            final_action = np.concatenate([arm_action_normalized, [gripper_ctrl]])

            # --- D. Step the environment ---
            obs, _, _, terminated, _ = env.step(final_action)
            obs = env.get_expert_obs()

            # --- E. Write Frame to Video ---
            frame_rgb = env.render(camera_name="fixed_camera")
            # OpenCV expects BGR format, so we need to convert from MuJoCo's RGB
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
            
            if expert.done:
                log.info("✅✅✅ TASK COMPLETE: Scripted expert finished its state machine.")
                # Continue recording for a few more steps to show the final state
                for _ in range(VIDEO_FPS): # Record 1 extra second
                    frame_rgb = env.render(camera_name="fixed_camera")
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_bgr)
                break
        
    finally:
        # This block will execute even if an error occurs in the loop
        log.info("Releasing video writer and closing file...")
        video_writer.release()
        log.info(f"Video saved successfully to {OUTPUT_VIDEO_PATH}")

    if not expert.done:
        log.warning("⚠️ TASK INCOMPLETE: Episode finished before expert reached 'DONE' state.")

    env.close()

if __name__ == "__main__":
    main()