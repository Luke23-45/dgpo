# In file: scripts/debug_scripted_pipeline.py
"""
Debug Script for Visualizing the Scripted Expert Pipeline.

This script runs a full pick-and-place trajectory using the deterministic
scripted expert and records the result as an MP4 video. It is an essential
tool for verifying and debugging the core robotics components:
1. PandaEnv: The MuJoCo simulation environment.
2. ScriptedExpert: The state-machine-based expert policy.
3. IKSolver: The inverse kinematics solver.

The script will create a directory named 'debug_output' and save the
trajectory video inside it.
"""
from __future__ import annotations

import logging
import os
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# --- Project Imports ---
# Ensure these paths are correct for your project structure
from envs.panda_env import PandaEnv
from utils.controls import gripper_action_to_ctrl
from utils.ik_solver import IKSolver
from utils.scripted_expert import ExpertConfig, ScriptedExpert

# --- Configuration ---
XML_PATH = "envs/panda_pick_place.xml"
URDF_PATH = "urdf/panda_mujoco_kinematics.urdf"
SEED = 42
MAX_EPISODE_STEPS = 10  # Give the expert enough time to complete the task
VIDEO_FPS = 30
OUTPUT_DIR = "debug_output"
OUTPUT_VIDEO_FILENAME = "scripted_trajectory.mp4"

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

log = logging.getLogger("SCRIPTED_DEBUG")


def world_to_base(env: PandaEnv, pose_world_7d: np.ndarray) -> np.ndarray:
    """
    Robustly converts a 7D pose from the world frame to the robot's base frame.
    """
    base_pos, base_quat_xyzw = env.get_base_pose()
    
    # Create Rotation objects from quaternions
    R_world_base = R.from_quat(base_quat_xyzw)
    R_target_world = R.from_quat(pose_world_7d[3:7])
    
    # Invert the base rotation to get the transformation from world to base
    R_base_world = R_world_base.inv()
    
    # Transform position and rotation
    pos_in_base = R_base_world.apply(pose_world_7d[:3] - base_pos)
    rot_in_base = R_base_world * R_target_world
    
    return np.concatenate([pos_in_base, rot_in_base.as_quat()]).astype(np.float32)


def main():
    """Initializes components, runs the simulation, and saves a video."""
    log.info("--- Starting Scripted Expert Pipeline Video Generation ---")
    
    # --- 1. Setup Environment and Components ---
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        video_path = os.path.join(OUTPUT_DIR, OUTPUT_VIDEO_FILENAME)
        log.info(f"Output video will be saved to: {video_path}")

        log.info("Initializing PandaEnv...")
        env = PandaEnv(xml_path=XML_PATH)

        log.info("Initializing IKSolver...")
        ik_solver = IKSolver(urdf_path=URDF_PATH)
        
        # Configure the expert for precision
        expert_cfg = ExpertConfig(pos_tolerance=0.01)
        expert = ScriptedExpert(cfg=expert_cfg)

    except Exception as e:
        log.critical(f"Failed during initialization: {e}", exc_info=True)
        return

    # --- 2. Reset Environment and Expert ---
    obs, _ = env.reset(seed=SEED)
    expert.reset()
    log.info(f"Environment reset. Initial object position: {np.round(obs['object_pos_world'], 3)}")

    # --- 3. Setup Video Writer ---
    frame = env.render()
    h, w, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (w, h))
    if not video_writer.isOpened():
        log.error(f"Failed to open video writer for path: {video_path}")
        env.close()
        return

    log.info(f"Video writer initialized for {w}x{h} frames at {VIDEO_FPS} FPS.")

    # --- 4. Main Simulation Loop ---
    # Use a try...finally block to ensure resources are always released
    try:
        for t in range(MAX_EPISODE_STEPS):
            # --- Get Current State ---
            # Get fresh observations, especially the object position which can change
            ee_pose = obs["ee_pose_world"]
            obj_pos = env.get_object_pos_expert()
            goal_pos = obs["goal_pos_world"]
            current_joints = obs["internal_full_proprio"][:7]

            log.debug(f"Step {t} | Expert state: '{expert.get_state()}'")

            # --- Get Expert's Target Pose ---
            target_pose_world, gripper_action = expert.get_target_pose(ee_pose, obj_pos, goal_pos)

            # --- Calculate Action via IK ---
            target_pose_base = world_to_base(env, target_pose_world)
            arm_action = ik_solver.compute_action(
                target_pose_base,
                current_joints,
                max_delta=0.05,  # A small, stable gain for smooth motion
                solution_position_tolerance=0.01
            )
            gripper_ctrl = gripper_action_to_ctrl(gripper_action)
            final_action = np.concatenate([arm_action, [gripper_ctrl]])

            # --- Step the Environment ---
            obs, _, terminated, truncated, _ = env.step(final_action)
            
            # --- Write Frame to Video ---
            # Render the scene *after* the step to show the result of the action
            frame_rgb = env.render()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

            if expert.is_done() or terminated or truncated:
                log_message = "Simulation ended:"
                if expert.is_done():
                    log_message += " Expert finished its state machine."
                if terminated:
                    log_message += " Environment terminated."
                if truncated:
                    log_message += " Episode truncated (max steps reached)."
                log.info(log_message)
                break
        
        # Add a pause at the end of the video
        # Add a pause at the end of the video
        if expert.is_done():
            log.info("Task complete. Holding final pose for 1 second in video.")
            # Get the very last frame state
            final_frame_rgb = env.render()
            final_frame_bgr = cv2.cvtColor(final_frame_rgb, cv2.COLOR_RGB2BGR) # <-- CORRECTED
            for _ in range(VIDEO_FPS):
                video_writer.write(final_frame_bgr)

    except Exception as e:
        log.critical(f"An error occurred during the simulation loop: {e}", exc_info=True)
    
    finally:
        # --- 5. Cleanup ---
        log.info("Releasing resources...")
        video_writer.release()
        env.close()
        log.info(f"Video saved successfully to {video_path}")

    if not expert.is_done():
        log.warning("⚠️ TASK INCOMPLETE: Episode finished before expert completed its task.")


if __name__ == "__main__":
    main()

