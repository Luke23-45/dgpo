# In file: scripts/verify_env_physics.py
"""
A hyper-focused script to verify ONLY the grasping physics of the PandaEnv.

This test bypasses all expert and IK logic. It sends a sequence of
hard-coded, "perfect" delta actions to the environment to answer one question:
"Is the environment physically capable of grasping and lifting the cube?"

If the block lifts in the output video, it proves that the XML weld constraint
and the environment's step method are working correctly.
"""
import argparse
import logging
from pathlib import Path
import time

import cv2
import numpy as np

# --- Project Imports ---
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from envs.panda_env import PandaEnv

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | [%(name)s] | %(message)s"
)
log = logging.getLogger("VERIFY_ENV_PHYSICS")


def main(args: argparse.Namespace):
    log.info("--- Starting PandaEnv Physics Verification Script ---")

    output_dir = Path("verification_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"env_physics_test_seed{args.seed}.mp4"

    # --- 1. Initialize ONLY the Environment ---
    log.info("Initializing PandaEnv...")
    env = PandaEnv(xml_path=args.xml_path)
    log.info("Environment initialized.")

    # --- 2. Setup Video Writer ---
    frame = env.render()
    h, w, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(video_path), fourcc, 30, (w, h))

    # --- 3. Run Hard-Coded Action Sequence ---
    log.info(f"Starting hard-coded sequence. Video will be saved to: {video_path}")
    obs, _ = env.reset(seed=args.seed)

    try:
        # Get the INITIAL object position. This is our main target.
        initial_object_pos = obs["object_pos_world"].copy()

        # Define the stages of our test
        stages = [
            {"name": "Move above block", "target_offset": np.array([0.0, 0.0, 0.10]), "gripper": -1.0, "steps": 100},
            {"name": "Lower to block", "target_offset": np.array([0.0, 0.0, 0.02]), "gripper": -1.0, "steps": 50},
            {"name": "Close gripper (activate weld)", "target_offset": np.array([0.0, 0.0, 0.02]), "gripper": 1.0, "steps": 50},
            {"name": "Lift block", "target_offset": np.array([0.0, 0.0, 0.15]), "gripper": 1.0, "steps": 100},
        ]

        for stage in stages:
            log.info(f"--- Executing Stage: {stage['name']} ---")
            for _ in range(stage["steps"]):
                # The target position is now DYNAMICALLY calculated based on the block's initial position
                target_pos = initial_object_pos + stage["target_offset"]
                
                # Get the most recent end-effector position
                current_pos = env.get_ee_pose()[:3]
                
                # --- This is the gentler controller from our previous attempt ---
                action = np.zeros(8)
                MAX_SPEED = 0.5
                gain = 5.0
                
                pos_delta = target_pos - current_pos
                velocity_command = pos_delta * gain
                speed = np.linalg.norm(velocity_command)
                if speed > MAX_SPEED:
                    velocity_command = velocity_command * (MAX_SPEED / speed)
                
                action[:3] = velocity_command
                action[7] = stage["gripper"]
                
                # Step the environment and render
                obs, _, _, _, _ = env.step(action)
                frame_rgb = env.render()
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)

        # Hold the final frame
        for _ in range(30): video_writer.write(frame_bgr)
        
        # --- 4. Final Verification Check ---
        final_block_pos = obs["object_pos_world"]
        log.info(f"Final block Z position: {final_block_pos[2]:.4f}")

        if final_block_pos[2] > 0.5: # A reasonable height threshold for a lift
            log.info("✅ SUCCESS: The block was successfully lifted off the table.")
        else:
            log.error("❌ FAILURE: The block was not lifted. Check XML weld and env.step() logic.")
        
    finally:
        log.info("Releasing resources...")
        video_writer.release()
        env.close()

    log.info("--- Verification complete. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify the grasping physics of the PandaEnv.")
    
    parser.add_argument(
        "--xml_path", type=str, default="envs/panda_pick_place.xml",
    )
    parser.add_argument(
        "--seed", type=int, default=123,
    )
    
    args = parser.parse_args()
    main(args)