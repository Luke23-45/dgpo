# In file: scripts/test_procedural_env.py
"""
Diagnostic Script for PandaEnv Procedural Generation.

This script generates a "showcase" video to visually verify the full
capabilities of the PandaEnv, including:
1. Structured Placement Scenarios (easy, hard, random).
2. Full Domain Randomization (lighting, textures, camera).

The output is a single MP4 video montage, with each segment annotated with
the parameters used to generate it, allowing for rapid manual review of the
data distribution.
"""
import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

# --- Project Imports ---
from envs.panda_env import DomainRandomizationConfig, PandaEnv

# --- Configuration ---
XML_PATH = "envs/panda_pick_place.xml"

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("ENV_TEST")


def add_text_to_frame(
    frame: np.ndarray, text_lines: list[str], start_x: int = 10, start_y: int = 20
) -> np.ndarray:
    """Adds multiple lines of text to an image frame using OpenCV."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color_bgr = (255, 255, 255)  # White
    line_type = 1
    line_height = 20

    for i, line in enumerate(text_lines):
        y = start_y + i * line_height
        cv2.putText(frame, line, (start_x, y), font, font_scale, font_color_bgr, line_type)
    return frame


def generate_and_record_segment(
    env: PandaEnv,
    video_writer: cv2.VideoWriter,
    segment_title: str,
    num_resets: int,
    steps_per_reset: int,
    reset_options: Optional[Dict] = None,
):
    """
    Runs one segment of the showcase video.

    Args:
        env: The PandaEnv instance.
        video_writer: The OpenCV video writer.
        segment_title: The name of the segment (e.g., "Easy Placements").
        num_resets: How many different scenes to generate for this segment.
        steps_per_reset: How many simulation steps to record for each scene.
        reset_options: The `options` dictionary to pass to `env.reset()`.
    """
    log.info(f"--- Generating Segment: {segment_title} ---")
    
    for i in range(num_resets):
        # Reset the environment with the specified options
        obs, _ = env.reset(options=reset_options)
        
        # Capture and record frames for this scene
        for t in range(steps_per_reset):
            # Render both camera views
            frame_primary_bgr = cv2.cvtColor(env.render("fixed_camera"), cv2.COLOR_RGB2BGR)
            frame_wrist_bgr = cv2.cvtColor(env.render("wrist_camera"), cv2.COLOR_RGB2BGR)
            
            # Combine into a side-by-side view
            h, w, _ = frame_primary_bgr.shape
            frame_wrist_resized = cv2.resize(frame_wrist_bgr, (w, h))
            combined_frame = np.concatenate((frame_primary_bgr, frame_wrist_resized), axis=1)

            # Add annotations
            annotations = [
                f"Segment: {segment_title}",
                f"Reset #{i+1}/{num_resets}",
                f"Timestep: {t}",
                f"Placement Mode: {reset_options.get('placement_mode', 'default') if reset_options else 'default'}",
            ]
            # ...
            annotated_frame = add_text_to_frame(combined_frame, annotations)
            
            # If the video_writer is real, write to it.
            if video_writer:
                video_writer.write(annotated_frame)
            # This function is also used to get a dummy frame for initialization.
            # In that case, we just return the first one we generate.
            else:
                return annotated_frame


def main(args: argparse.Namespace):
    """Main function to generate the diagnostic video."""
    log.info("--- Starting Environment Procedural Generation Test ---")

    # --- 1. Setup ---
    video_path = Path(args.output_dir) / args.output_filename
    
    # Now, get the parent directory of that file and ensure it exists
    video_path.parent.mkdir(parents=True, exist_ok=True)
    
    # We will instantiate two environments: one with DR, one without.
    log.info("Initializing environments...")
    try:
        # Standard environment for placement tests
        env_no_dr = PandaEnv(xml_path=XML_PATH)
        env_no_dr.reset(seed=args.seed)

        # Environment with Domain Randomization enabled
        env_with_dr = PandaEnv(xml_path=XML_PATH, enable_domain_randomization=True)
        env_with_dr.reset(seed=args.seed)
    except Exception as e:
        log.critical(f"Failed to initialize environment: {e}", exc_info=True)
        return
        
    # --- 2. Setup Video Writer ---
    # We get the frame size from our helper to ensure it's correct
    dummy_frame = generate_and_record_segment(env_no_dr, None, "", 1, 1, {})
    if dummy_frame is None: return # Error occurred
    h, w, _ = dummy_frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (w, h))
    if not video_writer.isOpened():
        log.error(f"Failed to open video writer for path: {video_path}")
        env_no_dr.close(); env_with_dr.close()
        return

    log.info(f"Video writer initialized. Output will be saved to: {video_path}")

    # --- 3. Generate and Record Segments ---
    try:
        # Segment 1: Easy Placements (no DR)
        generate_and_record_segment(
            env=env_no_dr, video_writer=video_writer, segment_title="Easy Placements",
            num_resets=args.num_resets, steps_per_reset=args.steps_per_reset,
            reset_options={"placement_mode": "easy"}
        )
        
        # Segment 2: Hard Placements (no DR)
        generate_and_record_segment(
            env=env_no_dr, video_writer=video_writer, segment_title="Hard Placements",
            num_resets=args.num_resets, steps_per_reset=args.steps_per_reset,
            reset_options={"placement_mode": "hard"}
        )

        # Segment 3: Full Domain Randomization (with random placement)
        generate_and_record_segment(
            env=env_with_dr, video_writer=video_writer, segment_title="Full Domain Randomization",
            num_resets=args.num_resets * 2, steps_per_reset=args.steps_per_reset,
            reset_options={"placement_mode": "random"} # Let placement be random during DR
        )

    except Exception as e:
        log.critical(f"An error occurred during video generation: {e}", exc_info=True)
    
    finally:
        # --- 4. Cleanup ---
        log.info("Releasing resources...")
        video_writer.release()
        env_no_dr.close()
        env_with_dr.close()
        log.info("✅ Showcase video saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and visualize PandaEnv procedural generation.")
    parser.add_argument(
        "--output_dir", type=str, default="debug_output",
        help="Directory to save the output video."
    )
    parser.add_argument(
        "--output_filename", type=str, default="env_showcase.mp4",
        help="Name of the output video file."
    )
    parser.add_argument(
        "--num_resets", type=int, default=5,
        help="Number of different scenes to show for each placement mode."
    )
    parser.add_argument(
        "--steps_per_reset", type=int, default=15,
        help="Number of static frames to record for each scene."
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second for the output video.")
    
    args = parser.parse_args()
    main(args)



"""
python -m tests.test_procedural_env --output_dir videos --output_filename panda_dr_showcase.mp4 --num_resets 10 --steps_per_reset 20 --fps 10
"""