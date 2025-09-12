# In file: scripts/analyze_expert_data.py
"""
Definitive Analysis Script for the Refactored ExpertDataset Pipeline.

This script is the final verification tool for our data generation stack.
It correctly consumes data from the stateful, trajectory-generating ExpertDataset
and saves a detailed report to a JSON file.

If the --video flag is provided, it also saves a video of the generated trajectory
for visual confirmation.

The output of this script provides the ground-truth evidence needed to verify:
1. The successful generation of complete pick-and-place trajectories.
2. The correct, sequential progression of the ScriptedExpert's state machine.
3. The generation of appropriate gripper actions at each stage of the task.
4. The diversity of task setups across multiple generated episodes.
"""
import argparse
import json
import logging
from pathlib import Path
from itertools import islice

import numpy as np
import cv2  # <-- Added for video generation
from tqdm import tqdm

# --- Project Imports ---
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.expert_dataset import ExpertDataset

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | [%(name)s] | %(message)s"
)
log = logging.getLogger("ANALYZE_EXPERT_DATA")


def main(args: argparse.Namespace):
    log.info("--- Starting Expert Data Analysis Script ---")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"expert_data_analysis_{args.num_samples}_samples.json"
    
    # --- START OF VIDEO SETUP ---
    video_writer = None
    if args.video:
        video_path = output_dir / f"expert_dataset_trajectory_seed{args.seed}.mp4"
        log.info(f"Video recording ENABLED. Output will be saved to: {video_path}")
    else:
        log.info("Video recording DISABLED.")
    # --- END OF VIDEO SETUP ---

    log.info(f"Generating {args.num_samples} sequential samples...")
    log.info(f"Analysis results will be saved to: {output_path}")

    # --- 1. Initialize the ExpertDataset ---
    # We use the dataset's public API, just as the DataLoader would.
    # We will default to the scripted expert to guarantee data quality.
    dataset = ExpertDataset(
        urdf_path=args.urdf_path,
        env_xml_path=args.xml_path,
        instruction=args.instruction,
        base_seed=args.seed,
        max_samples_per_epoch=args.num_samples,
        use_octo=False, 
        object_size=[float(d) for d in args.object_size.split(',')],
        object_grasp_width=args.object_grasp_width,
        yield_full_obs=True 
    )

    analysis_results = []
    
    try:
        # --- 2. Generate and Analyze Samples using the public API ---
        # Create a simple iterator from the dataset.
        data_iterator = iter(dataset)
        
        # Use itertools.islice to safely consume exactly `num_samples`.
        # This correctly handles the trajectory generation happening in the background.
        for i, (obs, expert_action) in enumerate(tqdm(
            islice(data_iterator, args.num_samples), 
            total=args.num_samples, 
            desc="Generating Samples"
        )):
            try:
                # --- START OF VIDEO FRAME WRITING ---
                if args.video:
                    # Initialize video writer on the first sample
                    if video_writer is None:
                        h, w, _ = obs["image_primary"].shape
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(str(video_path), fourcc, 30, (w, h))

                    # Convert RGB frame from observation to BGR for OpenCV and write to video
                    frame_rgb = obs["image_primary"]
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_bgr)
                # --- END OF VIDEO FRAME WRITING ---

                expert_source_int = obs.get("expert_source", -1)
                expert_source = "octo" if expert_source_int == 1 else "scripted"

                # Extract useful information for analysis
                sample_data = {
                    "sample_index": i,
                    "expert_source": expert_source,
                    # Read the state that was correctly saved with the observation.
                    "expert_fsm_state": obs.get("expert_fsm_state", "STATE_NOT_RECORDED"),
                    "ee_pos_world": obs["ee_pose_world"][:3].tolist(),
                    "object_pos_world": obs["object_pos_world"].tolist(),
                    "gripper_action_value": float(expert_action[-1]),
                }
                analysis_results.append(sample_data)
                
            except Exception as e:
                log.error(f"Error processing sample {i}: {e}", exc_info=True)
                analysis_results.append({"sample_index": i, "error": str(e)})

    finally:
        # --- 3. Save Results and Cleanup ---
        log.info(f"Generated {len(analysis_results)} data points. Saving to file.")
        with open(output_path, "w") as f:
            json.dump(analysis_results, f, indent=4)
        
        # --- START OF VIDEO CLEANUP ---
        if video_writer is not None:
            log.info("Releasing video writer...")
            video_writer.release()
        # --- END OF VIDEO CLEANUP ---
            
        if hasattr(dataset, "_env") and dataset._env is not None:
            dataset._env.close()
            
        log.info("--- Analysis complete. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the trajectory output of the ExpertDataset.")
    
    parser.add_argument("--urdf_path", type=str, default="urdf/panda_mujoco_kinematics.urdf")
    parser.add_argument("--xml_path", type=str, default="envs/panda_pick_place.xml")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of samples to generate.")
    parser.add_argument("--instruction", type=str, default="pick up the red block")
    parser.add_argument("--output_dir", type=str, default="analysis_reports")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--object-size", type=str, default="0.04,0.04,0.04")
    parser.add_argument("--object-grasp-width", type=float, default=0.6)
    
    # --- NEW ARGUMENT ---
    parser.add_argument("--video", action="store_true", help="Enable to generate a video of the trajectory.")
    
    args = parser.parse_args()
    main(args)