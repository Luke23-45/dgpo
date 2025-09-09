# In file: scripts/analyze_expert_data.py
"""
Dedicated Analysis Script for the Refactored ExpertDataset Pipeline.

This script acts as a perfect replica of the PyTorch DataLoader to consume
data from the stateful, trajectory-generating ExpertDataset.

It iterates through the dataset, collecting a specified number of sequential
(observation, action) samples and saves them to a detailed JSON file. This
provides a ground-truth report on the quality, variety, and correctness of
the expert demonstrations that the BC model will be trained on.

Key analysis points this script enables:
- Verifying that the expert's state machine progresses through all task stages.
- Confirming that gripper actions are generated at the appropriate times.
- Checking the distribution of expert sources (e.g., OCTO vs. scripted).
"""
import argparse
import json
import logging
from pathlib import Path
from itertools import islice

import numpy as np
from tqdm import tqdm

# --- Project Imports ---
# Add the project root to the path to import from utils/ and envs/
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.expert_dataset import ExpertDataset

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("ANALYZE_EXPERT_DATA")

# In file: scripts/analyze_expert_data.py

def main(args: argparse.Namespace):
    log.info("--- Starting Expert Data Analysis Script ---")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"expert_data_analysis_{args.num_samples}_samples.json"

    log.info(f"Generating {args.num_samples} sequential samples...")
    log.info(f"Analysis results will be saved to: {output_path}")

    # --- 1. Initialize the ExpertDataset ---
    log.info("CRITICAL: Forcing dataset into SCRIPTED-ONLY mode for fast analysis.")
    dataset = ExpertDataset(
        urdf_path=args.urdf_path,
        env_xml_path=args.xml_path,
        instruction=args.instruction,
        base_seed=args.seed,
        max_samples_per_epoch=args.num_samples,
        # THIS IS THE FIX for the "stuck for hours" problem.
        # We explicitly set use_octo=False to prevent slow CPU inference.
        use_octo=False, 
    )

    analysis_results = []
    
    try:
        # --- 2. Generate and Analyze Samples ---
        data_iterator = iter(dataset)
        
        for i, (obs, expert_action) in enumerate(tqdm(
            islice(data_iterator, args.num_samples), 
            total=args.num_samples, 
            desc="Generating Samples"
        )):
            try:
                # We know the source is scripted, but we keep this for API consistency.
                expert_source_int = obs.get("expert_source", -1)
                expert_source = "octo" if expert_source_int == 1 else "scripted"

                sample_data = {
                    "sample_index": i,
                    "expert_source": expert_source,
                    # This call will now work because of our fix to ScriptedExpert
                    "expert_fsm_state": dataset._scripted_expert.get_state(),
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
            
        if hasattr(dataset, "_env") and dataset._env is not None:
            dataset._env.close()
            
        log.info("--- Analysis complete. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the trajectory output of the ExpertDataset.")
    
    parser.add_argument(
        "--urdf_path", type=str, default="urdf/panda_mujoco_kinematics.urdf",
        help="Path to the URDF file for the IKSolver."
    )
    parser.add_argument(
        "--xml_path", type=str, default="envs/panda_pick_place.xml",
        help="Path to the MuJoCo XML file for the PandaEnv."
    )
    parser.add_argument(
        "--num_samples", type=int, default=200,
        help="Number of sequential data samples to generate and analyze from the dataset."
    )
    parser.add_argument(
        "--instruction", type=str, default="pick up the red block",
        help="The language instruction for the OCTO model."
    )
    parser.add_argument(
        "--output_dir", type=str, default="analysis_reports",
        help="Directory to save the JSON analysis file."
    )
    parser.add_argument(
        "--seed", type=int, default=1234,
        help="Base seed for reproducibility."
    )
    
    args = parser.parse_args()
    main(args)