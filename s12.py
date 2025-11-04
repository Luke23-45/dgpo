# FILE: scripts/inspect_heatmaps.py
#
# Definitive, SOTA, Visual Diagnostic Tool for ViP-C Heatmaps.
#
# This script loads a specified episode from a final, "training-ready"
# LMDB dataset, decodes the subgoal heatmaps, and saves them as a sequence
# of composite images for direct visual inspection. It is the final
# quality assurance check for our Phase 2 data enhancement pipeline.
#

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
from tqdm import tqdm

# It is assumed this script is run from the project's root directory
# or that the project's root is in the Python path.
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# We reuse the SoAEpisodeLoader, which is now fully patched and robust.
from scripts.visualize_dataset import SoAEpisodeLoader

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | [%(name)s] | %(message)s",
)
logger = logging.getLogger("inspect_heatmaps")

# We need the reverse mapping to get the state name from the phase integer
TASK_PHASE_TO_NAME = {
    0: "APPROACHING_OBJECT",
    1: "EXECUTING_GRASP",
    2: "TRANSPORTING_OBJECT_TO_GOAL",
    3: "PLACING_OBJECT",
    4: "RELEASING_AND_RETRACTING",
    -1: "DONE/INVALID"
}

def create_composite_image(
    base_image: np.ndarray,
    heatmap: np.ndarray,
    diagnostics: Dict[str, Any]
) -> np.ndarray:
    """
    Creates a single, diagnostic-rich visualization frame by overlaying
    the heatmap and text on the base image.
    """
    # Ensure base image is a mutable BGR copy
    vis_frame = cv2.cvtColor(base_image, cv2.COLOR_RGB2BGR)

    # Normalize heatmap to [0, 255] and apply a color map
    heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-9)
    heatmap_colored = cv2.applyColorMap((heatmap_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Resize heatmap to match the base image if necessary
    if vis_frame.shape[:2] != heatmap_colored.shape[:2]:
        heatmap_colored = cv2.resize(heatmap_colored, (vis_frame.shape[1], vis_frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Blend the images
    overlay = cv2.addWeighted(vis_frame, 0.6, heatmap_colored, 0.4, 0)

    # Add diagnostic text
    h, w, _ = overlay.shape
    step = diagnostics.get("step", -1)
    phase_id = diagnostics.get("phase", -1)
    phase_name = TASK_PHASE_TO_NAME.get(phase_id, "UNKNOWN")
    
    # Create a semi-transparent background for the text
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    
    cv2.putText(overlay, f"Step: {step}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(overlay, f"Phase: {phase_name} ({phase_id})", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1, cv2.LINE_AA)

    return overlay

def main(args):
    """Main orchestration function for the inspection process."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    loader = None
    try:
        logger.info(f"Loading dataset index from: {args.dataset_path}")
        loader = SoAEpisodeLoader(args.dataset_path)

        if not (0 <= args.episode_idx < len(loader)):
            logger.error(f"Invalid episode index {args.episode_idx}. Dataset has {len(loader)} episodes (0 to {len(loader)-1}).")
            return

        logger.info(f"Loading and reconstructing episode {args.episode_idx}...")
        episode_data = loader.get_episode(args.episode_idx)
        ep_id = episode_data.get("episode_id", f"ep{args.episode_idx}")
        logger.info(f"Episode '{ep_id}' loaded. It has {len(episode_data['obs_list'])} timesteps.")
        
        # Create a dedicated subdirectory for this episode's frames
        episode_frame_dir = output_dir / ep_id
        episode_frame_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving visualization frames to: {episode_frame_dir}")

        obs_list = episode_data['obs_list']

        for t in tqdm(range(len(obs_list)), desc=f"Processing frames for {ep_id}"):
            obs_step = obs_list[t]

            # --- Data Integrity Checks ---
            if "image_primary" not in obs_step:
                logger.warning(f"Skipping step {t}: 'image_primary' not found.")
                continue
            if "subgoal_heatmaps" not in obs_step:
                logger.warning(f"Skipping step {t}: 'subgoal_heatmaps' not found.")
                continue
            if "task_phases" not in obs_step:
                logger.warning(f"Skipping step {t}: 'task_phases' not found.")
                continue

            base_image = obs_step["image_primary"]
            heatmap = obs_step["subgoal_heatmaps"]
            
            diagnostics = {
                "step": t,
                "phase": obs_step["task_phases"],
            }
            
            # Create the final composite image
            composite_image = create_composite_image(base_image, heatmap, diagnostics)
            
            # Save the frame to disk
            output_path = episode_frame_dir / f"frame_{t:04d}.png"
            cv2.imwrite(str(output_path), composite_image)

        logger.info("="*50)
        logger.info("INSPECTION COMPLETE")
        logger.info(f"Visualizations for episode '{ep_id}' have been saved to:")
        logger.info(f"{episode_frame_dir}")
        logger.info("="*50)

    except Exception as e:
        logger.critical(f"A fatal error occurred: {e}", exc_info=True)
    finally:
        if loader:
            loader.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SOTA Visual Diagnostic Tool for ViP-C Heatmaps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the final, 'training-ready' LMDB dataset file to inspect.",
    )
    parser.add_argument(
        "--episode-idx",
        type=int,
        required=True,
        help="The numerical index of the episode you want to visualize (e.g., 0).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="heatmap_inspection",
        help="The root directory where the output frames will be saved.",
    )
    main(parser.parse_args())



# python -m s12 "C:\Users\Hellx\Documents\Programming\python\Project\redhot\data\validation\validation_dataset.lmdb" --episode-idx 0 --output-dir "data/visualizations"