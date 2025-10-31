# FILE: utils/extract_goal_images.py
# A standalone SOTA script to extract final frames as goal images from an expert dataset.

import argparse
from pathlib import Path
import sys
import numpy as np
import imageio
import logging
from tqdm import tqdm

# --- Add project root for imports ---
try:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from utils.expert_dataset import ExpertTrajectoryDataset
except ImportError as e:
    print(f"Error importing project modules: {e}. Make sure this script is in the `utils` directory.")
    sys.exit(1)

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
log = logging.getLogger(__name__)


def extract_goals(args):
    """Main function to extract and save goal images."""
    
    # --- 1. Setup ---
    log.info(f"Setting random seed to: {args.seed}")
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {output_dir.resolve()}")

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        log.error(f"Dataset not found at: {dataset_path}")
        return

    # --- 2. Load Dataset Reader ---
    log.info("Loading dataset index to access episode metadata...")
    try:
        # We only need the metadata, so horizons can be minimal.
        reader = ExpertTrajectoryDataset(
            demo_path=str(dataset_path),
            observation_horizon=1,
            action_horizon=1
        )
    except Exception as e:
        log.exception(f"Failed to load dataset reader. Is the path correct and the dataset valid? Error: {e}")
        return
        
    num_episodes = len(reader.episode_metadata)
    log.info(f"Dataset contains {num_episodes} episodes.")

    if num_episodes == 0:
        log.error("No episodes found in the dataset.")
        return

    # --- 3. Select Episodes ---
    num_to_extract = min(args.num_goals, num_episodes)
    
    # Generate a list of all possible episode indices and shuffle it
    all_episode_indices = list(range(num_episodes))
    np.random.shuffle(all_episode_indices)
    
    # Select the first `num_to_extract` indices from the shuffled list
    selected_indices = all_episode_indices[:num_to_extract]
    
    log.info(f"Randomly selected {len(selected_indices)} episodes to extract goals from: {selected_indices}")

    # --- 4. Extraction Loop ---
    for ep_idx in tqdm(selected_indices, desc="Extracting Goals"):
        try:
            ep_meta = reader.episode_metadata[ep_idx]
            ep_len = ep_meta['length']
            
            # The index of the last frame is length - 1
            last_frame_idx = ep_len - 1
            
            # Use the reader's internal method to get the full image array for the episode
            img_primary_meta = ep_meta["modalities"]["image_primary"]
            full_image_array = reader._get_full_modality_array(
                img_primary_meta["key"],
                img_primary_meta["compression"],
                img_primary_meta["dtype"],
                tuple(img_primary_meta["shape"])
            )
            
            # Slice the final frame
            goal_image_np = full_image_array[last_frame_idx] # Shape: (H, W, C), uint8
            
            # Construct the informative output filename
            output_filename = f"goal_ep{ep_idx}_seed{args.seed}.png"
            output_path = output_dir / output_filename
            
            # Save the image
            imageio.imwrite(output_path, goal_image_np)

        except Exception as e:
            log.error(f"Failed to extract goal for episode {ep_idx}. Error: {e}", exc_info=True)
            continue
            
    log.info(f"--- Extraction Complete ---")
    log.info(f"Successfully saved {len(selected_indices)} goal images to {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract final frames from episodes in an expert LMDB dataset to use as goal images."
    )
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the .lmdb expert dataset file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the goal images will be saved."
    )
    parser.add_argument(
        "--num_goals",
        type=int,
        default=5,
        help="Number of random episodes to extract goal images from. (default: 5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selecting episodes, ensuring reproducibility. (default: 42)"
    )
    
    args = parser.parse_args()
    extract_goals(args)

# python -m tests.extract_goal_images  --dataset_path C:/Users/Hellx/Documents/Programming/python/Project/redhot/data/validations/expert_validation_run_99914b93.lmdb --output_dir data/goal_images