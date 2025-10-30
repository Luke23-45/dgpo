# FILE: scripts/verify_dataset.py
# A SOTA Diagnostic Tool to Visually Verify Dataset Integrity

import argparse
import logging
from pathlib import Path
import numpy as np
import random
from tqdm import tqdm

# --- Dependency and Project Imports ---
import sys
try:
    # Essential for saving images
    import imageio
except ImportError:
    print("Error: 'imageio' package not found. Please install it with 'pip install imageio[pyav]'")
    sys.exit(1)

try:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    # We MUST use our own SOTA reader to verify the format is correct for the pipeline
    from utils.expert_dataset import ExpertTrajectoryDataset
except ImportError as e:
    print(f"Error importing project modules: {e}. Please run from project root or ensure PYTHONPATH is set.")
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_dataset")

def main(args):
    """
    Main function to load the dataset and save random goal image samples.
    """
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    num_samples = args.num_samples

    # 1. Validate inputs and create output directory
    if not dataset_path.exists():
        logger.error(f"Dataset LMDB file not found at: {dataset_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Will save verification images to: {output_dir.resolve()}")

    try:
        # 2. Load the dataset using our SOTA reader
        # The horizons can be minimal (1) as we are not using __getitem__ to get chunks.
        logger.info("Loading dataset index...")
        dataset = ExpertTrajectoryDataset(
            demo_path=str(dataset_path),
            observation_horizon=1,
            action_horizon=1
        )
        
        total_episodes = len(dataset.episode_metadata)
        if total_episodes == 0:
            logger.error("Dataset index is empty. No episodes found.")
            return
        
        logger.info(f"Dataset contains {total_episodes} episodes.")

        # 3. Select random, unique episode indices to verify
        num_to_sample = min(num_samples, total_episodes)
        indices_to_verify = sorted(random.sample(range(total_episodes), k=num_to_sample))
        
        logger.info(f"Randomly sampling {num_to_sample} episodes for verification...")
        logger.info(f"Indices to be checked: {indices_to_verify}")

        # 4. Loop, extract, and save
        for ep_idx in tqdm(indices_to_verify, desc="Saving Goal Images"):
            try:
                # This is the key API call we are testing
                goal_image_np = dataset.get_goal_image(ep_idx)

                # Basic validation of the returned image
                if not isinstance(goal_image_np, np.ndarray) or goal_image_np.ndim != 3:
                    logger.warning(f"Episode {ep_idx}: get_goal_image() returned invalid data type/shape. Skipping.")
                    continue

                # Construct a descriptive filename
                save_path = output_dir / f"episode_{ep_idx:06d}_goal_image.png"

                # Save the image (HWC, uint8, RGB format is handled by imageio)
                imageio.imwrite(save_path, goal_image_np)

            except KeyError:
                logger.error(f"Episode {ep_idx}: Failed to find 'goal_image_primary' metadata. "
                             "The dataset may not have been enhanced correctly.")
            except Exception as e:
                logger.error(f"An unexpected error occurred while processing episode {ep_idx}: {e}", exc_info=True)

        logger.info("="*50)
        logger.info("Verification Complete!")
        logger.info(f"Please check the saved images in the following directory:")
        logger.info(f"==> {output_dir.resolve()}")
        logger.info("="*50)

    except Exception as e:
        logger.exception(f"A fatal error occurred during dataset verification: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A diagnostic tool to visually verify the goal images in an enhanced SOTA dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the enhanced SOTA .lmdb dataset file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/dataset_verification",
        help="Directory to save the sampled goal images."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of random episodes to sample and verify."
    )
    args = parser.parse_args()
    main(args)

