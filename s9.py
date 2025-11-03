import logging
from pathlib import Path
from utils.expert_dataset import ExpertTrajectoryDataset

# IMPORTANT: Update this path to point to the LMDB file you are using for training.
# This should be the same path as `train_path` in your YAML config.
DATASET_PATH =   "C:/Users/Hellx/Documents/Programming/python/Project/redhot/data/training/training_data.lmdb"   # Replace with actual path

def verify_dimensions(dataset_path: str):
    """
    Loads an ExpertTrajectoryDataset and prints its key dimensions.
    """
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')

    if not Path(dataset_path).exists():
        log.error(f"Dataset not found at: {dataset_path}")
        log.error("Please update the DATASET_PATH variable in this script.")
        return

    log.info(f"Loading dataset from: {dataset_path}")
    try:
        reader = ExpertTrajectoryDataset(demo_path=dataset_path, observation_horizon=2, action_horizon=8)
        
        # Get metadata from the first episode
        first_episode_meta = reader.episode_metadata[0]
        
        # --- Find the action dimension ---
        action_shape = first_episode_meta['modalities']['actions']['shape']
        action_dim = action_shape[-1]
        
        # --- Find the proprioception dimension ---
        proprio_dim = reader.get_proprioception_dim()

        print("\n" + "="*50)
        print("    DATASET DIMENSION VERIFICATION REPORT")
        print("="*50)
        print(f"  Action Dimension      (action_dim): {action_dim}")
        print(f"  Proprioception Dim (proprio_dim): {proprio_dim}")
        print("="*50)
        print("\nACTION: Please ensure these values match the ones in your config YAML file.")

    except Exception as e:
        log.error("An error occurred while reading the dataset.", exc_info=True)

if __name__ == "__main__":
    verify_dimensions(DATASET_PATH)