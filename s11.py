# FILE: debug/test_dataloader.py
# A SOTA Diagnostic Script to Isolate and Debug DataLoader Issues

import os
import sys
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import time

# --- Project Imports ---
# Add project root to sys.path for robust execution
try:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    # Import the exact DataModule used by the failing training script
    from train.train_planner import PlannerDataModule
except ImportError as e:
    print(f"Error importing project modules: {e}. Please run from the project root.")
    sys.exit(1)

# --- Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
log = logging.getLogger("dataloader_debugger")

@hydra.main(version_base=None, config_path="./configs", config_name="train_planner_config")
def main(cfg: DictConfig):
    log.info("----------- DataLoader Debugger -----------")
    log.info("This script will attempt to replicate the PyTorch Lightning sanity check.")
    log.info(f"Using config: {OmegaConf.to_yaml(cfg.dataset)}")
    log.info("-----------------------------------------")

    try:
        # --- 1. Initialize the DataModule (Same as in trainer) ---
        log.info("Step 1: Initializing PlannerDataModule...")
        datamodule = PlannerDataModule(cfg)
        log.info("PlannerDataModule initialized successfully.")

        # --- 2. Setup the DataModule (Same as in trainer) ---
        log.info("\nStep 2: Calling datamodule.setup(stage='fit')...")
        datamodule.setup(stage='fit')
        log.info("datamodule.setup() completed successfully.")
        log.info(f"Train dataset size: {len(datamodule.train_dataset)}")
        log.info(f"Validation dataset size: {len(datamodule.val_dataset)}")


        # --- 3. Get the Validation DataLoader (Same as in trainer) ---
        log.info("\nStep 3: Calling datamodule.val_dataloader()...")
        val_dataloader = datamodule.val_dataloader()
        if val_dataloader is None:
            log.error("Validation dataloader is None. Check your config and dataset split.")
            return
        log.info("Validation dataloader created successfully.")
        log.info(f"Number of workers: {val_dataloader.num_workers}")
        if val_dataloader.num_workers > 0:
            log.warning("WARNING: Debugging with num_workers > 0 can be complex due to multiprocessing.")

        # --- 4. The Core Test: Attempt to fetch ONE batch ---
        log.info("\nStep 4: Attempting to fetch the first batch... (This is where the hang occurs)")
        log.info("If the script hangs here, the issue is inside the dataset's __getitem__ method or with a file lock.")
        
        start_time = time.time()
        
        # This is the command that hangs
        batch = next(iter(val_dataloader))
        
        end_time = time.time()
        log.info(f"SUCCESS! Fetched one batch in {end_time - start_time:.2f} seconds.")

        # --- 5. Inspect the Batch (If successful) ---
        log.info("\nStep 5: Inspecting the fetched batch...")
        if isinstance(batch, dict):
            for key, value in batch.items():
                log.info(f"  - Key: '{key}', Type: {type(value)}, Shape/Size: {value.shape if isinstance(value, torch.Tensor) else len(value)}")
        else:
            log.error(f"Batch is not a dictionary as expected. Type: {type(batch)}")

    except Exception as e:
        log.exception("An error occurred during the debug run.")

if __name__ == "__main__":
    main()