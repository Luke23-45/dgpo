# FILE: debug_sanity_check.py

import torch
import hydra
from omegaconf import DictConfig
import logging

# Import the actual modules we want to test
from train.train_planner import PlannerDataModule, PlannerLightningModule

# Configure logging to see outputs
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="./configs", config_name="train_planner_config")
def main(cfg: DictConfig):
    """
    Manually replicates the PyTorch Lightning sanity check process with detailed printouts.
    """
    print("----------- STARTING MANUAL SANITY CHECK -----------")
    
    # --- IMPORTANT: Override config for debugging ---
    # We already know the issue happens with num_workers=0
    cfg.dataset.num_workers = 0
    # Let's also disable pin_memory as it's a common suspect
    # In your DataLoader, you might need to manually set this to False if it's not in the config.
    
    # ----------------------------------------------------------------------
    # 1. INITIALIZE THE DATAMODULE AND MODEL (just like the trainer does)
    # ----------------------------------------------------------------------
    print("\n--- [PHASE 1] Initializing Modules ---")
    try:
        print("   -> Initializing PlannerDataModule...")
        datamodule = PlannerDataModule(cfg)
        print("   -> Initializing PlannerLightningModule (the model)...")
        model = PlannerLightningModule(cfg)
        print("--- [PHASE 1] SUCCESS: Modules initialized. ---\n")
    except Exception as e:
        print(f"--- [PHASE 1] FAILED: Could not initialize modules: {e}")
        return

    # ----------------------------------------------------------------------
    # 2. SETUP DATA AND GET DATALOADER (just like the trainer does)
    # ----------------------------------------------------------------------
    print("\n--- [PHASE 2] Setting up Data ---")
    try:
        print("   -> Calling datamodule.setup(stage='fit')...")
        datamodule.setup(stage='fit')
        print("   -> Getting validation dataloader...")
        
        # --- Manually create the DataLoader with debug settings ---
        # This bypasses the datamodule's method to ensure we control the settings
        val_loader = torch.utils.data.DataLoader(
            datamodule.val_dataset,
            batch_size=cfg.training.val_batch_size,
            shuffle=False,
            num_workers=0,  # Explicitly 0
            pin_memory=False, # Explicitly False
            drop_last=False
        )
        
        if val_loader is None:
            print("   -> Validation dataloader is None. Cannot proceed.")
            return
        print("--- [PHASE 2] SUCCESS: DataLoader is ready. ---\n")
    except Exception as e:
        print(f"--- [PHASE 2] FAILED: Could not setup data: {e}")
        return

    # ----------------------------------------------------------------------
    # 3. PREPARE FOR EXECUTION (just like the trainer does)
    # ----------------------------------------------------------------------
    print("\n--- [PHASE 3] Preparing for Execution ---")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() and cfg.trainer.accelerator != 'cpu' else "cpu")
        print(f"   -> Target device is: {device}")
        print(f"   -> Moving model to {device}...")
        model.to(device)
        # Manually set the model to eval mode for validation
        model.eval()
        print("--- [PHASE 3] SUCCESS: Model is on device and in eval mode. ---\n")
    except Exception as e:
        print(f"--- [PHASE 3] FAILED: Could not move model to device: {e}")
        return

    # ----------------------------------------------------------------------
    # 4. RUN THE SANITY CHECK LOOP (this is where the magic happens)
    # ----------------------------------------------------------------------
    num_batches_to_check = 2
    print(f"\n--- [PHASE 4] Starting manual sanity loop for {num_batches_to_check} batches ---")
    try:
        # Manually disable gradients, just like Lightning does for validation
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= num_batches_to_check:
                    break

                print(f"\n--- Processing Batch {i} ---")
                
                # STEP A: Getting the batch (already done by the for loop)
                print("   [STEP A] SUCCESS: Batch received from DataLoader.")
                print(f"      -> Batch keys: {batch.keys()}")

                # STEP B: Move the batch to the correct device
                print(f"   [STEP B] Moving batch to {device}...")
                batch_on_device = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                print("   [STEP B] SUCCESS: Batch moved to device.")

                # STEP C: Execute the validation_step
                print("   [STEP C] Executing model.validation_step...")
                model.validation_step(batch_on_device, i)
                print("   [STEP C] SUCCESS: validation_step completed.")

        print(f"\n--- [PHASE 4] SUCCESS: Manual sanity check loop finished. ---")

    except Exception as e:
        print(f"\n\n--- [PHASE 4] FAILED: An error occurred during the sanity loop! ---")
        import traceback
        traceback.print_exc()

    print("\n----------- DEBUG SCRIPT FINISHED -----------")

if __name__ == "__main__":
    main()