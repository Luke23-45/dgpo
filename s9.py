# FILE: scripts/convert_checkpoint_architecture.py

import sys
import torch
import logging
from pathlib import Path

# --- Project Imports ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Import the LightningModule (which contains the model)
from train.train_semantic_planner import SemanticPlannerLightningModule

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("SURGERY")

def main():
    # --- CONFIGURATION ---
    # Path to the "Best" checkpoint (trained with 3 layers unfrozen)
    OLD_CKPT_PATH = "/content/drive/MyDrive/pda/models/v1/backups/backup_epoch_008.ckpt"
    # Path for the "Clean" checkpoint (compatible with 1 layer unfrozen)
    NEW_CKPT_PATH = "/content/drive/MyDrive/pda/models/v1/backups/backup_epoch_008_FIXED.ckpt"
    
    device = torch.device("cpu") 
    
    log.info(f"--- Starting Checkpoint Surgery ---")
    
    # 1. Load the Old Checkpoint
    if not Path(OLD_CKPT_PATH).exists():
        log.error(f"❌ Source file not found: {OLD_CKPT_PATH}")
        return

    try:
        checkpoint = torch.load(OLD_CKPT_PATH, map_location=device)
        log.info(f"✅ Loaded: {OLD_CKPT_PATH}")
    except Exception as e:
        log.error(f"Failed to load checkpoint: {e}")
        return

    # 2. Extract Configuration
    # We reuse the config dictionary exactly as PL saved it to ensure compatibility
    hparams = checkpoint.get("hyper_parameters", {})
    cfg_dict = hparams.get("cfg")
    
    if cfg_dict is None:
        log.error("Could not find 'cfg' in hyper_parameters. Checkpoint structure might differ.")
        return

    # 3. Initialize FRESH Model
    # CRITICAL: This uses the current code in `models/semantic_planner.py`.
    # Ensure you have REVERTED that file to "Unfreeze 1 Layer" before running this.
    log.info("Initializing new model structure from current code...")
    try:
        new_pl_module = SemanticPlannerLightningModule(cfg_dict)
    except Exception as e:
        log.error(f"Failed to initialize model: {e}")
        log.error("Did you revert models/semantic_planner.py to the correct logic?")
        return

    # 4. Transplant Weights
    # The state_dict contains the values. The 'requires_grad' flag is not saved here,
    # so the shapes will match perfectly.
    log.info("Transplanting weights...")
    old_state_dict = checkpoint["state_dict"]
    
    # strict=True ensures every key matches exactly
    missing, unexpected = new_pl_module.load_state_dict(old_state_dict, strict=True)
    
    if len(missing) == 0 and len(unexpected) == 0:
        log.info("✅ Weights matched perfectly.")
    else:
        log.warning(f"⚠️ Weight mismatch! Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        return # Abort if keys don't match, something else is wrong

    # 5. Update the Checkpoint Dictionary
    checkpoint["state_dict"] = new_pl_module.state_dict()
    
    # --- CRITICAL FIX: WIPE OPTIMIZER STATES ---
    # The old optimizer tracked 3 layers. The new one tracks 1. 
    # Loading the old state would cause a shape mismatch crash.
    # Deleting this forces PyTorch Lightning to initialize a fresh optimizer.
    if "optimizer_states" in checkpoint:
        del checkpoint["optimizer_states"]
        log.info("✅ Optimizer states wiped (Essential for architecture change).")
    
    # Also reset the loop state to ensure a clean start
    if "loops" in checkpoint:
        del checkpoint["loops"]

    # 6. Save
    torch.save(checkpoint, NEW_CKPT_PATH)
    log.info(f"✅ Fixed Checkpoint saved to: {NEW_CKPT_PATH}")
    log.info("READY. Resume training with this file.")

if __name__ == "__main__":
    main()