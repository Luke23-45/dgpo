# verify_checkpoint.py
import torch
from pathlib import Path
import argparse

def main(ckpt_path: Path):
    if not ckpt_path.is_file():
        print(f"ERROR: Checkpoint file not found at: {ckpt_path}")
        return

    print(f"--- Verifying Checkpoint: {ckpt_path.name} ---")
    try:
        # Load the checkpoint onto the CPU to avoid needing a GPU
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))

        print("Checkpoint successfully loaded. Contents:")
        
        # Print the metadata stored in the checkpoint
        epoch = checkpoint.get("epoch")
        best_loss = checkpoint.get("best_loss")
        
        if epoch is not None:
            print(f"  - epoch_completed: {epoch}")
        else:
            print("  - 'epoch' key not found.")
            
        if best_loss is not None:
            print(f"  - best_loss: {best_loss}")
        else:
            print("  - 'best_loss' key not found.")
            
        # You can also check the model weights
        if "model_state_dict" in checkpoint:
            print("  - Contains 'model_state_dict': Yes")
            num_params = sum(p.numel() for p in checkpoint["model_state_dict"].values())
            print(f"  - Number of parameters in state_dict: {num_params:,}")
        else:
            print("  - Contains 'model_state_dict': No")
            
    except Exception as e:
        print(f"ERROR: Failed to load or inspect checkpoint. Reason: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a PyTorch checkpoint file.")
    parser.add_argument("checkpoint_path", type=str, help="Path to the .pth checkpoint file.")
    args = parser.parse_args()
    
    main(Path(args.checkpoint_path))