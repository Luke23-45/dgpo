# In file: scripts/evaluate_policy.py
"""
Evaluation Script for Trained Behavioral Cloning Policies.

This script loads a trained BCNet checkpoint and performs a closed-loop rollout
in the PandaEnv, saving the resulting trajectory as an MP4 video.

It serves as the primary tool for qualitative analysis of a policy's performance,
allowing visualization of its behavior, failure modes, and generalization
capabilities in the simulated environment.
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from gymnasium import spaces

# --- Project Imports ---
from envs.panda_env import PandaEnv
from models.bc_policy import BCNet

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("EVAL_POLICY")


# --- Helper Functions (Leveraging Best Practices) ---

def set_global_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def resolve_device(device_arg: str = "auto") -> torch.device:
    """Resolve 'auto' to 'cuda' if available, else 'cpu'."""
    if device_arg.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)

def load_checkpoint_for_evaluation(path: Path, model: torch.nn.Module) -> None:
    """
    Robustly loads a model state_dict from a .pth checkpoint file.
    Handles both raw state_dicts and common checkpoint dictionary formats.
    """
    log.info(f"Loading checkpoint from: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found at: {path}")

    # Load to CPU first to avoid GPU memory spikes
    ckpt = torch.load(path, map_location="cpu")

    # Determine the actual state dictionary
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt  # Assume it's a raw state_dict
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")

    # Load the weights into the model
    try:
        model.load_state_dict(state_dict, strict=False)
        log.info("Successfully loaded model weights.")
    except Exception as e:
        log.critical(f"Failed to load state_dict into the model. Error: {e}", exc_info=True)
        raise

def prepare_obs_for_model(
    obs: Dict[str, np.ndarray], device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Prepares a single observation dictionary from the environment for model inference.
    
    - Converts NumPy arrays to PyTorch tensors.
    - Adds a batch dimension of 1.
    - Moves all tensors to the specified device.
    - Handles both HWC and CHW image formats.
    """
    obs_torch = {}
    for key, value in obs.items():
        # Only process keys that are expected to be tensors
        if isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value).to(device)
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            
            # Ensure images are CHW if they are not already
            if "image" in key and tensor.ndim == 4:
                # Shape is (B, H, W, C), needs to be (B, C, H, W)
                if tensor.shape[3] in {1, 3, 4}:
                    tensor = tensor.permute(0, 3, 1, 2)
            
            obs_torch[key] = tensor

    return obs_torch

def process_action_for_env(action_tensor: torch.Tensor) -> np.ndarray:
    """
    Post-processes the model's output action tensor for the environment.
    
    - Detaches from the computation graph.
    - Moves to CPU.
    - Converts to a NumPy array.
    - Removes the batch dimension.
    """
    return action_tensor.detach().squeeze(0).cpu().numpy()


# --- Main Execution Function ---

def main(args: argparse.Namespace):
    """Initializes components, runs the evaluation rollout, and saves a video."""
    log.info("--- Starting BC Policy Evaluation Script ---")

    # --- 1. Setup ---
    set_global_seed(args.seed)
    device = resolve_device(args.device)
    
    # Create a unique, descriptive name for the output video
    checkpoint_name = Path(args.checkpoint_path).stem
    run_name = args.run_name or f"{checkpoint_name}_seed{args.seed}_{int(time.time())}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{run_name}.mp4"
    
    log.info(f"Device: {device}")
    log.info(f"Evaluation seed: {args.seed}")
    log.info(f"Output video will be saved to: {video_path}")

    # --- 2. Initialize Environment and Model ---
    try:
        log.info(f"Initializing PandaEnv from: {args.xml_path}")
        env = PandaEnv(xml_path=args.xml_path)
        
        # Infer model dimensions from the environment for robust instantiation
        action_dim = env.action_space.shape[0]
        proprio_dim = env.observation_space.spaces["proprio"].shape[0]
        
        log.info(f"Inferred model dimensions: action_dim={action_dim}, proprio_dim={proprio_dim}")
        
        model = BCNet(n_actions=action_dim, proprio_dim=proprio_dim)
        load_checkpoint_for_evaluation(Path(args.checkpoint_path), model)
        model.to(device)
        model.eval() # Set the model to evaluation mode

    except Exception as e:
        log.critical(f"Failed during initialization: {e}", exc_info=True)
        return

    # --- 3. Setup Video Writer ---
    frame = env.render()
    h, w, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(video_path), fourcc, 30, (w, h))
    if not video_writer.isOpened():
        log.error(f"Failed to open video writer for path: {video_path}")
        env.close()
        return

    # --- 4. Main Evaluation Loop ---
    obs, _ = env.reset(seed=args.seed)
    try:
        for t in range(args.max_steps):
            with torch.no_grad():
                # Prepare observation and get action from the model
                obs_for_model = prepare_obs_for_model(obs, device)
                action_tensor = model(obs_for_model)
                action_for_env = process_action_for_env(action_tensor)
            
            # Step the environment with the model's action
            obs, _, terminated, truncated, _ = env.step(action_for_env)
            
            # Render and write frame
            frame_rgb = env.render()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

            if terminated or truncated:
                log_message = "Episode finished:"
                if terminated: log_message += " Task successful (terminated)."
                if truncated: log_message += " Max steps reached (truncated)."
                log.info(log_message)
                break
        
        # Add a pause at the end of the video
        log.info("Holding final pose for 1 second in video.")
        for _ in range(30):
            video_writer.write(frame_bgr)
            
    except Exception as e:
        log.critical(f"An error occurred during the evaluation loop: {e}", exc_info=True)
    
    finally:
        # --- 5. Cleanup ---
        log.info("Releasing resources...")
        video_writer.release()
        env.close()
        log.info("Evaluation script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained BC policy.")
    
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to the .pth model checkpoint file to evaluate."
    )
    parser.add_argument(
        "--xml_path", type=str, default="envs/panda_pick_place.xml",
        help="Path to the MuJoCo XML file for the environment."
    )
    parser.add_argument(
        "--run_name", type=str, default=None,
        help="A specific name for the output video file. If not provided, a name is generated."
    )
    parser.add_argument(
        "--output_dir", type=str, default="evaluation_videos",
        help="Directory to save the evaluation video."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for the environment's randomization for reproducible rollouts."
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
        help="Device to use for model inference."
    )
    parser.add_argument(
        "--max_steps", type=int, default=400,
        help="Maximum number of steps to run the episode for."
    )
    
    args = parser.parse_args()
    main(args)