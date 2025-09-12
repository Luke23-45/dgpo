# In file: scripts/evaluate_rl_policy.py
"""
Evaluation Script for Trained Reinforcement Learning Policies from Stable Baselines 3.

This script loads a trained PPO agent from a .zip file and performs a closed-loop
rollout in the PandaEnv, saving the resulting trajectory as an MP4 video.

It correctly reconstructs the training environment, including all necessary wrappers
(RLRewardWrapper, OctoToSB3Adapter), to ensure a valid evaluation. The script
logs performance metrics (reward components) at each step for detailed analysis.
"""
from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from stable_baselines3 import PPO

# --- Project Imports ---
# This makes imports work reliably when running as a script
sys.path.append(str(Path(__file__).resolve().parent.parent))

from envs.panda_env import PandaEnv
from models.custom_sb3_extractor import BCFeaturesExtractor
from octo.model.octo_model import OctoModel
from utils.obs_adapters import OctoToSB3Adapter
from utils.rl_reward_wrapper import RLRewardWrapper

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("EVAL_RL_POLICY")


# --- Helper Functions ---

def set_global_seed(seed: int):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(args: argparse.Namespace):
    """Initializes components, runs the evaluation rollout, and saves a video."""
    log.info("--- Starting RL Policy Evaluation Script ---")

    # --- 1. Setup ---
    set_global_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_name = checkpoint_path.stem
    run_name = args.run_name or f"{checkpoint_name}_seed{args.seed}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{run_name}.mp4"
    
    log.info(f"Device: {device}")
    log.info(f"Evaluation seed: {args.seed}")
    log.info(f"Output video will be saved to: {video_path}")

    # --- 2. Initialize Environment and OCTO Model ---
    octo_model = None
    if args.w_plausibility > 0.0:
        log.info("Loading OCTO model for reward calculation...")
        try:
            octo_model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")
            log.info("OCTO model loaded successfully.")
        except Exception as e:
            log.error(f"Could not load OCTO model, divergence reward will be disabled. Error: {e}")
            args.w_plausibility = 0.0

    log.info("Initializing environment with full wrapper stack...")
    # The environment MUST have the same wrapper stack as used in training
    env = PandaEnv(xml_path=args.xml_path)
    env = RLRewardWrapper(
        env,
        octo_model=octo_model,
        w_plausibility=args.w_plausibility,
    )
    env = OctoToSB3Adapter(env)

    # --- 3. Load Trained PPO Agent ---
    log.info(f"Loading trained PPO agent from: {checkpoint_path}")
    if not checkpoint_path.is_file():
        log.critical(f"Checkpoint file not found at: {checkpoint_path}")
        return
        
    try:
        # PPO.load automatically handles device placement and policy setup
        agent = PPO.load(checkpoint_path, env=env, device=device)
        log.info("PPO agent loaded successfully.")
    except Exception as e:
        log.critical(f"Failed to load PPO agent. Error: {e}", exc_info=True)
        env.close()
        return

    # --- 4. Setup Video Writer ---
    # We get the frame by calling the render method of the *base* environment
    frame = env.unwrapped.render() 
    h, w, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(video_path), fourcc, 30, (w, h))

    # --- 5. Main Evaluation Loop ---
    obs, _ = env.reset(seed=args.seed)
    total_reward = 0.0
    try:
        for t in range(args.max_steps):
            # Use deterministic=True for evaluation to get the policy's best action
            action, _states = agent.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Log the detailed reward components from the wrapper's info dict
            reward_info = {k: v for k, v in info.items() if k.startswith("R_")}
            log.info(f"Step {t+1} | Action: {np.round(action, 2)} | Reward: {reward:.3f} | Total Reward: {total_reward:.3f} | Details: {reward_info}")

            frame_rgb = env.unwrapped.render()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

            if terminated or truncated:
                log.info(f"Episode finished after {t+1} steps. Final total reward: {total_reward:.3f}")
                break
        
        # Add a pause at the end of the video
        for _ in range(30):
            video_writer.write(frame_bgr)
            
    finally:
        # --- 6. Cleanup ---
        log.info("Releasing resources...")
        video_writer.release()
        env.close()
        log.info("Evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained SB3 RL policy.")
    
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to the .zip SB3 agent checkpoint file to evaluate."
    )
    parser.add_argument(
        "--xml_path", type=str, default="envs/panda_pick_place.xml",
        help="Path to the MuJoCo XML file for the environment."
    )
    parser.add_argument(
        "--run_name", type=str, default=None,
        help="A specific name for the output video. If not provided, a name is generated from the checkpoint."
    )
    parser.add_argument(
        "--output_dir", type=str, default="evaluation_videos_rl",
        help="Directory to save the RL evaluation video."
    )
    parser.add_argument(
        "--seed", type=int, default=101,
        help="Seed for the environment for a reproducible evaluation."
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
        help="Device to use for model inference."
    )
    parser.add_argument(
        "--max_steps", type=int, default=400,
        help="Maximum number of steps for the evaluation episode."
    )
    parser.add_argument(
        "--w_plausibility", type=float, default=0.1,
        help="Weight for the OCTO divergence reward. Should match the training config."
    )
    
    args = parser.parse_args()
    main(args)

"""
python scripts/evaluate_rl_policy.py --checkpoint_path "trained_models_rl/rl_finetune_from_BEST_bc_v1/checkpoints/rl_policy_80000_steps.zip" --seed 777

"""