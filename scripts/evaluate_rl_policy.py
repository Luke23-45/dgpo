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
from utils.scripted_expert import ScriptedExpert,ObjectProfile 
import cv2
import numpy as np
import torch
from stable_baselines3 import PPO
from run_experiment import setup_environment
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

# +++ REPLACE YOUR main FUNCTION WITH THIS +++
def main(args: argparse.Namespace):
    """Initializes components, runs the evaluation rollout, and saves a video."""
    log.info("--- Starting RL Policy Evaluation Script ---")

    set_global_seed(args.seed)
    device_str = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    
    checkpoint_path = Path(args.checkpoint_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{checkpoint_path.stem}_seed{args.seed}.mp4"
    
    log.info(f"Device: {device_str}")
    log.info(f"Output video will be saved to: {video_path}")

    # --- 2. Initialize Environment using the Centralized Function ---
    log.info("Initializing environment using the centralized setup_environment function...")
    # This ensures all wrappers and the control_mode are identical to training.
    # We create a single VecEnv (n_envs=1).
    env = setup_environment(
        xml_path=args.xml_path,
        seed=args.seed,
        control_mode='absolute',  # <-- THE CRITICAL FIX: Match training mode
        n_envs=1,
        add_monitor_wrapper=False, # We don't need episode logging for this single rollout
    )

    # --- 3. Load Trained PPO Agent ---
    log.info(f"Loading trained PPO agent from: {checkpoint_path}")
    if not checkpoint_path.is_file():
        log.critical(f"Checkpoint file not found at: {checkpoint_path}")
        return
        
    try:
        # Load the agent, passing the correctly configured VecEnv
        agent = PPO.load(checkpoint_path, env=env, device=device_str)
        log.info("PPO agent loaded successfully.")
    except Exception as e:
        log.critical(f"Failed to load PPO agent. Error: {e}", exc_info=True)
        env.close()
        return

    # --- 4. Setup Video Writer ---
    # The `render()` method of a VecEnv returns an RGB array for the first env.
    frame = env.render() 
    h, w, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(video_path), fourcc, 30, (w, h))

    # --- 5. Main Evaluation Loop ---
    obs = env.reset()  # VecEnv.reset() returns only the observation
    total_reward = 0.0
    try:
        for t in range(args.max_steps):
            action, _ = agent.predict(obs, deterministic=True)
            
            obs, rewards, dones, infos = env.step(action)
            
            # For a VecEnv (even with n_envs=1), outputs are arrays/lists.
            reward = rewards[0]
            done = dones[0]
            info = infos[0]

            total_reward += reward
            
            reward_info = {k: v for k, v in info.items() if k.startswith("R_")}
            log.info(f"Step {t+1:03d} | Reward: {reward:.3f} | Total Reward: {total_reward:.3f} | Details: {reward_info}")

            frame_rgb = env.render()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

            if done:
                log.info(f"Episode finished after {t+1} steps.")
                break
        
        log.info(f"--- Evaluation Summary ---")
        log.info(f"Final Total Reward: {total_reward:.3f}")
        log.info(f"Success Status: {info.get('is_success', False)}")
        
        for _ in range(30): video_writer.write(frame_bgr) # Add a pause
            
    finally:
        log.info("Releasing resources...")
        video_writer.release()
        env.close()
        log.info("Evaluation complete.")

# +++ END REPLACEMENT +++

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
        "--w_plausibility", type=float, default=0.0,
        help="Weight for the OCTO divergence reward. Should match the training config."
    )
    
    args = parser.parse_args()
    main(args)


"""
python -m run_experiment --run_name "rl_finetune_w_scripted_expert_v2" --bc_init_dir "artifacts/bc_sep_v2" --bc-init-type final --w_guidance 0.1 --w_plausibility 0.0 --total_timesteps 1000000

"""

"""
python -m scripts.evaluate_rl_policy --checkpoint_path "trained_models\advised_stage0_no_guidance_v2\backups\latest_backup.zip" --seed 795

"""