# FILE: scripts/train_rl.py
"""
Dedicated, high-quality script for Phase 3: RL Fine-tuning with TQC and HER.

This script fine-tunes a pre-trained DiffusionPolicy using the custom DiffusionTQC
algorithm. It leverages Hindsight Experience Replay (HER) for sample efficiency
and a dual-stream auxiliary BC loss for regularization.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
import gymnasium as gym
import numpy as np

# --- Project Imports ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from algos.diffusion_tqc import DiffusionTQC, DiffusionTQCPolicy
from envs.panda_env import PandaEnv
from envs.panda_env_wrapper import GoalPandaEnv # The HER wrapper
from models.custom_sb3_extractor import BCFeaturesExtractor
from models.diffusion_policy import ConditionalDenoiser, DiffusionPolicy
from utils.expert_dataset import ExpertDataset
from utils.rl_reward_wrapper import RLRewardWrapper
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.her import HerReplayBuffer

# --- Local Imports from Pre-training Script ---
from scripts.pretrain_diffusion import ExpertTrajectoryDataset, collate_fn

# Configure logging
logging.basicConfig(format="%(asctime)s | %(levelname)s | [%(name)s] | %(message)s", level=logging.INFO)
logger = logging.getLogger("dgpo.train_rl")

def main(args: argparse.Namespace):
    """Orchestrates the entire RL fine-tuning process."""
    
    run_name = args.run_name or f"tqc_her_finetune_{int(time.time())}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    with (run_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=4)
    
    logger.info(f"🚀 Starting TQC+HER RL Fine-tuning: {run_name}")
    logger.info(f"TensorBoard logs will be saved to: {run_dir / 'logs'}")

    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Create the Environment with HER Wrapper ---
    logger.info(f"Creating {args.n_envs} parallel environments for HER...")
    def make_env():
        env = PandaEnv(xml_path=args.xml_path, control_mode='delta')
        env = RLRewardWrapper(env)
        # CRITICAL: Wrap with GoalPandaEnv for HER compatibility
        env = GoalPandaEnv(env)
        return env
    
    env = make_vec_env(make_env, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)

    # --- 2. Create Offline Expert DataLoader for BC Loss ---
    logger.info("Setting up offline expert data loader...")
    offline_expert_dataset = ExpertTrajectoryDataset(Path(args.demo_path))
    offline_loader = DataLoader(
        offline_expert_dataset, batch_size=args.tqc_batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn, drop_last=True, persistent_workers=True
    )
    
    # --- 3. Define Replay Buffer and Policy Architecture ---
    # HER requires a specific replay buffer class
    replay_buffer_class = HerReplayBuffer
    replay_buffer_kwargs = {
        'n_sampled_goal': 4,
        'goal_selection_strategy': 'future',
        'online_sampling': True,
        'max_episode_length': args.max_episode_steps
    }

    # Define the arguments for our custom DiffusionPolicy actor
    obs_feature_dim = BCFeaturesExtractor(env.observation_space['observation']).features_dim
    policy_kwargs = {
        'features_extractor_class': BCFeaturesExtractor,
        'features_extractor_kwargs': dict(observation_space=env.observation_space['observation']),
        'net_arch': [256, 256], # Critic network architecture
        
        # Custom arguments for our DiffusionTQCPolicy
        'actor_kwargs': {
             'denoiser': ConditionalDenoiser(
                action_dim=env.action_space.shape[0],
                obs_feature_dim=obs_feature_dim
             ),
             'action_dim': env.action_space.shape[0]
        }
    }

    # --- 4. Instantiate the Custom TQC Agent ---
    logger.info("Instantiating DiffusionTQC agent...")
    agent = DiffusionTQC(
        policy=DiffusionTQCPolicy,
        env=env,
        offline_expert_loader=offline_loader,
        lambda_bc=args.lambda_bc,
        learning_rate=args.rl_lr,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.tqc_batch_size,
        gamma=args.gamma,
        train_freq=(1, "step"),
        gradient_steps=-1, # Run as many updates as possible
        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs=replay_buffer_kwargs,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(run_dir / "logs"),
        seed=args.seed,
        device=device,
        verbose=1,
    )

    # --- 5. Load Pre-trained Weights ---
    policy_path = Path(args.pretrained_policy_path)
    if policy_path.exists():
        # The actor's neural network is inside actor.mu
        agent.policy.actor.mu.load_state_dict(torch.load(policy_path, map_location=device))
        logger.info(f"✅ Successfully loaded pre-trained weights from {policy_path}")
    else:
        logger.warning(f"⚠️ Pre-trained policy not found at {policy_path}. Starting RL from scratch.")

    # --- 6. Setup Callbacks and Start Training ---
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 50000), # Save every 50k total steps
        save_path=str(checkpoints_dir),
        name_prefix="rl_model",
        save_replay_buffer=True,
    )

    logger.info("--- Starting RL Fine-tuning Loop ---")
    try:
        agent.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback, progress_bar=True)
        
        final_model_path = run_dir / "final_model.zip"
        agent.save(final_model_path)
        logger.info(f"✅ Training complete. Final model saved to {final_model_path}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving current model.")
        agent.save(run_dir / "interrupted_model.zip")
    finally:
        env.close()
        logger.info("--- RL Fine-tuning complete. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 3: RL Fine-tuning with TQC+HER")
    
    # --- Required Paths ---
    parser.add_argument("--demo_path", type=str, required=True, help="Path to the offline expert_demos.pkl file.")
    parser.add_argument("--pretrained_policy_path", type=str, required=True, help="Path to the pre-trained policy weights (.pth).")

    # --- Run Management ---
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="trained_models/tqc_her_finetune")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    # --- RL Hyperparameters ---
    parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Off-policy is more sample efficient.")
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--buffer_size", type=int, default=500_000)
    parser.add_argument("--learning_starts", type=int, default=10000)
    parser.add_argument("--tqc_batch_size", type=int, default=256)
    parser.add_argument("--rl_lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--lambda_bc", type=float, default=0.1, help="Weight for the Advantage-Weighted BC auxiliary loss.")
    
    # --- Environment/Expert Configs ---
    parser.add_argument("--urdf_path", type=str, default="urdf/panda_mujoco_kinematics.urdf")
    parser.add_argument("--xml_path", type=str, default="envs/panda_pick_place.xml")
    parser.add_argument("--max_episode_steps", type=int, default=200, help="Max steps per episode for HER buffer.")

    args = parser.parse_args()
    main(args)