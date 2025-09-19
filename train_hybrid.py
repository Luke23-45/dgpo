# FILE: scripts/train_hybrid.py

"""
Final, robust hybrid training script for DGPO-Foundation.

This script implements a "best practice" approach by combining Reinforcement
Learning (PPO) with Behavior Cloning (BC) regularization. It addresses all
previously identified failure modes:

1.  **Corrects Control Mismatch:** The environment is explicitly created in
    `control_mode='absolute'` to match the semantics of the BC policy.
2.  **Implements Imitation Regularization:** It uses an auxiliary BC loss to
    anchor the RL policy to the expert's behavior, preventing "catastrophic
    forgetting" during exploration.
3.  **Uses Balanced Expert Data:** The BC loss is calculated using data from the
    balanced ExpertDataset, ensuring the agent is anchored to a competent policy
    that has learned the critical (but rare) grasp/lift phases.
4.  **Annealing Schedule:** The weight of the BC loss (lambda) is annealed from
    a high value to zero over the course of training, allowing the agent to
    gradually rely more on the RL reward signal as it becomes more competent.
5.  **Robust and Clean:** The implementation wraps the SB3 training loop rather
    than modifying its internals, ensuring stability and maintainability.
"""

import argparse
import logging
import json
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# --- Project Imports (ensure this script is in the `scripts/` directory) ---
# Add project root to path to allow relative imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from envs.panda_env import PandaEnv
from models.bc_policy import BCNet
from scripts.arun_experiment import (
    initialize_ppo_agent,
    load_bc_checkpoint,
    transfer_bc_weights,
    setup_environment,
    SingleFileBackupCallback,
)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from utils.expert_dataset import ExpertDataset

# Configure logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
)
logger = logging.getLogger("dgpo.train_hybrid")


def run_hybrid_training(args: argparse.Namespace):
    """
    Main function to orchestrate the hybrid BC+RL training process.
    """
    # --- 1. SETUP AND INITIALIZATION ---
    # This section is very similar to `run_experiment.py`, but ensures
    # the environment is configured correctly for our hybrid strategy.
    
    run_name = args.run_name or f"hybrid_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(args.output_dir) / run_name
    checkpoints_dir = run_dir / "checkpoints"
    backups_dir = run_dir / "backups"
    
    # Create directories for artifacts
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)
    backups_dir.mkdir(exist_ok=True)

    # Save the configuration for this run
    with (run_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=4)
    logger.info(f"🚀 Starting Hybrid BC+RL experiment: {run_name}")
    logger.info(f"All artifacts will be saved in: {run_dir}")

    # Set random seeds for reproducibility
    set_random_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- CRITICAL FIX: Setup environment with ABSOLUTE control mode ---
    def make_absolute_control_env():
        # This wrapper function ensures every environment instance is created correctly.
        env = PandaEnv(xml_path=args.xml_path, control_mode='absolute')
        # We can add the reward wrapper and other wrappers here if needed,
        # but for this strategy, we'll start with the base environment reward
        # and focus on the BC + sparse reward signal.
        return env

    # Using setup_environment from run_experiment but overriding the env creation
    # to ensure absolute control mode. Let's create a simplified version here.
    from stable_baselines3.common.env_util import make_vec_env
    from utils.rl_reward_wrapper import RLRewardWrapper
    from utils.obs_adapters import OctoToSB3Adapter

    def setup_hybrid_environment():
        def make_env():
            env = PandaEnv(xml_path=args.xml_path, control_mode='absolute')
            env = RLRewardWrapper(
                env,
                grasp_reward=args.grasp_reward, # Use CLI-configurable rewards
                lift_reward=args.lift_reward,
                success_reward=args.success_reward,
                w_guidance_dense=0.0 # Disable dense guidance by default
            )
            env = OctoToSB3Adapter(env)
            return env
        
        vec_env = make_vec_env(lambda: make_env(), n_envs=args.n_envs, seed=args.seed)
        return vec_env

    env = setup_hybrid_environment()
    logger.info(f"Environment created with control_mode='absolute'")

    # Initialize PPO agent
    ppo_agent = initialize_ppo_agent(env, run_dir, args.seed, args.device)

    # Load the new, balanced BC model and transfer weights
    bc_model_path = str(Path(args.bc_init_dir) / "checkpoints" / "best_model.pth")
    logger.info(f"Loading balanced BC checkpoint from: {bc_model_path}")
    try:
        device = ppo_agent.policy.device
        state_dict = load_bc_checkpoint(bc_model_path, device)
        action_dim = env.action_space.shape[0]
        
        bc_net = BCNet(n_actions=action_dim).to(device)
        bc_net.load_state_dict(state_dict)
        
        if transfer_bc_weights:
            transfer_bc_weights(bc_net, ppo_agent)
            logger.info("Weight transfer from balanced BC model successful.")
        else:
            logger.warning("`transfer_bc_weights` utility not available. Agent is random.")
    except Exception as e:
        logger.error(f"Failed during BC weight transfer: {e}", exc_info=True)
        env.close()
        return

    # --- 2. PREPARE FOR IMITATION REGULARIZATION ---
    logger.info("\n--- Preparing for Imitation Regularization ---")
    
    # 2.1: Create the ExpertDataset and DataLoader for BC updates
    logger.info("Initializing Expert Dataloader for BC updates...")
    expert_dataset = ExpertDataset(
        urdf_path=args.urdf_path,
        base_seed=args.seed + 1000,
        max_samples_per_epoch=None,
        yield_full_obs=False
    )
    
    expert_loader = DataLoader(
        expert_dataset,
        batch_size=args.bc_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    expert_iterator = iter(expert_loader)
    logger.info("Expert Dataloader is ready.")

    # 2.2: Create a separate optimizer for the BC updates
    # We only update the policy network (actor), not the value network.
    policy_params_to_update = list(ppo_agent.policy.parameters())

    bc_optimizer = torch.optim.Adam(
        policy_params_to_update, lr=args.bc_lr, weight_decay=1e-6
    )
    bc_loss_fn = nn.MSELoss()
    logger.info("BC optimizer configured for policy network.")

    # --- 3. HYBRID TRAINING LOOP ---
    logger.info("\n--- Starting Hybrid Training Loop ---")

    # 3.1: Define Training Parameters
    rl_steps_per_iteration = args.n_steps * args.n_envs
    bc_updates_per_iteration = args.bc_updates
    initial_bc_lambda = args.bc_lambda_initial
    total_timesteps = args.total_timesteps

    # Setup SB3 Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.save_freq // args.n_envs),
        save_path=str(checkpoints_dir),
        name_prefix="hybrid_rl_policy"
    )
    backup_callback = SingleFileBackupCallback(
        save_freq=max(1, 5000 // args.n_envs),
        save_path=str(backups_dir),
        name_prefix="latest_backup"
    )
    callback_list = [checkpoint_callback, backup_callback]

    start_time = time.time()
    ppo_agent.num_timesteps = 0
    ppo_agent._episode_num = 0
    ppo_agent._total_timesteps = total_timesteps

    # 3.2: The Main Loop
    while ppo_agent.num_timesteps < total_timesteps:
        
        # --- PHASE A: Reinforcement Learning ---
        ppo_agent.learn(
            total_timesteps=rl_steps_per_iteration,
            callback=callback_list,
            reset_num_timesteps=False,
            log_interval=1
        )
        
        # --- PHASE B: Imitation Learning (BC Regularization) ---
        progress = ppo_agent.num_timesteps / total_timesteps
        current_bc_lambda = initial_bc_lambda * (1.0 - progress)

        total_bc_loss = 0.0
        if current_bc_lambda > 0:
            for _ in range(bc_updates_per_iteration):
                try:
                    obs_expert, act_expert = next(expert_iterator)
                except StopIteration:
                    expert_iterator = iter(expert_loader)
                    obs_expert, act_expert = next(expert_iterator)
                
                act_expert = act_expert.to(ppo_agent.policy.device)
                obs_expert_device = {k: v.to(ppo_agent.policy.device) for k, v in obs_expert.items()}
                
                # Get policy's predicted action distribution
                # For PPO, the policy outputs a distribution, not a raw action
                features = ppo_agent.policy.extract_features(obs_expert_device)
                latent_pi = ppo_agent.policy.mlp_extractor.forward_actor(features)
                distribution = ppo_agent.policy._get_action_dist_from_latent(latent_pi)
                predicted_actions = distribution.mode # Use the mode (most likely action) for BC loss
                
                bc_loss = bc_loss_fn(predicted_actions, act_expert)
                weighted_bc_loss = bc_loss * current_bc_lambda
                
                bc_optimizer.zero_grad()
                weighted_bc_loss.backward()
                nn.utils.clip_grad_norm_(policy_params_to_update, 1.0)
                bc_optimizer.step()
                
                total_bc_loss += bc_loss.item()
        
        # --- 3.3: Logging ---
        avg_bc_loss = total_bc_loss / bc_updates_per_iteration if bc_updates_per_iteration > 0 else 0
        
        ppo_agent.logger.record("custom/bc_loss", avg_bc_loss)
        ppo_agent.logger.record("custom/bc_lambda", current_bc_lambda)
        
        # Dump all logs (RL + custom) to TensorBoard
        ppo_agent.logger.dump(step=ppo_agent.num_timesteps)

        logger.info(f"Timesteps: {ppo_agent.num_timesteps}/{total_timesteps} | "
                    f"BC Lambda: {current_bc_lambda:.3f} | Avg BC Loss: {avg_bc_loss:.5f}")

    # --- 4. FINAL SAVE AND CLEANUP ---
    final_model_path = run_dir / "final_policy.zip"
    ppo_agent.save(final_model_path)
    logger.info(f"✅ Training complete. Final policy saved to: {final_model_path}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hybrid BC+RL Training for DGPO")

    # --- Run Management ---
    parser.add_argument("--run_name", type=str, default=None, help="Name for the training run.")
    parser.add_argument("--output_dir", type=str, default="trained_models", help="Directory to save training artifacts.")
    parser.add_argument("--bc_init_dir", type=str, required=True, help="Path to the COMPLETED, balanced BC run directory.")
    
    # --- Training Parameters ---
    parser.add_argument("--total_timesteps", type=int, default=2_000_000, help="Total timesteps for training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use ('cpu', 'cuda', 'auto').")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments.")
    parser.add_argument("--save_freq", type=int, default=50000, help="Frequency to save checkpoints.")
    
    # --- Environment Parameters ---
    parser.add_argument("--xml_path", type=str, default="envs/panda_pick_place.xml", help="Path to the MuJoCo XML file.")
    parser.add_argument("--urdf_path", type=str, default="urdf/panda_mujoco_kinematics.urdf", help="Path to the URDF file for IKSolver.")

    # --- Hybrid Training Hyperparameters ---
    parser.add_argument("--bc_batch_size", type=int, default=128, help="Batch size for the BC update steps.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for the expert dataloader.")
    parser.add_argument("--bc_lr", type=float, default=1e-4, help="Learning rate for the BC optimizer.")
    parser.add_argument("--bc_updates", type=int, default=10, help="Number of BC gradient updates per RL iteration.")
    parser.add_argument("--bc_lambda_initial", type=float, default=1.0, help="Initial weight for the BC loss.")
    
    # --- Reward Hyperparameters ---
    parser.add_argument("--grasp_reward", type=float, default=25.0, help="Sparse reward for grasping the object.")
    parser.add_argument("--lift_reward", type=float, default=50.0, help="Sparse reward for lifting the object.")
    parser.add_argument("--success_reward", type=float, default=200.0, help="Sparse reward for task success.")
    
    # --- PPO Hyperparameters (from run_experiment) ---
    parser.add_argument("--n_steps", type=int, default=2048, help="PPO rollout buffer size.")

    args = parser.parse_args()
    
    try:
        run_hybrid_training(args)
    except Exception as e:
        logger.exception("An error occurred during hybrid training.")
        sys.exit(1)


"""
python -m scripts.train_hybrid --run_name "final_hybrid_run_v1" \
    --bc_init_dir "artifacts/bc_retrained_balanced_v1" \
    --n_envs 8 \
    --total_timesteps 3000000
"""