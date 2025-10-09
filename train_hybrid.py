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
from models.custom_sb3_extractor import BCFeaturesExtractor 

from pathlib import Path
from typing import Optional
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from tqdm import tqdm
# --- Project Imports (ensure this script is in the `scripts/` directory) ---
# Add project root to path to allow relative imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from envs.panda_env import PandaEnv
from models.bc_policy import BCNet
from run_experiment import (
    initialize_ppo_agent,
    load_bc_checkpoint,
    transfer_bc_weights,
    setup_environment,
    SingleFileBackupCallback,
    resolve_device
)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from utils.expert_dataset import ExpertDataset
from collections import deque
# Configure logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
)
logger = logging.getLogger("dgpo.train_hybrid")

# ==============================================================================
# START OF NEW CODE BLOCK: PPOWithBCLoss IMPLEMENTATION
# ==============================================================================

import torch.nn.functional as F

class PPOWithBCLoss(PPO):
    """
    PPO that adds a Behavioral Cloning auxiliary loss during the PPO update phase.
    This is the robust, proven, and excellent architecture for hybrid RL.

    :param expert_dataloader: A PyTorch DataLoader that yields batches of expert
                              (observation, action) data.
    :param bc_schedule: A dictionary defining the annealing schedule for the BC loss weight.
                        Expected keys: 'initial', 'final', 'decay_steps'.
    :param bc_updates_per_ppo_update: How many BC gradient steps to perform for each PPO update cycle.
    """
    def __init__(
        self,
        *args,
        expert_dataloader: DataLoader,
        bc_schedule: dict,
        bc_updates_per_ppo_update: int = 10,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        if expert_dataloader is None:
            raise ValueError("`expert_dataloader` must be provided to PPOWithBCLoss.")
            
        self.expert_dataloader = expert_dataloader
        # Create a perpetual iterator for the expert data
        self.expert_iterator = iter(self.expert_dataloader)
        
        self.bc_initial = bc_schedule.get("initial", 1.0)
        self.bc_final = bc_schedule.get("final", 0.0) # Often decays to zero
        self.bc_decay_steps = bc_schedule.get("decay_steps", 1_000_000)
        self.bc_updates_per_ppo_update = bc_updates_per_ppo_update
        self.current_bc_lambda = self.bc_initial

    def _get_next_expert_batch(self) -> tuple[dict, torch.Tensor]:
        """Fetches the next batch from the expert dataloader, looping if necessary."""
        try:
            expert_obs, expert_actions = next(self.expert_iterator)
        except StopIteration:
            # Re-initialize the iterator if the epoch ends
            self.expert_iterator = iter(self.expert_dataloader)
            expert_obs, expert_actions = next(self.expert_iterator)
        
        # Move data to the correct device
        expert_actions = expert_actions.to(self.device)
        expert_obs = {key: tensor.to(self.device) for key, tensor in expert_obs.items()}
        return expert_obs, expert_actions

    def _update_bc_lambda(self) -> None:
        """Linearly anneals the BC loss weight lambda based on training progress."""
        progress = min(1.0, self.num_timesteps / self.bc_decay_steps)
        self.current_bc_lambda = self.bc_initial + progress * (self.bc_final - self.bc_initial)
        self.logger.record("custom/bc_lambda", self.current_bc_lambda)

    def train(self) -> None:
        """
        This is the core of the new architecture.
        It overrides the PPO `train` method to inject the BC auxiliary loss.
        """
        # 1. Run the standard PPO update.
        # This will collect rollouts, compute advantages, and update the policy
        # and value functions based on the RL objective.
        super().train()
        
        # 2. Update the BC loss weight based on our schedule.
        self._update_bc_lambda()
        if self.current_bc_lambda <= 0:
            return  # Skip BC update if the weight is zero

        # 3. Perform the auxiliary BC update.
        self.policy.set_training_mode(True)
        
        bc_losses = []
        # We perform multiple gradient steps on expert data for stability.
        for _ in range(self.bc_updates_per_ppo_update):
            # Get a fresh batch of expert data.
            expert_obs, expert_actions = self._get_next_expert_batch()

            # Perform a forward pass with the current policy on the expert observations.
            # We want the policy's deterministic action for the BC loss.
            policy_predicted_actions, _, _ = self.policy(expert_obs, deterministic=True)

            # Calculate the Behavioral Cloning loss (MSE for continuous actions).
            bc_loss = F.mse_loss(policy_predicted_actions, expert_actions)

            # Combine and optimize: Only the BC loss influences this gradient step.
            total_loss = self.current_bc_lambda * bc_loss

            self.policy.optimizer.zero_grad()
            total_loss.backward()
            # Clip gradients to prevent instability.
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            
            bc_losses.append(bc_loss.item())
        
        # Log the mean BC loss for this update cycle.
        mean_bc_loss = np.mean(bc_losses) if bc_losses else 0
        self.logger.record("custom/bc_loss", mean_bc_loss)



def run_hybrid_training(args: argparse.Namespace):
    """
    Main function to orchestrate the PPO + BC Auxiliary Loss training process.
    This is the robust, proven, and excellent implementation.
    """
    # --- 1. SETUP AND INITIALIZATION ---
    run_name = args.run_name or f"ppo_bc_aux_{Path(args.bc_init_dir).name}"
    run_dir = Path(args.output_dir) / run_name
    checkpoints_dir = run_dir / "checkpoints"
    backups_dir = run_dir / "backups"
    
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)
    backups_dir.mkdir(exist_ok=True)

    with (run_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=4)
    logger.info(f"🚀 Starting PPO + BC Auxiliary Loss experiment: {run_name}")

    set_random_seed(args.seed)
    device_str = resolve_device(args.device)
    device = torch.device(device_str)

    # --- 2. SETUP RL ENVIRONMENT (Unchanged from your working code) ---
    env = setup_environment(
        xml_path=args.xml_path,
        seed=args.seed,
        control_mode='absolute', 
        n_envs=args.n_envs,
        add_monitor_wrapper=True,
        scripted_expert=None, # The RL env does not need an expert
        w_guidance=args.w_guidance,
        w_guidance_dense=args.w_guidance_dense,
        guidance_clip=args.guidance_clip,
        grasp_reward=args.grasp_reward,
        lift_reward=args.lift_reward,
        success_reward=args.success_reward,
        pos_scale=args.pos_scale
    )
    logger.info("RL Environment created and wrapped successfully.")

    # --- 3. PREPARE EXPERT DATALOADER FOR BC LOSS (New Critical Step) ---
    logger.info("Loading ExpertDataset for auxiliary BC loss...")
    # This dataset runs in the background to provide expert data for regularization.
    expert_dataset = ExpertDataset(
        urdf_path=args.urdf_path,
        instruction="pick up the red block",
        use_octo=False,  # Use reliable scripted expert
        base_seed=args.seed + 1000, # Use a different seed for the loader
        yield_full_obs=True, # MUST be True to match the policy's observation space
    )
    expert_dataloader = DataLoader(
        expert_dataset,
        batch_size=args.batch_size, # Use same batch size as PPO
        shuffle=False, # IterableDataset is already randomized
        num_workers=2, # Use background workers to pre-fetch data
        pin_memory=True,
        drop_last=True,
    )

    # --- 4. CREATE THE PPO AGENT (Using PPOWithBCLoss) ---
    logger.info("Creating the PPOWithBCLoss agent...")
    bc_schedule = {
        "initial": args.bc_lambda_initial,
        "final": args.bc_lambda_final,
        "decay_steps": args.bc_decay_steps,
    }

    # Policy and PPO kwargs remain the same as they define the network architecture.
    policy_kwargs = {"features_extractor_class": BCFeaturesExtractor, "net_arch": {"pi": [512, 256], "vf": [512, 256]}}
    ppo_kwargs = {
        "policy": "MultiInputPolicy", "env": env, "policy_kwargs": policy_kwargs,
        "learning_rate": args.learning_rate, "n_steps": args.n_steps, "batch_size": args.batch_size,
        "n_epochs": 10, "gamma": 0.995, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01,
        "vf_coef": 0.5, "max_grad_norm": 0.5, "tensorboard_log": str(run_dir / "logs"),
        "seed": args.seed, "device": device_str, "verbose": 1,
    }

    # Instantiate our new, robust agent.
    agent = PPOWithBCLoss(
        expert_dataloader=expert_dataloader,
        bc_schedule=bc_schedule,
        bc_updates_per_ppo_update=args.bc_updates, # Use CLI arg
        **ppo_kwargs
    )
    
    # --- 5. INITIALIZE WEIGHTS FROM PRE-TRAINED BC MODEL (Best Practice) ---
    try:
        logger.info(f"Attempting weight transfer from BC checkpoint: {args.bc_init_dir}")
        bc_model_path = str(Path(args.bc_init_dir) / "checkpoints" / "best_model.pth")
        # We only need the state dict for the transfer function.
        ckpt_state_dict = load_bc_checkpoint(bc_model_path, device)
        
        # We create a temporary BCNet instance to hold the weights before transfer.
        temp_bc_net = BCNet(n_actions=env.action_space.shape[0]).to(device)
        temp_bc_net.load_state_dict(ckpt_state_dict, strict=False)
        
        transfer_bc_weights(temp_bc_net, agent)
        logger.info("Successfully transferred weights from BC checkpoint to PPO policy.")
        del temp_bc_net # Clean up memory
            
    except Exception as e:
        logger.error(f"Failed during BC weight transfer, continuing from scratch. Error: {e}", exc_info=True)

    if args.freeze_features:
        logger.info("--- FEATURE FREEZING ENABLED ---")
        frozen_keys = 0
        for name, param in agent.policy.named_parameters():
            if 'features_extractor' in name:
                param.requires_grad = False
                frozen_keys += 1
        logger.info(f"Froze {frozen_keys} parameters in the RL policy's feature extractor.")

    logger.info("PPOWithBCLoss agent is fully configured and ready for training.")

    # --- 6. SETUP CALLBACKS AND START TRAINING ---
    callbacks = [
        CheckpointCallback(save_freq=max(1, args.save_freq // args.n_envs), save_path=str(checkpoints_dir), name_prefix="rl_policy"),
        SingleFileBackupCallback(save_freq=max(1, 5000 // args.n_envs), save_path=str(backups_dir), name_prefix="latest_backup"),
    ]

    logger.info("\n--- Starting Robust PPO + BC Auxiliary Loss Training Loop ---")
    try:
        agent.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        agent.save(run_dir / "final_policy.zip")
        logger.info(f"✅ Training complete. Final policy saved to: {run_dir / 'final_policy.zip'}")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user; saving current policy and exiting.")
        agent.save(run_dir / "interrupted_policy.zip")
    except Exception as e:
        logger.exception("Unexpected error during training. Saving state and exiting.", exc_info=True)
        try:
            agent.save(run_dir / "error_policy.zip")
        except Exception as e:
            logger.error(f"Failed to save emergency policy after error. Error: {e}")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Robust PPO + BC Auxiliary Loss Training")

    # --- Run Management ---
    parser.add_argument("--run_name", type=str, default=None, help="Name for the training run.")
    parser.add_argument("--output_dir", type=str, default="trained_models", help="Directory to save artifacts.")
    parser.add_argument("--bc_init_dir", type=str, required=True, help="Path to the balanced BC run directory for weight initialization.")
    
    # --- Training Parameters ---
    parser.add_argument("--total_timesteps", type=int, default=3_000_000, help="Total timesteps for training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use ('cpu', 'cuda', 'auto').")
    parser.add_argument("--n_envs", type=int, default=8, help="Number of parallel environments.")
    parser.add_argument("--save_freq", type=int, default=50000, help="Frequency to save checkpoints.")
    
    # --- Environment Parameters ---
    parser.add_argument("--xml_path", type=str, default="envs/panda_pick_place.xml", help="Path to MuJoCo XML.")
    parser.add_argument("--urdf_path", type=str, default="urdf/panda_mujoco_kinematics.urdf", help="Path to URDF.")
    
    # --- PPO Hyperparameters ---
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the PPO agent.")
    parser.add_argument("--n_steps", type=int, default=2048, help="PPO rollout buffer size.")
    parser.add_argument("--batch_size", type=int, default=64, help="PPO and BC auxiliary loss batch size.")
    
    # --- NEW: BC Auxiliary Loss Hyperparameters ---
    parser.add_argument("--bc_lambda_initial", type=float, default=1.0, help="Initial weight for the BC auxiliary loss.")
    parser.add_argument("--bc_lambda_final", type=float, default=0.0, help="Final BC loss weight after decay.")
    parser.add_argument("--bc_decay_steps", type=int, default=1_500_000, help="Timesteps over which the BC loss weight decays.")
    parser.add_argument("--bc_updates", type=int, default=10, help="Number of BC gradient steps per PPO update cycle.")
    
    # --- Feature and Reward Hyperparameters (Unchanged) ---
    parser.add_argument("--freeze_features", action="store_true", help="If set, freeze the feature extractor layers.")
    parser.add_argument("--w_guidance", type=float, default=0.0, help="Weight for ScriptedExpert guidance terminal reward.")
    parser.add_argument("--w_guidance_dense", type=float, default=5.0, help="Weight for DENSE ScriptedExpert guidance reward.")
    parser.add_argument("--guidance_clip", type=float, default=1.0)
    parser.add_argument("--grasp_reward", type=float, default=50.0)
    parser.add_argument("--lift_reward", type=float, default=100.0)
    parser.add_argument("--success_reward", type=float, default=250.0)
    parser.add_argument("--pos_scale", type=float, default=0.05, help="Action scaling factor for the delta controller.")
    
    args = parser.parse_args()
    
    # --- DELETED command line examples for AdvisedPPO ---
    # Delete the old python -m train_hybrid ... lines
    # And add new, correct examples for yourself.
    
    # Example usage:
    # python -m scripts.train_hybrid --run_name "robust_hybrid_v1" \
    #   --bc_init_dir "artifacts/bc_final_balanced_v1" \
    #   --n_envs 8 \
    #   --total_timesteps 3000000 \
    #   --bc_lambda_initial 1.0 \
    #   --bc_lambda_final 0.0 \
    #   --bc_decay_steps 1500000 \
    #   --freeze_features
    
    try:
        run_hybrid_training(args)
    except Exception as e:
        logger.exception("An error occurred during training.")
        sys.exit(1)


"""
python -m scripts.train_hybrid --run_name "final_hybrid_run_v1" \
    --bc_init_dir "artifacts/bc_retrained_balanced_v1" \
    --n_envs 8 \
    --total_timesteps 3000000

python -m scripts.train_hybrid --run_name "final_hybrid_run_v1" \
    --bc_init_dir "artifacts/bc_final_balanced_v1" \
    --n_envs 8 \
    --total_timesteps 3000000 \
    --freeze_features

python -m train_hybrid --run_name "hybrid_stage0_no_guidance" --bc_init_dir "artifacts/bc_final_balanced_v1" --n_envs 1 --total_timesteps 200000 --device "cpu" --w_guidance 0.0 --w_guidance_dense 0.0 --bc_updates 4 --bc_lr 1e-5 --bc_lambda_initial 1.0 --freeze_features



python train_hybrid.py --run_name "hybrid_stage0_no_guidance" --bc_init_dir "artifacts/bc_final_balanced_v1" --n_envs 1 --total_timesteps 200000 --device "cpu" --w_guidance 0.0 --w_guidance_dense 0.0 --bc_updates 4 --bc_lr 1e-5 --bc_lambda_initial 1.0 --freeze_features


python -m train_hybrid --run_name "advised_stage0_no_guidance_v2" --bc_init_dir "artifacts/bc_final_balanced_v1" --n_envs 1 --total_timesteps 200000 --device "cpu" --w_guidance 0.0 --w_guidance_dense 0.0 --freeze_features --eps_initial 0.3 --eps_final 0.05 --eps_decay_steps 100000
"""