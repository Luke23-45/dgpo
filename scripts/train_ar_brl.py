import argparse
import logging
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch

# --- Project Imports ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from run_experiment import (
    initialize_ppo_agent, load_bc_checkpoint, setup_environment, SingleFileBackupCallback
)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.buffers import RolloutBuffer

from utils.expert_dataset import ExpertDataset
from utils.bootstrap_buffer import bootstrap_replay_buffer
from models.bc_policy import BCNet
from models.residual_policy import AdaptiveResidualPolicy

# Configure logging
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger("dgpo.train_ar_brl")
def resolve_device(device_arg: str) -> str:
    """Resolve 'auto' to 'cuda' if available else 'cpu' and validate."""
    if device_arg is None or device_arg.lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg

# In scripts/train_ar_brl.py

def run_ar_brl_training(args: argparse.Namespace):
    """
    Main function for the Adaptive Residual Bootstrapped RL training process.
    """
    # --- 1. SETUP AND INITIALIZATION ---
    run_name = args.run_name or f"ar_brl_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(args.output_dir) / run_name
    checkpoints_dir = run_dir / "checkpoints"
    backups_dir = run_dir / "backups"
    
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)
    backups_dir.mkdir(exist_ok=True)

    with (run_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=4)
    logger.info(f"🚀 Starting AR-BRL experiment: {run_name}")

    set_random_seed(args.seed)
    torch.manual_seed(args.seed)

    # <--- START OF FIX: RESOLVE DEVICE STRING EARLY --->
    device_str = resolve_device(args.device)
    device = torch.device(device_str)
    logger.info(f"Resolved device '{args.device}' to '{device_str}'")
    # <--- END OF FIX --->

    # --- 2. SETUP ENVIRONMENT & BC MODEL ---
    env = setup_environment(
        xml_path=args.xml_path, seed=args.seed, control_mode='absolute', n_envs=args.n_envs,
        w_guidance_dense=args.w_guidance_dense, grasp_reward=args.grasp_reward,
        lift_reward=args.lift_reward, success_reward=args.success_reward
    )

    logger.info("Loading FROZEN BC model to act as a base...")
    bc_model_path = str(Path(args.bc_init_dir) / "checkpoints" / "best_model.pth")
    # <--- FIX: Use the resolved `device` object --->
    state_dict = load_bc_checkpoint(bc_model_path, device)
    action_dim = env.action_space.shape[0]
    # <--- FIX: Use the resolved `device` object --->
    bc_model = BCNet(n_actions=action_dim).to(device)
    bc_model.load_state_dict(state_dict, strict=False)
    bc_model.eval() # Set to evaluation mode

    # --- 3. INITIALIZE THE PPO AGENT WITH THE CUSTOM RESIDUAL POLICY ---
    policy_kwargs = {
        "bc_model": bc_model,
        "net_arch": [], # We define nets inside the policy, so SB3's arch is disabled
        "ortho_init": False # Important for residual learning
    }

    # Use conservative PPO hyperparams for stable fine-tuning
    ppo_agent = PPO(
        AdaptiveResidualPolicy,
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.005, # Slightly higher entropy to encourage exploring residuals
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(run_dir / "logs"),
        seed=args.seed,
        # <--- FIX: Use the resolved `device_str` string --->
        device=device_str,
        verbose=1
    )
    
    logger.info("PPO agent with AdaptiveResidualPolicy created.")
    
    # --- 4. BOOTSTRAP THE REPLAY BUFFER ---
    logger.info("Preparing ExpertDataset for bootstrapping...")
    expert_dataset = ExpertDataset(
        urdf_path=args.urdf_path,
        base_seed=args.seed + 1000,
        env_xml_path=args.xml_path,
        use_octo=False # Use scripted expert for bootstrap data
    )
    
    # Bootstrap a fraction of the full buffer size
    bootstrap_size = int(args.bootstrap_fraction * ppo_agent.rollout_buffer.buffer_size)
    bootstrap_replay_buffer(
        ppo_agent.rollout_buffer,
        env,
        ppo_agent.policy,
        expert_dataset,
        num_samples=bootstrap_size,
        batch_size=args.batch_size
    )

    # --- 5. SETUP CALLBACKS AND START TRAINING ---
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.save_freq // args.n_envs),
        save_path=str(checkpoints_dir),
        name_prefix="ar_brl_policy"
    )
    backup_callback = SingleFileBackupCallback(
        save_freq=max(1, 5000 // args.n_envs),
        save_path=str(backups_dir),
        name_prefix="latest_backup"
    )

    logger.info("\n--- Starting AR-BRL Training Loop ---")
    try:
        ppo_agent.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, backup_callback],
            progress_bar=True
        )
        final_model_path = run_dir / "final_policy.zip"
        ppo_agent.save(final_model_path)
        logger.info(f"✅ Training complete. Final policy saved to: {final_model_path}")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AR-BRL Training for DGPO")
    # Add relevant arguments from train_hybrid.py
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="trained_models_ar_brl")
    parser.add_argument("--bc_init_dir", type=str, required=True)
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=50000)
    parser.add_argument("--xml_path", type=str, default="envs/panda_pick_place.xml")
    parser.add_argument("--urdf_path", type=str, default="urdf/panda_mujoco_kinematics.urdf")
    
    # New AR-BRL specific hypers
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--bootstrap_fraction", type=float, default=0.5, 
                        help="Fraction of the rollout buffer to pre-fill with expert data.")
                        
    # Reward Hyperparameters
    parser.add_argument("--w_guidance_dense", type=float, default=5.0)
    parser.add_argument("--grasp_reward", type=float, default=50.0)
    parser.add_argument("--lift_reward", type=float, default=100.0)
    parser.add_argument("--success_reward", type=float, default=250.0)
    
    args = parser.parse_args()
    run_ar_brl_training(args)

# Example CLI command:
# python -m scripts.train_ar_brl --run_name "ar_brl_v1" --bc_init_dir "artifacts/bc_final_balanced_v1" --n_envs 1 --total_timesteps 3000000