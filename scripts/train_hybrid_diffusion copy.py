# FILE: scripts/train_hybrid_diffusion.py
"""
Final, SOTA-aligned hybrid training script for DGPO-Foundation.

This script implements the "Hybrid Differentiable Diffusion-PPO" methodology with a
dual online/offline data stream for the auxiliary BC loss, ensuring utmost quality and robustness.

It orchestrates the full end-to-end pipeline:
1.  **Expert Demonstration Collection:** Generates and saves a high-quality offline dataset.
2.  **BC Pre-training:** Trains a DiffusionPolicy via pure behavioral cloning on the offline dataset.
3.  **RL Fine-tuning:** Fine-tunes the policy using our custom DiffusionPPO agent.
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
import sys
import time
from typing import Dict, Tuple
from itertools import islice
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import gymnasium as gym

# --- Project Imports ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from algos.diffusion_ppo import DiffusionPPO, DiffusionActorCriticPolicy
from envs.panda_env import PandaEnv
from models.custom_sb3_extractor import BCFeaturesExtractor
from models.diffusion_policy import ConditionalDenoiser, DiffusionPolicy
from utils.expert_dataset import ExpertDataset
from utils.rl_reward_wrapper import RLRewardWrapper
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

# Configure logging
logging.basicConfig(format="%(asctime)s | %(levelname)s | [%(name)s] | %(message)s", level=logging.INFO)
logger = logging.getLogger("dgpo.train_pipeline")

# --- Data Handling ---

class ExpertTrajectoryDataset(Dataset):
    """Simple map-style dataset for loading pickled expert trajectories."""
    def __init__(self, trajectory_path: Path):
        with open(trajectory_path, 'rb') as f:
            trajectories = pickle.load(f)
        
        self.observations = []
        self.actions = []
        for traj in trajectories:
            for obs, act in zip(traj['observations'], traj['actions']):
                self.observations.append({
                    'image_primary': obs['image_primary'],
                    'proprio': obs['proprio']
                })
                self.actions.append(act)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]

def collate_fn(batch):
    """Custom collate function to handle dictionary observations."""
    obs_list, act_list = zip(*batch)
    actions = torch.from_numpy(np.stack(act_list).astype(np.float32))
    obs_keys = obs_list[0].keys()
    observations = {key: torch.from_numpy(np.stack([obs[key] for obs in obs_list])) for key in obs_keys}
    if observations['image_primary'].dim() == 4 and observations['image_primary'].shape[-1] == 3:
        observations['image_primary'] = observations['image_primary'].permute(0, 3, 1, 2)
    return observations, actions




def collect_expert_demos(args: argparse.Namespace, demo_path: Path):
    """Stage 1: Collect and save expert demonstrations."""
    logger.info("--- STAGE 1: Collecting Expert Demonstrations ---")
    
    # The ExpertDataset is a self-contained episode runner and data converter.
    # We just need to iterate over it to get complete, successful trajectories.
    expert_dataset = ExpertDataset(
        urdf_path=args.urdf_path,
        use_octo=False,
        yield_full_obs=True, # We need the full obs for saving
        action_scaling_factor=PandaEnv.ACTION_SCALING_FACTOR
    )
    
    # Use itertools.islice to collect a specific number of items from the iterator.
    trajectories = list(tqdm(
        islice(expert_dataset, args.num_expert_demos),
        total=args.num_expert_demos,
        desc="Collecting Demos"
    ))
    
    with open(demo_path, 'wb') as f:
        pickle.dump(trajectories, f)
    logger.info(f"✅ Saved {len(trajectories)} successful expert demos to {demo_path}")


def pretrain_policy(args: argparse.Namespace, demo_path: Path, policy_path: Path, device: torch.device):
    """Stage 2: Pre-train the DiffusionPolicy using Behavioral Cloning."""
    logger.info("--- STAGE 2: BC Pre-training Diffusion Policy ---")

    dummy_obs_space = gym.spaces.Dict({
        'image_primary': gym.spaces.Box(0, 255, (256, 256, 3), np.uint8, shape=(256, 256, 3)),
        'proprio': gym.spaces.Box(-np.inf, np.inf, (22,), np.float32, shape=(22,))
    })
    obs_encoder = BCFeaturesExtractor(dummy_obs_space)
    obs_feature_dim = obs_encoder.features_dim
    denoiser = ConditionalDenoiser(action_dim=8, obs_feature_dim=obs_feature_dim)
    policy = DiffusionPolicy(obs_encoder, denoiser, action_dim=8).to(device)

    bc_dataset = ExpertTrajectoryDataset(demo_path)
    bc_dataloader = DataLoader(bc_dataset, batch_size=args.bc_batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn, persistent_workers=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.bc_lr)

    for epoch in tqdm(range(args.bc_epochs), desc="BC Pre-training"):
        for obs_batch, act_batch in bc_dataloader:
            obs_batch = {k: v.to(device) for k, v in obs_batch.items()}
            act_batch = act_batch.to(device)
            loss = policy.compute_loss(obs_batch, act_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(policy.state_dict(), policy_path)
    logger.info(f"✅ Saved pre-trained policy to {policy_path}")

def run_rl_finetuning(args: argparse.Namespace, demo_path: Path, policy_path: Path, run_dir: Path, device: torch.device):
    """Stage 3: Fine-tune the pre-trained policy with DiffusionPPO."""
    logger.info("--- STAGE 3: RL Fine-tuning with DiffusionPPO ---")

    # 1. Create vectorized RL environment
    def make_env():
        env = PandaEnv(xml_path=args.xml_path, control_mode='delta')
        env = RLRewardWrapper(env)
        return env
    
    vec_env = make_vec_env(make_env, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)
    vec_env = VecNormalize(vec_env, norm_obs=False, gamma=args.gamma)

    # 2. Create DataLoaders for BC loss
    offline_expert_dataset = ExpertTrajectoryDataset(demo_path)
    offline_loader = DataLoader(
        offline_expert_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn, drop_last=True, persistent_workers=True
    )
    # The online dataset needs its own env instance per worker
    online_expert_dataset = ExpertDataset(urdf_path=args.urdf_path, use_octo=False, action_scaling_factor=PandaEnv.ACTION_SCALING_FACTOR)
    online_loader = DataLoader(
        online_expert_dataset, batch_size=args.batch_size, num_workers=2,
        collate_fn=collate_fn, persistent_workers=True
    )



    policy_kwargs = {
        'features_extractor_class': BCFeaturesExtractor,
        # Let SB3 compute features_dim automatically
        'net_arch': [], # Actor and Critic will use the shared feature extractor
        
        # Pass the CLASSES and ARGS for our custom components
        'diffusion_policy_class': DiffusionPolicy,
        'diffusion_policy_kwargs': {
            # obs_feature_extractor will be passed in by the policy's _build_actor method
            'denoiser': ConditionalDenoiser(
                action_dim=vec_env.action_space.shape[0],
                # We can calculate the expected feature dim beforehand for the denoiser
                obs_feature_dim=BCFeaturesExtractor(vec_env.observation_space).features_dim
            ),
            'action_dim': vec_env.action_space.shape[0],
        }
    }

    # 4. Instantiate the DiffusionPPO agent
    agent = DiffusionPPO(
        policy=DiffusionActorCriticPolicy,
        env=vec_env,
        offline_expert_loader=offline_loader,
        online_expert_loader=online_loader,
        lambda_bc=args.lambda_bc,
        lambda_annealing_steps=args.lambda_annealing_steps,
        learning_rate=args.rl_lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=1.0,
        tensorboard_log=str(run_dir / "logs"),
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        device=device,
        verbose=1,
    )

    # Load pre-trained weights into the actor part of the policy
    agent.policy.actor.load_state_dict(torch.load(policy_path, map_location=device))
    logger.info("Successfully loaded pre-trained weights into RL policy's actor.")

    # 5. Set up callbacks and start training
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    callbacks = [
        CheckpointCallback(save_freq=max(1, 50000 // args.n_envs), save_path=str(checkpoints_dir), name_prefix="rl_model"),
    ]

    try:
        agent.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=True)
        agent.save(run_dir / "final_policy.zip")
        logger.info(f"✅ Training complete. Final policy saved to {run_dir / 'final_policy.zip'}")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving current model.")
        agent.save(run_dir / "interrupted_policy.zip")
    finally:
        vec_env.close()

def main():
    parser = argparse.ArgumentParser(description="Run Hybrid Diffusion-PPO Training Pipeline")
    # General
    parser.add_argument("--run_name", type=str, default=None, help="Name for the training run.")
    parser.add_argument("--output_dir", type=str, default="trained_models/diffusion_hybrid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    # Stage 1: Demo Collection
    parser.add_argument("--num_expert_demos", type=int, default=200, help="Number of successful expert trajectories to collect.")
    parser.add_argument("--urdf_path", type=str, default="urdf/panda_mujoco_kinematics.urdf")
    parser.add_argument("--xml_path", type=str, default="envs/panda_pick_place.xml")

    # Stage 2: BC Pre-training
    parser.add_argument("--bc_epochs", type=int, default=50, help="Number of epochs for BC pre-training.")
    parser.add_argument("--bc_batch_size", type=int, default=128)
    parser.add_argument("--bc_lr", type=float, default=1e-4)

    # Stage 3: RL Fine-tuning
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--n_envs", type=int, default=8, help="Number of parallel environments for RL.")
    parser.add_argument("--n_steps", type=int, default=2048, help="PPO rollout buffer size per environment.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--rl_lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda_bc", type=float, default=0.1, help="Initial weight for the BC auxiliary loss.")
    parser.add_argument("--lambda_annealing_steps", type=int, default=1_000_000)
    
    args = parser.parse_args()
    
    run_dir = Path(args.output_dir) / (args.run_name or f"run_{int(time.time())}")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    with (run_dir / "config.json").open("w") as f: json.dump(vars(args), f, indent=4)
    logger.info(f"🚀 Starting Hybrid Diffusion-PPO pipeline: {run_dir.name}")

    set_random_seed(args.seed)
    device = torch.device(resolve_device(args.device))

    # --- Execute Pipeline ---
    demo_path = run_dir / "expert_demos.pkl"
    if not demo_path.exists() or args.force_collect:
        collect_expert_demos(args, demo_path)
    else:
        logger.info(f"--- STAGE 1: Skipping demo collection, file exists at {demo_path} ---")

    policy_path = run_dir / "pretrained_policy.pth"
    if not policy_path.exists() or args.force_pretrain:
        pretrain_policy(args, demo_path, policy_path, device)
    else:
        logger.info(f"--- STAGE 2: Skipping BC pre-training, file exists at {policy_path} ---")

    run_rl_finetuning(args, demo_path, policy_path, run_dir, device)

# In scripts/train_hybrid_diffusion.py
# REPLACE the entire if __name__ == "__main__" block

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hybrid Diffusion-PPO Training Pipeline")
    
    # General
    parser.add_argument("--run_name", type=str, default=None, help="Name for the training run.")
    parser.add_argument("--output_dir", type=str, default="trained_models/diffusion_hybrid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--force_collect", action="store_true", help="Force re-collection of expert demos.")
    parser.add_argument("--force_pretrain", action="store_true", help="Force BC pre-training even if weights exist.")

    # Stage 1: Demo Collection
    parser.add_argument("--num_expert_demos", type=int, default=200)
    parser.add_argument("--urdf_path", type=str, default="urdf/panda_mujoco_kinematics.urdf")
    parser.add_argument("--xml_path", type=str, default="envs/panda_pick_place.xml")

    # Stage 2: BC Pre-training
    parser.add_argument("--bc_epochs", type=int, default=50)
    parser.add_argument("--bc_batch_size", type=int, default=128)
    parser.add_argument("--bc_lr", type=float, default=1e-4)

    # Stage 3: RL Fine-tuning
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--rl_lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda_bc", type=float, default=0.1)
    parser.add_argument("--lambda_annealing_steps", type=int, default=1_000_000)
    
    args = parser.parse_args()
    
    # The run_pipeline function is now removed, its logic is directly in main.
    run_dir = Path(args.output_dir) / (args.run_name or f"run_{int(time.time())}")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    with (run_dir / "config.json").open("w") as f: json.dump(vars(args), f, indent=4)
    logger.info(f"🚀 Starting Hybrid Diffusion-PPO pipeline: {run_dir.name}")

    set_random_seed(args.seed)
    # The device will be resolved inside pretrain and rl functions
    
    # --- Execute Pipeline ---
    demo_path = run_dir / "expert_demos.pkl"
    if not demo_path.exists() or args.force_collect:
        collect_expert_demos(args, demo_path)
    else:
        logger.info(f"--- STAGE 1: Skipping demo collection, file exists at {demo_path} ---")

    policy_path = run_dir / "pretrained_policy.pth"
    if not policy_path.exists() or args.force_pretrain:
        device = torch.device(resolve_device(args.device))
        pretrain_policy(args, demo_path, policy_path, device)
    else:
        logger.info(f"--- STAGE 2: Skipping BC pre-training, file exists at {policy_path} ---")
    
    device = torch.device(resolve_device(args.device))
    run_rl_finetuning(args, demo_path, policy_path, run_dir, device)