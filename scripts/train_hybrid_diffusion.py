# scripts/train_hybrid_diffusion.py
"""
Final, SOTA-aligned hybrid training script for DGPO-Foundation.

This script implements the "Hybrid Differentiable Diffusion-PPO" methodology with a
dual online/offline data stream for the auxiliary BC loss, ensuring utmost quality and robustness.

It orchestrates the full end-to-end pipeline:
1.  **Expert Demonstration Collection:** Generates and saves a high-quality offline dataset.
2.  **BC Pre-training:** Trains a DiffusionPolicy via pure behavioral cloning on the offline dataset.
3.  **RL Fine-tuning:** Fine-tunes the policy using a custom PPO agent (DiffusionPPO) that
    regularizes using both the offline dataset and a stream of fresh, on-the-fly expert data.
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
import sys
import time
from typing import Iterator, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- Project Imports ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from envs.panda_env import PandaEnv
from models.custom_sb3_extractor import BCFeaturesExtractor
from models.diffusion_policy import ConditionalDenoiser, DiffusionPolicy
from utils.expert_dataset import ExpertDataset
from utils.rl_reward_wrapper import RLRewardWrapper, RewardConfig

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# Configure logging
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger("dgpo.train_hybrid_diffusion")

class DiffusionActorCriticPolicy(ActorCriticPolicy):
    """
    Custom SB3 policy using DiffusionPolicy as the actor.
    """
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        # Separate diffusion-specific kwargs from standard ones
        self.diffusion_policy_kwargs = kwargs.pop('diffusion_policy_kwargs')
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        self._build(lr_schedule)

    def _build(self, lr_schedule):
        # Build the standard SB3 components (feature extractor, value net)
        super()._build(lr_schedule)
        
        # Now create our custom actor
        self.action_net = DiffusionPolicy(**self.diffusion_policy_kwargs)
        self.actor = self.action_net # SB3 convention

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the actor-critic.
        """
        latent_pi, latent_vf, _ = self._get_latent(obs)
        values = self.value_net(latent_vf)
        
        # Get actions from our diffusion actor (differentiable one-step)
        actions = self.actor(obs)
        
        # PPO requires a distribution. We create a pseudo-distribution around the action.
        # The log_prob is necessary for the PPO loss calculation.
        action_log_std = self.log_std.expand_as(actions)
        action_dist = torch.distributions.Normal(actions, torch.exp(action_log_std))
        log_prob = action_dist.log_prob(actions).sum(axis=1)
        
        return actions, values, log_prob

    def compute_loss(self, obs: Dict[str, torch.Tensor], clean_action: torch.Tensor) -> torch.Tensor:
        """Proxy method to call the diffusion actor's BC loss."""
        return self.actor.compute_loss(obs, clean_action)

# --- Custom DiffusionPPO Agent ---

class DiffusionPPO(PPO):
    """
    PPO agent that fine-tunes a DiffusionPolicy with a dual-stream auxiliary BC loss.
    """
    def __init__(
        self,
        policy,
        env,
        offline_expert_loader: DataLoader,
        online_expert_loader: DataLoader,
        lambda_bc: float = 1.0,
        lambda_annealing_steps: int = 1_000_000,
        **kwargs,
    ):
        super().__init__(policy=policy, env=env, **kwargs)
        
        self.offline_expert_loader = offline_expert_loader
        self.online_expert_loader = online_expert_loader
        self._offline_iter: Iterator = iter(self.offline_expert_loader)
        self._online_iter: Iterator = iter(self.online_expert_loader)
        
        self.initial_lambda_bc = lambda_bc
        self.current_lambda_bc = lambda_bc
        self.lambda_annealing_steps = lambda_annealing_steps

    def _get_expert_batch(self, loader_type: str):
        """Safely gets the next batch from the specified expert dataloader."""
        if loader_type == 'offline':
            try:
                return next(self._offline_iter)
            except StopIteration:
                self._offline_iter = iter(self.offline_expert_loader)
                return next(self._offline_iter)
        else: # 'online'
            return next(self._online_iter)

    def train(self) -> None:
        """
        Override the SB3 `train` method to incorporate the auxiliary BC loss.
        """
        # Update schedules
        self._update_learning_rate(self.policy.optimizer)
        progress = self.num_timesteps / self.lambda_annealing_steps
        self.current_lambda_bc = self.initial_lambda_bc * max(0.0, 1.0 - progress)
        
        # The original SB3 train() method handles the PPO update. We call it first.
        super().train()
        
        # --- Auxiliary BC Loss Update Step ---
        # We perform a separate optimization step for the BC loss. This is more stable
        # than trying to combine the gradients in a single step.
        for _ in range(self.n_epochs):
            # Get expert data from both streams
            offline_obs, offline_actions = self._get_expert_batch('offline')
            online_obs, online_actions = self._get_expert_batch('online')

            # Move data to device
            offline_obs = {k: v.to(self.device) for k, v in offline_obs.items()}
            offline_actions = offline_actions.to(self.device)
            online_obs = {k: v.to(self.device) for k, v in online_obs.items()}
            online_actions = online_actions.to(self.device)
            
            # Compute BC losses
            bc_loss_offline = self.policy.compute_loss(offline_obs, offline_actions)
            bc_loss_online = self.policy.compute_loss(online_obs, online_actions)
            
            bc_loss = 0.5 * (bc_loss_offline + bc_loss_online)
            total_aux_loss = self.current_lambda_bc * bc_loss
            
            # Optimization step for the auxiliary loss
            self.policy.optimizer.zero_grad()
            total_aux_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Logging
        self.logger.record("train/bc_loss", bc_loss.item())
        self.logger.record("train/lambda_bc", self.current_lambda_bc)

# --- Data Handling ---

class ExpertTrajectoryDataset(Dataset):
    """Simple map-style dataset for saved expert trajectories."""
    def __init__(self, trajectory_path: Path):
        with open(trajectory_path, 'rb') as f:
            trajectories = pickle.load(f)
        
        self.observations = []
        self.actions = []
        for traj in trajectories:
            for obs, act in zip(traj['observations'], traj['actions']):
                # Only store the keys the policy needs
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
    # Handle image data specifically if it's not CHW
    if observations['image_primary'].dim() == 4 and observations['image_primary'].shape[3] == 3:
        observations['image_primary'] = observations['image_primary'].permute(0, 3, 1, 2)
    return observations, actions


# --- Main Pipeline Orchestrator ---

def run_pipeline(args: argparse.Namespace):
    run_name = args.run_name or f"diffusion_ppo_{int(time.time())}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    with (run_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=4)
    logger.info(f"🚀 Starting Hybrid Diffusion-PPO pipeline: {run_name}")

    set_random_seed(args.seed)
    device = torch.device(resolve_device(args.device))

    # --- STAGE 1: Collect & Save Expert Demonstrations ---
    demo_path = run_dir / "expert_demos.pkl"
    if not demo_path.exists():
        logger.info("--- STAGE 1: Collecting Expert Demonstrations ---")
        expert_env = PandaEnv(xml_path=args.xml_path)
        expert_dataset = ExpertDataset(urdf_path=args.urdf_path, use_octo=False, yield_full_obs=True)
        
        trajectories = []
        pbar = tqdm(total=args.num_expert_demos, desc="Collecting Demos")
        # Since ExpertDataset is an IterableDataset, we can't index it. We must iterate.
        data_iter = iter(expert_dataset)
        while len(trajectories) < args.num_expert_demos:
            trajectory = {'observations': [], 'actions': []}
            successful = False
            for _ in range(expert_env.max_episode_steps * 2): # Iterate enough for one full trajectory
                try:
                    obs, action = next(data_iter)
                    trajectory['observations'].append(obs)
                    trajectory['actions'].append(action)
                    if expert_dataset._scripted_expert.is_done():
                        if expert_dataset._scripted_expert.was_successful():
                            successful = True
                        break
                except StopIteration:
                    break
            if successful:
                trajectories.append(trajectory)
                pbar.update(1)
        expert_env.close()
        
        with open(demo_path, 'wb') as f:
            pickle.dump(trajectories, f)
        logger.info(f"✅ Saved {len(trajectories)} expert demos to {demo_path}")
    else:
        logger.info(f"--- STAGE 1: Found existing expert demos at {demo_path} ---")

    # --- STAGE 2: BC Pre-training ---
    policy_path = run_dir / "pretrained_policy.pth"
    if not policy_path.exists():
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
    else:
        logger.info(f"--- STAGE 2: Found existing pre-trained policy at {policy_path} ---")

    # --- STAGE 3: RL Fine-tuning ---
    logger.info("--- STAGE 3: RL Fine-tuning with DiffusionPPO ---")

    def make_env():
        env = PandaEnv(xml_path=args.xml_path, control_mode='delta')
        env = RLRewardWrapper(env)
        return env
    
    vec_env = make_vec_env(make_env, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)
    vec_env = VecNormalize(vec_env, norm_obs=False, gamma=0.99)

    offline_expert_dataset = ExpertTrajectoryDataset(demo_path)
    offline_loader = DataLoader(offline_expert_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn, drop_last=True, persistent_workers=True)
    online_expert_dataset = ExpertDataset(urdf_path=args.urdf_path, use_octo=False)
    online_loader = DataLoader(online_expert_dataset, batch_size=args.batch_size, num_workers=2, collate_fn=collate_fn, persistent_workers=True)

    obs_encoder = BCFeaturesExtractor(vec_env.observation_space)
    obs_feature_dim = obs_encoder.features_dim
    policy_kwargs = {
        'features_extractor_class': BCFeaturesExtractor,
        'features_extractor_kwargs': dict(features_dim=obs_feature_dim),
        'net_arch': {'pi': [], 'vf': [256, 256]},
        'diffusion_policy_kwargs': {
            'obs_feature_extractor': obs_encoder,
            'denoiser': ConditionalDenoiser(action_dim=vec_env.action_space.shape[0], obs_feature_dim=obs_feature_dim),
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
        gamma=0.99,
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
        CheckpointCallback(save_freq=max(1, 50000 // args.n_envs), save_path=str(checkpoints_dir)),
        SingleFileBackupCallback(save_freq=max(1, 5000 // args.n_envs), save_path=str(run_dir / "backups"))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hybrid Diffusion-PPO Training")
    # Run Management
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="trained_models/diffusion_hybrid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    # Stage 1: Demo Collection
    parser.add_argument("--num_expert_demos", type=int, default=500)
    parser.add_argument("--urdf_path", type=str, default="urdf/panda_mujoco_kinematics.urdf")

    # Stage 2: BC Pre-training
    parser.add_argument("--bc_epochs", type=int, default=100)
    parser.add_argument("--bc_batch_size", type=int, default=128)
    parser.add_argument("--bc_lr", type=float, default=1e-4)

    # Stage 3: RL Fine-tuning
    parser.add_argument("--total_timesteps", type=int, default=3_000_000)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--rl_lr", type=float, default=3e-4)
    parser.add_argument("--lambda_bc", type=float, default=0.5)
    parser.add_argument("--lambda_annealing_steps", type=int, default=1_500_000)
    
    parser.add_argument("--xml_path", type=str, default="envs/panda_pick_place.xml")
    
    args = parser.parse_args()
    run_pipeline(args)