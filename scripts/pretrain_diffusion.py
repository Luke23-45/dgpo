# FILE: scripts/pretrain_diffusion.py
"""
Dedicated, high-quality script for Phase 2: BC Pre-training.

This script takes a dataset of expert demonstrations and performs Behavioral Cloning
to pre-train the DiffusionPolicy.

It incorporates SOTA best practices:
- Data-driven model initialization to prevent shape mismatches.
- Train/Validation split to monitor generalization and prevent overfitting.
- Checkpointing to save the best performing model based on validation loss.
- TensorBoard logging for real-time monitoring of training progress.
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import gymnasium as gym

# --- Project Imports ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from models.custom_sb3_extractor import BCFeaturesExtractor
from models.diffusion_policy import ConditionalDenoiser, DiffusionPolicy
from stable_baselines3.common.utils import set_random_seed

# Configure logging
logging.basicConfig(format="%(asctime)s | %(levelname)s | [%(name)s] | %(message)s", level=logging.INFO)
logger = logging.getLogger("dgpo.pretrain")

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

# --- Training Engine ---

class BCTrainer:
    def __init__(self, model, optimizer, train_loader, val_loader, scheduler, device, writer):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.device = device
        self.writer = writer
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} [Train]")
        for obs, actions in pbar:
            obs = {k: v.to(self.device) for k, v in obs.items()}
            actions = actions.to(self.device)
            
            loss = self.model.compute_loss(obs, actions)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar("Loss/train", avg_loss, epoch)
        return avg_loss

    @torch.no_grad()
    def validate_epoch(self, epoch: int, run_dir: Path):
        self.model.eval()
        total_loss = 0
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} [Val]")
        for obs, actions in pbar:
            obs = {k: v.to(self.device) for k, v in obs.items()}
            actions = actions.to(self.device)
            loss = self.model.compute_loss(obs, actions)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        self.writer.add_scalar("Loss/validation", avg_loss, epoch)
        
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            save_path = run_dir / "checkpoints" / "best_model.pth"
            torch.save(self.model.state_dict(), save_path)
            logger.info(f"✅ New best model saved with val loss: {avg_loss:.4f}")
        
        self.scheduler.step(avg_loss)
        return avg_loss

    def fit(self, args: argparse.Namespace, run_dir: Path):
        self.args = args
        for epoch in range(args.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch(epoch, run_dir)
            logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Best Val Loss={self.best_val_loss:.4f}")

# --- Main Orchestrator ---

def main(args: argparse.Namespace):
    run_name = args.run_name or f"bc_pretrain_{int(time.time())}"
    run_dir = Path(args.output_dir) / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    
    with (run_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=4)
    
    writer = SummaryWriter(log_dir=str(run_dir))
    logger.info(f"🚀 Starting BC Pre-training: {run_name}")
    logger.info(f"TensorBoard logs at: {run_dir}")

    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load data and infer model dimensions
    logger.info("Loading expert dataset and inferring model dimensions...")
    full_dataset = ExpertTrajectoryDataset(Path(args.demo_path))
    
    # Infer dimensions from the first sample
    sample_obs, sample_action = full_dataset[0]
    action_dim = len(sample_action)
    
    dummy_obs_space = gym.spaces.Dict({
        key: gym.spaces.Box(-np.inf, np.inf, shape=val.shape, dtype=val.dtype)
        for key, val in sample_obs.items()
    })
    
    logger.info(f"Inferred action_dim: {action_dim}")
    logger.info(f"Inferred observation space for feature extractor.")

    # 2. Create model
    obs_encoder = BCFeaturesExtractor(dummy_obs_space)
    obs_feature_dim = obs_encoder.features_dim
    denoiser = ConditionalDenoiser(action_dim=action_dim, obs_feature_dim=obs_feature_dim)
    policy = DiffusionPolicy(obs_encoder, denoiser, action_dim=action_dim).to(device)
    
    # 3. Create DataLoaders with Train/Val split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=4, collate_fn=collate_fn, persistent_workers=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=4, collate_fn=collate_fn, persistent_workers=True, pin_memory=True
    )
    logger.info(f"Dataset split: {train_size} training samples, {val_size} validation samples.")

    # 4. Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # 5. Launch Trainer
    trainer = BCTrainer(policy, optimizer, train_loader, val_loader, scheduler, device, writer)
    try:
        trainer.fit(args, run_dir)
        # Save final model as well
        final_path = run_dir / "checkpoints" / "final_model.pth"
        torch.save(policy.state_dict(), final_path)
        logger.info(f"✅ Final model saved to {final_path}")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
    finally:
        writer.close()
        logger.info("--- Pre-training complete. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BC Pre-training for Diffusion Policy")
    
    # Paths and Run Management
    parser.add_argument("--demo_path", type=str, required=True, help="Path to the expert_demos.pkl file.")
    parser.add_argument("--output_dir", type=str, default="trained_models/bc_pretrain")
    parser.add_argument("--run_name", type=str, default=None, help="Name for the training run.")
    parser.add_argument("--seed", type=int, default=42)

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of data to use for validation.")
    
    args = parser.parse_args()
    main(args)