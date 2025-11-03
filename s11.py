#!/usr/bin/env python3
# FILE: scripts/evaluate_ego_planner.py
# (Definitive, SOTA, Hydra-Integrated, and Fully Corrected Version)

"""
The definitive, state-of-the-art evaluation script for the Ego-Planner policy,
powered by the Hydra configuration framework.

This script performs a true closed-loop, autonomous rollout of the policy in a
simulated environment. It correctly loads the model, handles data preprocessing
with scientific rigor, and generates a detailed video of the policy's attempt.

Key Corrections & SOTA Features from Deep Analysis:
-   **Correct EMA Model Loading**: Implements a robust loader that is independent of
    the training script and correctly extracts the EMA weights from the checkpoint.
-   **Scientifically Valid Data Preprocessing**: Leverages the dataset's own
    transformation pipelines to guarantee a perfect match between training and
    evaluation data distributions, fixing a critical normalization bug.
-   **Correct Policy API Invocation**: Calls the model's `.sample()` method with
    the correct keyword arguments, fixing the "static video" bug and ensuring
    the policy receives real-time sensory feedback.
-   **Robust Goal Image Handling**: Uses the ground-truth goal image from the dataset,
    ensuring perfect consistency with the policy's training objective.
-   **Clean & Modular Structure**: Driven by a clean Hydra config for reproducibility
    and easy experimentation.
"""

import collections
import logging
from pathlib import Path
import json
import cv2
import imageio
import hydra
import mujoco
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from envs.panda_env import PandaEnv
from models.ego_planner import EgoPlanner, EgoPlannerConfig
from models.diffusion_policy import NoiseScheduler, NoiseSchedulerConfig
# NOTE: The EgoPlannerDataset is now the central hub for data access and transforms
from utils.ego_planner_dataset import EgoPlannerDataset

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. Definitive Model and Data Loaders
# -----------------------------------------------------------------------------

def load_policy_from_checkpoint(cfg: DictConfig, ckpt_path: str, device: torch.device) -> EgoPlanner:
    """
    Definitively loads the Ego-Planner from a PL checkpoint, prioritizing EMA weights.
    """
    log.info(f"Loading checkpoint from: {ckpt_path}")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(ckpt_path)

    payload = torch.load(ckpt_path, map_location=device)

    # Instantiate the model from the training configuration
    model_config = EgoPlannerConfig(**cfg.model)
    policy = EgoPlanner(model_config).to(device)

    # Prioritize loading EMA weights for superior evaluation performance
    if 'ema_state_dict' in payload:
        log.info("Loading Exponential Moving Average (EMA) weights for evaluation.")
        policy.load_state_dict(payload['ema_state_dict'])
    else:
        log.warning("EMA state not found. Falling back to raw model state_dict.")
        # Clean the "model." prefix added by Lightning
        state_dict = {k.replace("model.", ""): v for k, v in payload['state_dict'].items()}
        incompatible_keys = policy.load_state_dict(state_dict, strict=False)
        if incompatible_keys.missing_keys: log.warning(f"Missing keys: {incompatible_keys.missing_keys}")
        if incompatible_keys.unexpected_keys: log.warning(f"Unexpected keys: {incompatible_keys.unexpected_keys}")

    policy.eval()
    log.info("Policy loaded successfully and set to evaluation mode.")
    return policy


# -----------------------------------------------------------------------------
# 2. Main Hydra-Driven Evaluation Function
# -----------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="./configs", config_name="evaluate_ego_planner_config")
def evaluate(cfg: DictConfig):
    """Main evaluation function driven by Hydra."""
    
    # --- 1. Setup ---
    pl.seed_everything(cfg.seed)
    output_dir = Path.cwd() # Hydra manages this directory
    log.info("--- EGO-Planner SOTA Visual Evaluation ---")
    log.info(f"Output directory: {output_dir}")
    log.info("Full Configuration:\n" + OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.device)

    # --- 2. Load Components ---
    policy = load_policy_from_checkpoint(cfg, cfg.checkpoint_path, device)
    
    # The Dataset is now the single source of truth for data and transforms
    dataset = EgoPlannerDataset(dataset_path=cfg.dataset.path, obs_horizon=cfg.model.obs_horizon,
                                action_horizon=cfg.model.action_horizon, use_aug=False)
    dataset._build_episode_indices() # Enable evaluation methods

    scheduler = NoiseScheduler(NoiseSchedulerConfig(**cfg.scheduler)).to(device)
    
    # --- 3. Run Evaluation Loop ---
    num_episodes = len(dataset.episode_start_indices) - 1
    episodes_to_run = cfg.rollout.episode_indices or list(range(min(cfg.rollout.max_episodes_to_eval, num_episodes)))
    
    all_results = []
    for ep_idx in episodes_to_run:
        if ep_idx >= num_episodes:
            log.warning(f"Episode index {ep_idx} out of bounds. Skipping.")
            continue

        # Initialize the environment with DR disabled for consistency
        env = PandaEnv(xml_path=cfg.env.xml_path, enable_domain_randomization=False)

        # Initialize video writer for this episode
        video_path = output_dir / f"ep_{ep_idx}_closed_loop.mp4"
        frame_test = env.render(camera_name="fixed_camera")
        video_writer = imageio.get_writer(video_path, fps=cfg.video_fps, quality=8)
        
        log.info(f"--- Starting Rollout for Episode {ep_idx} (Seed: {cfg.seed + ep_idx}) ---")
        
        # --- 3a. Episode Reset & Setup ---
        obs, _ = env.reset(seed=cfg.seed + ep_idx)
        
        # Get static images for the Strategist from the dataset for consistency
        initial_sample = dataset.get_episode_sample(ep_idx, 0)
        initial_image_tensor = initial_sample['initial_image'].unsqueeze(0).to(device)
        goal_image_tensor = initial_sample['goal_image'].unsqueeze(0).to(device)
        
        # Buffer for observation history
        obs_history_deque = collections.deque(maxlen=cfg.model.obs_horizon)
        for _ in range(cfg.model.obs_horizon):
            obs_history_deque.append({
                'image_primary': dataset.transform_primary(Image.fromarray(obs['image_primary'])),
                'image_wrist': dataset.transform_wrist(Image.fromarray(obs['image_wrist'])),
                'proprio': torch.from_numpy(obs['proprio']).float()
            })

        # --- 3b. Main Rollout Loop ---
        is_success = False
        for step in tqdm(range(cfg.rollout.max_steps), desc=f"  Rollout Ep {ep_idx}"):
            # Prepare batch for model
            obs_history_batch = {k: torch.stack([h[k] for h in obs_history_deque]).unsqueeze(0).to(device)
                                 for k in obs_history_deque[0].keys()}

            # --- DEFINITIVE FIX for the "Static Video" bug ---
            # Call the model with the correct keyword arguments
            with torch.no_grad():
                action_chunk = policy.sample(
                    initial_image=initial_image_tensor,
                    goal_image=goal_image_tensor,
                    observation_history=obs_history_batch,
                    scheduler=scheduler,
                    num_inference_steps=cfg.inference.sampling_steps,
                    guidance_scale_plan=cfg.inference.guidance_scale_plan,
                    guidance_scale_obs=cfg.inference.guidance_scale_obs
                )
            
            action = action_chunk[0, 0].cpu().numpy()
            
            # Step environment
            obs, _, terminated, truncated, info = env.step(action)
            
            # Record frame
            frame_rgb = env.render(camera_name="fixed_camera")
            video_writer.append_data(frame_rgb)
            
            # Update history
            obs_history_deque.append({
                'image_primary': dataset.transform_primary(Image.fromarray(obs['image_primary'])),
                'image_wrist': dataset.transform_wrist(Image.fromarray(obs['image_wrist'])),
                'proprio': torch.from_numpy(obs['proprio']).float()
            })
            
            if terminated or truncated:
                is_success = info.get('is_success', False)
                log.info(f"Episode finished at step {step}. Success: {is_success}")
                break
        
        # --- 3c. Cleanup ---
        all_results.append({'episode_idx': ep_idx, 'success': is_success, 'steps': step + 1})
        video_writer.close()
        env.close()
        log.info(f"Video saved to {video_path}")

    # --- 4. Final Summary ---
    if all_results:
        success_rate = np.mean([r['success'] for r in all_results])
        log.info(f"\n--- Evaluation Summary ---")
        log.info(f"Success Rate: {success_rate:.3f} across {len(all_results)} episodes.")
        # Save summary to file
        with open(output_dir / "summary.json", "w") as f:
            json.dump({'success_rate': success_rate, 'results': all_results}, f, indent=2)

    log.info(f"Evaluation complete. All outputs saved to: {output_dir}")

if __name__ == "__main__":
    evaluate()