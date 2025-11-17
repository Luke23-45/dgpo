# FILE: evaluate/evaluate_finetuned_ego_planner.py
# (State-of-the-Art, Definitive V3 Synthesis for RL Evaluation)

"""
The definitive, state-of-the-art visual evaluation script for the RL-finetuned
Ego-Planner model.

This script is designed for "Deterministic Expert Evaluation": its sole purpose
is to showcase the best possible performance of the trained agent by eliminating
all sources of randomness from both the policy and the environment.

Key SOTA Features of this Definitive Version:
  - **Custom Environment Wrapper (`EvaluationEnvWrapper`)**: The cornerstone of
    this script. It acts as a critical "adapter" that solves the architectural
    mismatch between a standard environment's observation and the Ego-Planner's
    multi-part input requirements. It correctly injects the static `initial_image`
    and `goal_image` at every timestep.
  - **Correct Observation History Management**: The wrapper correctly implements
    the "warm-up" procedure, providing the policy with a temporally consistent
    initial history, which is critical for preventing first-step failures.
  - **Robust SB3 Model Loading**: Uses the correct `PPO.load()` methodology for
    models with custom policies. It first loads the pre-trained Ego-Planner
    "body" and then passes it to the PPO loader, which correctly reconstructs the
    full Actor-Critic architecture.
  - **Deterministic Inference**: Strictly uses `model.predict(..., deterministic=True)`
    to ensure the policy's actions are repeatable and represent its learned mean
    behavior without stochastic exploration.
  - **Full Hydra Integration**: Allows for clean, command-line configuration of
    the checkpoint path, video output, and other evaluation parameters.
"""

from __future__ import annotations

import logging
import collections
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
import mujoco
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from typing import Dict, Any, Tuple

import gymnasium as gym
from stable_baselines3 import PPO

# --- Project-Specific Imports ---
# This script assumes it is run from a location where these modules are importable.
from envs.panda_env import PandaEnv
from models.ego_planner import EgoPlanner, NoiseScheduler
from train.train_ego_planner import EgoPlannerLightningModule
# CRITICAL: We import our custom policy and feature extractor from the training script
from rl.finetune_ego_planner_rl import EgoPlannerActorCriticPolicy, EgoPlannerAsFeaturesExtractor

# Setup a logger for this module
log = logging.getLogger(__name__)


# ==============================================================================
# PHASE 1: THE CRITICAL ENVIRONMENT ADAPTER
# ==============================================================================

def get_goal_image(env: PandaEnv, obs: dict) -> np.ndarray:
    """
    SOTA Helper: Creates a 'goal_image' by saving the current state, teleporting
    the object to the goal, rendering, and then perfectly restoring the state.
    """
    original_mj_state = env.get_mj_state()
    try:
        goal_pos_world = obs['goal_pos_world']
        qpos_addr = env.model.jnt_qposadr[env.object_joint_id]
        env.data.qpos[qpos_addr:qpos_addr + 3] = goal_pos_world
        mujoco.mj_forward(env.model, env.data)
        goal_img_np = env.render(camera_name="fixed_camera")
    finally:
        env.set_mj_state(original_mj_state)
    return goal_img_np


class EvaluationEnvWrapper(gym.Wrapper):
    """
    The essential "glue" that adapts the PandaEnv's output to the Ego-Planner's
    complex, multi-part observation requirements for evaluation.
    """
    def __init__(self, env: PandaEnv, obs_horizon: int):
        super().__init__(env)
        self.env: PandaEnv
        self.obs_horizon = obs_horizon

        # --- Define the new, augmented observation space ---
        original_obs_space = self.env.observation_space
        img_space = original_obs_space['image_primary']

        # The new space includes the static images and a history of observations
        self.observation_space = gym.spaces.Dict({
            'initial_image': img_space,
            'goal_image': img_space,
            'observation_history.image_primary': gym.spaces.Box(0, 255, (obs_horizon,) + img_space.shape, np.uint8),
            'observation_history.image_wrist': gym.spaces.Box(0, 255, (obs_horizon,) + original_obs_space['image_wrist'].shape, np.uint8),
            'observation_history.proprio': gym.spaces.Box(-np.inf, np.inf, (obs_horizon,) + original_obs_space['proprio'].shape, np.float32),
        })

        # Internal state for the wrapper
        self._initial_image_np: np.ndarray | None = None
        self._goal_image_np: np.ndarray | None = None
        self._obs_history: collections.deque = collections.deque(maxlen=self.obs_horizon)

    def _get_stacked_history(self) -> Dict[str, np.ndarray]:
        """Stacks the deque of observation dicts into three NumPy arrays."""
        return {
            'image_primary': np.stack([obs['image_primary'] for obs in self._obs_history]),
            'image_wrist': np.stack([obs['image_wrist'] for obs in self._obs_history]),
            'proprio': np.stack([obs['proprio'] for obs in self._obs_history]),
        }

    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        
        # 1. Capture the static, episode-level images
        self._initial_image_np = obs['image_primary'].copy()
        self._goal_image_np = get_goal_image(self.env, obs)
        
        # 2. Perform the critical "warm-up" for the observation history
        self._obs_history.clear()
        for _ in range(self.obs_horizon):
            obs, _, _, _, _ = self.env.step(np.zeros(self.env.action_space.shape))
            self._obs_history.append(obs)
            
        # 3. Construct the first, augmented observation
        augmented_obs = {
            'initial_image': self._initial_image_np,
            'goal_image': self._goal_image_np,
            **{f'observation_history.{k}': v for k, v in self._get_stacked_history().items()}
        }
        
        return augmented_obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        self._obs_history.append(next_obs)
        
        # Inject the static images into the observation for the next step
        augmented_obs = {
            'initial_image': self._initial_image_np,
            'goal_image': self._goal_image_np,
            **{f'observation_history.{k}': v for k, v in self._get_stacked_history().items()}
        }
        
        return augmented_obs, reward, terminated, truncated, info


# ==============================================================================
# PHASE 2: THE MAIN EVALUATION ORCHESTRATOR SCRIPT
# ==============================================================================

@hydra.main(version_base=None, config_path="../configs", config_name="evaluate_finetuned_config.yaml")
def main(cfg: DictConfig):
    log.info("--- Definitive RL Fine-Tuning Evaluation Script (SOTA V3) ---")
    log.info(f"Full evaluation config:\n{OmegaConf.to_yaml(cfg)}")

    # --- 1. Setup ---
    pl.seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # --- 2. Load Pre-trained Components for Inference ---
    # We load the IL checkpoint to get the EgoPlanner "body" and its config
    pretrained_model, noise_scheduler, train_cfg = load_pretrained_model(
        cfg.il_checkpoint_path, device
    )
    
    # --- 3. Instantiate and Wrap the Environment ---
    log.info("Initializing and wrapping the evaluation environment...")
    env = PandaEnv(
        xml_path=train_cfg.env.xml_path,
        control_mode="delta",
        enable_domain_randomization=False # Ensure deterministic visuals
    )
    # The wrapper is the key to making the environment compatible with the policy
    eval_env = EvaluationEnvWrapper(env, obs_horizon=train_cfg.model.obs_horizon)

    # --- 4. Load the Final, Fine-Tuned PPO Model ---
    # This is the correct, SOTA way to load an SB3 model with a custom policy.
    log.info(f"Loading fine-tuned RL model from: {cfg.rl_model_path}")
    policy_kwargs = {
        'ego_planner_model': pretrained_model,
        'noise_scheduler': noise_scheduler,
        'inference_cfg': train_cfg.validation,
    }
    ppo_model = PPO.load(
        cfg.rl_model_path,
        env=eval_env,
        custom_objects={'policy': EgoPlannerActorCriticPolicy},
        policy_kwargs=policy_kwargs,
        device=device
    )

    # --- 5. Setup Video Recording ---
    video_path = output_dir / cfg.output_video
    H, W, _ = env.observation_space['image_primary'].shape
    video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (W, H))
    log.info(f"Recording evaluation video to: {video_path}")

    # --- 6. Run the Deterministic Evaluation Loop ---
    for ep_idx in tqdm(range(cfg.num_episodes), desc="Running Evaluation"):
        # Reset the wrapped environment to get the first augmented observation
        obs, info = eval_env.reset(seed=cfg.seed + ep_idx)
        
        for step in tqdm(range(env.max_episode_steps), desc=f"Episode {ep_idx+1}", leave=False):
            # Get action from the loaded PPO model
            action, _ = ppo_model.predict(obs, deterministic=True)
            
            # Step the wrapped environment
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            # Render the *unwrapped* environment to get a clean image for the video
            frame_rgb = env.render()
            video_writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            
            if terminated or truncated:
                log.info(f"Episode finished after {step + 1} steps. Success: {info.get('is_success', False)}")
                break

    # --- 7. Cleanup ---
    video_writer.release()
    eval_env.close()
    log.info("--- Evaluation Complete. ---")

if __name__ == "__main__":
    main()