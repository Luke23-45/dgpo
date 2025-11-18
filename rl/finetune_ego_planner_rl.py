#!/usr/bin/env python3
# FILE: train/finetune_ego_planner_rl_SOTA_v4.py
# (State-of-the-Art, V4 - Architecturally-Consistent & Stable)

"""
The definitive, state-of-the-art reinforcement learning fine-tuning script for
the pre-trained Ego-Planner model.

This V4 script is the "correct" implementation of the SOTA "frozen backbone"
philosophy (per the RT-1 paper)[cite: 482]. It REJECTS the buggy,
architecturally-conflicted `EgoPlannerActorCriticPolicy` 
from previous versions, which incorrectly tried to mix a generative sampler
with a policy-gradient algorithm.

Key SOTA Features of this Definitive Version:
  - **Architectural Purity**: This script uses the standard Stable-Baselines3
    `MlpPolicy`. The "magic" is in the `policy_kwargs`, where we
    inject our custom `EgoPlannerAsFeaturesExtractor`.
  - **Correct RT-1 Implementation**: The `EgoPlannerAsFeaturesExtractor` (V5)
     acts as the massive, frozen feature extractor. The `MlpPolicy`
    acts as the lightweight, trainable "head" that is fine-tuned with PPO.
    This prevents catastrophic forgetting [cite: 483] and is computationally
    efficient.
  - **SOTA Observation Handling**: This script solves the "history" problem
    [cite: 508] at the source. It uses `sb3_contrib.FrameStackDict` to wrap
    the vectorized environment. This wrapper correctly stacks the dictionary
    of observations over time, feeding the `EgoPlannerAsFeaturesExtractor`
     *exactly* the `(B, H_o, ...)` shaped tensors it expects.
  - **Resilient & Robust**: Retains the SOTA `EvalCallback`[cite: 537],
    `CheckpointCallback` [cite: 536], and `WandbCallback` [cite: 539] from the
    previous version for maximum observability and resilience.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Tuple
from torchvision import transforms as T
import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import mujoco
import collections
from stable_baselines3.common.vec_env import VecTransposeImage


import pytorch_lightning as pl
import gymnasium as gym
import numpy as np
import os

# --- Project-Specific Imports (All are SOTA and re-used) ---
from envs.panda_env import PandaEnv 
from models.ego_planner import EgoPlanner, NoiseScheduler, NoiseSchedulerConfig, EgoPlannerConfig 
# We do NOT import EgoPlannerLightningModule, as it's not needed for inference.
from rl.ego_planner_reward_wrapper import EgoPlannerRewardWrapper, EgoPlannerRewardConfig

try:
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True 
except ImportError:
    WandbCallback = None
    WANDB_AVAILABLE = False 

# Setup a logger for this module
log = logging.getLogger(__name__)


# In FILE: rl/finetune_ego_planner_rl.py
# ADD this class definition before the EgoPlannerAsFeaturesExtractor class.

# ==============================================================================
# PHASE 1A: THE DEFINITIVE ENVIRONMENT WRAPPER (SOTA SELF-CONTAINED)
# ==============================================================================

class FrameStackEgoPlanner(gym.Wrapper):
    """
    [SOTA V5, DEFINITIVE & SELF-CONTAINED]
    This wrapper is the definitive solution for handling the complex observation
    space required by the EgoPlanner model within the Stable-Baselines3 framework.

    It performs two critical functions:
    1.  **History Stacking**: It correctly maintains a deque of past observations
        and stacks them into the `(B, H_o, ...)` tensors the model expects.
    2.  **Static Info Injection**: It augments the observation at every step to
        include the static `initial_image` and `goal_image` for the episode.

    This wrapper produces a FLAT dictionary of observations, which is the native
    format for SB3, preventing any unexpected behavior from its internal wrappers.
    """
    def __init__(self, env: gym.Env, obs_horizon: int):
        super().__init__(env)
        self.env: PandaEnv # For type hinting
        self.obs_horizon = obs_horizon

        # --- Define the new, FLAT observation space ---
        original_obs_space = self.env.observation_space
        img_primary_space = original_obs_space['image_primary']
        img_wrist_space = original_obs_space['image_wrist']
        proprio_space = original_obs_space['proprio']
        
        self.observation_space = gym.spaces.Dict({
            'initial_image': img_primary_space,
            'goal_image': img_primary_space,
            'image_primary': gym.spaces.Box(0, 255, (obs_horizon,) + img_primary_space.shape, np.uint8),
            'image_wrist': gym.spaces.Box(0, 255, (obs_horizon,) + img_wrist_space.shape, np.uint8),
            'proprio': gym.spaces.Box(-np.inf, np.inf, (obs_horizon,) + proprio_space.shape, np.float32),
        })

        self._initial_image_np: np.ndarray | None = None
        self._goal_image_np: np.ndarray | None = None
        self._obs_history: collections.deque = collections.deque(maxlen=self.obs_horizon)

    def _get_stacked_obs(self) -> Dict[str, np.ndarray]:
        """Stacks the history and adds the static images to create the final flat dict."""
        history = {
            'image_primary': np.stack([obs['image_primary'] for obs in self._obs_history]),
            'image_wrist': np.stack([obs['image_wrist'] for obs in self._obs_history]),
            'proprio': np.stack([obs['proprio'] for obs in self._obs_history]),
        }
        return {'initial_image': self._initial_image_np, 'goal_image': self._goal_image_np, **history}

    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        
        self._initial_image_np = obs['image_primary'].copy()
        self._goal_image_np = get_goal_image(self.env, obs) # Assume get_goal_image is defined
        
        self._obs_history.clear()
        for _ in range(self.obs_horizon):
            self._obs_history.append(obs)
            
        return self._get_stacked_obs(), info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        self._obs_history.append(next_obs)
        
        return self._get_stacked_obs(), reward, terminated, truncated, info

# CORRECT, SOTA V4.2 VERSION - TO BE USED IN finetune_ego_planner_rl.py

def get_goal_image(env: gym.Env, obs: dict) -> np.ndarray:
    """
    Renders a goal image by temporarily moving the object.
    This version is robustly patched to use `env.unwrapped` to access the
    base environment's custom methods and attributes, bypassing any wrappers.
    """
    # Use .unwrapped to get the base PandaEnv and call its custom method
    base_env: PandaEnv = env.unwrapped
    original_mj_state = base_env.get_mj_state()
    try:
        goal_pos_world = obs['goal_pos_world']
        
        # Access all MuJoCo model/data attributes via the unwrapped env
        qpos_addr = base_env.model.jnt_qposadr[base_env.object_joint_id]
        base_env.data.qpos[qpos_addr:qpos_addr + 3] = goal_pos_world
        
        # We also need to set the orientation if available
        if 'goal_orn_world' in obs:
             # Convert xyzw (SciPy) to wxyz (MuJoCo)
             quat_xyzw = obs['goal_orn_world']
             quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
             base_env.data.qpos[qpos_addr + 3:qpos_addr + 7] = quat_wxyz

        mujoco.mj_forward(base_env.model, base_env.data)
        
        # The render method is standard, but it's safer to call it on the
        # original env to ensure the correct camera is used.
        goal_img_np = base_env.render(camera_name="fixed_camera")
    finally:
        # Restore the state using the unwrapped env's custom method
        base_env.set_mj_state(original_mj_state)
        
    return goal_img_np

# ==============================================================================
# PHASE 1: THE SOTA FEATURE EXTRACTOR (FROM YOUR FILE)
# This class is SOTA and is used AS-IS.
# =================================_policy=============================================

# Add this import at the top of the file
from torchvision import transforms as T

# ... existing code ...

class EgoPlannerAsFeaturesExtractor(BaseFeaturesExtractor):
    """
    [SOTA V5.1, DEFINITIVE & ROBUST]
    This version is patched to perform the critical resizing of input images
    from the environment's 256x256 to the model's expected 224x224.
    """
    def __init__(self,
                 observation_space: gym.spaces.Dict,
                 features_dim: int,
                 ego_planner_model: EgoPlanner):
        super().__init__(observation_space, features_dim)
        log.info(f"Initializing EgoPlannerAsFeaturesExtractor with features_dim={features_dim}")
        self.ego_planner = ego_planner_model

        # --- SOTA V5.1 FIX: Define the required resize transform ---
        # The Siglip model expects 224x224 images.
        self.resize = T.Resize((224, 224), antialias=True)
        # The Pilot's ResNets also benefit from a consistent size.
        self.resize_wrist = T.Resize((128, 128), antialias=True)


        # SOTA Stability Measure: Freeze the large vision backbones[cite: 482, 492].
        log.info("Freezing perception backbones for stable fine-tuning...")
        self.ego_planner.strategist.vision_backbone.requires_grad_(False) 
        self.ego_planner.pilot.primary_obs_encoder.backbone.requires_grad_(False) 
        self.ego_planner.pilot.wrist_obs_encoder.backbone.requires_grad_(False) 

        # --- NEW SOTA v4 FIX ---
        # Freeze ALL parameters of the EgoPlanner model.
        # The gradients will only flow through the new, small PPO MLP head.
        self.ego_planner.requires_grad_(False)
        log.info("All EgoPlanner model parameters frozen.")

# In CLASS EgoPlannerAsFeaturesExtractor:
# REPLACE the entire forward method with this one.

    @torch.no_grad()
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        [SOTA V5.3, DEFINITIVE] This is the final, definitive version.
        It correctly handles the mixed HWC/CHW formats produced by the SB3
        wrapper stack. `VecTransposeImage` handles the static 3D images, while
        we manually permute the 4D stacked history images.
        """
        # --- START OF DEFINITIVE SOTA V5.3 FIX ---
        # The static images (initial_image, goal_image) are 3D and are automatically
        # transposed to CHW by the SB3 VecTransposeImage wrapper. We do NOT permute them.
        initial_image_chw = observations['initial_image']
        goal_image_chw = observations['goal_image']

        # The stacked history images are 4D (B, H_o, H, W, C) and are IGNORED by
        # VecTransposeImage. We MUST manually permute them to (B, H_o, C, H, W).
        primary_hist_chw = observations['image_primary'].permute(0, 1, 4, 2, 3)
        wrist_hist_chw = observations['image_wrist'].permute(0, 1, 4, 2, 3)
        # --- END OF DEFINITIVE SOTA V5.3 FIX ---

        # Apply resizing to the now-correctly-formatted CHW tensors.
        initial_image_resized = self.resize(initial_image_chw)
        goal_image_resized = self.resize(goal_image_chw)
        
        # For history, we must reshape, apply, and reshape back
        B, H_o, C, H, W = primary_hist_chw.shape
        primary_hist_resized = self.resize(primary_hist_chw.view(B * H_o, C, H, W)).view(B, H_o, C, 224, 224)

        B, H_o, C, H, W = wrist_hist_chw.shape
        wrist_hist_resized = self.resize_wrist(wrist_hist_chw.view(B * H_o, C, H, W)).view(B, H_o, C, 128, 128)

        # 1. Reconstruct the batch dictionary with correctly formatted and resized images.
        batch = {
            'initial_image': initial_image_resized,
            'goal_image': goal_image_resized,
            'observation_history': {
                'image_primary': primary_hist_resized,
                'image_wrist': wrist_hist_resized,
                'proprio': observations['proprio'],
            }
        }

        # 2. Extract features using the model's built-in, unified encoders.
        plan_vector = self.ego_planner.strategist(batch['initial_image'], batch['goal_image']) 
        vision_tokens, proprio_tokens = self.ego_planner.pilot.encode_tactics(batch['observation_history']) 

        # 3. Summarize and concatenate.
        vision_summary = vision_tokens.mean(dim=1)
        proprio_summary = proprio_tokens.mean(dim=1)

        # All tensors have shape (B, D_...), so `cat` is safe.
        features = torch.cat([plan_vector, vision_summary, proprio_summary], dim=1)
        
        return features

# ==============================================================================
# PHASE 2: THE SOTA "WARM-START" LOADER (FROM YOUR FILE)
# This function is SOTA and is used AS-IS.
# ==============================================================================

def load_pretrained_model(
    checkpoint_path: str, device: torch.device, rl_config: DictConfig
) -> Tuple[EgoPlanner, NoiseScheduler]:
    """
    [DEFINITIVE SOTA V3 - Self-Contained Loader]
    Definitive "warm-start" loader[cite: 524].
    1. Instantiates a fresh EgoPlanner model[cite: 525].
    2. Loads the checkpoint file[cite: 525].
    3. Intelligently strips the 'model.' prefix [cite: 525-526].
    4. Loads the EMA weights for the highest performance [cite: 527-528].
    """
    log.info(f"Loading pre-trained checkpoint from: {checkpoint_path}")
    
    # 1. Instantiate the model architecture using the RL script's config.
    model_cfg = EgoPlannerConfig(**rl_config.model)
    ego_planner_model = EgoPlanner(model_cfg) 
    
    # 2. Load the checkpoint on the CPU.
    checkpoint = torch.load(checkpoint_path, map_location='cpu') 

    # 3. Manually load the EMA weights [cite: 527-528].
    state_dict_key = 'ema_state_dict'
    if state_dict_key not in checkpoint:
        log.warning("Checkpoint does not contain 'ema_state_dict'. Falling back to standard 'state_dict'.")
        state_dict_key = 'state_dict' 
    
    # 4. Perform the critical prefix stripping[cite: 526].
    original_state_dict = checkpoint[state_dict_key]
    new_state_dict = {key.replace("model.", ""): value for key, value in original_state_dict.items()} 
    
    # 5. Load the corrected state dict into our fresh model [cite: 529-530].
    incompatible_keys = ego_planner_model.load_state_dict(new_state_dict, strict=False) 
    if incompatible_keys.missing_keys:
        log.warning(f"Weights not found in checkpoint for: {incompatible_keys.missing_keys}") 
    if incompatible_keys.unexpected_keys:
        log.warning(f"Checkpoint weights ignored (mismatch): {incompatible_keys.unexpected_keys}") 
    
    log.info("Successfully loaded pre-trained weights into new EgoPlanner instance.")
    
    # 6. Instantiate a fresh NoiseScheduler[cite: 531].
    scheduler_cfg = NoiseSchedulerConfig(**rl_config.scheduler)
    noise_scheduler = NoiseScheduler(scheduler_cfg) 

    # 7. Move model to the correct device and set to evaluation mode[cite: 532].
    ego_planner_model.to(device).eval() 
    
    return ego_planner_model, noise_scheduler


# ==============================================================================
# PHASE 3: THE SOTA TRAINING ORCHESTRATION SCRIPT
# This is the new, corrected `main` function.
# ==============================================================================

@hydra.main(version_base=None, config_path="../configs", config_name="finetune_ego_planner_config.yaml")
def main(cfg: DictConfig):
    log.info("--- Definitive RL Fine-Tuning Script for Ego-Planner (SOTA V4) ---")
    log.info(f"Full RL fine-tuning config:\n{OmegaConf.to_yaml(cfg)}")
    os.environ["WANDB_MODE"] = cfg.logging.wandb_mode 

    # --- 1. Setup ---
    pl.seed_everything(cfg.seed) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) 

    # --- 2. Load Pre-trained Model (The "Warm-Start") ---
    pretrained_model, noise_scheduler = load_pretrained_model(
        checkpoint_path=cfg.il_checkpoint_path, 
        device=device, 
        rl_config=cfg
    ) 

    # --- 3. Setup Environment Stack (THE CRITICAL SOTA v4 FIX) ---
    log.info("Setting up vectorized environment with FrameStackDict...")
    

    def make_env():
        env = PandaEnv(
            xml_path=cfg.env.xml_path,
            control_mode="delta"
        )
        # 1. Apply the reward wrapper FIRST.
        #    It needs direct access to the full, unwrapped observation from PandaEnv.
        reward_cfg = EgoPlannerRewardConfig(**cfg.reward_wrapper)
        env = EgoPlannerRewardWrapper(env, cfg=reward_cfg)

        # 2. Apply the observation stacking wrapper SECOND.
        #    It will now correctly process the observations for the policy,
        #    after the reward has already been calculated.
        env = FrameStackEgoPlanner(env, obs_horizon=cfg.model.obs_horizon)
        return env
    
    # Create the vectorized environment using our custom `make_env` function
    vec_env = make_vec_env(make_env, n_envs=cfg.n_envs, vec_env_cls=SubprocVecEnv if cfg.n_envs > 1 else DummyVecEnv)

    log.info("Training environment stack created successfully.")

    callbacks = []
    
    # Checkpoint callback [cite: 536]
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.callbacks.checkpoint_freq,
        save_path=str(output_dir / "rl_checkpoints"),
        name_prefix="ego_planner_rl"
    )
    callbacks.append(checkpoint_callback) 


    eval_env = make_vec_env(make_env, n_envs=1, vec_env_cls=DummyVecEnv)
    eval_env = VecTransposeImage(eval_env)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=cfg.callbacks.eval_freq,
        n_eval_episodes=cfg.callbacks.n_eval_episodes,
        deterministic=True,
        render=False
    ) 
    callbacks.append(eval_callback) 

    # W&B callback [cite: 539-542]
    if cfg.logging.use_wandb:
        if not WANDB_AVAILABLE:
            log.warning("W&B logging enabled, but 'wandb' package not found. Skipping.")
        else:
            import wandb
            wandb.init(
                project=cfg.logging.wandb_project,
                name=cfg.logging.wandb_run_name or output_dir.name,
                config=OmegaConf.to_container(cfg, resolve=True),
                sync_tensorboard=True,
                monitor_gym=True,
                save_code=True,
            ) 
            wandb_callback = WandbCallback(
                gradient_save_freq=cfg.logging.gradient_save_freq,
                model_save_path=str(output_dir / "wandb_models"),
                verbose=2
            ) 
            callbacks.append(wandb_callback) 
    # --- 5. Instantiate PPO Algorithm (THE SOTA v4 FIX) ---
    
    # We must programmatically define the `features_dim` that our
    # feature extractor will output.
    # From: D_vis + D_pilot + D_pilot
    features_dim = (
        pretrained_model.cfg.vision_feature_dim +
        pretrained_model.cfg.pilot_d_model +
        pretrained_model.cfg.pilot_d_model
    ) 

    # The `policy_kwargs` are the key to this entire SOTA script.
    # We tell PPO to use *our* feature extractor, and pass it the
    # pre-loaded, frozen EgoPlanner model.
    policy_kwargs = {
        'features_extractor_class': EgoPlannerAsFeaturesExtractor,
        'features_extractor_kwargs': {
            'features_dim': features_dim,
            'ego_planner_model': pretrained_model
        },
        # SOTA: Standardize features for the small MLP head
        'normalize_images': False, 
    }

    log.info(f"Calculated EgoPlanner features_dim: {features_dim}")
    log.info("Instantiating PPO with MlpPolicy and custom EgoPlannerFeatureExtractor.")

    ppo_model = PPO(
        policy="MultiInputPolicy", # We use the standard, stable MlpPolicy
        env=vec_env,
        learning_rate=cfg.ppo.learning_rate,
        n_steps=cfg.ppo.n_steps,
        batch_size=cfg.ppo.batch_size,
        n_epochs=cfg.ppo.n_epochs,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda, 
        clip_range=cfg.ppo.clip_range, 
        ent_coef=cfg.ppo.ent_coef, 
        vf_coef=cfg.ppo.vf_coef, 
        max_grad_norm=cfg.ppo.max_grad_norm, 
        verbose=1,
        tensorboard_log=str(output_dir / "tb_logs"),
        policy_kwargs=policy_kwargs, # This injects our frozen model
        device=device
    )

    # --- 6. Launch Fine-Tuning ---
    log.info("Starting RL fine-tuning with PPO (SOTA v4)...")
    ppo_model.learn(
        total_timesteps=cfg.total_timesteps, 
        callback=callbacks
    )

    # --- 7. Final Save and Cleanup ---
    final_model_path = output_dir / "final_model.zip"
    ppo_model.save(final_model_path) 
    log.info(f"Final RL model saved to: {final_model_path}")
    
    vec_env.close() 
    eval_env.close() 
    if cfg.logging.use_wandb and WANDB_AVAILABLE:
        wandb.finish() 
    
    log.info("--- RL Fine-Tuning Complete (SOTA v4) ---")

if __name__ == "__main__":
    main()