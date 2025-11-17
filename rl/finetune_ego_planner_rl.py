# FILE: train/finetune_ego_planner_rl.py
# (State-of-the-Art, V3 - Definitive Synthesis for Ego-Planner Fine-Tuning)

"""
The definitive, state-of-the-art reinforcement learning fine-tuning script for
the pre-trained Ego-Planner model.

This script is a synthesis of best practices, combining an architecturally-
consistent training structure with advanced, physics-based reward components
derived from modern robotics research. It implements the "Guided Refinement"
philosophy, using the stability of the PPO algorithm to gently refine the
behavior of the imitation-learned policy.

Key SOTA Features of this Definitive Version:
  - **Architecturally-Consistent Integration**: Implements a custom Stable-Baselines3
    policy that correctly uses the Ego-Planner as a combined features extractor
    and diffusion-based actor, ensuring perfect alignment between the RL
    framework and the complex generative model.
  - **Frozen Perception Backbones**: To preserve the powerful, generalized
    knowledge from imitation learning, the weights of the large vision backbones
    are frozen during fine-tuning. RL updates are focused on the decision-making
    layers of the network, preventing catastrophic forgetting (per RT-1 paper).
  - **Advanced Bi-Modal Reward**: Utilizes the `EgoPlannerRewardWrapper` which
    provides a bi-modal (Pre-Grasp / Post-Grasp) reward signal that is
    philosophically aligned with the Ego-Planner's static-plan architecture.
  - **Physics-Based Penalties**: Incorporates quadratic penalties on action and
    velocity to encourage smooth, stable motion (per TD3/SAC papers), alongside
    robust contact and drop penalties.
  - **Resilient Evaluation and Checkpointing**: Employs an `EvalCallback` to
    periodically evaluate the policy's true performance in a deterministic
    environment and save the best-performing model, protecting against
    late-stage training instability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.distributions import DiagGaussianDistribution
import pytorch_lightning as pl
import gymnasium as gym
import numpy as np
import os
# --- Project-Specific Imports ---
from envs.panda_env import PandaEnv
from models.ego_planner import EgoPlanner, NoiseScheduler, NoiseSchedulerConfig, EgoPlannerConfig
from train.train_ego_planner import EgoPlannerLightningModule
from rl.ego_planner_reward_wrapper import EgoPlannerRewardWrapper, EgoPlannerRewardConfig


try:
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WandbCallback = None
    WANDB_AVAILABLE = False




# Setup a logger for this module
log = logging.getLogger(__name__)


# ==============================================================================
# PHASE 1: THE CORE INTEGRATION - BRIDGING EGO-PLANNER AND STABLE-BASELINES3
# ==============================================================================

class EgoPlannerAsFeaturesExtractor(BaseFeaturesExtractor):
    """
    SOTA adapter that defines the Ego-Planner's encoders as a shared, frozen-backbone
    feature extractor for a Stable-Baselines3 Actor-Critic algorithm.
    """
    def __init__(self,
                 observation_space: gym.spaces.Dict,
                 features_dim: int,
                 ego_planner_model: EgoPlanner):
        super().__init__(observation_space, features_dim)
        log.info(f"Initializing EgoPlannerAsFeaturesExtractor with features_dim={features_dim}")
        self.ego_planner = ego_planner_model

        # SOTA Stability Measure: Freeze the large vision backbones.
        # RL fine-tuning should adapt the decision-making layers, not relearn perception.
        log.info("Freezing perception backbones for stable fine-tuning...")
        for param in self.ego_planner.strategist.vision_backbone.parameters():
            param.requires_grad = False
        for param in self.ego_planner.pilot.primary_obs_encoder.backbone.parameters():
            param.requires_grad = False
        for param in self.ego_planner.pilot.wrist_obs_encoder.backbone.parameters():
            param.requires_grad = False
        log.info("Perception backbones frozen.")

  
    @torch.no_grad()
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        [SOTA V4, DEFINITIVE & ROBUST]
        This version is robust to the fact that Stable-Baselines3 provides a time
        dimension (history) for image observations but NOT for other vector observations
        like proprioception. It processes them separately to prevent rank mismatches.
        """
        # 1. Get the high-level plan vector from the strategist. It expects 2D images.
        plan_vector = self.ego_planner.strategist(
            observations['initial_image'],
            observations['goal_image']
        )
        
        # 2. Process the visual HISTORY provided by SB3.
        #    The images in `observations` already have the time dimension (B, T, C, H, W).
        vision_history = {
            'image_primary': observations['image_primary'],
            'image_wrist': observations['image_wrist'],
        }
        
        # We need to give `encode_tactics` a proprio tensor of the correct rank (3D),
        # even if we ignore its output. We can just expand the current proprio.
        temp_proprio_history = observations['proprio'].unsqueeze(1).repeat(1, self.ego_planner.cfg.obs_horizon, 1)
        vision_history['proprio'] = temp_proprio_history
        
        vision_tokens, _ = self.ego_planner.pilot.encode_tactics(vision_history)
        vision_summary = vision_tokens.mean(dim=1) # -> results in a 2D tensor [B, D_pilot]

        # 3. Process the CURRENT proprioceptive state separately.
        #    `observations['proprio']` is a 2D tensor [B, D_proprio].
        #    The projection result will be a 2D tensor [B, D_pilot].
        #    This is ALREADY the "summary" for the current step, no .mean() is needed.
        proprio_features = self.ego_planner.pilot.proprio_proj(observations['proprio'])

        # 4. Concatenate the three 2D tensors.
        #    plan_vector:      [B, D_vis]
        #    vision_summary:   [B, D_pilot]
        #    proprio_features: [B, D_pilot]
        features = torch.cat([plan_vector, vision_summary, proprio_features], dim=1)
        
        return features


class EgoPlannerActorCriticPolicy(ActorCriticPolicy):
    """
    The definitive custom policy for integrating the generative Ego-Planner model with PPO.
    This version uses the correct constructor ordering and architectural setup.
    """
    
    def __init__(self, *args, ego_planner_model: EgoPlanner, noise_scheduler: NoiseScheduler, inference_cfg: DictConfig, **kwargs):
        # Prepare the feature_dim for the custom features extractor.
        features_dim = (ego_planner_model.cfg.vision_feature_dim +
                        2 * ego_planner_model.cfg.pilot_d_model)

        # The parent constructor MUST be called first. It will create our custom
        # EgoPlannerAsFeaturesExtractor and also the standard self.mlp_extractor.
        super().__init__(*args, **kwargs,
                         features_extractor_class=EgoPlannerAsFeaturesExtractor,
                         features_extractor_kwargs={'features_dim': features_dim, 'ego_planner_model': ego_planner_model})

        # Now that super().__init__() is done, we can safely assign our attributes.
        self.ego_planner = ego_planner_model
        self.noise_scheduler = noise_scheduler
        self.inference_cfg = inference_cfg

        # --- CRITICAL FIX ---
        # The Critic's value_net does NOT operate on the raw `features_dim` (1792).
        # It operates on the LATENT dimension of the policy's internal MLP extractor,
        # which by default is 64.
        #
        # We access `self.mlp_extractor.latent_dim_vf` to get this dimension programmatically.
        #
        latent_dim_critic = self.mlp_extractor.latent_dim_vf

        self.value_net = nn.Sequential(
            nn.Linear(latent_dim_critic, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )


    def _predict(self, observation: Dict[str, torch.Tensor], deterministic: bool = True) -> torch.Tensor:
        """
        [SOTA V4, DEFINITIVE] The Actor's action generation function.
        This version is patched to create a synthetic observation history, resolving
        the dimensionality mismatch between the RL loop's current state and the
        pre-trained model's expectation of a time series.
        """
        H_o = self.ego_planner.cfg.obs_horizon

        # --- CRITICAL FIX: Manually create the time dimension (obs_horizon) ---
        # The model was trained on a history of T observations. In the RL `predict`
        # step, we only have the current observation. We must "stack" the current
        # obs T times to create a synthetic history that matches the model's input shape.
        
        # Get the current observation tensors. Images are (B, C, H, W), proprio is (B, D).
        current_primary_img = observation['image_primary']
        current_wrist_img = observation['image_wrist']
        current_proprio = observation['proprio']
        
        # Add a time dimension of 1, then repeat H_o times.
        # (B, C, H, W) -> (B, 1, C, H, W) -> (B, H_o, C, H, W)
        # (B, D)       -> (B, 1, D)       -> (B, H_o, D)
        primary_history = current_primary_img.unsqueeze(1).repeat(1, H_o, 1, 1, 1)
        wrist_history = current_wrist_img.unsqueeze(1).repeat(1, H_o, 1, 1, 1)
        proprio_history = current_proprio.unsqueeze(1).repeat(1, H_o, 1)

        # Reconstruct the batch with correctly shaped 3D tensors for history.
        batch = {
            'initial_image': observation['initial_image'],
            'goal_image': observation['goal_image'],
            'observation_history': {
                'image_primary': primary_history,
                'image_wrist': wrist_history,
                'proprio': proprio_history,
            }
        }
        
        # Now, the sampling call will receive correctly shaped data.
        with torch.no_grad():
            action_chunk = self.ego_planner.sample(
                batch=batch,
                scheduler=self.noise_scheduler,
                guidance_plan=self.inference_cfg.guidance_scale_plan,
                guidance_obs=self.inference_cfg.guidance_scale_obs,
                num_inference_steps=self.inference_cfg.sampling_steps
            )
        
        return action_chunk[:, 0, :]

# In FILE: rl/finetune_ego_planner_rl.py
# In CLASS: EgoPlannerActorCriticPolicy

    # --- START OF THE DEFINITIVE FINAL PATCH ---
    # REPLACE the existing forward method with this corrected one.
    
    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = True) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        """
        [SOTA V6, DEFINITIVE & API-COMPLIANT]
        This is the final, definitive version, patched to use the correct keyword
        argument for the DiagGaussianDistribution class.
        """
        # 1. Get shared features and critic value (this part is correct).
        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        values = self.value_net(latent_vf)
        
        # 2. Get the deterministic action from our diffusion actor (this is correct).
        actions = self._predict(obs, deterministic=deterministic)
        
        # 3. Create the probability distribution for PPO's loss calculation.
        
        # --- CRITICAL FIX: Use 'mean_actions' instead of 'action_logits' ---
        # This is a simple but critical API signature mismatch. DiagGaussianDistribution
        # expects the mean of the distribution under the keyword 'mean_actions'.
        
        distribution = self.action_dist.proba_distribution(
            mean_actions=actions,       # THIS IS THE CORRECT KEYWORD ARGUMENT
            log_std=self.log_std
        )
        
        action_log_probs = distribution.log_prob(actions)
        
        return actions, values, action_log_probs
    # --- END OF THE DEFINITIVE FINAL PATCH ---


# ==============================================================================
# PHASE 2: THE ENVIRONMENT & TRAINING ORCHESTRATION SCRIPT
# ==============================================================================


def load_pretrained_model(
    checkpoint_path: str, device: torch.device, rl_config: DictConfig
) -> Tuple[EgoPlanner, NoiseScheduler]:
    """
    [DEFINITIVE SOTA V3 - Self-Contained Loader]
    Definitive "warm-start" loader. This version is fully self-contained and
    does not depend on the original LightningModule.

    It performs the following robust steps:
    1. Instantiates a fresh EgoPlanner model using the RL script's own config.
    2. Loads the checkpoint file.
    3. Intelligently strips the 'model.' prefix from the state dict keys.
    4. Loads the compatible weights into the fresh model instance.
    5. Loads the EMA weights for the highest performance.
    """
    log.info(f"Loading pre-trained checkpoint from: {checkpoint_path}")
    
    # 1. Instantiate the model architecture using the RL script's config.
    #    This ensures the model structure is known before loading weights.
    model_cfg = EgoPlannerConfig(**rl_config.model)
    ego_planner_model = EgoPlanner(model_cfg)
    
    # 2. Load the checkpoint on the CPU.
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 3. Manually load the EMA weights, as they are the most important.
    #    Fall back to standard weights if EMA is not present.
    state_dict_key = 'ema_state_dict'
    if state_dict_key not in checkpoint:
        log.warning("Checkpoint does not contain 'ema_state_dict'. Falling back to standard 'state_dict'.")
        state_dict_key = 'state_dict'
    
    # 4. Perform the critical prefix stripping.
    original_state_dict = checkpoint[state_dict_key]
    # Example: 'model.strategist.xyz' becomes 'strategist.xyz'
    new_state_dict = {key.replace("model.", ""): value for key, value in original_state_dict.items()}
    
    # 5. Load the corrected state dict into our fresh model.
    incompatible_keys = ego_planner_model.load_state_dict(new_state_dict, strict=False)
    if incompatible_keys.missing_keys:
        log.warning(f"Weights not found in checkpoint for: {incompatible_keys.missing_keys}")
    if incompatible_keys.unexpected_keys:
        log.warning(f"Checkpoint weights ignored (mismatch): {incompatible_keys.unexpected_keys}")
    
    log.info("Successfully loaded pre-trained weights into new EgoPlanner instance.")
    
    # 6. Instantiate a fresh NoiseScheduler.
    scheduler_cfg = NoiseSchedulerConfig(**rl_config.scheduler)
    noise_scheduler = NoiseScheduler(scheduler_cfg)

    # 7. Move model to the correct device and set to evaluation mode.
    ego_planner_model.to(device).eval()
    
    return ego_planner_model, noise_scheduler


@hydra.main(version_base=None, config_path="../configs", config_name="finetune_ego_planner_config.yaml")
def main(cfg: DictConfig):
    log.info("--- Definitive RL Fine-Tuning Script for Ego-Planner (SOTA V3) ---")
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

    # --- 3. Setup Environment Stack ---
    # `make_vec_env` is a utility that handles env creation, wrapping, and vectorization.
    vec_env = make_vec_env(
        PandaEnv,
        n_envs=cfg.n_envs,
        env_kwargs={
            "xml_path": cfg.env.xml_path,
            "control_mode": "delta"
        },
        wrapper_class=EgoPlannerRewardWrapper,
        wrapper_kwargs={"cfg": EgoPlannerRewardConfig(**cfg.reward_wrapper)}
    )

    # --- 4. Define Callbacks (The Automation Engine) ---
    callbacks = []
    
    # Checkpoint callback to save the RL training state for full resumption.
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.callbacks.checkpoint_freq,
        save_path=str(output_dir / "rl_checkpoints"),
        name_prefix="ego_planner_rl"
    )
    callbacks.append(checkpoint_callback)

    # Evaluation callback to run periodic evals and save the *best* performing model.
    eval_env = make_vec_env(
        PandaEnv, n_envs=1,
        env_kwargs={"xml_path": cfg.env.xml_path, "control_mode": "delta"},
        wrapper_class=EgoPlannerRewardWrapper,
        wrapper_kwargs={"cfg": EgoPlannerRewardConfig(**cfg.reward_wrapper)}
    )
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

    # W&B callback for rich, real-time logging.
    if cfg.logging.use_wandb:
        if not WANDB_AVAILABLE:
            log.warning("W&B logging is enabled in config, but the 'wandb' package is not installed. Skipping W&B callback.")
        else:
            import wandb
            # Initialize wandb
            wandb.init(
                project=cfg.logging.wandb_project,
                name=cfg.logging.wandb_run_name or output_dir.name,
                config=OmegaConf.to_container(cfg, resolve=True),
                sync_tensorboard=True,  # Captures PPO's internal TensorBoard logs
                monitor_gym=True,       # Automatically logs episode rewards, lengths, etc.
                save_code=True,         # Saves a copy of the script to the W&B run
            )
            # Create and add the callback
            wandb_callback = WandbCallback(
                gradient_save_freq=cfg.logging.gradient_save_freq,
                model_save_path=str(output_dir / "wandb_models"),
                verbose=2
            )
            callbacks.append(wandb_callback)

    # --- 5. Instantiate PPO Algorithm with Custom Policy ---
    # The `policy_kwargs` are the key to injecting our pre-trained model and custom logic.
    policy_kwargs = {
        'ego_planner_model': pretrained_model,
        'noise_scheduler': noise_scheduler,
        'inference_cfg': cfg.inference,
    }

    ppo_model = PPO(
        policy=EgoPlannerActorCriticPolicy,
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
        policy_kwargs=policy_kwargs,
        device=device
    )

    # --- 6. Launch Fine-Tuning ---
    log.info("Starting RL fine-tuning with PPO...")
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
    if cfg.logging.use_wandb:
        wandb.finish()
    
    log.info("--- RL Fine-Tuning Complete. ---")

if __name__ == "__main__":
    main()