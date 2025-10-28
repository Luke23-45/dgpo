# FILE: scripts/test_weight_transfer.py
# Description: A diagnostic script to verify the weight transfer from a
# pre-trained BC diffusion policy into the RL actor setup.

import sys
import logging
from pathlib import Path
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import copy
from itertools import chain



# --- Project Imports ---
from models.diffusion_policy import DiffusionPolicy, NoiseSchedulerConfig
from scripts.train_rl import DiffusionActor, Critic # Import the wrappers from train_rl
from utils.expert_dataset import ExpertTrajectoryDataset, collate_fn
from torch.utils.data import DataLoader

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger("WeightTransferTest")

# --- The Core Test Function ---

@hydra.main(version_base=None, config_path="./configs", config_name="finetune_rl_config")
def main(cfg: DictConfig):
    """
    Main test function managed by Hydra. It loads the RL finetuning config,
    simulates the model loading process, and verifies the actor's output.
    """
    log.info("--- Starting Weight Transfer Verification Script ---")
    log.info("Loaded configuration:\n" + OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.device)

    # --- 1. Load a single sample from the expert dataset ---
    log.info(f"Loading one sample from dataset: {cfg.dataset.path}")
    try:
        # We only need one sample to test the actor
        val_dataset = ExpertTrajectoryDataset(
            demo_path=cfg.dataset.path,
            observation_horizon=cfg.model.observation_horizon,
            action_horizon=cfg.model.action_horizon,
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        obs_chunk, expert_action_chunk = next(iter(val_loader))
        log.info("Successfully loaded one sample trajectory chunk.")
    except Exception as e:
        log.error(f"Failed to load dataset. Check path in config. Error: {e}", exc_info=True)
        sys.exit(1)

    # --- 2. Simulate the model build and load process from RLFineTuner ---
    # This block is a direct copy of the logic in RLFineTuner._build_models_and_optimizers
    try:
        log.info("Simulating model build and weight loading process...")
        proprio_dim = val_dataset.get_proprioception_dim()
        model_cfg = cfg.model
        action_dim = model_cfg.action_dim
        
        policy_kwargs = {
            "proprio_dim": proprio_dim, "H_o": model_cfg.observation_horizon,
            "H_a": model_cfg.action_horizon, "action_dim": action_dim,
            "image_feat_dim": model_cfg.image_feat_dim,
            "scheduler_cfg": NoiseSchedulerConfig(**cfg.scheduler),
            "d_model": model_cfg.d_model, "denoiser_layers": model_cfg.denoiser_layers,
            "denoiser_heads": model_cfg.denoiser_heads,
            "cfg_p_uncond": 0.0, "ema_decay": None, "device": device
        }

        # --- Load Pre-trained Weights ---
        pretrained_path = Path(cfg.pretrained_policy_path)
        if not pretrained_path.is_file():
            log.error(f"Pre-trained policy path not found: {pretrained_path}")
            sys.exit(1)

        log.info(f"Loading checkpoint from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device,weights_only=False)
        if 'policy_state_dict' in checkpoint:
            pretrained_policy_state_dict = checkpoint.get('ema_state_dict', checkpoint['policy_state_dict'])
            log.info("Extracted EMA weights from pre-train checkpoint.")
        else:
            pretrained_policy_state_dict = checkpoint
            log.info("Loaded raw policy weights (checkpoint did not contain 'policy_state_dict' key).")

        # --- Instantiate and Load Actor ---
        main_diffusion_policy = DiffusionPolicy(**policy_kwargs)
        main_diffusion_policy.load_state_dict(pretrained_policy_state_dict)
        log.info("Successfully loaded weights into a DiffusionPolicy instance.")
        
        actor = DiffusionActor(
            main_diffusion_policy,
            cfg.rl_algorithm.guidance_scale,
            cfg.rl_algorithm.sampling_steps
        ).to(device)
        actor.eval() # Set to evaluation mode
        log.info("Successfully created and loaded the DiffusionActor.")
        
        # --- Instantiate and Load Critic (to verify encoder sharing) ---
        critic_feature_extractor = actor.diffusion_policy.vision_fusion_encoder
        critic = Critic(
            features_extractor=critic_feature_extractor,
            action_dim=action_dim, d_model=model_cfg.d_model,
            hidden_dims=cfg.rl_algorithm.critic_net_arch,
            use_layernorm=cfg.rl_algorithm.use_critic_layernorm,
            use_last_feature=cfg.rl_algorithm.critic_use_last_feature
        ).to(device)
        log.info("Successfully created Critic with shared vision encoder.")

    except Exception as e:
        log.error(f"An error occurred during model building or loading: {e}", exc_info=True)
        sys.exit(1)


    # --- 3. Perform Verification ---
    log.info("--- Performing Verification ---")
    
    # Move sample data to device
    obs_sample = {k: v.to(device).float() for k, v in obs_chunk.items()}
    expert_action_sample = expert_action_chunk.to(device).float()

    # Generate an action from the loaded actor
    with torch.no_grad():
        actor_action = actor.act(obs_sample)

    # Compare the first action in the sequence
    expert_first_action = expert_action_sample[:, 0, :]
    actor_first_action = actor_action

    # Calculate Mean Squared Error (MSE)
    mse = torch.nn.functional.mse_loss(actor_first_action, expert_first_action)
    
    # --- 4. Report Results ---
    log.info("Verification Complete. Results:")
    log.info(f"  Shape of expert action (first step): {expert_first_action.shape}")
    log.info(f"  Shape of actor action (first step):   {actor_first_action.shape}")
    
    np.set_printoptions(precision=4, suppress=True)
    log.info(f"\nExpert Action: {expert_first_action.cpu().numpy().flatten()}")
    log.info(f"Actor Action:  {actor_first_action.cpu().numpy().flatten()}")
    
    log.info(f"\nMean Squared Error (MSE): {mse.item():.6f}")

    # Heuristic check
    if mse.item() < 0.1: # Threshold is a heuristic, adjust as needed
        log.info(" SUCCESS: The MSE is low, indicating the loaded actor is behaving similarly to the expert.")
        log.info("This suggests the weight transfer process in 'train_rl.py' is likely correct.")
    else:
        log.warning(" WARNING: The MSE is high. This could indicate several issues:")
        log.warning("  1. The pre-trained model did not converge well (high BC loss).")
        log.warning("  2. A mismatch in normalization between training and inference.")
        log.warning("  3. A fundamental issue in the model loading or architecture.")
        log.warning("  Double-check the pre-training logs and model configurations.")

if __name__ == "__main__":
    main()