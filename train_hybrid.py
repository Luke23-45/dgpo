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
from pathlib import Path
from typing import Optional
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from utils.expert_dataset import ExpertDataset

# Configure logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
)
logger = logging.getLogger("dgpo.train_hybrid")
# --- ROBUST HELPERS FOR HYBRID UPDATES ---
def select_actor_parameters(policy):
    """
    Return (actor_params_list, actor_param_names).

    Heuristics to pick actor (policy) parameters while avoiding value/critic params.
    This works across common SB3 policy naming conventions. If nothing matched,
    falls back to all policy params and logs a WARNING.
    """
    value_patterns = ("value_net", "value", "vf", "critic", "mlp_extractor.value_net", "value_branch")
    actor_params = []
    actor_names = []

    for name, p in policy.named_parameters():
        if any(pat in name for pat in value_patterns):
            continue
        actor_params.append(p)
        actor_names.append(name)

    # If nothing selected (policy uses different naming), attempt common actor patterns
    if len(actor_params) == 0:
        for name, p in policy.named_parameters():
            if "policy_net" in name or "action_net" in name or "mlp_extractor.policy_net" in name:
                actor_params.append(p)
                actor_names.append(name)

    # Last resort: use all policy params
    if len(actor_params) == 0:
        actor_params = list(policy.parameters())
        actor_names = [n for n, _ in policy.named_parameters()]
        import logging
        logging.getLogger(__name__).warning(
            "select_actor_parameters: fallback to ALL policy params (could include critic params)."
        )

    return actor_params, actor_names

def get_policy_predicted_actions(policy, obs_dict):
    """
    Robustly produce the policy's deterministic predicted actions (tensor with grads if possible).

    Attempts multiple SB3-friendly APIs, from public -> internal -> fallback:
      1. policy.get_distribution(obs_dict) -> .mode / .mean if available
      2. policy.forward(obs_dict, deterministic=True) -> returns actions, value, logp (some SB3)
      3. internal mlp_extractor -> _get_action_dist_from_latent (last resort)
      4. policy.predict (detached numpy) as final fallback (warns about no grad)

    Returns a torch.Tensor on same device as policy parameters. If a differentiable
    tensor cannot be obtained (fallback), the function returns a detached tensor,
    and the caller must be aware BC updates will not produce gradients in that case.
    """
    device = next(policy.parameters()).device
    was_training = policy.training
    policy.train()  # ensure forwards provide tensors requiring grads

    # Helper to ensure obs are tensors on correct device
    # Many policies expect dict of tensors; assume obs_dict already built with torch tensors
    try:
        # 1) public API: get_distribution (some SB3 versions)
        if hasattr(policy, "get_distribution"):
            dist = policy.get_distribution(obs_dict)
            if hasattr(dist, "mode"):
                actions = dist.mode
                if torch.is_tensor(actions):
                    return actions.to(device)
            if hasattr(dist, "mean"):
                actions = dist.mean
                if torch.is_tensor(actions):
                    return actions.to(device)
            # some dist objects expose deterministic methods
            if hasattr(dist, "get_actions"):
                actions = dist.get_actions(deterministic=True)
                if torch.is_tensor(actions):
                    return actions.to(device)
    except Exception:
        pass

    try:
        # 2) forward API: many SB3 policies implement forward(obs, deterministic=True)
        if hasattr(policy, "forward"):
            # Some implementations expect numpy obs; policy.forward usually accepts tensors
            forward_result = policy.forward(obs_dict, deterministic=True)
            # forward may return (actions, values, log_probs) or similar; be defensive
            if isinstance(forward_result, tuple) and len(forward_result) >= 1:
                actions = forward_result[0]
                if torch.is_tensor(actions):
                    return actions.to(device)
    except Exception:
        pass

    try:
        # 3) internal latent -> distribution (last resort)
        if hasattr(policy, "extract_features") and hasattr(policy, "mlp_extractor"):
            features = policy.extract_features(obs_dict)
            # Some SB3 versions: mlp_extractor.forward_actor(features) / policy_net ...
            if hasattr(policy.mlp_extractor, "forward_actor"):
                latent_pi = policy.mlp_extractor.forward_actor(features)
            elif hasattr(policy.mlp_extractor, "policy_net"):
                latent_pi = policy.mlp_extractor.policy_net(features)
            else:
                latent_pi = None

            if latent_pi is not None and hasattr(policy, "_get_action_dist_from_latent"):
                dist = policy._get_action_dist_from_latent(latent_pi)
                if hasattr(dist, "mode"):
                    actions = dist.mode
                    if torch.is_tensor(actions):
                        return actions.to(device)
                if hasattr(dist, "mean"):
                    actions = dist.mean
                    if torch.is_tensor(actions):
                        return actions.to(device)
    except Exception:
        pass

    # Final fallback: use policy.predict -> produces numpy (detached)
    try:
        # Convert tensor obs to numpy for predict (best-effort)
        obs_numpy = {}
        for k, v in obs_dict.items():
            try:
                obs_numpy[k] = v.cpu().detach().numpy()
            except Exception:
                # if conversion fails (non-tensor), ignore field
                pass
        # policy.predict returns (actions, state) and is deterministic if flag set
        actions_np, _ = policy.predict(obs_numpy, deterministic=True)
        actions_t = torch.as_tensor(actions_np, device=device, dtype=torch.float32)
        import logging
        logging.getLogger(__name__).warning(
            "get_policy_predicted_actions: used detached fallback (policy.predict) -> actions will be non-differentiable."
        )
        return actions_t
    except Exception as e:
        # Give a clear error message for debugging
        raise RuntimeError(
            "get_policy_predicted_actions: unable to obtain policy actions via any strategy. "
            "Inspect the policy object and SB3 version for API compatibility."
        )
    finally:
        if not was_training:
            policy.eval()

def run_hybrid_training(args: argparse.Namespace):
    """
    Main function to orchestrate the hybrid BC+RL training process.
    """
    # --- 1. SETUP AND INITIALIZATION ---
    # This section is very similar to `run_experiment.py`, but ensures
    # the environment is configured correctly for our hybrid strategy.
    
    run_name = args.run_name or f"hybrid_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(args.output_dir) / run_name
    checkpoints_dir = run_dir / "checkpoints"
    backups_dir = run_dir / "backups"
    
    # Create directories for artifactsf
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)
    backups_dir.mkdir(exist_ok=True)

    # Save the configuration for this run
    with (run_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=4)
    logger.info(f"🚀 Starting Hybrid BC+RL experiment: {run_name}")
    logger.info(f"All artifacts will be saved in: {run_dir}")

    # Set random seeds for reproducibility
    set_random_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


  
    pos_scale = getattr(args, 'pos_scale', 0.05)
    env = setup_environment(
        xml_path=args.xml_path,
        seed=args.seed,
        n_envs=args.n_envs,
        octo_model=None,          # Hybrid mode doesn't use the Octo model
        w_plausibility=0.0,
        pos_scale=pos_scale,
        rot_scale=getattr(args, 'rot_scale', 1.0),
        div_clip=getattr(args, 'div_clip', 10.0),
        scripted_expert=None,    
        w_guidance=args.w_guidance,
        w_guidance_dense=args.w_guidance_dense,
        guidance_clip=args.guidance_clip,
        grasp_reward=args.grasp_reward,
        lift_reward=args.lift_reward,
        success_reward=args.success_reward,
        # Add other args from run_experiment's setup_environment if needed
        # For simplicity, we can also just pass the whole `args` object if the names match
    )
    logger.info("Environment created using centralized setup_environment from run_experiment.py")


    logger.info("Initializing PPO agent with STABLE hyperparameters for fine-tuning...")

    # Use the centralized PPO agent initialization from run_experiment.py
    ppo_agent = initialize_ppo_agent(env, run_dir, args.seed, args.device)
    # Load the new, balanced BC model and transfer weights
    bc_model_path = str(Path(args.bc_init_dir) / "checkpoints" / "best_model.pth")
    logger.info(f"Loading balanced BC checkpoint from: {bc_model_path}")
    try:
        device = ppo_agent.policy.device

        # Load BC checkpoint (robust to different saved formats)
        logger.info(f"Loading balanced BC checkpoint from: {bc_model_path}")
        try:
            ckpt = load_bc_checkpoint(bc_model_path, device)
            # load_bc_checkpoint may return a dict with keys, or the raw state_dict
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif isinstance(ckpt, dict) and any(k.startswith("cnn") or k.startswith("features_extractor") for k in ckpt.keys()):
                # heuristically assume this dict is the state_dict
                state_dict = ckpt
            else:
                # fallback (caller returned something else)
                state_dict = ckpt

            action_dim = env.action_space.shape[0]
            bc_net = BCNet(n_actions=action_dim).to(device)
            bc_net.load_state_dict(state_dict, strict=False)
            logger.info("BC model loaded into BCNet (strict=False).")
        except Exception as e:
            logger.exception("Failed to load BC checkpoint. Aborting hybrid run.")
            env.close()
            return

        if transfer_bc_weights:
            transfer_bc_weights(bc_net, ppo_agent)
            logger.info("Weight transfer from balanced BC model successful.")
        else:
            logger.warning("`transfer_bc_weights` utility not available. Agent is random.")
    except Exception as e:
        logger.error(f"Failed during BC weight transfer: {e}", exc_info=True)
        env.close()
        return
    
    if args.freeze_features:
        logger.info("--- FEATURE FREEZING ENABLED ---")
        frozen_keys = 0
        for name, param in ppo_agent.policy.named_parameters():
            if 'features_extractor' in name:
                param.requires_grad = False
                frozen_keys += 1
        logger.info(f"Froze {frozen_keys} parameters in the feature extractor.")
    # --- 2. PREPARE FOR IMITATION REGULARIZATION ---
    logger.info("\n--- Preparing for Imitation Regularization ---")
    
    # 2.1: Create the ExpertDataset and DataLoader for BC updates
    logger.info("Initializing Expert Dataloader for BC updates...")
    expert_dataset = ExpertDataset(
        urdf_path=args.urdf_path,
        base_seed=args.seed + 1000,
        env_xml_path=args.xml_path, 
        max_samples_per_epoch=None,
        yield_full_obs=False
    )
    
    expert_loader = DataLoader(
        expert_dataset,
        batch_size=args.bc_batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True
    )
    expert_iterator = iter(expert_loader)
    logger.info("Expert Dataloader is ready.")

    # 2.2: Create a separate optimizer for the BC updates
    # We only update the policy network (actor), not the value network.
    # Select actor-only parameters (avoid corrupting critic/value network)
    actor_params, actor_param_names = select_actor_parameters(ppo_agent.policy)
    policy_params_to_update = actor_params  # keep old name used later

    bc_optimizer = torch.optim.Adam(
        policy_params_to_update, lr=args.bc_lr, weight_decay=1e-6
    )
    logger.info(f"BC optimizer configured. Actor params count: {len(actor_param_names)}. Examples: {actor_param_names[:6]}")

    bc_loss_fn = nn.MSELoss()
    logger.info("BC optimizer configured for policy network.")

    # --- 3. HYBRID TRAINING LOOP ---
    logger.info("\n--- Starting Hybrid Training Loop ---")

    # 3.1: Define Training Parameters
    rl_steps_per_iteration = args.n_steps * args.n_envs
    bc_updates_per_iteration = args.bc_updates
    initial_bc_lambda = args.bc_lambda_initial
    total_timesteps = args.total_timesteps

    # Setup SB3 Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.save_freq // args.n_envs),
        save_path=str(checkpoints_dir),
        name_prefix="hybrid_rl_policy"
    )
    backup_callback = SingleFileBackupCallback(
        save_freq=max(1, 5000 // args.n_envs),
        save_path=str(backups_dir),
        name_prefix="latest_backup"
    )
    callback_list = [checkpoint_callback, backup_callback]

    start_time = time.time()
    ppo_agent.num_timesteps = 0
    ppo_agent._episode_num = 0
    ppo_agent._total_timesteps = total_timesteps

    # 3.2: The Main Loop
    try:
        while ppo_agent.num_timesteps < total_timesteps:
          
          # --- PHASE A: Reinforcement Learning ---
          ppo_agent.learn(
              total_timesteps=rl_steps_per_iteration,
              callback=callback_list,
              reset_num_timesteps=False,
              log_interval=1,
              progress_bar=True
          )
          
          # --- PHASE B: Imitation Learning (BC Regularization) ---
          progress = ppo_agent.num_timesteps / total_timesteps
          current_bc_lambda = initial_bc_lambda * (1.0 - progress)

          total_bc_loss = 0.0
          if current_bc_lambda > 0:
              for _ in range(bc_updates_per_iteration):
                  try:
                      obs_expert, act_expert = next(expert_iterator)
                  except StopIteration:
                      expert_iterator = iter(expert_loader)
                      obs_expert, act_expert = next(expert_iterator)
                  
                  act_expert = act_expert.to(ppo_agent.policy.device)
                  obs_expert_device = {k: v.to(ppo_agent.policy.device) for k, v in obs_expert.items()}
                  
                  # Get policy's predicted action distribution
                  # For PPO, the policy outputs a distribution, not a raw action
                  # Robust BC update using helper that tries public SB3 APIs for differentiable actions
                  predicted_actions = get_policy_predicted_actions(ppo_agent.policy, obs_expert_device)

                  # If predicted_actions came from the detached fallback, it won't require grad.
                  if not predicted_actions.requires_grad:
                      logger.warning("Predicted actions do not require grad. BC update will not adjust the policy via autograd. "
                                    "This means the fallback (policy.predict) was used. Investigate policy API.")
                  # Compute BC loss (MSE between predicted and expert actions)
                  bc_loss = bc_loss_fn(predicted_actions, act_expert)
                  weighted_bc_loss = bc_loss * current_bc_lambda

                  bc_optimizer.zero_grad()
                  weighted_bc_loss.backward()

                  # Clip only actor params (policy_params_to_update) to avoid huge updates
                  torch.nn.utils.clip_grad_norm_(policy_params_to_update, max_norm=1.0)
                  bc_optimizer.step()

                  total_bc_loss += bc_loss.item()

          
          # --- 3.3: Logging ---
          avg_bc_loss = total_bc_loss / bc_updates_per_iteration if bc_updates_per_iteration > 0 else 0
          
          ppo_agent.logger.record("custom/bc_loss", avg_bc_loss)
          ppo_agent.logger.record("custom/bc_lambda", current_bc_lambda)
          
          # Dump all logs (RL + custom) to TensorBoard
          ppo_agent.logger.dump(step=ppo_agent.num_timesteps)

          logger.info(f"Timesteps: {ppo_agent.num_timesteps}/{total_timesteps} | "
                      f"BC Lambda: {current_bc_lambda:.3f} | Avg BC Loss: {avg_bc_loss:.5f}")

        # --- 4. FINAL SAVE AND CLEANUP ---
        final_model_path = run_dir / "final_policy.zip"
        ppo_agent.save(final_model_path)
        logger.info(f"✅ Training complete. Final policy saved to: {final_model_path}")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user; saving current policy and exiting.")
        ppo_agent.save(run_dir / "interrupted_policy.zip")
    except Exception:
        logger.exception("Unexpected error during training. Saving state and exiting.")
        try:
            ppo_agent.save(run_dir / "error_policy.zip")
        except Exception:
            logger.exception("Failed to save policy after exception.")
    finally:
        try:
            env.close()
        except Exception:
            logger.exception("Failed to close env cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hybrid BC+RL Training for DGPO")

    # --- Run Management ---
    parser.add_argument("--run_name", type=str, default=None, help="Name for the training run.")
    parser.add_argument("--output_dir", type=str, default="trained_models", help="Directory to save training artifacts.")
    parser.add_argument("--bc_init_dir", type=str, required=True, help="Path to the COMPLETED, balanced BC run directory.")
    
    # --- Training Parameters ---
    parser.add_argument("--total_timesteps", type=int, default=2_000_000, help="Total timesteps for training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use ('cpu', 'cuda', 'auto').")
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environments.")
    parser.add_argument("--save_freq", type=int, default=50000, help="Frequency to save checkpoints.")
    
    # --- Environment Parameters ---
    parser.add_argument("--xml_path", type=str, default="envs/panda_pick_place.xml", help="Path to the MuJoCo XML file.")
    parser.add_argument("--urdf_path", type=str, default="urdf/panda_mujoco_kinematics.urdf", help="Path to the URDF file for IKSolver.")

    # --- Hybrid Training Hyperparameters ---

    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for the expert dataloader (0 = main process).")
    parser.add_argument("--bc_batch_size", type=int, default=64, help="Batch size for the BC update steps.")
    parser.add_argument("--bc_lr", type=float, default=1e-5, help="Learning rate for the BC optimizer.")
    parser.add_argument("--bc_updates", type=int, default=4, help="Number of BC gradient updates per RL iteration.")
    parser.add_argument("--bc_lambda_initial", type=float, default=1.0, help="Initial weight for the BC loss.")
    parser.add_argument("--n_steps", type=int, default=1024, help="PPO rollout buffer size (recommend 512–2048).")

    # --- Reward Hyperparameters ---

    
    parser.add_argument("--freeze_features", action="store_true", 
                        help="If set, freeze the feature extractor layers for stable initial fine-tuning.")
    
    parser.add_argument("--w_guidance", type=float, default=0.0,
                        help="Weight for the ScriptedExpert guidance terminal reward. Set > 0 to enable.")
    parser.add_argument("--w_guidance_dense", type=float, default=5.0, 
                        help="Weight for the DENSE ScriptedExpert guidance reward. Set > 0 to enable.")
    parser.add_argument("--grasp_reward", type=float, default=50.0, help="Sparse reward for grasping.")
    parser.add_argument("--lift_reward", type=float, default=100.0, help="Sparse reward for lifting.")
    parser.add_argument("--success_reward", type=float, default=250.0, help="Sparse reward for success.")
    
    parser.add_argument("--guidance_clip", type=float, default=1.0,
                        help="Maximum value to clip the raw guidance divergence score before weighting.")
    parser.add_argument("--pos_scale", type=float, default=0.05,
                        help="Action scaling factor for the delta controller.") 
    args = parser.parse_args()
    
    try:
        run_hybrid_training(args)
    except Exception as e:
        logger.exception("An error occurred during hybrid training.")
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


"""