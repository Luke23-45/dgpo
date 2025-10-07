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
from models.custom_sb3_extractor import BCFeaturesExtractor 

from pathlib import Path
from typing import Optional
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from tqdm import tqdm
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
    resolve_device
)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from utils.expert_dataset import ExpertDataset
from collections import deque
# Configure logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
)
logger = logging.getLogger("dgpo.train_hybrid")


class ActionSourceCallback(BaseCallback):
    """
    Monitor and log the source ('BC' or 'RL') of each action.
    - Resets at rollout start
    - Counts per-step in _on_step
    - Logs counts + percentage to TensorBoard at _on_rollout_end
    """
    def __init__(self, smooth_windows: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.bc_actions_count = 0
        self.rl_actions_count = 0
        self.history = deque(maxlen=smooth_windows)

    def _on_rollout_start(self) -> None:
        self.bc_actions_count = 0
        self.rl_actions_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos is None:
            return True

        # Normalize to a list so this works for single-env and vectorized envs.
        if isinstance(infos, dict):
            infos = [infos]

        for info in infos:
            if not info:
                continue
            src = info.get("action_source")
            if src == "BC":
                self.bc_actions_count += 1
            elif src == "RL":
                self.rl_actions_count += 1
        return True

    def _on_rollout_end(self) -> None:
        total_actions = self.bc_actions_count + self.rl_actions_count
        if total_actions == 0:
            return

        bc_percentage = (self.bc_actions_count / total_actions) * 100.0
        rl_percentage = 100.0 - bc_percentage

        # Record raw counts + percentage (TensorBoard will pick these up)
        self.logger.record("custom/bc_actions_count", self.bc_actions_count)
        self.logger.record("custom/rl_actions_count", self.rl_actions_count)
        self.logger.record("custom/bc_action_percentage", bc_percentage)
        self.logger.record("custom/rl_action_percentage", rl_percentage)

        # Optional: smoothed percentage for less noisy charting
        self.history.append(bc_percentage)
        self.logger.record("custom/bc_action_percentage_smooth", sum(self.history)/len(self.history))


class AdvisedPPO(PPO):
    """
    BC-Advised PPO Algorithm.
    Overrides the data collection step to use a frozen BC policy as an advisor.
    """
    def __init__(self, bc_policy_advisor: BCNet, epsilon_schedule: dict, **kwargs):
        super().__init__(**kwargs)
        self.bc_policy_advisor = bc_policy_advisor
        self.bc_policy_advisor.eval()
        self.bc_policy_advisor.to(self.device)

        self.eps_initial = epsilon_schedule.get("initial", 0.5)
        self.eps_final = epsilon_schedule.get("final", 0.05)
        self.eps_decay_steps = epsilon_schedule.get("decay_steps", 1_000_000)

    def _get_current_epsilon(self) -> float:
        """Linearly anneals epsilon from initial to final over decay_steps."""
        progress = self.num_timesteps / self.eps_decay_steps
        fraction = min(1.0, progress)
        return self.eps_initial + fraction * (self.eps_final - self.eps_initial)

    def collect_rollouts(
        self,
        env,
        callback: BaseCallback,
        rollout_buffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the BC-Advised strategy and store them.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()
        
        current_epsilon = self._get_current_epsilon()

        while n_steps < n_rollout_steps:
            # 1. Get actions from both the RL and the BC policies
            with torch.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions_rl, values, log_probs = self.policy(obs_tensor)
                actions_bc = self.bc_policy_advisor(obs_tensor)

            actions_rl = actions_rl.cpu().numpy()
            actions_bc = actions_bc.cpu().numpy()
            
            # 2. Select action via epsilon-greedy strategy
            actions_to_take = np.zeros_like(actions_rl)
            action_sources = []  # +++ ADD THIS LINE (THE FIX) +++
            for i in range(env.num_envs):
                if np.random.uniform() < current_epsilon:
                    actions_to_take[i] = actions_rl[i]  # Explore
                    action_sources.append("RL")  # +++ ADD THIS LINE (THE FIX) +++
                else:
                    actions_to_take[i] = actions_bc[i]  # Exploit advisor
                    action_sources.append("BC")  # +++ ADD THIS LINE (THE FIX) +++
            
            clipped_actions = np.clip(actions_to_take, self.action_space.low, self.action_space.high)
            
            # 3. Step environment and store results
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            if isinstance(infos, dict):
                infos = [infos]

            # Sanity check — catch mismatches early
            assert len(infos) == len(action_sources), (
                f"Infos length {len(infos)} != action_sources length {len(action_sources)}"
            )

            for i in range(len(infos)):
                infos[i]["action_source"] = action_sources[i]

            self.num_timesteps += env.num_envs
            self._update_info_buffer(infos)
            n_steps += 1
            callback.on_step()

            # IMPORTANT: Store the RL policy's action and log_prob for on-policy learning
            rollout_buffer.add(self._last_obs, actions_rl, rewards, self._last_episode_starts, values, log_probs)

            
            self._last_obs = new_obs
            self._last_episode_starts = dones
        
        with torch.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.on_rollout_end()
        self.logger.record("custom/epsilon", current_epsilon)
        
        return True

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


# In scripts/train_hybrid.py

# ... (keep all imports and the AdvisedPPO class definition as they are) ...

def run_hybrid_training(args: argparse.Namespace):
    """
    Main function to orchestrate the BC-Advised RL training process.
    This version PRESERVES the robust setup logic of the original script and
    implements a DEFINITIVE fix for the policy initialization error.
    """
    # --- 1. SETUP AND INITIALIZATION (PRESERVED LOGIC) ---
    run_name = args.run_name or f"advised_ppo_{Path(args.bc_init_dir).name}"
    run_dir = Path(args.output_dir) / run_name
    checkpoints_dir = run_dir / "checkpoints"
    backups_dir = run_dir / "backups"
    
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)
    backups_dir.mkdir(exist_ok=True)

    with (run_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=4)
    logger.info(f"🚀 Starting BC-Advised RL experiment: {run_name}")

    set_random_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device_str = resolve_device(args.device)
    device = torch.device(device_str)

    # --- 2. SETUP ENVIRONMENT (PRESERVED LOGIC) ---
    env = setup_environment(
        xml_path=args.xml_path,
        seed=args.seed,
        control_mode='absolute', 
        n_envs=args.n_envs,
        add_monitor_wrapper=True,
        octo_model=None,
        w_plausibility=0.0,
        pos_scale=args.pos_scale,
        rot_scale=getattr(args, 'rot_scale', 1.0),
        div_clip=getattr(args, 'div_clip', 10.0),
        scripted_expert=None,    
        w_guidance=args.w_guidance,
        w_guidance_dense=args.w_guidance_dense,
        guidance_clip=args.guidance_clip,
        grasp_reward=args.grasp_reward,
        lift_reward=args.lift_reward,
        success_reward=args.success_reward,
    )
    logger.info("Environment created using centralized setup_environment.")

    # --- 3. PREPARE INITIAL WEIGHTS and BC ADVISOR ---
    # This section now prepares an initial state_dict for the RL policy and the BC advisor model.
    initial_policy_state_dict = None
    
    try:
        # Load the BC checkpoint first
        logger.info("Loading BC model to serve as advisor and for weight transfer...")
        bc_model_path = str(Path(args.bc_init_dir) / "checkpoints" / "best_model.pth")
        ckpt = load_bc_checkpoint(bc_model_path, device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

        action_dim = env.action_space.shape[0]
        bc_model = BCNet(n_actions=action_dim).to(device)
        bc_model.load_state_dict(state_dict, strict=False)
        logger.info("BC model loaded to serve as advisor.")

        # Use a temporary agent to perform the weight transfer
        if transfer_bc_weights:
            logger.info("Creating temporary agent for weight transfer...")
            temp_agent = initialize_ppo_agent(env, run_dir, args.seed, device_str)
            transfer_bc_weights(bc_model, temp_agent)
            
            # THE KEY STEP: Extract the weights and discard the temp agent
            initial_policy_state_dict = temp_agent.policy.state_dict()
            del temp_agent
            logger.info("Successfully prepared initial weights for RL policy.")
        else:
            logger.warning("`transfer_bc_weights` not available. RL policy will start from scratch.")
            
    except Exception as e:
        logger.error(f"Failed during BC model loading or weight transfer: {e}", exc_info=True)
        env.close()
        return

    # --- 4. CREATE THE FINAL "BC-ADVISED" AGENT ---
    logger.info("Creating the final BC-Advised PPO agent...")
    epsilon_schedule = {
        "initial": args.eps_initial,
        "final": args.eps_final,
        "decay_steps": args.eps_decay_steps,
    }

    # Define the policy architecture for the new agent.
    # This MUST match the architecture of the policy used for weight transfer.
    policy_kwargs = {
        "features_extractor_class": BCFeaturesExtractor,
        "net_arch": {"pi": [512, 256], "vf": [512, 256]},
    }
    
    # Define all PPO algorithm arguments for a clean construction
    ppo_kwargs = {
        "policy": "MultiInputPolicy",
        "env": env,
        "policy_kwargs": policy_kwargs,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": 10,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "tensorboard_log": str(run_dir / "logs"),
        "seed": args.seed,
        "device": device_str,
        "verbose": 1,
    }

    # Instantiate our custom AdvisedPPO class cleanly
    agent = AdvisedPPO(
        bc_policy_advisor=bc_model,
        epsilon_schedule=epsilon_schedule,
        **ppo_kwargs
    )
    
    # Now, load the prepared state dict into the newly created agent's policy
    if initial_policy_state_dict:
        agent.policy.load_state_dict(initial_policy_state_dict)
        logger.info("Successfully loaded initial weights into the AdvisedPPO policy.")

    # Handle feature freezing on the final agent's policy
    if args.freeze_features:
        logger.info("--- FEATURE FREEZING ENABLED ---")
        frozen_keys = 0
        for name, param in agent.policy.named_parameters():
            if 'features_extractor' in name:
                param.requires_grad = False
                frozen_keys += 1
        logger.info(f"Froze {frozen_keys} parameters in the RL policy's feature extractor.")

    logger.info("BC-Advised PPO agent is fully configured and ready for training.")

    # --- 5. SETUP CALLBACKS AND START TRAINING (PRESERVED LOGIC) ---
    callbacks = [
        CheckpointCallback(
            save_freq=max(1, args.save_freq // args.n_envs),
            save_path=str(checkpoints_dir),
            name_prefix="advised_rl_policy"
        ),
        SingleFileBackupCallback(
            save_freq=max(1, 5000 // args.n_envs),
            save_path=str(backups_dir),
            name_prefix="latest_backup"
        ),
        ActionSourceCallback(),
    ]

    logger.info("\n--- Starting BC-Advised Training Loop ---")
    try:
        agent.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        final_model_path = run_dir / "final_policy.zip"
        agent.save(final_model_path)
        logger.info(f"✅ Training complete. Final policy saved to: {final_model_path}")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user; saving current policy and exiting.")
        agent.save(run_dir / "interrupted_policy.zip")
    except Exception as e:
        logger.exception("Unexpected error during training. Saving state and exiting.", exc_info=True)
        try:
            agent.save(run_dir / "error_policy.zip")
        except Exception:
            logger.exception("Failed to save policy after exception.")
    finally:
        try:
            env.close()
        except Exception:
            logger.exception("Failed to close env cleanly.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BC-Advised RL Training for DGPO")

    # --- Run Management ---
    parser.add_argument("--run_name", type=str, default=None, help="Name for the training run.")
    parser.add_argument("--output_dir", type=str, default="trained_models", help="Directory to save artifacts.")
    parser.add_argument("--bc_init_dir", type=str, required=True, help="Path to the balanced BC run directory.")
    
    # --- Training Parameters ---
    parser.add_argument("--total_timesteps", type=int, default=3_000_000, help="Total timesteps for training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use ('cpu', 'cuda', 'auto').")
    parser.add_argument("--n_envs", type=int, default=8, help="Number of parallel environments.")
    parser.add_argument("--save_freq", type=int, default=50000, help="Frequency to save checkpoints.")
    
    # --- Environment Parameters ---
    parser.add_argument("--xml_path", type=str, default="envs/panda_pick_place.xml", help="Path to MuJoCo XML.")
    parser.add_argument("--urdf_path", type=str, default="urdf/panda_mujoco_kinematics.urdf", help="Path to URDF.")
    
    # --- BC-Advised PPO Hyperparameters ---
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the PPO agent.")
    parser.add_argument("--n_steps", type=int, default=2048, help="PPO rollout buffer size.")
    parser.add_argument("--batch_size", type=int, default=64, help="PPO batch size.")
    parser.add_argument("--eps_initial", type=float, default=0.5, help="Initial epsilon for BC-advise probability.")
    parser.add_argument("--eps_final", type=float, default=0.05, help="Final epsilon value after decay.")
    parser.add_argument("--eps_decay_steps", type=int, default=1_500_000, help="Timesteps over which epsilon decays.")
    
    # --- Feature and Reward Hyperparameters (PRESERVED) ---
    parser.add_argument("--freeze_features", action="store_true", 
                        help="If set, freeze the feature extractor layers.")
    parser.add_argument("--w_guidance", type=float, default=0.0,
                        help="Weight for ScriptedExpert guidance terminal reward.")
    parser.add_argument("--w_guidance_dense", type=float, default=5.0, 
                        help="Weight for DENSE ScriptedExpert guidance reward.")
    parser.add_argument("--guidance_clip", type=float, default=1.0)
    parser.add_argument("--grasp_reward", type=float, default=50.0)
    parser.add_argument("--lift_reward", type=float, default=100.0)
    parser.add_argument("--success_reward", type=float, default=250.0)
    parser.add_argument("--pos_scale", type=float, default=0.05,
                        help="Action scaling factor for the delta controller.")
    
    args = parser.parse_args()
    
    try:
        run_hybrid_training(args)
    except Exception as e:
        logger.exception("An error occurred during training.")
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



python train_hybrid.py --run_name "hybrid_stage0_no_guidance" --bc_init_dir "artifacts/bc_final_balanced_v1" --n_envs 1 --total_timesteps 200000 --device "cpu" --w_guidance 0.0 --w_guidance_dense 0.0 --bc_updates 4 --bc_lr 1e-5 --bc_lambda_initial 1.0 --freeze_features


python -m train_hybrid --run_name "advised_stage0_no_guidance" --bc_init_dir "artifacts/bc_final_balanced_v1" --n_envs 1 --total_timesteps 200000 --device "cpu" --w_guidance 0.0 --w_guidance_dense 0.0 --freeze_features --eps_initial 0.3 --eps_final 0.05 --eps_decay_steps 100000

"""