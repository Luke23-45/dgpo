# run_experiment.py
"""
Robust experiment runner for DGPO-Foundation RL fine-tuning.

Improvements vs original:
 - Resolves 'auto' device selection.
 - Safe vec-env creation and seeding.
 - Robust BC checkpoint loading (several formats).
 - Validates action-dim alignment before weight transfer.
 - Creates checkpoint directories and handles SB3 API differences gracefully.
 - Clear logging and error messages for common pitfalls.
"""

import os
import time
import argparse
import logging
import random
from typing import Optional,Any
import gymnasium
import numpy as np
import torch
from envs.panda_env import PandaEnv
from utils.rl_reward_wrapper import RLRewardWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from octo.model.octo_model import OctoModel

# weight transfer utility: try to import from the most likely path
try:
    from utils.weight_transfer import transfer_bc_weights
except Exception:
    try:
        from utils.transfer_bc_to_ppo import transfer_bc_weights
    except Exception:
        transfer_bc_weights = None  # we'll check before calling

from models.bc_policy import BCNet

# Configure logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("dgpo.run_experiment")


def resolve_device(device_arg: str) -> str:
    """Resolve 'auto' to 'cuda' if available else 'cpu' and validate."""
    if device_arg is None or device_arg.lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def setup_environment(
    xml_path: str,
    seed: int,
    n_envs: int = 1,
    octo_model: Optional[Any] = None,
    w_plausibility: float = 0.1,
    pos_scale: float = 0.05,
    rot_scale: float = 1.0,
    div_clip: float = 10.0,
):
    """
    Create a vectorized environment and wrap it with RLRewardWrapper.
    Returns a VecEnv ready for SB3.
    """
    def make_env():
        env = PandaEnv(xml_path=xml_path)
        # Pass the new args to the wrapper
        env = RLRewardWrapper(
            env,
            octo_model=octo_model,
            w_plausibility=w_plausibility,
            pos_scale=pos_scale,
            rot_scale=rot_scale,
            div_clip=div_clip
        )
        return env

    vec_env = make_vec_env(lambda: make_env(), n_envs=n_envs, seed=seed)
    return vec_env


def initialize_ppo_agent(env, ppo_config: dict, seed: int, policy: str):
    """
    Initialize a PPO agent with given config and seed.
    ppo_config is passed directly to PPO() constructor.
    """
    # ensure seed present
    ppo_config = dict(ppo_config)
    ppo_config.setdefault("seed", seed)

    # set unique tensorboard log folder
    tblog = ppo_config.get("tensorboard_log", "./tensorboard_logs")
    run_log_dir = os.path.join(tblog, f"dgpo_run_{int(time.time())}")
    ppo_config["tensorboard_log"] = run_log_dir

    logger.info(f"Initializing PPO agent with '{policy}' (logdir={run_log_dir})") # Use policy in log
    agent = PPO(policy=policy, env=env, **ppo_config) 
    return agent

def load_bc_checkpoint(filepath: str, device: torch.device):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"BC checkpoint not found at: {filepath}")

    try:
        ckpt = torch.load(filepath, map_location='cpu')  # load to CPU first
    except Exception as e:
        raise RuntimeError(f"Failed to torch.load BC checkpoint '{filepath}': {e}")

    # common formats
    if isinstance(ckpt, dict):
        # accept wrappers
        for key in ("model_state_dict", "state_dict", "model_state"):
            if key in ckpt:
                return ckpt[key]
        # if it's already a raw state dict
        if all(isinstance(v, (torch.Tensor,)) for v in ckpt.values()):
            return ckpt

    raise RuntimeError("BC checkpoint loaded but format not recognized. Expected dict with 'model_state_dict' or raw state_dict.")



def run_experiment(
    xml_path: str,
    bc_model_path: Optional[str],
    total_timesteps: int,
    run_name: str,
    save_freq: int,
    seed: int,
    device_arg: str,
    n_envs: int = 1,
    w_plausibility: float = 0.1,
    pos_scale: float = 0.05,
    rot_scale: float = 1.0,
    div_clip: float = 10.0,
):
    logger.info("🚀 Starting DGPO-Foundation experiment")
    device_str = resolve_device(device_arg)
    device = torch.device(device_str)
    logger.info(f"Resolved device: {device_str}")

    # reproducibility
    logger.info(f"Setting random seed to {seed} for all libraries.")
    set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # --- NEW: Load the OCTO model for the divergence reward ---
    logger.info("Loading OCTO model for divergence reward calculation...")
    try:
        # Correctly load the model directly onto the CPU
        octo_device = torch.device("cpu")
        octo_model = OctoModel.load_pretrained(
            "hf://rail-berkeley/octo-small-1.5",
            device=octo_device
        )
        octo_model.eval() # Set to evaluation mode
        logger.info("OCTO model loaded successfully onto CPU.")
    except Exception as e:
        logger.error(f"Could not load OCTO model, divergence reward will be disabled. Error: {e}")
        octo_model = None
    # Make env
    logger.info("Creating vectorized environment...")
    env = setup_environment(
        xml_path=xml_path,
        seed=seed,
        n_envs=n_envs,
        octo_model=octo_model,
        w_plausibility=w_plausibility,
        pos_scale=pos_scale,
        rot_scale=rot_scale,
        div_clip=div_clip
    )
    logger.info("Environment created")
    # --- 2. Validate Environment and Auto-Select Policy ---
    # Access the underlying single environment to check its properties
    single_env = env.envs[0] if hasattr(env, 'envs') and env.envs else env
    obs_space = single_env.observation_space
    act_space = single_env.action_space
    logger.info(f"Validated Environment | Obs Space: {obs_space}, Act Space: {act_space}")

    if not isinstance(act_space, gymnasium.spaces.Box) or len(act_space.shape) != 1:
        raise TypeError(f"Action space must be a 1D Box, but got {type(act_space)} with shape {act_space.shape}")

    # Auto-select the correct policy based on the observation space type
    if isinstance(obs_space, gymnasium.spaces.Dict):
        policy = "MultiInputPolicy"
    elif isinstance(obs_space, gymnasium.spaces.Box):
        policy = "MlpPolicy"
    else:
        raise TypeError(f"Unsupported observation space type: {type(obs_space)}")
    logger.info(f"Auto-selected policy '{policy}' based on observation space type.")
    # PPO config (you can expand or override via function args)
    PPO_CONFIG = {
        "verbose": 1,
        "n_steps": 2048,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.001,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "device": device_str,
    }

    logger.info(f"Initializing PPO agent on device {device_str}...")
    ppo_agent = initialize_ppo_agent(env, PPO_CONFIG, seed, policy=policy)
    logger.info("PPO agent ready")

    # Weight transfer from BC if provided
    if bc_model_path:
        logger.info(f"BC model path provided: {bc_model_path}")
        try:
            logger.info(f"Attempting to load BC checkpoint from: {bc_model_path}")
            state_dict = load_bc_checkpoint(bc_model_path, device)
            
            action_dim = int(act_space.shape[0])
            logger.info(f"Inferred action dimension from environment: {action_dim}")

            # Instantiate BCNet on CPU first to safely load the state_dict
            bc_net = BCNet(n_actions=action_dim)
            
            # Log missing/unexpected keys for better debugging
            model_keys = set(bc_net.state_dict().keys())
            ckpt_keys = set(state_dict.keys())
            missing_keys = model_keys - ckpt_keys
            unexpected_keys = ckpt_keys - model_keys
            if missing_keys:
                logger.warning(f"BC checkpoint is missing keys: {list(missing_keys)}")
            if unexpected_keys:
                logger.warning(f"BC checkpoint has unexpected keys: {list(unexpected_keys)}")
            
            bc_net.load_state_dict(state_dict, strict=False)
            logger.info("Successfully loaded state_dict into BCNet instance.")

            if transfer_bc_weights is None:
                logger.warning("`transfer_bc_weights` utility not available. Skipping weight transfer.")
            else:
                # Move BC model to the target device before transfer
                bc_net.to(device)
                logger.info(f"Transferring weights from BCNet (on {device}) to PPO policy...")
                transfer_bc_weights(bc_net, ppo_agent)
                logger.info("Weight transfer complete.")

        except Exception as e:
            logger.error(f"Failed during BC model loading or weight transfer: {e}")
            logger.exception(e)
            logger.warning("Continuing RL training from scratch (random initialization).")
    else:
        logger.info("No BC model provided — starting RL from scratch.")

    # Callbacks and checkpointing
    save_dir = os.path.join("trained_models", run_name)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(save_freq=max(1, save_freq), save_path=save_dir, name_prefix="dgpo_policy")

    # Start RL training (handle potential API differences)
    logger.info(f"Starting RL fine-tuning for {total_timesteps} timesteps...")
    try:
        import inspect
        # Check if the `learn` method supports `progress_bar`
        learn_signature = inspect.signature(ppo_agent.learn)
        if "progress_bar" in learn_signature.parameters:
            logger.info("SB3 version supports `progress_bar`. Training with progress bar.")
            ppo_agent.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, progress_bar=True)
        else:
            logger.info("SB3 version does not support `progress_bar`. Training without it.")
            ppo_agent.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
            
    except Exception as e:
        logger.exception("An error occurred during training.")
        raise e # Re-raise the exception after logging

    # Final save
    final_model_path = os.path.join(save_dir, "final_policy.zip")
    ppo_agent.save(final_model_path)
    logger.info(f"Training complete. Final policy saved to: {final_model_path}")

    # Close envs
    try:
        env.close()
    except Exception as e:
        logger.warning(f"Error closing environment: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DGPO-Foundation RL training")
    parser.add_argument("--xml_path", type=str, default="envs/panda_pick_place.xml")
    parser.add_argument("--bc_model_path", type=str, default="trained_models/policy_pretrained_bc.pth")
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    parser.add_argument("--run_name", type=str, default=f"dgpo_run_{int(time.time())}")
    parser.add_argument("--save_freq", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--w_plausibility", type=float, default=0.1, 
                        help="Weight for the OCTO divergence terminal reward.")
    parser.add_argument("--pos_scale", type=float, default=0.05,
                        help="Typical position error scale (in meters) for divergence reward.")
    parser.add_argument("--rot_scale", type=float, default=1.0,
                        help="Weight for the rotation component of the divergence reward.")
    parser.add_argument("--div_clip", type=float, default=10.0,
                        help="Maximum value to clip the raw divergence score before weighting.")
    args = parser.parse_args()

    # normalize bc_model_path: accept 'None' literal
    if args.bc_model_path and args.bc_model_path.lower() == "none":
        args.bc_model_path = None

    run_experiment(
        xml_path=args.xml_path,
        bc_model_path=args.bc_model_path,
        total_timesteps=args.total_timesteps,
        run_name=args.run_name,
        save_freq=args.save_freq,
        seed=args.seed,
        device_arg=args.device,
        n_envs=args.n_envs,
        w_plausibility=args.w_plausibility,
        pos_scale=args.pos_scale,
        rot_scale=args.rot_scale,
        div_clip=args.div_clip,
    )
