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
from typing import Optional, Any, Dict, Tuple 
import gymnasium
import gymnasium as gym
import numpy as np
import torch
from envs.panda_env import PandaEnv
from utils.rl_reward_wrapper import RLRewardWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from octo.model.octo_model import OctoModel
from gymnasium import spaces 
from models.custom_sb3_extractor import BCFeaturesExtractor 
from utils.obs_adapters import OctoToSB3Adapter
from pathlib import Path
import sys
import json
try:
    from utils.transfer_bc_to_ppo import transfer_bc_weights
except Exception:
    try:
        from utils.transfer_bc_weights import transfer_bc_weights
    except Exception:
        transfer_bc_weights = None  # we'll check before calling

from models.bc_policy import BCNet

# Configure logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("dgpo.run_experiment")

# In run_experiment.py, add this new wrapper class


def _resize_hwc(img_hwc: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    Robust, anti-aliased downsample. Prefers OpenCV; falls back to PIL.
    """
    try:
        import cv2
        return cv2.resize(img_hwc, (out_w, out_h), interpolation=cv2.INTER_AREA)
    except ImportError:
        try:
            from PIL import Image
            pil_img = Image.fromarray(img_hwc)
            pil_img = pil_img.resize((out_w, out_h), resample=Image.Resampling.BOX)
            return np.asarray(pil_img, dtype=np.uint8)
        except ImportError as e:
            raise RuntimeError("Install 'opencv-python' or 'Pillow' for image downsampling.") from e

def resolve_device(device_arg: str) -> str:
    """Resolve 'auto' to 'cuda' if available else 'cpu' and validate."""
    if device_arg is None or device_arg.lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


class DownsampleImageWrapper(gym.ObservationWrapper):
    """Downsample selected image keys (CHW, uint8) to a target (H, W)."""
    def __init__(self, env, resolution_mapping: Dict[str, Tuple[int, int]]):
        super().__init__(env)
        self.resolution_mapping = dict(resolution_mapping)
        new_spaces: Dict[str, spaces.Space] = env.observation_space.spaces.copy()
        for key, sp in env.observation_space.spaces.items():
            if key in self.resolution_mapping and isinstance(sp, spaces.Box) and sp.dtype == np.uint8 and len(sp.shape) == 3:
                c, _, _ = sp.shape
                tgt_h, tgt_w = self.resolution_mapping[key]
                new_spaces[key] = spaces.Box(low=0, high=255, shape=(c, tgt_h, tgt_w), dtype=np.uint8)
        self.observation_space = spaces.Dict(new_spaces)

    def observation(self, obs):
        for key, val in obs.items():
            if key in self.resolution_mapping:
                c, h, w = val.shape
                tgt_h, tgt_w = self.resolution_mapping[key]
                hwc = np.transpose(val, (1, 2, 0))
                resized_hwc = _resize_hwc(hwc, tgt_h, tgt_w)
                if resized_hwc.ndim == 2: resized_hwc = resized_hwc[..., np.newaxis]
                obs[key] = np.transpose(resized_hwc, (2, 0, 1))
        return obs



def setup_environment(
    xml_path: str,
    seed: int,
    n_envs: int = 1,
    octo_model: Optional[Any] = None,
    w_plausibility: float = 0.1,
    pos_scale: float = 0.05,
    rot_scale: float = 1.0,
    div_clip: float = 10.0,
    enable_downsample: bool = False,
    primary_res: Tuple[int, int] = (128, 128),
    wrist_res: Tuple[int, int] = (96, 96),
):
    """
    Create a vectorized environment with the correct wrapper order, using the
    centralized OctoToSB3Adapter to handle all necessary transformations.
    """
    def make_env():
        # 1. Create the base environment. It produces nested OCTO-style observations.
        env = PandaEnv(xml_path=xml_path)

        # 2. Apply the reward wrapper. It receives the correct nested obs and can use the OCTO model.
        env = RLRewardWrapper(env, octo_model=octo_model,
                              w_plausibility=w_plausibility,
                              pos_scale=pos_scale, rot_scale=rot_scale, div_clip=div_clip)

        # 3. Apply the SB3 adapter LAST. It handles key filtering, image transposition,
        #    and flattening, preparing the observation perfectly for the PPO agent.
        env = OctoToSB3Adapter(env)

        # 4. (Optional) Apply downsampling after the main adapter.
        if enable_downsample:
            res_map = {
                "image_primary": primary_res,
                "image_wrist": wrist_res,
            }
            env = DownsampleImageWrapper(env, res_map)

        return env


    vec_env = make_vec_env(lambda: make_env(), n_envs=n_envs, seed=seed)
    return vec_env

def initialize_ppo_agent(env: gym.vector.VectorEnv, run_name: str, seed: int, device_str: str) -> PPO:
    """
    Initializes a new PPO agent with a standard configuration.
    """
    policy_kwargs = {
        "features_extractor_class": BCFeaturesExtractor,
        "net_arch": {"pi": [512, 256], "vf": [512, 256]}
    }
    
    # Define the PPO hyperparameters here.
    # In the future, these could be exposed as CLI arguments in `args`.
    ppo_config = {
        "verbose": 1,
        "n_steps": 2048,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "device": device_str,
        "tensorboard_log": os.path.join("trained_models", run_name),
        "policy_kwargs": policy_kwargs,
        "seed": seed
    }
    
    logger.info("Initializing new PPO agent with config:")
    for key, val in ppo_config.items():
        logger.info(f"  {key}: {val}")

    return PPO("MultiInputPolicy", env, **ppo_config)

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
    resume_from: Optional[str],
    output_dir: str,
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
    enable_downsample: bool = False,
    primary_res: Tuple[int, int] = (128, 128),
    wrist_res: Tuple[int, int] = (96, 96),
    resume_dir: Optional[str] = None,

):
    if resume_dir:
        run_dir = Path(resume_dir)
        # Load the original config and overwrite the necessary local variables for this run
        config_path = run_dir / "config.json"
        logger.info(f"RESUME mode: Loading config from {config_path}")
        if not config_path.is_file():
            logger.critical(f"Resume failed: config.json not found in {run_dir}"); sys.exit(1)
        with config_path.open("r") as f:
            original_args = json.load(f)
        
        # Overwrite key parameters for the resumed run
        xml_path = original_args.get("xml_path", xml_path)
        seed = original_args.get("seed", seed)
        run_name = original_args.get("run_name") # Use the original name
        # Find the latest checkpoint
        checkpoints = sorted(list((run_dir / "checkpoints").glob("*.zip")))
        if not checkpoints:
            logger.critical(f"Resume failed: No .zip checkpoints found in {run_dir / 'checkpoints'}"); sys.exit(1)
        resume_from = str(checkpoints[-1])
        bc_model_path = None # Disable BC init
    else:
        run_name = run_name or time.strftime("%Y%m%d-%H%M%S")
        run_dir = Path(output_dir) / run_name
        checkpoints_dir = run_dir / "checkpoints"
        logs_dir = run_dir / "logs"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        # Save the config
        config_to_save = {k: v for k, v in locals().items() if not k == 'octo_model'}
        with (run_dir / "config.json").open("w") as f:
            json.dump({k: str(v) for k, v in config_to_save.items()}, f, indent=4)
    logger.info("🚀 Starting DGPO-Foundation experiment")
    device_str = resolve_device(device_arg)
    device = torch.device(device_str)
    logger.info(f"Resolved device: {device_str}")
    run_name = time.strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join("trained_models", run_name)
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"All artifacts for this run will be saved in: {save_dir}")
    # reproducibility
    logger.info(f"Setting random seed to {seed} for all libraries.")
    set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # --- NEW: Load the OCTO model for the divergence reward ---
    octo_model = None
    if w_plausibility > 0.0:
        logger.info("Loading OCTO model for divergence reward calculation...")
        try:
            octo_model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")
            logger.info("OCTO model loaded successfully onto CPU.")
        except Exception as e:
            logger.error(f"Could not load OCTO model, divergence reward will be disabled. Error: {e}")

    logger.info("Creating vectorized environment...")
    env = setup_environment(
        xml_path=xml_path,
        seed=seed,
        n_envs=n_envs,
        octo_model=octo_model,
        w_plausibility=w_plausibility,
        pos_scale=pos_scale,
        rot_scale=rot_scale,
        div_clip=div_clip,
        enable_downsample=enable_downsample,
        primary_res=primary_res,
        wrist_res=wrist_res,
    )
    test_obs = env.reset()

    sample_obs = test_obs[0] if isinstance(test_obs, (list, tuple)) else test_obs
    logger.info(f"Sample obs keys: {list(sample_obs.keys())[:5]} ...")
    logger.info({k: (v.shape, v.dtype) for k, v in sample_obs.items()})
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

    if resume_from:
        logger.info(f"Resuming PPO training from checkpoint: {resume_from}")
        
        # We must provide the custom architecture "blueprint" to load the agent correctly.
        policy_kwargs = {
            "features_extractor_class": BCFeaturesExtractor,
            "net_arch": {"pi": [512, 256], "vf": [512, 256]}
        }
        
        # PPO.load restores the model, optimizer, timesteps, etc.
        ppo_agent = PPO.load(
            resume_from,
            env=env,
            custom_objects={"policy_kwargs": policy_kwargs},
        )
        logger.info(f"Agent loaded. Resuming from step {ppo_agent.num_timesteps}.")

    else:
        logger.info("Starting a new training run.")
        logger.info(f"Initializing PPO agent on device {device_str}...")
        ppo_agent = initialize_ppo_agent(env, run_name, seed, device_str)
        logger.info("PPO agent ready")

        # Correctly handle weight transfer from BC if provided
        if bc_model_path:
            logger.info(f"Attempting to load BC checkpoint from: {args.bc_model_path}")
            try:
                state_dict = load_bc_checkpoint(args.bc_model_path, device)
                action_dim = int(act_space.shape[0])
                logger.info(f"Inferred action dimension from environment: {action_dim}")

                logger.info("Instantiating temporary BCNet for weight transfer...")
                bc_net = BCNet(n_actions=action_dim)
                
                # Robust logging for debugging
                model_keys = set(bc_net.state_dict().keys())
                ckpt_keys = set(state_dict.keys())
                missing_keys = model_keys - ckpt_keys
                unexpected_keys = ckpt_keys - model_keys
                if missing_keys:
                    logger.warning(f"BC checkpoint is missing keys: {list(missing_keys)}")
                if unexpected_keys:
                    logger.warning(f"BC checkpoint has unexpected keys: {list(unexpected_keys)}")
                
                # Load weights into the temporary model
                bc_net.load_state_dict(state_dict, strict=False)
                logger.info("Successfully loaded state_dict into temporary BCNet.")
                
                if transfer_bc_weights:
                    bc_net.to(device)
                    transfer_bc_weights(bc_net, ppo_agent)
                    logger.info("Weight transfer from BC model to PPO agent is complete.")
                else:
                    logger.warning("`transfer_bc_weights` utility not available. Skipping transfer.")

            except Exception as e:
                    logger.error(f"Failed during BC model loading or weight transfer: {e}")
                    logger.exception(e)
                    logger.warning("Continuing RL training from scratch (random initialization).")
        else:
            logger.info("No BC model provided — starting RL")

    save_freq_per_env = max(1, save_freq // n_envs)
    logger.info(f"Checkpoint callback configured to save every {save_freq} total timesteps "
                f"({save_freq_per_env} steps per environment).")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_per_env,
        save_path=save_dir,
        name_prefix="dgpo_policy"
    )
    # Start RL training (handle potential API differences)
    logger.info(f"Starting RL fine-tuning for {total_timesteps} timesteps...")
    try:
        import inspect
        reset_timesteps = False if resume_from else True 
        # Check if the `learn` method supports `progress_bar`
        learn_signature = inspect.signature(ppo_agent.learn)
        if "progress_bar" in learn_signature.parameters:
            logger.info(f"Starting training (resuming={not reset_timesteps}) for {total_timesteps} timesteps...")
            ppo_agent.learn(
                total_timesteps=args.total_timesteps,
                callback=checkpoint_callback,
                reset_num_timesteps=reset_timesteps,
                progress_bar=True,
            )
        else:
            ppo_agent.learn(
                total_timesteps=args.total_timesteps,
                callback=checkpoint_callback,
                reset_num_timesteps=reset_timesteps
            )
            
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

def _parse_hw(s: str) -> Tuple[int, int]:
    """Helper to parse HxW string like '128x128' into a tuple."""
    try:
        h, w = map(int, s.lower().split("x"))
        return (h, w)
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid HxW format: '{s}'. Use '128x128'.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DGPO-Foundation RL training")
    run_group = parser.add_mutually_exclusive_group(required=True)

    parser.add_argument("--xml_path", type=str, default="envs/panda_pick_place.xml")
    parser.add_argument("--bc_model_path", type=str, default=None, help="Path to a .pth BC artifact to START a new run.")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a .zip SB3 checkpoint to RESUME a run.")
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    run_group.add_argument("--run_name", type=str,
                         help="Name for a NEW run. A directory will be created in --output_dir.")
    run_group.add_argument("--resume_dir", type=str,
                         help="Path to an existing run directory to RESUME training.")

    # --- Initialization ---
    parser.add_argument("--bc_init_dir", type=str, default=None,
                        help="Path to a COMPLETED BC run directory to initialize weights for a NEW run.")

    parser.add_argument("--save_freq", type=int, default=20_000)
    parser.add_argument("--enable_downsample", action="store_true",
                        help="Enable antialiased downsampling of image observations for the policy.")
    parser.add_argument("--primary_res", type=str, default="128x128",
                        help="Primary camera resolution HxW (e.g., '128x128').")
    parser.add_argument("--wrist_res", type=str, default="96x96",
                        help="Wrist camera resolution HxW (e.g., '96x96').")
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
    parser.add_argument("--output_dir", type=str, default="trained_models")

    args = parser.parse_args()
    if args.resume_dir and args.bc_init_dir:
        raise argparse.ArgumentTypeError("Cannot provide both --resume_dir and --bc_init_dir.")
    
    # This block acts as an "adapter" from the new CLI to the old function signature.
    if args.resume_dir:
        # --- RESUME MODE ---
        logger.info(f"Resume mode activated. Loading from: {args.resume_dir}")
        run_dir = Path(args.resume_dir)
        checkpoints_dir = run_dir / "checkpoints"
        checkpoints = sorted(list(checkpoints_dir.glob("*.zip")))
        if not checkpoints:
            logger.critical(f"Resume failed: No .zip checkpoints found in {checkpoints_dir}")
            sys.exit(1)
            
        # Set the variables for the function call
        bc_model_path_for_func = None
        resume_from_for_func = str(checkpoints[-1])
        run_name_for_func = run_dir.name # Use the existing directory name
        output_dir_for_func = str(run_dir.parent)
    else:
        # --- NEW RUN MODE ---
        if not args.run_name:
            args.run_name = time.strftime("run_%Y%m%d_%H%M%S")
            logger.info(f"No run name provided. Generated run name: {args.run_name}")

        run_name_for_func = args.run_name
        output_dir_for_func = args.output_dir
        resume_from_for_func = None

        if args.bc_init_dir:
            # Find the best BC model checkpoint to pass to the function
            bc_checkpoint_path = Path(args.bc_init_dir) / "checkpoints" / "best_model.pth"
            if not bc_checkpoint_path.is_file():
                logger.error(f"BC checkpoint 'best_model.pth' not found in {bc_checkpoint_path.parent}")
                sys.exit(1)
            bc_model_path_for_func = str(bc_checkpoint_path)
        else:
            logger.warning("No --bc_init_dir provided for new run. Training from random initialization.")
            bc_model_path_for_func = None

    primary_res = _parse_hw(args.primary_res)
    wrist_res = _parse_hw(args.wrist_res)

    run_experiment(
        xml_path=args.xml_path,
        bc_model_path=bc_model_path_for_func,
        resume_from=resume_from_for_func,
        total_timesteps=args.total_timesteps,
        run_name=run_name_for_func,
        resume_dir=args.resume_dir,
        save_freq=args.save_freq,
        output_dir=output_dir_for_func,
        seed=args.seed,
        device_arg=args.device,
        n_envs=args.n_envs,
        w_plausibility=args.w_plausibility,
        pos_scale=args.pos_scale,
        rot_scale=args.rot_scale,
        div_clip=args.div_clip,
        enable_downsample=args.enable_downsample,
        primary_res=primary_res,
        wrist_res=wrist_res,
    )
