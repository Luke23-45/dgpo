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

# weight transfer utility: try to import from the most likely path
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


class FlattenNestedDictObs(gym.ObservationWrapper):
    """
    Flattens a nested Dict observation space into a single-level Dict.
    This is necessary for compatibility with Stable Baselines3.
    """
    def __init__(self, env):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Dict):
            raise ValueError("This wrapper only works on environments with Dict observation spaces.")
        
        self.observation_space = self._flatten_space(env.observation_space)

    def _flatten_space(self, space, prefix=""):
        flat_spaces = {}
        for key, sub_space in space.spaces.items():
            new_prefix = f"{prefix}{key}/" if prefix else f"{key}/"
            if isinstance(sub_space, spaces.Dict):
                flat_spaces.update(self._flatten_space(sub_space, new_prefix))
            else:
                # Remove trailing slash from the key
                flat_spaces[new_prefix[:-1]] = sub_space
        return spaces.Dict(flat_spaces)

    def observation(self, obs):
        return self._flatten_obs(obs)

    def _flatten_obs(self, obs, prefix=""):
        flat_obs = {}
        for key, value in obs.items():
            new_prefix = f"{prefix}{key}/" if prefix else f"{key}/"
            if isinstance(value, dict):
                flat_obs.update(self._flatten_obs(value, new_prefix))
            else:
                flat_obs[new_prefix[:-1]] = value
        return flat_obs


class TransposeImageDict(gym.ObservationWrapper):
    """
    Transposes HWC image observations in a Dict space to CHW format for SB3.
    Keeps dtype as uint8 for SB3 CNN extractor.
    """
    def __init__(self, env):
        super().__init__(env)
        new_spaces = {}
        for key, space in self.observation_space.spaces.items():
            if isinstance(space, spaces.Box) and len(space.shape) == 3:
                # New shape is (C, H, W)
                new_shape = (space.shape[2], space.shape[0], space.shape[1])
                new_spaces[key] = spaces.Box(
                    low=0, high=255, shape=new_shape, dtype=space.dtype
                )
            else:
                new_spaces[key] = space
        self.observation_space = spaces.Dict(new_spaces)

    def observation(self, obs):
        new_obs = {}
        for key, value in obs.items():
            if key in self.observation_space.spaces and isinstance(self.observation_space.spaces[key], spaces.Box) and len(value.shape) == 3:
                # Transpose HWC -> CHW, keep uint8
                new_obs[key] = np.transpose(value, (2, 0, 1))
            else:
                new_obs[key] = value
        return new_obs


class DropKeysWrapper(gym.ObservationWrapper):
    def __init__(self, env, drop_prefixes=()):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Dict)
        self.drop_prefixes = tuple(drop_prefixes)
        kept = {k: v for k, v in env.observation_space.spaces.items()
                if not any(k.startswith(p) for p in self.drop_prefixes)}
        self.observation_space = spaces.Dict(kept)

    def observation(self, obs):
        return {k: v for k, v in obs.items()
                if k in self.observation_space.spaces}

# In file: run_experiment.py

# --- Replace the entire setup_environment function with this definitive version ---

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


class SanitizeDictObs(gym.ObservationWrapper):
    """
    - Expand scalar Box() -> Box(shape=(1,), dtype=float32)
    - Cast all non-image Box spaces to float32
    - Leave image Boxes (3,H,W or H,W,3) as-is (uint8)
    """
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.observation_space, spaces.Dict), "SanitizeDictObs expects Dict space"
        self.observation_space = self._sanitize_space(self.observation_space)

    @staticmethod
    def _is_image_space(space: spaces.Box) -> bool:
        if not isinstance(space, spaces.Box):
            return False
        if space.dtype != np.uint8:
            return False
        if len(space.shape) != 3:
            return False
        c, h, w = space.shape if space.shape[0] in (1, 3, 4) else (None, None, None)
        # allow CHW or HWC; we transpose earlier anyway
        return True

    def _sanitize_space(self, dict_space: spaces.Dict) -> spaces.Dict:
        new_spaces = {}
        for key, sp in dict_space.spaces.items():
            if isinstance(sp, spaces.Dict):
                new_spaces[key] = self._sanitize_space(sp)
                continue
            if isinstance(sp, spaces.Box):
                if self._is_image_space(sp):
                    new_spaces[key] = sp  # leave images unchanged
                else:
                    # ensure at least 1D and float32
                    shape = sp.shape
                    if shape == ():  # scalar -> (1,)
                        shape = (1,)
                    new_spaces[key] = spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=shape,
                        dtype=np.float32
                    )
            elif isinstance(sp, (spaces.MultiBinary, spaces.MultiDiscrete)):
                # leave as is (SB3 can handle), but they will arrive as int/uint8
                new_spaces[key] = sp
            else:
                # fallback: wrap as a 1D float32 Box of size 1
                new_spaces[key] = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                )
        return spaces.Dict(new_spaces)

    def observation(self, obs):
        return self._sanitize_obs(obs, self.observation_space)

    def _sanitize_obs(self, obs_dict, dict_space: spaces.Dict):
        out = {}
        for key, sp in dict_space.spaces.items():
            val = obs_dict[key]
            if isinstance(sp, spaces.Dict):
                out[key] = self._sanitize_obs(val, sp)
                continue
            if isinstance(sp, spaces.Box):
                if self._is_image_space(sp):
                    # ensure numpy array, uint8, correct shape already handled by other wrappers
                    if not isinstance(val, np.ndarray):
                        val = np.asarray(val, dtype=np.uint8)
                    else:
                        val = val.astype(np.uint8, copy=False)
                    out[key] = val
                else:
                    arr = np.asarray(val)
                    # expand scalar to (1,)
                    if arr.shape == ():
                        arr = arr.reshape(1)
                    out[key] = arr.astype(np.float32, copy=False)
            elif isinstance(sp, (spaces.MultiBinary, spaces.MultiDiscrete)):
                out[key] = np.asarray(val)
            else:
                # fallback to float32 (1,)
                arr = np.asarray(val)
                if arr.shape == ():
                    arr = arr.reshape(1)
                out[key] = arr.astype(np.float32, copy=False)
        return out



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
    enable_downsample: bool = False,
    primary_res: Tuple[int, int] = (128, 128),
    wrist_res: Tuple[int, int] = (96, 96),
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
    PPO_CONFIG = {
        "verbose": 1,
        "n_steps": 2048,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "learning_rate": 2.5e-4, # Slightly lower learning rate
        "clip_range": 0.15,      # Tighter clipping range
        "target_kl": 0.02,       # KL divergence target to prevent overly large updates
        "ent_coef": 0.001,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "device": device_str,
        "policy_kwargs": {
            "features_extractor_class": BCFeaturesExtractor,
            "net_arch": {
                "pi": [512, 256], # Policy network
                "vf": [512, 256], # Value network
            }
        }
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
    args = parser.parse_args()

    # normalize bc_model_path: accept 'None' literal
    if args.bc_model_path and args.bc_model_path.lower() == "none":
        args.bc_model_path = None
    def _parse_hw(s: str) -> Tuple[int, int]:
        try:
            h, w = map(int, s.lower().split("x"))
            return (h, w)
        except Exception:
            raise argparse.ArgumentTypeError(f"Invalid HxW format: '{s}'. Use '128x128'.")

    primary_res = _parse_hw(args.primary_res)
    wrist_res = _parse_hw(args.wrist_res)
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
        enable_downsample=args.enable_downsample,
        primary_res=primary_res,
        wrist_res=wrist_res,
    )
