# FILE: eval/vidhis.py
# SOTA Inference Script for Vision-Diffusion Hierarchical System (ViDHiS)

import os
import time
import logging
import random
from pathlib import Path
import sys
from typing import Dict, Any, Tuple, Optional, List
# from utils.data_utils import ObsHistoryBuffer
# Add project root for imports if necessary (adjust relative path as needed)
from utils.controller_dataset import HierarchicalControllerDataset
try:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except IndexError:
    print("Warning: Could not automatically determine project root. "
          "Ensure script is run from project root or PYTHONPATH is set.")

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import gymnasium as gym
from torchvision import transforms
import imageio # For saving videos
import matplotlib.pyplot as plt
import cv2 # For resizing if needed, ensure opencv-python-headless installed
from torch import nn
import json
# Project Imports
try:
    # Assuming standard project structure from previous steps
    from models.planner import VisualPlannerDiffusion
    # The 'Controller' is likely your adapted DiffusionPolicy
    from models.diffusion_policy import DiffusionPolicy, NoiseSchedulerConfig
    # Need ObsHistoryBuffer from train_rl.py context
    from scripts.train_rl import ObsHistoryBuffer # Adjust path if moved
    from envs.panda_env import PandaEnv
    # Need seeding function
    from scripts.train_rl import set_seed # Adjust path if moved
except ImportError as e:
    print(f"Error importing project modules: {e}. "
          "Ensure PYTHONPATH is correct or run from project root.")
    sys.exit(1)

log = logging.getLogger(__name__)



class OraclePlanner:
    """
    A "perfect" planner that provides ground-truth future images from the dataset.
    This replaces the neural network planner for diagnostic purposes.
    """
    def __init__(self, dataset, episode_idx: int, cfg: DictConfig, device: torch.device):
        self.dataset = dataset
        self.episode_idx = episode_idx
        self.cfg = cfg
        self.device = device
        self.subgoal_horizon_k = dataset.subgoal_horizon_k
        
        # Pre-load the full image array for the target episode for maximum speed.
        ep_meta = self.dataset.expert_reader.episode_metadata[self.episode_idx]
        img_primary_meta = ep_meta["modalities"]["image_primary"]
        self.full_image_array = self.dataset.expert_reader._get_full_modality_array(
            img_primary_meta["key"],
            img_primary_meta["compression"],
            img_primary_meta["dtype"],
            tuple(img_primary_meta["shape"])
        )
        self.episode_length = ep_meta['length']
        log.info(f"[OraclePlanner] Initialized for Episode {self.episode_idx} (Length: {self.episode_length})")

    def sample(self, current_timestep_t: int) -> torch.Tensor:
        """
        Retrieves the ground-truth subgoal image from the dataset.
        """
        # Calculate the target timestep for the subgoal
        subgoal_t = current_timestep_t + self.subgoal_horizon_k
        
        # Clamp the timestep to be within the valid range of the episode
        subgoal_t = min(subgoal_t, self.episode_length - 1)
        
        # Get the ground-truth image from the pre-loaded array
        gt_subgoal_np = self.full_image_array[subgoal_t]
        
        # Preprocess the numpy image into a batched tensor
        # NOTE: This uses the existing `preprocess_obs_image` helper
        subgoal_tensor = preprocess_obs_image(gt_subgoal_np, self.cfg, self.device)
        
        return subgoal_tensor.unsqueeze(0) # Add batch dimension

def reset_env_to_expert_state(env: PandaEnv, 
                              dataset, 
                              episode_idx: int, 
                              timestep_t: int) -> Dict[str, np.ndarray]:
    """
    Resets the simulation environment to a specific state from an expert trajectory.
    """
    # Get the state from the dataset's underlying reader
    ep_meta = dataset.expert_reader.episode_metadata[episode_idx]
    
    # --- START: ROBUST PATCH ---
    #
    # The key for the robot's state in our dataset is 'proprio', not 'state'.
    # We will use this to set the robot's joint configuration.
    # Note: This does not set the state of other objects like the cube.
    #
    if "proprio" not in ep_meta["modalities"]:
        raise KeyError("Dataset is missing the 'proprio' modality needed to reset the environment state.")
        
    state_meta = ep_meta["modalities"]["proprio"] # <-- CHANGED FROM "state"
    #
    # --- END: ROBUST PATCH ---

    full_state_array = dataset.expert_reader._get_full_modality_array(
        state_meta["key"],
        state_meta["compression"],
        state_meta["dtype"],
        tuple(state_meta["shape"])
    )
    target_state = full_state_array[timestep_t]
    
    # Use the environment's specific method to set its state
    # This requires your PandaEnv to have a `set_state` method
    if not hasattr(env.unwrapped, 'set_state'):
        raise NotImplementedError("The PandaEnv must have a `set_state(state)` method to support trajectory replay.")
        
    obs_dict, info = env.reset() # Do a standard reset first
    env.unwrapped.set_state(target_state)
    
    # After setting the state, we need to get the corresponding observation
    # We can do this by calling a `get_obs` method on the environment
    if not hasattr(env.unwrapped, 'get_obs'):
        raise NotImplementedError("The PandaEnv must have a `get_obs()` method to get observations after setting state.")
    
    obs_dict = env.unwrapped.get_obs()
    log.info(f"Environment reset to state of Episode {episode_idx} at Timestep {timestep_t}.")
    return obs_dict

# FILE: eval/run_vidhis_oracle.py

# --- START: ROBUST PATCH 2 (Model Loading) ---
# Delete the old load_models and load_and_preprocess_goal functions.
# Replace them with this single function.

def load_controller(cfg: DictConfig, device: torch.device) -> nn.Module:
    """
    Loads the Controller model from a checkpoint, robustly handling cases
    where the checkpoint's saved config might be incomplete.
    """
    log.info("--- Loading SOTA Controller Model ---")
    controller_path = Path(cfg.controller_ckpt_path)
    if not controller_path.exists():
        raise FileNotFoundError(f"Controller checkpoint not found: {controller_path}")

    log.info(f"Loading Controller from: {controller_path}")
    controller_ckpt = torch.load(controller_path, map_location=device, weights_only=False)
    
    ckpt_config = controller_ckpt['config']
    model_cfg = ckpt_config['model']
    scheduler_cfg_dict = ckpt_config['scheduler']

    # --- START: ROBUST PATCH 2 ---
    #
    # Implement the fallback logic for proprio_dim.
    # First, try to get it from the checkpoint's config.
    # If it's not there, get it from the current evaluation config (`cfg`).
    #
    proprio_dim = model_cfg.get('proprio_dim')
    if proprio_dim is None:
        log.warning("`proprio_dim` not found in checkpoint config. "
                    "Falling back to the value from the current evaluation config.")
        proprio_dim = cfg.model.controller.proprio_dim
    #
    # --- END: ROBUST PATCH 2 ---
    if 'schedule_type' in scheduler_cfg_dict:
        log.warning("Found legacy 'schedule_type' key in checkpoint config. Migrating to 'schedule'.")
        scheduler_cfg_dict['schedule'] = scheduler_cfg_dict.pop('schedule_type')
    #

    controller = DiffusionPolicy(
        proprio_dim=proprio_dim, # Use the robustly determined value
        H_o=model_cfg['observation_horizon'],
        H_a=model_cfg['action_horizon'],
        action_dim=model_cfg['action_dim'],
        image_feat_dim=model_cfg['image_feat_dim'],
        d_model=model_cfg['d_model'],
        denoiser_layers=model_cfg['denoiser_layers'],
        denoiser_heads=model_cfg['denoiser_heads'],
        scheduler_cfg=NoiseSchedulerConfig(**scheduler_cfg_dict),
        device=device
    )
    
    # Load the state dict for the EMA model, which is used for inference.
    if controller.ema and 'ema_state_dict' in controller_ckpt:
        controller.ema.load_state_dict(controller_ckpt['ema_state_dict'])
        log.info("Controller EMA weights loaded successfully.")
    else:
        log.warning("EMA weights not found in checkpoint. Using non-EMA weights.")
        controller.load_state_dict(controller_ckpt['policy_state_dict'])
    
    controller.eval()
    log.info("Controller loaded successfully.")

    # Apply torch.compile if configured
    if cfg.inference.use_torch_compile and hasattr(torch, "compile"):
        log.info(f"Applying torch.compile (mode='{cfg.inference.torch_compile_mode}')...")
        controller = torch.compile(controller, mode=cfg.inference.torch_compile_mode)

    return controller

# --- END: ROBUST PATCH 2 ---

def load_models(cfg: DictConfig, device: torch.device) -> Tuple[nn.Module, nn.Module]:
    """
    Loads Planner and Controller models from checkpoints using a SOTA, robust
    method that infers architecture from the checkpoint's hyperparameters.
    """
    log.info("--- Loading SOTA Models ---")

    # --- Load Planner ---
    planner_path = Path(cfg.planner_ckpt_path)
    if not planner_path.exists():
        raise FileNotFoundError(f"Planner checkpoint not found: {planner_path}")
    
    log.info(f"Loading Planner from: {planner_path}")
    planner_ckpt = torch.load(planner_path, map_location=device)
    
    # SOTA: Instantiate the model using hyperparameters saved in the checkpoint.
    # This ensures the architecture is identical to the one used during training.
    planner_hparams = planner_ckpt['hyper_parameters']
    planner = VisualPlannerDiffusion(**planner_hparams).to(device)
    
    # Robustly load the state dict, stripping the "model." prefix added by Lightning.
    planner_state_dict = {k.replace("model.", ""): v for k, v in planner_ckpt['state_dict'].items()}
    planner.load_state_dict(planner_state_dict)
    planner.eval()
    log.info("Planner loaded successfully.")

    # --- Load Controller ---
    controller_path = Path(cfg.controller_ckpt_path)
    if not controller_path.exists():
        raise FileNotFoundError(f"Controller checkpoint not found: {controller_path}")

    log.info(f"Loading Controller from: {controller_path}")
    controller_ckpt = torch.load(controller_path, map_location=device)
    
    # SOTA: Instantiate using the config saved in our manual checkpoint.
    controller_cfg = controller_ckpt['config']
    model_cfg = controller_cfg['model']
    scheduler_cfg_dict = controller_cfg['scheduler']

    controller = DiffusionPolicy(
        proprio_dim=model_cfg['proprio_dim'],
        H_o=model_cfg['observation_horizon'],
        H_a=model_cfg['action_horizon'],
        action_dim=model_cfg['action_dim'],
        image_feat_dim=model_cfg['image_feat_dim'],
        d_model=model_cfg['d_model'],
        denoiser_layers=model_cfg['denoiser_layers'],
        denoiser_heads=model_cfg['denoiser_heads'],
        scheduler_cfg=NoiseSchedulerConfig(**scheduler_cfg_dict),
        device=device
    )
    
    # Robustly load the state dicts for both the policy and the EMA model.
    controller.load_state_dict(controller_ckpt['policy_state_dict'])
    if controller.ema and 'ema_state_dict' in controller_ckpt:
        controller.ema.load_state_dict(controller_ckpt['ema_state_dict'])
        log.info("Controller EMA weights loaded successfully.")
    
    controller.eval()
    log.info("Controller loaded successfully.")

    # Apply torch.compile if configured
    if cfg.inference.use_torch_compile and hasattr(torch, "compile"):
        log.info(f"Applying torch.compile (mode='{cfg.inference.torch_compile_mode}')...")
        planner = torch.compile(planner, mode=cfg.inference.torch_compile_mode)
        controller = torch.compile(controller, mode=cfg.inference.torch_compile_mode)

    return planner, controller



def preprocess_obs_history(obs_history_dict: Dict[str, np.ndarray],
                           cfg: DictConfig,
                           device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Takes a dictionary of NumPy observation histories (H, ...), converts them
    to batched PyTorch Tensors (1, H, ...), and applies training-time normalization.
    """
    batched_tensors = {}
    
    # Define image transforms
    img_size = tuple(cfg.data_preprocessing.image_size)
    img_mean = tuple(cfg.data_preprocessing.img_mean)
    img_std = tuple(cfg.data_preprocessing.img_std)
    transform = transforms.Compose([
        transforms.ToTensor(), # HWC:uint8 -> CHW:float[0,1]
        transforms.Resize(img_size, antialias=True),
        transforms.Normalize(mean=img_mean, std=img_std)
    ])

    for key, value in obs_history_dict.items():
        if 'image' in key:
            # Handle image history (H, H_img, W_img, C)
            # Apply transform to each image in the history
            processed_imgs = [transform(img) for img in value]
            tensor = torch.stack(processed_imgs, dim=0)
        else:
            # Handle proprioception history (H, D_proprio)
            tensor = torch.from_numpy(value).float()
        
        # Add a batch dimension and move to the target device
        batched_tensors[key] = tensor.unsqueeze(0).to(device)
        
    return batched_tensors

def load_and_preprocess_goal(cfg: DictConfig, device: torch.device) -> torch.Tensor:
    """Loads and preprocesses the final goal image."""
    goal_img_path = Path(cfg.evaluation.goal_image_path)
    if not goal_img_path.exists():
        raise FileNotFoundError(f"Goal image not found: {goal_img_path}")
    try:
        # Load using OpenCV (robust)
        img_bgr = cv2.imread(str(goal_img_path))
        if img_bgr is None: raise IOError("Failed to load image.")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # HWC, uint8

        # Apply preprocessing (Resize, ToTensor, Normalize)
        # Use transforms consistent with training
        img_size = tuple(cfg.data_preprocessing.image_size)
        img_mean = tuple(cfg.data_preprocessing.img_mean)
        img_std = tuple(cfg.data_preprocessing.img_std)

        transform = transforms.Compose([
            transforms.ToTensor(), # HWC:uint8 -> CHW:float[0,1]
            transforms.Resize(img_size, antialias=True),
            transforms.Normalize(mean=img_mean, std=img_std)
        ])
        img_tensor = transform(img_rgb).to(device)
        log.info(f"Loaded and preprocessed goal image from {goal_img_path}")
        return img_tensor.unsqueeze(0) # Add batch dimension
    except Exception as e:
        log.exception(f"Error processing goal image: {e}")
        raise

# FILE: scripts/vidhis.py

def preprocess_obs_image(img_np: np.ndarray, cfg: DictConfig, device: torch.device) -> torch.Tensor:
    """Applies preprocessing to a NumPy observation image."""
    # Assumes img_np is HWC, uint8
    
    # --- START: ROBUST PATCH ---
    #
    # Add .copy() to the numpy array before passing it to transforms.
    # This resolves the "negative stride" error by creating a new, C-contiguous
    # array in memory that torch.from_numpy can handle.
    #
    img_np_copy = img_np.copy()
    #
    # --- END: ROBUST PATCH ---

    transform = transforms.Compose([
        transforms.ToTensor(), # HWC:uint8 -> CHW:float[0,1]
        transforms.Resize(tuple(cfg.data_preprocessing.image_size), antialias=True),
        transforms.Normalize(mean=tuple(cfg.data_preprocessing.img_mean),
                             std=tuple(cfg.data_preprocessing.img_std))
    ])
    
    # Pass the copied array to the transform
    return transform(img_np_copy).to(device)


def denormalize_image(img_tensor: torch.Tensor, cfg: DictConfig) -> np.ndarray:
    """Converts a normalized CHW tensor back to HWC uint8 NumPy image."""
    mean = torch.tensor(cfg.data_preprocessing.img_mean, device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor(cfg.data_preprocessing.img_std, device=img_tensor.device).view(3, 1, 1)
    img_denorm = torch.clamp(img_tensor * std + mean, 0, 1) # CHW, float[0,1]
    img_np = img_denorm.cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0)) # HWC, float[0,1]
    img_np = (img_np * 255).astype(np.uint8) # HWC, uint8
    return img_np

def save_video_from_frames(frames: List[np.ndarray], save_path: Path, fps: int):
    """Saves a list of HWC uint8 NumPy frames as a video."""
    if not frames:
        log.warning("No frames provided, cannot save video.")
        return
    try:
        log.info(f"Saving video with {len(frames)} frames to {save_path} (FPS={fps})...")
        with imageio.get_writer(save_path, fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)
        log.info("Video saved successfully.")
    except Exception as e:
        log.exception(f"Error saving video to {save_path}: {e}")

def save_subgoal_images(subgoals: List[np.ndarray], save_dir: Path, episode_idx: int):
    """Saves a sequence of subgoal images."""
    if not subgoals:
        return
    ep_dir = save_dir / f"episode_{episode_idx:03d}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    try:
        log.info(f"Saving {len(subgoals)} subgoal images for episode {episode_idx} to {ep_dir}...")
        for i, img_np in enumerate(subgoals):
            # Use matplotlib for simple saving
            plt.imsave(ep_dir / f"subgoal_{i:04d}.png", img_np)
        log.info("Subgoal images saved.")
    except Exception as e:
        log.exception(f"Error saving subgoal images to {ep_dir}: {e}")



# FILE: eval/run_vidhis_oracle.py

# --- START: ROBUST PATCH 3 (Main Evaluation Loop) ---
# Replace the old run_vidhis_evaluation function with this one.

class OraclePlanner:
    """
    A "perfect" planner that provides a sequence of ground-truth future images
    from a single, pre-loaded expert trajectory.
    """
    def __init__(self, dataset, episode_idx: int, cfg: DictConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.subgoal_horizon_k = dataset.subgoal_horizon_k
        
        # Load the entire sequence of primary images for the chosen expert episode.
        ep_meta = dataset.expert_reader.episode_metadata[episode_idx]
        img_primary_meta = ep_meta["modalities"]["image_primary"]
        self.expert_image_trajectory = dataset.expert_reader._get_full_modality_array(
            img_primary_meta["key"],
            img_primary_meta["compression"],
            img_primary_meta["dtype"],
            tuple(img_primary_meta["shape"])
        )
        self.episode_length = ep_meta['length']
        log.info(f"[OraclePlanner] Initialized with expert trajectory from Episode {episode_idx} (Length: {self.episode_length})")

    def sample(self, current_episode_step: int) -> torch.Tensor:
        """
        Retrieves the ground-truth subgoal image from the pre-loaded trajectory.
        """
        # Calculate the target timestep in the expert trajectory.
        subgoal_timestep = current_episode_step + self.subgoal_horizon_k
        
        # Clamp to the last frame if we're near the end.
        subgoal_timestep = min(subgoal_timestep, self.episode_length - 1)
        
        gt_subgoal_np = self.expert_image_trajectory[subgoal_timestep]
        
        # Preprocess the numpy image into a batched tensor.
        subgoal_tensor = preprocess_obs_image(gt_subgoal_np, self.cfg, self.device)
        return subgoal_tensor.unsqueeze(0) # Add batch dimension

def run_vidhis_evaluation(cfg: DictConfig):
    """
    Runs the ViDHiS evaluation using a "Guided Replay" with an Oracle Planner.
    """
    start_time = time.time()
    output_dir = Path.cwd()
    log.info(f"Oracle evaluation output directory: {output_dir}")

    # --- Setup ---
    set_seed(cfg.evaluation.seed)
    device = torch.device(cfg.inference.device)
    video_dir = output_dir / "videos"
    if cfg.visualization.save_video: video_dir.mkdir(exist_ok=True)

    # --- Load Controller and Validation Dataset ---
    controller = load_controller(cfg, device)
    log.info(f"Loading VALIDATION dataset for Oracle from: {cfg.dataset.val_path}")
    val_dataset = HierarchicalControllerDataset(
        dataset_path=cfg.dataset.val_path,
        observation_horizon=cfg.model.controller.observation_horizon,
        action_horizon=cfg.model.controller.action_horizon,
        subgoal_horizon_k=cfg.dataset.subgoal_horizon_k
    )


    # --- Initialize Environment ---
    log.info("Initializing environment...")
    env = PandaEnv(**cfg.environment.env_kwargs)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=cfg.environment.max_episode_steps)
    obs_horizon = cfg.model.controller.observation_horizon
    obs_space = env.observation_space # The TimeLimit wrapper keeps the same obs_space
    obs_history_buffer = ObsHistoryBuffer(n_envs=1, history_len=obs_horizon, obs_space=obs_space)

    # --- Evaluation Loop: Iterate over a selection of expert trajectories ---
    episode_results = []
    episodes_to_run = list(range(min(cfg.evaluation.num_episodes, len(val_dataset.episode_chunks))))
    
    for ep_idx in episodes_to_run:
        log.info(f"--- Starting Guided Replay guided by expert Episode {ep_idx} ---")
        
        # Instantiate the Oracle Planner with the chosen expert trajectory
        planner = OraclePlanner(dataset=val_dataset, episode_idx=ep_idx, cfg=cfg, device=device)
        
        step_count = 0
        done = False
        ep_frames = []
        
        # Start the episode with a standard environment reset
        obs_dict, info = env.reset(seed=cfg.evaluation.seed + ep_idx)
        obs_history_buffer.reset(0, obs_dict)
        
        # --- Inner MPC Loop ---
        while not done:
            # 1. Get current observation history from the buffer
            current_hist_obs_dict_np = obs_history_buffer.get_stacked(0)
            obs_history_batch_tensors = preprocess_obs_history(current_hist_obs_dict_np, cfg, device)

            # 2. Oracle Planner Inference
            # The oracle gets the current step in our live episode
            subgoal_img_tensor = planner.sample(current_episode_step=step_count)
            
            # 3. Controller Inference
            with torch.no_grad():
                action_trajectory_tensor = controller.sample(
                    obs=obs_history_batch_tensors,
                    subgoal_image=subgoal_img_tensor,
                    guidance_scale=cfg.inference.get('controller_guidance_scale', 1.0) # Use .get for safety
                )
            
            action_trajectory = action_trajectory_tensor[0].cpu().numpy()
            
            # 4. Execute only the FIRST action of the planned trajectory
            action = action_trajectory[0]
            next_obs_dict, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if cfg.visualization.save_video:
                frame = env.render()
                if frame is not None:
                    ep_frames.append(frame)

            # 5. Update history buffer with the NEW observation from the simulator
            obs_history_buffer.append(0, next_obs_dict)

        
        # --- Episode End ---
        success = info.get('is_success', False)
        episode_results.append({"episode_idx": ep_idx, "success": success, "steps": step_count})
        log.info(f"--- Guided Replay Finished: Success={success}, Steps={step_count} ---")
        if cfg.visualization.save_video:
            save_video_from_frames(ep_frames, video_dir / f"oracle_ep_{ep_idx:03d}.mp4", cfg.visualization.video_fps)

    # --- Aggregate Results ---
    if episode_results:
        success_rate = np.mean([r['success'] for r in episode_results])
        log.info("\n========== Oracle Evaluation Summary ==========")
        log.info(f"Success Rate with Perfect Planner: {success_rate:.3f}")
        log.info("==============================================")

    env.close()

# --- END: ROBUST PATCH 3 ---


# --- Hydra Main Entry Point ---

@hydra.main(version_base=None, config_path="../configs", config_name="run_vidhis_config")
def main(cfg: DictConfig):
    # Basic logging setup (Hydra might configure handlers too)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    log.info("----------- ViDHiS Evaluation Configuration -----------")
    log.info(OmegaConf.to_yaml(cfg))
    log.info("-----------------------------------------------------")

    try:
        run_vidhis_evaluation(cfg)
    except Exception as e:
        log.exception("An error occurred during the ViDHiS evaluation run.")
        sys.exit(1) # Exit with error code

if __name__ == "__main__":
    # Add dependency checks if needed (e.g., imageio)
    main()