# FILE: eval/run_vidhis.py
# SOTA Inference Script for Vision-Diffusion Hierarchical System (ViDHiS)

import os
import time
import logging
import random
from pathlib import Path
import sys
from typing import Dict, Any, Tuple, Optional, List

# Add project root for imports if necessary (adjust relative path as needed)
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

# --- Helper Functions ---

def load_models(cfg: DictConfig, device: torch.device) -> Tuple[nn.Module, nn.Module]:
    """Loads Planner and Controller models from checkpoints."""
    log.info("Loading models...")

    # --- Load Planner ---
    planner_path = Path(cfg.planner_ckpt_path)
    if not planner_path.exists():
        raise FileNotFoundError(f"Planner checkpoint not found: {planner_path}")
    try:
        # Assuming Planner uses PyTorch Lightning checkpoints
        # Need to instantiate model first, then load state_dict OR use PL load_from_checkpoint
        # Let's assume loading state_dict into a fresh instance for simplicity
        planner = VisualPlannerDiffusion(
             # Re-instantiate with config used during training (or load from ckpt's hparams)
             # Assuming config structure matches model's __init__
             image_size=cfg.model.planner.image_size,
             vit_model_name=cfg.model.planner.vit_model_name,
             # ... other planner args from its config ...
        )
        # Load state dict (adjust based on how checkpoints are saved)
        # If saved via PL: state = torch.load(planner_path, map_location=device)['state_dict']
        # Need to strip "model." prefix if saved by PL
        state = torch.load(planner_path, map_location=device)
        if 'state_dict' in state: state = state['state_dict']
        planner_state_dict = {k.replace("model.", ""): v for k, v in state.items()}
        planner.load_state_dict(planner_state_dict)
        log.info(f"Loaded Planner state dict from {planner_path}")

    except Exception as e:
        log.exception(f"Error loading Planner from {planner_path}: {e}")
        raise
    planner = planner.to(device).eval()

    # --- Load Controller (Adapted DiffusionPolicy) ---
    controller_path = Path(cfg.controller_ckpt_path)
    if not controller_path.exists():
        raise FileNotFoundError(f"Controller checkpoint not found: {controller_path}")
    try:
        # Re-instantiate adapted DiffusionPolicy
        # Needs config details matching its training
        # We need proprio_dim - should be in config or loadable from checkpoint
        proprio_dim = cfg.model.controller.proprio_dim # Add this to config
        controller = DiffusionPolicy(
            proprio_dim=proprio_dim,
            H_o=cfg.model.controller.observation_horizon,
            H_a=cfg.model.controller.action_horizon,
            action_dim=cfg.model.controller.action_dim, # Get from config
            image_feat_dim=cfg.model.controller.image_feat_dim,
            scheduler_cfg=NoiseSchedulerConfig(**cfg.scheduler), # Assuming same scheduler
            d_model=cfg.model.controller.d_model,
            denoiser_layers=cfg.model.controller.denoiser_layers,
            denoiser_heads=cfg.model.controller.denoiser_heads,
            # Ensure subgoal conditioning adaptation is included here if structural
            device=device
        )
        # Load state dict (adjust based on how controller checkpoints are saved)
        state = torch.load(controller_path, map_location=device)
        # Handle potential nesting (e.g., if saved within a trainer checkpoint)
        if 'state_dict' in state: state = state['state_dict'] # Common PL format
        if 'model' in state: state = state['model'] # Another common format
        if 'actor_state_dict' in state: state = state['actor_state_dict'] # From RL checkpoint
        # Strip "diffusion_policy." prefix if saved via DiffusionActor wrapper
        controller_state_dict = {k.replace("diffusion_policy.", ""): v for k, v in state.items()}
        controller.load_state_dict(controller_state_dict)
        log.info(f"Loaded Controller state dict from {controller_path}")

    except Exception as e:
        log.exception(f"Error loading Controller from {controller_path}: {e}")
        raise
    controller = controller.to(device).eval()

    # Apply torch.compile if configured and available
    if cfg.inference.use_torch_compile and hasattr(torch, "compile"):
        log.info(f"Applying torch.compile (mode='{cfg.inference.torch_compile_mode}')...")
        try:
            planner = torch.compile(planner, mode=cfg.inference.torch_compile_mode)
            controller = torch.compile(controller, mode=cfg.inference.torch_compile_mode)
            log.info("torch.compile applied successfully.")
        except Exception as e:
            log.warning(f"torch.compile failed: {e}. Continuing without compilation.")

    return planner, controller


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

def preprocess_obs_image(img_np: np.ndarray, cfg: DictConfig, device: torch.device) -> torch.Tensor:
    """Applies preprocessing to a NumPy observation image."""
    # Assumes img_np is HWC, uint8
    transform = transforms.Compose([
        transforms.ToTensor(), # HWC:uint8 -> CHW:float[0,1]
        transforms.Resize(tuple(cfg.data_preprocessing.image_size), antialias=True),
        transforms.Normalize(mean=tuple(cfg.data_preprocessing.img_mean),
                             std=tuple(cfg.data_preprocessing.img_std))
    ])
    return transform(img_np).to(device)


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


# --- Main Evaluation Function ---

def run_vidhis_evaluation(cfg: DictConfig):
    """Runs the ViDHiS evaluation loop."""
    start_time = time.time()
    output_dir = Path.cwd() # Hydra sets current working directory to output dir
    log.info(f"Evaluation output directory: {output_dir}")

    # --- Setup ---
    set_seed(cfg.evaluation.seed)
    device = torch.device(cfg.inference.device)
    if device.type == 'cpu':
        log.warning("Running inference on CPU. This will be slow!")
        # Set threads for CPU inference
        try:
            num_cores = os.cpu_count() or 1
            effective_cores = min(num_cores, cfg.inference.get("cpu_threads", num_cores)) # Configurable
            torch.set_num_threads(effective_cores)
            os.environ['OMP_NUM_THREADS'] = str(effective_cores)
            os.environ['MKL_NUM_THREADS'] = str(effective_cores)
            log.info(f"Set PyTorch/OMP/MKL CPU threads to: {effective_cores}")
        except Exception as e:
            log.warning(f"Failed to set CPU thread counts: {e}")

    # --- Create Output Dirs ---
    video_dir = output_dir / "videos"
    subgoal_dir = output_dir / "subgoals"
    if cfg.visualization.save_video: video_dir.mkdir(exist_ok=True)
    if cfg.visualization.save_subgoals: subgoal_dir.mkdir(exist_ok=True)

    # --- Load Models and Goal ---
    planner, controller = load_models(cfg, device)
    goal_image_tensor = load_and_preprocess_goal(cfg, device) # Shape (1, C, H, W)

    # --- Initialize Environment ---
    log.info("Initializing environment...")
    # Use _build_single_env for consistency, though VecEnv wrapper isn't strictly needed for n=1
    env = RLFineTuner._build_single_env(None, cfg, rank=0, is_eval=True) # Assuming helper is part of RLFineTuner
    # Need observation/action horizons from config
    obs_horizon = cfg.model.controller.observation_horizon
    # Need single_step_obs_space, get it from env
    # IMPORTANT: Need to handle potential wrappers like TimeLimit to get base env space
    base_env = env.unwrapped
    while hasattr(base_env, "env"): # Unwrap potential nested wrappers
         base_env = base_env.env
         if isinstance(base_env, PandaEnv): break # Found it
    if not isinstance(base_env, PandaEnv): raise TypeError("Could not find PandaEnv instance.")

    obs_history_buffer = ObsHistoryBuffer(n_envs=1, history_len=obs_horizon, obs_space=base_env.observation_space)
    log.info("Environment and history buffer initialized.")

    # --- Evaluation Loop ---
    episode_results = []
    max_steps = cfg.environment.max_episode_steps

    for ep_idx in range(cfg.evaluation.num_episodes):
        log.info(f"--- Starting Evaluation Episode {ep_idx+1}/{cfg.evaluation.num_episodes} ---")
        ep_start_time = time.time()
        step_count = 0
        done = False
        ep_frames = []
        ep_subgoals_denorm = []

        try:
            # Reset environment and history buffer
            obs_dict, info = env.reset(seed=cfg.evaluation.seed + ep_idx) # Seed each episode
            current_hist_obs_dict = obs_history_buffer.reset(0, obs_dict)
        except Exception as e:
             log.exception(f"Error resetting environment for episode {ep_idx+1}. Skipping.")
             continue

        # --- Inner MPC Loop ---
        while not done and step_count < max_steps:
            loop_start_time = time.time()

            # 1. Prepare Inputs
            obs_history_batch = { # Add batch dim for models
                k: torch.as_tensor(v[np.newaxis, ...], device=device).float()
                for k, v in current_hist_obs_dict.items()
            }
            # Assuming 'image_primary' is the key for the main camera
            current_image_tensor = obs_history_batch['image_primary'][:, -1, ...] # Get last image in history (B, C, H, W)
            progress_scalar = torch.tensor([[step_count / max_steps]], device=device, dtype=torch.float32) # Shape (1, 1)

            # 2. Planner Inference
            try:
                with torch.no_grad():
                    subgoal_img_tensor = planner.sample( # Shape (1, C, H, W)
                        current_image=current_image_tensor,
                        goal_image=goal_image_tensor,
                        progress=progress_scalar,
                        num_inference_steps=cfg.inference.planner_inference_steps
                    )
                if cfg.visualization.save_subgoals:
                    subgoal_np = denormalize_image(subgoal_img_tensor[0], cfg) # Denormalize first item
                    ep_subgoals_denorm.append(subgoal_np)
            except Exception as e:
                 log.exception("Error during Planner inference. Ending episode.")
                 break


            # 3. Controller Inference
            try:
                with torch.no_grad():
                    # Controller needs obs history and the generated subgoal
                    # Assuming controller adapted to take subgoal tensor directly
                    action_trajectory_tensor = controller.predict_action( # Check method name
                        obs_history_batch, # (B, H, ...)
                        subgoal_img_tensor # (B, C, H, W)
                    ) # Output shape (B, H_a, A_dim)
            except Exception as e:
                 log.exception("Error during Controller inference. Ending episode.")
                 break

            action_trajectory = action_trajectory_tensor[0].cpu().numpy() # Get batch 0, move to CPU

            # 4. Execute MPC Horizon (k steps)
            k = cfg.inference.mpc_horizon_k
            for i in range(k):
                if done: break # Check if episode ended during previous inner step

                action = action_trajectory[i]

                # Step environment
                try:
                    next_obs_dict, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                except Exception as e:
                    log.exception(f"Error during env.step at step {step_count+1}. Ending episode.")
                    done = True # Force end

                # Render frame (before history update)
                if cfg.visualization.save_video:
                    try:
                        frame = env.render()
                        if isinstance(frame, np.ndarray): ep_frames.append(frame)
                    except Exception as e: log.warning(f"Failed to render frame: {e}")

                # Update history buffer with the *new* observation
                obs_history_buffer.append(0, next_obs_dict)
                current_hist_obs_dict = obs_history_buffer.get_stacked(0) # Update for next MPC cycle

                step_count += 1
                if done:
                    log.info(f"Episode ended at step {step_count}. Terminated={terminated}, Truncated={truncated}")
                    break # Exit k loop

            # Timing for one MPC cycle (Planner + Controller + k env steps)
            cycle_time = time.time() - loop_start_time
            log.debug(f"MPC Cycle (Step {step_count-k}-{step_count}): {cycle_time:.3f}s")


        # --- Episode End ---
        ep_duration = time.time() - ep_start_time
        success = info.get('is_success', False) if 'info' in locals() else False # Check if placement succeeded
        episode_results.append({
            "episode": ep_idx + 1,
            "success": success,
            "steps": step_count,
            "duration": ep_duration
        })
        log.info(f"--- Episode {ep_idx+1} Finished: Success={success}, Steps={step_count}, Duration={ep_duration:.2f}s ---")

        # Save visualizations
        if cfg.visualization.save_video:
            save_video_from_frames(ep_frames, video_dir / f"episode_{ep_idx+1:03d}.mp4", cfg.visualization.video_fps)
        if cfg.visualization.save_subgoals:
            save_subgoal_images(ep_subgoals_denorm, subgoal_dir, ep_idx + 1)

    # --- Aggregate Results ---
    if episode_results:
        success_rate = np.mean([r['success'] for r in episode_results])
        mean_steps = np.mean([r['steps'] for r in episode_results if r['success']]) # Steps only for successful eps
        mean_duration = np.mean([r['duration'] for r in episode_results])
        log.info("\n========== ViDHiS Evaluation Summary ==========")
        log.info(f"Total Episodes: {len(episode_results)}")
        log.info(f"Success Rate:   {success_rate:.3f}")
        log.info(f"Mean Steps (Success): {mean_steps:.1f}" if not np.isnan(mean_steps) else "Mean Steps (Success): N/A (0 successes)")
        log.info(f"Mean Duration:  {mean_duration:.2f}s")
        log.info("==============================================")

        # Save results to a file
        results_path = output_dir / "evaluation_results.json"
        try:
            with open(results_path, 'w') as f:
                json.dump(episode_results, f, indent=2)
            log.info(f"Saved detailed results to {results_path}")
        except Exception as e:
            log.error(f"Failed to save results file: {e}")

    else:
        log.warning("No episodes were completed.")

    # --- Cleanup ---
    env.close()
    total_duration = time.time() - start_time
    log.info(f"Evaluation finished in {total_duration:.2f} seconds.")


# --- Hydra Main Entry Point ---

@hydra.main(version_base=None, config_path="../../configs", config_name="run_vidhis_config")
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