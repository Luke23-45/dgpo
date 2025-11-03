#!/usr/bin/env python3

"""
EGO-Planner Visual Evaluation Script

This script loads a trained EGO-Planner model and its corresponding Hydra
configuration to run a visual evaluation in the PandaEnv. It records a
video of the policy's performance.

Key Steps:
1.  Loads the Hydra config to build the model architecture.
2.  Loads the .ckpt file and extracts the EMA (Exponential Moving Average)
    weights, which are required for stable inference.
3.  Initializes the PandaEnv.
4.  For each episode:
    a.  Resets the environment and captures the 'initial_image'.
    b.  Manually moves the object to the goal position to render a 'goal_image'.
    c.  Resets the environment again to start the episode.
    d.  Maintains a history (deque) of observations, matching the
        `obs_horizon` the model was trained on.
    e.  At each step, passes the full batch (initial_img, goal_img, obs_history)
        to the model's `.sample()` method.
    f.  Executes the first action from the returned action plan.
    g.  Records the visual output to an MP4 video file.

Usage:
1.  Make sure you have an environment with all required packages
    (pytorch, hydra-core, omegaconf, opencv-python, torchvision, etc.).
2.  Place this script in a directory where it can import the project modules
    (like `envs.panda_env`, `models.ego_planner`, etc.).
3.  Run from the command line, pointing to your config and checkpoint:

    python evaluate_ego_planner.py \
        --config-path /path/to/your/configs \
        --config-name train_ego_planner_config.yaml \
        hydra.run.dir=. \
        output_video=ego_planner_eval.mp4 \
        checkpoint_path=/path/to/your/model/best.ckpt


python -m s7 --config-path "./configs" --config-name "evaluate_ego_planner_config.yaml" hydra.run.dir=. output_video=ego_planner_eval.mp4 checkpoint_path="./notes/checkpoints/backup_epoch_74.ckpt"

"""

import logging
import os
import collections
from pathlib import Path
import pytorch_lightning as pl
import cv2
import hydra
import numpy as np
import torch
import mujoco
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import transforms

# --- Import Project Modules ---
# Ensure this script is run from a location where these modules are importable
from envs.panda_env import PandaEnv, DomainRandomizationConfig
from models.ego_planner import EgoPlanner
from train.train_ego_planner import EgoPlannerLightningModule
from models.ego_planner import NoiseScheduler, NoiseSchedulerConfig

# Set up a logger
log = logging.getLogger(__name__)

# --- Helper Functions ---
# In your evaluation script (evaluate_ego_planner.py)
# REPLACE the entire function with this one.

def load_model_from_checkpoint(cfg: DictConfig, checkpoint_path: str, device: torch.device) -> EgoPlanner:
    """
    Loads the EGO-Planner model from a Lightning checkpoint.
    
    [DEFINITIVE VERSION] This function is robustly patched to:
    1.  Load the original training hyperparameters directly from the checkpoint file,
        preventing config mismatches during evaluation.
    2.  Correctly extract and load the EMA (Exponential Moving Average) weights,
        which are essential for stable inference.
    """
    log.info(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load the full checkpoint on CPU first to inspect its contents
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # --- START OF THE FIX ---
    # 1. Load the hyperparameters that were used during training
    if 'hyper_parameters' not in ckpt:
        raise KeyError(
            "Checkpoint is missing 'hyper_parameters'. It may be from an older version "
            "of PyTorch Lightning or was saved improperly."
        )
    
    # 2. Create the original training config from the stored hyperparameters
    #    This ensures the model architecture is built exactly as it was during training.
    original_train_cfg = OmegaConf.create(ckpt['hyper_parameters'])
    log.info("Successfully loaded original training config from checkpoint.")
    # --- END OF THE FIX ---

    # Check if 'ema_state_dict' exists. This is crucial.
    if 'ema_state_dict' not in ckpt:
        raise KeyError(
            "Checkpoint does not contain 'ema_state_dict'. "
            "This script requires the EMA weights for evaluation."
        )
        
    log.info("Found 'ema_state_dict'. Initializing model from original config...")
    
    # 3. Initialize the LightningModule with the ORIGINAL training config
    lightning_model = EgoPlannerLightningModule(original_train_cfg)
    
    # 4. Load the EMA state dict into the model's EMA object
    lightning_model.ema.load_state_dict(ckpt['ema_state_dict'])
    
    # 5. Get the *actual* model from the EMA wrapper
    model = lightning_model.ema.ema_model
    
    # 6. Move to the target device and set to evaluation mode
    model.to(device)
    model.eval()
    
    log.info("Model loaded successfully using EMA weights and set to eval mode.")
    return model


def get_goal_image(env: PandaEnv, obs: dict) -> np.ndarray:
    """
    Creates a 'goal_image' by saving the current state, moving the object
    to the goal position, rendering, and then restoring the original state.
    """
    log.debug("Capturing goal image...")
    
    # 1. Save the current complete simulation state
    try:
        state = env.get_mj_state()
    except Exception as e:
        log.error(f"Error getting MuJoCo state: {e}")
        return obs['image_primary'] # Fallback

    # 2. Get the goal position from the observation
    goal_pos_world = obs['goal_pos_world']
    
    # 3. Manually set the object's free joint to the goal position
    try:
        qpos_addr = env.model.jnt_qposadr[env.object_joint_id]
        env.data.qpos[qpos_addr:qpos_addr + 3] = goal_pos_world
        
        # We also need to set the orientation if available
        if 'goal_orn_world' in obs:
             # Convert xyzw (SciPy) to wxyz (MuJoCo)
             quat_xyzw = obs['goal_orn_world']
             quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
             env.data.qpos[qpos_addr + 3:qpos_addr + 7] = quat_wxyz

        # 4. Propagate this change through the simulation
        mujoco.mj_forward(env.model, env.data)
        
        # 5. Render the "goal" scene
        goal_img_np = env.render(camera_name="fixed_camera")
        
    except Exception as e:
        log.error(f"Error manually setting goal pose: {e}")
        goal_img_np = obs['image_primary'] # Fallback
    finally:
        # 6. Restore the original simulation state
        try:
            env.set_mj_state(state)
        except Exception as e:
            log.error(f"Error restoring MuJoCo state: {e}")
            
    log.debug("Goal image captured and state restored.")
    return goal_img_np


def preprocess_image(img_np: np.ndarray, transform: transforms.Compose) -> torch.Tensor:
    """
    Converts a NumPy image (H, W, C) to a preprocessed PyTorch tensor (C, H, W).
    """
    img_pil = Image.fromarray(img_np)
    return transform(img_pil)


# --- Main Evaluation Function ---

@hydra.main(version_base=None, config_path="configs", config_name="train_ego_planner_config")
def evaluate(cfg: DictConfig):
    """
    Main evaluation function driven by Hydra.
    """
    
    # --- 1. Configuration & Setup ---
    
    # Override config settings for evaluation
    OmegaConf.set_struct(cfg, False) # Allow adding new keys
    cfg.checkpoint_path = cfg.get("checkpoint_path", None)
    cfg.output_video = cfg.get("output_video", "evaluation_video.mp4")
    cfg.num_episodes = cfg.get("num_episodes", 5)
    OmegaConf.set_struct(cfg, True) # Re-lock config
    
    log.info("--- EGO-Planner Visual Evaluation ---")
    log.info(f"Config overrides:\n{OmegaConf.to_yaml(cfg)}")

    if not cfg.checkpoint_path:
        log.error("No 'checkpoint_path' provided. Exiting.")
        return

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    # Set seed for reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # --- 2. Load Model, Env, and Scheduler ---
    
    # Load the model using the helper
    model = load_model_from_checkpoint(cfg, cfg.checkpoint_path, device)
    
    # Get model-specific horizons
    obs_horizon = cfg.model.obs_horizon
    action_horizon = cfg.model.action_horizon
    
    # Initialize the noise scheduler
    scheduler_cfg = NoiseSchedulerConfig(
        beta_start=cfg.scheduler.beta_start,
        beta_end=cfg.scheduler.beta_end,
        schedule=cfg.scheduler.beta_schedule,
        timesteps=cfg.scheduler.timesteps,
    )
    noise_scheduler = NoiseScheduler(scheduler_cfg).to(device)
    
    # Initialize the environment
    # We disable domain randomization for consistent evaluation
    log.info("Initializing PandaEnv (Domain Randomization DISABLED for eval)...")
    dr_config = DomainRandomizationConfig() # Default config
    env = PandaEnv(
        xml_path="envs/panda_pick_place.xml", 
        enable_domain_randomization=False,
        control_mode="delta"
    )

    # --- 3. Setup Image Transforms ---
    # These must match the transforms used in utils/ego_planner_dataset.py
    transform_primary = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor()
    ])
    transform_wrist = transforms.Compose([
        transforms.Resize((128, 128), antialias=True),
        transforms.ToTensor()
    ])
    
    # --- 4. Setup Video Recording ---
    video_path = Path(cfg.output_video)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get frame size from a test render
    frame_test = env.render(camera_name="fixed_camera")
    H, W, _ = frame_test.shape
    
    video_writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        30.0,  # FPS
        (W, H) # Frame size
    )
    log.info(f"Recording video to: {video_path} ({W}x{H} @ 30fps)")

    # --- 5. Run Evaluation Loop ---
    
    for ep_idx in range(cfg.num_episodes):
        log.info(f"--- Starting Evaluation Episode {ep_idx + 1} / {cfg.num_episodes} ---")
        
        # --- 5a. Episode Reset & Setup ---
        episode_seed = cfg.seed + ep_idx
        obs, _ = env.reset(seed=episode_seed)
        
        # Get static images for the Strategist
        initial_image_np = obs['image_primary']
        goal_image_np = get_goal_image(env, obs)
        
        # Preprocess and batch them (B=1)
        initial_image_tensor = preprocess_image(initial_image_np, transform_primary).to(device).unsqueeze(0)
        goal_image_tensor = preprocess_image(goal_image_np, transform_primary).to(device).unsqueeze(0)
        
        # Initialize observation history deques
        primary_hist = collections.deque(maxlen=obs_horizon)
        wrist_hist = collections.deque(maxlen=obs_horizon)
        proprio_hist = collections.deque(maxlen=obs_horizon)
        
        # Pad the history with the first observation
        for _ in range(obs_horizon):
            primary_hist.append(obs['image_primary'])
            wrist_hist.append(obs['image_wrist'])
            proprio_hist.append(obs['proprio'])
            
        step_count = 0
        
        while step_count < env.max_episode_steps:
            
            # --- 5b. Prepare Model Batch ---
            
            # Stack history deques into tensors
            img_primary_hist_tensor = torch.stack(
                [preprocess_image(img, transform_primary) for img in primary_hist]
            ).to(device).unsqueeze(0) # (B, H_o, C, H, W)
            
            img_wrist_hist_tensor = torch.stack(
                [preprocess_image(img, transform_wrist) for img in wrist_hist]
            ).to(device).unsqueeze(0) # (B, H_o, C, H, W)
            
            proprio_hist_tensor = torch.from_numpy(
                np.stack(proprio_hist)
            ).float().to(device).unsqueeze(0) # (B, H_o, D_p)
            
            # Assemble the complete batch for the model
            batch = {
                'initial_image': initial_image_tensor,
                'goal_image': goal_image_tensor,
                'observation_history': {
                    'image_primary': img_primary_hist_tensor,
                    'image_wrist': img_wrist_hist_tensor,
                    'proprio': proprio_hist_tensor,
                }
            }
            
            # --- 5c. Get Action from Model ---
            try:
                with torch.no_grad():
                    action_chunk = model.sample(
                        batch=batch,
                        scheduler=noise_scheduler,
                        guidance_plan=cfg.inference.guidance_scale_plan,
                        guidance_obs=cfg.inference.guidance_scale_obs,
                        num_inference_steps=cfg.inference.sampling_steps
                    )
                
                # We only execute the first action in the plan
                # action_chunk shape is (B, H_a, D_a)
                action = action_chunk[0, 0, :].cpu().numpy()
                
            except Exception as e:
                log.error(f"Model inference failed: {e}", exc_info=True)
                break # Stop this episode
                
            # --- 5d. Step Environment ---
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # --- 5e. Record Frame ---
            frame_rgb = env.render(camera_name="fixed_camera")
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
            
            # --- 5f. Update History ---
            primary_hist.append(obs['image_primary'])
            wrist_hist.append(obs['image_wrist'])
            proprio_hist.append(obs['proprio'])
            
            if terminated or truncated:
                log.info(f"Episode finished after {step_count} steps. (Terminated: {terminated}, Truncated: {truncated})")
                break
                
        if step_count == env.max_episode_steps:
            log.info(f"Episode timed out after {env.max_episode_steps} steps.")

    # --- 6. Cleanup ---
    video_writer.release()
    env.close()
    log.info(f"--- Evaluation Complete ---")
    log.info(f"Video saved to: {os.path.abspath(cfg.output_video)}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
    )
    evaluate()
