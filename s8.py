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


python -m s7 --config-path "./configs" --config-name "evaluate_ego_planner_config.yaml" hydra.run.dir=. output_video=ego_planner_eval.mp4 checkpoint_path="C:\Users\Hellx\Documents\Programming\python\Project\redhot\notes\checkpoints\v1\backup_epoch_39.ckpt"

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
from tqdm.auto import tqdm 
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

@hydra.main(version_base=None, config_path="../configs", config_name="evaluate_uhp_config")
def evaluate(cfg: DictConfig):
    log.info("--- UHP v2.0 Hierarchical Visual Evaluation (SOTA Patched) ---")
    log.info(f"Full evaluation config:\n{OmegaConf.to_yaml(cfg)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl.seed_everything(cfg.seed, workers=True)

    # --- 1. Load the "Single Source of Truth" ---
    lightning_module = load_lightning_module_from_checkpoint(cfg.checkpoint_path, device)
    
    # --- 2. Unpack All Components from the Single Source of Truth ---
    model: UHP_Orchestrator = lightning_module.model
    action_normalizer = lightning_module.action_normalizer
    proprio_normalizer = lightning_module.proprio_normalizer
    joint_limits_low = lightning_module.joint_limits_low
    joint_limits_high = lightning_module.joint_limits_high
    noise_scheduler = lightning_module.noise_scheduler
    train_cfg = lightning_module.cfg # The original, correct training config
    
    # Get horizons and dimensions from the original training config for robustness
    obs_horizon = train_cfg.model.executor_cfg.obs_horizon
    action_dim = train_cfg.model.executor_cfg.action_dim
    action_horizon = train_cfg.model.executor_cfg.action_horizon
    
    # --- 3. Initialize Environment ---
    env = PandaEnv(
        xml_path=train_cfg.env.xml_path,
        control_mode="delta",
        enable_domain_randomization=False # Explicitly disable for consistency
    )
    
    # --- 4. Setup Image Transforms ---
    transform_planner_img = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_controller_primary = transforms.Compose([transforms.Resize((224, 224), antialias=True), transforms.ToTensor()])
    transform_controller_wrist = transforms.Compose([transforms.Resize((128, 128), antialias=True), transforms.ToTensor()])
    
    # --- 5. Setup Video & CSV Recording ---
    video_writer = None
    if cfg.logging.enable_video:
        video_path = Path(cfg.output_video)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        frame_test, _ = env.reset(seed=cfg.seed)
        H, W, _ = env.render().shape
        video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (W, H))
        log.info(f"Recording video to: {video_path}")

    # (Optional: Add CSV writer setup here if needed, mirroring Ego-Planner's)
    
    # --- 6. Run Hierarchical Evaluation Loop ---
    for ep_idx in tqdm(range(cfg.num_episodes), desc="Evaluating Episodes"):
        episode_seed = cfg.seed if cfg.eval_static_scene else cfg.seed + ep_idx
        obs, _ = env.reset(seed=episode_seed)
        
        goal_image_np = get_goal_image(env, obs)
        goal_image_tensor = transform_planner_img(Image.fromarray(goal_image_np)).to(device).unsqueeze(0)
        
        # [SOTA WARM-UP] Populate the history with unique, consecutive observations.
        obs_history = collections.deque(maxlen=obs_horizon)
        log.info(f"Warming up observation history for {obs_horizon} steps...")
        for _ in range(obs_horizon):
            obs, _, _, _, _ = env.step(np.zeros(action_dim))
            obs_history.append(obs)
        log.info("Warm-up complete. Starting policy.")
            
        current_phase = -1
        subgoal_embedding = None
        prev_is_grasped = False

        step_iterator = tqdm(range(env.max_episode_steps), desc=f"Episode {ep_idx+1}", leave=False)
        for step_count in step_iterator:
            
            # --- Hierarchical Control Logic with Corrected Oracle ---
            new_phase = get_current_task_phase(obs, prev_is_grasped)
            if new_phase != current_phase:
                log.info(f"Step {step_count}: Phase changed from {current_phase} -> {new_phase}. Re-planning...")
                current_phase = new_phase
                current_image_tensor = transform_planner_img(Image.fromarray(obs['image_primary'])).to(device).unsqueeze(0)
                task_phase_tensor = torch.tensor([current_phase], dtype=torch.long, device=device)
                
                # The `plan` method returns (subgoal_embedding, heatmap)
                subgoal_embedding, heatmap_viz = model.plan(
                    current_image_tensor,
                    goal_image_tensor,
                    task_phase_tensor
                )
            
            # --- Prepare Executor Inputs from the Warmed-Up History ---
            proprio_hist = torch.from_numpy(np.stack([h['proprio'] for h in obs_history])).float().to(device)
            primary_hist = torch.stack([transform_controller_primary(Image.fromarray(h['image_primary'])) for h in obs_history]).to(device)
            wrist_hist = torch.stack([transform_controller_wrist(Image.fromarray(h['image_wrist'])) for h in obs_history]).to(device)
            controller_obs_hist = {
                'image_primary': primary_hist.unsqueeze(0),
                'image_wrist': wrist_hist.unsqueeze(0),
                'proprio': proprio_hist.unsqueeze(0)
            }
            
            # --- Get Action from Policy ---
            action_chunk_raw = model.act(
                controller_obs_hist, subgoal_embedding, noise_scheduler,
                cfg.inference.inference_steps, action_normalizer, proprio_normalizer,
                joint_limits_low, joint_limits_high
            )
            action = action_chunk_raw[0, 0, :].cpu().numpy()
            
            # (Optional: Add CSV logging for the current step here)

            # Update state for the next oracle call BEFORE stepping the environment.
            prev_is_grasped = obs.get('is_grasped', [0.0])[0] > 0.5
            
            # --- Step Environment and Update History ---
            obs, reward, terminated, truncated, info = env.step(action)
            obs_history.append(obs)
            
            # --- Record Frame with Diagnostic Overlay ---
            if cfg.logging.enable_video:
                frame_rgb = env.render()
                
                # Add heatmap overlay for diagnostics
                heatmap_np = heatmap_viz[0, 0].cpu().numpy()
                heatmap_resized = cv2.resize(heatmap_np, (W, H))
                heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                
                # Blend the heatmap with the frame
                overlay_frame = cv2.addWeighted(frame_rgb, 0.6, heatmap_colored, 0.4, 0)
                
                # Add text overlay
                phase_text = f"Phase: {current_phase}"
                cv2.putText(overlay_frame, phase_text, (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                
                video_writer.write(cv2.cvtColor(overlay_frame, cv2.COLOR_RGB2BGR))
            
            if terminated or truncated:
                log.info(f"Episode finished after {step_count + 1} steps.")
                break
        
    # --- 7. Cleanup ---
    if video_writer is not None: video_writer.release()
    env.close()
    log.info("--- Evaluation Complete. ---")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
    )
    evaluate()
