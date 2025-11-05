#!/usr/bin/env python3

"""
ViP-C (Visual Planner-Controller) SOTA Evaluation Script

This script provides a comprehensive, diagnostic-rich evaluation for the
hierarchical ViP-C policy. It loads a trained model, its normalizers, and
kinematic limits from a checkpoint and executes the policy in the PandaEnv,
recording a video with rich visual overlays.

Key SOTA Features:
1.  **Hierarchical Control Loop:** Correctly implements the "Plan -> Act -> Re-plan"
    dialogue. The high-level Planner is called only when the task phase changes,
    and the low-level Controller executes the received subgoal.
2.  **Full State Restoration:** Robustly loads the complete training state,
    including the model (EMA weights), action/proprio normalizers, and
    kinematic limits directly from the PyTorch Lightning checkpoint.
3.  **Correct "Norm-to-Raw" Workflow:** Implements the full, symmetrical data
    flow required for inference:
    - Raw proprioception from the env is *normalized* before being passed to the model.
    - Normalized actions from the model are *un-normalized* and *clamped* before
      being sent to the environment.
4.  **Ground-Truth State Determination:** Programmatically determines the current
    `TaskPhase` at each step using ground-truth information from the environment,
    perfectly mirroring the data labeling logic.
5.  **Diagnostic Visualization:** Overlays the Planner's predicted subgoal
    heatmap directly onto the recorded video, providing invaluable insight into
    the model's high-level decision-making process in real-time.

Usage:
    python evaluate_vip_c.py \
        checkpoint_path=/path/to/your/vip_c.ckpt \
        output_video=vip_c_evaluation.mp4

python -m s2 checkpoint_path=/notes/checkpoints/backup_epoch_31.ckpt output_video=vip_c_evaluation.mp4
"""

import logging
import collections
from pathlib import Path
import cv2
import hydra
import numpy as np
import torch
import mujoco
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from typing import Dict
import pytorch_lightning as pl
# --- Import Project Modules ---
from envs.panda_env import PandaEnv
from models.vip_c import ViPC, LinearNormalizer
from train.train_vip_c import ViPCLightningModule
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(name)s] - %(message)s')
log = logging.getLogger(__name__)

# In FILE: evaluate_vip_c.py

# --- [REPLACE THE ENTIRE get_goal_image FUNCTION WITH THIS] ---

def get_goal_image(env: PandaEnv, obs: dict) -> np.ndarray:
    """
    [DEFINITIVE, CORRECTED VERSION - COMPATIBLE WITH OUR PandaEnv]
    Creates a 'goal_image' by saving the current state, moving the object
    to the goal position, rendering, and then restoring the original state.
    """
    log.debug("Capturing goal image by temporarily moving object...")
    
    # 1. Save the current complete simulation state using our env's method
    original_mj_state = env.get_mj_state()

    try:
        # 2. Get the goal position from the observation
        goal_pos_world = obs['goal_pos_world']
        
        # 3. Manually set the object's free joint to the goal position
        qpos_addr = env.model.jnt_qposadr[env.object_joint_id]
        env.data.qpos[qpos_addr:qpos_addr + 3] = goal_pos_world
        
        # Set orientation if available
        if 'goal_orn_world' in obs:
             quat_xyzw = obs['goal_orn_world']
             # Convert xyzw (SciPy) to wxyz (MuJoCo)
             quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
             env.data.qpos[qpos_addr + 3:qpos_addr + 7] = quat_wxyz

        # 4. Propagate this change through the simulation state
        mujoco.mj_forward(env.model, env.data)
        
        # 5. Render the "goal" scene from the correct camera
        goal_img_np = env.render(camera_name="fixed_camera")
        
    except Exception as e:
        log.error(f"Error manually setting goal pose: {e}", exc_info=True)
        goal_img_np = obs['image_primary'] # Fallback to current image
    finally:
        # 6. CRITICAL: Always restore the original simulation state
        env.set_mj_state(original_mj_state)
            
    log.debug("Goal image captured and state restored.")
    return goal_img_np



def load_lightning_module_from_checkpoint(
    checkpoint_path: str, device: torch.device
) -> ViPCLightningModule:
    """
    [DEFINITIVE, ROBUST VERSION]
    Loads a full ViPCLightningModule using the official `load_from_checkpoint`,
    which correctly handles model weights, hyperparameters, and custom attributes
    like normalizers and kinematic limits.
    """
    log.info(f"Loading Lightning checkpoint from: {checkpoint_path}")
    
    # Use the official PL method to load. It will initialize the module with
    # the saved hyper_parameters and then load the state_dict correctly.
    lightning_model = ViPCLightningModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device
    )
    
    # The `on_load_checkpoint` hook in the module has already restored the
    # normalizers and limits into the lightning_model attributes.
    
    # Restore the EMA weights, which are essential for stable inference
    if lightning_model.ema:
        # Replace the online model with the EMA model for evaluation
        lightning_model.model = lightning_model.ema.ema_model
        log.info("EMA weights successfully applied for inference.")
    else:
        log.warning("Checkpoint does not contain EMA state. Using standard weights.")
        
    # Set to evaluation mode
    lightning_model.eval()
    
    log.info("LightningModule loaded successfully and set to eval mode.")
    return lightning_model



def get_current_task_phase(obs: Dict[str, np.ndarray], proximity_threshold: float = 0.1) -> int:
    """
    Programmatically determines the current TaskPhase based on ground-truth
    state from the environment, mirroring the data labeling logic.
    """
    is_grasped = obs['is_grasped'][0] > 0.5
    ee_pos = obs['ee_pose_world'][:3]
    obj_pos = obs['object_pos_world']
    goal_pos = obs['goal_pos_world']

    dist_to_obj = np.linalg.norm(ee_pos - obj_pos)
    dist_to_goal = np.linalg.norm(ee_pos - goal_pos)

    # This logic should be a direct copy of your final data labeling script's logic
    if not is_grasped:
        if dist_to_obj < proximity_threshold:
            return 0 # APPROACHING_OBJECT (close enough to consider descent)
        else:
            return 0 # APPROACHING_OBJECT
    else: # is_grasped is True
        if dist_to_goal < proximity_threshold:
            return 3 # PLACING_OBJECT
        else:
            return 2 # TRANSPORTING_OBJECT_TO_GOAL

# --- Main Evaluation Function ---

@hydra.main(version_base=None, config_path="configs", config_name="evaluate_vip_c_config")
def evaluate(cfg: DictConfig):
    """
    Main evaluation function driven by Hydra.
    """
    log.info("--- ViP-C Hierarchical Visual Evaluation ---")
    log.info(f"Full evaluation config:\n{OmegaConf.to_yaml(cfg)}")
    
    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl.seed_everything(cfg.seed, workers=True)

    # --- 2. Load Model, Env, and Scheduler ---
    lightning_module = load_lightning_module_from_checkpoint(cfg.checkpoint_path, device)
    model = lightning_module.model # The EMA model
    
    # Extract the essential components for inference
    action_normalizer = lightning_module.action_normalizer
    proprio_normalizer = lightning_module.proprio_normalizer
    joint_limits_low = lightning_module.joint_limits_low
    joint_limits_high = lightning_module.joint_limits_high
    noise_scheduler = lightning_module.noise_scheduler
    
    # Get model-specific horizons
    obs_horizon = lightning_module.cfg.model.controller_cfg.obs_horizon
    
    log.info("Initializing PandaEnv for evaluation...")
    env = PandaEnv(xml_path=lightning_module.cfg.env.xml_path, control_mode="delta")
    
    # --- 3. Setup Image Transforms ---
    transform_planner_img = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_controller_primary = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    transform_controller_wrist = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    
    # --- 4. Setup Video Recording ---
    video_path = Path(cfg.output_video)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    frame_test = env.render()
    H, W, _ = frame_test.shape
    video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (W, H))
    log.info(f"Recording video to: {video_path}")

    # --- 5. Run Hierarchical Evaluation Loop ---
    for ep_idx in tqdm(range(cfg.num_episodes), desc="Evaluating Episodes"):
        log.info(f"--- Starting Evaluation Episode {ep_idx + 1}/{cfg.num_episodes} ---")
        
        # --- Episode Reset & Goal Image Capture ---
        obs, _ = env.reset(seed=cfg.seed + ep_idx)
        goal_image_np = get_goal_image(env, obs) # Using the same robust helper
        
        # Preprocess static inputs for the Planner
        goal_image_tensor = transform_planner_img(Image.fromarray(goal_image_np)).to(device).unsqueeze(0)
        
        # Initialize observation history deque
        obs_history = collections.deque(maxlen=obs_horizon)
        for _ in range(obs_horizon):
            obs_history.append(obs)
            
        current_phase = -1
        # Initialize subgoal_coord to None to ensure the planner runs on the first step.
        subgoal_coord = None

        # Initialize implicit_subgoal to a placeholder.
        implicit_subgoal_dim = model.planner.implicit_head[-1].out_features
        implicit_subgoal = torch.zeros((1, implicit_subgoal_dim), device=device)
        heatmap = torch.zeros((1, 1, 56, 56), device=device) # Placeholder heatmap
        
        step_iterator = tqdm(range(env.max_episode_steps), desc=f"Episode {ep_idx+1}", leave=False)
        for step_count in step_iterator:
            
            # --- 5a. Hierarchical Planning Step (The "Foreman") ---
            new_phase = get_current_task_phase(obs)
            # Re-plan if phase changes OR if it's the very first step (subgoal_coord is None)
            if new_phase != current_phase or subgoal_coord is None:
                log.info(f"Phase change detected at step {step_count}: {current_phase} -> {new_phase}")
                current_phase = new_phase
                
                # This is the "Re-Plan" trigger. We run the Planner.
                current_image_tensor = transform_planner_img(Image.fromarray(obs['image_primary'])).to(device).unsqueeze(0)
                task_phase_tensor = torch.tensor([current_phase], dtype=torch.long, device=device)
                
                with torch.no_grad():
                    # The .plan() method gives us everything we need for diagnostics
                    heatmap, implicit_subgoal, subgoal_coord = model.plan(
                        current_image=current_image_tensor,
                        goal_image=goal_image_tensor,
                        task_phase=task_phase_tensor
                    )
                log.info(f"Planner generated new subgoal: {subgoal_coord.cpu().numpy()}")
            
            # --- 5b. Prepare Controller Batch ---
            proprio_hist = torch.from_numpy(np.stack([h['proprio'] for h in obs_history])).float().to(device)
            primary_hist = torch.stack([transform_controller_primary(Image.fromarray(h['image_primary'])) for h in obs_history]).to(device)
            wrist_hist = torch.stack([transform_controller_wrist(Image.fromarray(h['image_wrist'])) for h in obs_history]).to(device)

            # Symmetrical "raw -> norm" pipeline for inference
            normalized_proprio = proprio_normalizer.normalize(proprio_hist)
            
            controller_obs_hist = {
                'image_primary': primary_hist.unsqueeze(0),
                'image_wrist': wrist_hist.unsqueeze(0),
                'proprio': normalized_proprio.unsqueeze(0)
            }
            
            # --- 5c. Get Action from Controller (The "Worker") ---
            with torch.no_grad():
                # The .act() method is fully kinematics- and normalization-aware
                action_chunk_raw = model.act(
                    observation_history=controller_obs_hist,
                    predicted_subgoal_coord=subgoal_coord,
                    implicit_subgoal=implicit_subgoal,
                    noise_scheduler=noise_scheduler,
                    num_inference_steps=cfg.inference_steps,
                    action_normalizer=action_normalizer,
                    proprio_normalizer=proprio_normalizer, # Pass for symmetry, even if unused in `act`
                    joint_limits_low=joint_limits_low,
                    joint_limits_high=joint_limits_high
                )
            
            action = action_chunk_raw[0, 0, :].cpu().numpy()
            
            # --- 5d. Step Environment & Record ---
            obs, reward, terminated, truncated, info = env.step(action)
            obs_history.append(obs)
            
            # Diagnostic Visualization
            frame_rgb = env.render()
            heatmap_resized = cv2.resize(heatmap[0, 0].cpu().numpy(), (frame_rgb.shape[1], frame_rgb.shape[0]))
            heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # Blend the heatmap onto the frame
            overlay_frame = cv2.addWeighted(frame_rgb, 0.6, heatmap_colored, 0.4, 0)
            
            video_writer.write(cv2.cvtColor(overlay_frame, cv2.COLOR_RGB2BGR))
            
            if terminated or truncated:
                log.info(f"Episode finished after {step_count + 1} steps.")
                break

    # --- 6. Cleanup ---
    video_writer.release()
    env.close()
    log.info(f"--- Evaluation Complete. Video saved to: {video_path.resolve()} ---")

if __name__ == "__main__":
    evaluate()