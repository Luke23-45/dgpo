#!/usr/bin/env python3

"""
UHP (Unified Hierarchical Policy) SOTA Live Evaluation Script

This script provides the definitive, closed-loop evaluation for the UHP policy.
It runs the trained agent live in the PandaEnv simulation.

Key SOTA Features:
1.  **Ground-Truth Goal Images:** Loads the validation dataset  purely as a
    source of episode seeds and ground-truth goal images,
    eliminating the need for the "teleport-and-render" hack.
2.  **Correct Normalization Pipeline:** Implements the critical train/test fix:
    - Raw proprioception is *normalized* before being passed to `model.plan()`.
    - Raw proprioception history is passed to `model.act()`, which handles
      normalization internally.
3.  **Robust Oracle:** Uses the ground-truth state-based oracle
    (`get_current_task_phase`) to provide the "perfect" phase label at
    each step, just like in training.
4.  **Live Closed-Loop Execution:** The policy's actions are executed
    in the environment[cite: 429], and the resulting observation is used
    for the next planning step, testing the policy's true stability.
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
from typing import Dict, Any

import pytorch_lightning as pl

# --- Import Project Modules ---
# Assume these are in the python path
from envs.panda_env import PandaEnv
from utils.expert_dataset import ExpertTrajectoryDataset

# These modules are from your *original* UHP codebase (akl.txt)
# We must assume they are available to run this script.
try:
    from train.train_uhp import UHPLightningModule
    from models.uhp import UHP_Orchestrator, LinearNormalizer
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
except ImportError as e:
    print("="*80)
    print(f"Error: Could not import UHP model files: {e}")
    print("This script MUST be run from an environment that has access to")
    print("UHPLightningModule, UHP_Orchestrator, and LinearNormalizer.")
    print("="*80)
    raise

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(name)s] - %(message)s')
log = logging.getLogger(__name__)


def load_lightning_module_from_checkpoint(
    checkpoint_path: str, device: torch.device
) -> UHPLightningModule:
    """
    [SOTA, "SINGLE SOURCE OF TRUTH" VERSION]
    Loads a full UHPLightningModule, which correctly handles model weights,
    hyperparameters, and all custom state like normalizers and kinematic limits.
    
    (This function is from your original akl.txt)
    """
    log.info(f"Loading Lightning checkpoint from: {checkpoint_path}")
    lightning_model = UHPLightningModule.load_from_checkpoint(
        checkpoint_path, map_location=device
    )
    
    if hasattr(lightning_model, 'ema') and lightning_model.ema:
        lightning_model.model = lightning_model.ema.ema_model
        log.info("EMA weights successfully extracted and applied for inference.")
    else:
        log.warning("Checkpoint does not contain EMA state. Using standard weights.")
        
    lightning_model.eval()
    log.info("LightningModule loaded and set to eval mode.")
    return lightning_model


def get_current_task_phase(obs: Dict[str, np.ndarray],
                           prev_is_grasped: bool,
                           dist_ee_to_obj_threshold: float = 0.04,
                           lift_height_threshold: float = 0.03,
                           dist_obj_to_goal_threshold: float = 0.08
                           ) -> int:
    """
    [SOTA ORACLE]
    Calculates the current task phase based on ground-truth environment state.
    This logic perfectly mirrors the expert's state machine.
    
    (This function is from your original akl.txt)
    """
    is_grasped = obs.get('is_grasped', [0.0])[0] > 0.5
    ee_pos = obs['ee_pose_world'][:3]
    obj_pos = obs['object_pos_world']
    goal_pos = obs['goal_pos_world']
    # Use the Z-height from the new env file for accuracy [cite: 251]
    table_z = PandaEnv.OBJECT_Z_HEIGHT - 0.02 # approx table height

    dist_ee_to_obj = np.linalg.norm(ee_pos - obj_pos)
    dist_obj_to_goal = np.linalg.norm(obj_pos - goal_pos)
    obj_lift_height = obj_pos[2] - table_z

    if not is_grasped and not prev_is_grasped:
        if dist_ee_to_obj < dist_ee_to_obj_threshold:
            return 1 # Phase 1: Close enough to grasp
        else:
            return 0 # Phase 0: Approaching
    elif is_grasped and not prev_is_grasped:
        return 1 # Phase 1: Just grasped
    elif is_grasped and prev_is_grasped:
        if obj_lift_height < lift_height_threshold:
            return 1 # Phase 1: Lifting
        elif dist_obj_to_goal < dist_obj_to_goal_threshold:
            return 3 # Phase 3: Arrived at goal
        else:
            return 2 # Phase 2: Transporting
    elif not is_grasped and prev_is_grasped:
        return 4 # Phase 4: Just released (retracting)
    
    return 0 # Default fallback


@hydra.main(version_base=None, config_path="./configs", config_name="evaluate_uhp_live_config")
def evaluate(cfg: DictConfig):
    log.info("--- UHP v3.0 SOTA Live Evaluation ---")
    log.info(f"Full evaluation config:\n{OmegaConf.to_yaml(cfg)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl.seed_everything(cfg.seed, workers=True)

    # --- 1. Load the "Single Source of Truth" Checkpoint ---
    lightning_module = load_lightning_module_from_checkpoint(cfg.checkpoint_path, device)
    
    # --- 2. Unpack All Components ---
    model: UHP_Orchestrator = lightning_module.model.to(device)
    action_normalizer = lightning_module.action_normalizer
    proprio_normalizer = lightning_module.proprio_normalizer
    joint_limits_low = lightning_module.joint_limits_low.to(device)
    joint_limits_high = lightning_module.joint_limits_high.to(device)
    noise_scheduler = lightning_module.noise_scheduler
    train_cfg = lightning_module.cfg
    
    # Get horizons from the *training config* for perfect consistency
    obs_horizon = train_cfg.model.executor_cfg.obs_horizon
    action_horizon = train_cfg.model.executor_cfg.action_horizon
    
    # --- 3. Load Environment (from aaaaa.txt) ---
    # We MUST use 'delta' control mode to match the dataset actions [cite: 256, 431]
    env = PandaEnv(
        xml_path=train_cfg.env.xml_path, 
        control_mode="delta",
        enable_domain_randomization=False # Turn off DR for consistent eval
    )
    
    # --- 4. Load Dataset (from aaaaa.txt) ---
    # We *only* use this to get seeds and ground-truth goal images
    log.info(f"Loading dataset index from: {cfg.dataset_path}")
    val_dataset = ExpertTrajectoryDataset(
        demo_path=cfg.dataset_path,
        observation_horizon=obs_horizon,
        action_horizon=action_horizon,
    )
    num_dataset_eps = val_dataset.get_num_episodes() 
    log.info(f"Found {num_dataset_eps} episodes in dataset index.")

    # --- 5. Setup Image Transforms ---
    # (These are from your original akl.txt)
    transform_planner_img = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_controller_primary = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor()
    ])
    transform_controller_wrist = transforms.Compose([
        transforms.Resize((128, 128), antialias=True),
        transforms.ToTensor()
    ])
    
    # --- 6. Setup Video Recording ---
    video_writer = None
    if cfg.output_video:
        video_path = Path(cfg.output_video)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        # Get render shape from the *new* env's obs space [cite: 330]
        H, W, _ = env.observation_space['image_primary'].shape
        video_writer = cv2.VideoWriter(
            str(video_path), 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            cfg.video_fps, 
            (W, H)
        )
        log.info(f"Recording video to: {video_path}")

    # --- 7. Run Hierarchical Evaluation Loop ---
    num_eval_episodes = min(cfg.num_episodes, num_dataset_eps)
    log.info(f"Running evaluation for {num_eval_episodes} episodes...")
    
    for ep_idx in tqdm(range(num_eval_episodes), desc="Evaluating Episodes"):
        # --- A. SOTA Reset and DYNAMIC Goal Data Fetching ---
        dataset_ep_meta = val_dataset.episode_metadata[ep_idx]
        seed = dataset_ep_meta.get("seed")
        if seed is None:
            seed = cfg.seed + ep_idx
            
        # 1. Reset Env to the *exact* seed from the dataset
        obs, _ = env.reset(seed=seed)

        # 2. [SOTA PATCH] Load arrays for DYNAMIC goal lookup
        # We must load all images and the index map for this episode
        # to mimic the training pipeline
        log.debug(f"Loading goal modalities for episode {ep_idx}...")
        try:
            def get_mod(modality_name: str):
                """Helper to load a full modality array for the episode."""
                meta = dataset_ep_meta["modalities"][modality_name]
                return val_dataset._get_full_modality_array(
                    key=meta["key"], compression=meta["compression"],
                    dtype_str=meta["dtype"], shape_list=tuple(meta["shape"])
                )

            # This array tells us which image index to use at each timestep
            all_goal_indices = get_mod("phase_goal_image_indices")
            # This is the full stack of all primary images for the episode
            all_images_primary = get_mod("image_primary")
            
            ep_demo_length = len(all_goal_indices)
            log.debug(f"Loaded {ep_demo_length} goal indices and {len(all_images_primary)} images.")
            
        except KeyError as e:
            log.error(f"FATAL: Dataset for ep {ep_idx} is missing required modalities: {e}")
            log.error("Please re-run utils/dataset_enhancer.py to fix the dataset.")
            continue # Skip this broken episode

        # This line is no longer needed and will FAIL, so it is REMOVED:
        # goal_image_np = val_dataset.get_goal_image(ep_idx) 

        # This tensor is now created *inside* the loop
        goal_image_tensor = None
        
        # --- B. Warm-up Observation History ---
        # (Using simple zero-action steps, as in akl.txt)
        obs_history = collections.deque(maxlen=obs_horizon)
        for _ in range(obs_horizon):
            # The new env's step function [cite: 429] handles zero actions
            obs, _, _, _, _ = env.step(np.zeros(env.action_space.shape))
            obs_history.append(obs)
            
        current_phase = -1
        prev_is_grasped = False
        subgoal_embedding = None

        step_iterator = tqdm(range(env.max_episode_steps), desc=f"Episode {ep_idx+1}", leave=False)
        for step_count in step_iterator:
            
            # --- 1. Get Oracle Phase ---
            new_phase = get_current_task_phase(obs, prev_is_grasped)
            if new_phase != current_phase:
                 log.debug(f"Step {step_count}: Oracle phase changed to {new_phase}.")
            current_phase = new_phase

            # --- 2. Prepare Planner Inputs (CRITICAL FIX) ---
            current_image_tensor = transform_planner_img(
                Image.fromarray(obs['image_primary'])
            ).to(device).unsqueeze(0)

            # --- [SOTA PATCH] DYNAMIC GOAL IMAGE LOOKUP ---
            # This mimics the training data pipeline exactly

            # Clamp step_count in case live episode > demo length
            clamped_step = min(step_count, ep_demo_length - 1)

            # 1. Get the goal *index* for this specific timestep
            current_goal_index = all_goal_indices[clamped_step]

            # 2. Get the goal *image* from that index
            goal_image_np = all_images_primary[current_goal_index]

            # 3. Transform the goal image for the planner
            goal_image_tensor = transform_planner_img(
                Image.fromarray(goal_image_np)
            ).to(device).unsqueeze(0)
            # --- END DYNAMIC GOAL PATCH ---

            # CRITICAL: Normalize proprio for the planner
            current_proprio_raw = torch.from_numpy(obs['proprio']).float().to(device).unsqueeze(0)
            current_proprio_norm = proprio_normalizer.normalize(current_proprio_raw)

            task_phase_tensor = torch.tensor([current_phase], dtype=torch.long, device=device)

            # --- 3. Plan() ---
            # Call `plan` at every step, now with the correct,
            # phase-consistent goal image for this timestep.
            subgoal_embedding = model.plan(
                current_image_tensor,
                goal_image_tensor, # Pass the new DYNAMIC tensor
                task_phase_tensor,
                current_proprio_norm # Pass *normalized*
            )
            
            # --- 4. Prepare Executor Inputs (CRITICAL FIX) ---
            # `act` expects RAW proprio history
            proprio_hist_raw = torch.from_numpy(
                np.stack([h['proprio'] for h in obs_history])
            ).float().to(device).unsqueeze(0)
            
            primary_hist = torch.stack([
                transform_controller_primary(Image.fromarray(h['image_primary']))
                for h in obs_history
            ]).to(device).unsqueeze(0)
            
            wrist_hist = torch.stack([
                transform_controller_wrist(Image.fromarray(h['image_wrist']))
                for h in obs_history
            ]).to(device).unsqueeze(0)

            controller_obs_hist = {
                'image_primary': primary_hist,
                'image_wrist': wrist_hist,
                'proprio': proprio_hist_raw # Pass *raw*
            }
            
            # --- 5. Act() ---
            action_chunk_raw = model.act(
                controller_obs_hist, 
                subgoal_embedding, 
                noise_scheduler,
                cfg.inference.inference_steps, 
                action_normalizer, 
                proprio_normalizer,
                joint_limits_low, 
                joint_limits_high
            )
            action = action_chunk_raw[0, 0, :].cpu().numpy()
            
            # --- 6. Update Oracle State (BEFORE stepping) ---
            prev_is_grasped = obs.get('is_grasped', [0.0])[0] > 0.5
            
            # --- 7. Step Environment ---
            obs, reward, terminated, truncated, info = env.step(action)
            obs_history.append(obs)
            
            # --- 8. Record Video Frame ---
            if video_writer is not None:
                frame_rgb = env.render(camera_name="fixed_camera") 
                # Add diagnostic text
                phase_text = f"Phase: {current_phase}"
                grasp_text = f"Grasped: {'TRUE' if prev_is_grasped else 'FALSE'}"
                cv2.putText(frame_rgb, phase_text, (10, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame_rgb, grasp_text, (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                video_writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            
            if terminated or truncated:
                log.info(f"Episode finished after {step_count + 1} steps.")
                break
        
    # --- 8. Cleanup ---
    if video_writer is not None:
        video_writer.release()
    env.close() 
    log.info("--- Live Evaluation Complete. ---")

if __name__ == "__main__":
    evaluate()