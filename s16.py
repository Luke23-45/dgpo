#!/usr/bin/env python3

"""
UHP (Unified Hierarchical Policy) v2.0 SOTA Evaluation Script

This script provides a comprehensive, diagnostic-rich evaluation for the
advanced UHP v2.0 architecture. It loads a trained model, its normalizers, and
kinematic limits from a checkpoint and executes the policy in the PandaEnv,
recording a video with rich visual overlays.

Key SOTA Features:
1.  **Hierarchical Control Loop:** Correctly implements the "Plan -> Act -> Re-plan"
    dialogue. The high-level Sequencer is called when the task phase changes
    to produce a dense `subgoal_embedding`.
2.  **Full State Restoration:** Robustly loads the complete training state by
    restoring the entire UHPLightningModule, including the model (EMA weights),
    action/proprio normalizers, and kinematic limits.
3.  **Correct "Norm-to-Raw" Workflow:** Implements the full, symmetrical data
    flow required for inference: raw env data is normalized for the model, and
    normalized model outputs are un-normalized and clamped for the env.
4.  **Oracle Task Phase:** Programmatically determines the current `TaskPhase` at
    each step using ground-truth state, mirroring the data labeling logic and
    providing a perfect signal to the Sequencer.
5.  **Diagnostic Visualization:** Overlays the Sequencer's predicted subgoal
    heatmap directly onto the recorded video, providing invaluable insight into
    the model's high-level spatial reasoning in real-time.

Usage:
    python -m evaluate.evaluate_uhp checkpoint_path=/path/to/your/uhp.ckpt
    
"""

import logging
import collections
from pathlib import Path
import cv2
import hydra
import numpy as np
import torch
import pytorch_lightning as pl
import mujoco
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from typing import Dict
import sys

# --- Add Project Root to `sys.path` for Robust Imports ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# --- Import Project Modules ---
from envs.panda_env import PandaEnv
from models.uhp import UHP_Orchestrator, LinearNormalizer
from train.train_uhp import UHPLightningModule
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(name)s] - %(message)s')
log = logging.getLogger(__name__)


def get_goal_image(env: PandaEnv, obs: dict) -> np.ndarray:
    """
    [SOTA, STATE-RESTORATION VERSION]
    Creates a 'goal_image' by saving the current state, temporarily moving the
    object to the goal position, rendering, and then perfectly restoring the
    original state. This is the only robust way to generate a goal image.
    """
    original_mj_state = env.get_mj_state()
    try:
        goal_pos_world = obs['goal_pos_world']
        qpos_addr = env.model.jnt_qposadr[env.object_joint_id]
        env.data.qpos[qpos_addr:qpos_addr + 3] = goal_pos_world
        
        if 'goal_orn_world' in obs:
             quat_xyzw = obs['goal_orn_world']
             quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
             env.data.qpos[qpos_addr + 3:qpos_addr + 7] = quat_wxyz

        mujoco.mj_forward(env.model, env.data)
        goal_img_np = env.render(camera_name="fixed_camera")
    finally:
        env.set_mj_state(original_mj_state)
    return goal_img_np


def load_lightning_module_from_checkpoint(
    checkpoint_path: str, device: torch.device
) -> UHPLightningModule:
    """
    [SOTA, "SINGLE SOURCE OF TRUTH" VERSION]
    Loads a full UHPLightningModule, which correctly handles model weights,
    hyperparameters, and all custom state like normalizers and kinematic limits.
    """
    log.info(f"Loading Lightning checkpoint from: {checkpoint_path}")
    lightning_model = UHPLightningModule.load_from_checkpoint(
        checkpoint_path, map_location=device
    )
    
    if hasattr(lightning_model, 'ema') and lightning_model.ema:
        lightning_model.model = lightning_model.ema.ema_model
        log.info("EMA weights successfully applied for inference.")
    else:
        log.warning("Checkpoint does not contain EMA state. Using standard weights.")
        
    lightning_model.eval()
    log.info("LightningModule loaded successfully and set to eval mode.")
    return lightning_model


def get_current_task_phase(obs: Dict[str, np.ndarray], proximity_threshold: float = 0.08) -> int:
    """
    [ORACLE VERSION]
    Programmatically determines the current TaskPhase based on ground-truth
    state from the environment, perfectly mirroring the data labeling logic.
    """
    is_grasped = obs.get('is_grasped', [0.0])[0] > 0.5
    ee_pos = obs['ee_pose_world'][:3]
    obj_pos = obs['object_pos_world']
    goal_pos = obs['goal_pos_world']

    dist_ee_to_obj = np.linalg.norm(ee_pos - obj_pos)
    dist_obj_to_goal = np.linalg.norm(obj_pos - goal_pos)

    if not is_grasped:
        # Phase 0: Approaching to grasp
        return 0
    else: # is_grasped is True
        if dist_obj_to_goal < proximity_threshold:
            # Phase 3: Object is near the goal, ready for placement.
            return 3
        else:
            # Phase 2: Object is grasped and being transported to the goal.
            return 2


@hydra.main(version_base=None, config_path="./configs", config_name="evaluate_uhp_config")
def evaluate(cfg: DictConfig):
    log.info("--- UHP v2.0 Hierarchical Visual Evaluation ---")
    log.info(f"Full evaluation config:\n{OmegaConf.to_yaml(cfg)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl.seed_everything(cfg.seed, workers=True)

    # --- 1. Load the "Single Source of Truth" Lightning Module ---
    lightning_module = load_lightning_module_from_checkpoint(cfg.checkpoint_path, device)
    model: UHP_Orchestrator = lightning_module.model
    
    # --- 2. Unpack All Components from the Loaded Module ---
    action_normalizer = lightning_module.action_normalizer
    proprio_normalizer = lightning_module.proprio_normalizer
    joint_limits_low = lightning_module.joint_limits_low
    joint_limits_high = lightning_module.joint_limits_high
    noise_scheduler = lightning_module.noise_scheduler
    
    # Use config parameters from the *checkpoint* for maximum reproducibility.
    train_cfg = lightning_module.cfg
    obs_horizon = train_cfg.model.executor_cfg.obs_horizon
    
    env = PandaEnv(xml_path=train_cfg.env.xml_path, control_mode="delta")
    
    # --- 3. Setup Image Transforms ---
    transform_planner_img = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_controller_primary = transforms.Compose([transforms.Resize((224, 224), antialias=True), transforms.ToTensor()])
    transform_controller_wrist = transforms.Compose([transforms.Resize((128, 128), antialias=True), transforms.ToTensor()])
    
    # --- 4. Setup Video Recording ---
    video_path = Path(cfg.output_video)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    frame_test = env.render()
    H, W, _ = frame_test.shape
    video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (W, H))
    
    # --- 5. Run Hierarchical Evaluation Loop ---
    for ep_idx in tqdm(range(cfg.num_episodes), desc="Evaluating Episodes"):
        obs, _ = env.reset(seed=cfg.seed + ep_idx)
        goal_image_np = get_goal_image(env, obs)
        goal_image_tensor = transform_planner_img(Image.fromarray(goal_image_np)).to(device).unsqueeze(0)
        
        obs_history = collections.deque(maxlen=obs_horizon)
        for _ in range(obs_horizon): obs_history.append(obs)
            
        current_phase = -1
        subgoal_embedding = None # Force re-planning on the first step.
        heatmap = torch.zeros((1, 1, 56, 56), device=device)

        step_iterator = tqdm(range(env.max_episode_steps), desc=f"Episode {ep_idx+1}", leave=False)
        for step_count in step_iterator:
            
            # --- 5a. Hierarchical Planning Step ("The Sequencer") ---
            new_phase = get_current_task_phase(obs)
            if new_phase != current_phase or subgoal_embedding is None:
                log.info(f"Phase change at step {step_count}: {current_phase} -> {new_phase}. Re-planning...")
                current_phase = new_phase
                
                current_image_tensor = transform_planner_img(Image.fromarray(obs['image_primary'])).to(device).unsqueeze(0)
                task_phase_tensor = torch.tensor([current_phase], dtype=torch.long, device=device)
                
                # The UHP `plan` method returns the command and the viz heatmap.
                subgoal_embedding, heatmap = model.plan(
                    current_image=current_image_tensor,
                    goal_image=goal_image_tensor,
                    task_phase=task_phase_tensor
                )
            
            # --- 5b. Prepare Executor Batch (Controller Inputs) ---
            proprio_hist = torch.from_numpy(np.stack([h['proprio'] for h in obs_history])).float().to(device)
            primary_hist = torch.stack([transform_controller_primary(Image.fromarray(h['image_primary'])) for h in obs_history]).to(device)
            wrist_hist = torch.stack([transform_controller_wrist(Image.fromarray(h['image_wrist'])) for h in obs_history]).to(device)
            
            controller_obs_hist = {
                'image_primary': primary_hist.unsqueeze(0),
                'image_wrist': wrist_hist.unsqueeze(0),
                'proprio': proprio_hist.unsqueeze(0)
            }
            
            # --- 5c. Get Action from Executor ("The Controller") ---
            action_chunk_raw = model.act(
                observation_history=controller_obs_hist,
                subgoal_embedding=subgoal_embedding,
                noise_scheduler=noise_scheduler,
                num_inference_steps=cfg.inference.inference_steps,
                action_normalizer=action_normalizer,
                proprio_normalizer=proprio_normalizer,
                joint_limits_low=joint_limits_low,
                joint_limits_high=joint_limits_high
            )
            action = action_chunk_raw[0, 0, :].cpu().numpy()
            
            # --- 5d. Step Environment & Record ---
            obs, reward, terminated, truncated, info = env.step(action)
            obs_history.append(obs)
            
            frame_rgb = env.render()
            heatmap_resized = cv2.resize(heatmap[0, 0].cpu().numpy(), (W, H))
            heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
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