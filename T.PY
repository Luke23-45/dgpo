#!/usr/bin/env python3

"""
UHP (Unified Hierarchical Policy) v2.0 SOTA Evaluation Script

This script provides a comprehensive, diagnostic-rich evaluation for the
advanced UHP v2.0 architecture. It loads a trained model, its normalizers, and
kinematic limits from a checkpoint and executes the policy in the PandaEnv,
recording a video with rich visual overlays and a detailed per-timestep CSV log.

Key SOTA Features:
... (all previous features are retained) ...
6.  **Diagnostic Data Logging:** Optionally saves a detailed, per-timestep CSV
    file containing the complete "state of mind" of the policy—ground truth
    state, planner outputs, and controller actions—for deep offline analysis.

Usage:
    python -m evaluate.evaluate_uhp checkpoint_path=/path/to/your/uhp.ckpt logging.enable_csv_logging=true
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
from typing import Dict, Any, List
import sys
import csv  # SOTA PATCH: Import the csv module

# --- Add Project Root to `sys.path` for Robust Imports ---
# ... (rest of imports are the same)
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


# --- [SOTA PATCH: Diagnostic Data Formatting Helper] ---
def _prepare_for_csv(data: Any) -> List[float]:
    """A robust helper to convert tensors, arrays, or scalars into a flat list of floats for CSV logging."""
    if data is None:
        return []
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if isinstance(data, np.ndarray):
        return data.flatten().tolist()
    if isinstance(data, (int, float)):
        return [float(data)]
    return []


# All other helper functions (`get_goal_image`, `load_lightning_module_from_checkpoint`, `get_current_task_phase`) remain unchanged.
# ...
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


@hydra.main(version_base=None, config_path="../configs", config_name="evaluate_uhp_config")
def evaluate(cfg: DictConfig):
    log.info("--- UHP v2.0 Hierarchical Visual Evaluation ---")
    log.info(f"Full evaluation config:\n{OmegaConf.to_yaml(cfg)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl.seed_everything(cfg.seed, workers=True)

    # --- 1. Load the "Single Source of Truth" ---
    lightning_module = load_lightning_module_from_checkpoint(cfg.checkpoint_path, device)
    model: UHP_Orchestrator = lightning_module.model
    
    # --- 2. Unpack All Components ---
    action_normalizer = lightning_module.action_normalizer
    proprio_normalizer = lightning_module.proprio_normalizer
    joint_limits_low = lightning_module.joint_limits_low
    joint_limits_high = lightning_module.joint_limits_high
    noise_scheduler = lightning_module.noise_scheduler
    train_cfg = lightning_module.cfg
    obs_horizon = train_cfg.model.executor_cfg.obs_horizon
    action_dim = train_cfg.model.executor_cfg.action_dim
    action_horizon = train_cfg.model.executor_cfg.action_horizon
    
    env = PandaEnv(xml_path=train_cfg.env.xml_path, control_mode="delta")
    
    # --- 3. Setup Image Transforms ---
    transform_planner_img = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_controller_primary = transforms.Compose([transforms.Resize((224, 224), antialias=True), transforms.ToTensor()])
    transform_controller_wrist = transforms.Compose([transforms.Resize((128, 128), antialias=True), transforms.ToTensor()])
    
    # --- 4. Setup Video & CSV Recording ---
    video_writer = None
    if cfg.logging.enable_video:
        video_path = Path(cfg.logging.video_output_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        frame_test = env.render()
        H, W, _ = frame_test.shape
        video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (W, H))
        log.info(f"Recording video to: {video_path}")

    csv_file = None
    csv_writer = None
    if cfg.logging.enable_csv_logging:
        csv_path = Path(cfg.logging.csv_output_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        
        # --- SOTA PATCH: Dynamically generate the header ---
        header = ['episode_idx', 'step', 'oracle_task_phase']
        header += [f'gt_ee_pos_{ax}' for ax in 'xyzw_quat_xyzw']
        header += ['gt_is_grasped']
        header += [f'gt_object_pos_{ax}' for ax in 'xyz']
        header += ['planner_subgoal_emb_norm', 'planner_heatmap_max_val']
        header += [f'planner_heatmap_peak_{ax}' for ax in 'uv']
        header += [f'action_executed_{i}' for i in range(action_dim)]
        for t in range(action_horizon):
            header += [f'action_pred_h{t}_d{i}' for i in range(action_dim)]
        csv_writer.writerow(header)
        log.info(f"Logging diagnostic data to: {csv_path}")

    # --- 5. Run Hierarchical Evaluation Loop ---
    for ep_idx in tqdm(range(cfg.num_episodes), desc="Evaluating Episodes"):
        obs, _ = env.reset(seed=cfg.seed + ep_idx)
        goal_image_np = get_goal_image(env, obs)
        goal_image_tensor = transform_planner_img(Image.fromarray(goal_image_np)).to(device).unsqueeze(0)
        
        obs_history = collections.deque(maxlen=obs_horizon)
        for _ in range(obs_horizon): obs_history.append(obs)
            
        current_phase = -1
        subgoal_embedding = None
        heatmap = torch.zeros((1, 1, 56, 56), device=device)
        episode_csv_data = []

        step_iterator = tqdm(range(env.max_episode_steps), desc=f"Episode {ep_idx+1}", leave=False)
        for step_count in step_iterator:
            
            new_phase = get_current_task_phase(obs)
            if new_phase != current_phase or subgoal_embedding is None:
                current_phase = new_phase
                current_image_tensor = transform_planner_img(Image.fromarray(obs['image_primary'])).to(device).unsqueeze(0)
                task_phase_tensor = torch.tensor([current_phase], dtype=torch.long, device=device)
                subgoal_embedding, heatmap = model.plan(current_image_tensor, goal_image_tensor, task_phase_tensor)
            
            # Prepare Executor inputs
            proprio_hist = torch.from_numpy(np.stack([h['proprio'] for h in obs_history])).float().to(device)
            primary_hist = torch.stack([transform_controller_primary(Image.fromarray(h['image_primary'])) for h in obs_history]).to(device)
            wrist_hist = torch.stack([transform_controller_wrist(Image.fromarray(h['image_wrist'])) for h in obs_history]).to(device)
            controller_obs_hist = {'image_primary': primary_hist.unsqueeze(0), 'image_wrist': wrist_hist.unsqueeze(0), 'proprio': proprio_hist.unsqueeze(0)}
            
            action_chunk_raw = model.act(
                controller_obs_hist, subgoal_embedding, noise_scheduler,
                cfg.inference_steps, action_normalizer, proprio_normalizer,
                joint_limits_low, joint_limits_high
            )
            action = action_chunk_raw[0, 0, :].cpu().numpy()
            
            # --- [SOTA PATCH: Data Logging] ---
            if cfg.logging.enable_csv_logging:
                heatmap_peak_val, heatmap_peak_idx = torch.max(heatmap.flatten(), 0)
                peak_y, peak_x = np.unravel_index(heatmap_peak_idx.cpu().numpy(), heatmap.shape[2:])
                
                log_row = (
                    _prepare_for_csv(ep_idx) +
                    _prepare_for_csv(step_count) +
                    _prepare_for_csv(current_phase) +
                    _prepare_for_csv(obs['ee_pose_world']) +
                    _prepare_for_csv(obs['is_grasped']) +
                    _prepare_for_csv(obs['object_pos_world']) +
                    _prepare_for_csv(torch.linalg.norm(subgoal_embedding)) +
                    _prepare_for_csv(heatmap_peak_val) +
                    _prepare_for_csv(np.array([peak_x, peak_y])) + # uv
                    _prepare_for_csv(action) +
                    _prepare_for_csv(action_chunk_raw)
                )
                episode_csv_data.append(log_row)

            obs, reward, terminated, truncated, info = env.step(action)
            obs_history.append(obs)
            
            if cfg.logging.enable_video:
                frame_rgb = env.render()
                heatmap_resized = cv2.resize(heatmap[0, 0].cpu().numpy(), (frame_rgb.shape[1], frame_rgb.shape[0]))
                heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                overlay_frame = cv2.addWeighted(frame_rgb, 0.6, heatmap_colored, 0.4, 0)
                video_writer.write(cv2.cvtColor(overlay_frame, cv2.COLOR_RGB2BGR))
            
            if terminated or truncated:
                log.info(f"Episode finished after {step_count + 1} steps.")
                break
        
        # --- [SOTA PATCH: Buffered Write to CSV] ---
        if cfg.logging.enable_csv_logging and csv_writer is not None:
            csv_writer.writerows(episode_csv_data)
            log.info(f"Wrote {len(episode_csv_data)} rows to CSV for episode {ep_idx}.")

    # --- 6. Cleanup ---
    if video_writer is not None: video_writer.release()
    if csv_file is not None: csv_file.close()
    env.close()
    log.info("--- Evaluation Complete. ---")

if __name__ == "__main__":
    evaluate()