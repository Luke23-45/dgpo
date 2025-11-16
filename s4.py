#!/usr/bin/env python3

"""
UHP (Unified Hierarchical Policy) SOTA Evaluation Script

This script provides a comprehensive, diagnostic-rich evaluation for the
hierarchical UHP policy. It is a definitive, SOTA implementation adapted
from the battle-tested Ego-Planner evaluation script.

Key SOTA Features:
1.  **Single Source of Truth Loading:** Loads the full UHPLightningModule and
    derives all model and training parameters directly from the checkpoint's
    stored hyperparameters. This prevents any configuration mismatch.
2.  **Correct EMA Weight Extraction:** Robustly extracts and uses the Exponential
    Moving Average (EMA) weights for inference, which is critical for the
    stability and performance of diffusion models.
3.  **Hierarchical Control Loop:** Correctly implements the "Plan -> Act -> Re-plan"
    dialogue, calling the state-aware Sequencer only when the task phase changes.
4.  **Reproducible Environment Setup:** Explicitly disables domain randomization and
    provides an option for a static evaluation scene (fixed seed) for consistent
    debugging and analysis.
5.  **Correct Observation History Handling:** Correctly pads the initial
    observation history to match the successful pattern from Ego-Planner.
6.  **Robust State Restoration & Diagnostic Logging:** Loads all necessary components
    (normalizers, kinematic limits) from the LightningModule and logs detailed
    per-step data to a CSV file.

Usage:
    python evaluate_uhp.py \
        checkpoint_path=/path/to/your/uhp_model.ckpt \
        output_video=uhp_evaluation.mp4 \
        num_episodes=5
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
import pytorch_lightning as pl
import csv
# --- SOTA: Add project root to path to ensure modules are found ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# --- Import Project Modules ---
from envs.panda_env import PandaEnv
from train.train_uhp import UHPLightningModule
from models.uhp import UHP_Orchestrator, LinearNormalizer

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(name)s] - %(message)s')
log = logging.getLogger(__name__)


def get_goal_image(env: PandaEnv, obs: dict) -> np.ndarray:
    """
    [DEFINITIVE, SOTA VERSION]
    Creates a 'goal_image' by saving the current state, teleporting the object
    to the goal position, rendering the scene, and then perfectly restoring the
    original state.
    """
    log.debug("Capturing goal image by temporarily moving object...")
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
    except Exception as e:
        log.error(f"Error manually setting goal pose: {e}", exc_info=True)
        goal_img_np = obs['image_primary']
    finally:
        env.set_mj_state(original_mj_state)
    log.debug("Goal image captured and state restored.")
    return goal_img_np


def get_current_task_phase(obs: Dict[str, np.ndarray],
                           prev_is_grasped: bool,
                           dist_ee_to_obj_threshold: float = 0.04,
                           lift_height_threshold: float = 0.03,
                           dist_obj_to_goal_threshold: float = 0.08
                           ) -> int:
    """
    [SOTA, STATEFUL ORACLE VERSION]
    Programmatically determines the current TaskPhase by mirroring the state
    machine logic of the ScriptedExpert used for data generation.
    """
    is_grasped = obs.get('is_grasped', [0.0])[0] > 0.5
    ee_pos = obs['ee_pose_world'][:3]
    obj_pos = obs['object_pos_world']
    goal_pos = obs['goal_pos_world']
    table_z = 0.4

    dist_ee_to_obj = np.linalg.norm(ee_pos - obj_pos)
    dist_obj_to_goal = np.linalg.norm(obj_pos - goal_pos)
    obj_lift_height = obj_pos[2] - table_z

    if not is_grasped and not prev_is_grasped:
        if dist_ee_to_obj < dist_ee_to_obj_threshold:
            return 1  # Phase 1: Close enough to grasp.
        else:
            return 0  # Phase 0: Approaching the object.
    elif is_grasped and not prev_is_grasped:
        return 1  # Just grasped.
    elif is_grasped and prev_is_grasped:
        if obj_lift_height < lift_height_threshold:
            return 1  # Still in the process of lifting.
        elif dist_obj_to_goal < dist_obj_to_goal_threshold:
            return 3  # Phase 3: Arrived at the goal, ready to place.
        else:
            return 2  # Phase 2: Transporting.
    elif not is_grasped and prev_is_grasped:
        return 4  # Phase 4: Just released, retracting.
    return 0


def _prepare_for_csv(data: Any) -> List[float]:
    """Robustly flattens data for CSV logging."""
    if data is None: return []
    if isinstance(data, torch.Tensor): data = data.detach().cpu().numpy()
    if isinstance(data, np.ndarray): return data.flatten().tolist()
    if isinstance(data, (int, float)): return [float(data)]
    return []


# --- Main Evaluation Function ---

@hydra.main(version_base=None, config_path="./configs", config_name="x")
def evaluate(cfg: DictConfig):
    log.info("--- UHP v3.0 State-Aware Visual Evaluation ---")
    log.info(f"Full evaluation config:\n{OmegaConf.to_yaml(cfg)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl.seed_everything(cfg.seed, workers=True)

    # --- 1. Load the "Single Source of Truth" ---
    # This loads the entire training state, including the original config.
    lightning_module = UHPLightningModule.load_from_checkpoint(cfg.checkpoint_path, map_location=device)
    
    # SOTA: The EMA model is the one we should always use for inference.
    if hasattr(lightning_module, 'ema') and lightning_module.ema:
        model: UHP_Orchestrator = lightning_module.ema.ema_model
        log.info("EMA weights successfully extracted and applied for inference.")
    else:
        model: UHP_Orchestrator = lightning_module.model
        log.warning("Checkpoint does not contain EMA state. Using standard weights.")
    
    model.eval()
    
    # --- 2. Unpack All Components from the Single Source of Truth ---
    train_cfg = lightning_module.cfg
    action_normalizer = lightning_module.action_normalizer
    proprio_normalizer = lightning_module.proprio_normalizer
    joint_limits_low = lightning_module.joint_limits_low
    joint_limits_high = lightning_module.joint_limits_high
    noise_scheduler = lightning_module.noise_scheduler
    
    obs_horizon = train_cfg.model.executor_cfg.obs_horizon
    action_dim = train_cfg.model.executor_cfg.action_dim
    action_horizon = train_cfg.model.executor_cfg.action_horizon
    
    # --- 3. Initialize Environment (with settings from loaded config) ---
    log.info("Initializing PandaEnv (Domain Randomization EXPLICITLY DISABLED for eval)...")
    env = PandaEnv(
        xml_path=train_cfg.env.xml_path,
        control_mode="delta",
        enable_domain_randomization=False  # Explicitly disable for consistency.
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

    csv_file, csv_writer = None, None
    if cfg.logging.enable_csv_logging:
        csv_path = Path(cfg.logging.csv_output_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        header = ['ep', 'step', 'phase', 'grasped'] + [f'ee_pos_{ax}' for ax in 'xyz'] + \
                 [f'obj_pos_{ax}' for ax in 'xyz'] + ['sg_norm'] + \
                 [f'act_{i}' for i in range(action_dim)] + \
                 [f'plan_{t}_{i}' for t in range(action_horizon) for i in range(action_dim)]
        csv_writer.writerow(header)
        log.info(f"Logging diagnostic data to: {csv_path}")

    # --- 6. Run Hierarchical Evaluation Loop ---
    for ep_idx in tqdm(range(cfg.num_episodes), desc="Evaluating Episodes"):
        # SOTA: Use static scene for debugging or varied scenes for generalization test.
        episode_seed = cfg.seed if cfg.eval_static_scene else cfg.seed + ep_idx
        obs, _ = env.reset(seed=episode_seed)
        
        goal_image_np = get_goal_image(env, obs)
        goal_image_tensor = transform_planner_img(Image.fromarray(goal_image_np)).to(device).unsqueeze(0)
        
        # SOTA: Correctly pad the initial observation history.
        obs_history = collections.deque(maxlen=obs_horizon)
        for _ in range(obs_horizon):
            obs_history.append(obs)
            
        current_phase = -1
        subgoal_embedding = None
        prev_is_grasped = False

        step_iterator = tqdm(range(env.max_episode_steps), desc=f"Episode {ep_idx+1}", leave=False)
        for step_count in step_iterator:
            # --- Hierarchical Control Logic ---
            new_phase = get_current_task_phase(obs, prev_is_grasped)
            if new_phase != current_phase:
                log.info(f"Step {step_count}: Phase changed from {current_phase} -> {new_phase}. Re-planning...")
                current_phase = new_phase
                current_image_tensor = transform_planner_img(Image.fromarray(obs['image_primary'])).to(device).unsqueeze(0)
                task_phase_tensor = torch.tensor([current_phase], dtype=torch.long, device=device)
                current_proprio_tensor = torch.from_numpy(obs['proprio']).float().to(device).unsqueeze(0)

                subgoal_embedding = model.plan(
                    current_image_tensor, goal_image_tensor, task_phase_tensor, current_proprio_tensor
                )
            
            # --- Prepare Executor Inputs ---
            proprio_hist_tensor = torch.from_numpy(np.stack([h['proprio'] for h in obs_history])).float().to(device).unsqueeze(0)
            primary_hist_tensor = torch.stack([transform_controller_primary(Image.fromarray(h['image_primary'])) for h in obs_history]).to(device).unsqueeze(0)
            wrist_hist_tensor = torch.stack([transform_controller_wrist(Image.fromarray(h['image_wrist'])) for h in obs_history]).to(device).unsqueeze(0)
            controller_obs_hist = {
                'image_primary': primary_hist_tensor,
                'image_wrist': wrist_hist_tensor,
                'proprio': proprio_hist_tensor
            }
            
            # --- Get Action from Policy ---
            action_chunk_raw = model.act(
                controller_obs_hist, subgoal_embedding, noise_scheduler,
                cfg.inference.inference_steps, action_normalizer, proprio_normalizer,
                joint_limits_low, joint_limits_high
            )
            action = action_chunk_raw[0, 0, :].cpu().numpy()
            
            if cfg.logging.enable_csv_logging:
                log_row = _prepare_for_csv(ep_idx) + _prepare_for_csv(step_count) + _prepare_for_csv(current_phase) + \
                          _prepare_for_csv(obs['is_grasped']) + _prepare_for_csv(obs['ee_pose_world'][:3]) + \
                          _prepare_for_csv(obs['object_pos_world']) + _prepare_for_csv(torch.linalg.norm(subgoal_embedding)) + \
                          _prepare_for_csv(action) + _prepare_for_csv(action_chunk_raw)
                csv_writer.writerow(log_row)

            # --- Step Environment and Update State ---
            prev_is_grasped = obs.get('is_grasped', [0.0])[0] > 0.5
            obs, reward, terminated, truncated, info = env.step(action)
            obs_history.append(obs)
            
            if cfg.logging.enable_video:
                frame_rgb = env.render()
                phase_text = f"Phase: {current_phase}"
                cv2.putText(frame_rgb, phase_text, (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                video_writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            
            if terminated or truncated:
                log.info(f"Episode finished after {step_count + 1} steps.")
                break
        
    # --- 7. Cleanup ---
    if video_writer is not None: video_writer.release()
    if csv_file is not None: csv_file.close()
    env.close()
    log.info("--- Evaluation Complete. ---")


if __name__ == "__main__":
    evaluate()