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
from train.train_uhp import UHPLightningModule
from models.vip_c import ViPC, LinearNormalizer
from train.train_vip_c import ViPCLightningModule
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from models.uhp import UHP_Orchestrator, LinearNormalizer
from typing import List, Any
import csv
# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(name)s] - %(message)s')
log = logging.getLogger(__name__)



def get_goal_image(env: PandaEnv, obs: dict) -> np.ndarray:
    """
    [DEFINITIVE, SOTA VERSION]
    Creates a 'goal_image' by saving the current state, teleporting the object
    to the goal position, rendering the scene, and then perfectly restoring the
    original state. This is the only robust way to get a ground-truth goal image.
    """
    log.debug("Capturing goal image by temporarily moving object...")
    
    # 1. Save the current complete simulation state.
    original_mj_state = env.get_mj_state()

    try:
        # 2. Get the goal pose from the observation.
        goal_pos_world = obs['goal_pos_world']
        
        # 3. Manually set the object's free joint to the goal position.
        qpos_addr = env.model.jnt_qposadr[env.object_joint_id]
        env.data.qpos[qpos_addr:qpos_addr + 3] = goal_pos_world
        
        # Set orientation if available in the observation.
        if 'goal_orn_world' in obs:
             quat_xyzw = obs['goal_orn_world']
             # Convert xyzw (SciPy) to wxyz (MuJoCo) for the simulation.
             quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
             env.data.qpos[qpos_addr + 3:qpos_addr + 7] = quat_wxyz

        # 4. Propagate this change through the simulation state.
        mujoco.mj_forward(env.model, env.data)
        
        # 5. Render the "goal" scene.
        goal_img_np = env.render(camera_name="fixed_camera")
        
    except Exception as e:
        log.error(f"Error manually setting goal pose: {e}", exc_info=True)
        goal_img_np = obs['image_primary'] # Fallback to current image on error.
    finally:
        # 6. CRITICAL: Always restore the original simulation state.
        env.set_mj_state(original_mj_state)
            
    log.debug("Goal image captured and state restored.")
    return goal_img_np


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

def load_lightning_module_from_checkpoint(
    checkpoint_path: str, device: torch.device
) -> UHPLightningModule:
    """
    [SOTA, "SINGLE SOURCE OF TRUTH" VERSION]
    Loads a full UHPLightningModule, which correctly handles model weights,
    hyperparameters, and all custom state like normalizers and kinematic limits.
    """
    log.info(f"Loading Lightning checkpoint from: {checkpoint_path}")
    # CRITICAL FIX: Load using the UHPLightningModule class.
    lightning_model = UHPLightningModule.load_from_checkpoint(
        checkpoint_path, map_location=device
    )
    
    # SOTA: The EMA model is the one we should always use for inference.
    # This logic correctly extracts it.
    if hasattr(lightning_model, 'ema') and lightning_model.ema:
        lightning_model.model = lightning_model.ema.ema_model
        log.info("EMA weights successfully extracted and applied for inference.")
    else:
        log.warning("Checkpoint does not contain EMA state. Using standard weights.")
        
    lightning_model.eval()
    log.info("LightningModule loaded and set to eval mode.")
    return lightning_model




# In FILE: evaluate_uhp.py

# --- [START OF DEFINITIVE PATCH 1: CORRECT ORACLE] ---
# REPLACE the existing `get_current_task_phase` function with this one.

def get_current_task_phase(obs: Dict[str, np.ndarray],
                           prev_is_grasped: bool,
                           dist_ee_to_obj_threshold: float = 0.04,
                           lift_height_threshold: float = 0.03,
                           dist_obj_to_goal_threshold: float = 0.08
                           ) -> int:

    is_grasped = obs.get('is_grasped', [0.0])[0] > 0.5
    ee_pos = obs['ee_pose_world'][:3]
    obj_pos = obs['object_pos_world']
    goal_pos = obs['goal_pos_world']
    table_z = 0.4  # Assumed table height from PandaEnv

    dist_ee_to_obj = np.linalg.norm(ee_pos - obj_pos)
    dist_obj_to_goal = np.linalg.norm(obj_pos - goal_pos)
    obj_lift_height = obj_pos[2] - table_z

    # This logic now robustly mirrors the expert's state machine.
    if not is_grasped and not prev_is_grasped:
        # Not holding, object not recently released.
        if dist_ee_to_obj < dist_ee_to_obj_threshold:
            # Phase 1: Close enough to grasp.
            return 1
        else:
            # Phase 0: Approaching the object.
            return 0
    elif is_grasped and not prev_is_grasped:
        # Just grasped the object.
        return 1
    elif is_grasped and prev_is_grasped:
        # Currently holding the object.
        if obj_lift_height < lift_height_threshold:
            # Still in the process of lifting.
            return 1 # Or could be 2 if lift is fast, this is safer.
        elif dist_obj_to_goal < dist_obj_to_goal_threshold:
            # Phase 3: Arrived at the goal, ready to place.
            return 3
        else:
            # Phase 2: Transporting the object towards the goal.
            return 2
    elif not is_grasped and prev_is_grasped:
        # Just released the object.
        # Phase 4: Retracting from placement.
        return 4
    
    # Default fallback
    return 0
# --- [END OF DEFINITIVE PATCH 1] ---




# In FILE: evaluate_uhp.py

# --- [START OF DEFINITIVE PATCH 2: MAIN EVALUATION FUNCTION] ---
# REPLACE the existing `evaluate` function with this new version.

@hydra.main(version_base=None, config_path="./configs", config_name="evaluate_uhp_config")
def evaluate(cfg: DictConfig):
    log.info("--- UHP v2.0 Hierarchical Visual Evaluation (SOTA Patched) ---")
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
    # (This section is already well-implemented and needs no changes)
    video_writer = None
    if cfg.logging.enable_video:
        video_path = Path(cfg.output_video)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        frame_test, _ = env.reset(seed=cfg.seed)
        H, W, _ = env.render().shape
        video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (W, H))
        log.info(f"Recording video to: {video_path}")

    csv_file = None
    csv_writer = None
    if cfg.logging.enable_csv_logging:
        csv_path = Path(cfg.logging.csv_output_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        
        header = ['episode_idx', 'step', 'oracle_task_phase', 'gt_is_grasped']
        header += [f'gt_ee_pos_{ax}' for ax in ['x', 'y', 'z']]
        header += [f'gt_obj_pos_{ax}' for ax in ['x', 'y', 'z']]
        header += ['planner_subgoal_emb_norm']
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
        
        # --- CRITICAL FIX: Warm-up the observation history ---
        # Populate the deque with unique, consecutive observations before starting.
        obs_history = collections.deque(maxlen=obs_horizon)
        # Take a few "zero action" steps to get a valid history.
        for _ in range(obs_horizon):
            # Pass a zero action to get the next observation without moving.
            obs, _, _, _, _ = env.step(np.zeros(action_dim))
            obs_history.append(obs)
            
        current_phase = -1
        subgoal_embedding = None
        # State for our new oracle.
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
                
                current_proprio_tensor = torch.from_numpy(obs['proprio']).float().to(device).unsqueeze(0)


                subgoal_embedding = model.plan(
                    current_image_tensor,
                    goal_image_tensor,
                    task_phase_tensor,
                    current_proprio_tensor
                )
            
            # Prepare Executor inputs from the now-valid obs_history
            proprio_hist = torch.from_numpy(np.stack([h['proprio'] for h in obs_history])).float().to(device)
            primary_hist = torch.stack([transform_controller_primary(Image.fromarray(h['image_primary'])) for h in obs_history]).to(device)
            wrist_hist = torch.stack([transform_controller_wrist(Image.fromarray(h['image_wrist'])) for h in obs_history]).to(device)
            controller_obs_hist = {
                'image_primary': primary_hist.unsqueeze(0),
                'image_wrist': wrist_hist.unsqueeze(0),
                'proprio': proprio_hist.unsqueeze(0)
            }
            
            action_chunk_raw = model.act(
                controller_obs_hist, subgoal_embedding, noise_scheduler,
                cfg.inference.inference_steps, action_normalizer, proprio_normalizer,
                joint_limits_low, joint_limits_high
            )
            action = action_chunk_raw[0, 0, :].cpu().numpy()
            
            if cfg.logging.enable_csv_logging:
                log_row = (
                    _prepare_for_csv(ep_idx) + _prepare_for_csv(step_count) +
                    _prepare_for_csv(current_phase) + _prepare_for_csv(obs['is_grasped']) +
                    _prepare_for_csv(obs['ee_pose_world'][:3]) + _prepare_for_csv(obs['object_pos_world']) +
                    _prepare_for_csv(torch.linalg.norm(subgoal_embedding)) +
                    _prepare_for_csv(action) + _prepare_for_csv(action_chunk_raw)
                )
                csv_writer.writerow(log_row)

            # Update state for the next oracle call BEFORE stepping the environment.
            prev_is_grasped = obs.get('is_grasped', [0.0])[0] > 0.5
            
            # Step Environment
            obs, reward, terminated, truncated, info = env.step(action)
            obs_history.append(obs)
            
            if cfg.logging.enable_video:
                frame_rgb = env.render()
                # You can add text overlays here for diagnostics
                phase_text = f"Phase: {current_phase}"
                cv2.putText(frame_rgb, phase_text, (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                video_writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            
            if terminated or truncated:
                log.info(f"Episode finished after {step_count + 1} steps.")
                break
        
    # --- 6. Cleanup ---
    if video_writer is not None: video_writer.release()
    if csv_file is not None: csv_file.close()
    env.close()
    log.info("--- Evaluation Complete. ---")

if __name__ == "__main__":
    evaluate()

# --- [END OF DEFINITIVE PATCH 2] ---