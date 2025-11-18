# FILE: evaluate_ego_planner.py
# (This is the final, definitive, phase-aware version)

import logging
import os
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
import sys
import pytorch_lightning as pl

# --- SOTA: Add project root to path to ensure modules are found ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# --- Import Project Modules ---
from envs.panda_env import PandaEnv
from models.ego_planner import EgoPlanner
from train.train_ego_planner import EgoPlannerLightningModule
from models.ego_planner import NoiseScheduler, NoiseSchedulerConfig

# Set up a logger
log = logging.getLogger(__name__)



def load_model_from_checkpoint(
    checkpoint_path: str, 
    train_cfg: DictConfig, # Now accepts the loaded training config
    device: torch.device
) -> EgoPlanner:
    """
    [DEFINITIVE, SOTA VERSION 2.0]
    Loads the EGO-Planner model by first instantiating it from the provided
    training config and then loading the EMA weights from the checkpoint.
    """
    log.info(f"Loading checkpoint from: {checkpoint_path}")
    
    # 1. Load the checkpoint file to get the weights.
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    if 'ema_state_dict' not in ckpt:
        raise KeyError("Checkpoint requires 'ema_state_dict' for evaluation.")
        
    log.info("Found 'ema_state_dict'. Initializing model from provided training config...")
    
    # 2. Initialize the LightningModule with the FULL training config.
    #    This ensures all components, including those not in `hyper_parameters`, are available.
    lightning_model = EgoPlannerLightningModule(train_cfg)
    
    # 3. Load the EMA state dict into the model's EMA object.
    lightning_model.ema.load_state_dict(ckpt['ema_state_dict'])
    
    # 4. Get the *actual* model from the EMA wrapper.
    model = lightning_model.ema.ema_model
    
    # 5. Move to the target device and set to evaluation mode.
    model.to(device)
    model.eval()
    
    log.info("Model loaded successfully using EMA weights and set to eval mode.")
    return model


def get_goal_image(env: PandaEnv, obs: dict) -> np.ndarray:
    """Correctly renders a goal image by temporarily moving the object."""
    state = env.get_mj_state()
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
        env.set_mj_state(state)
    return goal_img_np


def preprocess_image(img_np: np.ndarray, transform: transforms.Compose) -> torch.Tensor:
    return transform(Image.fromarray(img_np))

# --- [START OF DEFINITIVE PATCH 1: ADD THE TASK PHASE ORACLE] ---
def get_current_task_phase(obs: dict, prev_is_grasped: bool) -> int:
    """
    [SOTA, STATEFUL ORACLE]
    Determines the current TaskPhase by mirroring the ScriptedExpert's logic.
    """
    is_grasped = obs.get('is_grasped', [0.0])[0] > 0.5
    ee_pos = obs['ee_pose_world'][:3]
    obj_pos = obs['object_pos_world']
    goal_pos = obs['goal_pos_world']
    table_z = 0.4
    dist_ee_to_obj_threshold = 0.04
    lift_height_threshold = 0.03
    dist_obj_to_goal_threshold = 0.08

    dist_ee_to_obj = np.linalg.norm(ee_pos - obj_pos)
    dist_obj_to_goal = np.linalg.norm(obj_pos - goal_pos)
    obj_lift_height = obj_pos[2] - table_z

    if not is_grasped and not prev_is_grasped:
        if dist_ee_to_obj < dist_ee_to_obj_threshold: return 1
        else: return 0
    elif is_grasped and not prev_is_grasped: return 1
    elif is_grasped and prev_is_grasped:
        if obj_lift_height < lift_height_threshold: return 1
        elif dist_obj_to_goal < dist_obj_to_goal_threshold: return 3
        else: return 2
    elif not is_grasped and prev_is_grasped: return 4
    return 0
# --- [END OF DEFINITIVE PATCH 1] ---


@hydra.main(version_base=None, config_path="./configs", config_name="evaluate_ego_planner_config")
def evaluate(cfg: DictConfig):
    log.info("--- Phase-Aware EGO-Planner Visual Evaluation (Definitive v2.0) ---")
    log.info(f"Evaluation config:\n{OmegaConf.to_yaml(cfg)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl.seed_everything(cfg.seed, workers=True)

    # --- [START OF THE DEFINITIVE PATCH 2] ---
    # 1. Load the ORIGINAL training config from the specified path.
    if not cfg.train_config_path:
        raise ValueError("`train_config_path` must be provided in the evaluation config.")
    
    log.info(f"Loading original training config from: {cfg.train_config_path}")
    train_cfg = OmegaConf.load(cfg.train_config_path)

    # 2. Load the Model, passing it the full training config.
    model = load_model_from_checkpoint(cfg.checkpoint_path, train_cfg, device)
    
    # 3. All other parameters now come from the loaded `train_cfg`.
    obs_horizon = train_cfg.model.obs_horizon
    action_dim = train_cfg.model.action_dim
    
    scheduler_cfg = NoiseSchedulerConfig(
        beta_start=train_cfg.scheduler.beta_start,
        beta_end=train_cfg.scheduler.beta_end,
        schedule=train_cfg.scheduler.beta_schedule,
        timesteps=train_cfg.scheduler.timesteps,
    )
    noise_scheduler = NoiseScheduler(scheduler_cfg).to(device)
    
    # 4. Initialize the Environment using the now-available `train_cfg.env` block.
    log.info("Initializing PandaEnv from training config (Domain Randomization DISABLED for eval)...")
    env = PandaEnv( control_mode="delta")

    # --- 3. Setup Image Transforms (must match training) ---
    transform_primary = transforms.Compose([transforms.Resize((224, 224), antialias=True), transforms.ToTensor()])
    transform_wrist = transforms.Compose([transforms.Resize((128, 128), antialias=True), transforms.ToTensor()])
    
    # --- 4. Setup Video Recording ---
    video_path = Path(cfg.output_video)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    frame_test, _ = env.reset(seed=cfg.seed)
    H, W, _ = env.render().shape
    video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (W, H))
    log.info(f"Recording video to: {video_path}")

    # --- 5. Run Evaluation Loop ---
    for ep_idx in tqdm(range(cfg.num_episodes), desc="Evaluating Episodes"):
        episode_seed = cfg.seed if cfg.eval_static_scene else cfg.seed + ep_idx
        obs, _ = env.reset(seed=episode_seed)
        
        initial_image_np = obs['image_primary']
        goal_image_np = get_goal_image(env, obs)
        
        initial_image_tensor = preprocess_image(initial_image_np, transform_primary).to(device).unsqueeze(0)
        goal_image_tensor = preprocess_image(goal_image_np, transform_primary).to(device).unsqueeze(0)
        
        obs_history = collections.deque(maxlen=obs_horizon)
        
        # --- [START OF DEFINITIVE PATCH 2: DYNAMIC HISTORY WARM-UP] ---
        log.info("Warming up observation history with dynamic frames...")
        for _ in range(obs_horizon):
            obs, _, _, _, _ = env.step(np.zeros(action_dim)) # Take "do-nothing" steps
            obs_history.append(obs)
        # --- [END OF DEFINITIVE PATCH 2] ---
            
        prev_is_grasped = False
        step_iterator = tqdm(range(env.max_episode_steps), desc=f"Episode {ep_idx+1}", leave=False)
        for step_count in step_iterator:
            
            # --- Prepare Model Batch ---
            img_primary_hist = torch.stack([preprocess_image(h['image_primary'], transform_primary) for h in obs_history]).to(device)
            img_wrist_hist = torch.stack([preprocess_image(h['image_wrist'], transform_wrist) for h in obs_history]).to(device)
            proprio_hist = torch.from_numpy(np.stack([h['proprio'] for h in obs_history])).float().to(device)

            # --- [START OF DEFINITIVE PATCH 3: ADD TASK PHASE TO BATCH] ---
            current_phase = get_current_task_phase(obs, prev_is_grasped)
            task_phase_tensor = torch.tensor([current_phase], dtype=torch.long, device=device)
            # --- [END OF DEFINITIVE PATCH 3] ---
            
            batch = {
                'initial_image': initial_image_tensor,
                'goal_image': goal_image_tensor,
                'observation_history': {
                    'image_primary': img_primary_hist.unsqueeze(0),
                    'image_wrist': img_wrist_hist.unsqueeze(0),
                    'proprio': proprio_hist.unsqueeze(0),
                },
                'task_phase': task_phase_tensor, # Add the new key
            }
            
            # --- Get Action from Model ---
            with torch.no_grad():
                action_chunk = model.sample(
                    batch=batch,
                    scheduler=noise_scheduler,
                    guidance_plan=cfg.inference.guidance_scale_plan,
                    guidance_obs=cfg.inference.guidance_scale_obs,
                    num_inference_steps=cfg.inference.sampling_steps
                )
            action = action_chunk[0, 0, :].cpu().numpy()
            
            # Update state for next oracle call BEFORE stepping
            prev_is_grasped = obs.get('is_grasped', [0.0])[0] > 0.5
            
            # Step Environment and Record
            obs, _, terminated, truncated, _ = env.step(action)
            obs_history.append(obs)
            
            frame_rgb = env.render()
            phase_text = f"Phase: {current_phase}"
            cv2.putText(frame_rgb, phase_text, (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            video_writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            
            if terminated or truncated:
                break
    
    # --- 6. Cleanup ---
    video_writer.release()
    env.close()
    log.info("--- Evaluation Complete. ---")

if __name__ == "__main__":
    evaluate()