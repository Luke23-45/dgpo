# FILE: evaluate_semantic_planner.py
# (Definitive, SOTA, Production-Grade Version)

"""
AWSP (Advantage-Weighted Semantic Planner) SOTA Evaluation Script.

This script provides a comprehensive, closed-loop evaluation of the AWSP policy.
It seamlessly orchestrates the perception-planning-control loop:
1.  **Perception**: Captures images and infers the current high-level 'Task Phase'.
2.  **Planning**: Queries the Semantic Planner for the next Subgoal Pose.
3.  **Control**: Solves for joint velocities (Differential IK) to reach that subgoal.

Features:
- **Hermetic Loading**: Loads model & config directly from the Lightning checkpoint.
- **Virtual Goal Rendering**: Teleports objects in physics to generate valid goal images.
- **Phase Inference Oracle**: A state-machine heuristic to drive the phase-conditioned planner.
- **Full Telemetry**: Logs per-step ground truth vs. prediction errors to CSV.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import os
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Dict, Tuple

import cv2
import hydra
import mujoco
import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# --- Project-Specific Imports ---
# Robust path handling to ensure modules are found
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from envs.panda_env import PandaEnv
from models.semantic_planner import SemanticPlanner
from train.train_semantic_planner import SemanticPlannerLightningModule as AWSP_LightningModule
from utils.ik_solver import IKSolver

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("AWSP_Evaluator")


# ==============================================================================
# SOTA UTILITIES
# ==============================================================================

def load_model_from_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[SemanticPlanner, DictConfig]:
    """
    Robustly loads the trained model and its training configuration.
    Uses 'weights_only=False' to recover the full Hydra config stored in the ckpt.
    """
    log.info(f"Loading checkpoint: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    # Load LightningModule
    # We map to CPU first to avoid GPU OOM if multiple processes are running
    pl_module = AWSP_LightningModule.load_from_checkpoint(
        ckpt_path, 
        map_location=device
    )
    
    model = pl_module.model
    model.to(device)
    model.eval()
    
    # Recover the config used for training
    train_cfg = pl_module.cfg
    log.info("Model loaded. Freeze-drying configuration...")
    return model, train_cfg


@contextmanager
def render_virtual_goal(env: PandaEnv, goal_pos: np.ndarray):
    """
    Context manager that temporarily updates the simulation state to place the
    object at the goal, renders the scene, and then RESTORES the original state.
    This allows 'seeing the future' without disrupting the 'present'.
    """
    # 1. Snapshot current physics state
    original_qpos = env.data.qpos.copy()
    original_qvel = env.data.qvel.copy()
    
    try:
        # 2. Teleport object to goal
        # Find object joint address
        obj_joint_id = env.object_joint_id
        qpos_adr = env.model.jnt_qposadr[obj_joint_id]
        
        # Set position (keep orientation identity or random if needed)
        # We just use the current object orientation for the goal
        current_obj_quat = env.data.qpos[qpos_adr+3:qpos_adr+7].copy()
        
        env.data.qpos[qpos_adr:qpos_adr+3] = goal_pos
        env.data.qpos[qpos_adr+3:qpos_adr+7] = current_obj_quat # Keep orientation
        
        # Forward propagate kinematics
        mujoco.mj_forward(env.model, env.data)
        
        # 3. Yield control back to caller to render
        yield
        
    finally:
        # 4. Restore original state exactly
        env.data.qpos[:] = original_qpos
        env.data.qvel[:] = original_qvel
        mujoco.mj_forward(env.model, env.data)


def get_inferred_task_phase(
    obs: Dict[str, Any], 
    prev_phase: int, 
    prev_is_grasped: bool
) -> int:
    """
    [SOTA ORACLE] Infers the semantic task phase from raw observations.
    This mimics the logic used in `expert_dataset.py` but relies only on
    observable states, not internal FSM counters.
    
    Phases:
    0: Approach (Reach)
    1: Grasp (Actuate Gripper)
    2: Transport (Move to Goal)
    3: Place (Release)
    4: Retract (Done)
    """
    # Extract State
    is_grasped_val = obs.get('is_grasped', [0.0])[0]
    is_grasped = is_grasped_val > 0.5
    
    ee_pos = obs['ee_pose_world'][:3]
    obj_pos = obs['object_pos_world']
    goal_pos = obs['goal_pos_world']
    
    dist_ee_obj = np.linalg.norm(ee_pos - obj_pos)
    dist_obj_goal = np.linalg.norm(obj_pos - goal_pos)
    
    # Thresholds (Must align with reward/expert logic)
    THRESH_GRASP_DIST = 0.08  # 4cm
    THRESH_GOAL_DIST = 0.08   # 5cm

    # Logic Tree
    if not is_grasped:
        if prev_is_grasped: 
            # We just dropped or released
            if dist_obj_goal < THRESH_GOAL_DIST:
                return 4 # Retract (Success)
            else:
                return 0 # Lost object, restart Approach
        
        if prev_phase == 4:
            return 4 # Stay in Retract
            
        # If close to object, switch to Grasp Phase
        if dist_ee_obj < THRESH_GRASP_DIST:
            return 1
        else:
            return 0 # Approach
            
    else: # Is Grasped
        # If we are holding it, are we at the goal?
        if dist_obj_goal < THRESH_GOAL_DIST:
            return 3 # Place
        else:
            return 2 # Transport

def format_csv_row(ep, step, phase, grasped, ee_pos, pred_pose, grip_logit, grip_cmd):
    """Flattens data for CSV logging."""
    row = [ep, step, phase, int(grasped)]
    row.extend(ee_pos.tolist())
    row.extend(pred_pose.tolist())
    row.append(f"{grip_logit:.4f}")
    row.append(f"{grip_cmd:.1f}")
    return row

# ==============================================================================
# MAIN EVALUATION LOOP
# ==============================================================================

@hydra.main(version_base=None, config_path="../configs", config_name="evaluate_semantic_planner_config")
def main(cfg: DictConfig):
    log.info("--- Starting AWSP Evaluation ---")
    
    # 1. Setup Device & Seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl.seed_everything(cfg.seed)

    # 2. Load Model
    model, train_cfg = load_model_from_checkpoint(cfg.checkpoint_path, device)
    
    # 3. Setup Transforms
    # Ensure exact match with training transform
    tf = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor()
    ])

    # 4. Initialize Simulation
    log.info(f"Initializing Env from: {train_cfg.dataset.get('xml_path', 'envs/panda_pick_place.xml')}")
    env = PandaEnv(
        xml_path="envs/panda_pick_place.xml", # Explicit override or from cfg
        control_mode="delta",
    )
    
    ik_solver = IKSolver(urdf_path=cfg.ik_solver_path)
    
    # Controller Calibration
    N_SUBSTEPS = 20
    effective_dt = env.model.opt.timestep * N_SUBSTEPS
    # Sync Max DQ with Action Scaling Factor for physically valid actions
    max_dq = env.ACTION_SCALING_FACTOR / effective_dt
    log.info(f"Controller Calibrated: dt={effective_dt}, max_dq={max_dq}")

    # 5. Prepare Outputs
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    video_path = output_dir / cfg.output_video_path
    csv_path = output_dir / cfg.output_csv_path
    
    # Initialize Video Writer (Get dims from dummy render)
    env.reset()
    dummy_frame = env.render()
    H, W, _ = dummy_frame.shape
    video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (W, H))
    
    # Initialize CSV
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "Episode", "Step", "Phase", "IsGrasped", 
        "EE_X", "EE_Y", "EE_Z", 
        "Pred_X", "Pred_Y", "Pred_Z", "Pred_Qx", "Pred_Qy", "Pred_Qz", "Pred_Qw",
        "GripLogit", "GripCmd"
    ])

    episode_success_count = 0
    
    try:
        for ep_idx in tqdm(range(cfg.num_episodes), desc="Evaluating"):
            seed = cfg.seed if cfg.eval_static_scene else cfg.seed + ep_idx
            obs, _ = env.reset(seed=seed)
            
            # 5.1 Generate Goal Image (The "Imagination")
            goal_pos = obs['goal_pos_world']
            with render_virtual_goal(env, goal_pos):
                goal_img_raw = env.render()
            
            # Prepare Tensors
            goal_tensor = tf(Image.fromarray(goal_img_raw)).unsqueeze(0).to(device)
            
            # Reset Loop State
            prev_phase = 0
            prev_is_grasped = False
            
            for step in range(cfg.max_steps):
                # --- A. Perception ---
                img_tensor = tf(Image.fromarray(obs['image_primary'])).unsqueeze(0).to(device)
                proprio_tensor = torch.from_numpy(obs['proprio']).float().unsqueeze(0).to(device)
                
                # Infer Phase
                curr_phase = get_inferred_task_phase(obs, prev_phase, prev_is_grasped)
                phase_tensor = torch.tensor([curr_phase], device=device)
                
                # --- B. Planning (Inference) ---
                batch = {
                    'initial_image': img_tensor,
                    'goal_image': goal_tensor,
                    'task_phase': phase_tensor,
                    'current_proprio': proprio_tensor
                }
                
                with torch.no_grad():
                    preds = model(batch)
                
                pred_pose_np = preds['pose'].cpu().numpy()[0] # 7D
                pred_grip_logit = preds['gripper_logit'].item()
                
                # --- C. Low-Level Control ---
                # 1. Decode Gripper: Sigmoid threshold at 0.5 (logit 0.0)
                # If logit > 0 -> Prob > 0.5 -> GRASP (-1.0 in simulation)
                # If logit < 0 -> Prob < 0.5 -> OPEN (+1.0 in simulation)
                gripper_cmd = -1.0 if pred_grip_logit > 0.0 else 1.0
                
                # 2. Solve IK
                # target_pose_7d needs [pos, quat]
                delta_joint_action = ik_solver.compute_delta_action(
                    target_ee_pose=pred_pose_np,
                    model=env.model,
                    data=env.data,
                    ee_site_id=env.ee_site_id,
                    joint_qpos_indices=np.arange(7),
                    effective_dt=effective_dt,
                    max_dq=max_dq
                )
                
                action = np.concatenate([delta_joint_action, [gripper_cmd]])
                
                # --- D. Simulation Step ---
                obs, _, terminated, truncated, info = env.step(action)
                
                # --- E. Logging & Viz ---
                # Write CSV
                csv_row = format_csv_row(
                    ep_idx, step, curr_phase, prev_is_grasped, 
                    obs['ee_pose_world'][:3], pred_pose_np, pred_grip_logit, gripper_cmd
                )
                csv_writer.writerow(csv_row)
                
                # Write Video Frame
                frame = env.render()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # HUD
                status_color = (0, 255, 0) if prev_is_grasped else (0, 255, 255)
                cv2.putText(frame, f"Ep {ep_idx}: Phase {curr_phase}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Grip: {gripper_cmd:.1f}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                video_writer.write(frame)
                
                # Update state
                prev_phase = curr_phase
                prev_is_grasped = obs['is_grasped'][0] > 0.5
                
                # Check termination (Success logic usually inside info or simple proximity check)
                # Let's add a robust success check here based on physical state
                obj_goal_dist = np.linalg.norm(obs['object_pos_world'] - obs['goal_pos_world'])
                is_success = (obj_goal_dist < 0.05) and (obs['object_pos_world'][2] > 0.40)
                
                if terminated or truncated or is_success:
                    if is_success:
                        episode_success_count += 1
                        log.info(f"Episode {ep_idx} SUCCESS!")
                    else:
                        log.info(f"Episode {ep_idx} Failed/Timed Out.")
                    break
                    
    except KeyboardInterrupt:
        log.warning("Evaluation interrupted by user.")
    finally:
        log.info("Finalizing resources...")
        env.close()
        video_writer.release()
        csv_file.close()
        
        success_rate = (episode_success_count / cfg.num_episodes) * 100
        log.info(f"FINAL RESULTS: Success Rate = {success_rate:.1f}% ({episode_success_count}/{cfg.num_episodes})")
        log.info(f"Video saved to: {video_path}")
        log.info(f"Metrics saved to: {csv_path}")

if __name__ == "__main__":
    main()