# FILE: evaluate_hybrid_planner.py
# (Diagnostic: Neural Motion + Heuristic Grasping)

"""
Hybrid Evaluation System.

Diagnoses the vision backbone by decoupling the Grasp Logic.
- Neural Network: Controls the Arm (7D Pose).
- Geometric Rules: Control the Gripper (Open/Close).

If this script succeeds, it proves the Vision Model understands spatial structure,
and the only failure point in previous tests was the Gripper Decision Head.
"""

import logging
import os
import sys
import cv2
import hydra
import mujoco
import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from contextlib import contextmanager
from scipy.spatial.transform import Rotation as R

# --- Imports ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from envs.panda_env import PandaEnv
from models.semantic_planner import SemanticPlanner
from train.train_semantic_planner import SemanticPlannerLightningModule
from utils.ik_solver import IKSolver

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("HybridEval")

# --- Heuristic Policy ---
class GeometricGraspPolicy:
    """Deterministic rules for opening/closing the gripper."""
    def __init__(self):
        self.is_holding = False
        self.GRASP_THRESH = 0.03  # 3cm trigger
        self.DROP_THRESH = 0.05   # 5cm trigger

    def reset(self):
        self.is_holding = False

    def compute_command(self, obs: dict) -> float:
        # State
        ee_pos = obs['ee_pose_world'][:3]
        obj_pos = obs['object_pos_world']
        goal_pos = obs['goal_pos_world']
        
        dist_obj = np.linalg.norm(ee_pos - obj_pos)
        dist_goal = np.linalg.norm(obj_pos - goal_pos)
        
        cmd = 1.0 # Default Open

        if not self.is_holding:
            if dist_obj < self.GRASP_THRESH:
                cmd = -1.0 # Close
                # Assume success for next step if close
                self.is_holding = True 
        else:
            cmd = -1.0 # Keep Closed
            # Release conditions
            if dist_goal < self.DROP_THRESH and obs['object_pos_world'][2] > 0.41:
                cmd = 1.0 # Open
                self.is_holding = False
            
            # Lost object check
            if obs['is_grasped'][0] < 0.1:
                self.is_holding = False
        
        return cmd

# --- Simple Smoother ---
class PoseSmoother:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.pose = None
    
    def reset(self):
        self.pose = None
        
    def update(self, raw_pose):
        if self.pose is None:
            self.pose = raw_pose.copy()
        else:
            self.pose[:3] = self.alpha * raw_pose[:3] + (1-self.alpha)*self.pose[:3]
            # Slerp or simplistic quat mix
            self.pose[3:] = self.alpha * raw_pose[3:] + (1-self.alpha)*self.pose[3:]
            norm = np.linalg.norm(self.pose[3:])
            if norm > 1e-6: self.pose[3:] /= norm
        return self.pose

# --- Virtual Goal (Standard) ---
@contextmanager
def render_virtual_goal(env: PandaEnv, goal_pos: np.ndarray):
    saved = env.get_mj_state()
    try:
        # Move object
        obj_addr = env.model.jnt_qposadr[env.object_joint_id]
        curr_quat = env.data.qpos[obj_addr+3 : obj_addr+7].copy()
        env.data.qpos[obj_addr:obj_addr+3] = goal_pos
        env.data.qpos[obj_addr+3:obj_addr+7] = curr_quat
        
        # Move robot (Home)
        env.data.qpos[:7] = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        env.data.qpos[7:9] = 0.04
        
        mujoco.mj_forward(env.model, env.data)
        yield
    finally:
        env.set_mj_state(saved)

# --- Main Runner ---
@hydra.main(version_base=None, config_path="./configs", config_name="evaluate_semantic_planner_config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
        cfg.checkpoint_path, map_location=device
    )
    model = pl_module.model.eval().to(device)
    train_cfg = pl_module.cfg
    
    # 2. Setup Env
    env = PandaEnv(xml_path=train_cfg.dataset.get("xml_path", "envs/panda_pick_place.xml"), control_mode='delta')
    ik_solver = IKSolver(urdf_path=cfg.ik_solver_path)
    grasper = GeometricGraspPolicy()
    smoother = PoseSmoother()
    
    # Calib
    sim_steps = 20
    dt = env.model.opt.timestep * sim_steps
    max_dq = (env.ACTION_SCALING_FACTOR / dt) * 3.0 # 3x Boost
    
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor()
    ])
    
    video_file = "hybrid_eval.mp4"
    writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (256, 256))
    
    success_count = 0
    
    for ep in tqdm(range(cfg.num_episodes)):
        seed = cfg.seed + ep
        env.reset(seed=seed)
        obs = env.get_expert_obs()
        grasper.reset()
        smoother.reset()
        
        # Goal Image
        with render_virtual_goal(env, obs['goal_pos_world']):
            g_img = env.render()
        goal_t = transform(Image.fromarray(g_img)).unsqueeze(0).to(device)
        
        done = False
        
        for step in range(cfg.max_steps):
            # 1. Heuristic Control
            gripper_cmd = grasper.compute_command(obs)
            
            # 2. Neural Planning (Trajectory)
            img_t = transform(Image.fromarray(obs['image_primary'])).unsqueeze(0).to(device)
            prop_t = torch.from_numpy(obs['proprio']).float().unsqueeze(0).to(device)
            
            # Feed 'Task Phase' to model based on Heuristic State
            # If holding -> Transport (2). If not -> Reach (0).
            phase = 2 if grasper.is_holding else 0
            
            with torch.no_grad():
                pred = model({
                    'initial_image': img_t, 'goal_image': goal_t,
                    'task_phase': torch.tensor([phase], device=device),
                    'current_proprio': prop_t
                })
                
            raw_pose = pred['pose'].squeeze().cpu().numpy()
            target_pose = smoother.update(raw_pose)
            
            # 3. Execution
            try:
                d_arm = ik_solver.compute_delta_action(target_pose, env.model, env.data, 
                                                     env.ee_site_id, np.arange(7), dt, max_dq)
            except: d_arm = np.zeros(7)
            
            action = np.concatenate([d_arm, [gripper_cmd]])
            obs, _, _, _, _ = env.step(action)
            obs = env.get_expert_obs() # Ground Truth Update
            
            # Render
            frame = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
            cv2.putText(frame, f"Mode: {'HOLD' if grasper.is_holding else 'REACH'}", (10,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            writer.write(frame)
            
            # Check Success
            dist = np.linalg.norm(obs['object_pos_world'] - obs['goal_pos_world'])
            if dist < 0.05 and obs['object_pos_world'][2] > 0.41:
                success_count += 1
                done = True
                break
                
        if done: log.info(f"Ep {ep}: Success")
        else: log.info(f"Ep {ep}: Fail")
        
    writer.release()
    env.close()
    log.info(f"Final Success: {success_count}/{cfg.num_episodes}")

if __name__ == "__main__":
    main()