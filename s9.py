# FILE: eval/evaluate_hybrid_planner_fixed.py
# (v19.0 - Hybrid Diagnostic / Architecture Aligned)

import json
import logging
import sys
import cv2
import hydra
import mujoco
import numpy as np
import torch
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
from train.train_semantic_planner import SemanticPlannerLightningModule
from utils.ik_solver import IKSolver

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("HybridEval")

# --- 1. Normalization Helper ---
class DataNormalizer:
    def __init__(self, stats_path, device):
        self.device = device
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        self.stats = {k: torch.tensor(v, device=device) for k, v in stats.items()}

    def normalize_proprio(self, raw):
        return (raw - self.stats['proprio_mean']) / (self.stats['proprio_std'] + 1e-6)

    def unnormalize_pose(self, norm_pose):
        # Assumes action_mean/std covers the 7D pose
        # If your stats are [7], broadcast is handled
        mean = self.stats['action_mean'][:7] # slice for pose only
        std = self.stats['action_std'][:7]
        return (norm_pose * std) + mean

# --- 2. Heuristic Policy (Unchanged) ---
class GeometricGraspPolicy:
    def __init__(self):
        self.is_holding = False
        self.GRASP_THRESH = 0.03
        self.DROP_THRESH = 0.05

    def reset(self):
        self.is_holding = False

    def compute_command(self, obs: dict) -> float:
        ee_pos = obs['ee_pose_world'][:3]
        obj_pos = obs['object_pos_world']
        goal_pos = obs['goal_pos_world']
        dist_obj = np.linalg.norm(ee_pos - obj_pos)
        dist_goal = np.linalg.norm(obj_pos - goal_pos)
        
        cmd = 1.0 
        if not self.is_holding:
            if dist_obj < self.GRASP_THRESH:
                cmd = -1.0 
                self.is_holding = True 
        else:
            cmd = -1.0 
            if dist_goal < self.DROP_THRESH and obs['object_pos_world'][2] > 0.41:
                cmd = 1.0 
                self.is_holding = False
            if obs['is_grasped'][0] < 0.1:
                self.is_holding = False
        return cmd

# --- 3. Virtual Goal (Physics Safe) ---
@contextmanager
def render_physics_safe_goal(env, target_pos_world):
    saved_qpos = env.data.qpos.copy()
    saved_qvel = env.data.qvel.copy()
    saved_ctrl = env.data.ctrl.copy()
    try:
        home_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        env.data.qpos[:7] = home_qpos
        env.data.qpos[7:] = 0.04
        obj_addr = env.model.jnt_qposadr[env.object_joint_id]
        curr_quat = env.data.qpos[obj_addr+3 : obj_addr+7].copy()
        env.data.qpos[obj_addr:obj_addr+3] = target_pos_world
        env.data.qpos[obj_addr+3:obj_addr+7] = current_quat
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        yield
    finally:
        env.data.qpos[:] = saved_qpos
        env.data.qvel[:] = saved_qvel
        env.data.ctrl[:] = saved_ctrl
        mujoco.mj_forward(env.model, env.data)

# --- Main Runner ---
@hydra.main(version_base=None, config_path="./configs", config_name="evaluate_semantic_planner_config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
        cfg.checkpoint_path, map_location=device
    )
    model = pl_module.model.eval().to(device)
    
    # 2. Normalizer
    stats_path = cfg.get("stats_path", "dataset_stats.json")
    normalizer = DataNormalizer(stats_path, device)
    
    # 3. Setup Env
    env = PandaEnv(xml_path=cfg.get("xml_path", "envs/panda_pick_place.xml"), 
                   control_mode='delta', 
                   action_scaling_factor=0.5)
    ik_solver = IKSolver(urdf_path="urdf/panda_mujoco_kinematics.urdf")
    grasper = GeometricGraspPolicy()
    
    # Sync Controller
    dt = env.model.opt.timestep * 20
    max_dq = (env.ACTION_SCALING_FACTOR / dt) 
    
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    video_file = "hybrid_diagnostic.mp4"
    # Init video
    dummy = env.reset()[0]
    h, w, _ = env.render().shape
    writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    
    success_count = 0
    
    for ep in tqdm(range(cfg.num_episodes)):
        seed = cfg.seed + ep
        obs, _ = env.reset(seed=seed)
        grasper.reset()
        
        # Render Goal
        with render_physics_safe_goal(env, obs['goal_pos_world']):
            g_img = env.render()
        goal_t = transform(Image.fromarray(g_img)).unsqueeze(0).to(device)
        
        # History Buffer
        curr_img = Image.fromarray(obs['image_primary'])
        prev_img_t = transform(curr_img).unsqueeze(0).to(device)
        
        done = False
        
        for step in range(cfg.max_steps):
            # A. Heuristic Gripper Logic
            gripper_cmd = grasper.compute_command(obs)
            
            # B. Neural Network (Trajectory Only)
            curr_img = Image.fromarray(obs['image_primary'])
            curr_t = transform(curr_img).unsqueeze(0).to(device)
            
            # Normalize Proprio
            raw_prop = torch.from_numpy(obs['proprio']).float().unsqueeze(0).to(device)
            norm_prop = normalizer.normalize_proprio(raw_prop)
            
            batch = {
                'prev_image': prev_img_t,
                'curr_image': curr_t,
                'goal_image': goal_t,
                'curr_proprio': norm_prop
            }
            # Update history
            prev_img_t = curr_t.clone()
            
            with torch.no_grad():
                out = model(batch)
            
            # C. Extract & Un-Normalize Pose
            # Take 1st step of chunk [0]
            norm_pose = out['pose_chunk'][0][0] # (7,)
            target_pose = normalizer.unnormalize_pose(norm_pose).cpu().numpy()
            
            # D. Execution
            # Convert xyzw -> wxyz
            pos = target_pose[:3]
            quat_xyzw = target_pose[3:]
            target_ee_mj = np.concatenate([pos, [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]])
            
            try:
                d_arm = ik_solver.compute_delta_action(
                    target_ee_pose=target_ee_mj,
                    model=env.model, data=env.data,
                    ee_site_id=env.ee_site_id,
                    joint_qpos_indices=np.arange(7),
                    effective_dt=dt, max_dq=max_dq
                )
            except: d_arm = np.zeros(7)
            
            # Inverse Scaling
            d_arm = d_arm / env.ACTION_SCALING_FACTOR
            d_arm = np.clip(d_arm, -1.0, 1.0)
            
            action = np.concatenate([d_arm, [gripper_cmd]])
            obs, _, _, _, _ = env.step(action)
            
            # Render
            frame = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
            cv2.putText(frame, f"Mode: {'HOLD' if grasper.is_holding else 'REACH'}", (10,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            writer.write(frame)
            
            # Check Success
            dist = np.linalg.norm(obs['object_pos_world'] - obs['goal_pos_world'])
            if dist < 0.05:
                success_count += 1
                done = True
                break
                
        if done: log.info(f"Ep {ep}: Success")
        else: log.info(f"Ep {ep}: Fail")
        
    writer.release()
    env.close()
    log.info(f"Hybrid Success: {success_count}/{cfg.num_episodes}")

if __name__ == "__main__":
    main()