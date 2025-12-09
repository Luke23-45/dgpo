# FILE: eval/evaluate_semantic_planner_v17.py
# (v17.0 - Fixed Control Gain / Action Scaling)

import logging
import sys
import time
import csv
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional

import cv2
import hydra
import mujoco
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# --- Project Imports ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from envs.panda_env import PandaEnv
from train.train_semantic_planner import SemanticPlannerLightningModule
from utils.ik_solver import IKSolver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [SOTA-EVAL] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("SOTA_Eval")

# ==============================================================================
# 1. TEMPORAL ENSEMBLER (Unchanged)
# ==============================================================================
class TemporalEnsembler:
    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha
        self.smoothed_pose = None
        self.smoothed_grip = 0.0
        self.gripper_closed = False 

    def reset(self):
        self.smoothed_pose = None
        self.smoothed_grip = 0.0
        self.gripper_closed = False

    def update(self, target_pose: np.ndarray, grip_logit: float) -> Tuple[np.ndarray, float]:
        if self.smoothed_pose is None:
            self.smoothed_pose = target_pose
            self.smoothed_grip = grip_logit
        else:
            pos_smooth = (self.alpha * target_pose[:3]) + ((1 - self.alpha) * self.smoothed_pose[:3])
            quat_new = target_pose[3:]
            quat_old = self.smoothed_pose[3:]
            if np.dot(quat_new, quat_old) < 0.0: quat_new = -quat_new
            quat_smooth = (self.alpha * quat_new) + ((1 - self.alpha) * quat_old)
            quat_smooth /= (np.linalg.norm(quat_smooth) + 1e-8)
            self.smoothed_pose = np.concatenate([pos_smooth, quat_smooth])
            self.smoothed_grip = (self.alpha * grip_logit) + ((1 - self.alpha) * self.smoothed_grip)

        if not self.gripper_closed and self.smoothed_grip > 0.5: self.gripper_closed = True
        elif self.gripper_closed and self.smoothed_grip < -0.5: self.gripper_closed = False
        return self.smoothed_pose, (-1.0 if self.gripper_closed else 1.0)

# ==============================================================================
# 2. VIRTUAL IK & RENDERER (Unchanged)
# ==============================================================================
def solve_virtual_goal_qpos(env, ik_solver, target_pos_world, effective_dt, max_dq):
    temp_qpos = env.data.qpos[:7].copy()
    target_quat_wxyz = np.array([0.0, 1.0, 0.0, 0.0]) 
    target_ee_pose = np.concatenate([target_pos_world, target_quat_wxyz])
    for _ in range(20):
        env.data.qpos[:7] = temp_qpos
        mujoco.mj_kinematics(env.model, env.data)
        mujoco.mj_comPos(env.model, env.data)
        delta_q = ik_solver.compute_delta_action(
            target_ee_pose=target_ee_pose, model=env.model, data=env.data,
            ee_site_id=env.ee_site_id, joint_qpos_indices=np.arange(7),
            effective_dt=effective_dt, max_dq=max_dq
        )
        temp_qpos += delta_q
    return temp_qpos

@contextmanager
def render_physics_safe_goal(env, target_pos_world, goal_qpos_arm=None):
    saved_qpos = env.data.qpos.copy()
    saved_qvel = env.data.qvel.copy()
    saved_ctrl = env.data.ctrl.copy()
    try:
        if goal_qpos_arm is not None:
            env.data.qpos[:7] = goal_qpos_arm
        else:
            home_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
            env.data.qpos[:7] = home_qpos
        env.data.qpos[7:] = 0.04 
        obj_jnt_adr = env.model.jnt_qposadr[env.object_joint_id]
        current_quat = env.data.qpos[obj_jnt_adr+3 : obj_jnt_adr+7].copy()
        env.data.qpos[obj_jnt_adr : obj_jnt_adr+3] = target_pos_world
        env.data.qpos[obj_jnt_adr+3 : obj_jnt_adr+7] = current_quat
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        yield
    finally:
        env.data.qpos[:] = saved_qpos
        env.data.qvel[:] = saved_qvel
        env.data.ctrl[:] = saved_ctrl
        mujoco.mj_forward(env.model, env.data)

# ==============================================================================
# 3. MAIN EVALUATOR
# ==============================================================================

class AWSPEvaluator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        log.info(f"Loading Checkpoint: {cfg.checkpoint_path}")
        self.pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
            cfg.checkpoint_path, map_location=self.device
        )
        self.model = self.pl_module.model.eval().to(self.device)
        self.chunk_size = self.pl_module.cfg.model.get("chunk_size", 10)
        
        xml_path = cfg.get("xml_path", "envs/panda_pick_place.xml")
        self.env = PandaEnv(
            xml_path=xml_path, control_mode='delta', 
            render_mode="rgb_array", 
            action_scaling_factor=0.5 # Default scaling
        )
        
        self.ik_solver = IKSolver(urdf_path="urdf/panda_mujoco_kinematics.urdf")
        
        # Physics Sync
        self.effective_dt = self.env.model.opt.timestep * 20 
        self.max_dq = self.env.ACTION_SCALING_FACTOR / self.effective_dt
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.ensembler = TemporalEnsembler(alpha=0.7)
        self.prev_img_buffer = None

    def run(self):
        out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        video_path = out_dir / "sota_eval_fixed.mp4"
        csv_path = out_dir / "eval_telemetry.csv"
        
        dummy = self.env.reset()[0]
        h, w, _ = self.env.render().shape
        video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        
        telemetry_file = open(csv_path, 'w', newline='')
        telemetry_writer = csv.writer(telemetry_file)
        telemetry_writer.writerow(["ep", "step", "success", "dist", "phase", "target_z", "ee_z", "action_mag"])
        
        success_count = 0
        try:
            for ep_idx in tqdm(range(self.cfg.num_episodes), desc="Evaluator"):
                seed = self.cfg.seed + ep_idx
                success = self.run_episode(ep_idx, seed, video_writer, telemetry_writer)
                if success: success_count += 1
        finally:
            video_writer.release()
            telemetry_file.close()
            self.env.close()
            log.info(f"FINAL SUCCESS RATE: {(success_count/self.cfg.num_episodes)*100:.1f}%")

    def run_episode(self, ep_idx, seed, video_writer, telemetry_writer):
        obs, _ = self.env.reset(seed=seed)
        self.ensembler.reset()
        
        # Goal Image (Using Virtual IK Fix)
        goal_target_pos = obs['goal_pos_world'].copy()
        goal_target_pos[2] += 0.02 
        goal_qpos = solve_virtual_goal_qpos(
            self.env, self.ik_solver, goal_target_pos, 
            self.effective_dt, self.max_dq
        )
        with render_physics_safe_goal(self.env, obs['goal_pos_world'], goal_qpos):
            g_img = self.env.render()
        goal_t = self.transform(Image.fromarray(g_img)).unsqueeze(0).to(self.device)
        
        # History
        curr_img_pil = Image.fromarray(obs['image_primary'])
        self.prev_img_buffer = self.transform(curr_img_pil).unsqueeze(0).to(self.device)
        
        success = False
        
        for step in range(self.cfg.max_steps):
            # Input
            curr_img_pil = Image.fromarray(obs['image_primary'])
            curr_t = self.transform(curr_img_pil).unsqueeze(0).to(self.device)
            proprio_t = torch.from_numpy(obs['proprio']).float().unsqueeze(0).to(self.device)
            
            batch = {'prev_image': self.prev_img_buffer, 'curr_image': curr_t,
                     'goal_image': goal_t, 'curr_proprio': proprio_t}
            self.prev_img_buffer = curr_t.clone()
            
            # Inference
            with torch.no_grad():
                out = self.model(batch)
            
            chunk_pose = out['pose_chunk'][0].cpu().numpy()
            chunk_grip = out['gripper_chunk'][0].cpu().numpy()
            
            target_pose_raw = chunk_pose[0]
            target_grip_raw = chunk_grip[0].item()
            target_pose, grip_cmd = self.ensembler.update(target_pose_raw, target_grip_raw)
            
            # Control
            pos = target_pose[:3]
            quat_xyzw = target_pose[3:]
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            target_ee_pose = np.concatenate([pos, quat_wxyz])
            
            try:
                delta_joints = self.ik_solver.compute_delta_action(
                    target_ee_pose=target_ee_pose,
                    model=self.env.model, data=self.env.data,
                    ee_site_id=self.env.ee_site_id,
                    joint_qpos_indices=np.arange(7),
                    effective_dt=self.effective_dt, max_dq=self.max_dq
                )
            except: delta_joints = np.zeros(7)
            
            # --- [FIX] INVERSE SCALE ACTION ---
            # If PandaEnv multiplies by 0.5, we must divide by 0.5 to get the desired delta
            # But we clip to be safe.
            # The environment likely expects action in [-1, 1].
            # calculated delta_joints are in Radians (e.g. 0.05).
            # We want: action * 0.5 = 0.05  => action = 0.05 / 0.5 = 0.1
            
            scaled_delta_joints = delta_joints / self.env.ACTION_SCALING_FACTOR
            scaled_delta_joints = np.clip(scaled_delta_joints, -1.0, 1.0)
            
            action = np.concatenate([scaled_delta_joints, [grip_cmd]])
            obs, _, terminated, truncated, _ = self.env.step(action)
            
            # Metrics
            dist = np.linalg.norm(obs['object_pos_world'] - obs['goal_pos_world'])
            phase = np.argmax(out['phase_logits'][0].cpu().numpy())
            action_mag = np.linalg.norm(delta_joints)
            
            telemetry_writer.writerow([ep_idx, step, int(success), dist, phase, pos[2], obs['ee_pose_world'][2], action_mag])
            
            frame = self.env.render()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(frame, f"Err:{dist:.3f}m Act:{action_mag:.4f}", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            video_writer.write(frame)
            
            if dist < 0.05: success = True
            if terminated or truncated: break
            
        return success

@hydra.main(version_base=None, config_path="./configs", config_name="evaluate_semantic_planner_config")
def main(cfg: DictConfig):
    evaluator = AWSPEvaluator(cfg)
    evaluator.run()

if __name__ == "__main__":
    main()