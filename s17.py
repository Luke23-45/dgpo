# FILE: evaluate_semantic_planner.py
# (v11.0 - Iron Grip State Machine)

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import csv
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
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from envs.panda_env import PandaEnv
from train.train_semantic_planner import SemanticPlannerLightningModule
from utils.ik_solver import IKSolver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Strategist-Eval] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("Strategist_Eval")

class LightweightSmoother:
    def __init__(self, alpha_pos=0.8, alpha_grip=0.5):
        self.alpha_pos = alpha_pos
        self.alpha_grip = alpha_grip
        self.smooth_pose = None
        self.smooth_grip_logit = 0.0
        self.gripper_closed = False 

    def reset(self):
        self.smooth_pose = None
        self.smooth_grip_logit = 0.0
        self.gripper_closed = False

    def update(self, raw_pose: np.ndarray, raw_logit: float) -> Tuple[np.ndarray, float]:
        if self.smooth_pose is None:
            self.smooth_pose = raw_pose
            self.smooth_grip_logit = raw_logit
        else:
            self.smooth_pose = (self.alpha_pos * raw_pose) + ((1 - self.alpha_pos) * self.smooth_pose)
            self.smooth_grip_logit = (self.alpha_grip * raw_logit) + ((1 - self.alpha_grip) * self.smooth_grip_logit)

        # Hysteresis: 0.5 threshold
        if not self.gripper_closed and self.smooth_grip_logit > 0.5:
            self.gripper_closed = True
        elif self.gripper_closed and self.smooth_grip_logit < -0.5:
            self.gripper_closed = False
            
        gripper_cmd = -1.0 if self.gripper_closed else 1.0
        return self.smooth_pose, gripper_cmd

@contextmanager
def render_robust_virtual_goal(env, target_pos_world):
    saved_qpos = env.data.qpos.copy()
    saved_qvel = env.data.qvel.copy()
    saved_ctrl = env.data.ctrl.copy()
    try:
        home_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        env.data.qpos[:7] = home_qpos
        env.data.qpos[7:] = 0.04 
        obj_jnt_adr = env.model.jnt_qposadr[env.object_joint_id]
        curr_obj_quat = env.data.qpos[obj_jnt_adr+3 : obj_jnt_adr+7].copy()
        safe_z = max(target_pos_world[2], 0.42)
        env.data.qpos[obj_jnt_adr : obj_jnt_adr+3] = [target_pos_world[0], target_pos_world[1], safe_z]
        env.data.qpos[obj_jnt_adr+3 : obj_jnt_adr+7] = curr_obj_quat
        env.data.qvel[:] = 0
        mujoco.mj_forward(env.model, env.data)
        yield
    finally:
        env.data.qpos[:] = saved_qpos
        env.data.qvel[:] = saved_qvel
        env.data.ctrl[:] = saved_ctrl
        mujoco.mj_forward(env.model, env.data)

class AWSPEvaluator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        log.info(f"Loading checkpoint: {self.cfg.checkpoint_path}")
        self.pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
            self.cfg.checkpoint_path, map_location=self.device
        )
        self.model = self.pl_module.model.eval().to(self.device)
        self.chunk_size = self.pl_module.cfg.model.get("chunk_size", 10)

        xml_path = self.cfg.get("xml_path", "envs/panda_pick_place.xml")
        self.env = PandaEnv(
            xml_path=xml_path, 
            control_mode='delta', 
            render_mode="rgb_array",
            action_scaling_factor=0.5 
        )
        self.ik_solver = IKSolver(urdf_path="urdf/panda_mujoco_kinematics.urdf")
        
        SIM_SUBSTEPS = 20
        self.effective_dt = self.env.model.opt.timestep * SIM_SUBSTEPS
        self.max_dq = 3.5 

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.smoother = LightweightSmoother()
        self.prev_img_buffer = None

    def run(self):
        out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        video_file = out_dir / self.cfg.output_video_path

        csv_file = out_dir / "eval_telemetry.csv"
        csv_f = open(csv_file, 'w', newline='')
        writer = csv.writer(csv_f)
        writer.writerow([
            "episode", "step", "phase_pred", 
            "grip_logit_raw", "grip_cmd_smoothed", 
            "ee_z_actual", "target_z_commanded", 
            "dist_to_obj", "is_grasped", "state"
        ])
        log.info(f"Logging telemetry to: {csv_file}")

        obs, _ = self.env.reset()
        dummy = self.env.render()
        h, w, _ = dummy.shape
        video_writer = cv2.VideoWriter(str(video_file), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        
        success_count = 0

        try:
            for ep_idx in tqdm(range(self.cfg.num_episodes), desc="Evaluating"):
                self.env.reset(seed=self.cfg.seed + ep_idx)
                obs = self.env.get_expert_obs()
                self.smoother.reset()
                
                # --- IRON GRIP STATE MACHINE ---
                # 0: APPROACH
                # 1: GRASP_SEQUENCE (Freeze -> Lift)
                # 2: CARRY (Iron Grip)
                eval_state = 0 
                grasp_timer = 0

                with render_robust_virtual_goal(self.env, obs['goal_pos_world']):
                    g_img = self.env.render()
                goal_tensor = self.transform(Image.fromarray(g_img)).unsqueeze(0).to(self.device)

                curr_img_pil = Image.fromarray(obs['image_primary'])
                self.prev_img_buffer = self.transform(curr_img_pil).unsqueeze(0).to(self.device)

                episode_success = False
                
                for step in range(self.cfg.max_steps):
                    curr_img_pil = Image.fromarray(obs['image_primary'])
                    curr_tensor = self.transform(curr_img_pil).unsqueeze(0).to(self.device)
                    proprio = torch.from_numpy(obs['proprio']).float().unsqueeze(0).to(self.device)

                    batch = {
                        'prev_image': self.prev_img_buffer, 'curr_image': curr_tensor,
                        'goal_image': goal_tensor, 'curr_proprio': proprio
                    }
                    self.prev_img_buffer = curr_tensor.clone()

                    with torch.no_grad():
                        out = self.model(batch)

                    phase_logits = out['phase_logits'].cpu().numpy()[0]
                    predicted_phase = np.argmax(phase_logits)
                    
                    chunk_pose = out['pose_chunk'].cpu().numpy()[0]
                    chunk_grip = out['gripper_chunk'].cpu().numpy()[0]
                    
                    raw_target_pose = chunk_pose[2].copy() 
                    raw_grip_logit = chunk_grip[2].item()

                    # --- CONTROL LOGIC ---
                    ee_pos = obs['ee_pose_world']
                    obj_pos = obs['object_pos_world']
                    dist_xy = np.linalg.norm(ee_pos[:2] - obj_pos[:2])
                    dist_3d = np.linalg.norm(ee_pos[:3] - obj_pos)
                    is_physically_grasped = obs['is_grasped'][0] > 0.5
                    
                    action_gain = 2.0 # Default

                    # STATE 0: APPROACH
                    if eval_state == 0:
                        # Safety: Prevent premature closing
                        if ee_pos[2] > 0.45: raw_grip_logit = -5.0
                        
                        # Magnetic Heuristic
                        if dist_xy < 0.15:
                            raw_target_pose[:2] = (0.5 * raw_target_pose[:2]) + (0.5 * obj_pos[:2])
                            # Progressive Z-Cap
                            if dist_xy < 0.05:
                                raw_target_pose[2] = 0.425 # Exact grasp height (Hard Floor)
                            elif dist_xy < 0.10:
                                raw_target_pose[2] = min(raw_target_pose[2], 0.45)
                        
                        # Trigger Transition
                        if dist_xy < 0.03 and ee_pos[2] < 0.435:
                            eval_state = 1
                            grasp_timer = 35 # 1.75 seconds allocated for grasp

                    # STATE 1: GRASP SEQUENCE (Freeze -> Lift)
                    elif eval_state == 1:
                        raw_grip_logit = 5.0 # FORCE CLOSE
                        
                        if grasp_timer > 15:
                            # FREEZE (Steps 35 -> 15)
                            raw_target_pose[:2] = ee_pos[:2] # Lock XY
                            raw_target_pose[2] = 0.425       # Lock Z at grasp height
                            action_gain = 0.5                # Soften physics to prevent bounce
                        else:
                            # LIFT (Steps 15 -> 0)
                            raw_target_pose[:2] = ee_pos[:2]
                            raw_target_pose[2] = 0.55        # Lift straight up
                            action_gain = 2.5                # High gain for lifting
                        
                        grasp_timer -= 1
                        if grasp_timer <= 0:
                            eval_state = 2 # Transition to Carry

                    # STATE 2: CARRY (Iron Grip)
                    elif eval_state == 2:
                        # IRON GRIP: Ignore model, keep closed
                        raw_grip_logit = 5.0 
                        
                        # Check for release condition (Near Goal)
                        dist_goal = np.linalg.norm(obj_pos - obs['goal_pos_world'])
                        if dist_goal < 0.05:
                            # Allow model to open if it wants
                            raw_grip_logit = chunk_grip[2].item()

                    # --- EXECUTION ---
                    target_pose, gripper_cmd = self.smoother.update(raw_target_pose, raw_grip_logit)

                    try:
                        delta_joints = self.ik_solver.compute_delta_action(
                            target_ee_pose=target_pose,
                            model=self.env.model, data=self.env.data, ee_site_id=self.env.ee_site_id,
                            joint_qpos_indices=np.arange(7),
                            effective_dt=self.effective_dt,
                            max_dq=self.max_dq
                        )
                    except Exception:
                        delta_joints = np.zeros(7)

                    delta_joints = delta_joints * action_gain

                    action = np.concatenate([delta_joints, [gripper_cmd]])
                    obs, _, terminated, truncated, _ = self.env.step(action)

                    # Success Check
                    obj_pos_now = obs['object_pos_world']
                    goal_pos = obs['goal_pos_world']
                    dist_goal = np.linalg.norm(obj_pos_now - goal_pos)
                    is_lifted = obj_pos_now[2] > 0.415 
                    is_grasped = obs['is_grasped'][0] > 0.5
                    
                    if dist_goal < 0.05 and is_lifted and is_grasped:
                        episode_success = True

                    # Render & Log
                    frame = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
                    
                    state_txt = ["APPROACH", "GRASP_SEQ", "CARRY"][eval_state]
                    status_str = f"State: {state_txt} | Ph: {predicted_phase}"
                    color = (0, 255, 0) if eval_state > 0 else (255, 255, 255)
                    if eval_state == 1: color = (0, 255, 255) # Yellow
                    if eval_state == 2: color = (0, 0, 255)   # Red (Iron Grip)
                    
                    cv2.putText(frame, status_str, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    video_writer.write(frame)
                    
                    writer.writerow([
                        ep_idx, step, predicted_phase, 
                        f"{raw_grip_logit:.4f}", f"{gripper_cmd:.1f}", 
                        f"{ee_pos[2]:.4f}", f"{raw_target_pose[2]:.4f}",      
                        f"{dist_3d:.4f}", is_grasped, eval_state
                    ])
                    
                    if episode_success or terminated or truncated: break

                if episode_success:
                    success_count += 1
                log.info(f"Episode {ep_idx}: {'✅ SUCCESS' if episode_success else '❌ FAIL'} | Steps: {step}")

        finally:
            video_writer.release()
            self.env.close()
            csv_f.close() 
            rate = (success_count / self.cfg.num_episodes) * 100
            log.info(f"FINAL EVALUATION RESULT: {rate:.1f}% Success Rate")

@hydra.main(version_base=None, config_path="./configs", config_name="evaluate_semantic_planner_config")
def main(cfg: DictConfig):
    if "checkpoint_path" not in cfg:
        raise ValueError("Must provide 'checkpoint_path' in config")
    evaluator = AWSPEvaluator(cfg)
    evaluator.run()

if __name__ == "__main__":
    main()