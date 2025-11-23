# FILE: evaluate_semantic_planner.py
# (v9.8 - Fixed Smoother + Data Matching)

import logging
import sys
import numpy as np
import torch
import cv2
import hydra
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from contextlib import contextmanager
from typing import Any, Dict, Tuple
import mujoco
# --- Imports ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from envs.panda_env import PandaEnv
from train.train_semantic_planner import SemanticPlannerLightningModule
from utils.ik_solver import IKSolver

log = logging.getLogger("AWSP_Eval")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ==============================================================================
# 🚨 CRITICAL CONFIGURATION: DATA MATCHING 🚨
# ==============================================================================
# UPDATE THESE IF YOUR TRAINING USED PROPRIO NORMALIZATION
# If these are None, we assume Raw Proprioception.
PROPRIO_MEAN = None 
PROPRIO_STD = None

# SigLIP Defaults
IMG_MEAN = [0.5, 0.5, 0.5]
IMG_STD = [0.5, 0.5, 0.5]
# ==============================================================================

class TrajectorySmoother:
    def __init__(self, alpha=0.8): 
        self.alpha = alpha
        self.reset() # Initialize state

    def reset(self):
        self.smooth_pose = None
        self.smooth_grip = 0.0
        self.grip_state = 1.0

    def update(self, pose, grip_logit):
        if self.smooth_pose is None:
            self.smooth_pose = pose
            self.smooth_grip = grip_logit
        else:
            self.smooth_pose = self.alpha * self.smooth_pose + (1 - self.alpha) * pose
            self.smooth_grip = self.alpha * self.smooth_grip + (1 - self.alpha) * grip_logit
        
        # Hysteresis
        if self.grip_state == 1.0 and self.smooth_grip > 2.0: self.grip_state = -1.0
        elif self.grip_state == -1.0 and self.smooth_grip < -2.0: self.grip_state = 1.0
        return self.smooth_pose, self.grip_state

# Virtual Goal Helper
def calculate_retract_joints(env, ik_solver, target_pos, obj_quat):
    t_pos = np.array([target_pos[0], target_pos[1], 0.55])
    r_obj = R.from_quat([obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]])
    r_target = r_obj * R.from_euler('x', 180, degrees=True)
    guess = env.data.qpos[:7].copy()
    full = ik_solver.chain.inverse_kinematics(t_pos, r_target.as_matrix(), initial_position=guess)
    return np.array([full[i] for i in ik_solver._active_idx])

@contextmanager
def render_virtual_goal(env, ik_solver, goal_pos):
    saved = (env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy())
    try:
        oa = env.model.jnt_qposadr[env.object_joint_id]
        oq = env.data.qpos[oa+3:oa+7].copy()
        gp = goal_pos.copy(); gp[2] = max(gp[2], 0.42)
        env.data.qpos[oa:oa+3] = gp; env.data.qpos[oa+3:oa+7] = oq
        joints = calculate_retract_joints(env, ik_solver, gp, oq)
        env.data.qpos[:7] = joints; env.data.qvel[:] = 0
        mujoco.mj_forward(env.model, env.data)
        yield
    finally:
        env.data.qpos[:], env.data.qvel[:], env.data.ctrl[:] = saved
        mujoco.mj_forward(env.model, env.data)

class AWSPEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Model
        log.info(f"Loading: {cfg.checkpoint_path}")
        pl_module = SemanticPlannerLightningModule.load_from_checkpoint(cfg.checkpoint_path, map_location=self.device)
        self.model = pl_module.model.eval().to(self.device)

        # Arch Detection
        self.is_v9 = hasattr(self.model, 'traj_head')
        log.info(f"Architecture: {'v9.0 (History)' if self.is_v9 else 'v8.0 (Phase)'}")
        
        # Env & IK
        xml_path = cfg.get("xml_path", pl_module.cfg.dataset.get('xml_path', 'envs/panda_pick_place.xml'))
        self.env = PandaEnv(xml_path=xml_path, control_mode='delta', render_mode="rgb_array")
        self.ik = IKSolver(urdf_path="urdf/panda_mujoco_kinematics.urdf")
        
        # Physics Params
        SIM_SUBSTEPS = 20
        self.dt = self.env.model.opt.timestep * SIM_SUBSTEPS
        self.max_dq = 2.0 # Safe Limit

        # Transform (With Normalization)
        self.tf = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
        ])
        self.smoother = TrajectorySmoother()

    def normalize_proprio(self, proprio):
        if PROPRIO_MEAN is not None and PROPRIO_STD is not None:
             mean = torch.tensor(PROPRIO_MEAN, device=self.device).float()
             std = torch.tensor(PROPRIO_STD, device=self.device).float()
             return (proprio - mean) / (std + 1e-6)
        return proprio

    def run(self):
        out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        video_path = str(out_dir / self.cfg.output_video_path)
        
        # Dummy render to get shape
        self.env.reset()
        dummy = self.env.render()
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (dummy.shape[1], dummy.shape[0]))
        
        successes = []

        try:
            for ep in tqdm(range(self.cfg.num_episodes)):
                # Reset
                self.env.reset(seed=self.cfg.seed + ep)
                obs = self.env.get_expert_obs()
                self.smoother.reset() # <--- FIXED: Method now exists
                
                # Goal
                with render_virtual_goal(self.env, self.ik, obs['goal_pos_world']):
                    g_img = self.tf(Image.fromarray(self.env.render())).unsqueeze(0).to(self.device)
                
                # History (v9)
                prev_img = self.tf(Image.fromarray(obs['image_primary'])).unsqueeze(0).to(self.device)

                success = False
                for step in range(self.cfg.max_steps):
                    curr_img = self.tf(Image.fromarray(obs['image_primary'])).unsqueeze(0).to(self.device)
                    proprio = torch.from_numpy(obs['proprio']).float().unsqueeze(0).to(self.device)
                    
                    # 1. Normalize Data
                    proprio = self.normalize_proprio(proprio)

                    # 2. Batch Construction
                    batch = {'goal_image': g_img, 'curr_proprio': proprio}
                    if self.is_v9:
                        batch.update({'prev_image': prev_img, 'curr_image': curr_img})
                        prev_img = curr_img.clone()
                    else:
                        # Default Phase 0 (Reach) if not using Oracle
                        batch.update({'initial_image': curr_img, 'task_phase': torch.tensor([0], device=self.device)})

                    # 3. Inference
                    with torch.no_grad(): out = self.model(batch)

                    # 4. Unpack & Smooth
                    if self.is_v9:
                        chunk = out['pose_chunk'].cpu().numpy()[0]
                        raw_pose = chunk[4] # Use Middle of Chunk
                        raw_grip = out['gripper_chunk'].cpu().numpy()[0][0].item()
                    else:
                        raw_pose = out['pose'].cpu().numpy()[0]
                        raw_grip = out['gripper_logit'].item()

                    target, grip = self.smoother.update(raw_pose, raw_grip)

                    # 5. Act
                    try:
                        dq = self.ik.compute_delta_action(
                            target, self.env.model, self.env.data, 
                            self.env.ee_site_id, np.arange(7), self.dt, self.max_dq
                        )
                    except: dq = np.zeros(7)
                    
                    # Boost Gain slightly
                    dq = dq * 1.5
                    
                    obs, _, term, trunc, _ = self.env.step(np.concatenate([dq, [grip]]))
                    
                    # Visualize
                    frame = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
                    cv2.putText(frame, f"Step: {step}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    video_writer.write(frame)
                    
                    # Success Check
                    dist = np.linalg.norm(obs['object_pos_world'] - obs['goal_pos_world'])
                    if dist < 0.05 and obs['is_grasped'][0] > 0.5:
                        success = True
                        break
                    if term or trunc: break
                
                successes.append(success)
                log.info(f"Ep {ep}: {'✅ SUCCESS' if success else '❌ FAIL'}")

        finally:
            video_writer.release()
            self.env.close()
            sr = sum(successes)/len(successes) if successes else 0
            log.info(f"Final Success Rate: {sr*100:.1f}%")

@hydra.main(version_base=None, config_path="./configs", config_name="evaluate_semantic_planner_config")
def main(cfg): AWSPEvaluator(cfg).run()

if __name__ == "__main__": main()