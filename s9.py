# FILE: evaluate_semantic_planner.py
# (v9.6 - Heavy Stabilization)

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
from omegaconf import DictConfig
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
    format="%(asctime)s [%(levelname)s] [AWSP-Eval] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("AWSP_Eval")

# ==============================================================================
# 1. HEAVY TRAJECTORY SMOOTHER
# ==============================================================================
class TrajectorySmoother:
    def __init__(self, alpha_pos=0.15, alpha_grip=0.1):
        # Alpha 0.15 = Trust New Data 15%, Trust History 85%
        # This effectively ignores the 20cm jumps
        self.alpha_pos = alpha_pos
        self.alpha_grip = alpha_grip
        self.smooth_pose = None
        self.smooth_grip_logit = 0.0
        self.gripper_state = 1.0 

    def reset(self):
        self.smooth_pose = None
        self.smooth_grip_logit = 0.0
        self.gripper_state = 1.0

    def update(self, raw_pose: np.ndarray, raw_logit: float) -> Tuple[np.ndarray, float]:
        if self.smooth_pose is None:
            self.smooth_pose = raw_pose
            self.smooth_grip_logit = raw_logit
        else:
            # Strong Low-Pass Filter
            self.smooth_pose = (self.alpha_pos * raw_pose) + ((1 - self.alpha_pos) * self.smooth_pose)
            self.smooth_grip_logit = (self.alpha_grip * raw_logit) + ((1 - self.alpha_grip) * self.smooth_grip_logit)

        # Hysteresis
        if self.gripper_state == 1.0 and self.smooth_grip_logit > 2.0: 
            self.gripper_state = -1.0 
        elif self.gripper_state == -1.0 and self.smooth_grip_logit < -2.0:
            self.gripper_state = 1.0 

        return self.smooth_pose, self.gripper_state

# ==============================================================================
# 2. STATE ESTIMATOR
# ==============================================================================
class StateEstimator:
    def __init__(self):
        self.current_phase = 0
        self.thresh_grasp_enter = 0.12 
        self.thresh_grasp_exit = 0.20
        self.thresh_goal_enter = 0.05
        self.prev_is_grasped = False

    def reset(self):
        self.current_phase = 0
        self.prev_is_grasped = False

    def update(self, obs: Dict[str, Any]) -> int:
        is_grasped = obs['is_grasped'][0] > 0.5
        dist = np.linalg.norm(obs['ee_pose_world'][:3] - obs['object_pos_world'])
        dist_goal = np.linalg.norm(obs['object_pos_world'] - obs['goal_pos_world'])

        if is_grasped:
            self.prev_is_grasped = True
            self.current_phase = 3 if dist_goal < self.thresh_goal_enter else 2
        else:
            if self.prev_is_grasped:
                self.current_phase = 4 if dist_goal < self.thresh_goal_enter else 0
                self.prev_is_grasped = False
            else:
                if self.current_phase == 0 and dist < self.thresh_grasp_enter: self.current_phase = 1
                elif self.current_phase == 1 and dist > self.thresh_grasp_exit: self.current_phase = 0
        return self.current_phase

# ==============================================================================
# 3. VIRTUAL GOAL
# ==============================================================================
def calculate_retract_joints(env, ik_solver, target_pos, obj_quat):
    t_pos = np.array([target_pos[0], target_pos[1], 0.55])
    r_obj = R.from_quat([obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]])
    r_target = r_obj * R.from_euler('x', 180, degrees=True)
    
    current_q = env.data.qpos[:7].copy()
    guess = [0.0]*len(ik_solver.chain.links)
    for i, v in enumerate(current_q):
        if i < len(ik_solver._active_idx): guess[ik_solver._active_idx[i]] = v
            
    full = ik_solver.chain.inverse_kinematics(t_pos, r_target.as_matrix(), initial_position=guess)
    return np.array([full[i] for i in ik_solver._active_idx])

@contextmanager
def render_virtual_goal(env, ik_solver, goal_pos):
    saved = (env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy())
    try:
        oa = env.model.jnt_qposadr[env.object_joint_id]
        oq = env.data.qpos[oa+3:oa+7].copy()
        gp = goal_pos.copy()
        if gp[2] < 0.41: gp[2] = 0.42
        env.data.qpos[oa:oa+3] = gp
        env.data.qpos[oa+3:oa+7] = oq
        joints = calculate_retract_joints(env, ik_solver, gp, oq)
        env.data.qpos[:7] = joints
        env.data.qpos[7:] = 0.04
        env.data.qvel[:] = 0
        mujoco.mj_forward(env.model, env.data)
        yield
    finally:
        env.data.qpos[:], env.data.qvel[:], env.data.ctrl[:] = saved
        mujoco.mj_forward(env.model, env.data)

# ==============================================================================
# 4. EVALUATOR
# ==============================================================================
class AWSPEvaluator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
            self.cfg.checkpoint_path, map_location=self.device
        )
        self.model = pl_module.model.eval().to(self.device)

        self.arch_type = "UNKNOWN"
        if hasattr(self.model, 'pose_head') and hasattr(self.model, 'task_phase_embedding'):
            self.arch_type = "v8.0"
        elif hasattr(self.model, 'traj_head') and hasattr(self.model, 'phase_head'):
            self.arch_type = "v9.0"
        
        log.info(f"🧬 ARCHITECTURE: {self.arch_type} | MODE: STABILIZED")

        xml_path = self.cfg.get("xml_path", pl_module.cfg.dataset.get('xml_path', 'envs/panda_pick_place.xml'))
        self.env = PandaEnv(xml_path=xml_path, control_mode='delta', render_mode="rgb_array")
        self.ik_solver = IKSolver(urdf_path="urdf/panda_mujoco_kinematics.urdf")
        
        # Physics
        SIM_SUBSTEPS = 20 
        self.effective_dt = self.env.model.opt.timestep * SIM_SUBSTEPS
        self.max_dq = 3.5 

        # Transform (Normalized)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        
        self.state_estimator = StateEstimator()
        self.smoother = TrajectorySmoother()
        self.prev_img_buffer = None

    def run(self):
        out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        video_file = out_dir / self.cfg.output_video_path


        csv_file = out_dir / "eval_telemetry.csv"
        csv_f = open(csv_file, 'w', newline='')
        writer = csv.writer(csv_f)
        # Define columns for debugging
        writer.writerow([
            "episode", "step", "phase_pred", 
            "grip_logit_raw", "grip_cmd_smoothed", 
            "ee_z_actual", "target_z_commanded", 
            "dist_to_obj", "is_grasped"
        ])
        log.info(f"Logging telemetry to: {csv_file}")

        obs, _ = self.env.reset()
        dummy = self.env.render()
        h, w, _ = dummy.shape
        video_writer = cv2.VideoWriter(str(video_file), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        
        successes = []

        try:
            for ep_idx in tqdm(range(self.cfg.num_episodes), desc="Evaluating"):
                self.env.reset(seed=self.cfg.seed + ep_idx)
                obs = self.env.get_expert_obs()
                self.state_estimator.reset()
                self.smoother.reset()

                with render_virtual_goal(self.env, self.ik_solver, obs['goal_pos_world']):
                    g_img = self.env.render()
                goal_tensor = self.transform(Image.fromarray(g_img)).unsqueeze(0).to(self.device)

                curr_img_pil = Image.fromarray(obs['image_primary'])
                self.prev_img_buffer = self.transform(curr_img_pil).unsqueeze(0).to(self.device)

                episode_success = False
                for step in range(self.cfg.max_steps):
                    phase = self.state_estimator.update(obs)
                    curr_tensor = self.transform(Image.fromarray(obs['image_primary'])).unsqueeze(0).to(self.device)
                    proprio = torch.from_numpy(obs['proprio']).float().unsqueeze(0).to(self.device)


                    batch = {
                        'prev_image': self.prev_img_buffer, 'curr_image': curr_tensor,
                        'goal_image': goal_tensor, 'curr_proprio': proprio
                    }
                    self.prev_img_buffer = curr_tensor.clone()

                    with torch.no_grad():
                        out = self.model(batch)

                    target_pose = None
                    raw_grip = 0.0
                    
                    chunk = out['pose_chunk'].cpu().numpy()[0]
                    phase_logits = out['phase_logits'].cpu().numpy()[0]
                    predicted_phase = np.argmax(phase_logits)
                    # Lookahead: Index 4 (Middle of chunk) is safer than End
                    target_pose = chunk[4] 
                    raw_grip = out['gripper_chunk'].cpu().numpy()[0][0].item()
                    raw_target_pose = chunk[4] 
                    

                    # HEAVY SMOOTHING APPLIED HERE
                    target_pose, gripper_cmd = self.smoother.update(target_pose, raw_grip)

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

                    # Boost gain slightly to help against friction
                    delta_joints = delta_joints * 1.5

                    action = np.concatenate([delta_joints, [gripper_cmd]])
                    obs, _, terminated, truncated, _ = self.env.step(action)

                    dist_goal = np.linalg.norm(obs['object_pos_world'] - obs['goal_pos_world'])
                    if dist_goal < 0.06 and obs['object_pos_world'][2] > 0.41 and obs['is_grasped'][0] > 0.5:
                        episode_success = True

                    frame = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
                    phase_txt = ["REACH", "GRASP", "MOVE", "PLACE"][min(phase, 3)]
                    cv2.putText(frame, f"{self.arch_type} | {phase_txt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    video_writer.write(frame)


                    ee_z_actual = obs['ee_pose_world'][2]
                    target_z_cmd = raw_target_pose[2] 
                    dist_to_obj = np.linalg.norm(obs['ee_pose_world'][:3] - obs['object_pos_world'])
                    
                    writer.writerow([
                        ep_idx, 
                        step, 
                        predicted_phase, 
                        f"{raw_grip:.4f}",          # Fixed variable name (raw_grip vs raw_grip_logit)
                        f"{gripper_cmd:.1f}",       
                        f"{ee_z_actual:.4f}",       
                        f"{target_z_cmd:.4f}",      
                        f"{dist_to_obj:.4f}",       
                        obs['is_grasped'][0]
                    ])
                    if episode_success or terminated or truncated: break

                successes.append(episode_success)
                log.info(f"Ep {ep_idx}: {'✅ SUCCESS' if episode_success else '❌ FAIL'}")

        finally:
            video_writer.release()
            self.env.close()
            csv_f.close() 
            rate = (sum(successes) / len(successes)) * 100 if successes else 0.0
            log.info(f"FINAL SUCCESS RATE: {rate:.1f}%")

@hydra.main(version_base=None, config_path="./configs", config_name="evaluate_semantic_planner_config")
def main(cfg: DictConfig):
    evaluator = AWSPEvaluator(cfg)
    evaluator.run()

if __name__ == "__main__":
    main()