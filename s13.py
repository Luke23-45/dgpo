"""
SOTA Semantic Planner Evaluation System (vFinal).

This script implements a Production-Grade "Grey-Box" evaluation suite.
It bridges the gap between Deep Learning predictions and Physical Control
by enforcing strict coordinate transforms, phase consistency, and AR visualization.

Architecture:
1.  **Hybrid State Estimator**: Fuses geometric thresholds with physical sensor data 
    (gripper width) to prevent Phase-Lock.
2.  **Coordinate Transformer**: Converts Model Predictions (World) -> IK Targets (Base).
3.  **AR Visualizer**: Projects 3D neural predictions onto 2D camera frames for 
    instant visual debugging.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# --- Project Imports ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from envs.panda_env import PandaEnv
from models.semantic_planner import SemanticPlanner
from train.train_semantic_planner import SemanticPlannerLightningModule
from utils.ik_solver import IKSolver

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [SOTA-Eval] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("SOTA_Eval")


# ==============================================================================
# 1. UTILITY: AR VISUALIZATION (3D -> 2D Projection)
# ==============================================================================

def project_point_to_image(
    pos_world: np.ndarray, 
    camera_params: Dict[str, Any]
) -> Optional[Tuple[int, int]]:
    """
    Projects a 3D World Point onto the 2D Camera Image Plane.
    Used to draw the "Green Dot" (Target) and "Red Dot" (Actual EE).
    """
    try:
        # 1. Unpack Camera Extrinsics
        cam_pos = np.array(camera_params["pos"])
        cam_quat_xyzw = np.array(camera_params["quat_xyzw"])
        
        # World -> Camera Rotation
        R_wc = R.from_quat(cam_quat_xyzw).as_matrix()
        R_cw = R_wc.T
        
        # 2. Transform Point to Camera Frame
        # Vector from Camera to Point
        p_cam = R_cw @ (pos_world - cam_pos)
        
        # MuJoCo Camera looks down -Z axis. Points behind camera have Z > 0 (after transform sometimes)
        # Standard convention: Z should be negative for points in front.
        # Let's verify MuJoCo convention: Look -Z, Up +Y, Right +X.
        
        x, y, z = p_cam
        
        # Z-Buffer check (is point in front of camera?)
        if z > -0.01: 
            return None # Behind camera

        # 3. Intrinsics Projection (Pinhole Model)
        fovy = camera_params["fovy"] # Degrees
        height = camera_params["height"]
        width = camera_params["width"]
        
        # Focal Length calculation
        # f = (h / 2) / tan(fovy / 2)
        f = (height / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
        
        # Project
        u = (x / -z) * f + (width / 2.0)
        v = (y / -z) * f + (height / 2.0) # Assuming square pixels aspect ratio correction handled by f
        
        # Bounds check
        if 0 <= u < width and 0 <= v < height:
            return (int(u), int(v))
        return None
        
    except Exception:
        return None


# ==============================================================================
# 2. ROBUST STATE ESTIMATOR (The "Brain")
# ==============================================================================

class RobustStateEstimator:
    """
    A Hybrid FSM that combines Geometric Thresholds with Physical Sensor Data.
    This prevents the 'Hanging' issue where the robot gets stuck in Reach phase.
    """
    PHASE_REACH = 0
    PHASE_GRASP = 1
    PHASE_TRANSPORT = 2
    PHASE_PLACE = 3
    PHASE_RETRACT = 4

    def __init__(self):
        self.current_phase = self.PHASE_REACH
        self.prev_is_grasped = False
        
        # Tuned Thresholds (Meters)
        self.DIST_ENTER_GRASP = 0.10  # Enter Grasp phase when close
        self.DIST_EXIT_GRASP = 0.20   # Revert to Reach if we slip far away
        self.DIST_AT_GOAL = 0.07      # Enter Place phase
        self.Z_LIFT_HEIGHT = 0.45     # Table is ~0.42. If Obj > 0.45, it is lifted.

    def reset(self):
        self.current_phase = self.PHASE_REACH
        self.prev_is_grasped = False

    def update(self, obs: Dict[str, Any]) -> int:
        """
        Determines the Task Phase ID (0-4) based on world state.
        """
        # 1. Extract Physical Telemetry
        is_physically_grasped = obs['is_grasped'][0] > 0.5
        
        ee_pos = obs['ee_pose_world'][:3]
        obj_pos = obs['object_pos_world']
        goal_pos = obs['goal_pos_world']
        
        dist_ee_obj = np.linalg.norm(ee_pos - obj_pos)
        dist_obj_goal = np.linalg.norm(obj_pos - goal_pos)
        obj_z = obj_pos[2]

        # 2. Phase Logic
        if is_physically_grasped:
            self.prev_is_grasped = True
            
            # If we are holding it, we are either Transporting or Placing
            if dist_obj_goal < self.DIST_AT_GOAL:
                self.current_phase = self.PHASE_PLACE
            else:
                # Force Transport phase if holding, regardless of height initially
                self.current_phase = self.PHASE_TRANSPORT
        
        else:
            # Not holding
            if self.prev_is_grasped:
                # We WERE holding it. Did we succeed or drop it?
                if dist_obj_goal < self.DIST_AT_GOAL:
                    self.current_phase = self.PHASE_RETRACT # Success
                else:
                    # We dropped it mid-air. Restart.
                    self.current_phase = self.PHASE_REACH
                self.prev_is_grasped = False
            else:
                # Standard approach sequence
                if self.current_phase == self.PHASE_REACH:
                    if dist_ee_obj < self.DIST_ENTER_GRASP:
                        self.current_phase = self.PHASE_GRASP
                
                elif self.current_phase == self.PHASE_GRASP:
                    if dist_ee_obj > self.DIST_EXIT_GRASP:
                        self.current_phase = self.PHASE_REACH # Retry approach

        return self.current_phase


# ==============================================================================
# 3. TRAJECTORY SMOOTHER (The "Filter")
# ==============================================================================

class TrajectorySmoother:
    """
    Applies Exponential Moving Average (EMA) to Pose and
    Strict Hysteresis to Gripper commands to prevent jitter.
    """
    def __init__(self, alpha_pos=0.6, alpha_grip=0.3):
        self.alpha_pos = alpha_pos
        self.alpha_grip = alpha_grip
        self.smooth_pose = None
        self.smooth_grip_logit = None
        self.gripper_command = 1.0 # Default Open
        self.lock_timer = 0

    def reset(self):
        self.smooth_pose = None
        self.smooth_grip_logit = None
        self.gripper_command = 1.0
        self.lock_timer = 0

    def update(self, raw_pose: np.ndarray, raw_logit: float) -> Tuple[np.ndarray, float]:
        # 1. Pose Smoothing
        if self.smooth_pose is None:
            self.smooth_pose = raw_pose.copy()
            self.smooth_grip_logit = raw_logit
        else:
            # Linear interp for Pos
            self.smooth_pose[:3] = self.alpha_pos * raw_pose[:3] + (1 - self.alpha_pos) * self.smooth_pose[:3]
            # Linear interp for Quat (Approximation is fine for small steps)
            self.smooth_pose[3:] = self.alpha_pos * raw_pose[3:] + (1 - self.alpha_pos) * self.smooth_pose[3:]
            # Re-normalize quaternion
            self.smooth_pose[3:] /= np.linalg.norm(self.smooth_pose[3:])
            
            self.smooth_grip_logit = self.alpha_grip * raw_logit + (1 - self.alpha_grip) * self.smooth_grip_logit

        # 2. Gripper Latching (Hysteresis)
        CLOSE_THRESH = 0.0 # Logit > 0 means p > 0.5
        OPEN_THRESH = -0.5
        
        if self.lock_timer > 0:
            self.lock_timer -= 1
        else:
            if self.gripper_command > 0: # Open
                if self.smooth_grip_logit > CLOSE_THRESH:
                    self.gripper_command = -1.0
                    self.lock_timer = 10 # Lock for stability
            else: # Closed
                if self.smooth_grip_logit < OPEN_THRESH:
                    self.gripper_command = 1.0
                    self.lock_timer = 10

        return self.smooth_pose.copy(), self.gripper_command


# ==============================================================================
# 4. MAIN EVALUATOR CLASS
# ==============================================================================

class SOTAEvaluator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- Load Model ---
        log.info(f"Loading Model from: {cfg.checkpoint_path}")
        pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
            cfg.checkpoint_path, map_location=self.device
        )
        self.model = pl_module.model.eval().to(self.device)
        
        # --- Load Environment ---
        # Fallback to xml in checkpoint config if not provided
        xml_path = cfg.get("xml_path", pl_module.cfg.dataset.get("xml_path", "envs/panda_pick_place.xml"))
        log.info(f"Initializing Env: {xml_path}")
        
        self.env = PandaEnv(xml_path=xml_path, control_mode='delta', render_mode="rgb_array")
        
        # --- Load IK Solver ---
        urdf_path = cfg.get("urdf_path", "urdf/panda_mujoco_kinematics.urdf")
        self.ik_solver = IKSolver(urdf_path=urdf_path)
        
        # --- Calibrate Control ---
        # Hardcoded matching panda_env.py internals
        SIM_SUBSTEPS = 20
        self.effective_dt = self.env.model.opt.timestep * SIM_SUBSTEPS
        self.max_dq = self.env.ACTION_SCALING_FACTOR / self.effective_dt
        log.info(f"Control Calibrated: dt={self.effective_dt}, max_dq={self.max_dq}")

        # --- Components ---
        self.estimator = RobustStateEstimator()
        self.smoother = TrajectorySmoother()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor()
        ])

    def _transform_world_to_base(self, target_pose_world: np.ndarray) -> np.ndarray:
        """
        CRITICAL FIX: Transforms the Model's World-Frame prediction into the 
        Robot Base-Frame required by the IK Solver.
        """
        base_pos, base_quat_xyzw = self.env.get_base_pose()
        
        # Create transforms
        R_base_world = R.from_quat(base_quat_xyzw).inv()
        
        # Position: R_bw * (P_w - P_base)
        pos_base = R_base_world.apply(target_pose_world[:3] - base_pos)
        
        # Orientation: R_bw * R_w
        rot_world = R.from_quat(target_pose_world[3:])
        rot_base = R_base_world * rot_world
        
        return np.concatenate([pos_base, rot_base.as_quat()])

    @contextmanager
    def _dream_goal(self, goal_pos: np.ndarray):
        """Teleports objects to create the 'Goal Image' input."""
        saved_state = self.env.get_mj_state()
        try:
            # Teleport Object to Goal
            obj_addr = self.env.model.jnt_qposadr[self.env.object_joint_id]
            q_old = self.env.data.qpos[obj_addr+3:obj_addr+7].copy()
            
            # Safely place on table
            safe_goal = goal_pos.copy()
            if safe_goal[2] < 0.415: safe_goal[2] = 0.415
            
            self.env.data.qpos[obj_addr:obj_addr+3] = safe_goal
            self.env.data.qpos[obj_addr+3:obj_addr+7] = q_old
            
            # Teleport Robot Home (so it doesn't obscure goal)
            self.env.data.qpos[:7] = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
            self.env.data.qpos[7:9] = 0.04
            
            mujoco.mj_forward(self.env.model, self.env.data)
            yield
        finally:
            self.env.set_mj_state(saved_state)

    def run(self):
        out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        video_path = out_dir / self.cfg.output_video_path
        csv_path = out_dir / self.cfg.output_csv_path
        
        # Video Setup
        dummy = self.env.render()
        h, w, _ = dummy.shape
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        
        # CSV Setup
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["ep", "step", "phase", "success", "dist_goal", "z_height"])
        
        success_count = 0
        
        for ep_idx in tqdm(range(self.cfg.num_episodes), desc="Evaluating"):
            seed = self.cfg.seed + ep_idx
            self.env.reset(seed=seed)
            self.estimator.reset()
            self.smoother.reset()
            
            obs = self.env.get_expert_obs()
            
            # 1. Dream Goal Image
            with self._dream_goal(obs['goal_pos_world']):
                goal_img_np = self.env.render()
            
            # Preprocess inputs
            goal_tensor = self.transform(Image.fromarray(goal_img_np)).unsqueeze(0).to(self.device)
            
            done = False
            success = False
            
            for step in range(self.cfg.max_steps):
                # 2. State Estimation
                phase = self.estimator.update(obs)
                
                # 3. Inference
                img_tensor = self.transform(Image.fromarray(obs['image_primary'])).unsqueeze(0).to(self.device)
                proprio_tensor = torch.from_numpy(obs['proprio']).float().unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    out = self.model({
                        'initial_image': img_tensor,
                        'goal_image': goal_tensor,
                        'task_phase': torch.tensor([phase], device=self.device),
                        'current_proprio': proprio_tensor
                    })
                
                raw_pose = out['pose'].squeeze().cpu().numpy()
                raw_logit = out['gripper_logit'].item()
                
                # 4. Smoothing & Latching
                target_pose_world, gripper_cmd = self.smoother.update(raw_pose, raw_logit)
                
                # 5. Safety Clamps (Physics Guardrails)
                # Prevent diving through the table
                TABLE_Z = 0.41
                if target_pose_world[2] < TABLE_Z: 
                    target_pose_world[2] = TABLE_Z
                
                # 6. Coordinate Transform (World -> Base)
                target_pose_base = self._transform_world_to_base(target_pose_world)
                
                # 7. IK Solving
                try:
                    current_joints = obs['proprio'][:7] # Use purely proprioceptive joints
                    delta_joints = self.ik_solver.compute_delta_action(
                        target_ee_pose=target_pose_base,
                        model=self.env.model,
                        data=self.env.data,
                        ee_site_id=self.env.ee_site_id,
                        joint_qpos_indices=np.arange(7),
                        effective_dt=self.effective_dt,
                        max_dq=self.max_dq
                    )
                except Exception:
                    # Fallback: Mild retreat
                    delta_joints = np.zeros(7)
                    delta_joints[1] = -0.07 # Retract shoulder slightly

                # 8. Step
                action = np.concatenate([delta_joints, [gripper_cmd]])
                obs, _, _, _, _ = self.env.step(action)
                
                # 9. Success Check
                dist = np.linalg.norm(obs['object_pos_world'] - obs['goal_pos_world'])
                if dist < 0.07 and obs['object_pos_world'][2] > 0.42 and obs['is_grasped'][0] > 0.5:
                    success = True
                
                # 10. AR Visualization (The "Green Dot" Debugger)
                frame = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
                
                # Project Target (Green)
                cam_params = obs['camera_params']
                uv_target = project_point_to_image(target_pose_world[:3], cam_params)
                if uv_target:
                    cv2.circle(frame, uv_target, 6, (0, 255, 0), -1) # Green Dot = Network Plan
                
                # Project Actual EE (Red)
                uv_ee = project_point_to_image(obs['ee_pose_world'][:3], cam_params)
                if uv_ee:
                    cv2.circle(frame, uv_ee, 4, (0, 0, 255), -1) # Red Dot = Reality
                
                # HUD
                phases = ["REACH", "GRASP", "TRANS", "PLACE", "DONE"]
                cv2.putText(frame, f"Phase: {phases[min(phase, 4)]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"Grip: {gripper_cmd:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                
                writer.write(frame)
                csv_writer.writerow([ep_idx, step, phase, int(success), dist, obs['object_pos_world'][2]])
                
                if success:
                    break
            
            if success: success_count += 1
            
        writer.release()
        csv_file.close()
        self.env.close()
        
        log.info("="*40)
        log.info(f"FINAL RESULTS: {success_count}/{self.cfg.num_episodes} Successes ({(success_count/self.cfg.num_episodes)*100:.1f}%)")
        log.info(f"Video saved to: {video_path}")
        log.info("="*40)

@hydra.main(version_base=None, config_path="./configs", config_name="evaluate_semantic_planner_config")
def main(cfg: DictConfig):
    evaluator = SOTAEvaluator(cfg)
    evaluator.run()

if __name__ == "__main__":
    main()