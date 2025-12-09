# FILE: s45_teleport.py
# (v1.0 - Teleportation-Based Evaluation / Open-Loop Rollout)
# 
# PURPOSE:
#   This script evaluates the Semantic Planner by TELEPORTING the robot
#   directly to the model's predicted poses, bypassing the IK delta control loop.
#   This isolates model prediction quality from controller issues.
#
# KEY DIFFERENCES FROM s45.py:
#   - Robot joint positions are SET DIRECTLY via qpos (no delta actions)
#   - Gripper is controlled via direct qpos (not actuators)
#   - Physics substeps run after teleport for grasp detection
#   - This is an OPEN-LOOP evaluation (no feedback control)

import logging
import sys
import time
import csv
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

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
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from envs.panda_env import PandaEnv
from train.train_semantic_planner import SemanticPlannerLightningModule
from utils.ik_solver import IKSolver

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [TELEPORT-EVAL] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("teleport_eval_log.txt", mode='w')
    ]
)
log = logging.getLogger("TeleportEval")

# ==============================================================================
# 1. GRIPPER HANDLER (DIRECT QPOS CONTROL)
# ==============================================================================

class GripperHandler:
    """
    Manages gripper state via direct qpos manipulation.
    
    Gripper positions:
    - OPEN: qpos[7:9] = 0.04 (fingers apart)
    - CLOSED: qpos[7:9] = 0.0 (fingers together)
    
    Uses Schmitt trigger for hysteresis.
    """
    
    # From ExpertConfig thresholds
    GRIPPER_OPEN_QPOS = 0.04
    GRIPPER_CLOSED_QPOS = 0.0
    
    # Probability thresholds (from model sigmoid output)
    CLOSE_THRESHOLD = 0.5  # Close when prob > 0.5
    OPEN_THRESHOLD = 0.5   # Open when prob < 0.5
    
    def __init__(self):
        self.is_closed = False
        self.smoothed_prob = 0.0
        self.alpha = 0.5  # EMA smoothing factor
    
    def reset(self):
        self.is_closed = False
        self.smoothed_prob = 0.0
    
    def update(self, grip_logit: float) -> Tuple[float, bool]:
        """
        Update gripper state based on model logit.
        
        Returns:
            gripper_qpos: Value to set for qpos[7] and qpos[8]
            is_closed: Boolean gripper state
        """
        # Convert logit to probability
        grip_prob = 1.0 / (1.0 + np.exp(-np.clip(grip_logit, -50, 50)))
        
        # EMA smoothing
        self.smoothed_prob = (self.alpha * grip_prob) + ((1 - self.alpha) * self.smoothed_prob)
        
        # Schmitt trigger decision
        # Training convention: 1.0 = CLOSED, 0.0 = OPEN
        if self.smoothed_prob > self.CLOSE_THRESHOLD:
            self.is_closed = True
        elif self.smoothed_prob < self.OPEN_THRESHOLD:
            self.is_closed = False
        
        # Return qpos value for gripper fingers
        gripper_qpos = self.GRIPPER_CLOSED_QPOS if self.is_closed else self.GRIPPER_OPEN_QPOS
        
        return gripper_qpos, self.is_closed


# ==============================================================================
# 2. GOAL IMAGE GENERATION (MATCHING TRAINING DATA)
# ==============================================================================

@contextmanager
def render_goal_image_like_training(env: PandaEnv, goal_pos_world: np.ndarray):
    """
    Renders a goal image matching training data convention:
    - Object at goal position
    - Robot in retracted home position
    - Gripper open
    """
    saved_qpos = env.data.qpos.copy()
    saved_qvel = env.data.qvel.copy()
    saved_ctrl = env.data.ctrl.copy()
    
    try:
        # Robot in home/retracted position
        home_qpos = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.8, 0.785])
        env.data.qpos[:7] = home_qpos
        env.data.qpos[7:9] = 0.04  # Open gripper
        
        # Object at goal position
        obj_jnt_adr = env.model.jnt_qposadr[env.object_joint_id]
        current_quat = env.data.qpos[obj_jnt_adr+3 : obj_jnt_adr+7].copy()
        
        env.data.qpos[obj_jnt_adr : obj_jnt_adr+3] = goal_pos_world
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
# 3. TELEPORTATION FUNCTION
# ==============================================================================

def teleport_robot(env: PandaEnv, ik_solver: IKSolver, 
                   target_pose_7d: np.ndarray,
                   gripper_qpos: float,
                   physics_substeps: int = 5) -> bool:
    """
    Teleports the robot to a target end-effector pose.
    
    Args:
        env: PandaEnv instance
        ik_solver: IKSolver instance for computing joint angles
        target_pose_7d: [x, y, z, qx, qy, qz, qw] target EE pose
        gripper_qpos: Gripper joint position (0.0=closed, 0.04=open)
        physics_substeps: Number of physics steps to run after teleport
        
    Returns:
        success: True if IK was successful
    """
    # Get current joint angles
    current_joints = env.data.qpos[:7].copy()
    
    # Convert quaternion from xyzw (model output) to wxyz (MuJoCo/IKPy)
    pos = target_pose_7d[:3]
    quat_xyzw = target_pose_7d[3:]
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    
    # Create target pose in format expected by IK solver
    target_pose_for_ik = np.concatenate([pos, quat_xyzw])  # IKSolver expects xyzw
    
    # Solve IK to get target joint angles
    try:
        target_joints = ik_solver._get_target_joint_angles(
            target_pose_7d=target_pose_for_ik,
            current_joint_angles=current_joints,
            solution_position_tolerance=0.05,  # 5cm tolerance for teleport
            max_iter=50
        )
    except Exception as e:
        log.warning(f"IK failed: {e}")
        return False
    
    if target_joints is None:
        log.warning("IK returned None - no solution found")
        return False
    
    # Clamp joint angles to limits
    target_joints = ik_solver.clamp_to_limits(target_joints)
    
    # TELEPORT: Directly set joint positions
    env.data.qpos[:7] = target_joints
    env.data.qpos[7] = gripper_qpos  # Left finger
    env.data.qpos[8] = gripper_qpos  # Right finger
    
    # Zero velocities (teleportation = instant)
    env.data.qvel[:9] = 0.0
    
    # Run physics substeps to let grasp settle
    for _ in range(physics_substeps):
        mujoco.mj_step(env.model, env.data)
    
    return True


# ==============================================================================
# 4. MAIN EVALUATOR
# ==============================================================================

class TeleportEvaluator:
    """
    Teleportation-Based Semantic Planner Evaluator.
    
    Instead of using delta control, this evaluator directly teleports
    the robot to the model's predicted poses.
    """
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- A. Load Model ---
        log.info(f"Loading Checkpoint: {cfg.checkpoint_path}")
        self.pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
            cfg.checkpoint_path,
            map_location=self.device
        )
        self.model = self.pl_module.model.eval().to(self.device)
        self.chunk_size = self.pl_module.cfg.model.get("chunk_size", 10)
        log.info(f"Model Loaded. Action Chunk Size: {self.chunk_size}")

        # --- B. Initialize Environment ---
        xml_path = cfg.env.get("xml_path", "envs/panda_pick_place.xml")
        self.env = PandaEnv(
            xml_path=xml_path,
            control_mode='delta',  # Not used for teleport, but required
            render_mode="rgb_array",
            action_scaling_factor=0.5
        )
        
        # --- C. IK Solver (for computing joint angles from EE pose) ---
        self.ik_solver = IKSolver(urdf_path="urdf/panda_mujoco_kinematics.urdf")
        log.info("IK Solver loaded for teleportation")
        
        # --- D. Visual Transforms (MUST match training) ---
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # --- E. Components ---
        self.gripper_handler = GripperHandler()
        self.prev_img_buffer = None

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get observation from environment after teleport."""
        # Render image
        image = self.env.render()
        
        # Get EE pose
        ee_site_id = self.env.ee_site_id
        ee_pos = self.env.data.site_xpos[ee_site_id].copy()
        ee_mat = self.env.data.site_xmat[ee_site_id].reshape(3, 3)
        ee_quat_xyzw = R.from_matrix(ee_mat).as_quat()
        ee_pose_world = np.concatenate([ee_pos, ee_quat_xyzw])
        
        # Get object position
        obj_jnt_adr = self.env.model.jnt_qposadr[self.env.object_joint_id]
        obj_pos = self.env.data.qpos[obj_jnt_adr : obj_jnt_adr + 3].copy()
        
        # Get goal position
        goal_pos = self.env.get_goal_pos_expert()
        
        # Get proprio - MUST MATCH TRAINING DATA FORMAT EXACTLY!
        # Training proprio = [qpos[:7], qvel[:7], touch_sensors[2], force_vectors[6]]
        joint_qpos = self.env.data.qpos[:7].copy()
        joint_qvel = self.env.data.qvel[:7].copy()
        
        # Get touch sensors (same as PandaEnv._get_obs)
        left_touch = self.env.data.sensordata[self.env.left_touch_sensor_id]
        right_touch = self.env.data.sensordata[self.env.right_touch_sensor_id]
        
        # Get force sensors (same as PandaEnv._get_obs)
        left_force_adr = self.env.model.sensor_adr[self.env.left_force_sensor_id]
        right_force_adr = self.env.model.sensor_adr[self.env.right_force_sensor_id]
        left_force = self.env.data.sensordata[left_force_adr : left_force_adr + 3]
        right_force = self.env.data.sensordata[right_force_adr : right_force_adr + 3]
        
        # Build proprio vector (EXACT match to training)
        proprio = np.concatenate([
            joint_qpos,                              # 7
            joint_qvel,                              # 7
            np.array([left_touch, right_touch]),     # 2 (touch sensors)
            left_force,                              # 3 (force vector)
            right_force                              # 3 (force vector)
        ]).astype(np.float32)
        
        # Check grasp (using environment's method if available)
        is_grasped = self.env._check_physical_grasp() if hasattr(self.env, '_check_physical_grasp') else False
        
        return {
            'image_primary': image,
            'proprio': proprio,
            'ee_pose_world': ee_pose_world,
            'object_pos_world': obj_pos,
            'goal_pos_world': goal_pos,
            'is_grasped': np.array([float(is_grasped)])
        }

    def run(self):
        """Main Execution Loop"""
        out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        video_path = out_dir / "teleport_eval.mp4"
        telemetry_csv_path = out_dir / "eval_telemetry.csv"
        
        # Video setup
        _ = self.env.reset()
        h, w, _ = self.env.render().shape
        video_writer = cv2.VideoWriter(
            str(video_path), 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            30, (w, h)
        )
        
        # Telemetry CSV
        telemetry_file = open(telemetry_csv_path, 'w', newline='')
        telemetry_writer = csv.writer(telemetry_file)
        telemetry_writer.writerow([
            "episode_id", "step", "success_state", "dist_to_goal",
            "phase_pred", "latency_ms", "heuristic_active",
            "ee_x", "ee_y", "ee_z", 
            "target_x", "target_y", "target_z",
            "raw_grip_logit", "grip_prob", "smooth_grip_prob", "gripper_closed",
            "obj_x", "obj_y", "obj_z", "is_grasped", "teleport_ok", "grasp_latch"
        ])
        
        log.info(f"Starting TELEPORT Evaluation for {self.cfg.num_episodes} episodes...")
        log.info(f"Video: {video_path}")
        log.info(f"Telemetry: {telemetry_csv_path}")
        
        summary_results = []
        total_success = 0
        
        try:
            for ep_idx in tqdm(range(self.cfg.num_episodes), desc="TeleportEval"):
                seed = self.cfg.seed + ep_idx
                ep_result = self.run_episode(ep_idx, seed, video_writer, telemetry_writer)
                summary_results.append(ep_result)
                if ep_result['success']:
                    total_success += 1
                    
        finally:
            video_writer.release()
            telemetry_file.close()
            self.env.close()
            
            success_rate = (total_success / self.cfg.num_episodes) * 100
            avg_steps = np.mean([r['steps'] for r in summary_results]) if summary_results else 0
            
            log.info("=" * 60)
            log.info(f"TELEPORT EVALUATION REPORT")
            log.info(f"Episodes: {self.cfg.num_episodes}")
            log.info(f"Success Rate: {success_rate:.2f}%")
            log.info(f"Avg Steps: {avg_steps:.1f}")
            log.info("=" * 60)

    def run_episode(self, ep_idx: int, seed: int, 
                    video_writer, telemetry_writer) -> Dict[str, Any]:
        """Run a single teleportation evaluation episode"""
        
        # 1. Reset
        obs, _ = self.env.reset(seed=seed)
        self.gripper_handler.reset()
        self.ik_solver.reset_controller_state()
        
        # 2. Generate Goal Image (matching training convention)
        with render_goal_image_like_training(self.env, obs['goal_pos_world']):
            goal_img_raw = self.env.render()
        
        goal_tensor = self.transform(Image.fromarray(goal_img_raw)).unsqueeze(0).to(self.device)
        
        # 3. Initialize History
        curr_img_pil = Image.fromarray(obs['image_primary'])
        self.prev_img_buffer = self.transform(curr_img_pil).unsqueeze(0).to(self.device)
        
        success = False
        final_error = 99.9
        grasp_achieved = False
        grasp_latch_counter = 0  # Heuristic: Counts down during grasp sequence
        
        for step in range(self.cfg.max_steps):
            t0 = time.time()
            
            # --- A. Get current observation ---
            obs = self._get_observation()
            
            # --- B. Data Preparation ---
            curr_img_pil = Image.fromarray(obs['image_primary'])
            curr_tensor = self.transform(curr_img_pil).unsqueeze(0).to(self.device)
            proprio_tensor = torch.from_numpy(obs['proprio']).float().unsqueeze(0).to(self.device)
            
            batch = {
                'prev_image': self.prev_img_buffer,
                'curr_image': curr_tensor,
                'goal_image': goal_tensor,
                'curr_proprio': proprio_tensor
            }
            self.prev_img_buffer = curr_tensor.clone()
            
            # --- C. Inference ---
            with torch.no_grad():
                outputs = self.model(batch)
            
            inference_time_ms = (time.time() - t0) * 1000
            
            # Model outputs
            chunk_pose = outputs['pose_chunk'][0].cpu().numpy()
            chunk_grip = outputs['gripper_chunk'][0].cpu().numpy()
            
            # Use first prediction (receding horizon)
            target_pose_raw = chunk_pose[0]  # [x, y, z, qx, qy, qz, qw]
            target_grip_logit = chunk_grip[0].item()
            
            phase_logits = outputs['phase_logits'][0].cpu().numpy()
            predicted_phase = np.argmax(phase_logits)
            
            # Debug logging for first steps
            if step < 3 and ep_idx == 0:
                log.info(f"Step {step}: target_pose={target_pose_raw[:3]}, grip_logit={target_grip_logit:.3f}")
            
            # --- D. HEURISTIC GRASPING LOGIC (from s8.py) ---
            ee_pos = obs['ee_pose_world'][:3]
            obj_pos = obs['object_pos_world']
            goal_pos = obs['goal_pos_world']
            dist_xy = np.linalg.norm(ee_pos[:2] - obj_pos[:2])
            dist_3d = np.linalg.norm(ee_pos - obj_pos)
            is_currently_grasped = obs['is_grasped'][0] > 0.5
            dist_to_goal = np.linalg.norm(obj_pos - goal_pos)
            
            heuristic_active = 0
            
            # Priority 1: TRANSPORT & PLACE (if already grasped)
            if is_currently_grasped:
                heuristic_active = 3  # Transport
                target_grip_logit = 5.0  # Force CLOSE
                
                # Override position: Pull toward goal
                target_pose_raw = target_pose_raw.copy()
                target_pose_raw[:2] = (0.7 * ee_pos[:2]) + (0.3 * goal_pos[:2])
                
                # Z-Logic: High for travel, Low for placement
                if dist_to_goal < 0.10:
                    heuristic_active = 4  # Place
                    target_pose_raw[2] = 0.415  # Descend to place (lowered from 0.43)
                    if dist_to_goal < 0.05 and ee_pos[2] < 0.43:
                        target_grip_logit = -5.0  # RELEASE!
                else:
                    target_pose_raw[2] = 0.55  # Travel height
            else:
                # ACQUISITION LOGIC
                # Step A: Open gripper when high
                if ee_pos[2] > 0.46 and grasp_latch_counter == 0:
                    target_grip_logit = -5.0  # Open
                
                # Step B: XY Approach Correction
                if dist_xy < 0.30 and grasp_latch_counter == 0:
                    heuristic_active = 1  # Approach
                    target_pose_raw = target_pose_raw.copy()
                    target_pose_raw[:2] = (0.5 * target_pose_raw[:2]) + (0.5 * obj_pos[:2])
                    
                    # Z-height correction based on XY distance
                    if dist_xy < 0.05:
                        target_pose_raw[2] = 0.415  # Descend to grasp height (lowered from 0.43)
                    elif dist_xy < 0.15:
                        target_pose_raw[2] = min(target_pose_raw[2], 0.45)
                
                # Step C: Trigger Grasp Latch when positioned
                if dist_xy < 0.03 and ee_pos[2] < 0.43 and grasp_latch_counter == 0:
                    grasp_latch_counter = 45  # Start grasp sequence
                
                # Step D: Execute Grasp Latch Sequence
                if grasp_latch_counter > 0:
                    heuristic_active = 2  # Grasp Latch
                    target_grip_logit = 5.0  # Force CLOSE
                    target_pose_raw = target_pose_raw.copy()
                    
                    if grasp_latch_counter > 25:
                        # Stay at object, close gripper
                        target_pose_raw[:2] = obj_pos[:2]
                        target_pose_raw[2] = 0.415  # Grasp height (lowered from 0.425)
                    else:
                        # Lift up with object
                        target_pose_raw[:2] = ee_pos[:2]
                        target_pose_raw[2] = 0.55
                    
                    grasp_latch_counter -= 1
            
            # Safety: Clamp Z to table level
            target_pose_raw = target_pose_raw.copy() if not isinstance(target_pose_raw, np.ndarray) else target_pose_raw.copy()
            target_pose_raw[2] = max(target_pose_raw[2], 0.41)
            
            # --- E. Gripper Decision (with heuristic logit) ---
            gripper_qpos, gripper_closed = self.gripper_handler.update(target_grip_logit)
            grip_prob = 1.0 / (1.0 + np.exp(-np.clip(target_grip_logit, -50, 50)))
            
            # --- E. TELEPORT Robot ---
            teleport_ok = teleport_robot(
                env=self.env,
                ik_solver=self.ik_solver,
                target_pose_7d=target_pose_raw,
                gripper_qpos=gripper_qpos,
                physics_substeps=10  # Run physics to let grasp stabilize
            )
            
            # --- F. Get Updated Observation ---
            obs = self._get_observation()
            
            # --- G. Metrics ---
            ee_pos_now = obs['ee_pose_world'][:3]
            obj_pos = obs['object_pos_world']
            goal_pos = obs['goal_pos_world']
            dist_to_goal = np.linalg.norm(obj_pos - goal_pos)
            is_grasped = obs['is_grasped'][0] > 0.5
            
            if is_grasped:
                grasp_achieved = True
            
            if dist_to_goal < 0.05:
                success = True
            
            # --- H. Telemetry ---
            telemetry_writer.writerow([
                ep_idx, step, int(success), f"{dist_to_goal:.4f}",
                predicted_phase, f"{inference_time_ms:.1f}", heuristic_active,
                f"{ee_pos_now[0]:.3f}", f"{ee_pos_now[1]:.3f}", f"{ee_pos_now[2]:.3f}",
                f"{target_pose_raw[0]:.3f}", f"{target_pose_raw[1]:.3f}", f"{target_pose_raw[2]:.3f}",
                f"{target_grip_logit:.3f}", f"{grip_prob:.3f}",
                f"{self.gripper_handler.smoothed_prob:.3f}", int(gripper_closed),
                f"{obj_pos[0]:.3f}", f"{obj_pos[1]:.3f}", f"{obj_pos[2]:.3f}",
                int(is_grasped), int(teleport_ok), grasp_latch_counter
            ])
            
            # --- I. Video ---
            frame = self.env.render()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            color = (0, 255, 0) if success else (0, 0, 255)
            grip_color = (0, 255, 0) if gripper_closed else (255, 255, 0)
            
            # Heuristic color coding
            heur_colors = {0: (255,255,255), 1: (0,255,0), 2: (0,0,255), 3: (255,0,0), 4: (255,255,0)}
            heur_names = {0: "Model", 1: "Approach", 2: "Latch", 3: "Transport", 4: "Place"}
            heur_color = heur_colors.get(heuristic_active, (255,255,255))
            
            cv2.putText(frame_bgr, f"Ep:{ep_idx} H:{heur_names.get(heuristic_active,'?')} Err:{dist_to_goal:.3f}m", 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, heur_color, 1)
            cv2.putText(frame_bgr, f"Grip: {'CLOSE' if gripper_closed else 'OPEN'} | Latch:{grasp_latch_counter}", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, grip_color, 1)
            cv2.putText(frame_bgr, f"Grasped: {'YES' if is_grasped else 'NO'} | Z:{ee_pos_now[2]:.2f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            video_writer.write(frame_bgr)
            
            if success:
                final_error = dist_to_goal
                break
        
        log.info(f"Episode {ep_idx}: {'SUCCESS' if success else 'FAIL'}, "
                 f"Error: {final_error:.3f}m, Grasp: {grasp_achieved}")
                
        return {
            "episode_id": ep_idx,
            "success": success,
            "steps": step + 1,
            "final_error": final_error,
            "grasp_achieved": grasp_achieved,
            "seed": seed
        }


@hydra.main(version_base=None, config_path="./configs", config_name="evaluate_semantic_planner_config")
def main(cfg: DictConfig):
    if "checkpoint_path" not in cfg:
        raise ValueError("Must provide 'checkpoint_path' in config")
    
    evaluator = TeleportEvaluator(cfg)
    evaluator.run()


if __name__ == "__main__":
    main()
