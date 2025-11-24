# FILE: scripts/evaluate_ood_robustness.py
# (Phase 3: SOTA OOD & Robustness Stress Testing)

import logging
import sys
import csv
import cv2
import hydra
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
import mujoco
# --- Project Imports ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from envs.panda_env import PandaEnv
from train.train_semantic_planner import SemanticPlannerLightningModule
from utils.ik_solver import IKSolver 

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [OOD-EVAL] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("OODEval")

# ==============================================================================
# 1. ROBUSTNESS CONFIGURATION & DATA STRUCTURES
# ==============================================================================

@dataclass
class EpisodeMetrics:
    episode_id: int
    # Perturbation Parameters
    test_condition: str         # e.g., "ID", "Rotation_45", "Dark_Lighting"
    yaw_angle_deg: float
    light_intensity: float
    friction_coeff: float
    
    # Outcomes
    success: bool
    failure_mode: str           # "NONE", "REACH", "GRASP", "DROP", "TIMEOUT"
    
    # Performance Stats
    steps_taken: int
    min_dist_to_obj: float
    min_dist_to_goal: float
    avg_grip_confidence: float
    max_joint_velocity: float   # Jerkiness metric

class OOD_Scenario_Manager:
    """
    Manages the systematic sweeping of environmental parameters.
    Instead of random sampling, we force specific OOD conditions.
    """
    def __init__(self, env: PandaEnv):
        self.env = env
        
    def apply_condition(self, condition_name: str, seed: int):
        """
        Overrides the environment state to match a specific stress test.
        """
        # 1. Standard Reset first to clear physics state
        self.env.reset(seed=seed)
        
        # Default Parameters (In-Distribution)
        yaw = np.random.uniform(-0.2, 0.2) # Slight random yaw
        light_pos = np.array([0, 0, 2.0])
        friction = 1.0
        
        # 2. Apply Perturbations
        if condition_name == "ID":
            pass # Standard training distribution
            
        elif condition_name == "Rot_Easy":
            yaw = np.radians(45)
            
        elif condition_name == "Rot_Hard":
            # 90 degrees is often a singularity or visual ambiguity
            yaw = np.radians(90) 
            
        elif condition_name == "Rot_Extreme":
            # 135 degrees requires complex wrist planning
            yaw = np.radians(135)
            
        elif condition_name == "Light_Dim":
            # Simulate evening/shadows
            # PandaEnv lighting is controlled via model.light_*
            self.env.model.light_diffuse[self.env.light_id] = np.array([0.3, 0.3, 0.3])
            
        elif condition_name == "Physics_Slippery":
            # Low friction
            friction = 0.1
            self.env.model.geom_friction[self.env.object_geom_id][0] = friction
            
        # 3. Inject Geometry (Force Object Pose)
        # We must modify the physics state directly after reset
        obj_id = self.env.model.jnt_qposadr[self.env.object_joint_id]
        
        # Get current pos (randomly placed by reset)
        curr_pos = self.env.data.qpos[obj_id:obj_id+3].copy()
        
        # Force Orientation (Yaw)
        r = R.from_euler('z', yaw)
        # Convert scipy xyzw to mujoco wxyz
        quat_xyzw = r.as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        
        self.env.data.qpos[obj_id+3:obj_id+7] = quat_wxyz
        
        # Commit changes
        mujoco.mj_forward(self.env.model, self.env.data)
        
        return {
            "yaw": np.degrees(yaw),
            "friction": friction,
            "light": 1.0 if "Light" not in condition_name else 0.3
        }

# ==============================================================================
# 2. EVALUATOR
# ==============================================================================

class OOD_Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        
        # --- Load Model ---
        log.info(f"Loading Checkpoint: {self.cfg.checkpoint_path}")
        self.pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
            self.cfg.checkpoint_path, map_location=self.device
        )
        self.model = self.pl_module.model.eval().to(self.device)
        self.chunk_size = self.pl_module.cfg.model.get("chunk_size", 10)
        
        # --- Init Env ---
        self.env = PandaEnv(
            xml_path=self.cfg.get("xml_path", "envs/panda_pick_place.xml"),
            control_mode='delta',
            render_mode="rgb_array",
            action_scaling_factor=self.cfg.env.get("action_scaling_factor", 0.5),
            enable_domain_randomization=True # Enable so we can manipulate it
        )
        self.scenario_manager = OOD_Scenario_Manager(self.env)
        
        # --- Controller Sync ---
        SIM_SUBSTEPS = 20
        self.effective_dt = self.env.model.opt.timestep * SIM_SUBSTEPS
        self.max_dq = self.env.ACTION_SCALING_FACTOR / self.effective_dt
        self.ik_solver = IKSolver(urdf_path="urdf/panda_mujoco_kinematics.urdf")
        
        # --- Vision ---
        # Configurable Normalization Check
        steps = [transforms.Resize((224, 224), antialias=True), transforms.ToTensor()]
        if self.cfg.get("use_normalization", False):
            steps.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transform = transforms.Compose(steps)
        
        self.prev_image_buffer = None

    def _process_image(self, img_array):
        return self.transform(Image.fromarray(img_array)).unsqueeze(0).to(self.device)

    def run(self):
        csv_path = self.out_dir / "ood_robustness_report.csv"
        
        # Define Test Suite
        # 10 episodes per condition
        CONDITIONS = ["ID", "Rot_Easy", "Rot_Hard", "Rot_Extreme", "Light_Dim", "Physics_Slippery"]
        EPISODES_PER_COND = self.cfg.get("episodes_per_condition", 10)
        
        fieldnames = [field for field in EpisodeMetrics.__annotations__]
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for condition in CONDITIONS:
                log.info(f"--- Starting Condition: {condition} ---")
                
                for i in range(EPISODES_PER_COND):
                    seed = self.cfg.seed + i
                    
                    # 1. Configure Environment for OOD
                    params = self.scenario_manager.apply_condition(condition, seed)
                    
                    # 2. Run Episode
                    metrics = self.run_episode(
                        seed=seed, 
                        condition=condition, 
                        params=params
                    )
                    
                    # 3. Log
                    writer.writerow(asdict(metrics))
                    f.flush() # Ensure data is safe
                    
                    log.info(f"Ep {i} [{condition}]: Success={metrics.success}, FailMode={metrics.failure_mode}")

        log.info(f"OOD Evaluation Complete. Data saved to {csv_path}")

    def run_episode(self, seed, condition, params) -> EpisodeMetrics:
        """
        Executes a single episode using Receding Horizon Control.
        """
        obs = self.env.get_expert_obs() # Get initial observation after override
        
        # Buffers
        curr_img = self._process_image(obs['image_primary'])
        self.prev_image_buffer = curr_img.clone()
        
        if obs['goal_image'] is None:
            # Fallback if env didn't generate goal
            goal_tensor = torch.zeros_like(curr_img)
        else:
            goal_tensor = self._process_image(obs['goal_image'])

        # Metrics State
        min_dist_obj = float('inf')
        min_dist_goal = float('inf')
        grip_confidences = []
        max_j_vel = 0.0
        
        success = False
        failure_mode = "TIMEOUT" # Default if loop finishes
        
        # --- RHC LOOP ---
        for step in range(self.cfg.max_steps):
            # 1. Inference
            curr_img = self._process_image(obs['image_primary'])
            proprio = torch.from_numpy(obs['proprio']).float().unsqueeze(0).to(self.device)
            
            batch = {
                'prev_image': self.prev_image_buffer,
                'curr_image': curr_img,
                'goal_image': goal_tensor,
                'curr_proprio': proprio
            }
            self.prev_image_buffer = curr_img.clone() # Shift history
            
            with torch.no_grad():
                out = self.model(batch)
            
            # 2. RHC Selection (Index 0 for immediate action)
            pose_world = out['pose_chunk'].cpu().numpy()[0, 0]
            grip_logit = out['gripper_chunk'].cpu().numpy()[0, 0].item()
            
            grip_confidences.append(abs(grip_logit))
            
            # 3. Action Execution
            grip_act = -1.0 if grip_logit > 0.0 else 1.0
            
            try:
                dq = self.ik_solver.compute_delta_action(
                    target_ee_pose=pose_world,
                    model=self.env.model,
                    data=self.env.data,
                    ee_site_id=self.env.ee_site_id,
                    joint_qpos_indices=np.arange(7),
                    effective_dt=self.effective_dt,
                    max_dq=self.max_dq
                )
                max_j_vel = max(max_j_vel, np.max(np.abs(dq)))
            except:
                dq = np.zeros(7)
            
            action = np.concatenate([dq, [grip_act]])
            obs, _, terminated, truncated, _ = self.env.step(action)
            
            # 4. Metric Updates
            dist_ee_obj = np.linalg.norm(obs['ee_pose_world'][:3] - obs['object_pos_world'])
            dist_obj_goal = np.linalg.norm(obs['object_pos_world'] - obs['goal_pos_world'])
            is_grasped = obs['is_grasped'][0] > 0.5
            
            min_dist_obj = min(min_dist_obj, dist_ee_obj)
            min_dist_goal = min(min_dist_goal, dist_obj_goal)
            
            # 5. Success Logic
            # Success: Close to goal AND lifted
            if dist_obj_goal < 0.05 and obs['object_pos_world'][2] > 0.45:
                success = True
                failure_mode = "NONE"
                break
            
            if terminated or truncated:
                break
        
        # --- POST-HOC FAILURE CLASSIFICATION ---
        if not success:
            if min_dist_obj > 0.05:
                failure_mode = "REACH" # Never got close to object
            elif min_dist_obj <= 0.05 and min_dist_goal > 0.20:
                # Got close, but object didn't move to goal
                # Did we grasp it?
                if np.max(obs['is_grasped']) > 0.5: 
                    failure_mode = "DROP" # Had it, lost it
                else:
                    failure_mode = "GRASP" # Reached, but couldn't latch
            elif min_dist_goal <= 0.20:
                failure_mode = "DROP" # Moved it part way but failed final check
                
        return EpisodeMetrics(
            episode_id=seed,
            test_condition=condition,
            yaw_angle_deg=params['yaw'],
            light_intensity=params['light'],
            friction_coeff=params['friction'],
            success=success,
            failure_mode=failure_mode,
            steps_taken=step,
            min_dist_to_obj=min_dist_obj,
            min_dist_to_goal=min_dist_goal,
            avg_grip_confidence=np.mean(grip_confidences) if grip_confidences else 0.0,
            max_joint_velocity=max_j_vel
        )

@hydra.main(version_base=None, config_path="../configs", config_name="evaluate_ood_config")
def main(cfg):
    evaluator = OOD_Evaluator(cfg)
    evaluator.run()

if __name__ == "__main__":
    main()