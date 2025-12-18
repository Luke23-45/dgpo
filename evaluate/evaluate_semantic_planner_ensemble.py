"""
SOTA Evaluation Script for Semantic Planner with Temporal Ensembling (Aggregation).
Based on the verified correct logic of evaluate_semantic_planner_v2.py.

IMPROVEMENTS:
- Temporal Ensembling: Aggregates overlapping action chunks for SOTA smoothness.
- Exponential Weighting: Trusts earlier predictions in the chunk more than later ones.
- Robust Buffer Management: Handles infinite horizon streaming.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque, defaultdict

import cv2
import hydra
import mujoco
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from scipy.spatial.transform import Rotation
from torchvision import transforms
from tqdm import tqdm

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.panda_env import PandaEnv
from models.semantic_planner import SemanticPlanner
from train.train_semantic_planner import SemanticPlannerLightningModule
from utils.ik_solver import IKSolver

# Configure Logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ==============================================================================
# 1. TEMPORAL ENSEMBLER (THE SOTA COMPONENT)
# ==============================================================================

class TemporalEnsembler:
    """
    SOTA Inference Strategy: Temporal Ensembling (Aggregation).
    
    Instead of executing only the first step of a chunk, we aggregate ALL 
    overlapping predictions for a given timestep using weighted averaging.
    
    Math:
        Action_t = Sum( w_i * Prediction_{t-i}[i] ) / Sum( w_i )
        where w_i = exp(-decay * i)
    """
    def __init__(self, horizon: int, decay: float = 0.01):
        """
        Args:
            horizon (int): The chunk size (K) of the model.
            decay (float): Exponential decay rate for weights. 
                           Higher = Trust short-term more. Lower = smoother.
        """
        self.horizon = horizon
        self.decay = decay
        
        # Buffers to store weighted sums
        # Key: Global Timestep (int)
        # Value: np.ndarray (Weighted Action Sum)
        self.pose_buffer = defaultdict(lambda: np.zeros(7, dtype=np.float32))
        self.grip_buffer = defaultdict(lambda: 0.0)
        
        # Buffer to store total weights
        # Key: Global Timestep (int)
        # Value: float (Sum of weights)
        self.weight_buffer = defaultdict(lambda: 0.0)
        
        # Precompute weights for efficiency
        # w[i] corresponds to the weight of the i-th step in a chunk
        self.weights = np.exp(-self.decay * np.arange(self.horizon))

    def reset(self):
        """Clear all buffers."""
        self.pose_buffer.clear()
        self.grip_buffer.clear()
        self.weight_buffer.clear()

    def update(self, pose_chunk: np.ndarray, grip_chunk: np.ndarray, start_step: int):
        """
        Integrate a new predicted chunk into the ensemble.
        
        Args:
            pose_chunk: (K, 7) array [x, y, z, qx, qy, qz, qw]
            grip_chunk: (K, 1) array [logit]
            start_step: The global timestep where this chunk starts (t=0 of the chunk)
        """
        K = min(len(pose_chunk), self.horizon)
        
        for i in range(K):
            target_t = start_step + i
            w = self.weights[i]
            
            # 1. Accumulate Weighted Actions
            # Note: For quaternions, naive averaging works "okay" for small differences,
            # but for SOTA math we should treat them carefully.
            # However, ACT paper standard implementation averages 7D vectors directly 
            # and then normalizes result. We will follow this standard.
            self.pose_buffer[target_t] += pose_chunk[i] * w
            self.grip_buffer[target_t] += grip_chunk[i] * w
            
            # 2. Accumulate Weights
            self.weight_buffer[target_t] += w

    def get_action(self, step: int) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Retrieve the ensembled action for the specified timestep.
        
        Returns:
            (pose, grip) or (None, None) if no data available.
        """
        if self.weight_buffer[step] < 1e-6:
            return None, None
            
        # Normalize by total weight
        W = self.weight_buffer[step]
        avg_pose = self.pose_buffer[step] / W
        avg_grip = self.grip_buffer[step] / W
        
        # Re-normalize quaternion part to ensure it's a valid rotation
        # (Average of unit vectors is not necessarily a unit vector)
        pos = avg_pose[:3]
        quat = avg_pose[3:]
        quat = quat / (np.linalg.norm(quat) + 1e-8)
        
        final_pose = np.concatenate([pos, quat])
        
        # Clean up old history to prevent memory leak (optional but good practice)
        # We can safely delete anything older than 'step'
        if step > 0 and (step - 1) in self.weight_buffer:
            del self.weight_buffer[step - 1]
            del self.pose_buffer[step - 1]
            del self.grip_buffer[step - 1]
            
        return final_pose, avg_grip

# ==============================================================================
# 2. HELPER UTILS (Shared with v2)
# ==============================================================================

class EvaluationLogger:
    def __init__(self, csv_path: Path):
        self.file_handle = open(csv_path, 'w', newline='')
        self.writer = csv.writer(self.file_handle)
        self.headers = [
            "episode_id", "step", "time_sec", "task_phase", "is_holding_object",
            "success_flag", "dist_ee_obj", "dist_obj_goal", "obj_height",
            "target_x", "target_y", "target_z",
            "grip_logit", "commanded_gripper", "joint_vel_norm",
            "ensemble_weight" # New metric for confidence
        ]
        self.writer.writerow(self.headers)

    def log_step(self, data: dict):
        row = []
        for h in self.headers:
            val = data.get(h, 0)
            if isinstance(val, (np.floating, float)):
                val = f"{val:.6f}"
            row.append(val)
        self.writer.writerow(row)
        self.file_handle.flush()

    def close(self):
        self.file_handle.close()

def render_goal_image(env: PandaEnv, ik_solver: IKSolver, obs: dict) -> np.ndarray:
    """
    Render goal image matching Expert 'RETRACT' state (Hovering above goal).
    Dynamically recalculates the robot's target orientation to align with the
    goal object's faces, matching the ScriptedExpert's behavior.
    """
    saved_qpos = env.data.qpos.copy()
    saved_qvel = env.data.qvel.copy()
    saved_ctrl = env.data.ctrl.copy()
    
    # Expert Retract Parameters
    HOVER_HEIGHT = 0.10
    
    # Instantiate temporary expert to use its alignment logic
    from utils.scripted_expert import ScriptedExpert, ObjectProfile
    dummy_expert = ScriptedExpert(ObjectProfile(size=np.zeros(3), grasp_width_normalized=0.0))

    try:
        goal_pos_world = obs['goal_pos_world']
        goal_orn_world = obs['goal_orn_world']

        # 1. Move object to goal POSE (Position + Orientation)
        # FIX: goal_pos_world is at surface level (Z=0.401). Object qpos is center-of-mass.
        # We must lift the object by half-height (0.02 for 4cm cube) to place it ON the table.
        obj_addr = env.model.jnt_qposadr[env.object_joint_id]
        target_obj_pos = goal_pos_world + np.array([0.0, 0.0, 0.02])
        env.data.qpos[obj_addr:obj_addr+3] = target_obj_pos
        
        # Set Orientation: Convert xyzw (SciPy) -> wxyz (MuJoCo)
        goal_orn_wxyz = env._scipy_xyzw_to_mujoco_wxyz(goal_orn_world)
        env.data.qpos[obj_addr+3:obj_addr+7] = goal_orn_wxyz
        
        # 2. Calculate Robot Target Pose (Goal Pos + Hover Z)
        # FIX: Also lift robot by half-height (0.02) so it hovers relative to 
        # the object's center, not the table surface.
        target_pos = goal_pos_world + np.array([0.0, 0.0, HOVER_HEIGHT + 0.02])
        
        # 3. Calculate DYNAMIC Target Orientation
        # The expert aligns with the goal object. We calculate this alignment relative
        # to the goal orientation we just retrieved.
        # FIX: Expert uses [1, 0, 0, 0] (Rot X 180) as base. [0, 1, 0, 0] causes twisted arm.
        seed_downward_quat = np.array([1.0, 0.0, 0.0, 0.0])
        target_quat = dummy_expert._calculate_aligned_orientation(goal_orn_world, seed_downward_quat)
        
        target_pose_7d = np.concatenate([target_pos, target_quat])
        
        # 4. Solve IK for Hover Pose
        # Seed with current qpos or home
        home_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        
        goal_qpos = ik_solver.solve_ik_static(
            target_pose=target_pose_7d, 
            model=env.model, 
            data=env.data, 
            ee_site_id=env.ee_site_id,
            q0=home_qpos 
        )
        
        if goal_qpos is None:
            log.warning("IK failed for goal image generation. Using Home fallback.")
            goal_qpos = home_qpos
        
        # 4. Set Robot State
        env.data.qpos[:7] = goal_qpos
        env.data.ctrl[:7] = goal_qpos
        env.data.qpos[7:9] = 0.04 # Open Gripper
        
        mujoco.mj_forward(env.model, env.data)
        img = env.render()
        return img
    finally:
        env.data.qpos[:] = saved_qpos
        env.data.qvel[:] = saved_qvel
        env.data.ctrl[:] = saved_ctrl
        mujoco.mj_forward(env.model, env.data)

# ==============================================================================
# 3. ENSEMBLE EVALUATOR ENGINE
# ==============================================================================

class SemanticPlannerEvaluatorEnsemble:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load Model
        log.info(f"Loading Checkpoint: {self.cfg.checkpoint_path}")
        pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
            self.cfg.checkpoint_path, map_location=self.device, strict=False
        )
        self.model = pl_module.model.eval().to(self.device)
        self.train_cfg = pl_module.cfg
        
        # 2. Init Environment
        self.env = PandaEnv(
            xml_path=self.cfg.get("xml_path", "envs/panda_pick_place.xml"),
            control_mode='delta',
            render_mode="rgb_array"
        )
        
        # 3. Init IK Solver
        ik_path = self.cfg.get("urdf_path", "urdf/panda_mujoco_kinematics.urdf")
        self.ik_solver = IKSolver(urdf_path=ik_path)
        
        # Apply gains
        self.ik_solver.set_gains(
            kp=cfg.get("ik_kp", 400.0), 
            ki=cfg.get("ik_ki", 0.1), 
            kd=cfg.get("ik_kd", 20.0)
        )
        
        # 4. Control Timing
        SIM_SUBSTEPS = 20
        self.effective_dt = self.env.model.opt.timestep * SIM_SUBSTEPS
        self.max_dq = self.env.ACTION_SCALING_FACTOR / self.effective_dt
        
        # 5. Image Pipeline (MATCHING V2/TRAINING STRICTLY)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # 6. Init Temporal Ensembler
        self.chunk_size = self.model.cfg.chunk_size
        self.ensemble_decay = self.cfg.get("ensemble_decay", 0.01) # Default SOTA decay
        self.ensembler = TemporalEnsembler(horizon=self.chunk_size, decay=self.ensemble_decay)
        
        # Output
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        log.info(f"Initialized Ensemble Evaluator (Horizon={self.chunk_size}, Decay={self.ensemble_decay})")

    def _prepare_batch(self, prev, curr, goal, proprio):
        """Prepare batch for model inference."""
        return {
            "prev_image": self.transform(Image.fromarray(prev)).unsqueeze(0).to(self.device),
            "curr_image": self.transform(Image.fromarray(curr)).unsqueeze(0).to(self.device),
            "goal_image": self.transform(Image.fromarray(goal)).unsqueeze(0).to(self.device),
            "curr_proprio": torch.from_numpy(proprio).float().unsqueeze(0).to(self.device)
        }

    def run_episode(self, ep_id: int, seed: int, vid_writer, logger) -> bool:
        """Run single episode with Temporal Ensembling."""
        self.env.reset(seed=seed)
        self.ik_solver.reset_controller_state()
        self.ensembler.reset()
        
        obs = self.env.get_expert_obs()
        goal_img = render_goal_image(self.env, self.ik_solver, obs)
        prev_img = obs['image_primary'].copy()
        
        success = False
        
        # Initial Ensembling Warmup (Optional)
        # Some implementations run inference T times before moving.
        # Here we start moving immediately but confidence builds up over steps.
        
        for step in range(self.cfg.max_steps):
            # 1. Observation
            curr_img = obs['image_primary']
            proprio = obs['proprio']
            
            # 2. Inference (Full Chunk)
            batch = self._prepare_batch(prev_img, curr_img, goal_img, proprio)
            with torch.no_grad():
                out = self.model(batch)
            
            # 3. Temporal Ensemble Update
            # Get full chunks from CPU
            pose_chunk = out['pose_chunk'][0].cpu().numpy() # (K, 7)
            grip_chunk = out['gripper_chunk'][0].cpu().numpy() # (K, 1)
            
            # Add to ensembler at current global step
            self.ensembler.update(pose_chunk, grip_chunk, start_step=step)
            
            # 4. Get Ensembled Action for NOW
            target_pose, grip_logit = self.ensembler.get_action(step)
            
            # Fallback if ensembler isn't ready (shouldn't happen with k>=1)
            if target_pose is None:
                target_pose = pose_chunk[0]
                grip_logit = grip_chunk[0]
            
            # 5. IK & Execution
            try:
                d_qpos = self.ik_solver.compute_delta_action(
                    target_ee_pose=target_pose,
                    model=self.env.model,
                    data=self.env.data,
                    ee_site_id=self.env.ee_site_id,
                    max_dq=self.max_dq,
                    effective_dt=self.effective_dt
                )
            except Exception:
                d_qpos = np.zeros(7)
                
            action = np.concatenate([d_qpos, [grip_logit.item()]])
            obs, _, _, _, _ = self.env.step(action)
            
            # 6. Check Success
            # Logic: Object near goal, above table, grasped
            obj_pos = obs['object_pos_world']
            goal_pos = obs['goal_pos_world']
            dist = np.linalg.norm(obj_pos - goal_pos)
            
            if dist < 0.05 and obj_pos[2] > 0.41 and obs['is_grasped'][0] > 0.5:
                success = True
            
            # 7. Logging
            logger.log_step({
                "episode_id": ep_id, "step": step, 
                "time_sec": step * self.effective_dt,
                "target_x": target_pose[0], 
                "ensemble_weight": self.ensembler.weight_buffer[step],
                "success_flag": success,
                "dist_obj_goal": dist,
                "grip_logit": grip_logit
            })
            
            # Update history
            prev_img = curr_img.copy()
            
            # Video
            frame = self.env.render()
            vid_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            if success:
                break
                
        return success

    def run(self):
        csv_path = self.output_dir / "eval_ensemble_results.csv"
        video_path = self.output_dir / "eval_ensemble.mp4"
        
        logger = EvaluationLogger(csv_path)
        
        # Setup Video
        dummy = self.env.render()
        h, w, _ = dummy.shape
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        
        successes = []
        try:
            for i in tqdm(range(self.cfg.num_episodes), desc="Ensemble Eval"):
                seed = self.cfg.seed + i
                s = self.run_episode(i, seed, writer, logger)
                successes.append(s)
                
                # Live stats
                sr = np.mean(successes) * 100
                log.info(f"Ep {i} | Success: {s} | SR: {sr:.1f}%")
                
        finally:
            writer.release()
            logger.close()
            log.info(f"Final Success Rate: {np.mean(successes)*100:.2f}%")

@hydra.main(config_path="../config", config_name="train_config", version_base="1.2")
def main(cfg: DictConfig):
    # Override with eval-specific settings if not present
    if "output_dir" not in cfg:
        cfg.output_dir = "outputs/eval_ensemble"
    
    evaluator = SemanticPlannerEvaluatorEnsemble(cfg)
    evaluator.run()

if __name__ == "__main__":
    main()
