# FILE: evaluate_residual_policy.py
# Evaluation script for Residual Policy (BC + RL corrections)

import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

import cv2
import hydra
import mujoco
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms
from scipy.spatial.transform import Rotation as R

# --- Project Imports ---
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from envs.panda_env import PandaEnv
from models.residual_policy import ResidualPolicy, ResidualPolicyConfig, create_residual_policy
from utils.ik_solver import IKSolver
from utils.rl_utils import RunningMeanStd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvalResidual")


class ResidualPolicyEvaluator:
    """Evaluates trained residual policy in closed-loop."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load policy
        if cfg.get("residual_checkpoint"):
            # Load from RL-trained checkpoint
            logger.info(f"Loading RL-trained residual policy: {cfg.residual_checkpoint}")
            self.policy = create_residual_policy(cfg.bc_checkpoint, self.device)
            self.policy = create_residual_policy(cfg.bc_checkpoint, self.device)
            ckpt = torch.load(cfg.residual_checkpoint, map_location=self.device)
            self.policy.load_state_dict(ckpt['policy_state_dict'])
            
            # Load normalization stats
            self.proprio_normalizer = RunningMeanStd(shape=(self.policy.proprio_dim,))
            if 'proprio_normalizer_mean' in ckpt:
                self.proprio_normalizer.mean = ckpt['proprio_normalizer_mean']
                self.proprio_normalizer.var = ckpt['proprio_normalizer_var']
                self.proprio_normalizer.count = ckpt['proprio_normalizer_count']
                logger.info("Loaded proprioception normalization stats from checkpoint")
        else:
            # Just use BC + untrained residual (for baseline comparison)
            logger.info(f"Loading BC policy with untrained residual: {cfg.bc_checkpoint}")
            self.policy = create_residual_policy(cfg.bc_checkpoint, self.device)
            # Initialize empty normalizer (won't be used effectively or will be identity if count is small)
            # For baseline with untrained residual, maybe we shouldn't normalize? 
            # Or assume identity. But better to have it.
            self.proprio_normalizer = RunningMeanStd(shape=(self.policy.proprio_dim,))
        
        self.policy.eval()
        
        # Environment
        self.env = PandaEnv(
            xml_path=cfg.env.get("xml_path", "envs/panda_pick_place.xml"),
            control_mode='delta',
            render_mode="rgb_array"
        )
        
        # IK Solver
        self.ik_solver = IKSolver(urdf_path="urdf/panda_mujoco_kinematics.urdf")
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def _get_proprio_exact(self) -> np.ndarray:
        joint_qpos = self.env.data.qpos[:7].copy()
        joint_qvel = self.env.data.qvel[:7].copy()
        
        left_touch = self.env.data.sensordata[self.env.left_touch_sensor_id]
        right_touch = self.env.data.sensordata[self.env.right_touch_sensor_id]
        
        left_force_adr = self.env.model.sensor_adr[self.env.left_force_sensor_id]
        right_force_adr = self.env.model.sensor_adr[self.env.right_force_sensor_id]
        left_force = self.env.data.sensordata[left_force_adr : left_force_adr + 3]
        right_force = self.env.data.sensordata[right_force_adr : right_force_adr + 3]
        
        return np.concatenate([
            joint_qpos, joint_qvel,
            np.array([left_touch, right_touch]),
            left_force, right_force
        ]).astype(np.float32)
    
    def _render_goal_image(self, goal_pos: np.ndarray) -> np.ndarray:
        saved_qpos = self.env.data.qpos.copy()
        saved_qvel = self.env.data.qvel.copy()
        
        obj_jnt_adr = self.env.model.jnt_qposadr[self.env.object_joint_id]
        self.env.data.qpos[obj_jnt_adr : obj_jnt_adr + 3] = goal_pos
        self.env.data.qpos[:7] = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        self.env.data.qpos[7:9] = 0.04
        self.env.data.qvel[:] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)
        
        goal_image = self.env.render()
        
        self.env.data.qpos[:] = saved_qpos
        self.env.data.qvel[:] = saved_qvel
        mujoco.mj_forward(self.env.model, self.env.data)
        
        return goal_image
    
    def run_episode(self, seed: int) -> Dict[str, Any]:
        obs, _ = self.env.reset(seed=seed)
        self.ik_solver.reset_controller_state()
        
        # Get goal
        goal_pos = self.env.get_goal_pos_expert()
        goal_img = self._render_goal_image(goal_pos)
        goal_tensor = self.transform(Image.fromarray(goal_img)).unsqueeze(0).to(self.device)
        
        # Initialize
        curr_img = self.env.render()
        prev_tensor = self.transform(Image.fromarray(curr_img)).unsqueeze(0).to(self.device)
        
        success = False
        frames = []
        
        for step in range(self.cfg.max_steps):
            curr_img = self.env.render()
            curr_tensor = self.transform(Image.fromarray(curr_img)).unsqueeze(0).to(self.device)
            
            proprio_raw = self._get_proprio_exact()
            proprio_tensor = torch.from_numpy(proprio_raw).float().unsqueeze(0).to(self.device)
            
            # Normalize for residual net
            proprio_norm = self.proprio_normalizer.normalize(proprio_raw)
            proprio_norm_tensor = torch.from_numpy(proprio_norm).float().unsqueeze(0).to(self.device)
            
            batch = {
                'prev_image': prev_tensor,
                'curr_image': curr_tensor,
                'goal_image': goal_tensor,
                'prev_image': prev_tensor,
                'curr_image': curr_tensor,
                'goal_image': goal_tensor,
                'curr_proprio': proprio_tensor,
                'proprio_norm': proprio_norm_tensor
            }
            
            with torch.no_grad():
                outputs = self.policy.forward(batch, deterministic=True)
            
            pose = outputs['pose_chunk'][0, 0].cpu().numpy()
            grip_logit = outputs['gripper_chunk'][0, 0, 0].cpu().item()
            
            # Execute
            gripper_qpos = 0.0 if grip_logit > 0 else 0.04
            
            try:
                target_joints = self.ik_solver._get_target_joint_angles(
                    target_pose_7d=pose,
                    current_joint_angles=self.env.data.qpos[:7],
                    max_iter=30
                )
                if target_joints is not None:
                    delta = np.clip(target_joints - self.env.data.qpos[:7], -0.1, 0.1)
                    self.env.step(np.concatenate([delta, [gripper_qpos]]))
            except Exception:
                pass
            
            prev_tensor = curr_tensor.clone()
            
            # Check success
            obj_jnt_adr = self.env.model.jnt_qposadr[self.env.object_joint_id]
            obj_pos = self.env.data.qpos[obj_jnt_adr : obj_jnt_adr + 3]
            dist = np.linalg.norm(obj_pos - goal_pos)
            
            if dist < 0.05:
                success = True
                break
            
            # Save frame
            frame = self.env.render()
            frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        return {
            'success': success,
            'steps': step + 1,
            'final_dist': dist,
            'frames': frames
        }
    
    def run(self):
        out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        
        successes = 0
        results = []
        
        for ep in range(self.cfg.num_episodes):
            result = self.run_episode(self.cfg.seed + ep)
            results.append(result)
            if result['success']:
                successes += 1
            
            logger.info(f"Episode {ep}: {'SUCCESS' if result['success'] else 'FAIL'} "
                       f"(dist={result['final_dist']:.3f}, steps={result['steps']})")
        
        success_rate = (successes / self.cfg.num_episodes) * 100
        logger.info(f"Success Rate: {success_rate:.1f}% ({successes}/{self.cfg.num_episodes})")
        
        # Save video of last episode
        if results[-1]['frames']:
            h, w = results[-1]['frames'][0].shape[:2]
            video_path = out_dir / "residual_eval.mp4"
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
            for f in results[-1]['frames']:
                writer.write(f)
            writer.release()
            logger.info(f"Saved video: {video_path}")


@hydra.main(version_base=None, config_path="./configs", config_name="eval_residual_config")
def main(cfg: DictConfig):
    evaluator = ResidualPolicyEvaluator(cfg)
    evaluator.run()


if __name__ == "__main__":
    main()
