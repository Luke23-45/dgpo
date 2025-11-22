"""
Gated Residual Expert Aggregation (GREA) - Online DAgger Trainer.

Orchestrates the interactive learning loop:
1. Rollout with Student Policy.
2. Gate Actions via Scripted Expert (Safety Filter).
3. Aggregate 'Correction' Data.
4. Online Fine-Tuning (Residual Updates).

Usage:
    python train_online_dagger.py
"""

import copy
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import collections

# Project Imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from envs.panda_env import PandaEnv
from models.semantic_planner import SemanticPlanner
from train.train_semantic_planner import SemanticPlannerLightningModule
from utils.ik_solver import IKSolver
from utils.scripted_expert import ScriptedExpert, ObjectProfile, ExpertConfig
from utils.online_buffer import SlidingWindowBuffer, OnlineReplayBuffer
from utils.semantic_planner_dataset_post import SemanticPlannerDataset
from models.history_adapter import ResidualHistoryAdapter

log = logging.getLogger("OnlineDAgger")
logging.basicConfig(level=logging.INFO)

class DAggerOrchestrator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- 1. Initialize Models (Student & Teacher) ---
        log.info(f"Loading Student Model: {cfg.checkpoint_path}")
        self.pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
            cfg.checkpoint_path, map_location=self.device
        )
        
        # Extract Base Planner and Wrap with History Adapter
        self.base_planner = self.pl_module.model.eval().to(self.device)
        self.student = ResidualHistoryAdapter(
            base_planner=self.base_planner, 
            history_len=cfg.model.get("history_horizon", 10)
        ).to(self.device)
        
        # Optimizer: Only train the Adapter!
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=cfg.dagger.learning_rate
        )

        # Teacher (Scripted Expert)
        obj_profile = ObjectProfile(size=np.array([0.04, 0.04, 0.04]), grasp_width_normalized=0.6)
        expert_cfg = ExpertConfig(failure_timeout_steps=1000) 
        self.teacher = ScriptedExpert(obj_profile, expert_cfg)
        
        # --- 2. Initialize Environment ---
        self.env = PandaEnv(
            xml_path=cfg.get("xml_path", "envs/panda_pick_place.xml"),
            control_mode='delta',
            render_mode="rgb_array"
        )
        self.ik_solver = IKSolver(urdf_path=cfg.get("urdf_path", "urdf/panda_mujoco_kinematics.urdf"))
        
        # --- 3. Data Infrastructure ---
        self.history_len = cfg.model.get("history_horizon", 10)
        
        # Online Buffer (Stores Corrections)
        self.online_buffer = OnlineReplayBuffer(capacity=cfg.dagger.buffer_capacity)
        
        # Inference Buffer (Stores current episode history)
        self.window = SlidingWindowBuffer(
            horizon=self.history_len,
            proprio_dim=self.pl_module.cfg.model.proprio_dim,
            action_dim=8
        )
        
        # Offline Data Loader (For Replay Mixing)
        log.info("Loading Offline Dataset for Replay Mixing...")
        self.offline_dataset = SemanticPlannerDataset(
            dataset_path=cfg.dataset.train_path,
            history_horizon=self.history_len,
            use_aug=False
        )
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor()
        ])

        # Tuned Thresholds
        self.SAFE_DIST_THRESH = 0.03  # 3cm tolerance
        self.STAGNATION_VEL = 0.002   # 2mm/s
        self.STAGNATION_STEPS = 15
        self.last_executed_action = np.zeros(8) # Track previous action for history
        
        # Ensure checkpoints dir exists
        os.makedirs("checkpoints", exist_ok=True)

    def _get_student_action(self, phase: int, goal_tensor: torch.Tensor) -> Tuple[np.ndarray, float]:
        """
        Forward pass of the Student Model (Base + Adapter).
        Correctly slices history for Base (Current Frame) vs Adapter (Sequence).
        """
        # Get History from Window (1, T, ...)
        hist = self.window.get_history()
        
        img_hist = hist['initial_image'].to(self.device) # (1, T, C, H, W)
        prop_hist = hist['proprio_hist'].to(self.device) # (1, T, D)
        act_hist = hist['action_hist'].to(self.device)   # (1, T, D)
        
        batch = {
            # Base Planner inputs (Last Frame)
            'initial_image': img_hist[:, -1], 
            'goal_image': goal_tensor,
            'task_phase': torch.tensor([phase], device=self.device),
            'current_proprio': prop_hist[:, -1],
            
            # Adapter inputs (History Sequence)
            'proprio_hist': prop_hist,
            'action_hist': act_hist
        }
        
        with torch.no_grad():
            out = self.student(batch)
            
        pose = out['pose'].squeeze().cpu().numpy()
        grip_logit = out['gripper_logit'].item()
        return pose, grip_logit

    def _transform_world_to_base(self, pose_world: np.ndarray) -> np.ndarray:
        """Project World Frame prediction to Robot Base Frame for IK."""
        from scipy.spatial.transform import Rotation as R
        base_pos, base_quat = self.env.get_base_pose()
        R_bw = R.from_quat(base_quat).inv()
        
        pos_base = R_bw.apply(pose_world[:3] - base_pos)
        rot_base = R_bw * R.from_quat(pose_world[3:])
        return np.concatenate([pos_base, rot_base.as_quat()])


    def run(self):
        """Main DAgger Loop: Rollout -> Gate -> Aggregate -> Train"""
        log.info("Starting Online DAgger Loop...")
        
        sim_dt = self.env.model.opt.timestep * 20
        max_dq = self.env.ACTION_SCALING_FACTOR / sim_dt
        
        vel_history = collections.deque(maxlen=self.STAGNATION_STEPS)
        
        for epoch in range(self.cfg.dagger.num_epochs):
            # --- 1. Setup Episode ---
            seed = self.cfg.seed + epoch
            self.env.reset(seed=seed)
            self.teacher.reset()
            self.window.reset()
            self.last_executed_action = np.zeros(8)
            
            # Get Expert Obs
            obs = self.env.get_expert_obs()
            
            # Prepare Goal
            # Handle goal image resize via Transform (PIL)
            goal_img_np = obs['goal_image'] if obs['goal_image'] is not None else np.zeros((256, 256, 3), dtype=np.uint8)
            goal_tensor = self.transform(Image.fromarray(goal_img_np)).unsqueeze(0).to(self.device)
            
            done = False
            episode_steps = 0
            correction_count = 0
            
            pbar = tqdm(total=self.cfg.dagger.max_steps_per_epoch, desc=f"Epoch {epoch}")
            
            while not done and episode_steps < self.cfg.dagger.max_steps_per_epoch:
                # --- 2. Update History ---
                img_np = obs['image_primary']
                
                # [CRITICAL FIX] Resize 256x256 (Env) -> 224x224 (Model)
                # We resize here so the buffer stores the correct size for both Inference and Training
                img_resized = cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_AREA)
                
                proprio = obs['proprio']
                
                # Add (Img, Proprio, PrevAction) to sliding window
                self.window.add(img_resized, proprio, self.last_executed_action)
                
                # --- 3. Teacher (Oracle) Query ---
                teacher_pose, teacher_grip, info = self.teacher.get_target_pose(obs)
                current_phase = info['gt_phase']
                
                # --- 4. Student Prediction ---
                student_pose, student_grip_logit = self._get_student_action(current_phase, goal_tensor)
                
                # --- 5. Gating & Safety Logic ---
                dist_err = np.linalg.norm(student_pose[:3] - teacher_pose[:3])
                
                # Stagnation Logic
                curr_vel = np.linalg.norm(obs['proprio'][7:10])
                vel_history.append(curr_vel)
                is_stagnant = (len(vel_history) == self.STAGNATION_STEPS) and (max(vel_history) < self.STAGNATION_VEL)
                
                # Gripper Disagreement
                student_grip_cmd = -1.0 if student_grip_logit > 0 else 1.0
                grip_mismatch = (student_grip_cmd != teacher_grip)
                
                # Decision Tree
                is_correction = False
                
                if is_stagnant:
                    # Case C: Stagnation -> Hard Takeover
                    final_pose = teacher_pose
                    final_grip = teacher_grip
                    is_correction = True
                    pbar.set_postfix_str("Status: STAGNANT - Takeover")
                    vel_history.clear()
                    
                elif dist_err > self.SAFE_DIST_THRESH or grip_mismatch:
                    # Case B: Drift/Grip Error -> Correction
                    final_pose = teacher_pose
                    final_grip = teacher_grip
                    is_correction = True
                    pbar.set_postfix_str(f"Status: DRIFT ({dist_err*100:.1f}cm)")
                    
                else:
                    # Case A: Safe -> Student Control
                    final_pose = student_pose
                    final_grip = student_grip_cmd
                    is_correction = False
                    pbar.set_postfix_str(f"Status: SAFE ({dist_err*100:.1f}cm)")

                # --- 6. Data Aggregation ---
                # Prepare sample for Replay Buffer (CPU Tensors)
                hist = self.window.get_history()
                
                sample = {
                    'initial_image': hist['initial_image'].squeeze(0).cpu(), # Now correctly (T, C, 224, 224)
                    'proprio_hist': hist['proprio_hist'].squeeze(0).cpu(),
                    'action_hist': hist['action_hist'].squeeze(0).cpu(),
                    'goal_image': goal_tensor.squeeze(0).cpu(),
                    'task_phase': torch.tensor([current_phase], dtype=torch.long),
                    # Targets (Always use Teacher as Ground Truth)
                    'ground_truth_subgoal_pose': torch.tensor(teacher_pose, dtype=torch.float32),
                    'ground_truth_gripper_state': torch.tensor([1.0 if teacher_grip < 0 else 0.0], dtype=torch.float32)
                }
                
                self.online_buffer.add(sample, is_correction=is_correction)
                if is_correction: correction_count += 1

                # --- 7. Execution ---
                target_base = self._transform_world_to_base(final_pose)
                
                try:
                    delta_joints = self.ik_solver.compute_delta_action(
                        target_ee_pose=target_base,
                        model=self.env.model,
                        data=self.env.data,
                        ee_site_id=self.env.ee_site_id,
                        joint_qpos_indices=np.arange(7),
                        effective_dt=sim_dt,
                        max_dq=max_dq
                    )
                except:
                    delta_joints = np.zeros(7)
                
                action = np.concatenate([delta_joints, [final_grip]])
                self.last_executed_action = action 
                
                obs, _, term, trunc, _ = self.env.step(action)
                episode_steps += 1
                pbar.update(1)
                
                if term or trunc or self.teacher.is_done():
                    done = True
                    
                # --- 8. Interleaved Training ---
                if episode_steps % self.cfg.dagger.train_every_n_steps == 0:
                    self._train_step()
            
            pbar.close()
            log.info(f"Epoch {epoch} Summary: Corrections={correction_count} ({correction_count/episode_steps:.1%})")
            
            # Save Checkpoint
            if (epoch + 1) % self.cfg.dagger.save_every_n_epochs == 0:
                path = f"checkpoints/dagger_adapter_epoch_{epoch}.pt"
                torch.save(self.student.state_dict(), path)
                log.info(f"Saved Adapter: {path}")


    def _train_step(self):
        """
        Performs one Gradient Update using Mixed Data.
        Mix: 50% Online (Corrections) + 50% Offline (Replay).
        """
        if len(self.online_buffer.correction_buffer) < self.cfg.dagger.batch_size // 2:
            return 

        self.student.train()
        
        # 1. Sample Online
        batch_online = self.online_buffer.sample_batch(
            batch_size=self.cfg.dagger.batch_size // 2,
            correction_ratio=0.8,
            device=self.device
        )
        
        # 2. Sample Offline
        indices = np.random.randint(0, len(self.offline_dataset), self.cfg.dagger.batch_size // 2)
        offline_samples = [self.offline_dataset[i] for i in indices if self.offline_dataset[i] is not None]
        
        if not offline_samples: return
        from torch.utils.data import default_collate
        batch_offline = default_collate(offline_samples)
        
        # Move offline to device
        for k, v in batch_offline.items():
            batch_offline[k] = v.to(self.device)

        # 3. Merge Batches
        input_batch = {}
        keys_to_merge = [
            'initial_image', 'goal_image', 'proprio_hist', 'action_hist',
            'task_phase', 'ground_truth_subgoal_pose', 'ground_truth_gripper_state'
        ]

        for k in keys_to_merge:
            if k not in batch_online or k not in batch_offline:
                continue
                
            t_on = batch_online[k]
            t_off = batch_offline[k]
            
            # --- ROBUST DIMENSION MATCHING ---
            
            # Case 1: Scalars (Online (B, 1) vs Offline (B,))
            if t_on.ndim == 2 and t_off.ndim == 1:
                t_off = t_off.unsqueeze(1)
            elif t_on.ndim == 1 and t_off.ndim == 2:
                t_on = t_on.unsqueeze(1)
                
            # Case 2: History Mismatch (B, T, ...) vs (B, ...)
            elif t_on.ndim == t_off.ndim + 1:
                repeat_shape = [1] * t_off.ndim
                repeat_shape[0] = self.history_len 
                t_off = t_off.unsqueeze(1).repeat(1, *repeat_shape)

            # Fallback: If dimensions match in rank but not size (e.g. B, 1 vs B, 10)
            if t_on.ndim == t_off.ndim:
                 if t_on.shape[1] == 1 and t_off.shape[1] > 1:
                     t_on = t_on.repeat(1, t_off.shape[1], *([1]*(t_on.ndim-2)))
                 elif t_off.shape[1] == 1 and t_on.shape[1] > 1:
                     t_off = t_off.repeat(1, t_on.shape[1], *([1]*(t_off.ndim-2)))

            input_batch[k] = torch.cat([t_on, t_off], dim=0)
            
        # --- 4. SANITIZE SCALARS FOR BASE MODEL ---
        # The Base Planner's embeddings expect (B) or (B, 1), NOT (B, 1, 1)
        # Squeeze extra dimensions from task_phase if present
        if input_batch['task_phase'].ndim > 1:
             input_batch['task_phase'] = input_batch['task_phase'].view(-1) # Flatten to (B,) or (B)
             
        # --- 5. PREPARE INPUTS ---
        # Base Planner Needs: Single Frame (Last)
        # Adapter Needs: Sequence (All)
        
        img_seq = input_batch['initial_image']
        prop_seq = input_batch['proprio_hist']
        
        # Overwrite/Add keys for Base Planner compatibility
        input_batch['initial_image'] = img_seq[:, -1]      # Last Frame
        input_batch['current_proprio'] = prop_seq[:, -1]   # Last Proprio
        
        # Restore history for Adapter
        input_batch['proprio_hist'] = prop_seq
        
        # 6. Update
        self.optimizer.zero_grad()
        out = self.student(input_batch)
        
        pred_pose = out['pose']
        pred_grip = out['gripper_logit']
        
        gt_pose = input_batch['ground_truth_subgoal_pose']
        gt_grip = input_batch['ground_truth_gripper_state']
        
        # Handle potential extra dims in GT
        if gt_grip.ndim > 2: gt_grip = gt_grip.view(gt_grip.shape[0], -1)
        if gt_pose.ndim > 2: gt_pose = gt_pose.view(gt_pose.shape[0], -1)
        
        loss_pose = F.l1_loss(pred_pose, gt_pose)
        loss_grip = F.binary_cross_entropy_with_logits(pred_grip, gt_grip)
        loss_reg = out['delta_magnitude'] * 0.01
        
        total_loss = loss_pose + loss_grip + loss_reg
        
        total_loss.backward()
        self.optimizer.step()

@hydra.main(version_base=None, config_path="../configs", config_name="train_online_dagger_config")
def main(cfg: DictConfig):
    orchestrator = DAggerOrchestrator(cfg)
    orchestrator.run()

if __name__ == "__main__":
    main()