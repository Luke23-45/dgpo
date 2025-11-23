# FILE: scripts/run_online_dagger.py
# (Definitive, v13.0 Aligned - Transport Safety & Ratchet Logic)

"""
Online DAgger (Dataset Aggregation) Orchestrator.
ALIGNED WITH: evaluate_semantic_planner.py (v13.0)

Key Features:
1.  **Physics-Aware Sequencer**: Implements Transport Safety, Magnetic Approach, and Trigger/Latch logic
    during the Student Rollout to ensure data is collected in the valid distribution.
2.  **Lightweight Smoothing**: Uses EMA and Hysteresis to stabilize Student actions.
3.  **SOTA Architecture Support**: Handles Disentangled v8.0/v9.0 inputs/outputs (Chunking).
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import cv2
import hydra
import mujoco
import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image
from torch.utils.data import DataLoader
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
from utils.scripted_expert import ScriptedExpert, ObjectProfile, ExpertConfig
from utils.semantic_planner_dataset import SemanticPlannerDataset, semantic_planner_collate_fn
from utils.online_buffer import OnlineReplayBuffer
from utils.mixed_dataset import MixedDataset
from utils.samplers import EpisodeAwareSampler

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [DAgger-v13] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("DAgger")


# ==============================================================================
# 1. UTILITIES: SMOOTHER & ROBUST RENDERING (v13.0 Port)
# ==============================================================================

class LightweightSmoother:
    """EMA Smoother with Gripper Hysteresis (Ported from Eval v13.0)."""
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
        
        # Hysteresis
        if not self.gripper_closed and self.smooth_grip_logit > 0.5:
            self.gripper_closed = True
        elif self.gripper_closed and self.smooth_grip_logit < -0.5:
            self.gripper_closed = False
            
        gripper_cmd = -1.0 if self.gripper_closed else 1.0
        return self.smooth_pose, gripper_cmd

def calculate_retract_joints(env: PandaEnv, ik_solver: IKSolver, target_pos_world: np.ndarray, object_quat_wxyz: np.ndarray) -> np.ndarray:
    """Calculates joints for 'Done' state."""
    hover_z = 0.55 
    target_pos = np.array([target_pos_world[0], target_pos_world[1], hover_z])
    
    q_obj = np.array([object_quat_wxyz[1], object_quat_wxyz[2], object_quat_wxyz[3], object_quat_wxyz[0]])
    r_target = R.from_quat(q_obj) * R.from_euler('x', 180, degrees=True)
    
    base_pos, base_quat = env.get_base_pose()
    R_base_world = R.from_quat(base_quat).as_matrix()
    T_world_base = np.linalg.inv(np.vstack([
        np.hstack([R_base_world, base_pos.reshape(3,1)]),
        [0,0,0,1]
    ]))
    
    target_in_base_pos = (T_world_base @ np.append(target_pos, 1.0))[:3]
    target_in_base_rot = T_world_base[:3, :3] @ r_target.as_matrix()

    current_joints = env.data.qpos[:7].copy()
    initial_guess = [0.0]*len(ik_solver.chain.links)
    for i, v in enumerate(current_joints): initial_guess[ik_solver._active_idx[i]] = v

    full_joints = ik_solver.chain.inverse_kinematics(
        target_position=target_in_base_pos,
        target_orientation=target_in_base_rot,
        orientation_mode="all",
        initial_position=initial_guess
    )
    return np.array([full_joints[i] for i in ik_solver._active_idx])

@contextmanager
def render_robust_virtual_goal(env: PandaEnv, ik_solver: IKSolver, target_pos_world: np.ndarray):
    """
    Robust Virtual Goal Renderer (v13.0).
    Resets robot to home position before rendering to prevent occlusion.
    """
    saved_qpos = env.data.qpos.copy()
    saved_qvel = env.data.qvel.copy()
    saved_ctrl = env.data.ctrl.copy()
    try:
        # Reset robot to home to clear view
        home_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        env.data.qpos[:7] = home_qpos
        env.data.qpos[7:] = 0.04 # Open grippers
        
        # Move object to goal
        obj_jnt_adr = env.model.jnt_qposadr[env.object_joint_id]
        curr_obj_quat = env.data.qpos[obj_jnt_adr+3 : obj_jnt_adr+7].copy()
        
        # Safe Z enforcement
        safe_z = max(target_pos_world[2], 0.42)
        env.data.qpos[obj_jnt_adr : obj_jnt_adr+3] = [target_pos_world[0], target_pos_world[1], safe_z]
        env.data.qpos[obj_jnt_adr+3 : obj_jnt_adr+7] = curr_obj_quat
        
        # Calculate ideal retraction joints for visuals
        try:
             target_joints = calculate_retract_joints(env, ik_solver, target_pos_world, curr_obj_quat)
             env.data.qpos[:7] = target_joints
        except:
             pass

        env.data.qvel[:] = 0
        mujoco.mj_forward(env.model, env.data)
        yield
    finally:
        env.data.qpos[:] = saved_qpos
        env.data.qvel[:] = saved_qvel
        env.data.ctrl[:] = saved_ctrl
        mujoco.mj_forward(env.model, env.data)


# ==============================================================================
# 2. COMPONENT: DATA COLLECTOR (v13.0 LOGIC)
# ==============================================================================

class DAggerCollector:
    """
    Manages rollout with v13.0 Physics-Aware Logic.
    """
    def __init__(self, 
                 env: PandaEnv, 
                 student_model: SemanticPlanner, 
                 teacher_expert: ScriptedExpert,
                 ik_solver: IKSolver,
                 device: torch.device,
                 cfg: DictConfig):
        self.env = env
        self.student = student_model
        self.teacher = teacher_expert
        self.ik_solver = ik_solver
        self.device = device
        self.cfg = cfg

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            # Normalize included in transform to match Eval script
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Physics Sync
        SIM_SUBSTEPS = 20 
        self.effective_dt = env.model.opt.timestep * SIM_SUBSTEPS
        self.max_dq = env.ACTION_SCALING_FACTOR / self.effective_dt

        # v13.0 Utilities
        self.smoother = LightweightSmoother()
        self.chunk_size = cfg.model.get("chunk_size", 10) # Default for v9/v13 models

    @torch.no_grad()
    def collect_round(self, buffer: OnlineReplayBuffer, num_episodes: int) -> Dict[str, float]:
        self.student.eval()
        
        successes = 0
        corrections_added = 0
        
        for ep in tqdm(range(num_episodes), desc="Rollout"):
            seed = self.cfg.seed + 99999 + ep 
            obs, _ = self.env.reset(seed=seed)
            self.smoother.reset()
            grasp_latch_counter = 0
            
            self.teacher.reset()
            self.env.set_object_size(self.teacher.object.size)
            
            # 1. Robust Goal Rendering
            goal_pos = obs['goal_pos_world']
            with render_robust_virtual_goal(self.env, self.ik_solver, goal_pos):
                goal_img_np = self.env.render()
            goal_t = self.transform(Image.fromarray(goal_img_np)).unsqueeze(0).to(self.device)

            # Initialize history buffer (Inference) - NORMALIZED TENSOR
            curr_img_pil = Image.fromarray(obs['image_primary'])
            prev_img_buffer = self.transform(curr_img_pil).unsqueeze(0).to(self.device)

            # Initialize raw buffer (Storage) - RAW NUMPY
            prev_img_numpy = obs['image_primary'].copy() 

            episode_success = False
            
            for step in range(self.env.max_episode_steps):
                # --- A. TEACHER ---
                expert_obs = self.env.get_expert_obs()
                gt_pose, gt_grip_act, gt_info = self.teacher.get_target_pose(expert_obs)
                gt_phase = gt_info['gt_phase']
                gt_gripper_state = float(gt_info['gt_gripper_intent'])
                
                # --- B. STUDENT INFERENCE ---
                curr_img_pil = Image.fromarray(obs['image_primary'])
                curr_tensor = self.transform(curr_img_pil).unsqueeze(0).to(self.device)
                proprio = torch.from_numpy(obs['proprio']).float().unsqueeze(0).to(self.device)
                
                # Input Batch (v8.0 Structure)
                # [FIX]: Use 'prev_img_buffer' (Tensor) here, NOT 'prev_img_numpy'
                batch = {
                    'prev_image': prev_img_buffer, 
                    'curr_image': curr_tensor,
                    'goal_image': goal_t,
                    'curr_proprio': proprio
                }
                
                # Update Inference History
                prev_img_buffer = curr_tensor.clone()

                # Model Forward
                pred = self.student(batch)
                
                # Extract Output (Handling Chunks vs Single)
                if 'pose_chunk' in pred:
                     chunk_pose = pred['pose_chunk'].cpu().numpy()[0]
                     chunk_grip = pred['gripper_chunk'].cpu().numpy()[0]
                     lookahead_idx = min(4, self.chunk_size - 1)
                     raw_pose = chunk_pose[lookahead_idx].copy()
                     raw_logit = chunk_grip[lookahead_idx].item()
                else:
                     raw_pose = pred['pose'].squeeze().cpu().numpy()
                     raw_logit = pred['gripper_logit'].item()

                # --- C. DATA AGGREGATION ---
                gt_pose_chunk = np.tile(gt_pose, (self.chunk_size, 1))
                gt_grip_chunk = np.full((self.chunk_size, 1), gt_gripper_state, dtype=np.float32)

                # Store with keys matching SemanticPlannerDataset
                # [FIX]: Use 'prev_img_numpy' (Uint8) here for storage
                sample_data = {
                    'curr_image': obs['image_primary'], 
                    'prev_image': prev_img_numpy,          
                    'goal_image': goal_img_np,          
                    'curr_proprio': obs['proprio'],
                    'gt_pose_chunk': gt_pose_chunk.astype(np.float32),
                    'gt_grip_chunk': gt_grip_chunk.astype(np.float32),
                    'gt_phase_label': int(gt_phase),
                    'advantage': self.cfg.dagger.correction_advantage
                }

                buffer.add(sample=sample_data, is_correction=True)
                corrections_added += 1

                # Update Storage History
                prev_img_numpy = obs['image_primary'].copy()

                # --- D. v13.0 PHYSICS-AWARE SEQUENCER ---
                # ... (Rest of the loop remains unchanged)
                ee_pos = obs['ee_pose_world']
                obj_pos = obs['object_pos_world']
                dist_xy = np.linalg.norm(ee_pos[:2] - obj_pos[:2])
                is_physically_grasped = obs['is_grasped'][0] > 0.5
                
                current_action_gain = 2.0 

                if is_physically_grasped:
                    raw_logit = 5.0
                else:
                    if ee_pos[2] > 0.46 and grasp_latch_counter == 0:
                        raw_logit = -5.0
                    if dist_xy < 0.30 and grasp_latch_counter == 0:
                        raw_pose[:2] = (0.5 * raw_pose[:2]) + (0.5 * obj_pos[:2])
                        if dist_xy < 0.05: raw_pose[2] = 0.43 
                        elif dist_xy < 0.15: raw_pose[2] = min(raw_pose[2], 0.47)
                    if dist_xy < 0.03 and ee_pos[2] < 0.45 and grasp_latch_counter == 0:
                        grasp_latch_counter = 45 
                    if grasp_latch_counter > 0:
                        raw_logit = 5.0 
                        if grasp_latch_counter > 25:
                            raw_pose[:2] = obj_pos[:2]; raw_pose[2] = 0.425; current_action_gain = 0.5 
                        else:
                            raw_pose[:2] = ee_pos[:2]; raw_pose[2] = 0.55; current_action_gain = 1.0 
                        grasp_latch_counter -= 1

                raw_pose[2] = max(raw_pose[2], 0.405)
                target_pose, gripper_cmd = self.smoother.update(raw_pose, raw_logit)
                
                try:
                    delta_joints = self.ik_solver.compute_delta_action(
                        target_ee_pose=target_pose,
                        model=self.env.model, data=self.env.data, ee_site_id=self.env.ee_site_id,
                        joint_qpos_indices=np.arange(7), effective_dt=self.effective_dt, max_dq=self.max_dq
                    )
                except:
                    delta_joints = np.zeros(7)

                action = np.concatenate([delta_joints * current_action_gain, [gripper_cmd]])
                obs, _, terminated, truncated, _ = self.env.step(action)
                
                obj_pos_now = obs['object_pos_world']
                dist_goal = np.linalg.norm(obj_pos_now - goal_pos)
                if dist_goal < 0.05 and obj_pos_now[2] > 0.415 and obs['is_grasped'][0] > 0.5:
                    episode_success = True
                
                if episode_success or terminated or truncated:
                    break
            
            if episode_success: successes += 1
        
        return {"success_rate": successes / num_episodes, "corrections_added": corrections_added}
# ==============================================================================
# 3. MASTER ORCHESTRATOR
# ==============================================================================

@hydra.main(version_base=None, config_path="../configs", config_name="run_online_dagger_config")
def main(cfg: DictConfig):
    # 1. Setup
    pl.seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workspace_dir = Path(os.getcwd())
    log.info(f"DAgger Workspace: {workspace_dir}")

    # 2. Load Student (SOTA SURGICAL LOADING)
    log.info(f"Loading Base Policy: {cfg.model_checkpoint}")
    pl_module = SemanticPlannerLightningModule(cfg)
    student_model = pl_module.model
    student_model.to(device)

    # B. Surgical Weight Injection (Hot-Patching)
    if os.path.exists(cfg.model_checkpoint):
        log.info("Performing Surgical Weight Injection...")
        checkpoint = torch.load(cfg.model_checkpoint, map_location=device, weights_only=False)
        state_dict = checkpoint['state_dict']
        
        model_state = pl_module.state_dict()
        filtered_state_dict = {}
        
        for k, v in state_dict.items():
            if k in model_state:
                # Filter mismatches
                if v.shape != model_state[k].shape:
                    log.warning(f"Skipping shape mismatch: {k} | Ckpt: {v.shape} vs Model: {model_state[k].shape}")
                    continue
                filtered_state_dict[k] = v
        
        pl_module.load_state_dict(filtered_state_dict, strict=False)
        log.info("Weights Loaded Successfully.")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {cfg.model_checkpoint}")

    # [CRITICAL] Override Optimizer Warmup
    if 'optimizer' in pl_module.cfg:
        log.info("Overriding Optimizer Warmup for DAgger Fine-Tuning -> 0.0")
        # FIX: Handle both DictConfig (struct mode) and standard Dicts
        if isinstance(pl_module.cfg, DictConfig):
            with open_dict(pl_module.cfg):
                 pl_module.cfg.optimizer.warmup_percentage = 0.0
        else:
            # Fallback for standard mutable dicts (no context manager needed)
            # Try attribute access first, then item access
            try:
                pl_module.cfg.optimizer.warmup_percentage = 0.0
            except AttributeError:
                pl_module.cfg['optimizer']['warmup_percentage'] = 0.0

    # 3. Initialize Expert
    expert_cfg = ExpertConfig() 
    obj_profile = ObjectProfile(size=np.array([0.04, 0.04, 0.04]), grasp_width_normalized=0.6)
    teacher = ScriptedExpert(object_profile=obj_profile, cfg=expert_cfg)

    # 4. Environment & IK
    env = PandaEnv(xml_path="envs/panda_pick_place.xml", control_mode='delta', enable_domain_randomization=True)
    ik_solver = IKSolver(urdf_path=cfg.ik_solver_path)

    # 5. Datasets
    log.info(f"Loading Static Dataset: {cfg.static_dataset_path}")
    static_dataset = SemanticPlannerDataset(
        dataset_path=cfg.static_dataset_path,
        use_aug=True,
        chunk_size=cfg.model.get("chunk_size", 10)
    )
    
    online_buffer = OnlineReplayBuffer(
        capacity=cfg.dagger.buffer_capacity,
        default_advantage=cfg.dagger.correction_advantage
    )

    # 6. Init Collector (With v13.0 Logic)
    collector = DAggerCollector(env, student_model, teacher, ik_solver, device, cfg)

    # ============================
    # THE LOOP
    # ============================
    
    for round_idx in range(1, cfg.dagger.num_rounds + 1):
        log.info(f"\n{'='*20} DAGGER ROUND {round_idx}/{cfg.dagger.num_rounds} {'='*20}")
        
        # --- PHASE A: ROLLOUT ---
        log.info(">>> PHASE A: ROLLOUT (Data Collection)")
        metrics = collector.collect_round(
            buffer=online_buffer,
            num_episodes=cfg.dagger.rollout_episodes_per_round
        )
        log.info(f"Rollout Complete. Success Rate: {metrics['success_rate']*100:.1f}%")
        log.info(f"Buffer Size: {len(online_buffer)}")
        
        online_buffer.save_to_disk(str(workspace_dir / f"buffer_round_{round_idx}.pkl"))

        # --- PHASE B: UPDATE (Fine-Tuning) ---
        log.info(">>> PHASE B: UPDATE (Fine-Tuning)")
        
        mixed_dataset = MixedDataset(
            static_dataset=static_dataset,
            online_dataset=online_buffer,
            mix_ratio=cfg.dagger.mix_ratio
        )
        
        sampler = EpisodeAwareSampler(mixed_dataset, shuffle=True, seed=cfg.seed + round_idx)
        
        loader = DataLoader(
            mixed_dataset,
            batch_size=cfg.training.batch_size,
            sampler=sampler,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            collate_fn=semantic_planner_collate_fn
        )
        
        # Create a fresh trainer every round to ensure clean state
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=workspace_dir / "checkpoints",
            filename=f"dagger_round_{round_idx}_step={{step}}",
            save_top_k=1,
            monitor="train/loss"
        )
        
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            precision="16-mixed",
            max_epochs=cfg.dagger.epochs_per_round,
            callbacks=[checkpoint_callback],
            log_every_n_steps=10,
            default_root_dir=str(workspace_dir),
            enable_progress_bar=True
        )
        
        # Train
        pl_module.train()
        trainer.fit(pl_module, train_dataloaders=loader)
        
        round_ckpt = workspace_dir / f"model_round_{round_idx}.ckpt"
        trainer.save_checkpoint(round_ckpt)
        log.info(f"Round {round_idx} Model Saved: {round_ckpt}")
        
        # Model stays loaded in pl_module for next round, but we refresh the object to be safe
        student_model = pl_module.model
        if hasattr(loader, "_iterator"):
            del loader._iterator
        del loader
        import gc
        gc.collect() # Force garbage collection of worker processes

    log.info("DAgger Loop Complete.")
    final_ckpt = workspace_dir / "final_policy_dagger.ckpt"
    trainer.save_checkpoint(final_ckpt)
    log.info(f"Final Converged Model: {final_ckpt}")
    env.close()

if __name__ == "__main__":
    main()