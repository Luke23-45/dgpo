# FILE: scripts/run_online_dagger.py
# (Definitive, SOTA, DAgger Orchestrator)

"""
Online DAgger (Dataset Aggregation) Orchestrator.

This script implements the **Iterative Alignment** loop for the AWSP framework.
It bridges the "Simulation World" and the "Training World" to solve Covariate Shift.

Algorithm (DAgger):
1.  **Initialize**: Load Pre-trained Student (Policy) and Scripted Teacher (Expert).
2.  **Loop (Rounds)**:
    a.  **Rollout**: Student controls the robot. Teacher observes and labels the
        states visited by the Student (generating "Correction" tuples).
    b.  **Aggregate**: Add corrections to the RAM-based `OnlineReplayBuffer`.
    c.  **Update**: Train the Student on a mixture of Static Expert Data (Stability)
        and Online Correction Data (Recovery).
    d.  **Evaluate**: Check zero-shot performance.

Implementation Details:
-   **Dynamic Mixing**: Uses `MixedDataset` to interleave disk-based expert trajectories
    with RAM-based corrections without I/O blocking.
-   **Safety**: Persists the Replay Buffer to disk every round.
-   **Optimization**: Re-initializes the PyTorch Lightning Trainer every round to
    ensure clean optimizer states for fine-tuning.
"""

from __future__ import annotations

import logging
import os
import sys
import copy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import mujoco
import cv2
import hydra
import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    format="%(asctime)s [%(levelname)s] [DAgger] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("DAgger")


# ==============================================================================
# 1. DATA COLLECTION ENGINE (The Rollout Loop)
# ==============================================================================

class DAggerCollector:
    """
    Manages the interaction between Student, Teacher, and Environment.
    Generates 'Correction' samples.
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

        # Image Transform (Must match training exactly)
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor()
        ])

        # Calibration
        SIM_SUBSTEPS = 20 
        self.effective_dt = env.model.opt.timestep * SIM_SUBSTEPS
        self.max_dq = env.ACTION_SCALING_FACTOR / self.effective_dt

    @torch.no_grad()
    def collect_round(self, buffer: OnlineReplayBuffer, num_episodes: int, epsilon: float = 0.0) -> Dict[str, float]:
        """
        Runs simulation episodes.
        Student drives (mostly). Teacher corrects.
        
        Args:
            epsilon: Probability of executing Random actions (Exploration). 
                     Usually 0.0 for pure DAgger (Student drives).
        """
        self.student.eval() # Student in eval mode (no dropout)
        
        total_steps = 0
        successes = 0
        corrections_added = 0
        
        log.info(f"Starting Rollout: {num_episodes} Episodes...")
        
        for ep in tqdm(range(num_episodes), desc="Rollout"):
            # 1. Reset
            # Add noise to seed for diversity
            seed = self.cfg.seed + 9999 + ep 
            obs, _ = self.env.reset(seed=seed)
            
            # Teacher Reset
            self.teacher.reset()
            self.env.set_object_size(self.teacher.object.size)
            
            # 2. Goal Hallucination (For Student Input)
            # We use the virtual goal rendering trick from evaluation
            goal_pos = obs['goal_pos_world']
            # Inline render logic for speed (simplified version of context manager)
            saved_qpos = self.env.data.qpos.copy()
            saved_qvel = self.env.data.qvel.copy()
            try:
                q_adr = self.env.model.jnt_qposadr[self.env.object_joint_id]
                # Move obj to goal
                self.env.data.qpos[q_adr:q_adr+3] = goal_pos
                # Open gripper, retract arm (heuristic home)
                self.env.data.qpos[7:9] = 0.04
                mujoco.mj_forward(self.env.model, self.env.data)
                goal_img_np = self.env.render()
            finally:
                self.env.data.qpos[:] = saved_qpos
                self.env.data.qvel[:] = saved_qvel
                mujoco.mj_forward(self.env.model, self.env.data)
            
            goal_t = self.transform(Image.fromarray(goal_img_np)).unsqueeze(0).to(self.device)

            # 3. Step Loop
            episode_success = False
            
            for step in range(self.env.max_episode_steps):
                # --- A. Get Expert Correction (The Label) ---
                # We get what the expert *would* do in this state
                # Note: The expert needs the rich observation dict
                expert_obs = self.env.get_expert_obs()
                
                # Expert Logic
                # [PATCH] The expert now returns (target_pose, gripper_action, info) directly.
                # This matches the updated ScriptedExpert signature.
                gt_pose, gt_grip_act, gt_info = self.teacher.get_target_pose(expert_obs)
                
                # Convert Expert Action (Float -1/1) to State (1.0/0.0)
                # Logic: < -0.1 implies CLOSED (1.0), else OPEN (0.0)
                gt_gripper_state = 1.0 if gt_grip_act < -0.1 else 0.0
                
                # Get Phase (Explicit)
                gt_phase = gt_info['gt_phase']
                
                # --- B. Get Student Prediction (The Driver) ---
                img_t = self.transform(Image.fromarray(obs['image_primary'])).unsqueeze(0).to(self.device)
                prop_t = torch.from_numpy(obs['proprio']).float().unsqueeze(0).to(self.device)
                phase_t = torch.tensor([gt_phase], device=self.device) # Student knows phase (Assumption: Phase Oracle works)

                pred = self.student({
                    'initial_image': img_t, 'goal_image': goal_t,
                    'task_phase': phase_t, 'current_proprio': prop_t
                })
                
                # Student outputs
                raw_pose = pred['pose'].squeeze().cpu().numpy()
                raw_logit = pred['gripper_logit'].item()
                
                # --- C. Save Correction (DAgger) ---
                # Store: Student State -> Teacher Action
                buffer.add(
                    initial_image=obs['image_primary'], # Save raw uint8
                    goal_image=goal_img_np,             # Save raw uint8
                    task_phase=gt_phase,
                    current_proprio=obs['proprio'],
                    gt_pose=gt_pose,
                    gt_gripper=gt_gripper_state,
                    advantage=self.cfg.dagger.correction_advantage # High value (e.g. 10.0)
                )
                corrections_added += 1

                # --- D. Execute Action (Student Drives) ---
                # Decode Student Action
                student_grip_cmd = -1.0 if raw_logit > 0.0 else 1.0
                
                # Solve IK for Student Pose
                # [PATCH] Use the robust compute_delta_action with calibrated max_dq
                try:
                    delta_joints = self.ik_solver.compute_delta_action(
                        target_ee_pose=raw_pose,
                        model=self.env.model,
                        data=self.env.data,
                        ee_site_id=self.env.ee_site_id,
                        joint_qpos_indices=np.arange(7),
                        effective_dt=self.effective_dt,
                        max_dq=self.max_dq # Uses self.max_dq derived in __init__
                    )
                except Exception:
                    delta_joints = np.zeros(7)

                action = np.concatenate([delta_joints, [student_grip_cmd]])
                
                # Step
                obs, _, terminated, truncated, _ = self.env.step(action)
                
                # Check Success
                dist = np.linalg.norm(obs['object_pos_world'] - obs['goal_pos_world'])
                obj_z = obs['object_pos_world'][2]
                if dist < 0.05 and obj_z > 0.41 and obs['is_grasped'][0] > 0.5:
                    episode_success = True
                
                if episode_success or terminated or truncated:
                    break
                
                total_steps += 1
            
            if episode_success: successes += 1
        
        metrics = {
            "rollout/success_rate": successes / num_episodes,
            "rollout/corrections_added": corrections_added
        }
        return metrics


# ==============================================================================
# 2. MASTER ORCHESTRATOR
# ==============================================================================

@hydra.main(version_base=None, config_path="../configs", config_name="run_online_dagger_config")
def main(cfg: DictConfig):
    # 1. Setup & Reproducibility
    pl.seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    workspace_dir = Path(os.getcwd())
    log.info(f"DAgger Workspace: {workspace_dir}")

    # 2. Initialize Components
    
    # A. Model (Student) - Load from Pre-trained Checkpoint
    log.info(f"Loading Base Policy from: {cfg.model_checkpoint}")
    # We use the LightningModule wrapper to handle loading correctly
    pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
        cfg.model_checkpoint, map_location=device, strict=True
    )
    student_model = pl_module.model
    # Ensure config matches loaded model
    train_cfg = pl_module.cfg 

    # B. Expert (Teacher)
    expert_config = ExpertConfig() # Use defaults or load from hydra

    obj_profile = ObjectProfile(size=np.array([0.04, 0.04, 0.04]), grasp_width_normalized=0.6)

    teacher = ScriptedExpert(object_profile=obj_profile, cfg=expert_config)

    # C. Environment & IK
    env = PandaEnv(xml_path="envs/panda_pick_place.xml", control_mode='delta', enable_domain_randomization=True)
    ik_solver = IKSolver(urdf_path="urdf/panda_mujoco_kinematics.urdf")

    # D. Datasets
    # 1. Static Dataset (The Anchor)
    log.info(f"Loading Static Dataset: {cfg.static_dataset_path}")
    static_dataset = SemanticPlannerDataset(
        dataset_path=cfg.static_dataset_path,
        use_aug=True # Essential to maintain generalization
    )
    
    # 2. Online Buffer (The Correction Memory)
    online_buffer = OnlineReplayBuffer(
        capacity=cfg.dagger.buffer_capacity,
        default_advantage=cfg.dagger.correction_advantage
    )

    # E. Collector
    collector = DAggerCollector(env, student_model, teacher, ik_solver, device, cfg)

    # ============================
    # 3. THE DAGGER LOOP
    # ============================
    
    for round_idx in range(1, cfg.dagger.num_rounds + 1):
        log.info(f"\n{'='*20} DAGGER ROUND {round_idx}/{cfg.dagger.num_rounds} {'='*20}")
        
        # --- PHASE A: ROLLOUT ---
        log.info(">>> PHASE A: ROLLOUT (Data Collection)")
        rollout_metrics = collector.collect_round(
            buffer=online_buffer,
            num_episodes=cfg.dagger.rollout_episodes_per_round
        )
        log.info(f"Rollout Metrics: {rollout_metrics}")
        log.info(f"Buffer Size: {len(online_buffer)}")
        
        # Snapshot Buffer for safety
        buffer_save_path = workspace_dir / f"buffer_round_{round_idx}.pkl"
        online_buffer.save_to_disk(str(buffer_save_path))

        # --- PHASE B: UPDATE (Fine-Tuning) ---
        log.info(">>> PHASE B: UPDATE (Fine-Tuning)")
        
        # Create Mixed Dataset (Static + Online)
        mixed_dataset = MixedDataset(
            static_dataset=static_dataset,
            online_dataset=online_buffer,
            mix_ratio=cfg.dagger.mix_ratio
        )
        
        # Setup DataLoader with SOTA Sampler
        # The sampler will drive indices from the Static set. 
        # MixedDataset intercepts these and injects Online data probabilistically.
        sampler = EpisodeAwareSampler(mixed_dataset, shuffle=True, seed=cfg.seed + round_idx)
        
        loader = DataLoader(
            mixed_dataset,
            batch_size=cfg.training.batch_size,
            sampler=sampler,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            collate_fn=semantic_planner_collate_fn
        )
        
        # Initialize Fresh Trainer (To reset optimizer state for fine-tuning)
        # We use the same PL Module, just re-attach it.
        pl_module.train() # Set to train mode
        
        # Configure Trainer for this round
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
            default_root_dir=str(workspace_dir)
        )
        
        # Train
        trainer.fit(pl_module, train_dataloaders=loader)
        
        # Update Student Reference (Though PL Module updates in place, this is semantic)
        student_model = pl_module.model
        
        # --- PHASE C: EVALUATION ---
        log.info(">>> PHASE C: EVALUATION (Zero-Shot)")
        # (We reuse the collector in eval mode/low noise)
        # Note: Proper evaluation usually needs a separate seeded loop.
        # For DAgger progress tracking, we rely on the training metrics and next rollout.

    log.info("DAgger Loop Complete.")
    final_ckpt = workspace_dir / "final_policy.ckpt"
    trainer.save_checkpoint(final_ckpt)
    log.info(f"Final Model Saved: {final_ckpt}")

if __name__ == "__main__":
    main()