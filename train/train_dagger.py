# FILE: train/train_dagger.py
"""
DAgger-Style Trainer for SemanticPlanner Fine-Tuning

This trainer implements the DAgger (Dataset Aggregation) approach:
1. Expert EXECUTES actions → Robot follows successful trajectories
2. Policy PREDICTS actions → For computing imitation loss
3. Train policy to minimize divergence from expert

Key benefits over standard DGPO:
- 100% successful trajectories (expert drives the robot)
- On-distribution training (policy learns from expert's state distribution)
- Simpler than RL (no value network, no GAE, no PPO)

Usage:
    python train/train_dagger.py
    python train/train_dagger.py training.total_iterations=50
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Project Imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.panda_env import PandaEnv
from models.semantic_planner import SemanticPlanner, SemanticPlannerConfig
from train.train_semantic_planner import SemanticPlannerLightningModule
from utils.ik_solver import IKSolver
from utils.dgpo_expert import DGPOExpert, DGPOExpertConfig, ObjectProfile

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("DAgger")


# ==============================================================================
# 1. REPLAY BUFFER
# ==============================================================================

@dataclass
class DAggerReplayBuffer:
    """
    Simple replay buffer for DAgger rollouts.
    Stores PRE-TRANSFORMED tensors for fast training.
    """
    # Store pre-transformed tensors for speed (transforms done once during collection)
    prev_images: List[torch.Tensor] = field(default_factory=list)
    curr_images: List[torch.Tensor] = field(default_factory=list)
    goal_images: List[torch.Tensor] = field(default_factory=list)
    proprios: List[torch.Tensor] = field(default_factory=list)
    expert_poses: List[torch.Tensor] = field(default_factory=list)  # 7D target poses
    expert_grippers: List[float] = field(default_factory=list)  # -1.0 (close) or 1.0 (open)
    
    def add(
        self,
        prev_img_t: torch.Tensor,  # Pre-transformed tensor
        curr_img_t: torch.Tensor,  # Pre-transformed tensor
        goal_img_t: torch.Tensor,  # Pre-transformed tensor
        proprio: np.ndarray,
        expert_pose: np.ndarray,
        expert_gripper: float,
    ):
        """Add a single timestep to the buffer."""
        self.prev_images.append(prev_img_t.cpu())  # Store on CPU to save GPU memory
        self.curr_images.append(curr_img_t.cpu())
        self.goal_images.append(goal_img_t.cpu())
        self.proprios.append(torch.from_numpy(proprio.copy()).float())
        self.expert_poses.append(torch.from_numpy(expert_pose.copy()).float())
        self.expert_grippers.append(expert_gripper)
    
    def clear(self):
        """Clear all data from buffer."""
        self.prev_images.clear()
        self.curr_images.clear()
        self.goal_images.clear()
        self.proprios.clear()
        self.expert_poses.clear()
        self.expert_grippers.clear()
    
    def __len__(self) -> int:
        return len(self.prev_images)


# ==============================================================================
# 2. GOAL IMAGE RENDERING (From DGPO)
# ==============================================================================

def render_goal_image(env: PandaEnv, goal_pos: np.ndarray) -> np.ndarray:
    """Renders the goal image by teleporting object to goal position."""
    # Save current state
    original_qpos = env.data.qpos.copy()
    original_qvel = env.data.qvel.copy()
    
    # Move object to goal
    obj_joint_adr = env.model.jnt_qposadr[env.object_joint_id]
    env.data.qpos[obj_joint_adr:obj_joint_adr + 3] = goal_pos
    env.data.qvel[:] = 0
    
    # Forward and render
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    goal_image = env.render()
    
    # Restore state
    env.data.qpos[:] = original_qpos
    env.data.qvel[:] = original_qvel
    mujoco.mj_forward(env.model, env.data)
    
    return goal_image


# ==============================================================================
# 3. DAGGER TRAINER
# ==============================================================================

class DAggerTrainer:
    """
    DAgger-Style Trainer for SemanticPlanner Fine-Tuning.
    
    Core loop:
    1. Expert provides target pose → IK computes action → Robot executes
    2. Policy observes state → Makes prediction (not executed)
    3. Collect (observation, expert_action) pairs
    4. Train policy with imitation loss
    """
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Device: {self.device}")
        
        # 1. Load Pre-trained Policy
        log.info(f"Loading BC checkpoint: {cfg.bc_checkpoint}")
        pl_module = SemanticPlannerLightningModule.load_from_checkpoint(
            cfg.bc_checkpoint, map_location=self.device, strict=True
        )
        self.policy = pl_module.model.to(self.device)
        self.policy.train()  # Enable training mode
        
        # Freeze vision backbone if configured
        if cfg.get("freeze_vision", True):
            for param in self.policy.vision_backbone.parameters():
                param.requires_grad = False
            log.info("Froze vision backbone parameters.")
        
        # 2. Initialize Environment
        self.env = PandaEnv(
            xml_path=cfg.environment.xml_path,
            control_mode="delta",
            render_mode="rgb_array"
        )
        
        # 3. Initialize IK Solver
        self.ik_solver = IKSolver(urdf_path=cfg.environment.urdf_path)
        
        # 4. Initialize Expert (DGPOExpert - no timeouts)
        object_profile = ObjectProfile(
            size=np.array(cfg.expert.object_size),
            grasp_width_normalized=cfg.expert.grasp_width
        )
        expert_cfg = DGPOExpertConfig(
            hover_height=cfg.expert.get('hover_height', 0.15),
            grasp_offset_z=cfg.expert.get('grasp_offset_z', 0.025),
        )
        self.expert = DGPOExpert(
            object_profile=object_profile,
            cfg=expert_cfg,
        )
        
        # 5. Control calibration
        SIM_SUBSTEPS = 20
        self.effective_dt = self.env.model.opt.timestep * SIM_SUBSTEPS
        self.max_dq = self.env.ACTION_SCALING_FACTOR / self.effective_dt
        
        # 6. Optimizer (only trainable params)
        trainable_params = [p for p in self.policy.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=cfg.training.learning_rate
        )
        
        # 7. Image transform - MUST EXACTLY MATCH BC TRAINING!
        # BC uses: Resize(224, BICUBIC) + ToTensor() + Normalize(0.5, 0.5) -> [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # 8. Replay buffer
        self.buffer = DAggerReplayBuffer()
        
        # 9. Statistics
        self.iteration = 0
        self.total_steps = 0
        
        # 10. Ensure output directory exists
        os.makedirs(cfg.training.output_dir, exist_ok=True)
        
        log.info("DAgger Trainer initialized.")
    
    def _prepare_batch(
        self,
        prev_img: np.ndarray,
        curr_img: np.ndarray,
        goal_img: np.ndarray,
        proprio: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        """Prepares observation tensors for the policy (single sample)."""
        prev_t = self.transform(Image.fromarray(prev_img)).unsqueeze(0).to(self.device)
        curr_t = self.transform(Image.fromarray(curr_img)).unsqueeze(0).to(self.device)
        goal_t = self.transform(Image.fromarray(goal_img)).unsqueeze(0).to(self.device)
        proprio_t = torch.from_numpy(proprio).float().unsqueeze(0).to(self.device)
        
        return {
            "prev_image": prev_t,
            "curr_image": curr_t,
            "goal_image": goal_t,
            "curr_proprio": proprio_t
        }
    
    def _prepare_training_batch(
        self,
        indices: np.ndarray
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Prepares a batch of samples from the buffer for training.
        Buffer already contains pre-transformed tensors for speed.
        Returns: (input_batch, expert_poses, expert_grippers)
        """
        # Just stack pre-transformed tensors - no PIL transforms needed!
        prev_imgs = torch.stack([self.buffer.prev_images[idx] for idx in indices])
        curr_imgs = torch.stack([self.buffer.curr_images[idx] for idx in indices])
        goal_imgs = torch.stack([self.buffer.goal_images[idx] for idx in indices])
        proprios = torch.stack([self.buffer.proprios[idx] for idx in indices])
        expert_poses = torch.stack([self.buffer.expert_poses[idx] for idx in indices])
        
        # Convert gripper to binary: -1.0 (close) -> 1.0, 1.0 (open) -> 0.0
        expert_grippers = torch.tensor(
            [1.0 if self.buffer.expert_grippers[idx] < 0 else 0.0 for idx in indices],
            dtype=torch.float32
        ).unsqueeze(1)
        
        batch = {
            "prev_image": prev_imgs.to(self.device),
            "curr_image": curr_imgs.to(self.device),
            "goal_image": goal_imgs.to(self.device),
            "curr_proprio": proprios.to(self.device)
        }
        
        return batch, expert_poses.to(self.device), expert_grippers.to(self.device)
    
    def collect_rollouts(self, n_steps: int) -> Dict[str, float]:
        """
        Collects rollout data by executing EXPERT actions.
        
        Key difference from DGPO: Expert executes, policy only predicts for data collection.
        """
        self.buffer.clear()
        self.policy.eval()
        
        episode_count = 0
        success_count = 0
        
        # Reset environment and expert
        self.env.reset()
        obs = self.env.get_expert_obs()
        self.expert.reset()
        
        # Render goal image once per episode and pre-transform
        goal_img_np = render_goal_image(self.env, obs['goal_pos_world'])
        goal_img_t = self.transform(Image.fromarray(goal_img_np))
        prev_img_np = obs['image_primary'].copy()
        prev_img_t = self.transform(Image.fromarray(prev_img_np))
        
        pbar = tqdm(total=n_steps, desc=f"Collecting rollouts (Iter {self.iteration})")
        
        for step in range(n_steps):
            curr_img_np = obs['image_primary']
            proprio = obs['proprio']
            
            # Pre-transform current image
            curr_img_t = self.transform(Image.fromarray(curr_img_np))
            
            # 1. Get Expert's target pose (this is what we'll EXECUTE)
            expert_pose, expert_grip, info = self.expert.get_target_pose(obs)
            
            # 2. Store in buffer (pre-transformed tensors for fast training)
            self.buffer.add(
                prev_img_t=prev_img_t,
                curr_img_t=curr_img_t,
                goal_img_t=goal_img_t,
                proprio=proprio,
                expert_pose=expert_pose,
                expert_gripper=expert_grip,
            )
            
            # 5. Compute EXPERT's action via IK
            try:
                delta_joints = self.ik_solver.compute_delta_action(
                    target_ee_pose=expert_pose,  # EXPERT pose, not policy
                    model=self.env.model,
                    data=self.env.data,
                    ee_site_id=self.env.ee_site_id,
                    joint_qpos_indices=np.arange(7),
                    effective_dt=self.effective_dt,
                    max_dq=self.max_dq
                )
            except Exception as e:
                log.warning(f"IK failed at step {step}: {e}")
                delta_joints = np.zeros(7)
            
            action = np.concatenate([delta_joints, [expert_grip]])
            
            # 6. Execute EXPERT's action in environment
            next_obs, _, terminated, truncated, _ = self.env.step(action)
            next_obs = self.env.get_expert_obs()
            done = terminated or truncated or self.expert.is_done()
            
            # 7. Update state - carry tensor forward
            prev_img_t = curr_img_t
            obs = next_obs
            self.total_steps += 1
            pbar.update(1)
            
            # 8. Handle episode end
            if done:
                episode_count += 1
                
                # Check success: object near goal
                obj_pos = obs['object_pos_world']
                goal_pos = obs['goal_pos_world']
                if np.linalg.norm(obj_pos - goal_pos) < 0.05:
                    success_count += 1
                
                # Reset for next episode and pre-transform new images
                self.env.reset()
                obs = self.env.get_expert_obs()
                self.expert.reset()
                goal_img_np = render_goal_image(self.env, obs['goal_pos_world'])
                goal_img_t = self.transform(Image.fromarray(goal_img_np))
                prev_img_np = obs['image_primary'].copy()
                prev_img_t = self.transform(Image.fromarray(prev_img_np))
        
        pbar.close()
        
        return {
            "n_episodes": episode_count,
            "success_rate": success_count / max(episode_count, 1),
            "buffer_size": len(self.buffer)
        }
    
    def update_policy(self) -> Dict[str, float]:
        """
        Updates policy using imitation loss from expert demonstrations.
        
        Loss = MSE(policy_pose, expert_pose) + BCE(policy_gripper, expert_gripper)
        """
        self.policy.train()
        
        buffer_size = len(self.buffer)
        batch_size = min(self.cfg.training.batch_size, buffer_size)
        epochs = self.cfg.training.epochs_per_iter
        
        total_pose_loss = 0.0
        total_grip_loss = 0.0
        total_loss = 0.0
        n_updates = 0
        
        for epoch in range(epochs):
            # Shuffle indices each epoch
            indices = np.random.permutation(buffer_size)
            
            for start_idx in range(0, buffer_size - batch_size + 1, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                # Prepare batch
                batch, expert_poses, expert_grippers = self._prepare_training_batch(batch_indices)
                
                # Forward pass
                policy_out = self.policy(batch)
                
                # Extract first step of chunk (7D pose)
                policy_pose = policy_out['pose_chunk'][:, 0]  # (B, 7)
                policy_grip = policy_out['gripper_chunk'][:, 0]  # (B, 1)
                
                # Compute pose loss with position/orientation weighting
                pos_weight = self.cfg.loss.get('position_weight', 1.0)
                orn_weight = self.cfg.loss.get('orientation_weight', 0.3)
                
                pos_loss = F.mse_loss(policy_pose[:, :3], expert_poses[:, :3])
                orn_loss = F.mse_loss(policy_pose[:, 3:], expert_poses[:, 3:])
                pose_loss = pos_weight * pos_loss + orn_weight * orn_loss
                
                # Gripper loss (BCE - treating as classification)
                grip_loss = F.binary_cross_entropy_with_logits(policy_grip, expert_grippers)
                
                # Total loss
                loss = (
                    self.cfg.loss.pose_weight * pose_loss + 
                    self.cfg.loss.gripper_weight * grip_loss
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.cfg.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), 
                        self.cfg.training.max_grad_norm
                    )
                
                self.optimizer.step()
                
                # Accumulate stats
                total_pose_loss += pose_loss.item()
                total_grip_loss += grip_loss.item()
                total_loss += loss.item()
                n_updates += 1
        
        return {
            "pose_loss": total_pose_loss / max(n_updates, 1),
            "grip_loss": total_grip_loss / max(n_updates, 1),
            "total_loss": total_loss / max(n_updates, 1),
            "n_updates": n_updates
        }
    
    def save_checkpoint(self, path: str):
        """Saves model checkpoint (compatible with evaluate_dgpo.py)."""
        checkpoint = {
            "iteration": self.iteration,
            "total_steps": self.total_steps,
            "policy_state_dict": self.policy.state_dict(),  # Match DGPO format
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
        log.info(f"Saved checkpoint: {path}")
    
    def train(self):
        """Main training loop."""
        log.info("=" * 60)
        log.info("Starting DAgger Training")
        log.info("=" * 60)
        log.info(f"Total iterations: {self.cfg.training.total_iterations}")
        log.info(f"Steps per iteration: {self.cfg.training.steps_per_iter}")
        
        for self.iteration in range(self.cfg.training.total_iterations):
            # 1. Collect rollouts (expert executes)
            rollout_stats = self.collect_rollouts(self.cfg.training.steps_per_iter)
            
            # 2. Update policy (imitation learning)
            update_stats = self.update_policy()
            
            # 3. Log
            log.info(
                f"Iter {self.iteration:4d} | "
                f"Eps: {rollout_stats['n_episodes']:2d} | "
                f"Succ: {rollout_stats['success_rate']:.1%} | "
                f"Buf: {rollout_stats['buffer_size']} | "
                f"PL: {update_stats['pose_loss']:.4f} | "
                f"GL: {update_stats['grip_loss']:.4f}"
            )
            
            # 4. Save checkpoint
            if (self.iteration + 1) % self.cfg.training.save_freq == 0:
                ckpt_path = os.path.join(
                    self.cfg.training.output_dir,
                    f"dagger_iter_{self.iteration:04d}.pt"
                )
                self.save_checkpoint(ckpt_path)
        
        # Final save
        final_path = os.path.join(self.cfg.training.output_dir, "dagger_final.pt")
        self.save_checkpoint(final_path)
        log.info("Training complete!")


# ==============================================================================
# 4. MAIN ENTRY POINT
# ==============================================================================

@hydra.main(version_base=None, config_path="../configs", config_name="train_dagger_config")
def main(cfg: DictConfig):
    log.info("=" * 60)
    log.info("DAgger-Style Training")
    log.info("=" * 60)
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    trainer = DAggerTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
