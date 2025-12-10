# FILE: train/train_dgpo.py
"""
DGPO-Foundation Trainer (Dense Expert-Guided PPO Fine-Tuning)

This script implements the DGPO-Foundation algorithm for post-training a 
pre-trained SemanticPlanner policy using:
1. Dense per-step divergence rewards from a ScriptedExpert
2. PPO for policy optimization
3. GAE for advantage estimation

The key innovation is using the expert's recommended EE pose at each step
to provide immediate feedback to the policy, solving the credit assignment problem.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import hydra
import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# --- Robust Path Injection ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.panda_env import PandaEnv
from models.semantic_planner import SemanticPlanner, SemanticPlannerConfig
from train.train_semantic_planner import SemanticPlannerLightningModule
from utils.ik_solver import IKSolver
from utils.divergence import compute_step_divergence
from utils.scripted_expert import ScriptedExpert, ObjectProfile, ExpertConfig

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("DGPO")


# ==============================================================================
# 1. VALUE NETWORK
# ==============================================================================

class ValueNetwork(nn.Module):
    """Critic network for PPO value prediction."""
    
    def __init__(self, proprio_dim: int = 22, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.net(proprio).squeeze(-1)


# ==============================================================================
# 2. ROLLOUT BUFFER
# ==============================================================================

@dataclass
class RolloutBuffer:
    """Stores rollout data for PPO updates."""
    
    # Observation components
    prev_images: List[np.ndarray] = field(default_factory=list)
    curr_images: List[np.ndarray] = field(default_factory=list)
    goal_images: List[np.ndarray] = field(default_factory=list)
    proprios: List[np.ndarray] = field(default_factory=list)
    
    # Actions and probs
    actions: List[np.ndarray] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    
    # Rewards and values
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    
    # For divergence calculation
    achieved_ee_poses: List[np.ndarray] = field(default_factory=list)
    expert_ee_poses: List[np.ndarray] = field(default_factory=list)
    
    def add(
        self,
        prev_img: np.ndarray,
        curr_img: np.ndarray,
        goal_img: np.ndarray,
        proprio: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        achieved_ee: np.ndarray,
        expert_ee: np.ndarray,
    ):
        self.prev_images.append(prev_img)
        self.curr_images.append(curr_img)
        self.goal_images.append(goal_img)
        self.proprios.append(proprio)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.achieved_ee_poses.append(achieved_ee)
        self.expert_ee_poses.append(expert_ee)
    
    def clear(self):
        for attr in self.__dataclass_fields__:
            getattr(self, attr).clear()
    
    def __len__(self):
        return len(self.rewards)


# ==============================================================================
# 3. GAE COMPUTATION
# ==============================================================================

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)
    
    last_gae = 0.0
    last_value = 0.0
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]
        
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        advantages[t] = last_gae
        returns[t] = advantages[t] + values[t]
    
    return advantages, returns


# ==============================================================================
# 4. GOAL IMAGE RENDERING (From Evaluation Script)
# ==============================================================================

def render_goal_image(env: PandaEnv, goal_pos: np.ndarray) -> np.ndarray:
    """Renders the goal image by teleporting object to goal position."""
    saved_qpos = env.data.qpos.copy()
    saved_qvel = env.data.qvel.copy()
    
    try:
        obj_addr = env.model.jnt_qposadr[env.object_joint_id]
        env.data.qpos[obj_addr : obj_addr + 3] = goal_pos
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        goal_img = env.render()
    finally:
        env.data.qpos[:] = saved_qpos
        env.data.qvel[:] = saved_qvel
        mujoco.mj_forward(env.model, env.data)
    
    return goal_img


# ==============================================================================
# 5. DGPO TRAINER
# ==============================================================================

class DGPOTrainer:
    """
    DGPO-Foundation Trainer implementing dense divergence-guided PPO.
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
        
        # 2. Initialize Value Network
        self.value_net = ValueNetwork(
            proprio_dim=cfg.model.proprio_dim,
            hidden_dim=cfg.value_net.hidden_dim
        ).to(self.device)
        
        # 3. Initialize Environment
        self.env = PandaEnv(
            xml_path=cfg.environment.xml_path,
            control_mode="delta",
            render_mode="rgb_array"
        )
        
        # 4. Initialize IK Solver
        self.ik_solver = IKSolver(urdf_path=cfg.environment.urdf_path)
        
        # 5. Initialize Scripted Expert
        self.object_profile = ObjectProfile(
            size=np.array(cfg.expert.object_size),
            grasp_width_normalized=cfg.expert.grasp_width
        )
        self.expert = ScriptedExpert(
            object_profile=self.object_profile,
            cfg=ExpertConfig()
        )
        
        # 6. Control calibration
        SIM_SUBSTEPS = 20
        self.effective_dt = self.env.model.opt.timestep * SIM_SUBSTEPS
        self.max_dq = self.env.ACTION_SCALING_FACTOR / self.effective_dt
        
        # 7. Optimizers
        policy_params = [p for p in self.policy.parameters() if p.requires_grad]
        self.policy_optimizer = torch.optim.Adam(
            policy_params, lr=cfg.optimizer.policy_lr
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=cfg.optimizer.value_lr
        )
        
        # 8. Image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor()
        ])
        
        # 9. Rollout buffer
        self.buffer = RolloutBuffer()
        
        # 10. Statistics
        self.iteration = 0
        self.total_steps = 0
        
        log.info("DGPO Trainer initialized.")
    
    def _prepare_batch(
        self,
        prev_img: np.ndarray,
        curr_img: np.ndarray,
        goal_img: np.ndarray,
        proprio: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        """Prepares observation tensors for the policy."""
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
    
    def collect_rollouts(self, n_steps: int) -> Dict[str, float]:
        """
        Collects rollout data for n_steps using policy + divergence rewards.
        """
        self.buffer.clear()
        self.policy.eval()
        
        episode_rewards = []
        episode_successes = []
        current_ep_reward = 0.0
        
        # Reset environment and expert
        self.env.reset()
        obs = self.env.get_expert_obs()
        self.expert.reset()
        
        # Render goal image once per episode
        goal_img = render_goal_image(self.env, obs['goal_pos_world'])
        prev_img = obs['image_primary'].copy()
        
        for step in range(n_steps):
            curr_img = obs['image_primary']
            proprio = obs['proprio']
            
            # 1. Get Expert's recommended pose
            expert_pose, expert_grip, _ = self.expert.get_target_pose(obs)  # Returns (pose, gripper, info)
            
            # 2. Get Policy's predicted pose
            batch = self._prepare_batch(prev_img, curr_img, goal_img, proprio)
            with torch.no_grad():
                policy_out = self.policy(batch)
            
            # Extract first step of pose chunk
            policy_pose = policy_out['pose_chunk'][0, 0].cpu().numpy()  # (7,)
            policy_grip_logit = policy_out['gripper_chunk'][0, 0].cpu().numpy()[0]
            gripper_cmd = -1.0 if policy_grip_logit > 0 else 1.0
            
            # 3. Compute joint action via IK
            current_joints = self.env.data.qpos[:7].copy()
            delta_joints = self.ik_solver.compute_delta_action(
                target_ee_pose=policy_pose,
                model=self.env.model,
                data=self.env.data,
                ee_site_id=self.env.ee_site_id,
                joint_qpos_indices=np.arange(7),
                effective_dt=self.effective_dt,
                max_dq=self.max_dq
            )

            action = np.concatenate([delta_joints, [gripper_cmd]])
            
            # 4. Step environment
            next_obs, base_reward, terminated, truncated, info = self.env.step(action)
            next_obs = self.env.get_expert_obs()
            done = terminated or truncated
            
            # 5. Calculate DENSE divergence reward
            achieved_ee = self.env.get_ee_pose()
            div_penalty = compute_step_divergence(
                achieved_ee, expert_pose,
                position_weight=self.cfg.reward.position_weight,
                orientation_weight=self.cfg.reward.orientation_weight
            )
            
            # 6. Compute total reward
            # Task reward: distance to goal
            obj_pos = next_obs['object_pos_world']
            goal_pos = next_obs['goal_pos_world']
            dist_to_goal = np.linalg.norm(obj_pos - goal_pos)
            
            task_reward = -self.cfg.reward.w_dist * dist_to_goal
            div_reward = -self.cfg.reward.w_div * div_penalty
            
            # Success bonus
            success = dist_to_goal < 0.05
            success_bonus = self.cfg.reward.success_bonus if success else 0.0
            
            total_reward = task_reward + div_reward + success_bonus
            current_ep_reward += total_reward
            
            # 7. Get value estimate
            proprio_t = torch.from_numpy(proprio).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                value = self.value_net(proprio_t).item()
            
            # 8. Store in buffer (log_prob is placeholder for action-chunking policy)
            self.buffer.add(
                prev_img=prev_img,
                curr_img=curr_img,
                goal_img=goal_img,
                proprio=proprio,
                action=action,
                log_prob=0.0,  # Note: For chunking policies, we use MSE loss instead
                reward=total_reward,
                value=value,
                done=done,
                achieved_ee=achieved_ee,
                expert_ee=expert_pose
            )
            
            # 9. Update state
            prev_img = curr_img.copy()
            obs = next_obs
            self.total_steps += 1
            
            # 10. Handle episode end
            if done:
                episode_rewards.append(current_ep_reward)
                episode_successes.append(float(success))
                current_ep_reward = 0.0
                
                # Reset
                self.env.reset()
                obs = self.env.get_expert_obs()
                self.expert.reset()
                goal_img = render_goal_image(self.env, obs['goal_pos_world'])
                prev_img = obs['image_primary'].copy()
        
        self.policy.train()
        
        return {
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "success_rate": np.mean(episode_successes) if episode_successes else 0.0,
            "n_episodes": len(episode_rewards),
            "mean_div": np.mean([
                compute_step_divergence(a, e)
                for a, e in zip(self.buffer.achieved_ee_poses, self.buffer.expert_ee_poses)
            ])
        }
    
    def update_policy(self) -> Dict[str, float]:
        """
        Updates policy using PPO with imitation loss from expert poses.
        
        For action-chunking policies like SemanticPlanner, we use an imitation-style
        loss instead of standard PPO log-probability ratio, since the policy outputs
        continuous pose predictions rather than action distributions.
        """
        # Compute GAE
        rewards = np.array(self.buffer.rewards)
        values = np.array(self.buffer.values)
        dones = np.array(self.buffer.dones)
        advantages, returns = compute_gae(
            rewards, values, dones,
            gamma=self.cfg.ppo.gamma,
            lam=self.cfg.ppo.gae_lambda
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        advantages_t = torch.from_numpy(advantages).float().to(self.device)
        returns_t = torch.from_numpy(returns).float().to(self.device)
        
        policy_losses = []
        value_losses = []
        
        # PPO epochs
        for epoch in range(self.cfg.ppo.epochs):
            # Mini-batch updates
            indices = np.arange(len(self.buffer))
            np.random.shuffle(indices)
            
            for start in range(0, len(indices), self.cfg.ppo.batch_size):
                end = start + self.cfg.ppo.batch_size
                batch_indices = indices[start:end]
                
                # Prepare mini-batch
                batch_prev = torch.stack([
                    self.transform(Image.fromarray(self.buffer.prev_images[i]))
                    for i in batch_indices
                ]).to(self.device)
                batch_curr = torch.stack([
                    self.transform(Image.fromarray(self.buffer.curr_images[i]))
                    for i in batch_indices
                ]).to(self.device)
                batch_goal = torch.stack([
                    self.transform(Image.fromarray(self.buffer.goal_images[i]))
                    for i in batch_indices
                ]).to(self.device)
                batch_proprio = torch.stack([
                    torch.from_numpy(self.buffer.proprios[i]).float()
                    for i in batch_indices
                ]).to(self.device)
                batch_expert_poses = torch.stack([
                    torch.from_numpy(self.buffer.expert_ee_poses[i][:7]).float()
                    for i in batch_indices
                ]).to(self.device)
                
                batch_adv = advantages_t[batch_indices]
                batch_ret = returns_t[batch_indices]
                
                # Forward pass
                policy_out = self.policy({
                    "prev_image": batch_prev,
                    "curr_image": batch_curr,
                    "goal_image": batch_goal,
                    "curr_proprio": batch_proprio
                })
                
                # Policy update: Advantage-weighted imitation loss
                # L = -A * exp(-MSE(pred, expert))  ≈ weighted regression towards expert
                pred_poses = policy_out['pose_chunk'][:, 0, :]  # (B, 7)
                pose_error = F.mse_loss(pred_poses, batch_expert_poses, reduction='none').mean(dim=1)
                
                # Weighted by advantage (clamp to prevent extreme weights)
                weights = torch.exp(batch_adv.clamp(-10, 10) / self.cfg.ppo.temperature)
                policy_loss = (weights * pose_error).mean()
                
                # Value update
                values_pred = self.value_net(batch_proprio)
                value_loss = F.mse_loss(values_pred, batch_ret)
                
                # Optimize
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                
                total_loss = policy_loss + self.cfg.ppo.value_coef * value_loss
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.policy.parameters() if p.requires_grad],
                    self.cfg.ppo.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.value_net.parameters(),
                    self.cfg.ppo.max_grad_norm
                )
                
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
        
        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses)
        }
    
    def train(self):
        """Main training loop."""
        log.info(f"Starting DGPO training for {self.cfg.training.total_iterations} iterations")
        
        # Create output directory
        output_dir = Path(self.cfg.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for iteration in range(self.cfg.training.total_iterations):
            self.iteration = iteration
            
            # Collect rollouts
            rollout_stats = self.collect_rollouts(self.cfg.training.steps_per_iter)
            
            # Update policy
            update_stats = self.update_policy()
            
            # Logging
            log.info(
                f"Iter {iteration:4d} | "
                f"R: {rollout_stats['mean_reward']:.2f} | "
                f"Succ: {rollout_stats['success_rate']*100:.1f}% | "
                f"Div: {rollout_stats['mean_div']:.4f} | "
                f"PL: {update_stats['policy_loss']:.4f} | "
                f"VL: {update_stats['value_loss']:.4f}"
            )
            
            # Save checkpoint
            if (iteration + 1) % self.cfg.training.save_freq == 0:
                ckpt_path = output_dir / f"dgpo_iter_{iteration+1:04d}.pt"
                torch.save({
                    'iteration': iteration,
                    'policy_state_dict': self.policy.state_dict(),
                    'value_state_dict': self.value_net.state_dict(),
                    'policy_optimizer': self.policy_optimizer.state_dict(),
                    'value_optimizer': self.value_optimizer.state_dict(),
                }, ckpt_path)
                log.info(f"Saved checkpoint: {ckpt_path}")
        
        log.info("Training complete!")


# ==============================================================================
# 6. MAIN ENTRY POINT
# ==============================================================================

@hydra.main(version_base=None, config_path="../configs", config_name="train_dgpo_config")
def main(cfg: DictConfig):
    log.info("=" * 60)
    log.info("DGPO-Foundation Training")
    log.info("=" * 60)
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    trainer = DGPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
