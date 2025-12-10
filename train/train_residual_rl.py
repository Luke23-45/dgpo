# FILE: train/train_residual_rl.py
# (PPO-based Fine-tuning for Residual Policy - SOTA Version)
#
# PURPOSE:
#   This script trains the residual correction network on top of a frozen BC policy
#   using Proximal Policy Optimization (PPO). This enables closed-loop performance
#   by learning from environment interaction.
#
# SOTA FEATURES (2024):
#   - Running observation normalization
#   - Value function clipping
#   - KL divergence monitoring with early stopping
#   - Reward normalization
#   - Gradient norm logging
#
# REFERENCE:
#   - "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
#   - "Implementation Matters in Deep RL" (Engstrom et al., 2020)

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from scipy.spatial.transform import Rotation as R

# --- Project Imports ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from envs.panda_env import PandaEnv
from models.residual_policy import ResidualPolicy, ResidualPolicyConfig, create_residual_policy
from utils.ik_solver import IKSolver
from utils.rl_utils import RunningMeanStd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [ResidualRL] %(message)s"
)
logger = logging.getLogger("ResidualRL")


# ==============================================================================
# 1. RUNNING MEAN STD (for observation normalization)
# ==============================================================================




# ==============================================================================
# 2. VALUE NETWORK (CRITIC)
# ==============================================================================

class ValueNetwork(nn.Module):
    """
    Value function V(s) for PPO advantage estimation.
    
    Input: normalized proprio state (22-dim)
    Output: scalar value estimate
    """
    
    def __init__(self, input_dim: int = 22, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.net(proprio).squeeze(-1)


# ==============================================================================
# 3. REWARD FUNCTION
# ==============================================================================

def compute_reward(
    obj_pos: np.ndarray,
    goal_pos: np.ndarray,
    ee_pos: np.ndarray,
    is_grasped: bool,
    gripper_closed: bool,
    prev_dist_to_goal: Optional[float] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Compute reward for pick-and-place task.
    
    Reward components:
        1. Distance penalty (negative, dense)
        2. Grasp bonus
        3. Success bonus
        4. Anti-hover penalty (prevents staying high)
        5. Progress shaping
    """
    rewards = {}
    
    dist_to_goal = np.linalg.norm(obj_pos - goal_pos)
    dist_ee_to_obj = np.linalg.norm(ee_pos[:3] - obj_pos)
    
    # 1. Base Penalty (Distance)
    rewards['dist_penalty'] = -dist_to_goal
    
    # 2. Grasp & Success Bonuses
    if is_grasped:
        rewards['grasp_bonus'] = 1.0
    elif gripper_closed:
        if dist_ee_to_obj < 0.05:
            rewards['grasp_attempt'] = 0.1
        else:
            rewards['grasp_attempt'] = 0.0
    else:
        rewards['grasp_bonus'] = 0.0
    
    success = dist_to_goal < 0.05
    rewards['success_bonus'] = 10.0 if success else 0.0
    
    # 3. ANTI-HOVER PENALTY
    xy_dist = np.linalg.norm(ee_pos[:2] - obj_pos[:2])
    z_height = ee_pos[2]
    
    if xy_dist < 0.10 and z_height > 0.45 and not is_grasped:
        rewards['hover_penalty'] = -2.0 * (z_height - 0.42)
    else:
        rewards['hover_penalty'] = 0.0

    # 4. Progress Shaping
    if prev_dist_to_goal is not None:
        progress = prev_dist_to_goal - dist_to_goal
        rewards['progress'] = 5.0 * progress 
    else:
        rewards['progress'] = 0.0
    
    total_reward = sum(rewards.values())
    return total_reward, rewards


# ==============================================================================
# 4. ROLLOUT BUFFER (with old_values for value clipping)
# ==============================================================================

@dataclass
class RolloutBuffer:
    """Stores trajectory data for PPO training with SOTA features."""
    
    # Observations (stored as tensors)
    prev_images: List[torch.Tensor] = field(default_factory=list)
    curr_images: List[torch.Tensor] = field(default_factory=list)
    goal_images: List[torch.Tensor] = field(default_factory=list)
    proprios: List[torch.Tensor] = field(default_factory=list)
    proprio_norms: List[torch.Tensor] = field(default_factory=list) # Store specific normalized obs
    
    # Actions and values
    actions: List[torch.Tensor] = field(default_factory=list)
    raw_residuals: List[torch.Tensor] = field(default_factory=list) # Stored raw residuals
    log_probs: List[torch.Tensor] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)
    
    # Rewards and dones
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    
    # For advantage computation
    advantages: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None
    old_values: Optional[torch.Tensor] = None  # SOTA: For value clipping
    
    def add(
        self,
        prev_image: torch.Tensor,
        curr_image: torch.Tensor,
        goal_image: torch.Tensor,
        proprio: torch.Tensor,
        proprio_norm: torch.Tensor,
        action: torch.Tensor,
        raw_residual: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool
    ):
        self.prev_images.append(prev_image.cpu())
        self.curr_images.append(curr_image.cpu())
        self.goal_images.append(goal_image.cpu())
        self.proprios.append(proprio.cpu())
        self.proprio_norms.append(proprio_norm.cpu())
        self.actions.append(action.cpu())
        self.raw_residuals.append(raw_residual.cpu())
        self.log_probs.append(log_prob.cpu())
        self.values.append(value.cpu())
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_rewards: bool = True
    ):
        """Compute GAE advantages and returns with reward normalization."""
        rewards = np.array(self.rewards)
        
        # SOTA: Reward normalization
        if normalize_rewards and len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        dones = np.array(self.dones, dtype=np.float32)
        values = torch.stack(self.values).numpy().flatten() # Flatten to (T,) to match rewards
        
        T = len(rewards)
        advantages = np.zeros(T)
        last_gae = 0
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values
        
        self.advantages = torch.tensor(advantages, dtype=torch.float32)
        self.returns = torch.tensor(returns, dtype=torch.float32)
        self.old_values = torch.tensor(values, dtype=torch.float32)  # Store for value clipping
    
    def get_batches(self, batch_size: int, device: torch.device):
        """Yield mini-batches for PPO update."""
        T = len(self.rewards)
        indices = np.random.permutation(T)
        
        for start in range(0, T, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            yield {
                'prev_image': torch.stack([self.prev_images[i] for i in batch_indices]).to(device),
                'curr_image': torch.stack([self.curr_images[i] for i in batch_indices]).to(device),
                'goal_image': torch.stack([self.goal_images[i] for i in batch_indices]).to(device),
                'curr_proprio': torch.stack([self.proprios[i] for i in batch_indices]).to(device),
                'proprio_norms': torch.stack([self.proprio_norms[i] for i in batch_indices]).to(device),
                'actions': torch.stack([self.actions[i] for i in batch_indices]).to(device),
                'raw_residuals': torch.stack([self.raw_residuals[i] for i in batch_indices]).to(device),
                'old_log_probs': torch.stack([self.log_probs[i] for i in batch_indices]).flatten().to(device),
                'old_values': self.old_values[batch_indices].to(device),  # For value clipping
                'advantages': self.advantages[batch_indices].to(device),
                'returns': self.returns[batch_indices].to(device),
            }
    
    def clear(self):
        self.prev_images.clear()
        self.curr_images.clear()
        self.goal_images.clear()
        self.proprios.clear()
        self.proprio_norms.clear()
        self.actions.clear()
        self.raw_residuals.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.advantages = None
        self.returns = None
        self.old_values = None


# ==============================================================================
# 5. PPO TRAINER (SOTA VERSION)
# ==============================================================================

class ResidualRLTrainer:
    """
    PPO trainer for residual policy fine-tuning with SOTA features.
    
    Features:
        - Running observation normalization
        - Value function clipping
        - KL divergence monitoring
        - Learning rate scheduling
        - Gradient norm logging
    """
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- 1. Create Environment ---
        logger.info("Initializing environment...")
        self.env = PandaEnv(
            xml_path=cfg.env.get("xml_path", "envs/panda_pick_place.xml"),
            control_mode='delta',
            render_mode="rgb_array"
        )
        
        # --- 2. Create Residual Policy ---
        logger.info(f"Loading base policy from: {cfg.checkpoint_path}")
        residual_cfg = ResidualPolicyConfig(
            hidden_dim=cfg.residual.get("hidden_dim", 256),
            num_hidden_layers=cfg.residual.get("num_layers", 2),
            residual_scale=cfg.residual.get("initial_scale", 0.1),
            stochastic=True
        )
        self.policy = create_residual_policy(
            cfg.checkpoint_path,
            device=self.device,
            cfg=residual_cfg
        )
        
        # --- 3. Create Value Network ---
        self.value_net = ValueNetwork(
            input_dim=self.policy.proprio_dim,
            hidden_dim=cfg.value_net.get("hidden_dim", 256),
            num_layers=cfg.value_net.get("num_layers", 3)
        ).to(self.device)
        
        # --- 4. IK Solver ---
        self.ik_solver = IKSolver(urdf_path="urdf/panda_mujoco_kinematics.urdf")
        
        # --- 5. SOTA: Running Statistics for Observation Normalization ---
        self.proprio_normalizer = RunningMeanStd(shape=(self.policy.proprio_dim,))
        
        # --- 6. Optimizers ---
        self.policy_optimizer = torch.optim.Adam(
            self.policy.get_trainable_parameters(),
            lr=cfg.training.get("policy_lr", 1e-4)  # Reduced from 3e-4
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(),
            lr=cfg.training.get("value_lr", 1e-3)
        )
        
        # --- 7. SOTA: Learning Rate Schedulers ---
        self.num_iterations = cfg.training.get("num_iterations", 100)
        self.policy_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.policy_optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=self.num_iterations
        )
        
        # --- 8. Image Transform ---
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # --- 9. PPO Hyperparameters (SOTA tuned) ---
        self.clip_epsilon = cfg.ppo.get("clip_epsilon", 0.2)
        self.entropy_coef = cfg.ppo.get("entropy_coef", 0.005)  # Reduced from 0.01
        self.value_loss_coef = cfg.ppo.get("value_loss_coef", 0.5)
        self.max_grad_norm = cfg.ppo.get("max_grad_norm", 0.5)
        self.ppo_epochs = cfg.ppo.get("epochs", 4)  # Reduced from 10
        self.batch_size = cfg.ppo.get("batch_size", 128)  # Increased from 64
        self.gamma = cfg.ppo.get("gamma", 0.99)
        self.gae_lambda = cfg.ppo.get("gae_lambda", 0.95)
        self.target_kl = cfg.ppo.get("target_kl", 0.015)  # KL early stopping threshold
        
        # --- 10. Training Config ---
        self.steps_per_iteration = cfg.training.get("steps_per_iteration", 4096)  # Increased
        self.max_episode_steps = cfg.training.get("max_episode_steps", 200)
        
        # --- 11. Buffers ---
        self.buffer = RolloutBuffer()
        self.prev_img_buffer = None
        
        logger.info(f"ResidualRLTrainer (SOTA) initialized on {self.device}")
        logger.info(f"Policy trainable params: {self.policy.trainable_param_count()}")
        logger.info(f"PPO epochs: {self.ppo_epochs}, Batch size: {self.batch_size}")
    
    def _get_proprio_exact(self) -> np.ndarray:
        """Get proprio in exact training format."""
        joint_qpos = self.env.data.qpos[:7].copy()
        joint_qvel = self.env.data.qvel[:7].copy()
        
        left_touch = self.env.data.sensordata[self.env.left_touch_sensor_id]
        right_touch = self.env.data.sensordata[self.env.right_touch_sensor_id]
        
        left_force_adr = self.env.model.sensor_adr[self.env.left_force_sensor_id]
        right_force_adr = self.env.model.sensor_adr[self.env.right_force_sensor_id]
        left_force = self.env.data.sensordata[left_force_adr : left_force_adr + 3]
        right_force = self.env.data.sensordata[right_force_adr : right_force_adr + 3]
        
        proprio = np.concatenate([
            joint_qpos,
            joint_qvel,
            np.array([left_touch, right_touch]),
            left_force,
            right_force
        ]).astype(np.float32)
        
        return proprio
    
    def _get_normalized_proprio(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get proprio and normalized version."""
        proprio_raw = self._get_proprio_exact()
        self.proprio_normalizer.update(proprio_raw)
        proprio_norm = self.proprio_normalizer.normalize(proprio_raw)
        return proprio_raw, proprio_norm
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get full observation from environment."""
        image = self.env.render()
        proprio_raw, proprio_norm = self._get_normalized_proprio()
        
        ee_pos = self.env.data.site_xpos[self.env.ee_site_id].copy()
        
        obj_jnt_adr = self.env.model.jnt_qposadr[self.env.object_joint_id]
        obj_pos = self.env.data.qpos[obj_jnt_adr : obj_jnt_adr + 3].copy()
        
        goal_pos = self.env.get_goal_pos_expert()
        
        is_grasped = getattr(self.env, '_is_physically_grasped', False)
        
        return {
            'image': image,
            'proprio': proprio_raw,          # Raw for BC policy
            'proprio_norm': proprio_norm,    # Normalized for value net
            'ee_pos': ee_pos,
            'obj_pos': obj_pos,
            'goal_pos': goal_pos,
            'is_grasped': is_grasped
        }
    
    def _render_goal_image(self, goal_pos: np.ndarray) -> np.ndarray:
        """Render goal image with object at goal."""
        import mujoco
        
        saved_qpos = self.env.data.qpos.copy()
        saved_qvel = self.env.data.qvel.copy()
        
        obj_jnt_adr = self.env.model.jnt_qposadr[self.env.object_joint_id]
        self.env.data.qpos[obj_jnt_adr : obj_jnt_adr + 3] = goal_pos
        
        home_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.env.data.qpos[:7] = home_qpos
        self.env.data.qpos[7:9] = 0.04
        
        self.env.data.qvel[:] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)
        
        goal_image = self.env.render()
        
        self.env.data.qpos[:] = saved_qpos
        self.env.data.qvel[:] = saved_qvel
        mujoco.mj_forward(self.env.model, self.env.data)
        
        return goal_image
    
    def collect_rollouts(self) -> Dict[str, float]:
        """Collect trajectories using current policy."""
        self.policy.eval()
        self.buffer.clear()
        
        total_reward = 0.0
        num_episodes = 0
        episode_lengths = []
        successes = 0
        
        steps_collected = 0
        
        while steps_collected < self.steps_per_iteration:
            obs_dict, _ = self.env.reset()
            obs = self._get_observation()
            
            goal_img = self._render_goal_image(obs['goal_pos'])
            goal_tensor = self.transform(Image.fromarray(goal_img)).unsqueeze(0).to(self.device)
            
            curr_tensor = self.transform(Image.fromarray(obs['image'])).unsqueeze(0).to(self.device)
            self.prev_img_buffer = curr_tensor.clone()
            
            episode_reward = 0.0
            prev_dist = np.linalg.norm(obs['obj_pos'] - obs['goal_pos'])
            
            for step in range(self.max_episode_steps):
                curr_tensor = self.transform(Image.fromarray(obs['image'])).unsqueeze(0).to(self.device)
                
                # Use RAW proprio for BC policy (it was trained on raw)
                proprio_tensor = torch.from_numpy(obs['proprio']).float().unsqueeze(0).to(self.device)
                # Use NORMALIZED proprio for value network
                proprio_norm_tensor = torch.from_numpy(obs['proprio_norm']).float().unsqueeze(0).to(self.device)
                
                batch = {
                    'prev_image': self.prev_img_buffer,
                    'curr_image': curr_tensor,
                    'goal_image': goal_tensor,
                    'curr_proprio': proprio_tensor,
                    'proprio_norm': proprio_norm_tensor
                }
                
                with torch.no_grad():
                    action, log_prob, raw_residual = self.policy.get_action(batch, deterministic=False)
                    value = self.value_net(proprio_norm_tensor)
                
                pose_7d = action[0, :7].cpu().numpy()
                grip_logit = action[0, 7].cpu().item()
                
                gripper_closed = grip_logit > 0
                gripper_qpos = 0.0 if gripper_closed else 0.04
                
                try:
                    target_joints = self.ik_solver._get_target_joint_angles(
                        target_pose_7d=pose_7d,
                        current_joint_angles=self.env.data.qpos[:7],
                        max_iter=30
                    )
                    if target_joints is not None:
                        delta = target_joints - self.env.data.qpos[:7]
                        delta = delta / self.env.ACTION_SCALING_FACTOR
                        delta = np.clip(delta, -0.1, 0.1)
                        full_action = np.concatenate([delta, [gripper_qpos]])
                        
                        _, _, terminated, truncated, _ = self.env.step(full_action)
                        done = terminated or truncated
                    else:
                        done = False
                except Exception:
                    done = False
                
                new_obs = self._get_observation()
                
                reward, reward_info = compute_reward(
                    obj_pos=new_obs['obj_pos'],
                    goal_pos=new_obs['goal_pos'],
                    ee_pos=new_obs['ee_pos'],
                    is_grasped=new_obs['is_grasped'],
                    gripper_closed=gripper_closed,
                    prev_dist_to_goal=prev_dist
                )
                
                prev_dist = np.linalg.norm(new_obs['obj_pos'] - new_obs['goal_pos'])
                
                if prev_dist < 0.05:
                    successes += 1
                    done = True
                
                # Store normalized proprio for value network consistency
                self.buffer.add(
                    prev_image=self.prev_img_buffer.squeeze(0),
                    curr_image=curr_tensor.squeeze(0),
                    goal_image=goal_tensor.squeeze(0),
                    proprio=proprio_tensor.squeeze(0),
                    proprio_norm=proprio_norm_tensor.squeeze(0), # STORED NORM
                    action=action.squeeze(0),
                    raw_residual=raw_residual.squeeze(0),
                    log_prob=log_prob,
                    value=value,
                    reward=reward,
                    done=done
                )
                
                episode_reward += reward
                steps_collected += 1
                
                self.prev_img_buffer = curr_tensor.clone()
                obs = new_obs
                
                if done or steps_collected >= self.steps_per_iteration:
                    break
            
            total_reward += episode_reward
            num_episodes += 1
            episode_lengths.append(step + 1)
        
        with torch.no_grad():
            _, proprio_norm = self._get_normalized_proprio()
            proprio_tensor = torch.from_numpy(proprio_norm).float().unsqueeze(0).to(self.device)
            last_value = self.value_net(proprio_tensor).item()
        
        self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda,
            normalize_rewards=True
        )
        
        return {
            'avg_reward': total_reward / num_episodes if num_episodes > 0 else 0,
            'avg_episode_length': np.mean(episode_lengths) if episode_lengths else 0,
            'success_rate': successes / num_episodes if num_episodes > 0 else 0,
            'num_episodes': num_episodes
        }

    def update_policy(self) -> Dict[str, float]:
        """Perform PPO update with SOTA features."""
        self.policy.train()
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        num_updates = 0
        early_stop = False
        
        for epoch in range(self.ppo_epochs):
            if early_stop:
                break
                
            for batch in self.buffer.get_batches(self.batch_size, self.device):
                # Use stored normalized proprio for consistency
                curr_proprio_norm = batch['proprio_norms']

                new_log_prob, entropy = self.policy.evaluate_actions(
                    batch={
                        'prev_image': batch['prev_image'],
                        'curr_image': batch['curr_image'],
                        'goal_image': batch['goal_image'],
                        'curr_proprio': batch['curr_proprio'],
                        'proprio_norm': curr_proprio_norm
                    },
                    raw_residuals=batch['raw_residuals']
                )
                
                # SOTA: Compute approx KL for early stopping
                approx_kl = (batch['old_log_probs'] - new_log_prob).mean().item()
                if approx_kl > self.target_kl:
                    logger.info(f"KL divergence {approx_kl:.4f} > target {self.target_kl}, stopping epoch {epoch}")
                    early_stop = True
                    break
                
                # Value prediction (Use normalized proprio)
                values = self.value_net(curr_proprio_norm)
                
                # Normalize advantages (per-batch)
                advantages = batch['advantages']
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Policy loss (PPO clip)
                ratio = torch.exp(new_log_prob - batch['old_log_probs'])
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # SOTA: Value loss with clipping
                values_clipped = batch['old_values'] + torch.clamp(
                    values - batch['old_values'],
                    -self.clip_epsilon,
                    self.clip_epsilon
                )
                value_loss_unclipped = (values - batch['returns']).pow(2)
                value_loss_clipped = (values_clipped - batch['returns']).pow(2)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                
                # Entropy bonus
                entropy_val = entropy.mean() if entropy.dim() > 0 else entropy
                entropy_loss = -entropy_val
                
                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                policy_grad_norm = nn.utils.clip_grad_norm_(
                    self.policy.get_trainable_parameters(), self.max_grad_norm
                )
                value_grad_norm = nn.utils.clip_grad_norm_(
                    self.value_net.parameters(), self.max_grad_norm
                )
                
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_val.item()
                total_kl += approx_kl
                num_updates += 1
                
                # DEBUG PROBE
                if epoch == 0 and num_updates == 1:
                     logger.info(f"DEBUG: PGradNorm: {policy_grad_norm:.4f}, VGradNorm: {value_grad_norm:.4f}, "
                                 f"Adv Mean: {advantages.mean():.4f}, Adv Std: {advantages.std():.4f}, "
                                 f"KL: {approx_kl:.6f}, Ratio Mean: {ratio.mean():.4f}")
        
        # Step LR scheduler
        self.policy_scheduler.step()
        
        return {
            'policy_loss': total_policy_loss / num_updates if num_updates > 0 else 0,
            'value_loss': total_value_loss / num_updates if num_updates > 0 else 0,
            'entropy': total_entropy / num_updates if num_updates > 0 else 0,
            'approx_kl': total_kl / num_updates if num_updates > 0 else 0,
            'lr': self.policy_scheduler.get_last_lr()[0]
        }
    
    def train(self):
        """Main training loop."""
        logger.info(f"Starting Residual RL training (SOTA) for {self.num_iterations} iterations")
        
        best_success_rate = 0.0
        
        for iteration in range(self.num_iterations):
            t0 = time.time()
            
            rollout_stats = self.collect_rollouts()
            update_stats = self.update_policy()
            
            elapsed = time.time() - t0
            
            logger.info(
                f"Iter {iteration:4d} | "
                f"R: {rollout_stats['avg_reward']:.2f} | "
                f"Succ: {rollout_stats['success_rate']*100:.1f}% | "
                f"PL: {update_stats['policy_loss']:.4f} | "
                f"VL: {update_stats['value_loss']:.4f} | "
                f"KL: {update_stats['approx_kl']:.4f} | "
                f"LR: {update_stats['lr']:.2e} | "
                f"T: {elapsed:.1f}s"
            )
            
            if rollout_stats['success_rate'] > best_success_rate:
                best_success_rate = rollout_stats['success_rate']
                self._save_checkpoint("best_residual_policy.pt")
                logger.info(f"New best success rate: {best_success_rate*100:.1f}%")
            
            if (iteration + 1) % 10 == 0:
                self._save_checkpoint(f"residual_policy_iter_{iteration+1}.pt")
        
        logger.info(f"Training complete. Best success rate: {best_success_rate*100:.1f}%")
    
    def _save_checkpoint(self, filename: str):
        """Save policy checkpoint."""
        out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        path = out_dir / filename
        
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'proprio_normalizer_mean': self.proprio_normalizer.mean,
            'proprio_normalizer_var': self.proprio_normalizer.var,
            'proprio_normalizer_count': self.proprio_normalizer.count,
        }, path)
        
        logger.info(f"Saved checkpoint: {path}")


# ==============================================================================
# 6. MAIN
# ==============================================================================

@hydra.main(version_base=None, config_path="../configs", config_name="train_residual_rl_config")
def main(cfg: DictConfig):
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    trainer = ResidualRLTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
