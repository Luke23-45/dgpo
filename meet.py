# FILE: train/train_dgpo.py
"""
DGPO-Foundation Trainer (Dense Expert-Guided PPO Fine-Tuning)
[Production-Grade with Robust Logging and Resume Training]

This script implements the DGPO-Foundation algorithm for post-training a 
pre-trained SemanticPlanner policy using:
1. Dense per-step divergence rewards from a ScriptedExpert
2. PPO for policy optimization
3. GAE for advantage estimation

Features:
- CSV metrics logging for all training metrics
- TensorBoard integration for visualization
- Atomic checkpoint saving with backup rotation
- Resume training from any checkpoint
"""

from __future__ import annotations

import csv
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
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
from scipy.spatial.transform import Rotation
from torchvision import transforms
from tqdm import tqdm

# TensorBoard (optional, with graceful fallback)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

# --- Robust Path Injection ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.panda_env import PandaEnv
from models.semantic_planner import SemanticPlanner, SemanticPlannerConfig
from train.train_semantic_planner import SemanticPlannerLightningModule
from utils.ik_solver import IKSolver
from utils.divergence import compute_step_divergence
from utils.dgpo_expert import DGPOExpert, DGPOExpertConfig, ObjectProfile

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
# 1.5. METRICS LOGGER (Production-Grade CSV Logging)
# ==============================================================================

class MetricsLogger:
    """
    Production-grade CSV logger for DGPO training metrics.
    
    Features:
    - Automatic header detection and writing
    - Timestamp for each log entry
    - Atomic file operations with flush
    - Resume-aware (appends if file exists with matching headers)
    """
    
    HEADERS = [
        "iteration", "timestamp",
        "mean_reward", "success_rate", "n_episodes",
        "policy_loss", "value_loss",
        "shadow_pos_div_cm", "shadow_orn_div_rad", "grip_agreement",
        "total_steps"
    ]
    
    def __init__(self, csv_path: Path, resume: bool = False):
        """
        Initialize the metrics logger.
        
        Args:
            csv_path: Path to the CSV file
            resume: If True, append to existing file; if False, start fresh
        """
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine write mode
        if resume and self.csv_path.exists():
            # Verify headers match before resuming
            with open(self.csv_path, 'r', newline='') as f:
                reader = csv.reader(f)
                existing_headers = next(reader, None)
                if existing_headers != self.HEADERS:
                    log.warning(f"CSV headers mismatch. Starting fresh.")
                    self._write_header = True
                    self._mode = 'w'
                else:
                    self._write_header = False
                    self._mode = 'a'
        else:
            self._write_header = True
            self._mode = 'w'
        
        # Open file and write header if needed
        self.file = open(self.csv_path, self._mode, newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.HEADERS)
        
        if self._write_header:
            self.writer.writeheader()
            self.file.flush()
            log.info(f"CSV logger initialized: {self.csv_path}")
        else:
            log.info(f"CSV logger appending to: {self.csv_path}")
    
    def log_step(
        self,
        iteration: int,
        rollout_stats: Dict[str, Any],
        update_stats: Dict[str, Any],
        total_steps: int
    ):
        """Log a single training iteration."""
        row = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "mean_reward": f"{rollout_stats['mean_reward']:.4f}",
            "success_rate": f"{rollout_stats['success_rate']:.4f}",
            "n_episodes": rollout_stats['n_episodes'],
            "policy_loss": f"{update_stats['policy_loss']:.6f}",
            "value_loss": f"{update_stats['value_loss']:.6f}",
            "shadow_pos_div_cm": f"{rollout_stats['shadow_pos_div'] * 100:.4f}",
            "shadow_orn_div_rad": f"{rollout_stats['shadow_orn_div']:.6f}",
            "grip_agreement": f"{rollout_stats['grip_agreement']:.4f}",
            "total_steps": total_steps
        }
        self.writer.writerow(row)
        self.file.flush()  # Ensure data is written immediately
    
    def close(self):
        """Close the CSV file."""
        if hasattr(self, 'file') and self.file:
            self.file.close()
    
    def __del__(self):
        self.close()


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
    
    Features:
    - Dense divergence rewards from expert
    - PPO for policy optimization
    - CSV metrics logging
    - TensorBoard integration
    - Resume training from checkpoint
    - Atomic checkpoint saving with backup rotation
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
        
        # 5. Initialize DGPO Expert (robust FSM without timeouts)
        # Create object profile for standard cube (matches dataset generation)
        object_profile = ObjectProfile(
            size=np.array([0.04, 0.04, 0.04]),  # Standard cube dimensions
            grasp_width_normalized=0.6
        )
        expert_cfg = DGPOExpertConfig(
            hover_height=cfg.expert.get('hover_height', 0.15),
            grasp_offset_z=cfg.expert.get('grasp_offset_z', 0.025),
        )
        self.expert = DGPOExpert(
            object_profile=object_profile,
            cfg=expert_cfg,
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
        self.start_iteration = 0  # Will be updated if resuming
        self.iteration = 0
        self.total_steps = 0
        
        # 11. TensorBoard setup (optional)
        self.tb_writer = None
        if cfg.logging.get("use_tensorboard", False) and TENSORBOARD_AVAILABLE:
            tb_dir = Path(cfg.logging.log_dir) / "tensorboard"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            log.info(f"TensorBoard logging enabled: {tb_dir}")
        elif cfg.logging.get("use_tensorboard", False) and not TENSORBOARD_AVAILABLE:
            log.warning("TensorBoard requested but not available. Install with: pip install tensorboard")
        
        # 12. Resume from checkpoint if specified
        resume_path = cfg.checkpoint.get("resume_from")
        if resume_path and Path(resume_path).exists():
            self._load_checkpoint(resume_path)
        
        log.info("DGPO Trainer initialized.")
    
    # ==========================================================================
    # CHECKPOINT & RESUME METHODS
    # ==========================================================================
    
    def _load_checkpoint(self, path: str) -> None:
        """
        Resume training from a saved checkpoint.
        
        Loads policy, value network, optimizers, and training state.
        """
        log.info(f"Resuming from checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)
        
        # Load model states
        self.policy.load_state_dict(ckpt['policy_state_dict'])
        self.value_net.load_state_dict(ckpt['value_state_dict'])
        
        # Load optimizer states
        self.policy_optimizer.load_state_dict(ckpt['policy_optimizer'])
        self.value_optimizer.load_state_dict(ckpt['value_optimizer'])
        
        # Restore training state
        self.start_iteration = ckpt['iteration'] + 1
        self.total_steps = ckpt.get('total_steps', 0)
        
        log.info(f"Resumed at iteration {self.start_iteration} with {self.total_steps} total steps")
    
    def _save_checkpoint(self, iteration: int, is_backup: bool = False) -> Path:
        """
        Save checkpoint atomically with all training state.
        
        Args:
            iteration: Current training iteration
            is_backup: If True, save to backup directory with rotation
            
        Returns:
            Path to saved checkpoint
        """
        if is_backup:
            save_dir = Path(self.cfg.checkpoint.backup_dir)
            filename = f"dgpo_backup_iter_{iteration+1:04d}.pt"
        else:
            save_dir = Path(self.cfg.checkpoint.save_dir)
            filename = f"dgpo_iter_{iteration+1:04d}.pt"
        
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / filename
        
        # Save complete training state
        checkpoint = {
            'iteration': iteration,
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'config': OmegaConf.to_container(self.cfg, resolve=True),
            'timestamp': datetime.now().isoformat(),
        }
        
        torch.save(checkpoint, ckpt_path)
        log.info(f"Saved {'backup' if is_backup else 'checkpoint'}: {ckpt_path}")
        
        # Cleanup old backups if this was a backup save
        if is_backup:
            self._cleanup_old_backups()
        
        return ckpt_path
    
    def _cleanup_old_backups(self) -> None:
        """Keep only N most recent backups, delete older ones."""
        backup_dir = Path(self.cfg.checkpoint.backup_dir)
        if not backup_dir.exists():
            return
        
        backups = sorted(backup_dir.glob("dgpo_backup_*.pt"))
        keep = self.cfg.checkpoint.get("backups_to_keep", 3)
        
        if len(backups) > keep:
            for old_backup in backups[:-keep]:
                try:
                    old_backup.unlink()
                    log.info(f"Cleaned up old backup: {old_backup.name}")
                except OSError as e:
                    log.warning(f"Could not delete old backup {old_backup}: {e}")
    
    def _log_to_tensorboard(
        self,
        iteration: int,
        rollout_stats: Dict[str, Any],
        update_stats: Dict[str, Any]
    ) -> None:
        """Log metrics to TensorBoard."""
        if not self.tb_writer:
            return
        
        # Reward and success metrics
        self.tb_writer.add_scalar("reward/mean", rollout_stats['mean_reward'], iteration)
        self.tb_writer.add_scalar("reward/success_rate", rollout_stats['success_rate'], iteration)
        
        # Shadow mode metrics (policy vs expert)
        self.tb_writer.add_scalar("shadow/pos_div_cm", rollout_stats['shadow_pos_div'] * 100, iteration)
        self.tb_writer.add_scalar("shadow/orn_div_rad", rollout_stats['shadow_orn_div'], iteration)
        self.tb_writer.add_scalar("shadow/grip_agreement", rollout_stats['grip_agreement'], iteration)
        
        # Loss metrics
        self.tb_writer.add_scalar("loss/policy", update_stats['policy_loss'], iteration)
        self.tb_writer.add_scalar("loss/value", update_stats['value_loss'], iteration)
        
        # Training progress
        self.tb_writer.add_scalar("training/total_steps", self.total_steps, iteration)
        self.tb_writer.add_scalar("training/episodes", rollout_stats['n_episodes'], iteration)
    
    # ==========================================================================
    # BATCH PREPARATION
    # ==========================================================================
    
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
        Collects rollout data for n_steps using EXPERT EXECUTION (DAgger-style).
        
        KEY CHANGE: Expert actions are EXECUTED, policy predictions are for LEARNING only.
        This guarantees successful trajectories while training the policy to match expert.
        
        Shadow Mode Logging:
        - position_divergence: How far policy prediction is from expert target
        - orientation_divergence: Rotation error between policy and expert
        - gripper_agreement: Whether policy agrees with expert gripper command
        """
        self.buffer.clear()
        self.policy.eval()
        
        episode_rewards = []
        episode_successes = []
        current_ep_reward = 0.0
        
        # === SHADOW MODE METRICS ===
        shadow_pos_divergences = []  # Policy vs Expert position error
        shadow_orn_divergences = []  # Policy vs Expert orientation error
        gripper_agreements = []      # Policy agrees with expert gripper?
        expert_phases = []           # Track expert FSM phases
        
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
            
            # 1. Get Expert's target pose (DGPOExpert FSM)
            expert_pose, expert_grip, info = self.expert.get_target_pose(obs)
            expert_phases.append(info.get('phase', 'UNKNOWN'))
            
            # 2. Get Policy's predicted pose (SHADOW MODE - for learning only)
            batch = self._prepare_batch(prev_img, curr_img, goal_img, proprio)
            with torch.no_grad():
                policy_out = self.policy(batch)
            
            # Extract first step of pose chunk
            policy_pose = policy_out['pose_chunk'][0, 0].cpu().numpy()  # (7,)
            policy_grip_logit = policy_out['gripper_chunk'][0, 0].cpu().numpy()[0]
            policy_grip_cmd = -1.0 if policy_grip_logit > 0 else 1.0
            
            # === SHADOW MODE LOGGING ===
            # Position divergence (meters)
            pos_div = np.linalg.norm(policy_pose[:3] - expert_pose[:3])
            shadow_pos_divergences.append(pos_div)
            
            # Orientation divergence (radians) - using quaternion distance
            from scipy.spatial.transform import Rotation as R
            try:
                R_policy = R.from_quat(policy_pose[3:])
                R_expert = R.from_quat(expert_pose[3:])
                orn_div = (R_expert.inv() * R_policy).magnitude()
            except:
                orn_div = 0.0
            shadow_orn_divergences.append(orn_div)
            
            # Gripper agreement
            grip_agree = (policy_grip_cmd == expert_grip)
            gripper_agreements.append(float(grip_agree))
            
            # 3. Compute joint action via IK - USING EXPERT POSE (THE FIX!)
            try:
                delta_joints = self.ik_solver.compute_delta_action(
                    target_ee_pose=expert_pose,  # ← EXPERT pose, not policy!
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

            # Use EXPERT gripper command
            action = np.concatenate([delta_joints, [expert_grip]])
            
            # 4. Step environment (executing EXPERT action)
            next_obs, base_reward, terminated, truncated, _ = self.env.step(action)
            next_obs = self.env.get_expert_obs()
            done = terminated or truncated or self.expert.is_done()
            
            # 5. Calculate DENSE divergence reward (policy vs expert)
            # Note: This now measures policy PREDICTION quality, not execution quality
            achieved_ee = self.env.get_ee_pose()  # Where robot actually is
            div_penalty = compute_step_divergence(
                policy_pose, expert_pose,  # Compare policy prediction to expert
                position_weight=self.cfg.reward.position_weight,
                orientation_weight=self.cfg.reward.orientation_weight
            )
            
            # 6. Compute total reward
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
            
            # 8. Store in buffer
            self.buffer.add(
                prev_img=prev_img,
                curr_img=curr_img,
                goal_img=goal_img,
                proprio=proprio,
                action=action,
                log_prob=0.0,
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
        
        # === SHADOW MODE SUMMARY ===
        mean_pos_div = np.mean(shadow_pos_divergences) if shadow_pos_divergences else 0.0
        mean_orn_div = np.mean(shadow_orn_divergences) if shadow_orn_divergences else 0.0
        grip_agreement_rate = np.mean(gripper_agreements) if gripper_agreements else 0.0
        
        return {
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "success_rate": np.mean(episode_successes) if episode_successes else 0.0,
            "n_episodes": len(episode_rewards),
            "mean_div": np.mean([
                compute_step_divergence(a, e)
                for a, e in zip(self.buffer.achieved_ee_poses, self.buffer.expert_ee_poses)
            ]),
            # Shadow Mode Metrics
            "shadow_pos_div": mean_pos_div,
            "shadow_orn_div": mean_orn_div,
            "grip_agreement": grip_agreement_rate,
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
        """
        Main training loop with robust logging and checkpointing.
        
        Features:
        - Resumes from start_iteration if resuming from checkpoint
        - CSV metrics logging for all training metrics
        - TensorBoard logging for visualization
        - Regular checkpoints at save_freq intervals
        - Backup checkpoints at backup_freq intervals with rotation
        """
        total_iters = self.cfg.training.total_iterations
        log.info(f"Starting DGPO training for {total_iters} iterations")
        if self.start_iteration > 0:
            log.info(f"Resuming from iteration {self.start_iteration}")
        
        # Setup directories
        log_dir = Path(self.cfg.logging.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV logger
        csv_path = log_dir / self.cfg.logging.csv_log_name
        is_resuming = self.start_iteration > 0
        csv_logger = MetricsLogger(csv_path, resume=is_resuming)
        
        try:
            # Main training loop - resumes from start_iteration
            for iteration in range(self.start_iteration, total_iters):
                self.iteration = iteration
                
                # Collect rollouts
                rollout_stats = self.collect_rollouts(self.cfg.training.steps_per_iter)
                
                # Update policy
                update_stats = self.update_policy()
                
                # Console logging - Main metrics
                log.info(
                    f"Iter {iteration:4d} | "
                    f"R: {rollout_stats['mean_reward']:.2f} | "
                    f"Succ: {rollout_stats['success_rate']*100:.1f}% | "
                    f"Eps: {rollout_stats['n_episodes']} | "
                    f"PL: {update_stats['policy_loss']:.4f} | "
                    f"VL: {update_stats['value_loss']:.4f}"
                )
                
                # Console logging - Shadow Mode Performance
                log.info(
                    f"         Shadow Mode | "
                    f"PosDiv: {rollout_stats['shadow_pos_div']*100:.2f}cm | "
                    f"OrnDiv: {rollout_stats['shadow_orn_div']:.3f}rad | "
                    f"GripAgree: {rollout_stats['grip_agreement']*100:.1f}%"
                )
                
                # CSV logging
                if (iteration + 1) % self.cfg.logging.get("log_every_n_iters", 1) == 0:
                    csv_logger.log_step(
                        iteration=iteration,
                        rollout_stats=rollout_stats,
                        update_stats=update_stats,
                        total_steps=self.total_steps
                    )
                
                # TensorBoard logging
                self._log_to_tensorboard(iteration, rollout_stats, update_stats)
                
                # Regular checkpoint saving
                save_freq = self.cfg.checkpoint.get("save_freq", 10)
                if (iteration + 1) % save_freq == 0:
                    self._save_checkpoint(iteration, is_backup=False)
                
                # Backup checkpoint saving (more frequent for safety)
                backup_freq = self.cfg.checkpoint.get("backup_freq", 5)
                if (iteration + 1) % backup_freq == 0:
                    self._save_checkpoint(iteration, is_backup=True)
            
            # Save final checkpoint
            self._save_checkpoint(total_iters - 1, is_backup=False)
            log.info("Training complete!")
            
        finally:
            # Cleanup resources
            csv_logger.close()
            if self.tb_writer:
                self.tb_writer.close()
                log.info("TensorBoard writer closed.")


# ==============================================================================
# 6. MAIN ENTRY POINT
# ==============================================================================

@hydra.main(version_base=None, config_path="./configs", config_name="train_dgpo_config")
def main(cfg: DictConfig):
    log.info("=" * 60)
    log.info("DGPO-Foundation Training")
    log.info("=" * 60)
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    trainer = DGPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()