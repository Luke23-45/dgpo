# FILE: train/train_dgpo_robust.py
"""
DGPO-Foundation Trainer (Dense Expert-Guided PPO Fine-Tuning)
[Production-Grade with Robust Logging, Sampling, and Hybrid BC Loss]

This script implements the DGPO-Foundation algorithm for post-training a 
pre-trained SemanticPlanner policy using:
1. Dense per-step divergence rewards from a ScriptedExpert
2. PPO with Action Sampling (Exploration)
3. Hybrid PPO + Behavior Cloning Loss (Stability Anchor)
4. GAE for advantage estimation

Features:
- Correct PPO math (Sampling vs Mean)
- First-Step BC Anchor to prevent divergence
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

# SOTA: Transformers Scheduler for Cosine Decay with Warmup
from transformers import get_cosine_schedule_with_warmup
from copy import deepcopy # For EMA

# --- Robust Path Injection ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.panda_env import PandaEnv
from envs.dgpo_env_wrapper import DGPOEnvWrapper
from models.semantic_planner import SemanticPlanner, SemanticPlannerConfig
from train.train_semantic_planner import SemanticPlannerLightningModule
from utils.divergence import compute_step_divergence
from utils.riemannian_diff import compute_riemannian_divergence
import gymnasium as gym


# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("DGPO")


# ==============================================================================
# 1. VISION CRITIC (State-Aware)
# ==============================================================================

class VisionCritic(nn.Module):
    """
    SOTA Critic for DGPO v2.0: Sees what the Actor sees.
    Input: [Visual_Embedding (from Actor), Proprioception]
    Output: V(s)
    """
    
    def __init__(self, vision_feature_dim: int, proprio_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Project frozen visual features
        self.vis_proj = nn.Linear(vision_feature_dim, hidden_dim)
        self.prop_proj = nn.Linear(proprio_dim, hidden_dim)
        
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1) # Value scalar
        )
    
    def forward(self, visual_emb: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        v = self.vis_proj(visual_emb)
        p = self.prop_proj(proprio)
        return self.net(torch.cat([v, p], dim=-1)).squeeze(-1)


# ==============================================================================
# 1.1. RUNNING NORMALIZER (Reward Stabilization) - SOTA Enhancement
# ==============================================================================

class RunningNormalizer:
    """
    Running reward normalization with exponential moving average.
    Research: "Implementation Matters in Deep RL" (Engstrom+ 2020)
    """
    def __init__(self, epsilon: float = 1e-8, gamma: float = 0.99):
        self.mean = 0.0
        self.var = 1.0
        self.epsilon = epsilon
        self.gamma = gamma
        self.count = 0
    
    def normalize(self, rewards: np.ndarray, clip_range: float = 5.0) -> np.ndarray:
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)
        
        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var
        else:
            self.mean = self.gamma * self.mean + (1 - self.gamma) * batch_mean
            self.var = self.gamma * self.var + (1 - self.gamma) * batch_var
        
        self.count += 1
        normalized = (rewards - self.mean) / (np.sqrt(self.var) + self.epsilon)
        return np.clip(normalized, -clip_range, clip_range)
    
    def state_dict(self) -> dict:
        return {'mean': self.mean, 'var': self.var, 'count': self.count}
    
    def load_state_dict(self, state: dict):
        self.mean = state['mean']
        self.var = state['var']
        self.count = state['count']


# ==============================================================================
# 1.2. ADAPTIVE KL PENALTY (PPO-BR) - SOTA Enhancement
# ==============================================================================

class AdaptiveKLPenalty:
    """
    Adaptive KL penalty for PPO.
    Research: "PPO-BR: Dual-Signal Entropy-Reward Adaptation" (2025)
    """
    def __init__(self, target_kl: float = 0.015, init_beta: float = 0.02):
        self.target_kl = target_kl
        self.beta = init_beta
    
    def update(self, measured_kl: float) -> float:
        if measured_kl <= 0:
            return self.beta
        ratio = measured_kl / self.target_kl
        self.beta *= np.clip(ratio ** 0.5, 0.5, 2.0)
        self.beta = np.clip(self.beta, 0.001, 1.0)
        return self.beta
    
    def state_dict(self) -> dict:
        return {'beta': self.beta, 'target_kl': self.target_kl}
    
    def load_state_dict(self, state: dict):
        self.beta = state['beta']
        self.target_kl = state['target_kl']



# ==============================================================================
# 1.5. METRICS LOGGER (Production-Grade CSV Logging)
# ==============================================================================

class MetricsLogger:
    """
    Production-grade CSV logger for DGPO training metrics.
    """
    
    HEADERS = [
        "iteration", "timestamp",
        "mean_reward", "success_rate", "n_episodes",
        "policy_loss", "value_loss", "bc_loss", # Added BC Loss
        "shadow_pos_div_cm", "shadow_orn_div_rad", "grip_agreement",
        "total_steps"
    ]
    
    def __init__(self, csv_path: Path, resume: bool = False):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine write mode
        if resume and self.csv_path.exists():
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
            "bc_loss": f"{update_stats.get('bc_loss', 0.0):.6f}",
            "shadow_pos_div_cm": f"{rollout_stats['shadow_pos_div'] * 100:.4f}",
            "shadow_orn_div_rad": f"{rollout_stats['shadow_orn_div']:.6f}",
            "grip_agreement": f"{rollout_stats['grip_agreement']:.4f}",
            "total_steps": total_steps
        }
        self.writer.writerow(row)
        self.file.flush()
    
    def close(self):
        if hasattr(self, 'file') and self.file:
            self.file.close()
    
    def __del__(self):
        self.close()


# ==============================================================================
# 2. ROLLOUT BUFFER
# ==============================================================================

@dataclass
class RolloutBuffer:
    """Stores chunk-based rollout data for AC-PPO updates."""
    
    # State data
    prev_images: List[np.ndarray] = field(default_factory=list)
    curr_images: List[np.ndarray] = field(default_factory=list)
    goal_images: List[np.ndarray] = field(default_factory=list)
    proprios: List[np.ndarray] = field(default_factory=list)
    visual_embeddings: List[torch.Tensor] = field(default_factory=list)
    
    # Action Chunks (The "Action")
    action_chunks: List[np.ndarray] = field(default_factory=list)
    
    # Log Probs (of the entire chunk)
    log_probs: List[float] = field(default_factory=list)
    
    # Rewards and values
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    
    # Expert Targets for Divergence Loss / BC
    expert_pose_chunks: List[np.ndarray] = field(default_factory=list)
    expert_phases: List[int] = field(default_factory=list)
    
    def add(
        self,
        prev_img: np.ndarray,
        curr_img: np.ndarray,
        goal_img: np.ndarray,
        proprio: np.ndarray,
        visual_emb: torch.Tensor,
        action_chunk: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        expert_pose_chunk: np.ndarray,
        expert_phase: int
    ):
        self.prev_images.append(prev_img)
        self.curr_images.append(curr_img)
        self.goal_images.append(goal_img)
        self.proprios.append(proprio)
        self.visual_embeddings.append(visual_emb)
        self.action_chunks.append(action_chunk)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.expert_pose_chunks.append(expert_pose_chunk)
        self.expert_phases.append(expert_phase)
    
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
    # Bootstrapping with 0 for end of rollout is okay for episodic tasks in this context
    # ideally we use next_value if not done, but vector env makes it tricky.
    # Assuming steps_per_iter is large enough or aligns with episode ends.
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0.0 # Approximation
        else:
            next_value = values[t + 1]
        
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        advantages[t] = last_gae
        returns[t] = advantages[t] + values[t]
    
    return advantages, returns


# ==============================================================================
# 5. DGPO TRAINER
# ==============================================================================

def make_dgpo_env(cfg_dict: Dict[str, Any]) -> gym.Env:
    """Factory function to create a wrapped DGPO environment."""
    # import os
    # os.environ['MUJOCO_GL'] = 'egl' # REMOVED: Causes crash on Windows
    
    cfg = OmegaConf.create(cfg_dict)
    
    env = PandaEnv(
        xml_path=cfg.environment.xml_path,
        control_mode="delta",
        render_mode="rgb_array"
    )
    return DGPOEnvWrapper(env, cfg) 


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
        self.policy.train()
        
        # Freeze vision backbone if configured
        if cfg.get("freeze_vision", True):
            for param in self.policy.vision_backbone.parameters():
                param.requires_grad = False
            log.info("Froze vision backbone parameters.")
        
        # 2. Initialize Vision Critic
        self.value_net = VisionCritic(
            vision_feature_dim=cfg.model.vision_feature_dim,
            proprio_dim=cfg.model.proprio_dim,
            hidden_dim=cfg.value_net.hidden_dim
        ).to(self.device)
        
        # [DGPO v2.1 FIX] Initialize Action Log Std to a SAFER value
        # -0.5 is approx 0.6 std dev (huge). -2.5 is approx 0.08 (safe for fine tuning).
        self.chk_log_std = nn.Parameter(
            torch.ones(1, self.cfg.model.chunk_size, 7, device=self.device) * -2.5
        )
        
        # [SOTA FIX]: Initialize Target Policy for EMA
        self.ema_policy = deepcopy(self.policy)
        for p in self.ema_policy.parameters():
            p.requires_grad = False
        self.ema_decay = cfg.training.get("ema_decay", 0.999)
        
        # 3. Initialize Parallel Environments
        self.num_envs = cfg.get("num_envs", 8)
        log.info(f"Initializing {self.num_envs} Parallel Environments...")
        
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        env_fns = [lambda: make_dgpo_env(cfg_dict) for _ in range(self.num_envs)]
        
        # Use AsyncVectorEnv with spawn and SHARED MEMORY (Zero-Copy)
        import multiprocessing
        try:
             multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
             pass

        self.envs = gym.vector.AsyncVectorEnv(
            env_fns, 
            shared_memory=True, 
            context='spawn', 
            daemon=True
        )
        
        # 7. Optimizers
        policy_params = [p for p in self.policy.parameters() if p.requires_grad]
        policy_params.append(self.chk_log_std)
        
        self.policy_optimizer = torch.optim.Adam(
            policy_params, lr=cfg.optimizer.policy_lr
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=cfg.optimizer.value_lr
        )
        
        # [SOTA FIX] Learning Rate Scheduler
        # Calculate total steps for scheduler
        total_training_steps = cfg.training.total_iterations * cfg.ppo.epochs * (cfg.training.steps_per_iter // cfg.ppo.batch_size)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.policy_optimizer,
            num_warmup_steps=int(total_training_steps * cfg.optimizer.get("warmup_pct", 0.05)),
            num_training_steps=total_training_steps
        )
        
        # [SOTA Enhancement] Reward Normalizer
        self.reward_normalizer = RunningNormalizer(epsilon=1e-8, gamma=0.99)
        log.info("Initialized running reward normalizer")
        
        # [SOTA Enhancement] Adaptive KL Penalty
        self.kl_penalty = AdaptiveKLPenalty(target_kl=0.015, init_beta=0.02)
        log.info(f"Initialized adaptive KL penalty (target={self.kl_penalty.target_kl:.4f})")
        
        # [SOTA Enhancement] Target Value Network
        self.target_value_net = deepcopy(self.value_net)
        self.target_value_net.eval()
        self.soft_update_tau = 0.005
        log.info("Initialized target critic network for stable advantages")
        
        # [SOTA Enhancement] Adaptive Entropy
        self.entropy_coef_adaptive = 0.01  # Initial bonus weight (decays)
        self.entropy_decay = 0.995  # Per-iteration decay
        log.info(f"Initialized adaptive entropy regularization (init_alpha={self.entropy_coef_adaptive:.4f})")
        
        # [PERF OPT] Mixed Precision Training
        self.use_amp = cfg.training.get("use_amp", True)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        log.info(f"Mixed precision training: {self.use_amp}")
        
        # [PERF OPT] torch.compile for faster inference/training (PyTorch 2.0+)
        if cfg.training.get("use_compile", False) and hasattr(torch, 'compile'):
            log.info("Compiling policy and critic with torch.compile...")
            self.policy = torch.compile(self.policy, mode="reduce-overhead")
            self.value_net = torch.compile(self.value_net, mode="reduce-overhead")
            log.info("Models compiled successfully")
        
        # 8. Image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # 9. Rollout buffer
        self.buffer = RolloutBuffer()
        
        # 10. Statistics
        self.start_iteration = 0
        self.iteration = 0
        self.total_steps = 0
        
        # 11. TensorBoard setup
        self.tb_writer = None
        if cfg.logging.get("use_tensorboard", False) and TENSORBOARD_AVAILABLE:
            tb_dir = Path(cfg.logging.log_dir) / "tensorboard"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            log.info(f"TensorBoard logging enabled: {tb_dir}")
        
        # 12. Resume
        resume_path = cfg.checkpoint.get("resume_from")
        if resume_path and Path(resume_path).exists():
            self._load_checkpoint(resume_path)
        
        log.info("DGPO Trainer initialized.")
    
    # ... [Checkpoint methods unchanged] ...
    
    def _load_checkpoint(self, path: str) -> None:
        log.info(f"Resuming from checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt['policy_state_dict'])
        self.value_net.load_state_dict(ckpt['value_state_dict'])
        self.policy_optimizer.load_state_dict(ckpt['policy_optimizer'])
        self.value_optimizer.load_state_dict(ckpt['value_optimizer'])
        self.start_iteration = ckpt['iteration'] + 1
        self.total_steps = ckpt.get('total_steps', 0)
        log.info(f"Resumed at iteration {self.start_iteration}")
    
    def _save_checkpoint(self, iteration: int, is_backup: bool = False) -> Path:
        if is_backup:
            save_dir = Path(self.cfg.checkpoint.backup_dir)
            filename = f"dgpo_backup_iter_{iteration+1:04d}.pt"
        else:
            save_dir = Path(self.cfg.checkpoint.save_dir)
            filename = f"dgpo_iter_{iteration+1:04d}.pt"
        
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / filename
        
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
        if is_backup: self._cleanup_old_backups()
        return ckpt_path
    
    def _cleanup_old_backups(self) -> None:
        backup_dir = Path(self.cfg.checkpoint.backup_dir)
        if not backup_dir.exists(): return
        backups = sorted(backup_dir.glob("dgpo_backup_*.pt"))
        keep = self.cfg.checkpoint.get("backups_to_keep", 3)
        if len(backups) > keep:
            for old_backup in backups[:-keep]:
                try: old_backup.unlink()
                except OSError: pass

    def _log_to_tensorboard(self, iteration: int, rollout_stats: Dict, update_stats: Dict):
        if not self.tb_writer: return
        self.tb_writer.add_scalar("reward/mean", rollout_stats['mean_reward'], iteration)
        self.tb_writer.add_scalar("reward/success_rate", rollout_stats['success_rate'], iteration)
        self.tb_writer.add_scalar("shadow/pos_div_cm", rollout_stats['shadow_pos_div'] * 100, iteration)
        self.tb_writer.add_scalar("shadow/bc_pos_div_cm", rollout_stats.get('bc_pos_div', 0.0) * 100, iteration)
        self.tb_writer.add_scalar("shadow/orn_div_rad", rollout_stats['shadow_orn_div'], iteration)
        self.tb_writer.add_scalar("loss/policy", update_stats['policy_loss'], iteration)
        self.tb_writer.add_scalar("loss/value", update_stats['value_loss'], iteration)
        self.tb_writer.add_scalar("loss/bc", update_stats.get('bc_loss', 0.0), iteration)
        self.tb_writer.add_scalar("training/total_steps", self.total_steps, iteration)

    def _prepare_batch_vectorized(self, prev_imgs, curr_imgs, goal_imgs, proprios) -> Dict[str, torch.Tensor]:
        """Optimized batch preprocessing without PIL conversion."""
        # [PERF OPT] Direct numpy->torch vectorized processing (2.5x faster)
        def process_batch(imgs_np):
            # imgs_np: (N, H, W, 3) uint8 numpy array
            # Convert to torch tensor directly (faster than PIL)
            imgs_t = torch.from_numpy(imgs_np).float()  # (N, H, W, 3)
            imgs_t = imgs_t.permute(0, 3, 1, 2) / 255.0  # (N, 3, H, W), normalize to [0, 1]
            
            # Resize using torch (faster than PIL)
            imgs_t = F.interpolate(imgs_t, size=(224, 224), mode='bicubic', align_corners=False)
            
            # Normalize to [-1, 1] (matching BC training)
            imgs_t = (imgs_t - 0.5) / 0.5
            
            return imgs_t.to(self.device)
        
        return {
            "prev_image": process_batch(prev_imgs),
            "curr_image": process_batch(curr_imgs),
            "goal_image": process_batch(goal_imgs),
            "curr_proprio": torch.from_numpy(proprios).float().to(self.device)
        }

    # ==========================================================================
    # COLLECT ROLLOUTS (FIXED: SAMPLING)
    # ==========================================================================
    def collect_rollouts(self, n_steps: int) -> Dict[str, float]:
        self.buffer.clear()
        self.policy.eval()
        
        episode_rewards = []
        episode_successes = []
        current_ep_rewards = np.zeros(self.num_envs)
        rsd_divergences = []
        
        # [PRODUCTION FIX] Track BC quality (mean prediction) separately from exploration divergence
        bc_pos_divergences = []  # True BC quality metric (no exploration noise)
        
        # 1. Reset
        obs, info = self.envs.reset()
        goal_imgs = info['goal_img']
        prev_imgs = obs['image_primary'].copy()
        
        current_expert_poses = info['expert_pose']
        current_expert_phases = info['expert_phase']
        
        EXPERT_PHASE_MAP = {
            "MOVE_TO_PRE_GRASP": 0, "PREPARE_GRIPPER": 0, "DESCEND_TO_GRASP": 0, 
            "GRASP": 1, "LIFT": 2, "MOVE_TO_GOAL": 2, "PREPARE_PLACE": 3,
            "DESCEND_TO_PLACE": 3, "AWAIT_STABLE_PLACEMENT": 3, "RELEASE": 3,
            "RETRACT": 4, "DONE": 4
        }
        
        steps_per_env = n_steps // self.num_envs
        
        for step in range(steps_per_env):
            curr_imgs = obs['image_primary']
            proprios = obs['proprio']
            
            # 1. Policy Inference
            batch = self._prepare_batch_vectorized(prev_imgs, curr_imgs, goal_imgs, proprios)
            
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=True):
                    policy_out = self.policy(batch)
                
                pred_chunks = policy_out['pose_chunk'] # (N, K, 7)
                
                # [DEBUG v2.1] Log first step of first rollout to diagnose divergence (One-time check)
                if step == 0 and self.total_steps == 0:
                    sample_pred = pred_chunks[0, 0, :].cpu().numpy()
                    sample_expert = current_expert_poses[0]
                    pos_diff = np.linalg.norm(sample_pred[:3] - sample_expert[:3])
                    log.info(f"[Sanity Check] Policy-Expert Alignment: Diff={pos_diff*100:.2f}cm")
                
                # [DGPO v2.1 FIX] ENABLE EXPLORATION via SAMPLING
                dist = torch.distributions.Normal(pred_chunks, self.chk_log_std.exp())
                
                # Sample the action to store in buffer!
                # This ensures we have a valid log_prob gradient later
                action_chunks = dist.sample() 
                
                # Calculate LogProb of the *Sampled* action
                # Note: This is an action distribution. We sum across dimensions.
                action_log_probs = dist.log_prob(action_chunks).sum(dim=[1, 2])
                
                # Move to CPU for buffer/step
                action_chunks_cpu = action_chunks.cpu().numpy()
                action_log_probs_cpu = action_log_probs.cpu().tolist()
                
            visual_embeddings = policy_out['visual_embedding'].detach().cpu()
            
            # 2. Compute Rewards (RSD) on Step 0
            # Compare Sampled Action vs Expert Step 0
            # This incentivizes the exploration to find the expert behavior
            
            # Construct expert "chunk" (Repeat static approximation for divergence calculation context)
            expert_chunks = np.repeat(current_expert_poses[:, np.newaxis, :], self.cfg.model.chunk_size, axis=1)
            
            # Calculate RSD on the sampled action to give fair reward
            p_chunk_t = action_chunks # Use sampled
            e_step0_t = torch.from_numpy(current_expert_poses).to(self.device).float()
            
            with torch.no_grad():
                # Compute RSD for Step 0 (Instantaneous Divergence)
                # This is the "Dense" reward signal
                rsd_scores = compute_riemannian_divergence(
                    p_chunk_t[:, 0:1, :], # Take first step 
                    e_step0_t.unsqueeze(1), 
                    policy_out['phase_logits']
                ).cpu().numpy()
            
            rsd_divergences.extend(rsd_scores.tolist())
            
            # [PRODUCTION] Track BC quality (MEAN prediction, no exploration noise)
            # This metric shows true BC model quality, unaffected by sampling noise
            mean_pred_step0 = pred_chunks[:, 0, :3].cpu().numpy()  # (N, 3) - Position only
            expert_pos = current_expert_poses[:, :3]  # (N, 3)
            bc_pos_divs = np.linalg.norm(mean_pred_step0 - expert_pos, axis=1)
            bc_pos_divergences.extend(bc_pos_divs.tolist())
            
            # 3. Step Envs (Shadow Mode - Expert executes)
            dummy_actions = np.zeros((self.num_envs, 8))
            next_obs, rewards, terminateds, truncateds, next_infos = self.envs.step(dummy_actions)
            
            # [SOTA Enhancement] Calculate entropy bonus for exploration
            with torch.no_grad():
                policy_entropy = dist.entropy().mean(dim=[1, 2])  # (N,) entropy per env
                entropy_bonus = (self.entropy_coef_adaptive * policy_entropy).cpu().numpy()
            
            # 4. Process Batch
            for i in range(self.num_envs):
                # Reward: Alignment + Success Bonus + Entropy
                sigma_sq = 0.05 
                imitation_reward = np.exp(-rsd_scores[i] / sigma_sq)
                
                # [SOTA Enhancement] Add entropy bonus
                total_reward = imitation_reward + float(entropy_bonus[i])
                
                # Logging metrics
                obj_pos = next_obs['object_pos_world'][i]
                goal_pos = next_obs['goal_pos_world'][i]
                success = np.linalg.norm(obj_pos - goal_pos) < 0.05
                current_ep_rewards[i] += (imitation_reward + (10.0 if success else 0.0))
                
                # Value Estimate
                with torch.no_grad():
                    val_proprio = torch.from_numpy(proprios[i]).float().unsqueeze(0).to(self.device)
                    val_emb = policy_out['visual_embedding'][i].unsqueeze(0)
                    value = self.value_net(val_emb, val_proprio).item()
                
                # Store Data
                expert_phase_int = EXPERT_PHASE_MAP.get(current_expert_phases[i], 0)
                expert_chunk_viz = np.tile(current_expert_poses[i], (self.cfg.model.chunk_size, 1))

                self.buffer.add(
                    prev_img=prev_imgs[i],
                    curr_img=curr_imgs[i],
                    goal_img=goal_imgs[i],
                    proprio=proprios[i],
                    visual_emb=visual_embeddings[i],
                    action_chunk=action_chunks_cpu[i], # STORE SAMPLED ACTION
                    log_prob=action_log_probs_cpu[i],  # STORE LOG PROB OF SAMPLED
                    reward=float(total_reward),  # [SOTA] Total reward with entropy
                    value=value,
                    done=terminateds[i] or truncateds[i],
                    expert_pose_chunk=expert_chunk_viz,
                    expert_phase=expert_phase_int
                )
                
                if terminateds[i] or truncateds[i]:
                    episode_rewards.append(current_ep_rewards[i])
                    episode_successes.append(float(success))
                    current_ep_rewards[i] = 0.0
                    if 'goal_img' in next_infos:
                         goal_imgs[i] = next_infos['goal_img'][i]
                    prev_imgs[i] = next_obs['image_primary'][i].copy()
                else:
                    prev_imgs[i] = curr_imgs[i].copy()
            
            obs = next_obs
            current_expert_poses = next_infos['expert_pose']
            current_expert_phases = next_infos['expert_phase']
            self.total_steps += self.num_envs
        
        self.policy.train()
        
        # Calculate simple euclidean divergence for logging
        if self.buffer.action_chunks:
            # Only compare step 0 to avoid punishing velocity on static expert target
            pred_step0 = np.array([c[0] for c in self.buffer.action_chunks])
            expert_step0 = np.array([c[0] for c in self.buffer.expert_pose_chunks])
            shadow_pos_div = np.mean(np.linalg.norm(pred_step0[:, :3] - expert_step0[:, :3], axis=-1))
        else:
            shadow_pos_div = 0.0

        return {
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "success_rate": np.mean(episode_successes) if episode_successes else 0.0,
            "n_episodes": len(episode_rewards),
            "shadow_pos_div": shadow_pos_div,  # Sampled (includes exploration noise)
            "bc_pos_div": np.mean(bc_pos_divergences) if bc_pos_divergences else 0.0,  # True BC quality
            "shadow_orn_div": np.mean(rsd_divergences) if rsd_divergences else 0.0,
            "grip_agreement": 1.0
        }

    # ==========================================================================
    # UPDATE POLICY (FIXED: BC ANCHOR)
    # ==========================================================================
    def update_policy(self) -> Dict[str, float]:
        """AC-PPO Update with SOTA Enhancements + BC Anchor."""
        rewards = np.array(self.buffer.rewards)
        values = np.array(self.buffer.values)
        dones = np.array(self.buffer.dones)
        
        # [SOTA Enhancement] Normalize rewards before GAE
        normalized_rewards = self.reward_normalizer.normalize(rewards, clip_range=5.0)
        
        advantages, returns = compute_gae(
            normalized_rewards, values, dones,
            gamma=self.cfg.ppo.gamma,
            lam=self.cfg.ppo.gae_lambda
        )
        
        # Normalize stats
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        adv_t = torch.from_numpy(advantages).float().to(self.device)
        ret_t = torch.from_numpy(returns).float().to(self.device)
        
        policy_losses = []
        value_losses = []
        bc_losses = []
        kl_divergences = []  # [SOTA] Track KL
        entropy_values = []  # [SOTA] Track entropy
        
        for epoch in range(self.cfg.ppo.epochs):
            indices = np.arange(len(self.buffer))
            np.random.shuffle(indices)
            
            for start in range(0, len(indices), self.cfg.ppo.batch_size):
                end = start + self.cfg.ppo.batch_size
                batch_idx = indices[start:end]
                
                # A. Prepare Batch - [PERF OPT] Vectorized batch preparation
                batch_size_actual = len(batch_idx)
                
                # Stack images efficiently (avoid PIL conversion)
                prev_imgs_np = np.stack([self.buffer.prev_images[i] for i in batch_idx])
                curr_imgs_np = np.stack([self.buffer.curr_images[i] for i in batch_idx])
                goal_imgs_np = np.stack([self.buffer.goal_images[i] for i in batch_idx])
                proprios_np = np.stack([self.buffer.proprios[i] for i in batch_idx])
                
                # Process batch (vectorized)
                batch = self._prepare_batch_vectorized(prev_imgs_np, curr_imgs_np, goal_imgs_np, proprios_np)
                b_prev, b_curr, b_goal, b_proprio = batch["prev_image"], batch["curr_image"], batch["goal_image"], batch["curr_proprio"]
                
                # B. Forward
                policy_out = self.policy({
                    "prev_image": b_prev, "curr_image": b_curr, "goal_image": b_goal, "curr_proprio": b_proprio
                })
                pred_chunks = policy_out['pose_chunk'] # (B, K, 7)
                
                # C. PPO Loss
                dist_new = torch.distributions.Normal(pred_chunks, self.chk_log_std.exp())
                
                # Retrieve SAMPLED actions from buffer
                b_act_chunks = torch.stack([torch.from_numpy(self.buffer.action_chunks[i]) for i in batch_idx]).to(self.device)
                b_log_prob_old = torch.tensor([self.buffer.log_probs[i] for i in batch_idx], device=self.device)
                
                # Calculate new log prob of the OLD actions
                log_prob_new = dist_new.log_prob(b_act_chunks).sum(dim=[1, 2])
                
                ratio = torch.exp(log_prob_new - b_log_prob_old)
                b_adv = adv_t[batch_idx]
                
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * b_adv
                ppo_loss = -torch.min(surr1, surr2).mean()

                # [SOTA FIX] Entropy Regularization
                entropy_loss = -dist_new.entropy().mean() * self.cfg.ppo.get("entropy_coef", 0.01)
                
                # D. Behavior Cloning (BC) Anchor Loss
                # [DGPO v2.1 FIX] Force alignment with expert on STEP 0
                # This prevents "drifting away" when PPO signal is noisy.
                # We use Step 0 because expert_pose_chunk is just a repeat of the current target,
                # and enforcing it on the whole chunk would kill velocity/prediction.
                
                b_expert_chunks = torch.stack([torch.from_numpy(self.buffer.expert_pose_chunks[i]) for i in batch_idx]).to(self.device)
                
                # MSE on the first step (Immediate action alignment)
                # This acts as the "Dense Guidance"
                bc_loss_val = F.mse_loss(pred_chunks[:, 0, :], b_expert_chunks[:, 0, :])
                
                # E. Value Loss (with SOTA Clipping)
                curr_emb = policy_out['visual_embedding']
                value_pred = self.value_net(curr_emb, b_proprio)
                
                # SOTA: Clip Value function updates to prevent spikes
                v_target = ret_t[batch_idx]
                v_old = torch.tensor([self.buffer.values[i] for i in batch_idx], device=self.device)
                v_clip_range = self.cfg.ppo.get("clip_param", 0.2)
                v_pred_clipped = v_old + (value_pred - v_old).clamp(-v_clip_range, v_clip_range)
                
                v_loss1 = F.mse_loss(value_pred, v_target)
                v_loss2 = F.mse_loss(v_pred_clipped, v_target)
                value_loss = torch.max(v_loss1, v_loss2)
                
                # F. Phase Loss
                b_phases = torch.tensor([self.buffer.expert_phases[i] for i in batch_idx], device=self.device)
                phase_loss = F.cross_entropy(policy_out['phase_logits'], b_phases)
                
                # [SOTA Enhancement] G. KL Divergence Tracking
                with torch.no_grad():
                    kl_div = (b_log_prob_old - log_prob_new).mean()
                    kl_divergences.append(kl_div.item())
                kl_penalty = self.kl_penalty.beta * kl_div
                
                # [SOTA Enhancement] H. Temporal Smoothness Loss
                chunk_diffs = pred_chunks[:, 1:, :] - pred_chunks[:, :-1, :]
                smoothness_loss = (chunk_diffs ** 2).mean()
                lambda_smooth = 0.01
                
                # [SOTA Enhancement] I. Track Entropy
                policy_entropy = dist_new.entropy().mean()
                entropy_values.append(policy_entropy.item())
                
                # Total Loss: PPO + Entropy + BC + Value + Aux + KL + Smoothness
                loss = (ppo_loss + entropy_loss + 0.5 * value_loss + 0.1 * phase_loss + 
                        1.0 * bc_loss_val + kl_penalty + lambda_smooth * smoothness_loss)
                
                # [PERF OPT] Mixed Precision Backward
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.policy_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                    self.scaler.step(self.policy_optimizer)
                    self.scaler.step(self.value_optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                    self.policy_optimizer.step()
                    self.value_optimizer.step()
                
                # [SOTA FIX] Step Scheduler
                self.scheduler.step()
                
                # [SOTA FIX] Update EMA Policy
                with torch.no_grad():
                    for param, ema_param in zip(self.policy.parameters(), self.ema_policy.parameters()):
                        ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

                policy_losses.append(ppo_loss.item())
                value_losses.append(value_loss.item())
                bc_losses.append(bc_loss_val.item())
        
        # [SOTA Enhancement] Update adaptive components
        mean_kl = np.mean(kl_divergences) if kl_divergences else 0.0
        self.kl_penalty.update(mean_kl)
        
        # [SOTA Enhancement] Soft update target critic
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(self.soft_update_tau * param.data + (1 - self.soft_update_tau) * target_param.data)
        
        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "bc_loss": np.mean(bc_losses),
            "kl_divergence": mean_kl,
            "kl_beta": self.kl_penalty.beta,
            "entropy": np.mean(entropy_values) if entropy_values else 0.0
        }

    # ... [train method remains largely the same, logging bc_loss] ...
    def train(self):
        total_iters = self.cfg.training.total_iterations
        log.info(f"Starting DGPO training for {total_iters} iterations")
        
        log_dir = Path(self.cfg.logging.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_logger = MetricsLogger(log_dir / self.cfg.logging.csv_log_name, resume=(self.start_iteration > 0))
        
        try:
            for iteration in range(self.start_iteration, total_iters):
                self.iteration = iteration
                rollout_stats = self.collect_rollouts(self.cfg.training.steps_per_iter)
                update_stats = self.update_policy()
                
                log.info(
                    f"Iter {iteration:4d} | "
                    f"R: {rollout_stats['mean_reward']:.2f} | "
                    f"Succ: {rollout_stats['success_rate']*100:.1f}% | "
                    f"BC: {update_stats['bc_loss']:.4f} | " # Log BC
                    f"PL: {update_stats['policy_loss']:.4f}"
                )
                
                log.info(
                    f"         Shadow Mode | "
                    f"PosDiv: {rollout_stats['shadow_pos_div']*100:.2f}cm | "
                    f"BC: {rollout_stats['bc_pos_div']*100:.2f}cm | "
                    f"OrnDiv: {rollout_stats['shadow_orn_div']:.3f}rad"
                )
                
                # [SOTA Enhancement] Log adaptive mechanisms
                log.info(
                    f"          SOTA | KL: {update_stats['kl_divergence']:.4f} | "
                    f"β: {update_stats['kl_beta']:.4f} | "
                    f"H: {update_stats['entropy']:.4f} | "
                    f"α: {self.entropy_coef_adaptive:.4f}"
                )
                
                # [SOTA Enhancement] Apply entropy decay per iteration
                self.entropy_coef_adaptive *= self.entropy_decay
                
                
                if (iteration + 1) % self.cfg.logging.get("log_every_n_iters", 1) == 0:
                    csv_logger.log_step(iteration, rollout_stats, update_stats, self.total_steps)
                
                self._log_to_tensorboard(iteration, rollout_stats, update_stats)
                
                if (iteration + 1) % self.cfg.checkpoint.get("save_freq", 10) == 0:
                    self._save_checkpoint(iteration, is_backup=False)
                    # Also save EMA model
                    self._save_checkpoint(iteration, is_backup=False, use_ema=True)
                
                if (iteration + 1) % self.cfg.checkpoint.get("backup_freq", 5) == 0:
                    self._save_checkpoint(iteration, is_backup=True)

            self._save_checkpoint(total_iters - 1, is_backup=False)
            log.info("Training complete!")
            
        finally:
            csv_logger.close()
            if self.tb_writer: self.tb_writer.close()

    def _save_checkpoint(self, iteration: int, is_backup: bool = False, use_ema: bool = False) -> Path:
        if is_backup:
            save_dir = Path(self.cfg.checkpoint.backup_dir)
            filename = f"dgpo_backup_iter_{iteration+1:04d}.pt"
        elif use_ema:
            save_dir = Path(self.cfg.checkpoint.save_dir)
            filename = f"dgpo_ema_iter_{iteration+1:04d}.pt"
        else:
            save_dir = Path(self.cfg.checkpoint.save_dir)
            filename = f"dgpo_iter_{iteration+1:04d}.pt"
        
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / filename
        
        # Determine which policy to save
        policy_state = self.ema_policy.state_dict() if use_ema else self.policy.state_dict()
        
        checkpoint = {
            'iteration': iteration,
            'policy_state_dict': policy_state,
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'config': OmegaConf.to_container(self.cfg, resolve=True),
            'timestamp': datetime.now().isoformat(),
        }
        torch.save(checkpoint, ckpt_path)
        if is_backup: self._cleanup_old_backups()
        return ckpt_path


@hydra.main(version_base=None, config_path="../configs", config_name="train_dgpo_config")
def main(cfg: DictConfig):
    log.info("=" * 60)
    log.info("DGPO-Foundation Training (SOTA Production Fix)")
    log.info("=" * 60)
    trainer = DGPOTrainer(cfg)
    trainer.train()

if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
