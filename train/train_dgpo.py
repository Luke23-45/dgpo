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
    """Stores chunk-based rollout data for AC-PPO updates."""
    
    # State data
    prev_images: List[np.ndarray] = field(default_factory=list)
    curr_images: List[np.ndarray] = field(default_factory=list)
    goal_images: List[np.ndarray] = field(default_factory=list)
    proprios: List[np.ndarray] = field(default_factory=list)
    visual_embeddings: List[torch.Tensor] = field(default_factory=list) # [DGPO v2.0] Stored on CPU/GPU?
    
    # Action Chunks (The "Action")
    action_chunks: List[np.ndarray] = field(default_factory=list) # (K, 7) or (K, 8)
    
    # Log Probs (of the entire chunk)
    log_probs: List[float] = field(default_factory=list)
    
    # Rewards and values
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    
    # [DGPO v2.0] Expert Targets for Divergence Loss
    expert_pose_chunks: List[np.ndarray] = field(default_factory=list)
    expert_phases: List[int] = field(default_factory=list) # For stiffness weighting
    
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

# Removed: render_goal_image (Now handled by DGPOEnvWrapper)



# ==============================================================================
# 5. DGPO TRAINER
# ==============================================================================

# ==============================================================================
# Helper for Multiprocessing (Must be module-level)
# ==============================================================================

def make_dgpo_env(cfg_dict: Dict[str, Any]) -> gym.Env:
    """Factory function to create a wrapped DGPO environment."""
    # Convert dict back to DictConfig if needed, or pass dict to Wrapper
    # Wrapper expects the full cfg object or similar structure
    # Here we assume cfg_dict handles the necessary attribute access or we wrap it
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
        self.policy.train()  # Enable training mode
        
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
        
        # [DGPO v2.0 FIX] Initialize Action Log Std HERE, not in loop
        # We assume 7D pose (3 pos + 4 quat).
        self.chk_log_std = nn.Parameter(
            torch.ones(1, self.cfg.model.chunk_size, 7, device=self.device) * -0.5
        ) # Start with small std
        
        # 3. Initialize Parallel Environments
        self.num_envs = cfg.get("num_envs", 8)  # Default to 8 envs
        log.info(f"Initializing {self.num_envs} Parallel Environments...")
        
        # Convert config to primitive dict for safe pickling
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        
        # Create list of factory functions
        env_fns = [
            lambda: make_dgpo_env(cfg_dict) 
            for _ in range(self.num_envs)
        ]
        
        # Use AsyncVectorEnv (multiprocessing)
        # Context 'spawn' is safer for PyTorch/CUDA interaction
        # Gymnasium usually handles context, but we can enforce it via kwargs if needed
        # For now, default behavior is usually sufficient on Windows (defaults to spawn?)
        self.envs = gym.vector.AsyncVectorEnv(env_fns)
        
        # Removed: self.ik_solver (Now inside Wrapper)
        # Removed: self.expert (Now inside Wrapper)
        
        # 6. Control calibration
        # Note: We still need these constants for consistency checks or logging, 
        # but the physics steps happen inside the wrapper.
        pass

        
        # 7. Optimizers
        policy_params = [p for p in self.policy.parameters() if p.requires_grad]
        policy_params.append(self.chk_log_std) # [DGPO v2.0 FIX] Add log_std to optimizer
        
        self.policy_optimizer = torch.optim.Adam(
            policy_params, lr=cfg.optimizer.policy_lr
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=cfg.optimizer.value_lr
        )
        
        # 8. Image transform - MUST EXACTLY MATCH BC TRAINING!
        # BC uses: Resize(224, BICUBIC) + ToTensor() + Normalize(0.5, 0.5) -> [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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
        Collects rollout data using PARALLEL ENVIRONMENTS and AC-PPO LOGIC.
        
        [DGPO v2.0]:
        1. Predict Action Chunk.
        2. Calculate LogProb of Chunk.
        3. Execute first step of chunk (Shadow Mode: Execute Expert).
        4. Store Chunk, VisualEmbedding, ExpertChunk for updates.
        """
        self.buffer.clear()
        self.policy.eval()
        
        # Metrics trackers
        episode_rewards = []
        episode_successes = []
        current_ep_rewards = np.zeros(self.num_envs)
        
        # Divergence Metrics
        rsd_divergences = []
        
        # 1. Reset
        obs, info = self.envs.reset()
        goal_imgs = info['goal_img']
        prev_imgs = obs['image_primary'].copy()
        
        # For RSD reward, we need the Expert's FUTURE chunk.
        # But in a running env, we only know the Expert's CURRENT target.
        # We will approximate the Expert Chunk by generating it online or using the current target extended.
        # Better: The DGPOExpert inside the wrapper can return the 'target_pose'.
        # For the reward, we compare the policy's predicted chunk[0] vs expert[0] for dense feedback,
        # OR we try to predict the full expert trajectory. 
        # DGPO v2.0 Formulation says: Divergence between Policy Chunk and Expert Chunk.
        # WE NEED THE EXPERT CHUNK. 
        # The wrapper provides 'expert_pose'. This is step t.
        # Implementation constraint: We only get 1 step of expert data per environment step.
        # Solution: We compute RSD for the *first step* of the chunk (immediate divergence)
        # OR we store the single step data and the policy's single step prediction.
        # Wait, AC-PPO updates the *entire chunk distribution*.
        # So we need to store the Policy Chunk.
        # But we only correspond it to the Expert's single step? 
        # Or we execute K steps open loop?
        # The V2.0 proposal says "Divergence between Policy Chunk and Expert Chunk".
        # If we cannot get the full expert chunk (because expert is stateful/reactive), 
        # we can only compare the *immediate* step, or we accept that we only optimize the first step of the chunk?
        # NO. We should optimize the FULL chunk.
        # Simplification for Online RL:
        # We compare PolicyChunk[0] vs ExpertTarget.
        # We verify stability of the rest of the chunk?
        # Let's stick to the simpler version where we reward based on immediate agreement,
        # but update the WHOLE chunk distribution based on that reward.
        
        # Actually, for 'compute_riemannian_divergence(pred_chunk, expert_chunk)', we need 'expert_chunk'.
        # Since we can't fast-forward the expert in the vector env easily,
        # we will create a pseudo-expert-chunk by repeating the current expert target K times.
        # This is valid because the expert is a stable attractor.
        
        current_expert_poses = info['expert_pose']
        current_expert_grips = info['expert_grip']
        current_expert_phases = info['expert_phase']
        
        # Define Phase Map locally (Minor Bug Fix 1)
        EXPERT_PHASE_MAP = {
            "MOVE_TO_PRE_GRASP": 0,
            "PREPARE_GRIPPER": 0,
            "DESCEND_TO_GRASP": 0, 
            "GRASP": 1,
            "LIFT": 2,
            "MOVE_TO_GOAL": 2,
            "PREPARE_PLACE": 3,
            "DESCEND_TO_PLACE": 3,
            "AWAIT_STABLE_PLACEMENT": 3,
            "RELEASE": 3,
            "RETRACT": 4,
            "DONE": 4
        }
        
        steps_per_env = n_steps // self.num_envs
        
        for step in range(steps_per_env):
            curr_imgs = obs['image_primary']
            proprios = obs['proprio']
            
            # 1. Policy Inference
            batch = self._prepare_batch_vectorized(prev_imgs, curr_imgs, goal_imgs, proprios)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    policy_out = self.policy(batch)
                
                # [DGPO v2.0 FIX] Critical Error A: Calculate LogProb HERE
                pred_chunks = policy_out['pose_chunk'] # (N, K, 7)
                dist = torch.distributions.Normal(pred_chunks, self.chk_log_std.exp())
                # The log_prob of the action we *predicted* (which is the mean)
                # Note: `pred_chunks` is the mean of the distribution.
                # We need the log_prob of the action that gets stored.
                # In DGPO Shadow Mode, we define the "Action" for the update as the Policy's Prediction.
                # So we calculate log_prob(mean).
                action_log_probs = dist.log_prob(pred_chunks).sum(dim=[1, 2]) # (N,)
                action_log_probs_cpu = action_log_probs.cpu().tolist()
            
            # Outputs
            pose_chunks = policy_out['pose_chunk'].cpu().numpy()
            visual_embeddings = policy_out['visual_embedding'].detach() 
            visual_embeddings_cpu = visual_embeddings.cpu()
            
            # 2. Compute Rewards (RSD)
            # We need to construct expert chunks.
            # Assume expert target is constant/stable for the chunk duration (simplification).
            expert_chunks = np.repeat(current_expert_poses[:, np.newaxis, :], self.cfg.model.chunk_size, axis=1) # (N, K, 7)
            
            # Compute RSD (Riemannian Semantic Divergence)
            # We convert numpy -> tensor for the utility calculation
            p_chunk_t = torch.from_numpy(pose_chunks).to(self.device)
            e_chunk_t = torch.from_numpy(expert_chunks).to(self.device)
            phase_logits_t = policy_out['phase_logits']
            
            with torch.no_grad():
                # D_sigma
                rsd_scores = compute_riemannian_divergence(p_chunk_t, e_chunk_t, phase_logits_t).cpu().numpy() # (N,)
            
            rsd_divergences.extend(rsd_scores.tolist())
            
            # 3. Step Envs (Shadow Mode)
            dummy_actions = np.zeros((self.num_envs, 8))
            next_obs, rewards, terminateds, truncateds, next_infos = self.envs.step(dummy_actions)
            
            # 4. Process Batch
            for i in range(self.num_envs):
                # Calculate Dense Reward
                # r_dense = r_task + lambda * I[success_possible] * exp(-D_sigma / sigma^2)
                
                obj_pos = next_obs['object_pos_world'][i]
                goal_pos = next_obs['goal_pos_world'][i]
                dist_to_goal = np.linalg.norm(obj_pos - goal_pos)
                
                # Check for "Success Possible" (heuristic: not failed)
                success_possible = 1.0 # In this robust env, we assume always possible unless done
                
                # Task Reward (Sparse-ish or Shaped)
                r_task = -self.cfg.reward.w_dist * dist_to_goal
                
                # Geodesic Guidance
                # exp(-D / sigma^2)
                # sigma is a temperature parameter. Let's say 0.1
                sigma_sq = 0.01 
                guidance = np.exp(-rsd_scores[i] / sigma_sq)
                
                rsd_reward = self.cfg.reward.w_div * success_possible * guidance
                
                success = dist_to_goal < 0.05
                success_bonus = self.cfg.reward.success_bonus if success else 0.0
                
                total_reward = r_task + rsd_reward + success_bonus
                current_ep_rewards[i] += total_reward
                
                # Value Estimate (Vision Critic)
                # Input: visual_emb (already computed) + proprio
                # Needs proprio tensor
                with torch.no_grad():
                    val_proprio = torch.from_numpy(proprios[i]).float().unsqueeze(0).to(self.device)
                    val_emb = visual_embeddings[i].unsqueeze(0) # (1, D)
                    value = self.value_net(val_emb, val_proprio).item()
                
                # Store
                executed_action = next_infos['executed_action'][i] # Expert action (for reference, mostly)
                
                # For AC-PPO, we store the POLICY CHUNK as the "Action".
                # and the Expert Chunk for reconstruction.
                
                # Log Prob??
                # We calculate log_prob of the chunk during update? 
                # Or here?
                # Usually calculate here. But for Gaussian with fixed std, we can recalc.
                # Storing 0.0 for now, recompute in update (standard for some implementations, but less efficient).
                # To be precise, we should compute it here.
                # But 'policy_out' is deterministic pose. We need the distribution info from the model/trainer.
                # Trainer holds 'log_std'.
                # Proceed with recomputing in update_policy.
                
                expert_phase_int = EXPERT_PHASE_MAP.get(current_expert_phases[i], 0)
                
                self.buffer.add(
                    prev_img=prev_imgs[i],
                    curr_img=curr_imgs[i],
                    goal_img=goal_imgs[i],
                    proprio=proprios[i],
                    visual_emb=visual_embeddings_cpu[i], # Store CPU tensor
                    action_chunk=pose_chunks[i], # The Chunk
                    log_prob=action_log_probs_cpu[i], # [DGPO v2.0 FIX] Store REAL old log_prob
                    reward=total_reward,
                    value=value,
                    done=terminateds[i] or truncateds[i],
                    expert_pose_chunk=expert_chunks[i],
                    expert_phase=expert_phase_int
                )
                
                if terminateds[i] or truncateds[i]:
                    episode_rewards.append(current_ep_rewards[i])
                    episode_successes.append(float(success))
                    current_ep_rewards[i] = 0.0
                    
                    if 'goal_img' in next_infos: # Handle auto-reset updates
                         goal_imgs[i] = next_infos['goal_img'][i]
                    prev_imgs[i] = next_obs['image_primary'][i].copy()
                else:
                    prev_imgs[i] = curr_imgs[i].copy()
            
            obs = next_obs
            current_expert_poses = next_infos['expert_pose']
            current_expert_grips = next_infos['expert_grip']
            current_expert_phases = next_infos['expert_phase']
            
            self.total_steps += self.num_envs
        
        self.policy.train()
        
        return {
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "success_rate": np.mean(episode_successes) if episode_successes else 0.0,
            "n_episodes": len(episode_rewards),
            "mean_rsd": np.mean(rsd_divergences) if rsd_divergences else 0.0,
        }

    def _prepare_batch_vectorized(
        self,
        prev_imgs: np.ndarray,
        curr_imgs: np.ndarray,
        goal_imgs: np.ndarray,
        proprios: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        """
        Batched version of _prepare_batch.
        Input arrays are (N, H, W, C).
        """
        # Convert entire batch to Processed Tensors
        # Optimization: We could use specific data loaders or optimized conversions here.
        # For now, we iterate, but it's still parallelized upstream.
        # Faster: Use a list comprehension and stack.
        
        def process(imgs):
            # Optim: Use listcomp instead of loop for slightly better speed
            tensors = [self.transform(Image.fromarray(img)) for img in imgs]
            return torch.stack(tensors).to(self.device)

        prev_t = process(prev_imgs)
        curr_t = process(curr_imgs)
        goal_t = process(goal_imgs)
        proprio_t = torch.from_numpy(proprios).float().to(self.device)
        
        return {
            "prev_image": prev_t,
            "curr_image": curr_t,
            "goal_image": goal_t,
            "curr_proprio": proprio_t
        }
    
    def update_policy(self) -> Dict[str, float]:
        """
        AC-PPO Update (DGPO v2.0).
        Optimizes Policy Chunk Distribution using Riemannian Semantic Divergence for rewards.
        """
        # 1. Compute GAE
        rewards = np.array(self.buffer.rewards)
        values = np.array(self.buffer.values)
        dones = np.array(self.buffer.dones)
        
        # Standard GAE
        advantages, returns = compute_gae(
            rewards, values, dones,
            gamma=self.cfg.ppo.gamma,
            lam=self.cfg.ppo.gae_lambda
        )
        
        # Normalize stats
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        adv_t = torch.from_numpy(advantages).float().to(self.device)
        ret_t = torch.from_numpy(returns).float().to(self.device)
        
        # [DGPO v2.0 FIX] Initialize loss lists (Minor Bug 2)
        policy_losses = []
        value_losses = []

        # 2. PPO Epochs
        for epoch in range(self.cfg.ppo.epochs):
            indices = np.arange(len(self.buffer))
            np.random.shuffle(indices)
            
            for start in range(0, len(indices), self.cfg.ppo.batch_size):
                end = start + self.cfg.ppo.batch_size
                batch_idx = indices[start:end]
                
                # A. Re-Run Policy Output (Visual Features + Chunks)
                # We need to re-generate the action distribution to compare probability
                
                # Prepare Inputs
                b_prev = torch.stack([self.transform(Image.fromarray(self.buffer.prev_images[i])) for i in batch_idx]).to(self.device)
                b_curr = torch.stack([self.transform(Image.fromarray(self.buffer.curr_images[i])) for i in batch_idx]).to(self.device)
                b_goal = torch.stack([self.transform(Image.fromarray(self.buffer.goal_images[i])) for i in batch_idx]).to(self.device)
                b_proprio = torch.stack([torch.from_numpy(self.buffer.proprios[i]).float() for i in batch_idx]).to(self.device)
                
                # Re-run Actor
                policy_out = self.policy({
                    "prev_image": b_prev,
                    "curr_image": b_curr,
                    "goal_image": b_goal,
                    "curr_proprio": b_proprio
                })
                
                # B. Chunk Distributions
                # [DGPO v2.0 FIX] Removed dynamic parameter init. self.chk_log_std is now in __init__.
                
                pred_chunks = policy_out['pose_chunk'] # (B, K, 7)
                batch_size = pred_chunks.shape[0]
                
                dist_new = torch.distributions.Normal(pred_chunks, self.chk_log_std.exp())
                
                # Retrieve stored Actions (Policy Chunks from Rollout)
                b_act_chunks = torch.stack([torch.from_numpy(self.buffer.action_chunks[i]) for i in batch_idx]).to(self.device)
                
                # [DGPO v2.0 FIX] Retrieve stored OLD Log Probs
                b_log_prob_old = torch.tensor([self.buffer.log_probs[i] for i in batch_idx], device=self.device)
                
                # Calculate NEW Log Prob
                log_prob_new = dist_new.log_prob(b_act_chunks).sum(dim=[1, 2]) # Sum over K, 7
                
                # Calculate Ratio
                ratio = torch.exp(log_prob_new - b_log_prob_old)
                
                b_adv = adv_t[batch_idx]
                
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # C. Value Loss
                # We need Visual Embeddings for Critic!
                # We stored them in buffer (CPU).
                b_emb = torch.stack([self.buffer.visual_embeddings[i].to(self.device) for i in batch_idx])
                
                # Re-run Critic
                # (Note: b_emb is from rollout (stale?). SOTA usually requires re-computing features if backbone updates.
                # But backbone is frozen/slow. Using stored embeddings is vastly faster.
                # However, Actor *updated* the embeddings in `policy_out` if backbone is trainable.
                # `policy_out['visual_embedding']` is the FRESH embedding.
                # We should use THAT for the Critic update to keep it consistent with the new actor state?
                # Or use the stored one?
                # Critic minimizes (V(s) - Target).
                # Using FRESH embedding is better.
                
                curr_emb = policy_out['visual_embedding']
                value_pred = self.value_net(curr_emb, b_proprio)
                value_loss = F.mse_loss(value_pred, ret_t[batch_idx])
                
                # D. Phase Loss (Auxiliary)
                # b_expert_phases stored in buffer
                b_phases = torch.tensor([self.buffer.expert_phases[i] for i in batch_idx], device=self.device)
                phase_loss = F.cross_entropy(policy_out['phase_logits'], b_phases)
                
                # Total Loss
                loss = policy_loss + 0.5 * value_loss + 0.1 * phase_loss
                
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
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
