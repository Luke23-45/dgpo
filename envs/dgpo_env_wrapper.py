
import gymnasium as gym
import numpy as np
import torch
from typing import Dict, Any, Tuple
import logging

# Logic Imports (Must be available in PYTHONPATH)
# Logic Imports (Must be available in PYTHONPATH)
from utils.dgpo_expert import DGPOExpert, DGPOExpertConfig, ObjectProfile
from utils.ik_solver import IKSolver
from utils.divergence import compute_step_divergence
from envs.panda_env import PandaEnv
from omegaconf import OmegaConf, DictConfig

# Setup localized logger (will likely print to stderr in subprocesses)
log = logging.getLogger(__name__)

# ==============================================================================
# Helper for Multiprocessing (Moved here to ensure picklability/importability)
# ==============================================================================
def make_dgpo_env(cfg_dict: Dict[str, Any]) -> gym.Env:
    """Factory function to create a wrapped DGPO environment."""
    # [CRITICAL FIX] Ensure Subprocess knows to use EGL
    import os
    os.environ['MUJOCO_GL'] = 'egl'
    
    # Convert dict back to DictConfig if needed, or pass dict to Wrapper
    cfg = OmegaConf.create(cfg_dict)
    
    env = PandaEnv(
        xml_path=cfg.environment.xml_path,
        control_mode="delta",
        render_mode="rgb_array"
    )
    return DGPOEnvWrapper(env, cfg)


class DGPOEnvWrapper(gym.Wrapper):
    """
    Wraps PandaEnv to bundle DGPO Expert and IK Solver logic inside the environment.
    This enables usage with AsyncVectorEnv where the main process cannot access 
    inner simulation data (MjData) required for Jacobians or Expert FSM.
    
    Functionality:
    1. Manages internal 'DGPOExpert' and 'IKSolver' instances.
    2. Supports BLENDED execution: (1-α)*Expert + α*Policy
    3. Returns 'expert_pose' and 'expert_grip' in the info dict for the Trainer to calculate loss.
    
    Enhanced DGPO v3.0: Progressive Policy Blending Mode
    """
    
    def __init__(self, env, cfg):
        super().__init__(env)
        self.cfg = cfg
        
        # 1. Initialize IK Solver (Needs URDF path from config or env)
        # We assume cfg is the full Hydraconfig or a dict with necessary keys
        urdf_path = getattr(cfg.environment, "urdf_path", "urdf/panda_mujoco_kinematics.urdf")
        self.ik_solver = IKSolver(urdf_path=urdf_path)
        
        # 2. Initialize Expert
        # Create object profile (assuming standard cube for now)
        object_size = getattr(cfg.expert, "object_size", [0.04, 0.04, 0.04])
        grasp_width = getattr(cfg.expert, "grasp_width", 0.6)
        
        object_profile = ObjectProfile(
            size=np.array(object_size), 
            grasp_width_normalized=grasp_width
        )
        
        expert_cfg = DGPOExpertConfig(
            hover_height=getattr(cfg.expert, 'hover_height', 0.15),
            grasp_offset_z=getattr(cfg.expert, 'grasp_offset_z', 0.025),
            # Pass workspace if available, else expert uses default
        )
        
        self.expert = DGPOExpert(
            object_profile=object_profile,
            cfg=expert_cfg,
        )
        
        # Control calibration (match Trainer's logic)
        SIM_SUBSTEPS = 20
        self.effective_dt = self.env.model.opt.timestep * SIM_SUBSTEPS
        
        # Cache for stateful execution
        self.latest_expert_pose = None
        self.latest_expert_grip = None
        
        # [ENHANCED DGPO v3.0] Blending mode configuration
        # FIX: Config is a plain dict when using AsyncVectorEnv, not DictConfig
        # So we need to use dict-style access, not attribute access
        self.blend_alpha = 0.1  # Start with small policy influence
        
        # Check if training config exists (handles both dict and DictConfig)
        training_cfg = cfg.get('training', {}) if isinstance(cfg, dict) else getattr(cfg, 'training', {})
        if isinstance(training_cfg, dict):
            self.use_blending = training_cfg.get('use_policy_blending', True)
            self.blend_alpha = 0.1  # Initial alpha (will be updated by trainer)
        else:
            self.use_blending = getattr(training_cfg, 'use_policy_blending', True)
        
        log.info(f"[DGPOEnvWrapper] Blending enabled: {self.use_blending}, initial_alpha: {self.blend_alpha}")
        
    def reset(self, **kwargs):
        """
        Resets environment and expert. 
        Calculates the INITIAL expert target for the first observation.
        """
        obs, info = self.env.reset(**kwargs)
        self.expert.reset()
        
        # Get expert target based on INITIAL observation
        expert_obs = self.env.get_expert_obs() # Wrapper must ensure it calls the underlying get_expert_obs
        
        target_pose, grip, _ = self.expert.get_target_pose(expert_obs)
        
        self.latest_expert_pose = target_pose
        self.latest_expert_grip = grip
        
        # Add to info for Trainer
        info['expert_pose'] = target_pose
        info['expert_grip'] = grip
        info['expert_phase'] = self.expert.get_state()
        info['executed_action'] = np.zeros(8, dtype=np.float32) # [FIX] Prevent KeyError on reset/done common in VectorEnv
        
        # Helper: Render goal image (since main process cannot access MuJoCo state)
        # Use simple try/except in case the internal method name changes or fails
        try:
            info['goal_img'] = self.env._render_goal_image()
        except Exception as e:
            # Fallback for stability
            info['goal_img'] = np.zeros_like(obs['image_primary'])
        
        return obs, info
        
    def step(self, action):
        """
        Executes a step with PROGRESSIVE POLICY BLENDING.
        
        [ENHANCED DGPO v3.0]
        - When blend_alpha = 0: Pure Shadow Mode (Expert executes)
        - When blend_alpha = 1: Pure Policy Mode (Policy executes)
        - In between: Blended execution = (1-α)*Expert + α*Policy
        
        Args:
            action: Can be:
                   - 8D: [7 joints + 1 gripper] - uses self.blend_alpha
                   - 9D: [7 joints + 1 gripper + 1 alpha] - uses provided alpha
                   If blending disabled, policy action is IGNORED.
            
        Returns:
            Standard Gym 5-tuple. 'info' contains the 'expert_pose' for the NEXT step.
        """
        
        # 1. Compute Expert Action (IK) based on the LATEST target state
        try:
            delta_joints = self.ik_solver.compute_delta_action(
                target_ee_pose_chunk=np.array([self.latest_expert_pose]),
                model=self.env.model,
                data=self.env.data,
                ee_site_id=self.env.ee_site_id,
                joint_qpos_indices=np.arange(7),
                effective_dt=self.effective_dt,
                max_dq=self.env.ACTION_SCALING_FACTOR / self.effective_dt
            )
        except Exception as e:
            delta_joints = np.zeros(7)
        
        expert_action = np.concatenate([delta_joints, [self.latest_expert_grip]])
        
        # 2. [ENHANCED DGPO v3.0] Compute BLENDED action
        if self.use_blending and action is not None:
            action_flat = np.asarray(action).flatten()
            
            # Check if alpha is provided in action (9D)
            if len(action_flat) >= 9:
                policy_action = action_flat[:8]
                alpha = float(np.clip(action_flat[8], 0.0, 1.0))
            else:
                policy_action = action_flat[:8]
                if len(policy_action) < 8:
                    policy_action = np.concatenate([policy_action, np.zeros(8 - len(policy_action))])
                alpha = self.blend_alpha
            
            # Blend: (1-α)*Expert + α*Policy
            blended_action = (1 - alpha) * expert_action + alpha * policy_action
            
            # Safety clipping
            blended_action[:7] = np.clip(blended_action[:7], -1.0, 1.0)
            blended_action[7] = np.clip(blended_action[7], -1.0, 1.0)
            
            executed_action = blended_action
            used_alpha = alpha
        else:
            # Pure Shadow Mode
            executed_action = expert_action
            used_alpha = 0.0
        
        # 3. Step the Environment with the EXECUTED action
        next_obs, reward, terminated, truncated, info = self.env.step(executed_action)
        
        # 4. Handle Expert Finish / Done
        if self.expert.is_done():
            terminated = True
            
        # 5. Compute NEXT Expert Target
        expert_obs = self.env.get_expert_obs()
        target_pose, grip, exp_info = self.expert.get_target_pose(expert_obs)
        
        self.latest_expert_pose = target_pose
        self.latest_expert_grip = grip
        
        # 6. Populate Info for Trainer
        info['expert_pose'] = target_pose
        info['expert_grip'] = grip
        info['expert_phase'] = exp_info.get('phase', 'UNKNOWN')
        info['executed_action'] = executed_action
        info['expert_action'] = expert_action
        info['blend_alpha'] = used_alpha
        
        return next_obs, reward, terminated, truncated, info

    def set_blend_alpha(self, alpha: float):
        """
        [ENHANCED DGPO v3.0] Set the blending coefficient.
        
        Args:
            alpha: Float in [0, 1].
                   0 = Pure Expert (Shadow Mode)
                   1 = Pure Policy (Full Autonomy)
        """
        self.blend_alpha = np.clip(alpha, 0.0, 1.0)
        self.use_blending = (alpha > 0.0)

    def get_expert_obs(self):
        return self.env.get_expert_obs()

    def get_ee_pose(self):
        return self.env.get_ee_pose()
