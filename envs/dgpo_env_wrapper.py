
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
    2. Overrides 'step()' to IGNORE the input action and instead execute the EXPERT's action.
    3. Returns 'expert_pose' and 'expert_grip' in the info dict for the Trainer to calculate loss.
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
        Executes a step in SHADOW MODE.
        
        Args:
            action: The POLICY'S predicted action (or dummy). IGNORED for physics control.
            
        Returns:
            Standard Gym 5-tuple. 'info' contains the 'expert_pose' for the NEXT step.
        """
        
        # 1. Compute Expert Action (IK) based on the LATEST target state
        #    (computed at the end of the previous step or reset)
        try:
            delta_joints = self.ik_solver.compute_delta_action(
                target_ee_pose_chunk=np.array([self.latest_expert_pose]), # [FIX] Wrap as chunk for Adaptive IK
                model=self.env.model,
                data=self.env.data,
                ee_site_id=self.env.ee_site_id,
                joint_qpos_indices=np.arange(7),
                effective_dt=self.effective_dt,
                max_dq=self.env.ACTION_SCALING_FACTOR / self.effective_dt
            )
        except Exception as e:
            # Fallback for stability
            delta_joints = np.zeros(7)
        
        expert_action = np.concatenate([delta_joints, [self.latest_expert_grip]])
        
        # 2. Step the Environment with EXPERT action
        next_obs, reward, terminated, truncated, info = self.env.step(expert_action)
        
        # 3. Handle Expert Finish / Done
        if self.expert.is_done():
            # If expert says done, we treat it as terminated (success or failure)
            terminated = True
            
        # 4. Compute NEXT Expert Target (for the NEW observation)
        #    This prepares 'latest_expert_pose' for the NEXT step() call.
        expert_obs = self.env.get_expert_obs()
        target_pose, grip, exp_info = self.expert.get_target_pose(expert_obs)
        
        self.latest_expert_pose = target_pose
        self.latest_expert_grip = grip
        
        # 5. Populate Info for Trainer
        #    The Trainer needs 'expert_pose' to compare against the Policy's prediction for 'next_obs'
        #    Wait, standard RL loop:
        #      obs_t -> Policy -> Pred_t.
        #      obs_t -> Expert -> Target_t.
        #      Loss(Pred_t, Target_t).
        #      Env.step(Target_t).
        #      -> obs_{t+1}.
        
        #    Here, we return obs_{t+1}.
        #    The info needs to contain Target_{t+1} (calculated from obs_{t+1}) 
        #    so the implementation in collect_rollouts can batch it easily.
        
        info['expert_pose'] = target_pose
        info['expert_grip'] = grip
        info['expert_phase'] = exp_info.get('phase', 'UNKNOWN')
        info['executed_action'] = expert_action # Useful for debugging/buffer
        
        return next_obs, reward, terminated, truncated, info

    def get_expert_obs(self):
        return self.env.get_expert_obs()

    def get_ee_pose(self):
        return self.env.get_ee_pose()
