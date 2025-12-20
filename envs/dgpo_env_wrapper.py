
import gymnasium as gym
import numpy as np
import torch
import mujoco
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
        
        # [CRITICAL FIX] Update Action Space to include Alpha (9D)
        # This prevents AsyncVectorEnv from truncating the 9th element
        low = np.full(9, -1.0, dtype=np.float32)
        high = np.full(9, 1.0, dtype=np.float32)
        # Alpha is [0, 1] but we keep bounds [-1, 1] for simplicity (clipped later)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        
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
        
        # [ENHANCED v3.0] Accurate Goal Image Generation
        # Uses specific IK logic and robot positioning to match training distribution
        try:
            info['goal_img'] = self._render_accurate_goal_image(obs)
        except Exception as e:
            log.warning(f"Goal Image Generation Failed: {e}")
            # Fallback to simple render or zeros
            try:
                info['goal_img'] = self.env._render_goal_image()
            except:
                info['goal_img'] = np.zeros_like(obs['image_primary'])
        
        return obs, info

    def _render_accurate_goal_image(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Renders the goal image using the EXACT logic from `evaluate/test_visualize_goal.py`.
        This ensures the Semantic Planner receives goal images consistent with its training distribution.
        """
        # Save current state
        saved_qpos = self.env.data.qpos.copy()
        saved_qvel = self.env.data.qvel.copy()
        saved_ctrl = self.env.data.ctrl.copy()
        
        goal_pos_world = obs['goal_pos_world']
        goal_orn_world = obs['goal_orn_world']
        HOVER_HEIGHT = 0.10
        
        try:
            # 1. Move Object to Goal Pose (Lifted)
            obj_addr = self.env.model.jnt_qposadr[self.env.object_joint_id]
            # [MATCHING LOGIC] Lift object by 0.02 to prevent sinking
            target_obj_pos = goal_pos_world + np.array([0.0, 0.0, 0.02])
            self.env.data.qpos[obj_addr:obj_addr+3] = target_obj_pos
            
            # Set Orientation: Convert xyzw (SciPy) -> wxyz (MuJoCo)
            goal_orn_wxyz = self.env._scipy_xyzw_to_mujoco_wxyz(goal_orn_world)
            self.env.data.qpos[obj_addr+3:obj_addr+7] = goal_orn_wxyz
            
            # 2. Calculate Robot Target Pose (Goal Pos + Hover Z)
            # [MATCHING LOGIC] Lift robot by HOVER_HEIGHT + 0.02
            target_pos = goal_pos_world + np.array([0.0, 0.0, HOVER_HEIGHT + 0.02])
            
            # 3. Calculate DYNAMIC Target Orientation
            # [MATCHING LOGIC] Use seed [1, 0, 0, 0] (Rot X 180) which matches test_visualize_goal.py
            # The expert default is [0, 1, 0, 0], so we MUST override it here.
            seed_downward_quat = np.array([1.0, 0.0, 0.0, 0.0])
            
            # Reuse expert's robust alignment logic
            target_quat = self.expert._calculate_aligned_orientation(goal_orn_world, seed_downward_quat)
            
            target_pose_7d = np.concatenate([target_pos, target_quat])
            
            # 4. Solve IK for Hover Pose
            # Seed with home position (standard neutral pose)
            home_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
            
            goal_qpos = self.ik_solver.solve_ik_static(
                target_pose=target_pose_7d, 
                model=self.env.model, 
                data=self.env.data, 
                ee_site_id=self.env.ee_site_id,
                q0=home_qpos 
            )
            
            if goal_qpos is not None:
                self.env.data.qpos[:7] = goal_qpos
            else:
                self.env.data.qpos[:7] = home_qpos # Fallback
    
            # 5. Open Gripper (Retract state has open gripper)
            # 0.04 represents full open in this env
            self.env.data.qpos[7:9] = 0.04 
            
            # 6. Forward Prop
            mujoco.mj_forward(self.env.model, self.env.data)
            
            # 7. Render
            # Use the environment's render method to get the correct camera/settings
            goal_img = self.env.render()
            
            return goal_img
            
        finally:
            # Restore State
            self.env.data.qpos[:] = saved_qpos
            self.env.data.qvel[:] = saved_qvel
            self.env.data.ctrl[:] = saved_ctrl
            mujoco.mj_forward(self.env.model, self.env.data)
        
    def step(self, action):
        """
        Executes a step with PROGRESSIVE POLICY BLENDING in POSE SPACE.
        
        [ENHANCED DGPO v3.0 - CRITICAL FIX]
        Blending happens at the POSE level (not joint level):
        - Blended_pose = (1-α)*Expert_pose + α*Policy_pose
        - Joint_command = IK(Blended_pose)
        
        Args:
            action: Can be:
                   - 8D: [7 pose (x,y,z,qw,qx,qy,qz) + 1 gripper]
                   - 9D: [7 pose + 1 gripper + 1 alpha]
                   If blending disabled, policy action is IGNORED.
            
        Returns:
            Standard Gym 5-tuple. 'info' contains the 'expert_pose' for the NEXT step.
        """
        
        # 1. Get Expert target pose (already computed)
        expert_pose = self.latest_expert_pose
        expert_grip = self.latest_expert_grip
        
        # [DEBUG] Print status (will show in console)
        # print(f"DEBUG: use_blending={self.use_blending}, action type={type(action)}")
        
        # 2. [CRITICAL FIX] Blend at POSE level, not joint level
        if self.use_blending and action is not None:
            action_flat = np.asarray(action).flatten()
            
            # [DEBUG] Check action shape
            # if len(action_flat) < 9:
            #    print(f"DEBUG: Action shape {action_flat.shape} < 9! Using internal alpha {self.blend_alpha}")
            
            # Extract policy pose and alpha
            if len(action_flat) >= 9:
                policy_pose = action_flat[:7]
                policy_grip = action_flat[7]
                alpha = float(np.clip(action_flat[8], 0.0, 1.0))
            else:
                policy_pose = action_flat[:7]
                policy_grip = action_flat[7] if len(action_flat) > 7 else expert_grip
                alpha = self.blend_alpha
            
            # [DEBUG] Verify alpha
            # if alpha == 0.0 and self.blend_alpha > 0:
            #    print(f"DEBUG: Alpha is 0.0 but self.blend_alpha is {self.blend_alpha}")

            
            # Blend POSES (physically meaningful!)
            blended_pose = (1 - alpha) * expert_pose + alpha * policy_pose
            blended_grip = (1 - alpha) * expert_grip + alpha * policy_grip
            
            # Normalize quaternion in blended pose
            quat = blended_pose[3:7]
            quat_norm = np.linalg.norm(quat)
            if quat_norm > 1e-6:
                blended_pose[3:7] = quat / quat_norm
            
            used_alpha = alpha
        else:
            # Pure Shadow Mode
            blended_pose = expert_pose
            blended_grip = expert_grip
            used_alpha = 0.0
        
        # 3. Convert blended pose to joint commands via IK
        # [AUDIT] Input `blended_pose` is already SciPy format [x,y,z,w] from DGPOExpert/PandaEnv.
        # No permutation needed. Passing directly to IKSolver.
        
        try:
            delta_joints = self.ik_solver.compute_delta_action(
                target_ee_pose=blended_pose,  # [FIX] Singular arg name, 1D array, already xyzw
                model=self.env.model,
                data=self.env.data,
                ee_site_id=self.env.ee_site_id,
                joint_qpos_indices=np.arange(7),
                effective_dt=self.effective_dt,
                max_dq=self.env.ACTION_SCALING_FACTOR / self.effective_dt
            )
        except Exception as e:
            # [DEBUG] Print exception to see why IK fails
            log.warning(f"IK Failed in Step: {e}")
            delta_joints = np.zeros(7)
        
        executed_action = np.concatenate([delta_joints, [blended_grip]])
        
        # [ENHANCED] Capture Expert Action for Divergence Tracking
        
        try:
            expert_delta_joints = self.ik_solver.compute_delta_action(
                target_ee_pose=expert_pose, # [FIX] Singular arg name, already xyzw
                model=self.env.model,
                data=self.env.data,
                ee_site_id=self.env.ee_site_id,
                joint_qpos_indices=np.arange(7),
                effective_dt=self.effective_dt,
                max_dq=self.env.ACTION_SCALING_FACTOR / self.effective_dt
            )
        except Exception as e:
            log.warning(f"Expert IK Failed in Step: {e}")
            expert_delta_joints = delta_joints # Fallback to executed action so diff is 0
        expert_action = np.concatenate([expert_delta_joints, [expert_grip]])
        
        # 4. Step the Environment
        next_obs, reward, terminated, truncated, info = self.env.step(executed_action)
        
        # 5. Handle Expert Finish / Done
        if self.expert.is_done():
            terminated = True
            
        # 6. Compute NEXT Expert Target
        expert_obs = self.env.get_expert_obs()
        target_pose, grip, exp_info = self.expert.get_target_pose(expert_obs)
        
        self.latest_expert_pose = target_pose
        self.latest_expert_grip = grip
        
        # 7. Populate Info for Trainer
        info['expert_pose'] = target_pose
        info['expert_grip'] = grip
        info['expert_phase'] = exp_info.get('phase', 'UNKNOWN')
        info['executed_action'] = executed_action
        info['expert_action'] = expert_action
        info['blend_alpha'] = used_alpha
        info['blended_pose'] = blended_pose  # For debugging
        
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
