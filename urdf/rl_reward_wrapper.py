# In file: utils/rl_reward_wrapper.py
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, List
import gymnasium as gym
import mujoco
import numpy as np

# Import the ScriptedExpert, which will be our new source of guidance.
from utils.scripted_expert import ScriptedExpert

logger = logging.getLogger(__name__)


def _safe_norm(vec: np.ndarray) -> float:
    """Calculates the L2 norm of a vector with numerical stability."""
    try:
        val = float(np.linalg.norm(vec))
        return val if np.isfinite(val) else 0.0
    except (ValueError, TypeError):
        return 0.0


class RLRewardWrapper(gym.Wrapper):
    """
    A dense reward wrapper for a pick-and-place task, now with the option to use
    a ScriptedExpert for a terminal guidance reward.
    """
    def __init__(
        self,
        env: gym.Env,
        *,
        # --- NEW ARGUMENTS for ScriptedExpert guidance ---
        scripted_expert: Optional[ScriptedExpert] = None,
        w_guidance: float = 0.1,
        guidance_clip: float = 1.0,
        # --- Standard task reward arguments ---
        ee_site_name: str = "attachment_site",
        object_geom_name: str = "object_geom",
        goal_body_name: str = "goal",
        reach_scale: float = 10.0,
        place_scale: float = 20.0,
        grasp_reward: float = 2.0,
        lift_reward: float = 5.0,
        success_reward: float = 100.0,
        lift_z_threshold: float = 0.45,
        grasp_distance_threshold: float = 0.04,
        success_distance_threshold: float = 0.05,
        gripper_threshold: float = 0.5,
        action_penalty: float = 0.001,
    ):
        super().__init__(env)

        # Store the expert instance and its parameters
        self.scripted_expert = scripted_expert
        self.w_guidance = w_guidance
        self.guidance_clip = guidance_clip
        self._episode_trajectory: List[Dict[str, np.ndarray]] = []
        
        # Store parameters for the task reward
        self.ee_site_name = ee_site_name
        self.object_geom_name = object_geom_name
        self.goal_body_name = goal_body_name
        self.reach_scale = float(reach_scale)
        self.place_scale = float(place_scale)
        self.grasp_reward = float(grasp_reward)
        self.lift_reward = float(lift_reward)
        self.success_reward = float(success_reward)
        self.lift_z_threshold = float(lift_z_threshold)
        self.grasp_distance_threshold = float(grasp_distance_threshold)
        self.success_distance_threshold = float(success_distance_threshold)
        self.gripper_threshold = float(gripper_threshold)
        self.action_penalty = float(action_penalty)

        # Internal state for reward shaping
        self._last_dist_ee_to_cube: float = 0.0
        self._last_dist_cube_to_goal: float = 0.0
        self._grasp_achieved: bool = False
        self._lift_achieved: bool = False
        self._warned = {"ee": False, "geom": False, "goal": False}

    def _safe_site_pos(self, name: str) -> Optional[np.ndarray]:
        try:
            idx = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if idx == -1: raise ValueError()
            return np.array(self.env.data.site_xpos[idx], dtype=np.float64)
        except Exception:
            if not self._warned["ee"]: logger.warning(f"MuJoCo site '{name}' not found."); self._warned["ee"] = True
            return None

    def _safe_geom_pos(self, name: str) -> Optional[np.ndarray]:
        try:
            idx = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if idx == -1: raise ValueError()
            return np.array(self.env.data.geom_xpos[idx], dtype=np.float64)
        except Exception:
            if not self._warned["geom"]: logger.warning(f"MuJoCo geom '{name}' not found."); self._warned["geom"] = True
            return None

    def _safe_body_pos(self, name: str) -> Optional[np.ndarray]:
        try:
            idx = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if idx == -1: raise ValueError()
            return np.array(self.env.data.xpos[idx], dtype=np.float64)
        except Exception:
            if not self._warned["goal"]: logger.warning(f"MuJoCo body '{name}' not found."); self._warned["goal"] = True
            return None

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        
        self._episode_trajectory = []
        self._grasp_achieved = False
        self._lift_achieved = False
        
        ee_pos = self._safe_site_pos(self.ee_site_name)
        cube_pos = self._safe_geom_pos(self.object_geom_name)
        goal_pos = self._safe_body_pos(self.goal_body_name)

        if ee_pos is not None and cube_pos is not None:
            self._last_dist_ee_to_cube = _safe_norm(ee_pos - cube_pos)
        else:
            self._last_dist_ee_to_cube = 0.0

        if cube_pos is not None and goal_pos is not None:
            self._last_dist_cube_to_goal = _safe_norm(cube_pos[:2] - goal_pos[:2])
        else:
            self._last_dist_cube_to_goal = 0.0

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        
        # Collect rich trajectory data if expert guidance is enabled
        if self.scripted_expert and hasattr(self.env.unwrapped, 'get_expert_obs'):
            self._episode_trajectory.append(self.env.unwrapped.get_expert_obs())

        # Calculate the primary task reward
        reward_total, terminated, info = self._calculate_task_reward(
            action=action, terminated=terminated, info=info
        )

        # At the end of the episode, add the terminal guidance reward
        if self.scripted_expert and (terminated or truncated):
            divergence = self._calculate_scripted_divergence()
            clipped_divergence = np.clip(divergence, 0.0, self.guidance_clip)
            R_T_guidance = -self.w_guidance * clipped_divergence
            
            reward_total += R_T_guidance
            info['R_T_guidance'] = R_T_guidance
            info['guidance_divergence_raw'] = divergence
                
        reward_total = float(np.nan_to_num(reward_total))
        return obs, reward_total, terminated, truncated, info
    
    def _calculate_scripted_divergence(self) -> float:
        """
        Calculates the divergence between the agent's trajectory and the ideal path
        of the stateless ScriptedExpert. Returns Mean Squared Positional Error.
        """
        if not self._episode_trajectory:
            return 0.0

        # Create a temporary, stateless expert to get the ideal poses for the trajectory
        temp_expert = ScriptedExpert(self.scripted_expert.object)
        temp_expert.reset()
        
        total_sq_error = 0.0
        for expert_obs_at_step in self._episode_trajectory:
            agent_pose = expert_obs_at_step['ee_pose_world']
            
            # Get what the expert's target pose would have been from the same state
            ideal_pose, _ = temp_expert.get_target_pose(
                expert_obs_at_step['ee_pose_world'],
                expert_obs_at_step['object_pos_world'],
                expert_obs_at_step['object_orn_world'],
                expert_obs_at_step['goal_pos_world'],
                expert_obs_at_step['is_grasped']
            )
            # temp_expert's state machine advances with each call, simulating the ideal path.
            
            total_sq_error += np.sum(np.square(agent_pose[:3] - ideal_pose[:3]))

        return total_sq_error / len(self._episode_trajectory)

    def _calculate_task_reward(
        self, action: np.ndarray, terminated: bool, info: Dict[str, Any]
    ) -> Tuple[float, bool, Dict[str, Any]]:
        # This function contains the original, reliable task-based reward logic.
        a_np = np.asarray(action, dtype=float).ravel()
        ee_pos = self._safe_site_pos(self.ee_site_name)
        cube_pos = self._safe_geom_pos(self.object_geom_name)
        goal_pos = self._safe_body_pos(self.goal_body_name)
        
        if ee_pos is None or cube_pos is None or goal_pos is None:
            R_penalty = -self.action_penalty * float(np.sum(np.square(a_np)))
            info.update({"warning": "degraded_reward"})
            return R_penalty, terminated, info

        dist_ee_to_cube = _safe_norm(ee_pos - cube_pos)
        R_reach = (self._last_dist_ee_to_cube - dist_ee_to_cube) * self.reach_scale
        self._last_dist_ee_to_cube = dist_ee_to_cube

        gripper_value = a_np[-1] # Simple assumption: gripper is the last action dimension
        is_gripping = gripper_value > self.gripper_threshold

        R_grasp = 0.0
        if is_gripping and dist_ee_to_cube < self.grasp_distance_threshold and not self._grasp_achieved:
            R_grasp = self.grasp_reward
            self._grasp_achieved = True

        is_lifted = cube_pos[2] > self.lift_z_threshold
        R_lift = 0.0
        if self._grasp_achieved and is_lifted and not self._lift_achieved:
            R_lift = self.lift_reward
            self._lift_achieved = True

        dist_cube_to_goal = _safe_norm(cube_pos[:2] - goal_pos[:2])
        R_place = 0.0
        if self._grasp_achieved and is_lifted:
            R_place = (self._last_dist_cube_to_goal - dist_cube_to_goal) * self.place_scale
            self._last_dist_cube_to_goal = dist_cube_to_goal

        R_success = 0.0
        if self._grasp_achieved and dist_cube_to_goal < self.success_distance_threshold:
            R_success = self.success_reward
            terminated = True

        R_penalty = -self.action_penalty * float(np.sum(np.square(a_np)))

        reward_total = R_reach + R_grasp + R_lift + R_place + R_success + R_penalty
        
        info.update({
            "R_reach": R_reach, "R_grasp": R_grasp, "R_lift": R_lift, 
            "R_place": R_place, "R_success": R_success, "R_penalty": R_penalty
        })
        
        return float(reward_total), terminated, info