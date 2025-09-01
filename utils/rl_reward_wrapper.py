"""
Robust RLRewardWrapper for Panda pick-and-place tasks.

This Gymnasium wrapper provides a dense, multi-stage reward function suitable
for reinforcement learning on a pick-and-place task. It is designed to be
robust against common simulation issues like missing MuJoCo elements or
unstable physics.

The reward is composed of the following components:
- R_reach: Dense reward for moving the end-effector towards the object.
- R_grasp: Sparse reward for successfully grasping the object.
- R_lift: Sparse reward for lifting the grasped object off the table.
- R_place: Dense reward for moving the lifted object towards the goal.
- R_success: Large sparse reward for placing the object at the goal.
- R_penalty: Small penalty on the magnitude of actions to encourage efficiency.

The wrapper is designed for testability by separating the reward calculation
logic from the environment's physics step.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, List
import gymnasium as gym
import mujoco
import numpy as np
import torch 
logger = logging.getLogger(__name__)


def _safe_norm(vec: np.ndarray) -> float:
    """
    Calculates the L2 norm of a vector with numerical stability.

    If the norm calculation fails or results in a non-finite number (inf, nan),
    it safely returns 0.0.

    Args:
        vec: The input numpy array.

    Returns:
        The L2 norm as a float, or 0.0 if an error occurs.
    """
    try:
        val = float(np.linalg.norm(vec))
        if not np.isfinite(val):
            return 0.0
        return val
    except (ValueError, TypeError):
        return 0.0


class RLRewardWrapper(gym.Wrapper):
    """
    A dense reward wrapper for a pick-and-place task.

    This wrapper computes a multi-component reward and handles episode state
    tracking (e.g., whether an object has been grasped or lifted).

    Args:
        env: The Gymnasium environment to wrap.
        ee_site_name: Name of the end-effector site in the MuJoCo model.
        object_geom_name: Name of the object's geometry.
        goal_body_name: Name of the goal's body.
        gripper_action_index: Index of the gripper control in the action vector.
        reach_scale: Scaling factor for the reach reward component.
        place_scale: Scaling factor for the place reward component.
        grasp_reward: One-time reward for a successful grasp.
        lift_reward: One-time reward for lifting the object.
        success_reward: One-time reward for completing the task.
        lift_z_threshold: Z-coordinate threshold to consider the object "lifted".
        grasp_distance_threshold: Max distance between EE and object for a grasp.
        success_distance_threshold: Max distance between object and goal for success.
        gripper_threshold: Action value threshold to consider the gripper "closing".
        action_penalty: Scaling factor for the L2 action penalty.
        base_reward_weight: Weight to mix in the environment's base reward.
    """
    def __init__(
        self,
        env: gym.Env,
        *,
        ee_site_name: str = "attachment_site",
        object_geom_name: str = "object_geom",
        goal_body_name: str = "goal",
        gripper_action_index: int = 7,
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
        base_reward_weight: float = 0.0,
        octo_model: Optional[Any] = None,
        w_plausibility: float = 0.1,
        quat_format: str = "xyzw", 
        pos_scale: float = 0.05,
        rot_scale: float = 1.0,
        div_clip: float = 10.0,
        div_frame_stride: int = 1,
    ):
        super().__init__(env)

        # Names for MuJoCo elements
        self.ee_site_name = ee_site_name
        self.object_geom_name = object_geom_name
        self.goal_body_name = goal_body_name

        self.gripper_action_index = int(gripper_action_index)

        # Reward parameters
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
        self.base_reward_weight = float(base_reward_weight)

        # Internal state tracking for one-time rewards and dense shaping
        self._last_dist_ee_to_cube: float = 0.0
        self._last_dist_cube_to_goal: float = 0.0
        self._grasp_achieved: bool = False
        self._lift_achieved: bool = False

        # Cache for one-time warnings to avoid log spam
        self._warned = {"ee": False, "geom": False, "goal": False, "gripper": False}
        self.octo_model = octo_model
        self.w_plausibility = w_plausibility
        self.div_clip = div_clip
        self.quat_format = quat_format
        self.pos_weight = 1.0 / (pos_scale**2) if pos_scale > 1e-6 else 1.0
        self.rot_weight = rot_scale
        if self.octo_model:
            self.octo_task = self.octo_model.create_tasks(texts=["pick up the red block"])
        
        self._episode_trajectory: List[Dict[str, np.ndarray]] = []
        self.div_frame_stride = max(1, div_frame_stride)


    def _safe_site_pos(self, name: str) -> Optional[np.ndarray]:
        """Safely get a site's position using the core MuJoCo API."""
        try:
            idx = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if idx == -1:
                raise ValueError(f"Site '{name}' not found.")
            pos = self.env.data.site_xpos[idx]
            return np.array(pos, dtype=np.float64)
        except (ValueError, IndexError):
            if not self._warned["ee"]:
                logger.warning(f"[RewardWrapper] MuJoCo site '{name}' not found.")
                self._warned["ee"] = True
            return None

    def _safe_geom_pos(self, name: str) -> Optional[np.ndarray]:
        """Safely get a geom's position using the core MuJoCo API."""
        try:
            idx = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if idx == -1:
                raise ValueError(f"Geom '{name}' not found.")
            pos = self.env.data.geom_xpos[idx]
            return np.array(pos, dtype=np.float64)
        except (ValueError, IndexError):
            if not self._warned["geom"]:
                logger.warning(f"[RewardWrapper] MuJoCo geom '{name}' not found.")
                self._warned["geom"] = True
            return None

    def _safe_body_pos(self, name: str) -> Optional[np.ndarray]:
        """Safely get a body's position using the core MuJoCo API."""
        try:
            idx = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if idx == -1:
                raise ValueError(f"Body '{name}' not found.")
            pos = self.env.data.xpos[idx]
            return np.array(pos, dtype=np.float64)
        except (ValueError, IndexError):
            if not self._warned["goal"]:
                logger.warning(f"[RewardWrapper] MuJoCo body '{name}' not found.")
                self._warned["goal"] = True
            return None

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Resets the environment and the wrapper's internal state."""
        obs, info = self.env.reset(**kwargs)

        # Reset internal state flags
        self._grasp_achieved = False
        self._lift_achieved = False

        # Recalculate initial distances for dense reward shaping
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
        self._episode_trajectory = []
        return obs, info

    def _calculate_rewards_and_info(
        self, action: np.ndarray, base_reward: float, terminated: bool, info: Dict[str, Any]
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Calculates all reward components based on the current simulation state.
        This method is separated from `step` to allow for direct testing of the
        reward logic without running the physics simulation.
        """
        try:
            if hasattr(action, "cpu") and hasattr(action, "numpy"):
                a_np = action.cpu().numpy().ravel()
            else:
                a_np = np.asarray(action, dtype=float).ravel()
        except Exception:
            a_np = np.zeros_like(self.action_space.sample(), dtype=float)

        ee_pos = self._safe_site_pos(self.ee_site_name)
        cube_pos = self._safe_geom_pos(self.object_geom_name)
        goal_pos = self._safe_body_pos(self.goal_body_name)

        if ee_pos is None or cube_pos is None or goal_pos is None:
            R_penalty = -self.action_penalty * float(np.sum(np.square(a_np)))
            reward_total = self.base_reward_weight * float(base_reward) + R_penalty
            info.update({
                "R_reach": 0.0, "R_grasp": 0.0, "R_lift": 0.0, "R_place": 0.0, 
                "R_success": 0.0, "R_penalty": R_penalty, "warning": "degraded_reward"
            })
            return reward_total, terminated, info

        # -- R_reach: Dense reward for moving EE to the cube --
        dist_ee_to_cube = _safe_norm(ee_pos - cube_pos)
        R_reach = (self._last_dist_ee_to_cube - dist_ee_to_cube) * self.reach_scale
        self._last_dist_ee_to_cube = dist_ee_to_cube

        # -- Gripper state --
        if a_np.size > self.gripper_action_index:
            gripper_value = a_np[self.gripper_action_index]
        else:
            gripper_value = a_np[-1]
            if not self._warned["gripper"]:
                logger.warning(
                    f"[RewardWrapper] gripper_action_index={self.gripper_action_index} "
                    f"out of bounds for action size {a_np.size}; using last element."
                )
                self._warned["gripper"] = True
        is_gripping = float(gripper_value) > self.gripper_threshold

        # -- R_grasp: Sparse reward for grasping the cube --
        R_grasp = 0.0
        if is_gripping and dist_ee_to_cube < self.grasp_distance_threshold and not self._grasp_achieved:
            R_grasp = self.grasp_reward
            self._grasp_achieved = True

        # -- R_lift: Sparse reward for lifting the cube --
        is_lifted = cube_pos[2] > self.lift_z_threshold
        R_lift = 0.0
        if self._grasp_achieved and is_lifted and not self._lift_achieved:
            R_lift = self.lift_reward
            self._lift_achieved = True

        # -- R_place: Dense reward for moving cube to the goal --
        dist_cube_to_goal = _safe_norm(cube_pos[:2] - goal_pos[:2])
        R_place = 0.0
        if self._grasp_achieved and is_lifted:
            R_place = (self._last_dist_cube_to_goal - dist_cube_to_goal) * self.place_scale
            self._last_dist_cube_to_goal = dist_cube_to_goal

        # -- R_success: Sparse reward for task completion --
        R_success = 0.0
        if self._grasp_achieved and dist_cube_to_goal < self.success_distance_threshold:
            R_success = self.success_reward
            terminated = True  # Terminate episode on success

        # -- R_penalty: Penalty for large actions --
        R_penalty = -self.action_penalty * float(np.sum(np.square(a_np)))

        # -- Total Reward --
        reward_total = (
            R_reach + R_grasp + R_lift + R_place + R_success + R_penalty +
            self.base_reward_weight * float(base_reward)
        )
        reward_total = float(np.nan_to_num(reward_total))

        # -- Update info dictionary for logging/debugging --
        info.update({
            "R_reach": float(R_reach),
            "R_grasp": float(R_grasp),
            "R_lift": float(R_lift),
            "R_place": float(R_place),
            "R_success": float(R_success),
            "R_penalty": float(R_penalty),
            "dist_ee_to_cube": float(dist_ee_to_cube),
            "dist_cube_to_goal": float(dist_cube_to_goal),
            "is_gripping": bool(is_gripping),
            "is_lifted": bool(is_lifted),
            "gripper_value": float(gripper_value),
        })

        return reward_total, terminated, info

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Executes one step in the environment, computes the custom reward,
        and returns the results.
        """
        # Clip action to avoid MuJoCo instability from extreme values
        if hasattr(self.env.action_space, "low"):
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

        # 1. Run the physics simulation in the underlying environment
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        if self.octo_model and hasattr(self.env.unwrapped, 'get_ee_pose'):
            current_ee_pose = self.env.unwrapped.get_ee_pose()
            self._episode_trajectory.append({
                "obs": obs,
                "ee_pose": current_ee_pose
            })
        # 2. Calculate our custom rewards on the new state
        reward_total, terminated, info = self._calculate_rewards_and_info(
            action=action, base_reward=base_reward, terminated=terminated, info=info
        )
        if self.octo_model and (terminated or truncated):
            divergence_mse = self._calculate_divergence()
            # Clip the raw divergence score before applying weight
            clipped_divergence = np.clip(divergence_mse, 0.0, self.div_clip)
            R_T = -self.w_plausibility * clipped_divergence
            
            reward_total += R_T
            info['R_T_divergence'] = float(R_T)
            info['divergence_raw'] = float(divergence_mse) # Telemetry
            info['divergence_clipped'] = float(clipped_divergence) # Telemetry
            info['divergence_enabled'] = True
            info['divergence_steps'] = len(self._episode_trajectory)
                
        reward_total = float(np.nan_to_num(reward_total))
        return obs, reward_total, terminated, truncated, info

    @torch.inference_mode()
    def _calculate_divergence(self) -> float:
        """
        Calculates a scaled, robust divergence between the agent's trajectory
        and OCTO's predictions. This version includes critical robustness checks.
        """
        if not self._episode_trajectory or self.octo_model is None:
            return 0.0

        # 1. Subsample the trajectory to save memory and compute
        trajectory = self._episode_trajectory[::self.div_frame_stride]
        if not trajectory:
            return 0.0

        required_keys = ("image_primary", "internal_full_proprio")
        if not all(k in trajectory[0]['obs'] for k in required_keys):
            logger.error(f"Missing one of required keys {required_keys} in observation; skipping divergence.")
            return 0.0

        obs_list = [t['obs'] for t in trajectory]
        agent_poses = np.stack([t['ee_pose'] for t in trajectory])

        # 2. Prepare data for OCTO, ensuring correct dtype and format
        images_batch = np.stack([np.asarray(o["image_primary"]) for o in obs_list])
        proprios_batch = np.stack([np.asarray(o["internal_full_proprio"]) for o in obs_list])

        # Ensure images are HWC float32 in [0, 1] for OCTO
        if images_batch.shape[1] in (1, 3):
            images_batch = np.transpose(images_batch, (0, 2, 3, 1))
        if images_batch.dtype == np.uint8:
            images_batch = images_batch.astype(np.float32) / 255.0
        
        octo_input = {
            "image_primary": images_batch[:, np.newaxis, ...],
            "proprio": proprios_batch[:, np.newaxis, ...],
        }

        # 3. Get OCTO predictions with error handling
        try:
            predicted_actions_raw = self.octo_model.sample_actions(octo_input, self.octo_task)
            octo_poses = np.asarray(predicted_actions_raw[:, 0, :])
        except Exception as e:
            logger.error(f"OCTO inference failed during divergence calculation: {e}")
            return 0.0 # Return zero divergence on failure

        # 4. Align and normalize poses
        T = min(len(agent_poses), len(octo_poses))
        if T == 0: return 0.0

        pos_error = agent_poses[:, :3] - octo_poses[:, :3]
        pos_mse = float(np.mean(np.sum(pos_error * pos_error, axis=-1)))
        agent_poses, octo_poses = agent_poses[:T], octo_poses[:T]

        def _normalize(q: np.ndarray) -> np.ndarray:
            """Normalizes a batch of quaternions to unit length."""
            norm = np.linalg.norm(q, axis=-1, keepdims=True)
            return q / np.clip(norm, 1e-8, None)

        q_agent = _normalize(agent_poses[:, 3:])
        q_octo_raw = octo_poses[:, 3:]

        if self.quat_format == "wxyz":
            q_octo_raw = q_octo_raw[:, [1, 2, 3, 0]]

        q_octo = _normalize(q_octo_raw)

        dot_product = np.clip(np.abs(np.sum(q_agent * q_octo, axis=-1)), -1.0, 1.0)
        quat_dist_mean = float(np.mean(1.0 - dot_product**2))

        divergence = self.pos_weight * pos_mse + self.rot_weight * quat_dist_mean
        return float(divergence)