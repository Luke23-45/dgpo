# FILE: goal_env_wrapper.py
# (State-of-the-Art, Configurable, Multi-Component-Goal Version)

"""
An advanced, configurable goal-based environment wrapper for Hindsight Experience
Replay (HER). [cite: 266]

This wrapper transforms a standard environment into one compatible with goal-conditioned
RL algorithms like HER. [cite: 267] It goes beyond a minimal implementation by offering key
flexibility features that are crucial for modern robotics research. [cite: 268]

Key Advancements:
  - **Configurable Multi-Component Goals**: Users specify a mapping of goal
    components (e.g., 'position', 'orientation') to keys in the environment's
    observation dictionary (e.g., 'object_pos_world', 'object_orn_world'). [cite: 269]
    This allows the wrapper to be easily adapted to complex, multi-part goals.
  - **Mathematically Sound Reward Computation**: The `compute_reward` method
    intelligently handles multi-component goals. It computes distances for
    position (L2 norm) and orientation (angular distance) separately and
    combines them, preventing the mathematically invalid operation of adding
    meters and quaternion components.
  - **Multiple Reward Computation Modes**: The wrapper supports several vectorized
    reward functions ('binary', 'dense', 'shaped'), selectable via a
    configuration parameter. [cite: 270, 271, 272, 293]
  - **Robust and Type-Safe**: Built with clear interfaces, type hinting, and
    extensive validation to ensure ease of use and integration. [cite: 273, 274]
"""

from typing import Literal, Optional, Dict as TypingDict, List
import gymnasium as gym
from gymnasium.spaces import Box, Dict as DictSpace
import numpy as np
from scipy.spatial.transform import Rotation as R

# Define the types for reward modes
RewardType = Literal['binary', 'dense', 'shaped']


class HERGoalEnvWrapper(gym.Wrapper):
    """
    A flexible wrapper to make an environment compatible with Hindsight Experience Replay (HER).

    This wrapper reshapes the observation space and provides a configurable, vectorized
    `compute_reward` method essential for HER's goal relabeling strategy. [cite: 275, 276]
    It is designed to handle multi-component goals, such as position and orientation,
    by correctly processing them from the base environment's observation dictionary.
    """

    def __init__(
        self,
        env: gym.Env,
        achieved_goal_keys: TypingDict[str, str],
        desired_goal_keys: TypingDict[str, str],
        reward_type: RewardType = 'binary',
        pos_dist_thresh: float = 0.04,
        orn_dist_thresh: float = 0.1,
        shaped_reward_scale: float = 5.0,
        orientation_weight: float = 0.1
    ):
        """
        Args:
            env: The environment to wrap. [cite: 277]
            achieved_goal_keys: A mapping from a goal component (e.g., 'position')
                to the key in the observation dict for the achieved state
                (e.g., 'object_pos_world').
            desired_goal_keys: A mapping from a goal component (e.g., 'position')
                to the key in the observation dict for the desired state
                (e.g., 'goal_pos_world').
            reward_type: The type of reward to compute ('binary', 'dense', or 'shaped'). [cite: 280]
            pos_dist_thresh: The tolerance for position distance (in meters) for
                the binary reward mode. [cite: 281]
            orn_dist_thresh: The tolerance for orientation distance (in radians)
                for the binary reward mode.
            shaped_reward_scale: The scaling factor 'k' for the shaped reward exp(-k * dist). [cite: 282]
            orientation_weight: The weight to apply to the orientation distance when
                calculating 'dense' or 'shaped' rewards.
        """
        super().__init__(env)
        self.achieved_goal_keys = achieved_goal_keys
        self.desired_goal_keys = desired_goal_keys
        self.reward_type = reward_type
        self.pos_dist_thresh = pos_dist_thresh
        self.orn_dist_thresh = orn_dist_thresh
        self.shaped_reward_scale = shaped_reward_scale
        self.orientation_weight = orientation_weight

        if not isinstance(env.observation_space, DictSpace):
            raise ValueError("HERGoalEnvWrapper requires an environment with a Dict observation space.") 

        if self.achieved_goal_keys.keys() != self.desired_goal_keys.keys():
            raise ValueError(f"Achieved goal keys {self.achieved_goal_keys.keys()} and "
                             f"desired goal keys {self.desired_goal_keys.keys()} do not match.")

        self.goal_components = sorted(list(self.achieved_goal_keys.keys()))
        if 'position' not in self.goal_components:
            raise ValueError("`achieved_goal_keys` must contain a 'position' component.")

        # --- Validate keys and build new goal spaces ---
        self.achieved_goal_space = self._build_goal_space(self.achieved_goal_keys, env.observation_space)
        self.desired_goal_space = self._build_goal_space(self.desired_goal_keys, env.observation_space)

        # --- Define the new HER-compatible observation space ---
        self.observation_space = DictSpace({
            'observation': env.observation_space,
            'achieved_goal': self.achieved_goal_space,
            'desired_goal': self.desired_goal_space
        }) 

    def _build_goal_space(self, key_mapping: TypingDict[str, str], env_obs_space: DictSpace) -> DictSpace:
        """Helper to validate keys and construct a new DictSpace for the goal."""
        goal_spaces = {}
        for component, env_key in key_mapping.items():
            if env_key not in env_obs_space.spaces:
                raise ValueError(f"Goal key '{env_key}' (for component '{component}') "
                                 f"not in environment observation space.") 
            goal_spaces[component] = env_obs_space.spaces[env_key]
        return DictSpace(goal_spaces)

    def reset(self, **kwargs) -> tuple[TypingDict, TypingDict]:
        obs, info = self.env.reset(**kwargs)
        return self._obs_to_her_dict(obs), info 

    def step(self, action: np.ndarray) -> tuple[TypingDict, float, bool, bool, TypingDict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        # The reward from the base env is ignored, as HER will compute its own.
        # However, for algorithms like TD3+BC, we might use the env's reward.
        # Here we pass the env's reward through, as it's the standard gym.Wrapper behavior.
        return self._obs_to_her_dict(obs), reward, terminated, truncated, info

    def _obs_to_her_dict(self, obs: TypingDict) -> TypingDict:
        """Converts the environment's flat observation into the HER-compatible dict format."""
        achieved_goal = {
            component: obs[env_key]
            for component, env_key in self.achieved_goal_keys.items()
        }
        desired_goal = {
            component: obs[env_key]
            for component, env_key in self.desired_goal_keys.items()
        }

        return {
            'observation': obs,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal
        }

    def compute_reward(
        self,
        achieved_goal: TypingDict[str, np.ndarray],
        desired_goal: TypingDict[str, np.ndarray],
        info: Optional[TypingDict] = None  # Info is unused but part of the API [cite: 287]
    ) -> np.ndarray:
        """
        Computes the vectorized reward for HER based on the selected reward type. [cite: 288]
        This method is called by the HER replay buffer to relabel goals. [cite: 289]

        Args:
            achieved_goal: A batch of achieved goals, as a dictionary of components
                           (e.g., {'position': (N, 3), 'orientation': (N, 4)}).
            desired_goal: A batch of desired goals, in the same format. [cite: 290]
            info: Auxiliary information (unused). [cite: 291]

        Returns:
            A batch of rewards, shape (N,). [cite: 292]
        """
        # --- Position Distance (Required) ---
        pos_ach = achieved_goal['position']
        pos_des = desired_goal['position']
        pos_dist = np.linalg.norm(pos_ach - pos_des, axis=-1)
        is_pos_success = (pos_dist < self.pos_dist_thresh)

        # Initialize final success and combined distance
        is_success = is_pos_success
        combined_dist = pos_dist

        # --- Orientation Distance (Optional) ---
        if 'orientation' in self.achieved_goal_space.spaces:
            orn_ach = achieved_goal['orientation']
            orn_des = desired_goal['orientation']

            try:
                # Use Scipy to compute batched quaternion distance
                R_ach = R.from_quat(orn_ach)
                R_des = R.from_quat(orn_des)
                # This computes the angle of the difference rotation, in radians
                orn_dist = (R_ach * R_des.inv()).magnitude()
            except Exception:
                # Fallback for invalid quaternions
                orn_dist = np.full_like(pos_dist, np.pi)

            is_orn_success = (orn_dist < self.orn_dist_thresh)
            is_success = np.logical_and(is_success, is_orn_success)
            combined_dist = combined_dist + self.orientation_weight * orn_dist

        # --- Compute Final Reward Based on Mode ---
        if self.reward_type == 'binary':
            # Returns 0.0 for success, -1.0 for failure. [cite: 292]
            return -(~is_success).astype(np.float32)
        elif self.reward_type == 'dense':
            # Returns the negative weighted combined distance.
            return -combined_dist
        elif self.reward_type == 'shaped':
            # Returns a smooth reward based on the combined distance. [cite: 293]
            return np.exp(-self.shaped_reward_scale * combined_dist) - 1.0
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")