# FILE: goal_env_wrapper.py
# (State-of-the-Art, Configurable, Multi-Reward-Mode Version)

"""
An advanced, configurable goal-based environment wrapper for Hindsight Experience
Replay (HER).

This wrapper transforms a standard environment into one compatible with goal-conditioned
RL algorithms like HER. It goes beyond a minimal implementation by offering key
flexibility features that are crucial for modern robotics research.

Key Advancements:
  - **Configurable Goal Representation**: Users can specify which keys from the
    observation dictionary should be used as the 'achieved_goal' and 'desired_goal'.
    This allows the wrapper to be easily adapted to tasks where the goal is more
    than just an object's position (e.g., including orientation).

  - **Multiple Reward Computation Modes**: The wrapper supports several vectorized
    reward functions, selectable via a configuration parameter. This allows
    researchers to easily experiment with different reward formulations for HER
    without changing code:
      - 'binary': The standard 0/-1 reward for HER.
      - 'dense': Continuous reward based on negative Euclidean distance.
      - 'shaped': A smooth, exponential reward function that provides a bounded
        and continuous signal.

  - **Robust and Type-Safe**: Built with clear interfaces, type hinting, and
    extensive documentation to ensure ease of use and integration into complex
    training pipelines.
"""

from typing import Literal, Optional, Dict as TypingDict
import gymnasium as gym
from gymnasium.spaces import Box, Dict as DictSpace
import numpy as np

# Define the types for reward modes
RewardType = Literal['binary', 'dense', 'shaped']

class HERGoalEnvWrapper(gym.Wrapper):
    """
    A flexible wrapper to make an environment compatible with Hindsight Experience Replay (HER).

    This wrapper reshapes the observation space and provides a configurable, vectorized
    `compute_reward` method essential for HER's goal relabeling strategy.
    """

    def __init__(
        self,
        env: gym.Env,
        achieved_goal_key: str = 'object_pos_world',
        desired_goal_key: str = 'goal_pos_world',
        reward_type: RewardType = 'binary',
        goal_dist_thresh: float = 0.04,
        shaped_reward_scale: float = 5.0
    ):
        """
        Args:
            env: The environment to wrap.
            achieved_goal_key: The key in the observation dict for the achieved goal.
            desired_goal_key: The key in the observation dict for the desired goal.
            reward_type: The type of reward to compute ('binary', 'dense', or 'shaped').
            goal_dist_thresh: The tolerance for the binary reward mode.
            shaped_reward_scale: The scaling factor 'k' for the shaped reward exp(-k * dist).
        """
        super().__init__(env)
        self.achieved_goal_key = achieved_goal_key
        self.desired_goal_key = desired_goal_key
        self.reward_type = reward_type
        self.goal_dist_thresh = goal_dist_thresh
        self.shaped_reward_scale = shaped_reward_scale

        # --- Validate that the goal keys exist in the original space ---
        if not isinstance(env.observation_space, DictSpace):
            raise ValueError("HERGoalEnvWrapper requires an environment with a Dict observation space.")
        if achieved_goal_key not in env.observation_space.spaces:
            raise ValueError(f"Achieved goal key '{achieved_goal_key}' not in environment observation space.")
        if desired_goal_key not in env.observation_space.spaces:
            raise ValueError(f"Desired goal key '{desired_goal_key}' not in environment observation space.")

        self.achieved_goal_space = env.observation_space.spaces[achieved_goal_key]
        self.desired_goal_space = env.observation_space.spaces[desired_goal_key]

        if self.achieved_goal_space.shape != self.desired_goal_space.shape:
             raise ValueError("Achieved and desired goal spaces must have the same shape.")

        # --- Define the new HER-compatible observation space ---
        self.observation_space = DictSpace({
            'observation': env.observation_space,
            'achieved_goal': self.achieved_goal_space,
            'desired_goal': self.desired_goal_space
        })

    def reset(self, **kwargs) -> tuple[TypingDict, TypingDict]:
        obs, info = self.env.reset(**kwargs)
        return self._obs_to_her_dict(obs), info

    def step(self, action: np.ndarray) -> tuple[TypingDict, float, bool, bool, TypingDict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._obs_to_her_dict(obs), reward, terminated, truncated, info

    def _obs_to_her_dict(self, obs: TypingDict) -> TypingDict:
        """Converts the environment's flat observation into the HER-compatible dict format."""
        return {
            'observation': obs,
            'achieved_goal': obs[self.achieved_goal_key],
            'desired_goal': obs[self.desired_goal_key]
        }

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Optional[TypingDict] = None # Info is unused but part of the API
    ) -> np.ndarray:
        """
        Computes the vectorized reward for HER based on the selected reward type.

        This method is called by the HER replay buffer to relabel goals and calculate
        new rewards for sampled transitions.

        Args:
            achieved_goal: A batch of achieved goals, shape (N, D_goal).
            desired_goal: A batch of desired goals, shape (N, D_goal).
            info: Auxiliary information (unused).

        Returns:
            A batch of rewards, shape (N,).
        """
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)

        if self.reward_type == 'binary':
            # Returns 0.0 for success, -1.0 for failure.
            return -(distance > self.goal_dist_thresh).astype(np.float32)
        elif self.reward_type == 'dense':
            # Returns the negative Euclidean distance.
            return -distance
        elif self.reward_type == 'shaped':
            # Returns a smooth reward between 0 (goal) and e^-k*inf -> -1 (far).
            return np.exp(-self.shaped_reward_scale * distance) - 1.0
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")