import gymnasium as gym
from gymnasium.spaces import Box, Dict
import numpy as np
from envs.panda_env import PandaEnv

class GoalPandaEnv(gym.Wrapper):
    """
    A wrapper to make the PandaEnv compatible with Hindsight Experience Replay (HER).
    It reshapes the observation space and implements a vectorized compute_reward method.
    """
    def __init__(self, env: PandaEnv):
        super().__init__(env)
        # The new observation space required by HER
        self.observation_space = Dict({
            'observation': env.observation_space,
            'achieved_goal': Box(low=-np.inf, high=np.inf, shape=(3,)), # Object position
            'desired_goal': Box(low=-np.inf, high=np.inf, shape=(3,))  # Goal position
        })

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._obs_to_dict(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._obs_to_dict(obs), reward, terminated, truncated, info

    def _obs_to_dict(self, obs: dict) -> dict:
        """Converts the environment's observation into the HER-compatible dict format."""
        return {
            'observation': obs,
            'achieved_goal': obs['object_pos_world'],
            'desired_goal': obs['goal_pos_world']
        }
    
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict) -> np.ndarray:
        """
        HER's reward function. It's binary: 0 for success, -1 for failure.
        Our dense rewards will still be used for the main RL update.
        """
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        # Reward is 0 if the distance is less than a small tolerance, otherwise -1.
        return -(distance > 0.04).astype(np.float32)