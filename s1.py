import numpy as np
import torch

import gymnasium as gym
from gymnasium.spaces import Box, Dict as SpaceDict

from utils.rl_reward_wrapper import RLRewardWrapper
import mujoco
mujoco.mj_name2id = lambda *args, **kwargs: 0


class DummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.observation_space = SpaceDict({
            "image_primary": Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "internal_full_proprio": Box(low=-1.0, high=1.0, shape=(14,), dtype=np.float32),
        })
        self.t = 0

        # Fake MuJoCo API
        class DummyModel: pass
        class DummyData:
            site_xpos = np.zeros((1, 3))
            geom_xpos = np.zeros((1, 3))
            xpos = np.zeros((1, 3))
        self.model = DummyModel()
        self.data = DummyData()

    def reset(self, *, seed=None, options=None):
        self.t = 0
        obs = {
            "image_primary": np.zeros((64, 64, 3), dtype=np.uint8),
            "internal_full_proprio": np.zeros((14,), dtype=np.float32),
        }
        return obs, {}

    def step(self, action):
        self.t += 1
        obs = {
            "image_primary": np.zeros((64, 64, 3), dtype=np.uint8),
            "internal_full_proprio": np.ones((14,), dtype=np.float32) * 0.5,
        }
        reward = 0.0
        terminated = self.t >= 3
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def get_ee_pose(self):
        return np.array([0, 0, 0.5, 0, 0, 0, 1], dtype=np.float32)


class DummyOctoModel:
    """
    Minimal stub OCTO model for schema validation.
    """

    def __init__(self):
        # Example batch should match what our wrapper will produce
        self.example_batch = {
            "observations": {
                "image_primary": np.zeros((1, 1, 64, 64, 3), dtype=np.float32),
                "image_wrist": np.zeros((1, 1, 128, 128, 3), dtype=np.float32),
                "internal_full_proprio": np.zeros((1, 1, 14), dtype=np.float32),
                "task_completed": np.zeros((1, 1, 4), dtype=np.float32),
                "timestep": np.zeros((1, 1), dtype=np.int32),
                "pad_mask_dict": {
                    "image_primary": np.ones((1, 1), dtype=bool),
                    "image_wrist": np.ones((1, 1), dtype=bool),
                    "internal_full_proprio": np.ones((1, 1), dtype=bool),
                    "task_completed": np.ones((1, 1), dtype=bool),
                    "timestep": np.ones((1, 1), dtype=bool),
                },
                "timestep_pad_mask": np.ones((1, 1), dtype=bool),
            }
        }

    def create_tasks(self, texts):
        return {"dummy": 0}

    def sample_actions(self, batch, task):
        # Pretend OCTO outputs pose predictions
        B, T = batch["observations"]["image_primary"].shape[:2]
        # Return (B,T,D) with D=7 (pos+quat)
        return np.tile(np.array([[0, 0, 0.5, 0, 0, 0, 1]], dtype=np.float32), (B, T, 1))


def test_rl_reward_wrapper_divergence():
    env = DummyEnv()
    octo = DummyOctoModel()
    wrapped = RLRewardWrapper(env, octo_model=octo)

    obs, info = wrapped.reset()
    done = False
    while not done:
        action = wrapped.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped.step(action)
        done = terminated or truncated

    # Force divergence calculation
    div = wrapped._calculate_divergence()
    print("Divergence:", div)

    # Check that divergence is a finite float
    assert isinstance(div, float)
    assert np.isfinite(div)


if __name__ == "__main__":
    test_rl_reward_wrapper_divergence()
    print("✅ RLRewardWrapper divergence test passed.")
