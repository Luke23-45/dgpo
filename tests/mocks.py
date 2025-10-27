# FILE: tests/mocks.py (Create this new file)

import gymnasium as gym
from gymnasium.spaces import Box, Dict as DictSpace
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional

# --- Mock MuJoCo Contact ---
@dataclass
class MockMjContact:
    geom1: int = -1
    geom2: int = -1

# --- Mock MuJoCo Data ---
@dataclass
class MockMjData:
    ncon: int = 0
    contact: list = field(default_factory=list) # List of MockMjContact
    # Add other attributes if needed by the wrapper in the future

# --- Mock MuJoCo Model ---
@dataclass
class MockMjModel:
    ngeom: int = 10 # Example number of geoms
    geom_size: np.ndarray = field(default_factory=lambda: np.zeros((10, 3))) # Example geom sizes
    # Add other attributes if needed

# --- Mock Unwrapped Env ---
@dataclass
class MockUnwrappedEnv:
    """Mocks the unwrapped env attributes needed by the reward wrapper."""
    model: MockMjModel = field(default_factory=MockMjModel)
    data: MockMjData = field(default_factory=MockMjData)
    object_geom_id: int = 1 # Example geom ID for the object

    def mj_name2id(self, obj_type, name):
        """Basic mock name to ID mapping."""
        if name == "table_geom":
            return 0 # Table is geom 0
        return -1 # Default unknown

    def mj_id2name(self, obj_type, geom_id):
        """Basic mock ID to name mapping."""
        if geom_id == 0:
            return "table_geom"
        if geom_id in [2, 3, 4]: # Example robot geoms
            return f"link{geom_id}_c0"
        if geom_id == 5:
             return "finger_geom_left" # Excluded from collision check
        return None


# --- Mock Gymnasium Env ---
class MockPandaEnv(gym.Env):
    """
    A mock Gymnasium environment mimicking PandaEnv's observation space
    and allowing manual setting of observation values.
    """
    def __init__(self):
        super().__init__()
        # Define a realistic observation space matching PandaEnv
        self.observation_space = DictSpace({
            'ee_pose_world': Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
            'object_pos_world': Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'object_orn_world': Box(low=-1, high=1, shape=(4,), dtype=np.float32),
            'goal_pos_world': Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'goal_orn_world': Box(low=-1, high=1, shape=(4,), dtype=np.float32),
            'is_grasped': Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'proprio': Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32), # Includes gripper forces
            'object_vel': Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            'robot_base_pos_world': Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            # Add other keys if your AdvancedRewardWrapper._extract_state uses them
        })
        # Define a simple action space
        self.action_space = Box(low=-1, high=1, shape=(8,), dtype=np.float32)

        # Observation buffer that can be set externally
        self._current_obs = {key: np.zeros(space.shape, dtype=space.dtype)
                             for key, space in self.observation_space.spaces.items()}

        # === START OF FIX ===
        # Create the private _unwrapped instance first
        self._unwrapped = MockUnwrappedEnv()
        # === END OF FIX ===
        
        # Set some default sizes needed by reward wrapper init
        # Now access via self._unwrapped
        self._unwrapped.model.geom_size[0] = np.array([1.0, 1.0, 0.01])
        self._unwrapped.model.geom_size[1] = np.array([0.02, 0.02, 0.02])
        self._unwrapped.data.geom_xpos = np.zeros((self._unwrapped.model.ngeom, 3))
        self._unwrapped.data.geom_xpos[0, 2] = 0.4

    # === START OF FIX ===
    @property
    def unwrapped(self):
        """Expose the private _unwrapped attribute as a read-only property."""
        return self._unwrapped
    # === END OF FIX ===

    def set_observation(self, obs_dict: Dict[str, np.ndarray]):
        """Manually set the next observation the env will return."""
        for key, value in obs_dict.items():
            if key in self._current_obs:
                # Ensure correct dtype and shape
                expected_space = self.observation_space[key]
                self._current_obs[key] = np.array(value, dtype=expected_space.dtype).reshape(expected_space.shape)
            else:
                print(f"Warning: Key '{key}' not in mock observation space.")

    def set_collisions(self, collisions: list[Tuple[int, int]]):
        """Set mock collisions (list of geom1, geom2 tuples)."""
        self.unwrapped.data.contact = [MockMjContact(geom1=g1, geom2=g2) for g1, g2 in collisions]
        self.unwrapped.data.ncon = len(collisions)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[dict, dict]:
        super().reset(seed=seed)
        # Return the current observation buffer and empty info
        return self._current_obs.copy(), {}

    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, bool, dict]:
        # In a mock env, step just returns the current obs buffer
        # Reward, terminated, truncated are placeholders - wrapper calculates real reward
        terminated = False
        truncated = False
        reward = 0.0
        info = {}
        return self._current_obs.copy(), reward, terminated, truncated, info

    def render(self):
        # Mock render method if needed for video tests (optional)
        return np.zeros((480, 640, 3), dtype=np.uint8) # Return dummy image