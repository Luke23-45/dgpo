# envs/panda_env.py (Final, Production-Grade, OCTO-Compliant Version)
import warnings
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from typing import Tuple, Dict
from scipy.spatial.transform import Rotation as R
from utils.mujoco_utils import set_joint_qpos_by_name

class PandaEnv(gym.Env):
    """
    Final, Production-Grade PandaEnv for OCTO Data Generation.

    This environment is specifically tailored to generate (observation, action) pairs
    for training a policy using the `hf://rail-berkeley/octo-small-1.5` model as an expert.

    Key Features:
    - **OCTO-Compliant Observations**: Produces observation dictionaries that precisely
      match the data structure expected by the OCTO model, including required keys
      like 'image_wrist', 'task_completed', and nested padding masks.
    - **Placeholder Generation**: Safely generates placeholder data (e.g., black images
      for the wrist camera) for required keys that are not available in this simulation.
    - **Decoupled Internal State**: Provides the full 14D proprioceptive state under the
      'internal_full_proprio' key. This key is used by downstream components like the
      IKSolver but is safely ignored by the OCTO model.
    - **API-Level Robustness**: Incorporates defensive programming practices from its
      predecessor, including fallbacks for different MuJoCo API versions for rendering,
      simulation stepping, and object manipulation to prevent crashes.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    def __init__(self, xml_path: str = "envs/panda_pick_place.xml", render_mode: str = "rgb_array"):
        super().__init__()


        assert render_mode is None or render_mode in self.metadata["render_modes"]
        # --- Load model and data ---
        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            raise FileNotFoundError(f"Could not load MuJoCo XML from '{xml_path}'. Error: {e}")

        # --- Renderer ---
        self.render_mode = render_mode
        try:
            self.renderer = mujoco.Renderer(self.model, height=256, width=256)
        except Exception:
            warnings.warn("mujoco.Renderer not available — running headless.")
            self.renderer = None

        # --- Episode bookkeeping ---
        self.max_episode_steps = 250
        self.timestep = 0

        # --- Define Observation and Action Spaces (CRITICAL SECTION) ---
        self._define_spaces()

        # --- Random Number Generator ---
        self.np_random, _ = seeding.np_random(None)

        self.ee_site_name = "attachment_site"
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name)
        if self.ee_site_id == -1:
            raise ValueError(f"Site '{self.ee_site_name}' not found in the MuJoCo model.")

    # (Inside PandaEnv class)
    def _define_spaces(self):
        """Defines observation and action spaces to be fully OCTO-compliant."""
        self.observation_space = spaces.Dict({
            # --- Primary modalities ---
            "image_primary": spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
            "image_wrist":   spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
            "proprio":       spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32),
            # Add the internal key to the space for completeness, even if OCTO ignores it
            "internal_full_proprio": spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32),

            # --- Control fields required by OCTO ---
            "task_completed": spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
            "timestep":       spaces.Box(low=0, high=np.iinfo(np.int32).max, shape=(), dtype=np.int32),

            # --- Nested dictionary for padding masks ---
            "pad_mask_dict": spaces.Dict({
                "image_primary": spaces.MultiBinary(1),
                "image_wrist":   spaces.MultiBinary(1),
                "proprio":       spaces.MultiBinary(1),
                "timestep":      spaces.MultiBinary(1),
                "task_completed":spaces.MultiBinary(1),
            }),
        })
        act_dim = int(getattr(self.model, "nu", 8))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)

        
    def render(self):
        """
        Handles rendering for the 'rgb_array' mode, per Gymnasium API.
        """
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                # Handle the case where the renderer wasn't initialized
                warnings.warn("Renderer not available, returning a black frame.")
                return np.zeros((256, 256, 3), dtype=np.uint8)
            
            try:
                self.renderer.update_scene(self.data, camera="fixed_camera")
            except TypeError: # Fallback for different mujoco-python APIs
                try: self.renderer.update_scene(self.data)
                except Exception: pass
            except Exception: pass

            try:
                image_raw = self.renderer.render()
                return np.asarray(image_raw, dtype=np.uint8)
            except Exception:
                warnings.warn("Failed to render primary image. Returning a black frame.")
                return np.zeros((256, 256, 3), dtype=np.uint8)
        # If other render modes were supported, they would be handled here.

    # (Inside the PandaEnv class in envs/panda_env.py)

    def get_ee_pose(self) -> np.ndarray:
        """
        Calculates and returns the current 7D pose of the end-effector.
        This version is optimized by using a cached site ID.
        """
        mujoco.mj_forward(self.model, self.data)
        
        # Use the cached ID for efficient access
        pos = self.data.site_xpos[self.ee_site_id].copy()
        
        # site_xmat is a flat 9-element array (row-major 3x3 matrix)
        rot_matrix = self.data.site_xmat[self.ee_site_id].copy().reshape(3, 3)
        
        quat_xyzw = R.from_matrix(rot_matrix).as_quat()
        
        return np.concatenate([pos, quat_xyzw]).astype(np.float32)

    def _get_obs(self) -> dict:
        """Returns a single-timestep observation that is fully OCTO-compliant."""
        qpos = np.asarray(self.data.qpos, dtype=np.float32)
        qvel = np.asarray(self.data.qvel, dtype=np.float32)
        proprio = np.concatenate([qpos[:7], qvel[:7]])

        # Create placeholders and control fields with the EXACT shapes and types
        return {
            "image_primary": self.render(),
            "image_wrist": np.zeros((128, 128, 3), dtype=np.uint8),
            "proprio": proprio,
            "internal_full_proprio": proprio, # Keep alias for IK
            "task_completed": np.zeros(4, dtype=np.float32),
            "timestep": np.int32(self.timestep), # scalar int32

            "pad_mask_dict": {
                "image_primary": np.array(True, dtype=bool),
                "image_wrist":   np.array(True, dtype=bool),
                "proprio":       np.array(True, dtype=bool),
                "timestep":      np.array(True, dtype=bool),
                "task_completed":np.array(True, dtype=bool),
            },
        }

    def reset(self, seed: int = None, options: dict = None) -> Tuple[Dict, Dict]:
        """Resets the environment to a new, randomized state."""
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
        
        self.timestep = 0
        mujoco.mj_resetData(self.model, self.data)

        # Randomize object position
        cube_qpos = np.array([
            self.np_random.uniform(0.45, 0.65),
            self.np_random.uniform(-0.1, 0.1),
            0.42, 1.0, 0.0, 0.0, 0.0
        ], dtype=float)
        
        try:
            set_joint_qpos_by_name(self.model, self.data, "object_joint", cube_qpos)
        except Exception as e:
            warnings.warn(f"set_joint_qpos_by_name failed for 'object_joint': {e}")
        
        # Randomize goal position
        goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal")
        if goal_body_id >= 0:
            self.model.body_pos[goal_body_id][:2] = [
                self.np_random.uniform(0.45, 0.65),
                self.np_random.uniform(0.15, 0.25)
            ]
        else:
            warnings.warn("Goal body 'goal' not found in model; skipping goal placement.")

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Applies an action and steps the simulation forward."""
        self.timestep += 1
        
        # --- Robust Action Application ---
        action = np.asarray(action, dtype=float).ravel()
        if action.size != self.model.nu:
            raise ValueError(f"Action dimension mismatch: got {action.size}, expected {self.model.nu}")
        
        try:
            ctrl_range = self.model.actuator_ctrlrange
            lo, hi = ctrl_range[:, 0], ctrl_range[:, 1]
            scaled_action = lo + 0.5 * (action + 1.0) * (hi - lo)
        except Exception:
            scaled_action = action # Pass through if scaling fails
            
        self.data.ctrl[:scaled_action.size] = scaled_action

        # --- Robust Simulation Stepping ---
        try:
            mujoco.mj_step(self.model, self.data, nstep=5)
        except TypeError: # Fallback for APIs that don't support nstep
            for _ in range(5):
                mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = (self.timestep >= self.max_episode_steps)
        
        return obs, reward, terminated, truncated, {}

    def close(self):
        """Cleans up resources, primarily the renderer."""
        if hasattr(self, "renderer") and self.renderer is not None:
            self.renderer.close()

    def get_base_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the world-frame pose of the robot's base ('link0')."""
        base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link0")
        if base_body_id < 0:
            raise ValueError("Body 'link0' not found in the MuJoCo model.")
        
        pos = self.data.xpos[base_body_id].copy()
        quat_wxyz = self.data.xquat[base_body_id].copy()
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        
        return pos, quat_xyzw