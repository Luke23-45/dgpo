# envs/panda_env.py (Final, Production-Grade, OCTO-Compliant Version)
import warnings
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from typing import Tuple, Dict

# Assumes this utility exists and is correct.
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

    def _define_spaces(self):
        """Defines the observation and action spaces for the environment."""
        self.observation_space = spaces.Dict({
              # CORRECT: All image spaces are correct.
              "image_primary": spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
              "image_wrist": spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
              # CORRECT: task_completed is a Box space for a vector of shape (4,).
              "task_completed": spaces.Box(low=0, high=1, shape=(4,), dtype=np.int64),
              # CORRECT: timestep is a Box for a vector of shape (1,).
              "timestep": spaces.Box(low=0, high=np.iinfo(np.int32).max, shape=(1,), dtype=np.int32),
              # CORRECT: All Discrete spaces for scalar pad masks are correct.
              "timestep_pad_mask": spaces.Discrete(2),
              "pad_mask_dict": spaces.Dict({
                  "image_primary": spaces.Discrete(2),
                  "image_wrist": spaces.Discrete(2),
                  "timestep": spaces.Discrete(2),
              }),
              # CORRECT: Internal proprioception space is correct.
              "internal_full_proprio": spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
          })
        num_actuators = int(getattr(self.model, "nu", 0))
        if num_actuators <= 0:
            raise RuntimeError("Loaded model has no actuators (model.nu <= 0). Check your XML.")
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_actuators,), dtype=np.float32)


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

        The pose consists of the 3D Cartesian position and the 4D orientation
        quaternion, formatted as [x, y, z, qx, qy, qz, qw] to align with
        common robotics standards and the expected IK solver input.

        Returns:
            A NumPy array of shape (7,) containing the end-effector pose.
        """
        # Ensure the physics state is up-to-date with the latest joint positions
        mujoco.mj_forward(self.model, self.data)

        # Get the 3D position of the end-effector site
        ee_pos = self.data.site("attachment_site").xpos.copy()

        # Get the 3x3 orientation matrix of the end-effector site
        ee_orientation_matrix = self.data.site("attachment_site").xmat.copy().reshape(3, 3)
        
        # Convert the rotation matrix to a quaternion [x, y, z, w]
        # This is the standard format for scipy's Rotation library
        ee_quat_xyzw = R.from_matrix(ee_orientation_matrix).as_quat()

        # Concatenate position and orientation to form the 7D pose
        return np.concatenate([ee_pos, ee_quat_xyzw]).astype(np.float32)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        Gathers the current observation, formatted precisely for the OCTO model.
        Rendering is now delegated to the `render()` method.
        """
        # 1. Get Real Data from Simulation
        image_primary = self.render()
        
        # --- Robust Proprioception Reading ---
        qpos = np.asarray(getattr(self.data, "qpos", np.array([])), dtype=float)
        qvel = np.asarray(getattr(self.data, "qvel", np.array([])), dtype=float)
        qpos7 = qpos[:7] if qpos.size >= 7 else np.pad(qpos, (0, max(0, 7 - qpos.size)))
        qvel7 = qvel[:7] if qvel.size >= 7 else np.pad(qvel, (0, max(0, 7 - qvel.size)))
        full_proprio_internal = np.concatenate([qpos7, qvel7]).astype(np.float32)

        # 2. Create Placeholder Data required by OCTO spec
        wrist_image_placeholder = np.zeros((128, 128, 3), dtype=np.uint8)
        
        # --- THIS IS THE FIX ---
        # The OCTO model expects a vector of shape (4,) for this key.
        task_completed_placeholder = np.zeros(4, dtype=np.int64)

        # 3. Construct the Final Observation Dictionary
        return {
            # --- Data for OCTO ---
            "image_primary": image_primary,
            "image_wrist": wrist_image_placeholder,
            "task_completed": task_completed_placeholder,
            "timestep": np.array([self.timestep], dtype=np.int32),
            "timestep_pad_mask": 1,
            "pad_mask_dict": {
                "image_primary": 1,
                "image_wrist": 0,
                "timestep": 1,
            },
            # --- Data for Our Use (IKSolver) ---
            "internal_full_proprio": full_proprio_internal,
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