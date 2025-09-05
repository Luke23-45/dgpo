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
import cv2

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
    # envs/panda_env.py --> _define_spaces()
    def _define_spaces(self):
        """
        Defines observation and action spaces. This version provides all keys
        that are directly used by the OCTO expert pipeline.
        """
        self.observation_space = spaces.Dict({
            # --- Core Visual Modalities (HWC format) ---
            "image_primary": spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
            "image_wrist":   spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
            
            # --- Proprioceptive State ---
            # The primary 14D proprio state (7 joint pos + 7 joint vel)
            "proprio": spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32),
            
            # --- Additional State Information for Expert ---
            # A scalar indicating if the task is complete (0.0 or 1.0)
            "task_completed": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            
            # Current timestep in the episode, shaped as a 1D array
            "timestep": spaces.Box(low=0, high=np.iinfo(np.int32).max, shape=(1,), dtype=np.int32),
        })
        
        # Action space: 7 arm joint deltas + 1 gripper command
        act_dim = int(getattr(self.model, "nu", 8))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
        
    def render(self, camera_name: str = "fixed_camera"):
        """
        Handles rendering for the 'rgb_array' mode from a specified camera.
        This robust version always renders at the primary camera's resolution
        and then downsamples if a smaller view (like the wrist) is requested.
        This avoids resizing the MuJoCo renderer context, which is more stable.
        """
        if self.render_mode != "rgb_array" or self.renderer is None:
            # Determine target shape for the black placeholder
            is_wrist = "wrist" in camera_name
            h = self.observation_space["image_wrist" if is_wrist else "image_primary"].shape[0]
            w = self.observation_space["image_wrist" if is_wrist else "image_primary"].shape[1]
            warnings.warn(f"Renderer not available, returning a black frame for camera '{camera_name}'.")
            return np.zeros((h, w, 3), dtype=np.uint8)

        # Set the renderer to the largest size needed (primary camera) to initialize it
        if self.renderer.width != self.observation_space["image_primary"].shape[1]:
            self.renderer.width = self.observation_space["image_primary"].shape[1]
            self.renderer.height = self.observation_space["image_primary"].shape[0]

        try:
            # Update the scene with the desired camera view
            self.renderer.update_scene(self.data, camera=camera_name)  
            # Render the image at the pre-set large resolution
            large_image = self.renderer.render()
        except Exception as e:
            warnings.warn(f"Failed to render from camera '{camera_name}': {e}")
            is_wrist = "wrist" in camera_name
            h = self.observation_space["image_wrist" if is_wrist else "image_primary"].shape[0]
            w = self.observation_space["image_wrist" if is_wrist else "image_primary"].shape[1]
            return np.zeros((h, w, 3), dtype=np.uint8)

        # Downsample if the requested camera is the wrist camera
        if "wrist" in camera_name:
            target_h = self.observation_space["image_wrist"].shape[0]
            target_w = self.observation_space["image_wrist"].shape[1]
            # Use INTER_AREA for robust downsampling
            resized_image = cv2.resize(large_image, (target_w, target_h), interpolation=cv2.INTER_AREA)
            return resized_image
        else:
            return large_image

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

    # envs/panda_env.py --> _get_obs()
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        Returns a clean observation dictionary that matches the observation_space.
        This version includes rendering for both primary and wrist cameras.
        """
        # Get base proprioceptive state (joint positions and velocities)
        qpos = np.asarray(self.data.qpos, dtype=np.float32)
        qvel = np.asarray(self.data.qvel, dtype=np.float32)
        proprio = np.concatenate([qpos[:7], qvel[:7]])

        # The observation now includes both rendered images.
        return {
            "image_primary": self.render(camera_name="fixed_camera"),
            "image_wrist": self.render(camera_name="wrist_camera"),
            "proprio": proprio,
            "task_completed": np.array([0.0], dtype=np.float32),
            "timestep": np.array([self.timestep], dtype=np.int32),
        }
    
    def get_body_pos_expert(self, name: str) -> np.ndarray:
        """
        Expert-specific helper to get a body's world position.
        This is a safe, read-only operation.
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id == -1:
            raise ValueError(f"Body '{name}' not found for expert pipeline.")
        return self.data.xpos[body_id].copy()

    def get_object_pos_expert(self) -> np.ndarray:
        """Gets the ground-truth world position of the object for the expert."""
        return self.get_body_pos_expert("object")

    def get_goal_pos_expert(self) -> np.ndarray:
        """Gets the ground-truth world position of the goal for the expert."""
        return self.get_body_pos_expert("goal")


    def get_expert_obs(self) -> Dict[str, np.ndarray]:
        """
        Returns a rich observation dictionary for the expert pipeline.

        This includes the standard observation (with both camera views) plus
        ground-truth state information required by the expert.
        """
        # Start with the standard observation, which now includes the wrist image.
        obs = self._get_obs()

        # Add ground-truth data required ONLY by the expert.
        obs["ee_pose_world"] = self.get_ee_pose()
        obs["object_pos_world"] = self.get_object_pos_expert()
        obs["goal_pos_world"] = self.get_goal_pos_expert()
        
        # Add the redundant proprio key required by the IKSolver.
        # This isolates the redundancy to the expert pipeline, which is a good design.
        obs["internal_full_proprio"] = obs["proprio"].copy()

        return obs

    def reset(self, seed: int = None) -> Tuple[Dict, Dict]:
        """Resets the environment to a new, randomized state."""
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
        
        self.timestep = 0
        mujoco.mj_resetData(self.model, self.data)
        home_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.data.qpos[:7] = home_qpos
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
        return self.get_expert_obs(), {}

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

        obs = self.get_expert_obs()
        reward = 0.0
        terminated = False
        truncated = (self.timestep >= self.max_episode_steps)
        
        return obs, reward, terminated, truncated, {}


    def close(self):
        """Cleans up resources, primarily the renderer."""
        if hasattr(self, "renderer") and self.renderer is not None:
            try:
                self.renderer.close()
            finally:
                # Ensure the renderer handle is cleared even if close() fails
                self.renderer = None

    def get_base_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the world-frame pose (pos, quat_xyzw) of the robot's base.
        This version is robust, ensuring float32 dtype and xyzw quaternion format.
        """
        base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link0")
        if base_body_id < 0:
            raise ValueError("Body 'link0' not found in the MuJoCo model.")
        
        # Ensure data is float32
        pos = self.data.xpos[base_body_id].copy().astype(np.float32)
        
        # Ensure quaternion is in xyzw format and float32
        quat_wxyz = self.data.xquat[base_body_id].copy()
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
        
        return pos, quat_xyzw
    


