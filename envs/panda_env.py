# envs/panda_env.py (Final, Production-Grade, OCTO-Compliant Version)
import warnings
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from typing import Tuple, Dict,List, Callable
from scipy.spatial.transform import Rotation as R
import cv2
from dataclasses import dataclass, field 
from typing import Optional
from scipy.spatial.transform import Rotation, Slerp
from typing import Any
import copy
#self.max_episode_steps
@dataclass
class RenderPostConfig:
    """
    Controls photometric post-processing for enhanced realism.
    """
    apply_tonemap: bool = True
    tonemap_curve: str = "aces"          # ["aces", "reinhard"]
    # Auto-exposure aims to map the chosen luminance percentile to target_white
    auto_exposure_percentile: float = 0.98
    target_white: float = 0.75        # in linear space
    min_exposure: float = 0.7
    max_exposure: float = 1.3
    flip_vertical: Optional[bool] = None
    # When True and flip_vertical is None, the environment will attempt to determine
    # if flip is required by testing the camera up-vector sign (robust in most cases).
    flip_vertical_auto_detect: bool = True

    # Occasionally you want to flip horizontally (keep here for completeness)
    flip_horizontal: bool = False

    # Color space management
    assume_input_is_srgb: bool = True    # MuJoCo returns 8-bit sRGB-like frames
    output_srgb: bool = True             # Final dataset should be sRGB 8-bit

    # Finishing touches
    add_sharpen: bool = False            # Off by default; enable if images look soft
    sharpen_amount: float = 0.15         # Unsharp mask strength
    dithering: bool = True               # Add subtle noise before 8-bit quantization

@dataclass
class CameraShot:
    """Defines a single, known-good camera position and target."""
    pos: Tuple[float, float, float]
    target: Tuple[float, float, float]


# FILE: envs/panda_env.py (Replace the dataclass)

@dataclass
class DomainRandomizationConfig:
    """Holds all parameters for domain randomization."""
    # Lighting and Texture randomization parameters remain the same.
    light_pos_range: Tuple[Tuple[float, float], ...] = ((-1.0, 1.0), (-1.0, 1.0), (1.5, 2.5))
    light_color_range: Tuple[Tuple[float, float], ...] = ((0.6, 1.0), (0.6, 1.0), (0.6, 1.0))
    table_textures: List[str] = field(default_factory=lambda: [
        "mat_table_wood_light", "mat_table_wood_stripe", "mat_table_marble_white",
        "mat_table_metal_brushed", "mat_table_noise_low",   "mat_table_noise_high" 
    ])
    floor_textures: List[str] = field(default_factory=lambda: [
        "mat_floor_checker_blue", "mat_floor_checker_green",
        "mat_floor_wood_dark", "mat_floor_wood_paquet"
    ])

    # ============================ CURATED EXEMPLAR SHOTS ============================
    # FINAL PATCH: This new list is mined from the best results in your JSON data.
    # It provides a wider, more robust, and higher-quality set of base viewpoints.
    # camera_shots: List[CameraShot] = field(default_factory=lambda: [
    #     # --- Right Three-Quarter Views ---
    #     # [Source: Shot_01/sample_00] Perfect, balanced right view. Elevation: 38.3°
    #     CameraShot(pos=(0.90, -0.57, 1.03), target=(0.41, -0.02, 0.45)),
    #     # (From Shot_04/sample_01) - A slightly wider right view. Elevation: 45.1°
    #     CameraShot(pos=(1.07, -0.25, 1.09), target=(0.49, -0.03, 0.44)),

    #     # --- Left Three-Quarter Views ---
    #     # [Source: Shot_04/sample_00] Excellent, clear left-side composition. Elevation: 53.0°
    #     # CameraShot(pos=(0.57, 0.58, 1.01), target=(0.44, -0.01, 0.43)),
    #     # (From Shot_06/sample_15) - A wide, cinematic left view. Elevation: 32°
    #     # CameraShot(pos=(0.91, 0.49, 0.92), target=(0.38, 0.06, 0.45)),
    #     # [Source: Shot_12/sample_01] Another strong left view, slightly different framing. Elevation: 39.2°
    #     CameraShot(pos=(1.10, 0.41, 1.01), target=(0.52, 0.06, 0.45)),

    #     CameraShot(pos=(0.98, 0.53, 0.98), target=(0.48, -0.01, 0.44)),
        
    #     # --- Frontal Views ---
    #     # [Source: Shot_08/sample_00] A perfect, direct frontal shot. Elevation: 31.0°
    #     CameraShot(pos=(1.09, 0.14, 0.87), target=(0.40, 0.03, 0.44)),
    #     # [Source: Shot_09/sample_12] A slightly higher frontal view, great for context. Elevation: 36.1°
    #     # CameraShot(pos=(1.17, 0.06, 0.98), target=(0.45, -0.02, 0.45)),
    #     # [Source: Shot_06/sample_02] A wider frontal view. Elevation: 41.3°
    #     CameraShot(pos=(1.02, 0.10, 0.95), target=(0.39, 0.04, 0.45)),

    #     # --- High-Angle / Near Top-Down Views ---
    #     # [Source: Shot_09/sample_00] A well-composed high three-quarter view. Elevation: 48.7°
    #     CameraShot(pos=(1.02, 0.05, 1.10), target=(0.45, -0.02, 0.44)),
    #     # [Source: Shot_10/sample_01 - MODIFIED] A safe top-down, clamped away from the extreme 74°. Elevation: 68.0°
    #     # CameraShot(pos=(0.85, -0.15, 1.25), target=(0.40, -0.04, 0.45)),

    #     # --- Dynamic / Lower Views (Still Safe) ---
    #     # [Source: Shot_11/sample_00] The perfect low-angle shot, just above the limit. Elevation: 25.2°
    #     CameraShot(pos=(1.02, 0.10, 0.95), target=(0.39, 0.04, 0.45)),
    #     # [Source: Shot_13/sample_01] A fantastic wide, low-left perspective. Elevation: 27.4°
    #     CameraShot(pos=(0.98, 0.53, 0.82), target=(0.48, -0.01, 0.44)),

    #     # --- Extra Views for Maximum Variety ---    
    #     # [Source: Shot_03/sample_00] An interesting over-the-shoulder left view. Elevation: 35.0°
    #     # CameraShot(pos=(0.75, 0.54, 0.91), target=(0.40, -0.02, 0.44)),
    #     # (From Shot_06/sample_15) - A wide, cinematic left view. Elevation: 32°
    #     # [Source: Shot_07/sample_00] A very wide three-quarter view, good for seeing the whole table. Elevation: 23.5° (clamped to 25)
    #     CameraShot(pos=(1.20, 0.55, 0.95), target=(0.52, 0.01, 0.45)),
    # ])
    camera_shots: List[CameraShot] = field(default_factory=lambda: [
        # --- Right Three-Quarter Views (Safe High Angles) ---
        CameraShot(pos=(0.90, -0.57, 1.03), target=(0.41, -0.02, 0.45)),
        CameraShot(pos=(1.07, -0.25, 1.09), target=(0.49, -0.03, 0.44)),

        # --- Left Three-Quarter Views (Balanced) ---
        CameraShot(pos=(1.10, 0.41, 1.01), target=(0.52, 0.06, 0.45)),
        
        # [FIXED] "Side Left" - Raised Z from 0.82 to 0.98 for safety
        CameraShot(pos=(0.98, 0.53, 0.98), target=(0.48, -0.01, 0.44)),
        
        # Wide Left
        CameraShot(pos=(1.20, 0.55, 0.95), target=(0.52, 0.01, 0.45)),

        # --- Frontal Views (Varied Heights) ---
        # Low Front (Face Level) - Lowest safe frontal shot
        CameraShot(pos=(1.09, 0.14, 0.87), target=(0.40, 0.03, 0.44)),
        
        # Mid Front - Standard view
        CameraShot(pos=(1.02, 0.10, 0.95), target=(0.39, 0.04, 0.45)),

        # High Front - Near top-down
        CameraShot(pos=(1.02, 0.05, 1.10), target=(0.45, -0.02, 0.44)),
    ])

    # radius_jitter: float = 0.10      # meters (reduced from 0.15)
    # azimuth_jitter: float = 0.26     # radians (~15 degrees) (reduced from 0.35)
    # elevation_jitter: float = 0.17   # radians (~10 degrees) (reduced from 0.26)
    # target_pos_jitter: float = 0.05  # meters (reduced from 0.08)
    # fovy_jitter: float = 3.0         # degrees (reduced from 5.0)

    radius_jitter: float = 0.05      # meters (reduced from 0.10). Keeps scale consistent.
    azimuth_jitter: float = 0.15     # radians (~8.5 deg). Prevents swinging behind robot arm.
    elevation_jitter: float = 0.10   # radians (~5.7 deg). Prevents becoming "Top Down".
    target_pos_jitter: float = 0.02  # meters (2cm). Keeps object perfectly centered.
    fovy_jitter: float = 2.0         # degrees. Minor zoom variation.

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

    TABLE_CENTER = np.array([0.6, 0.0]) # XY center of the table in world frame
    TABLE_DIMS = np.array([0.4, 0.4])   # Half-widths of the table geom

    # Placement zones are defined as [min_offset, max_offset] from the table center
    # Placement zones are defined as [min_offset, max_offset] from the table center
    # These have been adjusted to be more central and guarantee visibility.
    PLACEMENT_ZONES = {
        # "center": (np.array([-0.05, -0.05]), np.array([0.05, 0.05])),
        # "left":   (np.array([-0.15, -0.1]), np.array([-0.05, 0.1])),
        # "right":  (np.array([0.05, -0.1]), np.array([0.15, 0.1])),
        # "front":  (np.array([-0.15, -0.15]), np.array([0.15, -0.05])),
        # "back":   (np.array([-0.15, 0.05]), np.array([0.15, 0.15])),
        "full_table": (np.array([-0.25, -0.35]), np.array([0.15, 0.35]))

    }

    # Z-height for the object on the table
    OBJECT_Z_HEIGHT = 0.42 
    # Z-height for the goal on the table
    GOAL_Z_HEIGHT = 0.401
    LONG_REACH_THRESHOLD = 0.5

    # Parameters for the "Three-Quarter Detail View" strategy.
    # Placing them here makes them easy to tune.
    CAM_BASE_DISTANCE = 0.8
    CAM_DISTANCE_SCALE_FACTOR = 1.5
    CAM_BASE_FOVY = 45.0
    CAM_FOVY_SCALE_FACTOR = 25.0
    CAM_HEIGHT_ABOVE_TARGET = 0.8
    CAM_MIN_DISTANCE = 0.4   # Don't let the camera get too close
    CAM_MAX_DISTANCE = 2.0   # Don't let the camera get too far
    CAM_MIN_FOVY = 25.0      # Min zoom
    CAM_MAX_FOVY = 90.0      # Max zoom (wide-angle)
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    ACTION_SCALING_FACTOR = 0.5


    # REPLACE THE ENTIRE __init__ METHOD WITH THIS
    def __init__(
            self,
            xml_path: str = "envs/panda_pick_place.xml",
            render_mode: str = "rgb_array",
            dr_config: DomainRandomizationConfig = None,
            enable_domain_randomization: bool = True,
            post_config: RenderPostConfig = None,
            control_mode: str = "delta",
            grasp_mode: str = "stateful", 
            action_scaling_factor: float = 0.5,
        ):
        super().__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        assert control_mode in ["absolute", "delta"], "control_mode must be 'absolute' or 'delta'"
        self.control_mode = control_mode

        # 1. Load Model (Must be first)
        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self.data = mujoco.MjData(self.model)

        except Exception as e:
            raise FileNotFoundError(f"Could not load MuJoCo XML from '{xml_path}'. Error: {e}")

        # 2. Initialize Renderer and Configs
        self.render_mode = render_mode
        try:
            self.renderer = mujoco.Renderer(self.model, height=256, width=256)
        except Exception:
            warnings.warn("mujoco.Renderer not available — running headless.")
            self.renderer = None
        self.post = post_config or RenderPostConfig()
        
        # 3. Initialize Episode Bookkeeping and RNG
        self.max_episode_steps = 400
        self.timestep = 0
        self.np_random, _ = seeding.np_random(None)
        self.enable_domain_randomization = enable_domain_randomization
        self.dr_config = dr_config or DomainRandomizationConfig()
        self.ACTION_SCALING_FACTOR = action_scaling_factor
        self.proprio_dim = 7 + 7 + 2 + 6 
        self._initial_image: Optional[np.ndarray] = None
        self._goal_image: Optional[np.ndarray] = None

        # 4. Consolidated Block: Cache all MuJoCo IDs and initialize state
        #    This block runs AFTER the model is loaded and BEFORE spaces are defined.
        self._cache_dr_element_ids()

        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        if self.ee_site_id == -1:
            raise ValueError("Site 'attachment_site' not found in the MuJoCo model.")

        self.object_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_joint")
        if self.object_joint_id == -1:
            raise ValueError("Joint 'object_joint' not found in the MuJoCo model.")
        self.object_qpos_addr = self.model.jnt_qposadr[self.object_joint_id]
        
        self.object_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        if self.object_geom_id == -1:
            raise ValueError("Geom 'object_geom' not found in the XML.")
            
        self.left_touch_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "left_finger_touch_sensor")
        self.right_touch_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "right_finger_touch_sensor")
        if self.left_touch_sensor_id == -1 or self.right_touch_sensor_id == -1:
            raise ValueError("Touch sensors for fingertips not found. Check the XML.")
            
        self.left_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        if self.left_finger_id == -1 or self.right_finger_id == -1:
            raise ValueError("Could not find 'left_finger' or 'right_finger' bodies in the XML.")
        self.left_force_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "left_finger_force_sensor")
        self.right_force_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "right_finger_force_sensor")
        if self.left_force_sensor_id == -1 or self.right_force_sensor_id == -1:
            raise ValueError("Force sensors for fingertips not found. Check the XML.")
            
        self._is_physically_grasped = False
        self._grasp_stabilization_counter = 0

        # Cache body and geom IDs for efficient contact checking
        self.object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.left_finger_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_finger_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

        if any(id_ == -1 for id_ in [self.object_body_id, self.left_finger_body_id, self.right_finger_body_id]):
            raise ValueError("Could not find body IDs for object, left_finger, or right_finger.")

        # Collect all geoms belonging to the relevant bodies
        self.object_geoms = np.where(self.model.geom_bodyid == self.object_body_id)[0]
        self.left_finger_geoms = np.where(self.model.geom_bodyid == self.left_finger_body_id)[0]
        self.right_finger_geoms = np.where(self.model.geom_bodyid == self.right_finger_body_id)[0]

        self._is_physically_grasped = False
        self._grasp_stabilization_counter = 0
        # 5. Define Spaces (Must be last)
        assert grasp_mode in ["stateful", "physical"], "grasp_mode must be 'stateful' or 'physical'"
        self.grasp_mode = grasp_mode
        self._define_spaces()


    def _render_goal_image(self) -> np.ndarray:
        """Saves the current state, moves object to goal, renders, and restores state."""
        original_state = self.get_mj_state()

        goal_pos = self.get_goal_pos_expert()
        goal_orn_xyzw = self.get_goal_orientation_expert()
        goal_orn_wxyz = self._scipy_xyzw_to_mujoco_wxyz(goal_orn_xyzw)

        qpos_addr = self.model.jnt_qposadr[self.object_joint_id]
        self.data.qpos[qpos_addr : qpos_addr + 3] = goal_pos
        self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = goal_orn_wxyz
        mujoco.mj_forward(self.model, self.data)
        
        goal_img = self.render(camera_name="fixed_camera")

        self.set_mj_state(original_state)
        return goal_img

    def _has_geom_contact(self, group1_geoms: np.ndarray, group2_geoms: np.ndarray) -> bool:
        """Checks if any geom in group1 is in contact with any geom in group2."""
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            # Check if the contact involves one geom from each group
            is_in_group1_geom1 = np.any(con.geom1 == group1_geoms)
            is_in_group2_geom2 = np.any(con.geom2 == group2_geoms)
            is_in_group2_geom1 = np.any(con.geom1 == group2_geoms)
            is_in_group1_geom2 = np.any(con.geom2 == group1_geoms)
            
            if (is_in_group1_geom1 and is_in_group2_geom2) or \
               (is_in_group2_geom1 and is_in_group1_geom2):
                return True
        return False


    def _cache_dr_element_ids(self):
        """Finds and caches the integer IDs of all elements used in DR."""
        if not self.enable_domain_randomization:
            self._dr_mat_ids = {}
            return

        # --- Cache critical element IDs (fail fast if missing) ---
        self.light_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_LIGHT, "main_light")
        self.fill_light_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_LIGHT, "fill_light") 
        self.back_light_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_LIGHT, "back_light")
        self.table_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed_camera")

        if any(id_ == -1 for id_ in [self.light_id, self.table_geom_id, self.floor_geom_id, self.camera_id, self.back_light_id, self.fill_light_id]):
            raise ValueError("One or more critical elements (main_light, table_geom, floor, fixed_camera) "
                             "are missing a 'name' attribute in the XML and cannot be randomized.")

        # --- Validate and cache material IDs (fail gracefully) ---
        self._dr_mat_ids = {}
        all_textures = self.dr_config.table_textures + self.dr_config.floor_textures
        for name in all_textures:
            mat_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, name)
            if mat_id != -1:
                self._dr_mat_ids[name] = mat_id
            else:
                warnings.warn(f"Domain Randomization asset '{name}' not found in XML. It will be ignored.")
        
        # Update the config to only include valid textures
        self.dr_config.table_textures = [n for n in self.dr_config.table_textures if n in self._dr_mat_ids]
        self.dr_config.floor_textures = [n for n in self.dr_config.floor_textures if n in self._dr_mat_ids]
    # ---------- Photometric helpers (sRGB/linear, exposure, tone map) ----------
    def set_object_size(self, size: np.ndarray):
        """
        Dynamically sets the size of the object geom for the current episode.
        NOTE: MuJoCo geoms are defined by half-extents (half-widths).
        """
        size = np.asarray(size, dtype=float)
        if size.shape != (3,):
            raise ValueError(f"Size must be a 3-element array, but got shape {size.shape}")
        
        # We need to set the geom_size to half of the full dimension
        self.model.geom_size[self.object_geom_id] = size / 2.0
        try:
            mujoco.mj_forward(self.model, self.data)
        except Exception as e:
            print(f"mj_forward failed after setting object size: {e}")
            # Depending on the desired robustness, you might want to raise the exception
            raise
    @staticmethod
    def _srgb_to_linear(img: np.ndarray) -> np.ndarray:
        """img in [0,1] sRGB -> linear RGB (float32)."""
        img = img.astype(np.float32)
        a = 0.055
        low = img <= 0.04045
        high = ~low
        out = np.empty_like(img, dtype=np.float32)
        out[low]  = img[low] / 12.92
        out[high] = ((img[high] + a) / (1 + a)) ** 2.4
        return out

    @staticmethod
    def _linear_to_srgb(img: np.ndarray) -> np.ndarray:
        """linear RGB in [0,1] -> sRGB [0,1] (float32)."""
        img = np.clip(img, 0.0, 1.0).astype(np.float32)
        a = 0.055
        low = img <= 0.0031308
        high = ~low
        out = np.empty_like(img, dtype=np.float32)
        out[low]  = img[low] * 12.92
        out[high] = (1 + a) * (img[high] ** (1/2.4)) - a
        return out

    @staticmethod
    def _luminance_linear(img_lin: np.ndarray) -> np.ndarray:
        """Rec.709 luminance of a linear RGB image in [0, +inf)."""
        return 0.2126 * img_lin[..., 0] + 0.7152 * img_lin[..., 1] + 0.0722 * img_lin[..., 2]

    def _auto_exposure_scale(self, img_lin: np.ndarray) -> float:
        """Percentile-based exposure so that p% luminance maps to target_white."""
        lum = self._luminance_linear(img_lin).reshape(-1)
        # robust against pure-black frames
        p = np.percentile(lum, self.post.auto_exposure_percentile * 100.0) if lum.size > 0 else 0.0
        if p <= 1e-6:
            return 1.0
        scale = self.post.target_white / float(p)
        return float(np.clip(scale, self.post.min_exposure, self.post.max_exposure))

    @staticmethod
    def _tonemap_aces(img_lin: np.ndarray) -> np.ndarray:
        """ACES fitted filmic curve (applied in linear space)."""
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        num = img_lin * (a * img_lin + b)
        den = img_lin * (c * img_lin + d) + e
        return np.clip(num / den, 0.0, 1.0)

    @staticmethod
    def _tonemap_reinhard(img_lin: np.ndarray) -> np.ndarray:
        """Classic Reinhard global operator: x/(1+x) in linear space."""
        return img_lin / (1.0 + img_lin)

    def _mujoco_quat_to_scipy_xyzw(self, quat_wxyz: np.ndarray) -> np.ndarray:
        """
        MuJoCo stores quaternions in [w, x, y, z] order in model arrays.
        SciPy Rotation.from_quat expects [x, y, z, w].
        This helper returns the SciPy-compatible ordering.
        """
        q = np.asarray(quat_wxyz, dtype=float).reshape(4)
        return np.array([q[1], q[2], q[3], q[0]], dtype=float)

    def _scipy_xyzw_to_mujoco_wxyz(self, quat_xyzw: np.ndarray) -> np.ndarray:
        """
        Convert scipy-style [x,y,z,w] quaternion into MuJoCo [w,x,y,z] format.
        """
        q = np.asarray(quat_xyzw, dtype=float).reshape(4)
        return np.array([q[3], q[0], q[1], q[2]], dtype=float)
    
    def _decide_flip_for_camera(self, camera_name: str) -> bool:
        # 1) explicit override via config
        if self.post.flip_vertical is not None:
            return bool(self.post.flip_vertical)

        # 2) cached calibration per camera (set by calibration or earlier render)
        if hasattr(self, "_camera_flip_cache") and camera_name in self._camera_flip_cache:
            return bool(self._camera_flip_cache[camera_name])

        # 3) auto detect using quaternion up-vector (with small margin)
        flip = False
        if self.post.flip_vertical_auto_detect:
            try:
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
                if cam_id != -1:
                    quat_wxyz = np.asarray(self.model.cam_quat[cam_id], dtype=float)
                    quat_xyzw = self._mujoco_quat_to_scipy_xyzw(quat_wxyz)
                    R_wc = R.from_quat(quat_xyzw).as_matrix()
                    cam_up_world = R_wc @ np.array([0.0, 1.0, 0.0], dtype=float)
                    flip = float(cam_up_world[2]) < -1e-3
            except Exception:
                flip = False

        # cache the decision for the camera for this runtime (can be invalidated if camera moves)
        if not hasattr(self, "_camera_flip_cache"):
            self._camera_flip_cache = {}
        self._camera_flip_cache[camera_name] = bool(flip)
        return bool(flip)

    def _postprocess_image(self, img_u8: np.ndarray) -> np.ndarray:
        """End-to-end post pipeline: sRGB->linear, auto-exposure, tonemap, linear->sRGB."""
        img = img_u8.astype(np.float32) / 255.0
        img_lin = self._srgb_to_linear(img) if self.post.assume_input_is_srgb else img

        exposure = self._auto_exposure_scale(img_lin)
        img_lin *= exposure

        if self.post.apply_tonemap:
            if self.post.tonemap_curve.lower() == "aces":
                img_lin = self._tonemap_aces(img_lin)
            else:
                img_lin = self._tonemap_reinhard(img_lin)

        img_out = self._linear_to_srgb(img_lin) if self.post.output_srgb else np.clip(img_lin, 0.0, 1.0)

        if self.post.add_sharpen:
            blur = cv2.GaussianBlur(img_out, ksize=(0, 0), sigmaX=0.8)
            img_out = np.clip(img_out + self.post.sharpen_amount * (img_out - blur), 0.0, 1.0)

        if self.post.dithering:
            noise = (self.np_random.random(img_out.shape).astype(np.float32) - 0.5) / 255.0
            img_out = np.clip(img_out + noise, 0.0, 1.0)

        return np.clip(np.round(img_out * 255.0), 0, 255).astype(np.uint8)  
          
    @staticmethod
    def _calculate_look_at_quat(camera_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        """Calculates a quaternion for a camera to look at a target. Returns xyzw."""
        up_vector = np.array([0, 0, 1])
        forward = target_pos - camera_pos
        # Add a small epsilon to prevent normalization of a zero vector
        if np.linalg.norm(forward) < 1e-6:
            return np.array([0, 0, 0, 1], dtype=float) # Return identity quat
        forward /= np.linalg.norm(forward)
        
        right = np.cross(up_vector, forward)
        if np.linalg.norm(right) < 1e-6: # Handle gimbal lock case
            # If forward is aligned with up, choose a different right vector
            right = np.array([1, 0, 0], dtype=float)
        right /= np.linalg.norm(right)

        cam_up = np.cross(right, forward)
        rot_matrix = np.eye(3)
        rot_matrix[:, 0] = right
        rot_matrix[:, 1] = cam_up
        rot_matrix[:, 2] = -forward # MuJoCo cameras look along their -Z axis
        return R.from_matrix(rot_matrix).as_quat()
    @staticmethod
    def _cartesian_to_spherical(pos: np.ndarray, target: np.ndarray) -> Tuple[float, float, float]:
        """Converts a camera position to spherical coordinates (r, az, el) relative to a target."""
        vec = pos - target
        radius = np.linalg.norm(vec)
        # Add a small epsilon to prevent division by zero for radius
        if radius < 1e-6:
            return 0.0, 0.0, np.pi / 2
        azimuth = np.arctan2(vec[1], vec[0])
        elevation = np.arcsin(vec[2] / radius)
        return radius, azimuth, elevation
    
    @staticmethod
    def _safe_normalize(vec: np.ndarray, default: np.ndarray = None) -> np.ndarray:
        """Normalizes a vector, returning a default if the norm is close to zero."""
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            if default is None:
                return np.zeros_like(vec)
            return default
        return vec / norm


    @staticmethod
    def _spherical_to_cartesian(radius: float, azimuth: float, elevation: float, target: np.ndarray) -> np.ndarray:
        """Converts spherical coordinates back to a Cartesian camera position."""
        x = radius * np.cos(elevation) * np.cos(azimuth)
        y = radius * np.cos(elevation) * np.sin(azimuth)
        z = radius * np.sin(elevation)
        return target + np.array([x, y, z])


    def get_camera_params(self, camera_name: str) -> Dict[str, Any]:
        """
        Returns a dictionary of parameters for a given camera,
        essential for 3D-to-2D projections.
        """
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id == -1:
            raise ValueError(f"Camera '{camera_name}' not found in the model.")

        # Get extrinsics
        pos = self.model.cam_pos[cam_id].copy()
        quat_wxyz = self.model.cam_quat[cam_id].copy()
        quat_xyzw = self._mujoco_quat_to_scipy_xyzw(quat_wxyz) # Use existing helper

        # Get intrinsics-related info
        fovy = self.model.cam_fovy[cam_id]
        
        # Get image dimensions from the observation space for this camera
        target_key = "image_wrist" if "wrist" in camera_name else "image_primary"
        height, width, _ = self.observation_space.spaces[target_key].shape

        return {
            "pos": pos.astype(np.float32),
            "quat_xyzw": quat_xyzw.astype(np.float32),
            "fovy": float(fovy),
            "height": int(height),
            "width": int(width),
        }


    def _apply_domain_randomization(self, gripper_pos: np.ndarray, goal_pos: np.ndarray):
        """
        Original Domain Randomization: Object-Centric / Panning.
        The camera tracks the midpoint between the gripper and the goal.
        """
        if not self.enable_domain_randomization:
            return

        # --- 1. Photometric Randomization ---
        # Center lighting on the action
        action_midpoint = (gripper_pos + goal_pos) / 2.0
        self._randomize_photometrics(action_midpoint)

        # --- 2. Geometric Randomization (Camera Pose) ---
        chosen_shot = self.np_random.choice(self.dr_config.camera_shots)
        base_cam_pos = np.array(chosen_shot.pos)
        base_target_ref = np.array(chosen_shot.target)

        # A. Define Target: The Action Midpoint + Jitter
        # This causes the "Panning" effect as the gripper moves
        target_jitter = self.np_random.uniform(
            -self.dr_config.target_pos_jitter,
            self.dr_config.target_pos_jitter, 
            size=3
        )
        final_target_pos = action_midpoint + target_jitter
        
        # Safety: Keep focus above table
        final_target_pos[2] = max(final_target_pos[2], self.GOAL_Z_HEIGHT)

        # B. Spherical Conversion
        radius, azimuth, elevation = self._cartesian_to_spherical(base_cam_pos, base_target_ref)
        
        # Apply Noise
        radius += self.np_random.uniform(-self.dr_config.radius_jitter, self.dr_config.radius_jitter)
        azimuth += self.np_random.uniform(-self.dr_config.azimuth_jitter, self.dr_config.azimuth_jitter)
        elevation += self.np_random.uniform(-self.dr_config.elevation_jitter, self.dr_config.elevation_jitter)

        # C. Safety Clamps
        elevation = np.clip(elevation, np.deg2rad(25), np.deg2rad(75))
        radius = np.clip(radius, 0.6, 1.5)

        # D. Reconstruct Camera
        final_cam_pos = self._spherical_to_cartesian(radius, azimuth, elevation, final_target_pos)
        new_quat_xyzw = self._calculate_look_at_quat(final_cam_pos, final_target_pos)
        
        # E. FOV Randomization
        base_fovy = 45.0 
        final_fovy = base_fovy + self.np_random.uniform(-self.dr_config.fovy_jitter, self.dr_config.fovy_jitter)
        final_fovy = np.clip(final_fovy, 35.0, 65.0)

        # --- 3. Apply to Simulation ---
        self.model.cam_pos[self.camera_id] = final_cam_pos
        self.model.cam_quat[self.camera_id] = self._scipy_xyzw_to_mujoco_wxyz(new_quat_xyzw)
        self.model.cam_fovy[self.camera_id] = final_fovy
        
        if hasattr(self, "_camera_flip_cache"):
            self._camera_flip_cache.clear()

    def _is_pos_in_camera_view(
        self, pos_world: np.ndarray, camera_name: str, margin: int = 0
    ) -> Tuple[bool, Dict]:
        """
        Returns a tuple: (is_visible, debug_info).
        `is_visible` is True if the point is in the camera's view.
        `debug_info` contains intermediate values for debugging.
        """
        debug_info = {}

        # Image size from your observation_space
        # Choose correct observation size for this camera (wrist vs primary)
        target_key = "image_wrist" if "wrist" in camera_name else "image_primary"
        height, width, _ = self.observation_space.spaces[target_key].shape
        debug_info["image_shape"] = (height, width)

        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id == -1:
            return False, {"error": f"Camera '{camera_name}' not found."}

        # Read the static camera pose from the model definition
        cam_pos = self.model.cam_pos[cam_id].copy()
        quat_wxyz = self.model.cam_quat[cam_id].copy()
        debug_info["cam_pos_model"] = cam_pos
        debug_info["cam_quat_model"] = quat_wxyz
        
        quat_xyzw = self._mujoco_quat_to_scipy_xyzw(quat_wxyz)
        R_wc = R.from_quat(quat_xyzw).as_matrix()
        R_cw = R_wc.T

        # Transform the world point into the camera frame
        Pw = np.asarray(pos_world, dtype=float).reshape(3)
        Pc = R_cw @ (Pw - cam_pos)
        debug_info["point_in_camera_frame"] = Pc

        # In MuJoCo, camera looks along -Z. Points in front must have z_c < 0.
        if Pc[2] >= -1e-5:
            debug_info["failure_reason"] = "Point is behind or on the camera plane."
            return False, debug_info

        # Intrinsics
        fovy_deg = float(self.model.cam_fovy[cam_id])
        fovy = np.deg2rad(fovy_deg)
        fy = 0.5 * height / np.tan(0.5 * fovy)
        fx = fy * (width / float(height))
        debug_info["intrinsics"] = {"fovy": fovy_deg, "fx": fx, "fy": fy}

        # Pinhole projection
        z_c_safe = -Pc[2] if -Pc[2] > 1e-6 else 1e-6
        u = fx * (Pc[0] / z_c_safe) + 0.5 * width
        v = -fy * (Pc[1] / z_c_safe) + 0.5 * height
        debug_info["projected_pixel"] = (u, v)


        flip_vertical = self._decide_flip_for_camera(camera_name)


        debug_info["render_flip_applied"] = bool(flip_vertical)

        if flip_vertical:
            v_img = (height - 1) - v
        else:
            v_img = v

        debug_info["projected_pixel_after_flip"] = (u, v_img)
        # Final bounds check
        adaptive_margin = int(0.05 * min(width, height))  # 5% of image size
        is_visible = (
            adaptive_margin <= u < (width - adaptive_margin)
            and adaptive_margin <= v_img < (height - adaptive_margin)
)

        if not is_visible:
            debug_info["failure_reason"] = "Projected pixel is outside the image margin."
            debug_info["bounds"] = {"u": u, "v": v, "adaptive_margin": adaptive_margin, "width": width, "height": height}
        
        return is_visible, debug_info

    # def _define_spaces(self):
    #     """
    #     Defines observation and action spaces.
    #     """
    #     # START OF MODIFIED BLOCK
    #     proprio_dim = 7 + 7 + 2 + 6 # 7 qpos, 7 qvel, 2 touch, 6 force (3D x 2)
    #     self.observation_space = spaces.Dict({
    #         # --- Core Visual Modalities (HWC format) ---
    #         "image_primary": spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
    #         "image_wrist":   spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
            
    #         # --- Proprioceptive State ---
    #         # 7 jnt_pos + 7 jnt_vel + 2 touch_sensor + 2x3D force_sensor
    #         "proprio": spaces.Box(low=-np.inf, high=np.inf, shape=(proprio_dim,), dtype=np.float32),
    #         "is_grasped": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            
    #         # --- Additional State Information for Expert ---
    #         "task_completed": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            
    #         # Current timestep in the episode, shaped as a 1D array
    #         "timestep": spaces.Box(low=0, high=np.iinfo(np.int32).max, shape=(1,), dtype=np.int32),
    #     })
    #     # END OF MODIFIED BLOCK
        
    #     act_dim = 8
    #     self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)  
    def _define_spaces(self):
        """
        Defines observation and action spaces, including all expert keys
        needed for the data generation and RL fine-tuning pipeline.
        """
        proprio_dim = 7 + 7 + 2 + 6 # 7 qpos, 7 qvel, 2 touch, 6 force (3D x 2)
        
        # Define common bounds for floating point Box spaces
        FLOAT_BOX = lambda shape: spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            # --- Core Visual Modalities (HWC format) ---
            "image_primary": spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
            "image_wrist":   spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
            "initial_image": spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
            "goal_image": spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),

            # --- Proprioceptive State (Used by Policy) ---
            "proprio": FLOAT_BOX((proprio_dim,)),
            "is_grasped": FLOAT_BOX((1,)),
            
            # --- Additional State Information (Meta/Task) ---
            "task_completed": FLOAT_BOX((1,)),
            "timestep": spaces.Box(low=0, high=np.iinfo(np.int32).max, shape=(1,), dtype=np.int32),
            
            # --- Expert/Ground Truth Keys (Used by Expert Policy & Reward Wrapper) ---
            # These must be defined here for stable-baselines3 compatibility.
            "ee_pose_world": FLOAT_BOX((7,)), # 3 pos + 4 quat (xyzw)
            "object_pos_world": FLOAT_BOX((3,)),
            "goal_pos_world": FLOAT_BOX((3,)),
            "object_orn_world": FLOAT_BOX((4,)),
            "goal_orn_world": FLOAT_BOX((4,)),
            "goal_size_world": FLOAT_BOX((3,)), # Goal object size (for reward/planning)
            
            # --- Internal/IKSolver Keys ---
            "internal_full_proprio": FLOAT_BOX((proprio_dim,)), # Redundant copy for IKSolver compatibility
            "gripper_qpos": FLOAT_BOX((2,)),
            "robot_base_pos_world": FLOAT_BOX((3,)),
            "ee_vel": FLOAT_BOX((6,)),
            "object_vel": FLOAT_BOX((6,)),
            "gripper_vel": FLOAT_BOX((2,)),
            
        })
        # END OF MODIFIED BLOCK
        
        act_dim = 8
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)

    def _randomize_photometrics(self, light_target: np.ndarray):
            """
            Implements an advanced 3-point lighting strategy with material randomization
            to create a more realistic and visually diverse scene.
            """
            # --- Part 1: Randomize Textures ---
            if self.dr_config.table_textures:
                chosen_table_tex = self.np_random.choice(self.dr_config.table_textures)
                if chosen_table_tex in self._dr_mat_ids:
                    self.model.geom_matid[self.table_geom_id] = self._dr_mat_ids[chosen_table_tex]

            if self.dr_config.floor_textures:
                chosen_floor_tex = self.np_random.choice(self.dr_config.floor_textures)
                if chosen_floor_tex in self._dr_mat_ids:
                    self.model.geom_matid[self.floor_geom_id] = self._dr_mat_ids[chosen_floor_tex]

            # --- Part 2: Randomize Table Material Properties ---
            table_mat_id = self.model.geom_matid[self.table_geom_id]
            # PATCH: Tightly constrained ranges to prevent overly glossy/plastic looks.
            self.model.mat_shininess[table_mat_id] = self.np_random.uniform(30, 50)
            spec_intensity = self.np_random.uniform(0.05, 0.2)
            self.model.mat_specular[table_mat_id] = spec_intensity

            # --- Part 3: Advanced 3-Point Lighting Randomization ---
            target_pos_light = light_target + self.np_random.uniform(-0.05, 0.05, size=3)
            # print(f"DEBUG: Lighting target is at {np.round(target_pos_light, 2)}") 
            target_pos_light[2] = max(target_pos_light[2], 0.4)

            # PATCH: More robust color logic for natural lighting.
            # Generate a base color temperature (from warm white to cool white).
            base_color_temp = self.np_random.uniform(0.9, 1.0)
            key_color = np.array([1.0, base_color_temp, base_color_temp - 0.15])
            key_color = self._safe_normalize(key_color) * self.np_random.uniform(0.9, 1.1)
            key_color = np.clip(key_color, 0.0, 1.0)

            # The fill light is intentionally made cooler (more blue) for nice contrast.
            fill_tint = np.array([0.85, 0.9, 1.0])
            fill_color = key_color * fill_tint
            fill_color = self._safe_normalize(fill_color) * self.np_random.uniform(0.9, 1.1)
            fill_color = np.clip(fill_color, 0.0, 1.0)
            
            key_intensity = self.np_random.uniform(0.7, 0.9)
            fill_intensity = self.np_random.uniform(0.3, 0.45)
            back_intensity = self.np_random.uniform(0.25, 0.4)

            # 1. KEY LIGHT (Main Light)
            key_azimuth = self.np_random.uniform(np.deg2rad(-60), np.deg2rad(60))
            key_elevation = self.np_random.uniform(np.deg2rad(45), np.deg2rad(70))
            key_radius = self.np_random.uniform(1.5, 2.2)
            key_pos = target_pos_light + np.array([
                key_radius * np.cos(key_elevation) * np.cos(key_azimuth),
                key_radius * np.cos(key_elevation) * np.sin(key_azimuth),
                key_radius * np.sin(key_elevation)
            ])
            self.model.light_pos[self.light_id] = key_pos
            self.model.light_dir[self.light_id] = self._safe_normalize(target_pos_light - key_pos)
            self.model.light_diffuse[self.light_id] = key_intensity * key_color

            # 2. FILL LIGHT
            fill_azimuth_offset = self.np_random.uniform(np.deg2rad(90), np.deg2rad(150)) * self.np_random.choice([-1, 1])
            fill_azimuth = key_azimuth + fill_azimuth_offset
            fill_elevation = self.np_random.uniform(np.deg2rad(30), np.deg2rad(50))
            fill_radius = self.np_random.uniform(1.2, 1.8)
            fill_pos = target_pos_light + np.array([
                fill_radius * np.cos(fill_elevation) * np.cos(fill_azimuth),
                fill_radius * np.cos(fill_elevation) * np.sin(fill_azimuth),
                fill_radius * np.sin(fill_elevation)
            ])
            self.model.light_pos[self.fill_light_id] = fill_pos
            self.model.light_dir[self.fill_light_id] = self._safe_normalize(target_pos_light - fill_pos)
            self.model.light_diffuse[self.fill_light_id] = fill_intensity * fill_color

            # 3. BACK LIGHT (Rim Light)
            back_azimuth = key_azimuth + np.pi + self.np_random.uniform(np.deg2rad(-45), np.deg2rad(45))
            back_elevation = self.np_random.uniform(np.deg2rad(40), np.deg2rad(60))
            back_radius = self.np_random.uniform(1.5, 2.0)
            back_pos = target_pos_light + np.array([
                back_radius * np.cos(back_elevation) * np.cos(back_azimuth),
                back_radius * np.cos(back_elevation) * np.sin(back_azimuth),
                back_radius * np.sin(back_elevation)
            ])
            self.model.light_pos[self.back_light_id] = back_pos
            self.model.light_dir[self.back_light_id] = self._safe_normalize(target_pos_light - back_pos)
            self.model.light_diffuse[self.back_light_id] = back_intensity * key_color
            # print(f"DEBUG: Main Light DIR: {np.round(self.model.light_dir[self.light_id], 2)}")
            # print(f"DEBUG: Fill Light DIR: {np.round(self.model.light_dir[self.fill_light_id], 2)}")
    def render(self, camera_name: str = "fixed_camera"):
        """
        Robust and FAST renderer with filmic post-processing.
        
        This optimized version maintains a single, large renderer (256x256) and
        downsamples for smaller camera views. This avoids the extremely slow process
        of destroying and re-creating the renderer every step.
        """
        is_wrist = "wrist" in camera_name
        target_key = "image_wrist" if is_wrist else "image_primary"
        target_h, target_w, _ = self.observation_space[target_key].shape

        if self.render_mode != "rgb_array":
            warnings.warn(f"Render mode is not 'rgb_array', returning a black frame for camera '{camera_name}'.")
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        try:
            # --- START OF FAST RENDERING LOGIC ---
            # 1. Ensure the renderer exists and is at the MAXIMUM resolution (256x256).
            # This logic only runs if the renderer is missing or has been closed.
            max_h, max_w, _ = self.observation_space["image_primary"].shape

            if (self.renderer is None) or (getattr(self.renderer, "width", None) != max_w or getattr(self.renderer, "height", None) != max_h):
                if self.renderer is not None:
                    try:
                        self.renderer.close()
                    except Exception:
                        pass

            # 2. Render the scene at the native 256x256 resolution.
            self.renderer.update_scene(self.data, camera=camera_name)
            img_raw_large = self.renderer.render()
            
            # 3. Apply necessary patches (flipping and cache clearing).
            if self._decide_flip_for_camera(camera_name):
                img_raw_large = np.flipud(img_raw_large)
            
            if is_wrist and hasattr(self, "_camera_flip_cache"):
                if camera_name in self._camera_flip_cache:
                    del self._camera_flip_cache[camera_name]
            # --- END OF FAST RENDERING LOGIC ---

        except Exception as e:
            warnings.warn(f"Failed to render from camera '{camera_name}': {e}")
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # 4. Apply post-processing ON THE LARGE IMAGE.
        # This is optional and can be disabled for even more speed.
        if self.post:
            img_processed_large = self._postprocess_image(img_raw_large)
        else:
            img_processed_large = img_raw_large

        # 5. Downsample to the target shape ONLY IF NECESSARY (e.g., for wrist camera).
        if img_processed_large.shape[0] != target_h or img_processed_large.shape[1] != target_w:
            # Use INTER_AREA for high-quality downsampling.
            img_out = cv2.resize(img_processed_large, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            img_out = img_processed_large

        return img_out
    def get_ee_pose(self) -> np.ndarray:
        """
        Calculates and returns the current 7D pose of the end-effector.
        This version is optimized by using a cached site ID.
        """
        # Use the cached ID for efficient access
        pos = self.data.site_xpos[self.ee_site_id].copy()
        
        # site_xmat is a flat 9-element array (row-major 3x3 matrix)
        rot_matrix = self.data.site_xmat[self.ee_site_id].copy().reshape(3, 3)
        
        quat_xyzw = R.from_matrix(rot_matrix).as_quat()
        
        return np.concatenate([pos, quat_xyzw]).astype(np.float32)

    def set_rendering_enabled(self, enabled: bool):
        """Toggle rendering on/off to speed up the physics-only pass."""
        self._rendering_enabled = getattr(self, "_rendering_enabled", True)
        self._rendering_enabled = enabled


    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        Returns a clean observation dictionary that matches the observation_space.
        """
        # Get base proprioceptive state (joint positions and velocities)
        qpos = np.asarray(self.data.qpos, dtype=np.float32)
        qvel = np.asarray(self.data.qvel, dtype=np.float32)
        left_touch = self.data.sensordata[self.left_touch_sensor_id]
        right_touch = self.data.sensordata[self.right_touch_sensor_id]
        left_force = self.data.sensordata[self.left_force_sensor_id : self.left_force_sensor_id + 3]
        right_force = self.data.sensordata[self.right_force_sensor_id : self.right_force_sensor_id + 3]
        
        proprio = np.concatenate([
            qpos[:7], 
            qvel[:7], 
            np.array([left_touch, right_touch]),
            left_force,
            right_force
        ])
        if getattr(self, "_rendering_enabled", True):
            image_primary = self.render(camera_name="fixed_camera")
            image_wrist = self.render(camera_name="wrist_camera")
        else:
            # Fast placeholders (shapes must match space)
            image_primary = np.zeros((256, 256, 3), dtype=np.uint8)
            image_wrist = np.zeros((128, 128, 3), dtype=np.uint8)

        return {
            "image_primary": image_primary,
            "image_wrist": image_wrist,
            "proprio": proprio,
            "is_grasped": np.array([self._is_physically_grasped], dtype=np.float32),
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

    def get_object_orientation_expert(self) -> np.ndarray:
        """
        Retrieves the object's orientation quaternion from the MuJoCo simulation data.
        Assumes object_joint is a free joint (pos + quat).
        Returns quaternion in [x, y, z, w] format for compatibility with scipy.spatial.transform.Rotation.
        """

        object_qpos_addr = self.model.jnt_qposadr[self.object_joint_id]

        quat_wxyz = self.data.qpos[object_qpos_addr + 3 : object_qpos_addr + 7].copy()

        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)

        return quat_xyzw
    def get_goal_orientation_expert(self) -> np.ndarray:
        """Gets the ground-truth world orientation of the goal for the expert as an xyzw quat."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal")
        if body_id == -1:
            raise ValueError("Body 'goal' not found for expert pipeline.")
        # Goals are simple bodies, their orientation is in `xquat`
        quat_wxyz = self.data.xquat[body_id].copy()
        return self._mujoco_quat_to_scipy_xyzw(quat_wxyz)
        
    def get_expert_obs(self) -> Dict[str, np.ndarray]:
        """
        Returns a rich observation dictionary for the expert pipeline.

        This includes the standard observation (with both camera views) plus
        ground-truth state information required by the expert.
        """
        # Start with the standard observation, which now includes the wrist image.
        obs = self._get_obs()
        goal_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_geom")
        goal_size_full = self.model.geom_size[goal_geom_id] * 2
        obs["goal_size_world"] = goal_size_full.astype(np.float32)
        # Add ground-truth data required ONLY by the expert.
        obs["ee_pose_world"] = self.get_ee_pose()
        obs["object_pos_world"] = self.get_object_pos_expert()
        obs["goal_pos_world"] = self.get_goal_pos_expert()
        obs['initial_image'] = self._initial_image
        obs['goal_image'] = self._goal_image
        # Add the redundant proprio key required by the IKSolver.
        # This isolates the redundancy to the expert pipeline, which is a good design.
        obs["internal_full_proprio"] = obs["proprio"].copy()
        # Use the single, correct flag and match the float32 dtype of the observation space
        obs["is_grasped"] = np.array([self._is_physically_grasped], dtype=np.float32)
        obs["object_orn_world"] = self.get_object_orientation_expert()
        obs["goal_orn_world"] = self.get_goal_orientation_expert()
        finger_joint1_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
        finger_joint2_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
        
        obs["gripper_qpos"] = np.array([
            self.data.qpos[self.model.jnt_qposadr[finger_joint1_idx]],
            self.data.qpos[self.model.jnt_qposadr[finger_joint2_idx]]
        ], dtype=np.float32)
        base_pos, _ = self.get_base_pose()
        obs["robot_base_pos_world"] = base_pos
        ee_vel_6d = np.zeros(6, dtype=np.float64) 
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_id, ee_vel_6d, 0)
        obs["ee_vel"] = ee_vel_6d.astype(np.float32) # Cast to float32 for observation consistency

        # 2. Get Object 6D Velocity (Twist)
        # MuJoCo functions require float64 arrays
        object_vel_6d = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.object_body_id, object_vel_6d, 0)
        obs["object_vel"] = object_vel_6d.astype(np.float32) # Cast to float32

        # 3. Get Gripper Joint Velocities
        finger_joint1_vel_idx = self.model.jnt_dofadr[finger_joint1_idx]
        finger_joint2_vel_idx = self.model.jnt_dofadr[finger_joint2_idx]
        obs["gripper_vel"] = np.array([
            self.data.qvel[finger_joint1_vel_idx],
            self.data.qvel[finger_joint2_vel_idx]
        ], dtype=np.float32)

        obs["camera_params"] = self.get_camera_params("fixed_camera")

        return obs
      
    def compensatory_clamp_xy(self, orig_xy, goal_xy, x_min, x_max, y_min, y_max, max_y_offset=None):
        """
        Clamp orig_xy = (x_orig, y_orig) with respect to goal_xy = (gx, gy),
        so that x_new ∈ [x_min, x_max], while preserving distance to goal as much as possible,
        by compensating in y-direction. Optionally limit how far Y can shift.
        Returns new (x_new, y_new).
        """
        x_orig, y_orig = float(orig_xy[0]), float(orig_xy[1])
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        
        x_new = x_orig
        y_new = y_orig

        # If within bounds already, just clip and return
        if x_min <= x_orig <= x_max:
            y_new = np.clip(y_new, y_min, y_max)
            return x_new, y_new

        # Determine which bound is violated
        if x_orig > x_max:
            x_new = x_max
        elif x_orig < x_min:
            x_new = x_min

        # Compute squared distances
        dx0 = x_orig - gx
        dy0 = y_orig - gy
        dist_sq = dx0 * dx0 + dy0 * dy0

        dx_new = x_new - gx
        dx_new_sq = dx_new * dx_new

        # Clamp negative underflows
        rem = dist_sq - dx_new_sq
        if rem < 0:
            rem = 0.0

        # Compute candidate Y offsets
        y_offset = np.sqrt(rem)
        # Choose sign consistent with original side of goal
        if y_orig >= gy:
            y_new = gy + y_offset
        else:
            y_new = gy - y_offset

        # Optionally limit how far Y can shift
        if max_y_offset is not None:
            # cap |y_new − original y|
            max_shift = abs(max_y_offset)
            delta_y = y_new - y_orig
            if abs(delta_y) > max_shift:
                y_new = y_orig + np.sign(delta_y) * max_shift

        # Finally, clip y_new to workspace bounds
        y_new = np.clip(y_new, y_min, y_max)

        return x_new, y_new
    


    def reset(self, seed: int = None, options: dict = None) -> Tuple[Dict, Dict]:
        """
        Reset with Object-Centric Randomization (Legacy Panning Logic).
        Sequence: 1. Physics -> 2. Robot -> 3. Place Objects -> 4. Randomize Camera (Tracking)
        """
        super().reset(seed=seed)
        if seed is not None: self.np_random, _ = seeding.np_random(seed)
        
        self.timestep = 0
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0
        
        # 1. Physics Randomization
        object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        new_mass = self.np_random.uniform(low=0.1, high=0.5)
        self.model.geom_friction[self.object_geom_id][0] = self.np_random.uniform(low=0.5, high=1.2)
        self.model.body_mass[object_body_id] = new_mass

        # 2. Robot Initialization
        home_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        qpos_jitter = self.np_random.uniform(-0.05, 0.05, size=home_qpos.shape)
        self.data.qpos[:7] = home_qpos + qpos_jitter
        
        # 3. Object Placement (Before Camera Randomization)
        # We use the safe zone we defined
        safe_full_zone = (np.array([-0.20, -0.20]), np.array([0.15, 0.20]))
        
        # Note: We DISABLE check_visibility here because the camera hasn't moved yet.
        # We rely on the camera moving TO the object in the next step.
        object_pos = self._place_object_in_zone(
            "object", "full_table", safe_full_zone, self.OBJECT_Z_HEIGHT,
            camera_name="fixed_camera", check_visibility=False
        )
        
        goal_pos = self._place_object_in_zone(
            "goal", "full_table", safe_full_zone, self.GOAL_Z_HEIGHT,
            camera_name="fixed_camera", check_visibility=False
        )

        # 4. Rotational Randomization
        obj_yaw = self.np_random.uniform(-np.pi, np.pi)
        obj_quat_wxyz = self._scipy_xyzw_to_mujoco_wxyz(R.from_euler('z', obj_yaw).as_quat())
        goal_yaw = self.np_random.uniform(-np.pi, np.pi)
        goal_quat_wxyz = self._scipy_xyzw_to_mujoco_wxyz(R.from_euler('z', goal_yaw).as_quat())

        # 5. Apply Objects to Physics
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_joint")
        qpos_adr = int(self.model.jnt_qposadr[joint_id])
        self.data.qpos[qpos_adr : qpos_adr+3] = object_pos
        self.data.qpos[qpos_adr+3 : qpos_adr+7] = obj_quat_wxyz

        goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal")
        self.data.xpos[goal_body_id] = goal_pos
        self.data.xquat[goal_body_id] = goal_quat_wxyz
        
        # Update physics so get_ee_pose is correct
        mujoco.mj_forward(self.model, self.data)

        # 6. Camera Randomization (Tracking)
        # Now that objects are placed, we move the camera to look at them.
        initial_ee_pos = self.get_ee_pose()[:3]
        self._apply_domain_randomization(initial_ee_pos, goal_pos)

        # 7. Final Settle
        mujoco.mj_forward(self.model, self.data)
        
        self._is_physically_grasped = False
        self._grasp_stabilization_counter = 0
        self._grasp_pos_offset = None
        self._grasp_orn_offset = None
        
        # 8. Render
        if getattr(self, "_rendering_enabled", True):
            self._initial_image = self.render(camera_name="fixed_camera")
            self._goal_image = self._render_goal_image()
        else:
            h, w, c = self.observation_space["image_primary"].shape
            self._initial_image = np.zeros((h, w, c), dtype=np.uint8)
            self._goal_image = np.zeros((h, w, c), dtype=np.uint8)
        
        return self.get_expert_obs(), {}


    def _place_object_in_zone(
        self,
        object_name: str,
        zone_key: str,
        zone: Tuple[np.ndarray, np.ndarray],
        z_plane: float,
        camera_name: str,
        check_visibility: bool = True,
        max_attempts: int = 100,
        margin: int = 20,
    ) -> np.ndarray:
        """
        Sample points inside the zone and return the first position that projects
        inside the provided camera view. Falls back to zone center if none found.
        Returns an (x,y,z) world position.
        """
        # FIX: Moved this line to the top of the function.
        # It now runs before any logic that depends on it.
        min_offset, max_offset = zone
        if not check_visibility:
            # Perform a blind placement without any visibility checks.
            offset = self.np_random.uniform(low=min_offset, high=max_offset)
            return np.append(self.TABLE_CENTER + offset, z_plane)

        # The rest of the logic for visibility checks.
        debug_printed = False
        for attempt in range(max_attempts):
            offset = self.np_random.uniform(low=min_offset, high=max_offset)
            candidate = np.append(self.TABLE_CENTER + offset, z_plane)

            visible, debug = self._is_pos_in_camera_view(candidate, camera_name, margin=margin)
            if visible:
                return candidate

            if not debug_printed:
                print("\n" + "="*80)
                print(f"DEBUG: Visibility Check FAILED for '{object_name}' in zone '{zone_key}'")
                print(f"Attempt {attempt+1}/{max_attempts}. Candidate: {candidate}")
                for k,v in debug.items():
                    if isinstance(v, np.ndarray):
                        print(f"  - {k}: {np.array2string(v, precision=4, suppress_small=True)}")
                    else:
                        print(f"  - {k}: {v}")
                print("="*80 + "\n")
                debug_printed = True

        # Fallback to zone center
        center_offset = 0.5 * (min_offset + max_offset)
        center = np.append(self.TABLE_CENTER + center_offset, z_plane)
        visible_center, _ = self._is_pos_in_camera_view(center, camera_name, margin=margin)
        if visible_center:
            return center

        warnings.warn(
            f"Could not find visible position for '{object_name}' in zone '{zone_key}'. "
            "Falling back to zone center. Check camera placement and table coordinates."
        )
        return center

    def _camera_extrinsics(self, camera_id: int):
        """Return cam_pos (3,), R_wc (3x3 rotation camera->world), fx, fy, cx, cy, height, width."""
        # image size

        cam_name_bytes = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)
        camera_name = cam_name_bytes.decode() if isinstance(cam_name_bytes, (bytes, bytearray)) else str(cam_name_bytes)
        target_key = "image_wrist" if "wrist" in camera_name else "image_primary"
        height, width, _ = self.observation_space.spaces[target_key].shape


        cam_pos = np.array(self.model.cam_pos[camera_id], dtype=float)

        # MuJoCo stores cam_quat as WXYZ ; scipy expects (x,y,z,w)
        quat_wxyz = np.asarray(self.model.cam_quat[camera_id], dtype=float)
        quat_xyzw = self._mujoco_quat_to_scipy_xyzw(quat_wxyz)
        R_wc = R.from_quat(quat_xyzw).as_matrix()  # rotation: camera -> world


        # intrinsics from model.cam_fovy (vertical FOV in degrees)
        if self.model.cam_fovy.size > camera_id:
            fovy_deg = float(self.model.cam_fovy[camera_id])
        else:
            fovy_deg = 45.0
        fovy = np.deg2rad(fovy_deg)
        fy = 0.5 * height / np.tan(0.5 * fovy)
        fx = fy * (width / height)
        cx = 0.5 * width
        cy = 0.5 * height

        return cam_pos, R_wc, fx, fy, cx, cy, height, width
    
    
    def get_object_orientation_expert(self) -> np.ndarray:
        """Gets the ground-truth world orientation of the object for the expert as an xyzw quat."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        if body_id == -1:
            raise ValueError("Body 'object' not found for expert pipeline.")
        quat_wxyz = self.data.xquat[body_id].copy()
        return self._mujoco_quat_to_scipy_xyzw(quat_wxyz)

    def _update_grasp_physical(self, gripper_action: float):
        """
        Updates grasp state based purely on physical conditions.
        The is_grasped flag can flicker, representing the noisy physical reality.
        """
        # Asymmetric stabilization counters for robustness against jitter
        GRASP_REQUIRED_STEPS = 5
        RELEASE_REQUIRED_STEPS = 15

        # Determine the raw physical state of the gripper/object contact
        # (This logic is the same as our previous robust physical checker)
        left_force_adr = self.model.sensor_adr[self.left_force_sensor_id]
        right_force_adr = self.model.sensor_adr[self.right_force_sensor_id]
        left_force_vec = self.data.sensordata[left_force_adr : left_force_adr + 3]
        right_force_vec = self.data.sensordata[right_force_adr : right_force_adr + 3]

        has_bilateral_contact = (
            self._has_geom_contact(self.left_finger_geoms, self.object_geoms) and
            self._has_geom_contact(self.right_finger_geoms, self.object_geoms)
        )
        has_sufficient_force = (np.linalg.norm(left_force_vec) > 0.5) and (np.linalg.norm(right_force_vec) > 0.5)
        is_gripping_command = gripper_action < -0.1
        
        is_grasp_conditions_met = is_gripping_command and has_bilateral_contact and has_sufficient_force

        # Update the grasp flag using asymmetric debouncing
        if self._is_physically_grasped:
            if not is_grasp_conditions_met:
                self._grasp_condition_counter += 1
                if self._grasp_condition_counter >= RELEASE_REQUIRED_STEPS:
                    self._is_physically_grasped = False
            else:
                self._grasp_condition_counter = 0
        else:
            if is_grasp_conditions_met:
                self._grasp_condition_counter += 1
                if self._grasp_condition_counter >= GRASP_REQUIRED_STEPS:
                    self._is_physically_grasped = True
            else:
                self._grasp_condition_counter = 0

    def _update_grasp_stateful(self, gripper_action: float):
        """
        Updates grasp state using a latched, stateful model.
        Once grasped, the object is kinematically welded until an explicit release.
        """
        GRASP_CONFIRM_STEPS = 5
        RELEASE_CONFIRM_STEPS = 5

        # Determine the raw physical state
        has_bilateral_contact = (
            self._has_geom_contact(self.left_finger_geoms, self.object_geoms) and
            self._has_geom_contact(self.right_finger_geoms, self.object_geoms)
        )
        left_force_adr = self.model.sensor_adr[self.left_force_sensor_id]
        right_force_adr = self.model.sensor_adr[self.right_force_sensor_id]
        left_force_vec = self.data.sensordata[left_force_adr : left_force_adr + 3]
        right_force_vec = self.data.sensordata[right_force_adr : right_force_adr + 3]
        has_sufficient_force = (np.linalg.norm(left_force_vec) > 0.5) and (np.linalg.norm(right_force_vec) > 0.5)
        physical_conditions_met = has_bilateral_contact and has_sufficient_force

        # State machine logic
        if not self._is_physically_grasped:
            # Entry condition: command and physics must align
            is_gripping_command = gripper_action < -0.1
            if is_gripping_command and physical_conditions_met:
                self._grasp_condition_counter += 1
            else:
                self._grasp_condition_counter = 0

            if self._grasp_condition_counter >= GRASP_CONFIRM_STEPS:
                self._is_physically_grasped = True
                ee_pos, ee_quat_xyzw = self.get_ee_pose()[:3], self.get_ee_pose()[3:]
                R_ee_world = R.from_quat(ee_quat_xyzw)
                obj_pos, obj_quat_wxyz = self.data.xpos[self.object_body_id].copy(), self.data.xquat[self.object_body_id].copy()
                R_obj_world = R.from_quat(self._mujoco_quat_to_scipy_xyzw(obj_quat_wxyz))
                self._grasp_pos_offset = R_ee_world.inv().apply(obj_pos - ee_pos)
                self._grasp_orn_offset = R_ee_world.inv() * R_obj_world
        else:
            # Exit condition: command and loss of contact must align
            is_releasing_command = gripper_action > 0.1
            if is_releasing_command and not has_bilateral_contact:
                self._grasp_condition_counter += 1
            else:
                self._grasp_condition_counter = 0

            if self._grasp_condition_counter >= RELEASE_CONFIRM_STEPS:
                self._is_physically_grasped = False
                self._grasp_pos_offset, self._grasp_orn_offset = None, None

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        self.timestep += 1
        action = np.asarray(action, dtype=float).ravel()
        arm_action = action[:7]
        gripper_action = action[7]
        
        if self.control_mode == 'absolute':
            # --- This is your ORIGINAL logic, used by the expert ---
            # The action is an absolute, normalized target joint position [-1, 1].
            arm_ctrl_range = self.model.actuator_ctrlrange[:7]
            arm_lo, arm_hi = arm_ctrl_range[:, 0], arm_ctrl_range[:, 1]
            scaled_arm_action = arm_lo + 0.5 * (arm_action + 1.0) * (arm_hi - arm_lo)
            self.data.ctrl[:7] = scaled_arm_action

        elif self.control_mode == 'delta':
            # This is the physically corrected delta mode.
            physical_delta = arm_action * self.ACTION_SCALING_FACTOR
            current_qpos = self.data.qpos[:7].copy()
            target_qpos = current_qpos + physical_delta

            joint_limits = self.model.jnt_range[:7]
            jnt_lo, jnt_hi = joint_limits[:, 0], joint_limits[:, 1]
            target_qpos = np.clip(target_qpos, jnt_lo, jnt_hi)

            if getattr(self, 'debug_instant_move', False):
                # --- START OF FINAL FIX ---
                # 1. Teleport the joint to the target position
                self.data.qpos[:7] = target_qpos
                
                # 2. CRITICAL: Also update the controller's target to match.
                #    This prevents the controller from fighting the teleport.
                denom = np.where(np.abs(jnt_hi - jnt_lo) > 1e-9, (jnt_hi - jnt_lo), 1.0)
                norm = 2.0 * (target_qpos - jnt_lo) / denom - 1.0
                
                arm_ctrl_range = np.asarray(self.model.actuator_ctrlrange[:7], dtype=float)
                act_lo, act_hi = arm_ctrl_range[:, 0], arm_ctrl_range[:, 1]
                scaled_ctrl = act_lo + 0.5 * (norm + 1.0) * (act_hi - act_lo)
                self.data.ctrl[:7] = scaled_ctrl

                mujoco.mj_forward(self.model, self.data)
                # --- END OF FINAL FIX ---
            else:
                # For real training: Use the physically realistic controller
                denom = np.where(np.abs(jnt_hi - jnt_lo) > 1e-9, (jnt_hi - jnt_lo), 1.0)
                norm = 2.0 * (target_qpos - jnt_lo) / denom - 1.0
                
                arm_ctrl_range = np.asarray(self.model.actuator_ctrlrange[:7], dtype=float)
                act_lo, act_hi = arm_ctrl_range[:, 0], arm_ctrl_range[:, 1]
                scaled_ctrl = act_lo + 0.5 * (norm + 1.0) * (act_hi - act_lo)
                self.data.ctrl[:7] = scaled_ctrl


        scaled_gripper_action = (gripper_action + 1.0) / 2.0 * 0.04
        self.data.ctrl[7] = scaled_gripper_action
        self.data.ctrl[8] = scaled_gripper_action
        
        # --- START OF GRASP DETECTION PATCH ---
        # Update substeps to maintain control frequency with smaller timestep
        N_SUBSTEPS = 20
        for _ in range(N_SUBSTEPS):
            # Dispatch to the correct grasp update logic based on the environment's mode
            if self.grasp_mode == "stateful":
                self._update_grasp_stateful(gripper_action)
            else: # self.grasp_mode == "physical"
                self._update_grasp_physical(gripper_action)

            mujoco.mj_step(self.model, self.data)

            # The kinematic weld is only applied in stateful mode
            if self.grasp_mode == "stateful" and self._is_physically_grasped:
                ee_pos, ee_quat_xyzw = self.get_ee_pose()[:3], self.get_ee_pose()[3:]
                R_ee_world = R.from_quat(ee_quat_xyzw)
                new_obj_pos = ee_pos + R_ee_world.apply(self._grasp_pos_offset)
                new_obj_orn = (R_ee_world * self._grasp_orn_offset)
                qpos_addr = self.model.jnt_qposadr[self.object_joint_id]
                self.data.qpos[qpos_addr : qpos_addr + 3] = new_obj_pos
                self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = self._scipy_xyzw_to_mujoco_wxyz(new_obj_orn.as_quat())
        
        mujoco.mj_forward(self.model, self.data)    


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
    


# Add this method
    def get_mj_state(self) -> mujoco.MjData:
        """Returns a deep copy of the full simulation state (qpos, qvel, etc.)."""
        return copy.deepcopy(self.data)

    # Add this method
    def set_mj_state(self, state: mujoco.MjData):
        """Sets the full simulation state from a saved MjData object."""
        self.data.qpos[:] = state.qpos
        self.data.qvel[:] = state.qvel
        self.data.act[:] = state.act
        self.data.time = state.time
        # You may need to copy other fields depending on your env, but these are the core ones.
        mujoco.mj_forward(self.model, self.data)


    def get_action_space_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        [SOTA, DEFINITIVE VERSION]
        Programmatically extracts the action space limits directly from the MuJoCo model.

        This is the single source of truth for robot kinematics, ensuring that
        the policy's constraints always match the simulation's constraints.

        Returns:
            A tuple of (low_limits, high_limits) for the entire 8-DoF action space.
        """
        low_limits = []
        high_limits = []

        # 1. Get limits for the 7 arm joints
        for i in range(1, 8): # Joints are named joint1 through joint7
            joint_name = f"joint{i}"
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id == -1:
                raise ValueError(f"Joint '{joint_name}' not found in the MuJoCo model.")
            
            # The range is stored in model.jnt_range
            jnt_range = self.model.jnt_range[joint_id]
            low_limits.append(jnt_range[0])
            high_limits.append(jnt_range[1])
            
        # 2. Add the abstract limits for the gripper action (-1 to 1)
        # This corresponds to the normalized gripper command, not the physical range.
        low_limits.append(-1.0)
        high_limits.append(1.0)
        
        return np.array(low_limits, dtype=np.float32), np.array(high_limits, dtype=np.float32)
    

    def debug_force_camera_shot(self, shot_index: int):
        """
        DEBUG ONLY: Forces the camera to a specific index from the config list.
        Bypasses all jitter and randomization to visualize the 'Ideal' shot.
        """
        if not (0 <= shot_index < len(self.dr_config.camera_shots)):
            print(f"Warning: Shot index {shot_index} out of bounds.")
            return

        shot = self.dr_config.camera_shots[shot_index]
        
        # 1. Set Position
        cam_pos = np.array(shot.pos)
        target_pos = np.array(shot.target)
        
        # 2. Calculate Orientation (Look-At)
        quat_xyzw = self._calculate_look_at_quat(cam_pos, target_pos)
        quat_wxyz = self._scipy_xyzw_to_mujoco_wxyz(quat_xyzw)
        
        # 3. Apply to Model
        self.model.cam_pos[self.camera_id] = cam_pos
        self.model.cam_quat[self.camera_id] = quat_wxyz
        
        # 4. Reset FOV to a standard value (e.g. 45) to judge the position purely
        self.model.cam_fovy[self.camera_id] = 45.0
        
        # 5. Update scene
        mujoco.mj_forward(self.model, self.data)