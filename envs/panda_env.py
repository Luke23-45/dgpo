# envs/panda_env.py (Final, Production-Grade, OCTO-Compliant Version)
import warnings
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from typing import Tuple, Dict,List, Callable
from scipy.spatial.transform import Rotation as R
from utils.mujoco_utils import set_joint_qpos_by_name
import cv2
from dataclasses import dataclass, field 

@dataclass
class RenderPostConfig:
    """
    Controls photometric post-processing for enhanced realism.
    """
    apply_tonemap: bool = True
    tonemap_curve: str = "aces"          # ["aces", "reinhard"]
    # Auto-exposure aims to map the chosen luminance percentile to target_white
    auto_exposure_percentile: float = 0.98
    target_white: float = 0.8          # in linear space
    min_exposure: float = 0.7
    max_exposure: float = 1.2

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
        "mat_table_metal_brushed", "mat_table_noise_low"
    ])
    floor_textures: List[str] = field(default_factory=lambda: [
        "mat_floor_checker_blue", "mat_floor_checker_green",
        "mat_floor_wood_dark", "mat_floor_wood_paquet"
    ])

    # ============================ CURATED EXEMPLAR SHOTS ============================
    # FINAL PATCH: This new list is mined from the best results in your JSON data.
    # It provides a wider, more robust, and higher-quality set of base viewpoints.
    camera_shots: List[CameraShot] = field(default_factory=lambda: [
        # --- Right Three-Quarter Views ---
        # (From Shot_01/sample_00) - A perfect classic view. Elevation: 39°
        CameraShot(pos=(0.88, -0.44, 0.95), target=(0.42, -0.02, 0.44)),
        # (From Shot_04/sample_01) - A slightly wider right view. Elevation: 45.1°
        CameraShot(pos=(1.07, -0.25, 1.09), target=(0.49, -0.03, 0.44)),
        # (From Shot_04/sample_01) - A slightly different angle, good composition. Elevation: 53°
        CameraShot(pos=(0.57, 0.58, 1.01), target=(0.44, -0.01, 0.43)),


        # --- Left Three-Quarter Views ---
        # (From Shot_02/sample_00) - Excellent left-side view. Elevation: 34°
        CameraShot(pos=(0.83, 0.49, 0.87), target=(0.48, -0.03, 0.43)),
        # (From Shot_06/sample_01) - A wider left-side view. Elevation: 35°
        CameraShot(pos=(1.08, 0.36, 0.97), target=(0.47, 0.06, 0.44)),

        # --- Frontal Views ---
        # (From Shot_09/sample_12) - A centered, slightly higher frontal view. Elevation: 41°
        # CameraShot(pos=(1.14, -0.01, 0.99), target=(0.52, 0.03, 0.43)),
        # (From Shot_09/sample_12) - A centered, slightly higher frontal view. Elevation: 38.4°
        CameraShot(pos=(1.02, 0.10, 0.95), target=(0.39, 0.04, 0.45)),
        # (From Shot_08/sample_00) - A well-balanced frontal shot. Elevation: 30°
        CameraShot(pos=(1.35, 0.34, 1.00), target=(0.44, -0.01, 0.45)),


        # --- High-Angle / Near Top-Down Views ---
        # (From Shot_03/sample_11) - A high three-quarter view, very informative. Elevation: 44°
        CameraShot(pos=(1.04, -0.04, 1.04), target=(0.43, 0.01, 0.43)),
        # (From Shot_07/sample_00) - A balanced top-down view, not too extreme. Elevation: 74°
        CameraShot(pos=(0.72, 0.00, 1.42), target=(0.45, 0.02, 0.43)),

        # --- Dynamic / Lower Views (Still Safe) ---
        # (From Shot_01/sample_09) - A lower, more dynamic angle that still works well. Elevation: 27.8°
        CameraShot(pos=(1.05, -0.37, 0.82), target=(0.44, -0.00, 0.45)),
        # (From Shot_05/sample_13) - A lower, more dynamic angle that still works well. Elevation: 18°
        # CameraShot(pos=(1.16, -0.27, 0.72), target=(0.48, -0.07, 0.45)),
        # (From Shot_06/sample_15) - A wide, cinematic left view. Elevation: 32°
        CameraShot(pos=(0.91, 0.49, 0.92), target=(0.38, 0.06, 0.45)),

    ])

    radius_jitter: float = 0.10      # meters (reduced from 0.15)
    azimuth_jitter: float = 0.26     # radians (~15 degrees) (reduced from 0.35)
    elevation_jitter: float = 0.17   # radians (~10 degrees) (reduced from 0.26)
    target_pos_jitter: float = 0.05  # meters (reduced from 0.08)
    fovy_jitter: float = 3.0         # degrees (reduced from 5.0)


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
        "center": (np.array([-0.05, -0.05]), np.array([0.05, 0.05])),
        "left":   (np.array([-0.15, -0.1]), np.array([-0.05, 0.1])),
        "right":  (np.array([0.05, -0.1]), np.array([0.15, 0.1])),
        "front":  (np.array([-0.15, -0.15]), np.array([0.15, -0.05])),
        "back":   (np.array([-0.15, 0.05]), np.array([0.15, 0.15])),
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

    def __init__(
            self,
            xml_path: str = "envs/panda_pick_place.xml",
            render_mode: str = "rgb_array",
            dr_config: DomainRandomizationConfig = None,
            enable_domain_randomization: bool = True,
            post_config: RenderPostConfig = None,
        ):
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
        self.post = post_config or RenderPostConfig()
        # --- Episode bookkeeping ---
        self.max_episode_steps = 250
        self.timestep = 0

        # --- Define Observation and Action Spaces (CRITICAL SECTION) ---
        self._define_spaces()

        # --- Random Number Generator ---
        self.np_random, _ = seeding.np_random(None)
        self.enable_domain_randomization = enable_domain_randomization
        self.dr_config = dr_config or DomainRandomizationConfig()

        # Cache IDs of elements to be randomized for performance
        self._cache_dr_element_ids()
        allowed_prefs = ["random", "left", "right", "front"]

        self.ee_site_name = "attachment_site"
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name)
        if self.ee_site_id == -1:
            raise ValueError(f"Site '{self.ee_site_name}' not found in the MuJoCo model.")





    def _cache_dr_element_ids(self):
        """Finds and caches the integer IDs of all elements used in DR."""
        if not self.enable_domain_randomization:
            self._dr_mat_ids = {}
            return

        # --- Cache critical element IDs (fail fast if missing) ---
        self.light_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_LIGHT, "main_light")
        self.table_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed_camera")

        if any(id_ == -1 for id_ in [self.light_id, self.table_geom_id, self.floor_geom_id, self.camera_id]):
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
  
    def _apply_domain_randomization(self, gripper_pos: np.ndarray, goal_pos: np.ndarray):
        """
        Handles all domain randomization using the new "Exemplar-Based Spherical Jitter" strategy.
        This guarantees high-quality, varied, and well-framed shots every time.
        """
        if not self.enable_domain_randomization:
            return

        # --- Part 1: Photometric Randomization (Lighting and Textures) ---
        self._randomize_photometrics()

        # --- Part 2: Geometric Randomization (Principled Camera Placement) ---

        # 1. Select a random high-quality "exemplar" shot from our curated list.
        chosen_shot = self.np_random.choice(self.dr_config.camera_shots)
        base_cam_pos = np.array(chosen_shot.pos)
        base_target_pos = np.array(chosen_shot.target)

        # 2. Define the dynamic "center of action" for this specific task.
        # We will aim the camera at the midpoint between the gripper and the goal.
        action_midpoint = (gripper_pos + goal_pos) / 2.0

        # 3. Add bounded, random jitter to the target position.
        # This creates small variations in framing (e.g., rule of thirds).
        target_jitter = self.np_random.uniform(-self.dr_config.target_pos_jitter,
                                               self.dr_config.target_pos_jitter,
                                               size=3)
        final_target_pos = action_midpoint + target_jitter
        # Ensure the camera isn't looking at the floor.
        final_target_pos[2] = max(final_target_pos[2], self.GOAL_Z_HEIGHT)

        # 4. Convert the exemplar's camera position to spherical coordinates relative to its target.
        radius, azimuth, elevation = self._cartesian_to_spherical(base_cam_pos, base_target_pos)

        # 5. Apply bounded, random jitter in the more intuitive spherical coordinate space.
        radius += self.np_random.uniform(-self.dr_config.radius_jitter, self.dr_config.radius_jitter)
        azimuth += self.np_random.uniform(-self.dr_config.azimuth_jitter, self.dr_config.azimuth_jitter)
        elevation += self.np_random.uniform(-self.dr_config.elevation_jitter, self.dr_config.elevation_jitter)

        # Clamp the elevation to prevent extreme low or high angles. This is a critical safety check.
        elevation = np.clip(elevation, np.deg2rad(20), np.deg2rad(75))
        radius = np.clip(radius, 0.8, 2.0) # Prevent camera from getting too close or far

        # 6. Reconstruct the new Cartesian camera position using the jittered spherical coords
        #    and the NEW dynamic target position.
        final_cam_pos = self._spherical_to_cartesian(radius, azimuth, elevation, final_target_pos)

        # 7. Calculate the final camera orientation and FOV.
        new_quat_xyzw = self._calculate_look_at_quat(final_cam_pos, final_target_pos)
        base_fovy = self.model.cam_fovy[self.camera_id]
        final_fovy = base_fovy + self.np_random.uniform(-self.dr_config.fovy_jitter,
                                                        self.dr_config.fovy_jitter)
        final_fovy = np.clip(final_fovy, 35.0, 80.0)

        # --- Part 3: Apply Final Camera Pose to the MuJoCo Model ---
        self.model.cam_pos[self.camera_id] = final_cam_pos
        self.model.cam_quat[self.camera_id] = [new_quat_xyzw[3], new_quat_xyzw[0], new_quat_xyzw[1], new_quat_xyzw[2]]
        self.model.cam_fovy[self.camera_id] = final_fovy


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
        height, width, _ = self.observation_space.spaces["image_primary"].shape
        debug_info["image_shape"] = (height, width)

        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id == -1:
            return False, {"error": f"Camera '{camera_name}' not found."}

        # Read the static camera pose from the model definition
        cam_pos = self.model.cam_pos[cam_id].copy()
        quat_wxyz = self.model.cam_quat[cam_id].copy()
        debug_info["cam_pos_model"] = cam_pos
        debug_info["cam_quat_model"] = quat_wxyz
        
        R_wc = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_matrix()
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

        # Final bounds check
        is_visible = (margin <= u < (width - margin)) and (margin <= v < (height - margin))
        if not is_visible:
            debug_info["failure_reason"] = "Projected pixel is outside the image margin."
            debug_info["bounds"] = {"u": u, "v": v, "margin": margin, "width": width, "height": height}
        
        return is_visible, debug_info
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


    def _randomize_photometrics(self):
        """Randomizes scene lighting and textures to improve policy robustness."""
        # --- Randomize Textures ---
        if self.dr_config.table_textures:
            chosen_table_tex = self.np_random.choice(self.dr_config.table_textures)
            if chosen_table_tex in self._dr_mat_ids:
                self.model.geom_matid[self.table_geom_id] = self._dr_mat_ids[chosen_table_tex]
        
        if self.dr_config.floor_textures:
            chosen_floor_tex = self.np_random.choice(self.dr_config.floor_textures)
            if chosen_floor_tex in self._dr_mat_ids:
                self.model.geom_matid[self.floor_geom_id] = self._dr_mat_ids[chosen_floor_tex]

        # --- Advanced Lighting Randomization ---
        # (This can remain largely the same as your previous version, as the XML change is more impactful)
        angle = self.np_random.uniform(0, 2 * np.pi)
        radius = self.np_random.uniform(1.0, 1.5)
        light_z = self.np_random.uniform(1.5, 2.5)
        self.model.light_pos[self.light_id] = [
            self.TABLE_CENTER[0] + radius * np.cos(angle),
            self.TABLE_CENTER[1] + radius * np.sin(angle),
            light_z
        ]
        target_pos_light = np.append(self.TABLE_CENTER, 0.4) + self.np_random.uniform(-0.1, 0.1, size=3)
        direction = target_pos_light - self.model.light_pos[self.light_id]
        self.model.light_dir[self.light_id] = self._safe_normalize(direction, default=np.array([0,0,-1]))

        kelvin_shift = self.np_random.uniform(-500, 500)
        tint = np.array([1.0 + (kelvin_shift / 2500.0), 1.0, 1.0 - (kelvin_shift / 2500.0)])
        tint = np.clip(tint, 0.85, 1.15)
        
        self.model.light_diffuse[self.light_id] = self.np_random.uniform(0.7, 1.0, 3) * tint

    def render(self, camera_name: str = "fixed_camera"):
        """
        Robust renderer with filmic post-processing.
        Always returns tonemapped sRGB uint8 frames at the requested shape.
        """
        is_wrist = "wrist" in camera_name
        target_key = "image_wrist" if is_wrist else "image_primary"
        target_h, target_w, _ = self.observation_space[target_key].shape

        if self.render_mode != "rgb_array" or self.renderer is None:
            warnings.warn(f"Renderer not available, returning a black frame for camera '{camera_name}'.")
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # Render once at the native 256x256 resolution
        try:
            # Ensure renderer is at the primary camera's resolution
            if self.renderer.width != self.observation_space["image_primary"].shape[1]:
                 self.renderer.width = self.observation_space["image_primary"].shape[1]
                 self.renderer.height = self.observation_space["image_primary"].shape[0]

            self.renderer.update_scene(self.data, camera=camera_name)
            img_raw = self.renderer.render()  # uint8, sRGB-ish
        except Exception as e:
            warnings.warn(f"Failed to render from camera '{camera_name}': {e}")
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # Photometric post-processing (tone map etc.) at the native render resolution
        img_processed = self._postprocess_image(img_raw)

        # Downsample to target shape if needed (e.g., for the wrist camera)
        if img_processed.shape[0] != target_h or img_processed.shape[1] != target_w:
            img_out = cv2.resize(img_processed, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            img_out = img_processed

        return img_out

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

    def reset(self, seed: int = None, options: dict = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        if seed is not None: self.np_random, _ = seeding.np_random(seed)
        
        self.timestep = 0
        mujoco.mj_resetData(self.model, self.data)

        # === STAGE 1: UNBIASED TASK GENERATION ===
        # 1a. Reset robot to a jittered home position.
        home_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        qpos_jitter = self.np_random.uniform(-0.03, 0.03, size=home_qpos.shape)
        self.data.qpos[:7] = home_qpos + qpos_jitter
        mujoco.mj_forward(self.model, self.data) # CRITICAL: Update kinematics for a valid gripper pose.

        # 1b. Perform BLIND placement of goal and object to get candidate positions.
        #     This ensures a truly random and unbiased task distribution.
        obj_zone_key, goal_zone_key = self.np_random.choice(list(self.PLACEMENT_ZONES.keys()), 2, replace=True)
        obj_zone, goal_zone = self.PLACEMENT_ZONES[obj_zone_key], self.PLACEMENT_ZONES[goal_zone_key]
        
        object_pos = self._place_object_in_zone("object", obj_zone_key, obj_zone, self.OBJECT_Z_HEIGHT, check_visibility=False)
        goal_pos   = self._place_object_in_zone("goal", goal_zone_key, goal_zone, self.GOAL_Z_HEIGHT, check_visibility=False)

        # === STAGE 2: ADAPTIVE CAMERA PLACEMENT & DOMAIN RANDOMIZATION ===
        # 2a. Get key positions to inform the camera logic.
        gripper_pos = self.get_ee_pose()[:3]
        
        # 2b. Delegate all camera and DR logic to the refactored helper function.
        self._apply_domain_randomization(gripper_pos, goal_pos)

        # === STAGE 3: COMMIT SCENE & FINALIZE ===
        # 3a. Now that the camera is set, commit the object and goal positions to the simulation state.
        if np.linalg.norm(goal_pos[:2] - object_pos[:2]) < 0.05:
            goal_pos[0] += 0.05 # Ensure a small separation if they spawn too close.
        
        self.data.joint("object_joint").qpos[:3] = object_pos
        self.model.body("goal").pos = goal_pos
        
        # 3b. Final forward pass to ensure all changes (camera, objects) are reflected.
        mujoco.mj_forward(self.model, self.data)
        
        return self.get_expert_obs(), {}

    # ======================== REPLACE THIS ENTIRE METHOD ========================

    def _place_object_in_zone(
        self,
        object_name: str,
        zone_key: str,
        zone: Tuple[np.ndarray, np.ndarray],
        z_plane: float,
        check_visibility: bool = True,
        camera_name: str = "fixed_camera",
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
        height, width, _ = self.observation_space["image_primary"].shape

        cam_pos = np.array(self.model.cam_pos[camera_id], dtype=float)

        # MuJoCo stores cam_quat as WXYZ ; scipy expects (x,y,z,w)
        quat_wxyz = np.array(self.model.cam_quat[camera_id], dtype=float)
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=float)
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
    


