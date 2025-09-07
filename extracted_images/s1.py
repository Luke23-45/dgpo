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
    # This new list is mined from the best results in your JSON data.
    # It provides a wider, more robust, and higher-quality set of base viewpoints.
    camera_shots: List[CameraShot] = field(default_factory=lambda: [
        # --- Right Three-Quarter Views ---
        # (From Shot_01/sample_00) - A perfect classic view. Elevation: 43.5°
        CameraShot(pos=(0.87, -0.50, 1.02), target=(0.52, -0.01, 0.45)),
        # (From Shot_04/sample_01) - A slightly wider right view. Elevation: 45.1°
        CameraShot(pos=(1.07, -0.25, 1.09), target=(0.49, -0.03, 0.44)),
        
        # --- Left Three-Quarter Views ---
        # (From Shot_02/sample_00) - Excellent left-side view. Elevation: 34°
        CameraShot(pos=(0.83, 0.49, 0.87), target=(0.48, -0.03, 0.43)),
        # (From Shot_06/sample_15) - A wide, cinematic left view. Elevation: 32°
        CameraShot(pos=(0.91, 0.49, 0.92), target=(0.38, 0.06, 0.45)),
        # (From Shot_04/sample_00 - adapted) - Another good left-side view for variety. Elevation: 53°
        CameraShot(pos=(0.57, 0.58, 1.01), target=(0.44, -0.01, 0.43)),

        # --- Frontal Views ---
        # (From Shot_09/sample_12) - A centered, slightly higher frontal view. Elevation: 41°
        CameraShot(pos=(1.14, -0.01, 0.99), target=(0.52, 0.03, 0.43)),
        # (From Shot_08/sample_00) - A well-balanced frontal shot. Elevation: 30°
        CameraShot(pos=(1.35, 0.34, 1.00), target=(0.44, -0.01, 0.45)),
        # (From Shot_02/sample_13 - adapted) - A slightly different frontal composition. Elevation: 38.4°
        CameraShot(pos=(1.02, 0.10, 0.95), target=(0.39, 0.04, 0.45)),
        
        # --- High-Angle / Near Top-Down Views ---
        # (From Shot_03/sample_11) - A high three-quarter view, very informative. Elevation: 44°
        CameraShot(pos=(1.04, -0.04, 1.04), target=(0.43, 0.01, 0.43)),
        # (From Shot_07/sample_00 - adapted) - A balanced top-down view, NOT the extreme 74°. Elevation: 65°
        CameraShot(pos=(0.72, 0.00, 1.25), target=(0.45, 0.02, 0.43)),

        # --- Dynamic / Lower Views (Still Safe) ---
        # (From Shot_01/sample_09) - A lower, more dynamic angle that still works well. Elevation: 27.8°
        CameraShot(pos=(1.05, -0.37, 0.82), target=(0.44, -0.00, 0.45)),
        # (From Shot_06/sample_01) - Another strong, slightly lower left view. Elevation: 35°
        CameraShot(pos=(1.08, 0.36, 0.97), target=(0.47, 0.06, 0.44)),

        # --- REJECTED SHOTS FOR REFERENCE ---
        # (From Shot_05/sample_13) - REJECTED: Elevation 18° is too low, sees under the table.
        # CameraShot(pos=(1.16, -0.27, 0.72), target=(0.48, -0.07, 0.45)),
        # (From Shot_07/sample_00) - REJECTED: Elevation 74° is too high, causes occlusion.
        # CameraShot(pos=(0.72, 0.00, 1.42), target=(0.45, 0.02, 0.43)),
    ])

    # --- TIGHTLY-TUNED JITTER PARAMETERS ---
    # These values add sufficient variety without compromising shot quality.
    radius_jitter: float = 0.10      # meters (previously 0.15)
    azimuth_jitter: float = 0.26     # radians (~15 degrees) (previously 0.35)
    elevation_jitter: float = 0.17   # radians (~10 degrees) (previously 0.26)
    target_pos_jitter: float = 0.05  # meters (previously 0.08)
    fovy_jitter: float = 3.0         # degrees (previously 5.0)


# FILE: envs/panda_env.py (Replace this method)

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
    # SAFETY: Ensure the camera isn't looking at the floor.
    final_target_pos[2] = max(final_target_pos[2], self.GOAL_Z_HEIGHT)

    # 4. Convert the exemplar's camera position to spherical coordinates relative to its target.
    radius, azimuth, elevation = self._cartesian_to_spherical(base_cam_pos, base_target_pos)

    # 5. Apply bounded, random jitter in the more intuitive spherical coordinate space.
    radius += self.np_random.uniform(-self.dr_config.radius_jitter, self.dr_config.radius_jitter)
    azimuth += self.np_random.uniform(-self.dr_config.azimuth_jitter, self.dr_config.azimuth_jitter)
    elevation += self.np_random.uniform(-self.dr_config.elevation_jitter, self.dr_config.elevation_jitter)

    # 6. CRITICAL SAFETY CLAMPS: Enforce the "Good Zone" limits we discovered.
    # This single-handedly prevents the vast majority of bad shots.
    elevation = np.clip(elevation, np.deg2rad(25), np.deg2rad(70)) # Clamp between 25° and 70°
    radius = np.clip(radius, 0.8, 2.0) # Prevent camera from getting too close or far

    # 7. Reconstruct the new Cartesian camera position using the jittered spherical coords
    #    and the NEW dynamic target position.
    final_cam_pos = self._spherical_to_cartesian(radius, azimuth, elevation, final_target_pos)

    # 8. Calculate the final camera orientation and FOV.
    new_quat_xyzw = self._calculate_look_at_quat(final_cam_pos, final_target_pos)
    base_fovy = self.model.cam_fovy[self.camera_id]
    final_fovy = base_fovy + self.np_random.uniform(-self.dr_config.fovy_jitter,
                                                    self.dr_config.fovy_jitter)
    final_fovy = np.clip(final_fovy, 35.0, 80.0) # Clamp FOV for good measure

    # --- Part 3: Apply Final Camera Pose to the MuJoCo Model ---
    self.model.cam_pos[self.camera_id] = final_cam_pos
    # MuJoCo uses w,x,y,z format for quaternions
    self.model.cam_quat[self.camera_id] = [new_quat_xyzw[3], new_quat_xyzw[0], new_quat_xyzw[1], new_quat_xyzw[2]]
    self.model.cam_fovy[self.camera_id] = final_fovy