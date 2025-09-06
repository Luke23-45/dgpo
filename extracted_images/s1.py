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
        # (From Shot_04/sample_01) - A slightly different angle, good composition. Elevation: 53°
        CameraShot(pos=(0.63, -0.44, 1.07), target=(0.49, -0.00, 0.45)),

        # --- Left Three-Quarter Views ---
        # (From Shot_02/sample_00) - Excellent left-side view. Elevation: 34°
        CameraShot(pos=(0.83, 0.49, 0.87), target=(0.48, -0.03, 0.43)),
        # (From Shot_06/sample_01) - A wider left-side view. Elevation: 35°
        CameraShot(pos=(1.08, 0.36, 0.97), target=(0.47, 0.06, 0.44)),

        # --- Frontal Views ---
        # (From Shot_08/sample_00) - A well-balanced frontal shot. Elevation: 30°
        CameraShot(pos=(1.35, 0.34, 1.00), target=(0.44, -0.01, 0.45)),
        # (From Shot_09/sample_12) - A centered, slightly higher frontal view. Elevation: 41°
        CameraShot(pos=(1.14, -0.01, 0.99), target=(0.52, 0.03, 0.43)),

        # --- High-Angle / Top-Down Views ---
        # (From Shot_03/sample_11) - A high three-quarter view, very informative. Elevation: 44°
        CameraShot(pos=(1.04, -0.04, 1.04), target=(0.43, 0.01, 0.43)),
        # (From Shot_07/sample_00) - A balanced top-down view, not too extreme. Elevation: 74°
        CameraShot(pos=(0.72, 0.00, 1.42), target=(0.45, 0.02, 0.43)),

        # --- Creative / Dynamic Views ---
        # (From Shot_05/sample_13) - A lower, more dynamic angle that still works well. Elevation: 18°
        CameraShot(pos=(1.16, -0.27, 0.72), target=(0.48, -0.07, 0.45)),
        # (From Shot_06/sample_15) - A wide, cinematic left view. Elevation: 32°
        CameraShot(pos=(0.91, 0.49, 0.92), target=(0.38, 0.06, 0.45)),
    ])

    # ============================ REFINED JITTER PARAMETERS ============================
    # FINAL PATCH: Tighter bounds to keep variations closer to our golden samples.
    radius_jitter: float = 0.10      # meters (reduced from 0.15)
    azimuth_jitter: float = 0.26     # radians (~15 degrees) (reduced from 0.35)
    elevation_jitter: float = 0.17   # radians (~10 degrees) (reduced from 0.26)
    target_pos_jitter: float = 0.05  # meters (reduced from 0.08)
    fovy_jitter: float = 3.0         # degrees (reduced from 5.0)

    