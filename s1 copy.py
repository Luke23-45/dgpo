# FILE: envs/panda_env.py (Replace this entire dataclass)

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
        # (From Shot_01/sample_00) - A perfect classic view. Elevation: 38.9°
        CameraShot(pos=(0.87, -0.36, 0.94), target=(0.41, 0.05, 0.43)),
        # (From Shot_04/sample_01) - A slightly wider right view. Elevation: 45.1°
        CameraShot(pos=(1.07, -0.25, 1.09), target=(0.49, -0.03, 0.44)),

        # --- Left Three-Quarter Views ---
        # (From Shot_02/sample_12) - Excellent left-side view. Elevation: 44.2°
        CameraShot(pos=(0.57, 0.58, 1.01), target=(0.44, -0.01, 0.43)),
        # (From Shot_03/sample_15) - A high-angle left view. Elevation: 44.6°
        CameraShot(pos=(0.78, 0.47, 1.00), target=(0.45, 0.02, 0.44)),

        # --- Frontal Views ---
        # (From Shot_08/sample_15) - A well-balanced frontal shot. Elevation: 41.6°
        CameraShot(pos=(0.86, -0.48, 1.03), target=(0.48, 0.05, 0.45)),
        # (From Shot_09/sample_12) - A centered, slightly higher frontal view. Elevation: 38.4°
        CameraShot(pos=(1.02, 0.10, 0.95), target=(0.39, 0.04, 0.45)),

        # --- High-Angle / Near Top-Down Views ---
        # (From Shot_07/sample_15) - A balanced top-down view, not too extreme. Elevation: 53.5°
        CameraShot(pos=(0.70, 0.00, 1.19), target=(0.50, -0.03, 0.44)),
        # (From Shot_06/sample_11) - A high three-quarter view, very informative. Elevation: 40.3°
        CameraShot(pos=(1.00, -0.06, 0.93), target=(0.41, -0.06, 0.43)),

        # --- Dynamic / Lower Views (Still Safe) ---
        # (From Shot_01/sample_09) - A lower, more dynamic angle that still works well. Elevation: 27.8°
        CameraShot(pos=(1.05, -0.37, 0.82), target=(0.44, -0.00, 0.45)),
        # (From Shot_02/sample_00) - A wide, cinematic left view. Elevation: 51.0°
        CameraShot(pos=(0.59, -0.49, 1.05), target=(0.50, -0.00, 0.43)),
    ])

    # ============================ REFINED JITTER PARAMETERS ============================
    # FINAL PATCH: Tighter bounds to keep variations closer to our golden samples.
    radius_jitter: float = 0.10      # meters (reduced from 0.15)
    azimuth_jitter: float = 0.26     # radians (~15 degrees) (reduced from 0.35)
    elevation_jitter: float = 0.17   # radians (~10 degrees) (reduced from 0.26)
    target_pos_jitter: float = 0.05  # meters (reduced from 0.08)
    fovy_jitter: float = 3.0         # degrees (reduced from 5.0)