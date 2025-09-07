# ... after CAM_HEIGHT_ABOVE_TARGET = 0.8 ...

    # --- NEW: Min/Max clamping values for robustness ---
    CAM_MIN_DISTANCE = 0.4   # Don't let the camera get too close
    CAM_MAX_DISTANCE = 2.0   # Don't let the camera get too far
    CAM_MIN_FOVY = 25.0      # Min zoom
    CAM_MAX_FOVY = 90.0      # Max zoom (wide-angle)

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
# ...