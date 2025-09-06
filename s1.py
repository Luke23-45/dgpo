# tests/test_camera_tuning.py

import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from envs.panda_env import PandaEnv, DomainRandomizationConfig, CameraShot
import warnings

# --- Configuration ---
NUM_SAMPLES_PER_SHOT = 16  # We will generate a 4x4 grid of images for each shot
OUTPUT_DIR = "debug_output/camera_tuning" # A new directory for the tuning tests

def create_contact_sheet(image_files, output_path, grid_size=(4, 4)):
    """Creates a grid of images and saves it."""
    images = [cv2.imread(f) for f in image_files if os.path.exists(f)]
    if not images:
        print(f"Warning: No images found for {output_path}")
        return

    base_shape = images[0].shape
    images = [cv2.resize(img, (base_shape[1], base_shape[0])) for img in images]
    
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        ax.axis('off')
    
    title = f"Visualization for: {os.path.basename(output_path).replace('_contact_sheet.png', '')}"
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close(fig)
    print(f"✔️ Contact sheet saved to: {output_path}")

# ==================== MODIFIED FUNCTION: NOW RETURNS DATA =====================
def get_debug_info(env: PandaEnv) -> dict:
    """
    Collects and returns a dictionary of detailed camera and task information.
    """
    cam_id = env.camera_id
    
    final_cam_pos = env.model.cam_pos[cam_id].copy()
    final_cam_quat_wxyz = env.model.cam_quat[cam_id].copy()
    final_fovy = env.model.cam_fovy[cam_id]

    gripper_pos = env.get_ee_pose()[:3]
    goal_pos = env.get_goal_pos_expert()
    task_center = (gripper_pos + goal_pos) / 2.0
    
    radius, azimuth_rad, elevation_rad = env._cartesian_to_spherical(final_cam_pos, task_center)
    
    debug_data = {
        "camera_config": {
            "final_pos_xyz": final_cam_pos.tolist(),
            "final_quat_wxyz": final_cam_quat_wxyz.tolist(),
            "final_fovy_degrees": float(final_fovy)
        },
        "task_geometry": {
            "gripper_pos_xyz": gripper_pos.tolist(),
            "goal_pos_xyz": goal_pos.tolist(),
            "task_center_xyz": task_center.tolist()
        },
        "relative_spherical_coords": {
            "radius": float(radius),
            "azimuth_radians": float(azimuth_rad),
            "azimuth_degrees": float(np.rad2deg(azimuth_rad)),
            "elevation_radians": float(elevation_rad),
            "elevation_degrees": float(np.rad2deg(elevation_rad))
        }
    }
    # Return the dictionary instead of saving it
    return debug_data
# ================================================================================
# FILE: envs/panda_env.py (Replace this entire dataclass)
"""



"""


def main():
    print("--- Starting Camera Shot Tuning and Visualization Test ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    shots_to_test = [
                # --- Right Three-Quarter Views ---
        # (From Shot_01/sample_00) - A perfect classic view. Elevation: 39°
        CameraShot(pos=(0.88, -0.44, 0.95), target=(0.42, -0.02, 0.44)),
        # (From Shot_04/sample_01) - A slightly wider right view. Elevation: 45.1°
        CameraShot(pos=(1.07, -0.25, 1.09), target=(0.49, -0.03, 0.44)),
        # (From Shot_04/sample_01) - A slightly different angle, good composition. Elevation: 53°
        CameraShot(pos=(0.63, -0.44, 1.07), target=(0.49, -0.00, 0.45)),


        # --- Left Three-Quarter Views ---
        # (From Shot_02/sample_00) - Excellent left-side view. Elevation: 34°
        CameraShot(pos=(0.83, 0.49, 0.87), target=(0.48, -0.03, 0.43)),
        # (From Shot_06/sample_01) - A wider left-side view. Elevation: 35°
        CameraShot(pos=(1.08, 0.36, 0.97), target=(0.47, 0.06, 0.44)),

        # --- Frontal Views ---
        # (From Shot_09/sample_12) - A centered, slightly higher frontal view. Elevation: 41°
        CameraShot(pos=(1.14, -0.01, 0.99), target=(0.52, 0.03, 0.43)),
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
        CameraShot(pos=(1.16, -0.27, 0.72), target=(0.48, -0.07, 0.45)),
        # (From Shot_06/sample_15) - A wide, cinematic left view. Elevation: 32°
        CameraShot(pos=(0.91, 0.49, 0.92), target=(0.38, 0.06, 0.45)),
    ]
    
    print(f"Found {len(shots_to_test)} camera shots to visualize and tune.")

    # --- NEW: Create a master dictionary to hold all data ---
    all_tuning_data = {}

    for i, shot in enumerate(shots_to_test):
        shot_name = f"Shot_{i+1:02d}"
        print(f"\n--- Testing {shot_name}: pos={shot.pos}, target={shot.target} ---")
        
        temp_img_dir = os.path.join(OUTPUT_DIR, shot_name)
        os.makedirs(temp_img_dir, exist_ok=True)
        
        temp_dr_config = DomainRandomizationConfig()
        temp_dr_config.camera_shots = [shot]
        
        env = PandaEnv(
            xml_path="envs/panda_pick_place.xml", 
            dr_config=temp_dr_config,
            enable_domain_randomization=True
        )

        image_files = []
        # --- NEW: Create a dictionary to hold data for this specific shot ---
        shot_records = {}
        for j in range(NUM_SAMPLES_PER_SHOT):
            print(f"  Generating sample {j+1}/{NUM_SAMPLES_PER_SHOT}...")
            env.reset()
            
            img = env.render(camera_name="fixed_camera")
            
            # Use a simple filename as the key
            base_filename = f"sample_{j:02d}"
            img_filename = f"{base_filename}.png"
            img_filepath = os.path.join(temp_img_dir, img_filename)
            
            cv2.imwrite(img_filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            image_files.append(img_filepath)
            
            # --- MODIFIED: Get data and store it in our dictionary ---
            debug_info = get_debug_info(env)
            shot_records[img_filename] = debug_info
            
        env.close()

        # Add this shot's records to the master dictionary
        all_tuning_data[shot_name] = shot_records

        # Create the contact sheet (this logic is unchanged)
        contact_sheet_path = os.path.join(OUTPUT_DIR, f"{shot_name}_contact_sheet.png")
        create_contact_sheet(image_files, contact_sheet_path)

    # --- NEW: Save the single master JSON file at the very end ---
    print("\n--- Aggregating all data into a single file ---")
    final_json_path = os.path.join(OUTPUT_DIR, "camera_tuning_results.json")
    with open(final_json_path, 'w') as f:
        json.dump(all_tuning_data, f, indent=4)
    print(f"✔️ All tuning data saved to: {final_json_path}")


if __name__ == "__main__":
    main()