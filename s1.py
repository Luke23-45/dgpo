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

# =================================================================================
# --- VISUALIZATION FUNCTIONS (ORIGINAL AND NEW) ---
# =================================================================================

def create_contact_sheet(image_files, output_path, grid_size=(4, 4)):
    """Creates a grid of images for a single shot and saves it."""
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

# --- NEW FUNCTION TO STACK TWO SHOTS ---
def create_comparison_sheet(image_files1, shot_name1, image_files2, shot_name2, output_path, grid_size=(4, 4)):
    """Creates a stacked contact sheet to compare two camera shots in one image."""
    images1 = [cv2.imread(f) for f in image_files1 if os.path.exists(f)]
    images2 = [cv2.imread(f) for f in image_files2 if os.path.exists(f)]

    if not images1 or not images2:
        print(f"Warning: Missing images for one or both shots for {output_path}. Skipping.")
        return
    
    # Ensure all images are resized to a common shape
    base_shape = images1[0].shape
    images1 = [cv2.resize(img, (base_shape[1], base_shape[0])) for img in images1]
    images2 = [cv2.resize(img, (base_shape[1], base_shape[0])) for img in images2]

    rows = grid_size[0] * 2  # Stack two grids vertically
    cols = grid_size[1]
    
    # Use a taller figure to accommodate the stacked layout
    fig, axes = plt.subplots(rows, cols, figsize=(12, 22)) 
    
    # Plot the first set of images in the top half
    for i, ax in enumerate(axes.flat[:len(images1)]):
        ax.imshow(cv2.cvtColor(images1[i], cv2.COLOR_BGR2RGB))
        ax.axis('off')

    # Plot the second set of images in the bottom half
    offset = grid_size[0] * grid_size[1]
    for i, ax in enumerate(axes.flat[offset:offset + len(images2)]):
        ax.imshow(cv2.cvtColor(images2[i], cv2.COLOR_BGR2RGB))
        ax.axis('off')

    # Turn off any unused axes
    for ax in axes.flat:
        if not ax.images:
            ax.axis('off')
        
    # Add titles for clarity
    fig.suptitle(f"Comparison: {shot_name1} vs. {shot_name2}", fontsize=20)
    fig.text(0.5, 0.92, shot_name1, ha='center', va='center', fontsize=16, weight='bold')
    fig.text(0.5, 0.48, shot_name2, ha='center', va='center', fontsize=16, weight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout for main title
    plt.subplots_adjust(hspace=0.1, wspace=0.1) # Fine-tune spacing
    
    plt.savefig(output_path)
    plt.close(fig)
    print(f"✔️ Comparison sheet saved to: {output_path}")


# ================================================================================
# --- DATA COLLECTION FUNCTIONS ---
# ================================================================================

def get_debug_info(env: PandaEnv) -> dict:
    """Collects and returns a dictionary of detailed camera and task information."""
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
    return debug_data

# --- NEW HELPER FUNCTION TO PROCESS A SINGLE SHOT ---
def process_shot(shot_config, shot_name, base_output_dir):
    """
    Generates images and debug data for a single CameraShot configuration.
    Returns a list of image file paths and a dictionary of the debug data.
    """
    temp_img_dir = os.path.join(base_output_dir, shot_name)
    os.makedirs(temp_img_dir, exist_ok=True)
    
    # Configure the environment for this specific shot
    temp_dr_config = DomainRandomizationConfig()
    temp_dr_config.camera_shots = [shot_config]
    
    env = PandaEnv(
        xml_path="envs/panda_pick_place.xml", 
        dr_config=temp_dr_config,
        enable_domain_randomization=True
    )

    image_files = []
    shot_records = {}
    for j in range(NUM_SAMPLES_PER_SHOT):
        print(f"\r  Generating sample {j+1}/{NUM_SAMPLES_PER_SHOT} for {shot_name}...", end="")
        env.reset()
        
        img = env.render(camera_name="fixed_camera")
        
        img_filename = f"sample_{j:02d}.png"
        img_filepath = os.path.join(temp_img_dir, img_filename)
        
        cv2.imwrite(img_filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        image_files.append(img_filepath)
        
        debug_info = get_debug_info(env)
        shot_records[img_filename] = debug_info
        
    print() # Add a newline after the progress indicator is done
    env.close()
    
    return image_files, shot_records


# ================================================================================
# --- MAIN EXECUTION LOGIC ---
# ================================================================================

def main():
    print("--- Starting Camera Shot Tuning and Visualization Test ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    shots_to_test = [
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
        # CameraShot(pos=(1.14, -0.01, 0.99), target=(0.52, 0.03, 0.43)),
        # (From Shot_02/sample_13 - adapted) - A slightly different frontal composition. Elevation: 38.4°
        CameraShot(pos=(1.02, 0.10, 0.95), target=(0.39, 0.04, 0.45)),
        # (From Shot_08/sample_00) - A well-balanced frontal shot. Elevation: 30°
        CameraShot(pos=(1.35, 0.34, 1.00), target=(0.44, -0.01, 0.45)),
        # (From Shot_09/sample_12) - A centered, slightly higher frontal view. Elevation: 41°
        CameraShot(pos=(1.14, -0.01, 0.99), target=(0.52, 0.03, 0.43)),

        # --- High-Angle / Near Top-Down Views ---
        # (From Shot_03/sample_11) - A high three-quarter view, very informative. Elevation: 44°
        CameraShot(pos=(1.04, -0.04, 1.04), target=(0.43, 0.01, 0.43)),
        # (From Shot_07/sample_00) - A balanced top-down view, not too extreme. Elevation: 74°
        CameraShot(pos=(0.72, 0.00, 1.42), target=(0.45, 0.02, 0.43)),

        # --- Dynamic / Lower Views (Still Safe) ---
        # (From Shot_01/sample_09) - A lower, more dynamic angle that still works well. Elevation: 27.8°
        CameraShot(pos=(1.05, -0.37, 0.82), target=(0.44, -0.00, 0.45)),
        # (From Shot_06/sample_01) - Another strong, slightly lower left view. Elevation: 35°
        CameraShot(pos=(1.08, 0.36, 0.97), target=(0.47, 0.06, 0.44)),
        # (From Shot_06/sample_15) - A wide, cinematic left view. Elevation: 32°
        CameraShot(pos=(0.91, 0.49, 0.92), target=(0.38, 0.06, 0.45)),
    ]
    
    print(f"Found {len(shots_to_test)} camera shots to test.")
    all_tuning_data = {}

    # --- REVISED LOOP: Process shots in pairs for comparison ---
    for i in range(0, len(shots_to_test), 2):
        shot1_config = shots_to_test[i]
        shot1_name = f"Shot_{i+1:02d}"
        
        print(f"\n--- Processing Pair starting with {shot1_name} ---")
        image_files1, shot1_records = process_shot(shot1_config, shot1_name, OUTPUT_DIR)
        all_tuning_data[shot1_name] = shot1_records
        
        # Check if a second shot exists in the pair
        if i + 1 < len(shots_to_test):
            shot2_config = shots_to_test[i+1]
            shot2_name = f"Shot_{i+2:02d}"

            image_files2, shot2_records = process_shot(shot2_config, shot2_name, OUTPUT_DIR)
            all_tuning_data[shot2_name] = shot2_records
            
            # Create a stacked comparison sheet for the pair
            comparison_path = os.path.join(OUTPUT_DIR, f"{shot1_name}_vs_{shot2_name}_comparison.png")
            create_comparison_sheet(image_files1, shot1_name, image_files2, shot2_name, comparison_path)
        else:
            # If it's the last shot and it's unpaired, create a normal contact sheet
            print(f"  {shot1_name} is the last shot, creating a single contact sheet.")
            contact_sheet_path = os.path.join(OUTPUT_DIR, f"{shot1_name}_contact_sheet.png")
            create_contact_sheet(image_files1, contact_sheet_path)

    # --- Save the single master JSON file at the very end ---
    print("\n--- Aggregating all data into a single file ---")
    final_json_path = os.path.join(OUTPUT_DIR, "camera_tuning_results.json")
    with open(final_json_path, 'w') as f:
        json.dump(all_tuning_data, f, indent=4)
    print(f"✔️ All tuning data saved to: {final_json_path}")


if __name__ == "__main__":
    main()