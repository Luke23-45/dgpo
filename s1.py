# tests/test_camera_shots.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from envs.panda_env import PandaEnv, DomainRandomizationConfig, CameraShot
import warnings

# --- Configuration ---
NUM_SAMPLES_PER_SHOT = 16  # We will generate a 4x4 grid of images for each shot
OUTPUT_DIR = "debug_output/camera_shot_tests" # A new directory for the new tests

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

def main():
    print("--- Starting Camera Shot Visualization Test ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Get the "master list" of shots from the default environment config
    master_config = DomainRandomizationConfig()
    shots_to_test = master_config.camera_shots
    print(f"Found {len(shots_to_test)} predefined camera shots to visualize.")

    for i, shot in enumerate(shots_to_test):
        shot_name = f"Shot_{i+1:02d}"
        print(f"\n--- Testing {shot_name}: pos={shot.pos}, target={shot.target} ---")
        
        temp_img_dir = os.path.join(OUTPUT_DIR, shot_name)
        os.makedirs(temp_img_dir, exist_ok=True)
        
        # 2. Create a temporary config that ONLY uses this one shot
        temp_dr_config = DomainRandomizationConfig()
        temp_dr_config.camera_shots = [shot] # Isolate the single shot we want to test
        
        env = PandaEnv(
            xml_path="envs/panda_pick_place.xml", 
            dr_config=temp_dr_config,
            enable_domain_randomization=True
        )

        image_files = []
        for j in range(NUM_SAMPLES_PER_SHOT):
            print(f"  Generating sample {j+1}/{NUM_SAMPLES_PER_SHOT}...")
            env.reset()
            img = env.render(camera_name="fixed_camera")
            filepath = os.path.join(temp_img_dir, f"sample_{j:02d}.png")
            cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            image_files.append(filepath)
            
        env.close()

        # 3. Create a contact sheet specifically for this shot
        contact_sheet_path = os.path.join(OUTPUT_DIR, f"{shot_name}_contact_sheet.png")
        create_contact_sheet(image_files, contact_sheet_path)

if __name__ == "__main__":
    main()