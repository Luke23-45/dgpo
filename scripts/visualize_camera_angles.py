# FILE: scripts/visualize_camera_angles.py
#python -m scripts.visualize_camera_angles
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

# --- Project Imports ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from envs.panda_env import PandaEnv

def main():
    # 1. Setup Environment
    print("Initializing Environment...")
    # enable_domain_randomization=False is strictly required here so we can 
    # manually force specific camera shots without the reset() overriding us.
    # (Assumes you applied the fix to _cache_dr_element_ids provided previously)
    env = PandaEnv(
        xml_path="envs/panda_pick_place.xml",
        render_mode="rgb_array",
        control_mode="delta"
    )
    
    # 2. Setup Output Directories
    output_root = ROOT / "camera_audit_output"
    
    # Optional: Clean up previous run
    if output_root.exists():
        try:
            shutil.rmtree(output_root)
        except Exception:
            pass
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Output files will be saved to: {output_root}")

    video_file = output_root / "summary_audit.mp4"
    
    # 3. Initialize Video Writer
    obs, _ = env.reset()
    dummy_frame = env.render()
    h, w, _ = dummy_frame.shape
    fps = 1
    frames_per_shot = fps * 2  # 2 seconds per angle
    
    writer = cv2.VideoWriter(
        str(video_file), 
        cv2.VideoWriter_fourcc(*'mp4v'), 
        fps, 
        (w, h)
    )
    
    shots = env.dr_config.camera_shots
    print(f"Found {len(shots)} camera angles to audit.")
    
    try:
        for i in range(len(shots)):
            print(f"Processing Shot Index {i:02d}...")
            
            # Create a folder for this specific shot ID
            # Naming format: shot_XX
            shot_folder = output_root / f"shot_{i:02d}"
            shot_folder.mkdir(exist_ok=True)
            
            # Reset physics to center the object for consistent viewing
            env.reset(seed=65) 
            
            # Force the camera to the specific index
            env.debug_force_camera_shot(i)
            
            # Render loop for the "wiggle" animation
            for f in range(frames_per_shot):
                # Wiggle the robot slightly so it's not static
                # This helps visualize depth and occlusion
                action = np.zeros(8)
                action[0] = np.sin(f * 0.1) * 0.2 
                env.step(action)
                
                frame = env.render()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # --- Overlay Info ---
                shot_info = shots[i]
                text_lines = [
                    f"ID: {i}",
                    f"Pos: [{shot_info.pos[0]:.2f}, {shot_info.pos[1]:.2f}, {shot_info.pos[2]:.2f}]",
                    f"Tgt: [{shot_info.target[0]:.2f}, {shot_info.target[1]:.2f}, {shot_info.target[2]:.2f}]"
                ]
                
                y0, dy = 30, 25
                for k, line in enumerate(text_lines):
                    y = y0 + k * dy
                    # Black border for readability
                    cv2.putText(frame_bgr, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                    # Yellow text
                    cv2.putText(frame_bgr, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                
                # 1. Write to Video
                writer.write(frame_bgr)
                
                # 2. Write individual frame to the shot folder
                frame_filename = shot_folder / f"frame_{f:03d}.png"
                cv2.imwrite(str(frame_filename), frame_bgr)
                
    finally:
        writer.release()
        env.close()
        print(f"\nDone!")
        print(f"1. Video saved to: {video_file}")
        print(f"2. Individual frames saved in folders at: {output_root}")

if __name__ == "__main__":
    main()