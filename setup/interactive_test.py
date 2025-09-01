import time
import glfw  # Import the glfw library
import numpy as np
import mujoco
from mujoco import viewer
from typing import Dict

# Import your custom environment class
# Make sure this path is correct for your project structure
from envs.panda_env import PandaEnv

# --- Keyboard Control State ---
# This dictionary will be modified by the key_callback function.
# We declare it globally so the callback and main loop can access it.
keys: Dict[str, bool] = {
    'W': False, 'S': False, 'A': False, 'D': False,  # EE Position: X, Y
    'Q': False, 'E': False,                         # EE Position: Z
    'J': False, 'L': False,                         # EE Rotation: Yaw
    'G': False,                                     # Gripper
    'R': False                                      # Reset
}

def key_callback(window, key: int, scancode: int, action: int, mods: int) -> None:
    """
    A callback function that updates the global `keys` dictionary.
    This function is registered with GLFW and is called every time a
    key is pressed or released.
    """
    global keys
    
    key_map = {
        glfw.KEY_W: 'W', glfw.KEY_S: 'S',
        glfw.KEY_A: 'A', glfw.KEY_D: 'D',
        glfw.KEY_Q: 'Q', glfw.KEY_E: 'E',
        glfw.KEY_J: 'J', glfw.KEY_L: 'L',
        glfw.KEY_G: 'G', glfw.KEY_R: 'R'
    }
    
    key_char = key_map.get(key)
    if key_char:
        if action == glfw.PRESS:
            keys[key_char] = True
        elif action == glfw.RELEASE:
            keys[key_char] = False

def main() -> None:
    """
    Final, robust interactive visualization script.
    
    This version uses the official MuJoCo viewer with the public GLFW
    API for stable and correct key handling. It also includes robust
    real-time pacing and non-conflicting key-to-action mapping.
    """
    print("--- Interactive Environment Verification Script (Robust Version) ---")

    print("\n⏳ Creating the environment...")
    try:
        # We test the base environment to ensure the XML and model are correct.
        env = PandaEnv(xml_path="envs/panda_pick_place.xml")
        print("✅ Environment created successfully.")
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return

    # Get the low-level MuJoCo model and data from our env
    model = env.model
    data = env.data

    print("\n⏳ Launching MuJoCo Viewer...")
    
    # Use launch_passive to create a viewer that we can control manually.
    viewer_handle = viewer.launch_passive(model, data)
    
    # Set a fixed camera view for consistency
    viewer_handle.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer_handle.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed_camera")
    
    # --- THIS IS THE CORRECT, ROBUST WAY TO SET A KEY CALLBACK ---
    # We use the underlying window from the viewer handle and the public GLFW API.
    glfw.set_key_callback(viewer_handle.window, key_callback)
    
    print("✅ Viewer launched successfully. A window should now be visible.")
    
    print("\n--- CONTROLS ---")
    print(" W / S: Move Forward / Backward (X-axis)")
    print(" A / D: Move Left / Right (Y-axis)")
    print(" Q / E: Move Up / Down (Z-axis)")
    print(" J / L: Rotate Gripper (Yaw)")
    print(" G:     Close Gripper (Hold)")
    print(" R:     Reset Environment")
    print(" Mouse: Rotate/zoom camera (MuJoCo default)")
    print("\nClose the window to exit.")
    
    obs, info = env.reset()
    
    # Determine the correct time per step for real-time simulation
    # A standard Gym env advances time by `env.dt` seconds per `step()`.
    # This is typically model.opt.timestep * frame_skip.
    time_per_step = env.dt if hasattr(env, 'dt') else model.opt.timestep

    try:
        # The main loop runs as long as the viewer window is open.
        while viewer_handle.is_running():
            step_start_time = time.time()
            
            # --- Convert Keyboard Input to a Delta Action ---
            action = np.zeros(env.action_space.shape)
            
            # Use if/elif to prevent opposing keys from canceling each other out.
            if keys['W']:   action[0] = 1.0   # Positive X delta
            elif keys['S']: action[0] = -1.0  # Negative X delta

            if keys['A']:   action[1] = 1.0   # Positive Y delta
            elif keys['D']: action[1] = -1.0  # Negative Y delta

            if keys['Q']:   action[2] = 1.0   # Positive Z delta
            elif keys['E']: action[2] = -1.0  # Negative Z delta
            
            if keys['J']:   action[6] = 1.0   # Positive Yaw delta (joint 7)
            elif keys['L']: action[6] = -1.0  # Negative Yaw delta (joint 7)

            # Gripper control: [-1, 1] maps to [open, closed].
            # When 'G' is held, command "close". Otherwise, command "open".
            # This is a common teleop choice.
            action[7] = 1.0 if keys['G'] else -1.0

            # Handle environment reset
            if keys['R']:
                print("Resetting environment...")
                obs, info = env.reset()
                keys['R'] = False # Consume the reset key press

            # Step the environment with the calculated action
            env.step(action)
            
            # Synchronize the viewer with the new state of the simulation
            viewer_handle.sync()
            
            # Maintain a real-time simulation speed
            time_until_next_step = time_per_step - (time.time() - step_start_time)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        
        print("\nViewer closed by user.")

    except Exception as e:
        print(f"\n❌ An error occurred during the simulation loop: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Ensure resources are cleanly released.
        viewer_handle.close()
        env.close()
        print("\n--- ✅ Verification Complete ---")

if __name__ == "__main__":
    main()