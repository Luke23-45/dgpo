import mujoco
import numpy as np
import time
import os
import glfw  # Make sure to 'pip install glfw'
from mujoco.viewer import launch_passive

# --- Main Test Script ---

def main():
    """
    This script loads the custom Panda environment, verifies its setup,
    and launches an interactive viewer for visualization.
    """
    print("--- DGPO Project Verification Script ---")

    # --- 1. Load the MuJoCo Model ---
    # Construct an absolute path to the XML file to ensure it's always found.
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Join it with the relative path to the XML file
        xml_path = os.path.join(script_dir, 'envs', 'panda_pick_place.xml')
        print(f"Loading model from: {xml_path}")

        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        print("✅ Model and data loaded successfully.")
    except Exception as e:
        print("\n--- ❌ ERROR ---")
        print(f"Failed to load the model. MuJoCo error: {e}")
        print("\n🚨 Troubleshooting:")
        print("1. Ensure you are running this script from the 'dgpo_project' root directory.")
        print("2. Verify the directory structure is correct: 'dgpo_project/envs/panda_pick_place.xml'.")
        print("3. Check that all model files exist in 'dgpo_project/mujoco_menagerie/franka_emika_panda/'.")
        return

    # --- 2. Create a Renderer for Observations ---
    try:
        renderer = mujoco.Renderer(model, height=240, width=320)
        print("✅ Renderer created successfully.")
    except Exception as e:
        print(f"\n--- ⚠️ WARNING ---")
        print(f"Failed to create renderer. Visual observations will not be available.")
        print(f"This can happen on a headless server. Error: {e}")
        renderer = None

    # --- 3. Define Core Environment Functions ---

    def get_observation():
        """Returns a camera image as a NumPy array."""
        if renderer:
            renderer.update_scene(data, camera="fixed_camera")
            return renderer.render()
        else:
            # Return a black image if the renderer is not available
            return np.zeros((240, 320, 3), dtype=np.uint8)

    def reset_environment():
        """Resets the simulation and randomizes object/goal positions."""
        mujoco.mj_resetData(model, data)

        # Randomize cube position on the table surface
        table_pos = model.body('table').pos
        table_size = model.geom('table_geom').size
        # More robust randomization to avoid edges
        cube_x = np.random.uniform(table_pos[0] - table_size[0]/2 + 0.1, table_pos[0] + table_size[0]/2 - 0.1)
        cube_y = np.random.uniform(table_pos[1] - table_size[1]/2 + 0.1, table_pos[1] + table_size[1]/2 - 0.1)
        
        data.joint('object_joint').qpos[:2] = [cube_x, cube_y]

        # Randomize goal position
        goal_x = np.random.uniform(table_pos[0] - table_size[0]/2 + 0.1, table_pos[0] + table_size[0]/2 - 0.1)
        goal_y = np.random.uniform(table_pos[1] - table_size[1]/2 + 0.1, table_pos[1] + table_size[1]/2 - 0.1)
        model.body('goal').pos[:2] = [goal_x, goal_y]
        
        # BEST PRACTICE: Use mj_forward to update the simulation state after changing positions.
        mujoco.mj_forward(model, data)
        
        print(f"🔄 Environment Reset. Cube at ({data.geom('object_geom').xpos[0]:.2f}, {data.geom('object_geom').xpos[1]:.2f}), "
              f"Goal at ({model.body('goal').pos[0]:.2f}, {model.body('goal').pos[1]:.2f})")
        return model.body('goal').pos.copy() # Return a copy

    def check_success(cube_position, goal_position, threshold=0.03):
        """Checks if the cube is within a certain distance of the goal in the XY plane."""
        distance = np.linalg.norm(cube_position[:2] - goal_position[:2])
        return distance < threshold

    # --- 4. Run Verification Steps ---
    
    print("\n--- Verifying Environment Functions ---")
    
    # Test reset
    goal_pos = reset_environment()
    cube_start_pos = data.geom('object_geom').xpos.copy()
    
    # Test observation
    obs = get_observation()
    print(f"📸 Observation received with shape: {obs.shape} and dtype: {obs.dtype}")

    # Test a simple action
    # Action space: 7 for arm joint velocities, 1 for gripper position
    action = np.zeros(model.nu) # model.nu is the number of actuators
    action[2] = -0.3  # Apply a downward velocity to joint 3
    action[7] = 0.00  # Command gripper to open
    
    data.ctrl[:] = action
    mujoco.mj_step(model, data, nstep=200) # Simulate for 200 steps
    
    cube_end_pos = data.geom('object_geom').xpos.copy()
    print(f"📦 Cube started at {cube_start_pos.round(3)}")
    print(f"📦 Cube ended at   {cube_end_pos.round(3)}")

    # Test success check
    is_success_false = check_success(cube_end_pos, goal_pos)
    print(f"🏆 Is task successful (should be False)? {is_success_false}")
    
    # Manually move cube to goal to test the True case
    data.joint('object_joint').qpos[:2] = goal_pos[:2]
    mujoco.mj_forward(model, data) # Update data after changing qpos
    is_success_true = check_success(data.geom('object_geom').xpos, goal_pos)
    print(f"🏆 Is task successful (should be True)? {is_success_true}")

    print("\n--- ✅ SETUP VERIFIED ---")

    # --- 5. Launch Interactive Viewer ---
    print("\nLaunching interactive viewer. Close the window to exit.")
    reset_environment() # Reset for a clean start in the viewer
    
    try:
        with launch_passive(model, data) as viewer:
            while viewer.is_running():
                step_start = time.time()
                
                # Example of continuous action: a slow circular motion
                t = time.time()
                action[0] = 0.1 * np.sin(t)
                action[1] = 0.1 * np.cos(t)
                action[7] = 0.04 # Close gripper
                data.ctrl[:] = action
                
                mujoco.mj_step(model, data)
                viewer.sync()
                
                # Maintain simulation real-time factor
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    except Exception as e:
        print(f"\n--- ❌ Viewer Error ---")
        print(f"Could not launch viewer. Error: {e}")
        print("Please ensure 'pip install glfw' was successful.")


if __name__ == "__main__":
    main()
