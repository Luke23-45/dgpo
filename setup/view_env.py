# run_joint_test.py (Final Version with Video Recording)
import mujoco
import numpy as np
import time
import argparse
import os
from datetime import datetime

# Import optional dependencies at the top level
try:
    import glfw
except ImportError:
    glfw = None

try:
    import cv2
except ImportError:
    cv2 = None

# --- Configuration ---
XML_PATH = "envs/panda_pick_place.xml"
OUTPUT_LOG_FILE = "joint_test_results.log"
VIDEO_FILENAME = "joint_test_simulation.mp4"
VIDEO_FPS = 60 # A smooth framerate for the output video

SIMULATION_STEPS_PER_ACTION = 500
TEST_DELTA_RAD = 0.4
SUCCESS_THRESHOLD_RAD = 0.05
SUCCESS_THRESHOLD_METER = 0.001 

def init_simulation(xml_path):
    """Loads the MuJoCo model and initializes the simulation data."""
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found at '{xml_path}'.")
    print(f"⏳ Loading model from: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    print("✅ Model loaded successfully.")
    return model, data

def run_simulation_steps(model, data, steps, window=None, renderer=None, video_writer=None):
    """
    Runs the simulation for a given number of steps, with optional rendering and video recording.
    """
    for _ in range(steps):
        if window and glfw.window_should_close(window):
            return False 

        mujoco.mj_step(model, data)

        if renderer:
            renderer.update_scene(data, camera="fixed_camera")
            if window:
                glfw.poll_events()
                glfw.swap_buffers(window)
            
            if video_writer:
                # Render the frame to a NumPy array
                frame = renderer.render()
                # OpenCV expects BGR format, MuJoCo provides RGB. We need to convert.
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
            
    return True

def test_actuator(model, data, actuator_index, actuator_name, log_file, window=None, renderer=None, video_writer=None):
    """
    Tests a single actuator. Passes the video_writer object to the simulation steps.
    """
    log_file.write(f"\n{'='*50}\n")
    log_file.write(f"🔬 Testing Actuator {actuator_index}: {actuator_name}\n")
    log_file.write(f"{'='*50}\n")
    print(f"\n🔬 Testing Actuator {actuator_index}: {actuator_name}")

    joint_id = model.actuator_trnid[actuator_index, 0]
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    log_file.write(f"   (Controls Joint: {joint_name})\n")
    is_gripper = "actuator8" in actuator_name

    mujoco.mj_resetData(model, data)
    if not run_simulation_steps(model, data, 10, window, renderer, video_writer): return False

    initial_qpos = data.qpos.copy()
    if is_gripper:
        initial_qpos[joint_id] = 0.0
    initial_joint_pos = initial_qpos[joint_id]

    test_targets = [200.0, 0.0] if is_gripper else [initial_joint_pos + TEST_DELTA_RAD, initial_joint_pos - TEST_DELTA_RAD]

    for i, target_pos in enumerate(test_targets):
        print(f"  ▶️  Action {i+1}: Commanding target position {target_pos:.2f}")

        mujoco.mj_resetData(model, data)
        data.qpos[:] = initial_qpos
        mujoco.mj_forward(model, data)

        start_pos = data.qpos[joint_id]
        log_file.write(f"\n  ▶️  Action: Set target to {target_pos:.4f}\n")
        log_file.write(f"      Initial Position (qpos[{joint_id}]): {start_pos:.4f}\n")

        control_signal = initial_qpos[:model.nu].copy()
        control_signal[actuator_index] = target_pos
        data.ctrl[:model.nu] = control_signal

        if not run_simulation_steps(model, data, SIMULATION_STEPS_PER_ACTION, window, renderer, video_writer):
            return False

        final_pos = data.qpos[joint_id]
        delta = final_pos - start_pos
        success = np.abs(delta) > (SUCCESS_THRESHOLD_METER if is_gripper else SUCCESS_THRESHOLD_RAD)

        log_file.write(f"      Final Position:   {final_pos:.4f}\n")
        log_file.write(f"      Position Delta:   {delta:.4f}\n")
        log_file.write(f"      Result:           {'✅ SUCCESS' if success else '❌ FAILED'}\n")
        print(f"      Result: {'✅ SUCCESS' if success else '❌ FAILED'} (Delta: {delta:.4f})")
    
    return True

def main(visualize, record):
    """Main function to run the entire test suite."""
    model, data = init_simulation(XML_PATH)

    renderer = None
    window = None
    video_writer = None
    frame_size = (1280, 720) # Define a standard size for window and video

    # A renderer is needed for both visualization and recording
    if visualize or record:
        renderer = mujoco.Renderer(model, height=frame_size[1], width=frame_size[0])

    if visualize:
        if not glfw: raise ImportError("GLFW is required for visualization.")
        if not glfw.init(): raise Exception("GLFW could not be initialized.")
        window = glfw.create_window(frame_size[0], frame_size[1], "Automated Joint Test", None, None)
        if not window: glfw.terminate(); raise Exception("GLFW window could not be created.")
        glfw.make_context_current(window)
        glfw.swap_interval(1)

    if record:
        if not cv2: raise ImportError("opencv-python is required for recording.")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files
        video_writer = cv2.VideoWriter(VIDEO_FILENAME, fourcc, VIDEO_FPS, frame_size)
        print(f"📹 Recording video to '{VIDEO_FILENAME}' at {VIDEO_FPS} FPS.")

    # Display initial scene if visualizing or recording
    if renderer:
        run_simulation_steps(model, data, steps=1, window=window, renderer=renderer, video_writer=video_writer)

    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]

    with open(OUTPUT_LOG_FILE, "w", encoding="utf-8") as log_file:
        log_file.write(f"MuJoCo Automated Actuator Control Test\n")
        log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Model: {XML_PATH}\n")

        for i in range(model.nu):
            should_continue = test_actuator(model, data, i, actuator_names[i], log_file, window, renderer, video_writer)
            if not should_continue:
                print("\n⚠️ User interrupted the test by closing the window.")
                break
            # Add a small pause between tests for better video pacing
            if renderer:
                if not run_simulation_steps(model, data, 30, window, renderer, video_writer): break

    print(f"\n🎉 Test complete. Results saved to '{OUTPUT_LOG_FILE}'")

    if video_writer:
        video_writer.release()
        print(f"✅ Video saved successfully to '{VIDEO_FILENAME}'")

    if window:
        print("Closing visualization in 5 seconds...")
        time.sleep(5)
        glfw.terminate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated test for Panda robot joints and gripper in MuJoCo.")
    parser.add_argument('--visualize', action='store_true', help="Show the MuJoCo simulation window during the test.")
    parser.add_argument('--record', action='store_true', help="Record the entire simulation to an MP4 video file.")
    args = parser.parse_args()
    main(args.visualize, args.record)