import numpy as np
import os
import traceback  # Import the traceback module
from utils.ik_solver import IKSolver

# --- Configuration ---
URDF_PATH = "urdf/panda.urdf"

def main():
    """
    Final, robust test script for the IKSolver class.
    Includes detailed error reporting.
    """
    print("--- Testing IKSolver (Verbose Error Reporting) ---")

    # --- 1. Check if the URDF file exists ---
    if not os.path.exists(URDF_PATH):
        print(f"❌ FATAL ERROR: Cannot find the URDF file at '{URDF_PATH}'")
        print("   Please ensure you have downloaded the file and placed it in the 'urdf' directory.")
        return

    # --- 2. Initialize the Solver ---
    try:
        solver = IKSolver(urdf_path=URDF_PATH)
    except Exception as e:
        print(f"\n--- ❌ Test Failed During IKSolver Initialization ---")
        print(f"The IKSolver class raised an exception. This is the root cause.")
        # --- CRITICAL ADDITION: Print the full error traceback ---
        traceback.print_exc()
        return

    # --- 3. Run a Test Case ---
    print("\n--- Running a test case ---")
    
    current_joints = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
    print(f"  Current Joint Angles: {np.round(current_joints, 2)}")
    
    target_pose = np.array([
        0.5, -0.2, 0.5,           # XYZ
        0.924, 0.383, 0, 0         # QXYZW
    ])
    target_pose[3:] /= np.linalg.norm(target_pose[3:])
    print(f"  Target Pose (XYZ, QXYZW): {np.round(target_pose, 2)}")

    # --- 4. Compute the Action ---
    try:
        computed_action = solver.compute_action(target_pose, current_joints)
        
        print("\n✅ IK computation successful!")
        print(f"  Computed Action (scaled to [-1, 1]): {np.round(computed_action, 2)}")
        print(f"  Action Shape: {computed_action.shape}")

        if computed_action.shape == (8,):
            print("\n--- ✅ IK Solver Test Passed ---")
        else:
            print(f"\n--- ❌ IK Solver Test Failed: Incorrect action shape! ---")

    except Exception as e:
        print(f"\n--- ❌ Test Failed During IK Computation ---")
        traceback.print_exc()

if __name__ == "__main__":
    main()