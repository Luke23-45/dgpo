# test_ik_solver_thorough.py
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
import io
# The class we are testing
from utils.ik_solver import IKSolver
import contextlib

# --- Configuration ---
MUJOCO_XML_PATH = "envs/panda_pick_place.xml"
IK_URDF_PATH = "urdf/panda_mujoco_kinematics.urdf"
EE_BODY_NAME = "hand"
BASE_BODY_NAME = "link0"

# --- Helper Functions ---
def get_body_pose_in_base_frame(model, data, body_name, base_name):
    """Gets the pose of a body relative to a base body's frame."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_name)
    
    if body_id < 0 or base_id < 0:
        raise ValueError(f"Body '{body_name}' or '{base_name}' not found.")

    # Get world poses
    p_world_body = data.xpos[body_id]
    R_world_body = data.xmat[body_id].reshape(3, 3)
    p_world_base = data.xpos[base_id]
    R_world_base = data.xmat[base_id].reshape(3, 3)

    # Calculate relative pose
    R_base_world = R_world_base.T
    p_base_body = R_base_world @ (p_world_body - p_world_base)
    R_base_body = R_base_world @ R_world_body
    
    return p_base_body, R_base_body


def run_tests():
    """Execute a series of validation tests on the IKSolver."""
    print("--- 🚀 Starting Thorough IKSolver Validation ---")

    # --- Setup ---
    try:
        model = mujoco.MjModel.from_xml_path(MUJOCO_XML_PATH)
        data = mujoco.MjData(model)
        solver = IKSolver(urdf_path=IK_URDF_PATH)
    except Exception as e:
        print(f"❌ FAILED during setup: {e}")
        return

    # === Test 1: Initialization Sanity Check ===
    print("\n--- [Test 1/4] Initialization Sanity Check ---")
    try:
        assert len(solver._active_idx) == 7, "Should have found 7 active joints."
        assert solver.chain.links[-1].name == "hand_joint", "Chain should end at 'hand_joint'."
        print("✅ Correct number of active joints (7) and correct end-effector found.")
    except AssertionError as e:
        print(f"❌ FAILED: {e}")
        return

    # === Test 2: Forward Kinematics (FK) Sync Check ===
    print("\n--- [Test 2/4] Forward Kinematics Sync Check ---")
    print("\n--- [Test 2/4] Forward Kinematics Sync Check ---")
    try:
        q_test = np.array([0.1, -0.2, 0.3, -1.5, 0.1, 1.5, 0.2])
        print(f"Test joint angles (radians): {q_test}")

        data.qpos[:7] = q_test
        mujoco.mj_forward(model, data)

        # Get ground truth pose from MuJoCo
        p_mj, R_mj = get_body_pose_in_base_frame(model, data, EE_BODY_NAME, BASE_BODY_NAME)
        print(f"MuJoCo EE position (base frame): {p_mj}")
        print(f"MuJoCo EE rotation matrix (base frame):\n{R_mj}")

        # Prepare joint angles for IKPy FK
        q_full = [0.0] * len(solver.chain.links)
        for k, idx in enumerate(solver._active_idx):
            q_full[idx] = q_test[k]
        print(f"Full joint vector for IKPy (with zeros for inactive joints): {q_full}")

        # Compute FK with IKPy
        T_ik = solver.chain.forward_kinematics(q_full)
        p_ik, R_ik = T_ik[:3, 3], T_ik[:3, :3]
        print(f"IKPy EE position: {p_ik}")
        print(f"IKPy EE rotation matrix:\n{R_ik}")

        # Compute errors
        pos_error = np.linalg.norm(p_mj - p_ik)
        
        R_diff = R_mj.T @ R_ik
        trace = np.trace(R_diff)
        trace = np.clip(trace, -1.0, 3.0)  # safeguard trace range
        rot_error_rad = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        rot_error_deg = np.degrees(rot_error_rad)

        print(f"Position error: {pos_error:.6f} m")
        print(f"Rotation error: {rot_error_deg:.6f} degrees")

        # Print difference vector
        diff_vec = p_mj - p_ik
        print(f"Position difference vector (MuJoCo - IKPy): {diff_vec}")

        # Assert tolerances
        assert pos_error < 1e-5, f"Position error is too high: {pos_error:.3e} m"
        assert rot_error_rad < 1e-4, f"Rotation error is too high: {rot_error_deg:.3e} deg"

        print(f"✅ FK sync passed. Position error: {pos_error:.3e} m, Rotation error: {rot_error_deg:.3e} deg.")

    except Exception as e:
        print(f"❌ FAILED: {e}")
        # Optional: raise to see full traceback or debug in IDE
        raise


    # === Test 3: IK Round-Trip Accuracy ===
    print("\n--- [Test 3/4] IK Round-Trip Accuracy (FK -> IK -> FK) ---")
    N_SAMPLES = 10
    total_pos_error = 0.0
    total_rot_error = 0.0
    try:
        for i in range(N_SAMPLES):
            # 1. Start with a random, known joint configuration
            q_known = np.random.uniform(
                low=[lim[0] for lim in solver._joint_limits],
                high=[lim[1] for lim in solver._joint_limits]
            )
            
            # 2. FK Step 1: Find the ground truth pose for this configuration using MuJoCo
            data.qpos[:7] = q_known
            mujoco.mj_forward(model, data)
            p_target, R_target = get_body_pose_in_base_frame(model, data, EE_BODY_NAME, BASE_BODY_NAME)
            
            # 3. IK Step: Solve for joint angles that produce the target pose
            initial_guess_full = [0.0] * len(solver.chain.links)
            q_initial_guess = q_known + np.random.uniform(-0.1, 0.1, size=7)
            for k, idx in enumerate(solver._active_idx):
                initial_guess_full[idx] = q_initial_guess[k]
            
            q_solved_full = solver.chain.inverse_kinematics(
                target_position=p_target,
                target_orientation=R_target,
                orientation_mode="all",
                initial_position=initial_guess_full
            )
            q_solved = np.array([q_solved_full[idx] for idx in solver._active_idx])
            
            # 4. FK Step 2: Calculate the pose resulting from the IK solution
            q_full_solved = [0.0] * len(solver.chain.links)
            for k, idx in enumerate(solver._active_idx):
                q_full_solved[idx] = q_solved[k]
            T_ik_final = solver.chain.forward_kinematics(q_full_solved)
            p_final, R_final = T_ik_final[:3, 3], T_ik_final[:3, :3]
            
            # 5. Verification: The final pose must match the target pose
            pos_error = np.linalg.norm(p_target - p_final)
            rot_error_rad = np.arccos(np.clip((np.trace(R_target.T @ R_final) - 1) / 2, -1, 1))
            
            total_pos_error += pos_error
            total_rot_error += rot_error_rad
            
            assert pos_error < 1e-5, f"IK result position error for sample {i} is too high: {pos_error:.3e} m"
            assert rot_error_rad < 1e-4, f"IK result rotation error for sample {i} is too high: {np.degrees(rot_error_rad):.3e} deg"

        avg_pos_error = total_pos_error / N_SAMPLES
        avg_rot_error_deg = np.degrees(total_rot_error / N_SAMPLES)
        print(f"✅ IK round-trip test passed for {N_SAMPLES} random poses.")
        print(f"   Average final pose error: Position={avg_pos_error:.3e} m, Rotation={avg_rot_error_deg:.3e} deg.")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return

    # === Test 4: compute_action() API and Edge Cases ===
    # === Test 4: compute_action() API and Edge Cases (Robust Version) ===
    print("\n--- [Test 4/4] compute_action() API and Edge Cases ---")

    try:
        # --- Sub-test 4.1: Guaranteed Reachable Target ---
        print("   [4.1] Testing a guaranteed reachable target...")
        
        # Create a known-good target pose using FK
        q_target_known = np.array([0.2, 0.2, -0.1, -1.0, 0.1, 1.2, 0.2])
        data.qpos[:7] = q_target_known
        mujoco.mj_forward(model, data)
        p_target, R_target = get_body_pose_in_base_frame(model, data, EE_BODY_NAME, BASE_BODY_NAME)
        quat_target = R.from_matrix(R_target).as_quat() # xyzw
        target_pose_7d = np.concatenate([p_target, quat_target])
        
        # Start from a different configuration (e.g., zero)
        current_q = np.zeros(7)
        action = solver.compute_action(target_pose_7d, current_q)
        
        # Assertions for a successful IK solve
        assert action.shape == (8,), f"Action shape was {action.shape}, expected (8,)"
        assert np.all(action >= -1.0) and np.all(action <= 1.0), "Action was not clipped to [-1, 1]"
        assert not np.all(action == 0.0), "IK failed for a reachable target! Returned the zero-vector fallback."
        print("   ✅ compute_action() SUCCEEDED for a reachable target and returned a valid, non-zero action.")

        # --- Sub-test 4.2: Unreachable Target with Failure Verification ---
        print("   [4.2] Testing an unreachable target...")
        unreachable_target = np.array([5.0, 0, 0, 1, 0, 0, 0]) # 5 meters away
        
        # MODIFICATION START: Capture both stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            action_fail = solver.compute_action(unreachable_target, current_q)
        
        # Combine both streams into a single output string for checking
        output = stdout_capture.getvalue() + stderr_capture.getvalue()
        # MODIFICATION END
        
        # Assertions for a graceful failure
        assert "IK failed (all attempts)" in output, "The expected failure message was not printed to console."

        assert np.all(action_fail == 0.0), "Expected a zero-vector action on confirmed IK failure."
        print("   ✅ compute_action() handled an unreachable target gracefully (printed error, returned zeros).")
        
        # --- Sub-test 4.3: max_delta Scaling ---
        print("   [4.3] Testing max_delta scaling behavior...")
        # Use the same reachable target from 4.1
        
        # With a tiny max_delta, the action should be clipped (saturated)
        action_clipped = solver.compute_action(target_pose_7d, current_q, max_delta=0.001)
        arm_action_clipped = action_clipped[:7]
        assert np.any(np.isclose(np.abs(arm_action_clipped), 1.0)), "With small max_delta, action should be saturated to +/-1.0"
        print("   ✅ Correctly saturated action with small max_delta.")

        # With a huge max_delta, the action should NOT be clipped
        action_unclipped = solver.compute_action(target_pose_7d, current_q, max_delta=100.0)
        arm_action_unclipped = action_unclipped[:7]
        assert np.all(np.abs(arm_action_unclipped) < 1.0), "With large max_delta, action should not be saturated."
        print("   ✅ Correctly produced a non-saturated action with large max_delta.")
        
        # --- Sub-test 4.4: Input Validation ---
        print("   [4.4] Testing input validation for malformed inputs...")
        try:
            solver.compute_action(np.zeros(6), current_q)
            raise AssertionError("Failed to raise ValueError for wrong target shape.")
        except ValueError:
            pass # Expected
        
        try:
            solver.compute_action(target_pose_7d, np.zeros(8))
            raise AssertionError("Failed to raise ValueError for wrong current_q shape.")
        except ValueError:
            pass # Expected
        print("   ✅ Correctly raised ValueError for malformed inputs.")

    except Exception as e:
        print(f"❌ FAILED: {e}")
        # Re-raise to get a full traceback for debugging
        raise
    print("\n\n--- 🎉 All IKSolver tests passed! The component is robust and accurate. ---")


if __name__ == "__main__":
    run_tests()