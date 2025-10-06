import unittest
import numpy as np
import os
import sys
import ikpy
from scipy.spatial.transform import Rotation as R 

# Make sure the project root is in the path to import project files
sys.path.append('.') 
from utils.ik_solver import IKSolver

# --- Configuration ---
# This URDF file must exist at the specified path for the tests to run.
URDF_PATH = "urdf/panda_mujoco_kinematics.urdf" 
TOLERANCE_POS = 5e-4  # Position tolerance in meters for assertions
TOLERANCE_ANGLE = 1e-3 # Angle tolerance in radians for assertions

# --- ANSI Color Codes for Pretty Printing ---
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_BOLD = "\033[1m"
C_RESET = "\033[0m"

class TestIKSolver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Load the IKSolver once for all tests to save time.
        This method is run by unittest before any tests in the class.
        """
        print(f"\n{C_BOLD}--- Setting up IKSolver for testing ---{C_RESET}")
        if not os.path.exists(URDF_PATH):
            raise FileNotFoundError(
                f"{C_RED}CRITICAL: URDF file not found at '{URDF_PATH}'. "
                "Cannot run IKSolver tests.{C_RESET}"
            )
        try:
            cls.solver = IKSolver(urdf_path=URDF_PATH)
        except Exception as e:
            raise RuntimeError(f"{C_RED}Failed to initialize IKSolver: {e}{C_RESET}")
        print(f"{C_GREEN}IKSolver loaded successfully.{C_RESET}")

    # def test_01_initialization(self):
    #     """
    #     [Test 1] Verify that the solver initializes correctly.
    #     - Checks if the chain is loaded.
    #     - Checks if it found the correct number of active joints (7 for Panda).
    #     """
    #     print(f"\n{C_YELLOW}[TEST 1/6] Running: Initialization Test{C_RESET}")
    #     self.assertIsNotNone(self.solver.chain, "IK chain should not be None.")
    #     num_active_joints = len(self.solver._active_idx)
    #     print(f"  - Found {num_active_joints} active joints.")
    #     self.assertEqual(num_active_joints, 7, "Panda arm should have 7 active joints.")
    #     print(f"  {C_GREEN}SUCCESS: Correct number of active joints found.{C_RESET}")

    # def test_02_forward_kinematics_consistency(self):
    #     """
    #     [Test 2] Verify Forward Kinematics (FK) against a known configuration.
    #     - We set the robot to a known 'home' position.
    #     - We check if the calculated end-effector pose matches the expected pose.
    #     """
    #     print(f"\n{C_YELLOW}[TEST 2/6] Running: Forward Kinematics Consistency Test{C_RESET}")
    #     # A known "ready" pose for the Panda arm (in radians)
    #     home_joint_angles = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
        
    #     # The expected end-effector position for this pose in the robot's base frame
    #     # This value is typically found by running FK once and recording the result.
    #     expected_ee_pos = np.array([0.307, 0.0, 0.59])
        
    #     # Create the full joint vector including non-active links (all zeros)
    #     full_joint_vector = np.zeros(len(self.solver.chain.links))
    #     for i, active_idx in enumerate(self.solver._active_idx):
    #         full_joint_vector[active_idx] = home_joint_angles[i]
            
    #     # Calculate the FK pose
    #     fk_frame = self.solver.chain.forward_kinematics(full_joint_vector)
    #     calculated_ee_pos = fk_frame[:3, 3]
        
    #     print(f"  - Home joint angles: {np.round(home_joint_angles, 3)}")
    #     print(f"  - Expected EE pos:   {np.round(expected_ee_pos, 4)}")
    #     print(f"  - Calculated EE pos: {np.round(calculated_ee_pos, 4)}")
        
    #     # Assert that the calculated position is very close to the expected one
    #     np.testing.assert_allclose(
    #         calculated_ee_pos,
    #         expected_ee_pos,
    #         atol=TOLERANCE_POS,
    #         err_msg="Forward kinematics position does not match expected value."
    #     )
    #     print(f"  {C_GREEN}SUCCESS: Forward kinematics is consistent.{C_RESET}")

    def test_03_inverse_kinematics_solvability(self):
        """
        [Test 3] Verify Inverse Kinematics (IK) for a reachable target.
        - We define a reachable target pose.
        - We ask the IK solver to find the joint angles for it.
        - We then use FK on the result to see if we get the original target back.
        """
        print(f"\n{C_YELLOW}[TEST 3/6] Running: Inverse Kinematics Solvability Test{C_RESET}")
        # An arbitrary but reachable target pose [x, y, z, qx, qy, qz, qw]
        target_pose_7d = np.array([0.5, 0.2, 0.5, 0, 1, 0, 0]) # Downward pointing

        initial_guess_joints = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        print(f"  - Target EE pose: {np.round(target_pose_7d[:3], 3)}")
        
        # Use the internal IK method to get target joint angles
        solved_joints = self.solver._get_target_joint_angles(
            target_pose_7d, initial_guess_joints, solution_position_tolerance=0.01
        )
        
        self.assertIsNotNone(solved_joints, "IK solver should find a solution for a reachable target.")
        print(f"  - IK solver found a solution: {np.round(solved_joints, 3)}")
        
        # Verify by running FK on the solved joints
        full_joint_vector = np.zeros(len(self.solver.chain.links))
        for i, active_idx in enumerate(self.solver._active_idx):
            full_joint_vector[active_idx] = solved_joints[i]
        
        fk_frame = self.solver.chain.forward_kinematics(full_joint_vector)
        reconstructed_pos = fk_frame[:3, 3]
        
        print(f"  - Reconstructed EE pos (from FK): {np.round(reconstructed_pos, 4)}")
        
        # The reconstructed position should be very close to our original target
        np.testing.assert_allclose(
            reconstructed_pos,
            target_pose_7d[:3],
            atol=0.01, # Use the solver's own tolerance for this check
            err_msg="Reconstructed FK pose does not match the original IK target."
        )
        print(f"  {C_GREEN}SUCCESS: IK solver can find a valid solution.{C_RESET}")
        
    # def test_04_compute_action_moves_toward_target(self):
    #     """
    #     [Test 4] Verify that compute_action() generates a sensible action.
    #     - We place the arm far from the target.
    #     - The resulting action should be non-zero.
    #     - The action should be normalized correctly (between -1 and 1).
    #     """
    #     print(f"\n{C_YELLOW}[TEST 4/6] Running: Action Computation Test{C_RESET}")
    #     current_joints = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
    #     # A target pose that is definitely not at the origin
    #     target_pose_7d = np.array([0.5, 0.0, 0.5, 0, 1, 0, 0])
        
    #     action = self.solver.compute_action(target_pose_7d, current_joints)
        
    #     print(f"  - Current joints (all zero), Target: [0.5, 0, 0.5]")
    #     print(f"  - Computed action: {np.round(action, 3)}")
        
    #     self.assertEqual(action.shape, (7,), "Action should have shape (7,)")
    #     self.assertTrue(np.linalg.norm(action) > 0.1, "Action should be non-zero when far from target.")
    #     self.assertTrue(np.all(action >= -1.0) and np.all(action <= 1.0), "Action values must be within [-1, 1].")
    #     print(f"  {C_GREEN}SUCCESS: Computed action is valid and non-zero.{C_RESET}")
        
    # def test_05_graceful_failure_on_unreachable_target(self):
    #     """
    #     [Test 5] Verify graceful failure for an unreachable target.
    #     - We give a target that is physically impossible to reach (too far).
    #     - We expect the solver to fail to find a solution.
    #     - compute_action() should return a safe zero-vector, not crash.
    #     """
    #     print(f"\n{C_YELLOW}[TEST 5/6] Running: Unreachable Target Edge Case Test{C_RESET}")
    #     current_joints = np.zeros(7)
    #     # This target is 3 meters away, far beyond the robot's reach
    #     unreachable_target_pose = np.array([3.0, 0.0, 0.0, 0, 1, 0, 0])
        
    #     print(f"  - Giving solver an unreachable target at x=3.0m...")
        
    #     action = self.solver.compute_action(unreachable_target_pose, current_joints)
        
    #     print(f"  - Computed action: {action}")
        
    #     # We expect the IK to fail, resulting in a zero action
    #     np.testing.assert_allclose(
    #         action,
    #         np.zeros(7),
    #         atol=1e-9,
    #         err_msg="Solver should return a zero action for an unreachable target."
    #     )
    #     print(f"  {C_GREEN}SUCCESS: Solver failed gracefully and returned a zero action.{C_RESET}")

    # def test_06_zero_action_when_at_target(self):
    #     """
    #     [Test 6] Verify zero action when the arm is already at the target.
    #     - We use FK to find the exact pose for a set of joint angles.
    #     - We then ask compute_action() to solve for that *same* pose.
    #     - The resulting action should be a zero-vector, as no movement is needed.
    #     """
    #     print(f"\n{C_YELLOW}[TEST 6/6] Running: At Target (No Movement) Edge Case Test{C_RESET}")
    #     # An arbitrary but valid joint configuration
    #     current_joints = np.array([0.1, 0.2, 0.3, -0.5, 0.1, 0.8, 0.2])
        
    #     # 1. Use FK to find the end-effector pose for these exact joints
    #     full_joint_vector = np.zeros(len(self.solver.chain.links))
    #     for i, active_idx in enumerate(self.solver._active_idx):
    #         full_joint_vector[active_idx] = current_joints[i]
        
    #     fk_frame = self.solver.chain.forward_kinematics(full_joint_vector)
    #     target_pos = fk_frame[:3, 3]
    #     # target_rot = rot.matrix_to_quaternion(fk_frame[:3, :3]) 

    #     # target_quat_xyzw = np.array([target_rot[1], target_rot[2], target_rot[3], target_rot[0]])

    #     # target_quat_xyzw = np.array([target_rot[1], target_rot[2], target_rot[3], target_rot[0]])
    #     target_quat_xyzw = R.from_matrix(fk_frame[:3, :3]).as_quat()
    #     at_target_pose = np.concatenate([target_pos, target_quat_xyzw])

    #     print(f"  - Set current joints to an arbitrary configuration.")
    #     print(f"  - Used FK to find the exact EE pose for these joints.")
    #     print(f"  - Now asking IK to solve for the pose it's already at.")

    #     # 2. Now ask the solver to compute an action to reach the pose it is already at
    #     action = self.solver.compute_action(at_target_pose, current_joints, max_delta=0.1)

    #     print(f"  - Computed action: {np.round(action, 5)}")
        
    #     # The action should be effectively zero
    #     np.testing.assert_allclose(
    #         action,
    #         np.zeros(7),
    #         atol=1e-5, # Use a small tolerance for floating point inaccuracies
    #         err_msg="Solver should return a zero action when already at the target."
    #     )
    #     print(f"  {C_GREEN}SUCCESS: Solver correctly returned a zero action.{C_RESET}")


if __name__ == "__main__":
    # To run the tests from the command line: python -m tests.test_ik_solver
    unittest.main()