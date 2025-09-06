import numpy as np
import os
import unittest # Using Python's built-in unittest framework

# Import the components we want to test
from utils.ik_solver import IKSolver
from envs.panda_env import PandaEnv

# --- Configuration for Tests ---
URDF_PATH = "urdf/panda.urdf"
XML_PATH = "envs/panda_pick_place.xml"

class TestDGPOComponents(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up objects once for all tests in this class."""
        print("\n--- Setting up test suite ---")
        if not os.path.exists(URDF_PATH):
            raise FileNotFoundError("URDF file not found for tests.")
        if not os.path.exists(XML_PATH):
            raise FileNotFoundError("MuJoCo XML file not found for tests.")
            
        cls.ik_solver = IKSolver(urdf_path=URDF_PATH)
        cls.env = PandaEnv(xml_path=XML_PATH)

    def test_ik_solver_initialization(self):
        """Test 1: Does the IKSolver initialize correctly?"""
        print("\nRunning test_ik_solver_initialization...")
        self.assertIsNotNone(self.ik_solver)
        # Check that it found the 7 active arm joints
        self.assertEqual(len(self.ik_solver.joint_limits[0]), 7)
        print("✅ PASSED")

    def test_ik_solver_computation(self):
        """Test 2: Does compute_action return a valid, finite action?"""
        print("\nRunning test_ik_solver_computation...")
        current_joints = np.zeros(7)
        target_pose = np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]) # Simple forward pose
        
        action = self.ik_solver.compute_action(target_pose, current_joints)
        
        self.assertEqual(action.shape, (8,))
        self.assertTrue(np.all(np.isfinite(action)))
        print("✅ PASSED")
        
    def test_env_initialization(self):
        """Test 3: Does the PandaEnv initialize correctly?"""
        print("\nRunning test_env_initialization...")
        self.assertIsNotNone(self.env)
        # Check that the action space was sized correctly
        self.assertEqual(self.env.action_space.shape[0], self.env.model.nu)
        print("✅ PASSED")
        
    def test_env_reset_and_step(self):
        """Test 4: Can the environment reset and take a step?"""
        print("\nRunning test_env_reset_and_step...")
        obs, info = self.env.reset()
        self.assertIn("image_primary", obs)
        self.assertIn("proprio", obs)
        
        action = self.env.action_space.sample()
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.assertIn("image_primary", obs)
        self.assertIn("proprio", obs)
        print("✅ PASSED")

if __name__ == '__main__':
    unittest.main()