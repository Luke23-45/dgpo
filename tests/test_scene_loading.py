# s1_robust.py
import unittest
import os
import numpy as np
import mujoco

class TestSceneLoadingRobust(unittest.TestCase):
    """
    Robustly tests the main MuJoCo XML scene file.

    This test suite acts as a "contract" for the simulation scene, ensuring
    critical elements are present and correctly configured.

    Key improvements over the basic version:
    - **Data-Driven**: Scene expectations are defined in a single, easy-to-read
      dictionary, separating configuration from test logic.
    - **Extensible**: Adding new objects to test is a one-line change in the
      configuration dictionary.
    - **Detailed Failure Reports**: Uses `subTest` to run checks for every
      object and report all failures at once, instead of stopping at the first error.
    - **Comprehensive Checks**: Verifies not only positions but also orientations (quaternions)
      and the existence of other critical elements like cameras and joints.
    """

    # --- Test Configuration (The "Scene Contract") ---
    # Define all expectations here. Makes the test easy to read and modify.
    XML_PATH = os.path.join("envs", "panda_pick_place.xml")

    # MuJoCo quaternions are in [w, x, y, z] format. [1, 0, 0, 0] is the identity (no rotation).
    EXPECTED_BODY_POSES = {
        "link0":  {"pos": [0, 0, 0]},
        "table":  {"pos": [0.6, 0, 0.2]},
        "object": {"pos": [0.6, 0.0, 0.41], "quat": [1, 0, 0, 0]},
    }

    EXPECTED_ELEMENTS = {
        "camera": ["fixed_camera"],
        "joint": ["object_joint"],
    }
    # ---------------------------------------------------

    @classmethod
    def setUpClass(cls):
        """Load the model once for the entire test class for efficiency."""
        cls.assertTrue(os.path.exists(cls.XML_PATH), f"XML file not found at: {cls.XML_PATH}")
        try:
            cls.model = mujoco.MjModel.from_xml_path(cls.XML_PATH)
        except Exception as e:
            cls.fail(f"Failed to load MuJoCo model from {cls.XML_PATH}. Error: {e}")

    def test_A_model_compiles(self):
        """Test that the model was loaded successfully."""
        self.assertIsNotNone(self.model)
        print("\n✅ [TestSceneLoading] MuJoCo XML loaded successfully.")

    def test_B_body_poses_are_correct(self):
        """Verify the position and orientation of all critical bodies."""
        for body_name, expected_pose in self.EXPECTED_BODY_POSES.items():
            # Using subTest allows the test to continue after a failure,
            # reporting all mismatched bodies at once.
            with self.subTest(body_name=body_name):
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                self.assertGreaterEqual(body_id, 0, f"Body '{body_name}' not found in the model.")

                # Check position if it's defined in our contract
                if "pos" in expected_pose:
                    actual_pos = self.model.body_pos[body_id]
                    expected_pos = expected_pose["pos"]
                    np.testing.assert_allclose(
                        actual_pos, expected_pos, atol=1e-6,
                        err_msg=f"Position mismatch for body '{body_name}'."
                    )

                # Check orientation if it's defined in our contract
                if "quat" in expected_pose:
                    actual_quat = self.model.body_quat[body_id]
                    expected_quat = expected_pose["quat"]
                    np.testing.assert_allclose(
                        actual_quat, expected_quat, atol=1e-6,
                        err_msg=f"Orientation (quaternion) mismatch for body '{body_name}'."
                    )
        print("✅ [TestSceneLoading] All specified body poses are correct.")

    def test_C_critical_elements_exist(self):
        """Verify that other essential named elements (cameras, joints) exist."""
        element_map = {
            "camera": (mujoco.mjtObj.mjOBJ_CAMERA, "Camera"),
            "joint": (mujoco.mjtObj.mjOBJ_JOINT, "Joint"),
            "actuator": (mujoco.mjtObj.mjOBJ_ACTUATOR, "Actuator"),
        }

        for elem_type, names in self.EXPECTED_ELEMENTS.items():
            with self.subTest(element_type=elem_type):
                self.assertIn(elem_type, element_map, f"Unknown element type '{elem_type}' specified in test.")
                obj_type, readable_name = element_map[elem_type]
                for name in names:
                    elem_id = mujoco.mj_name2id(self.model, obj_type, name)
                    self.assertGreaterEqual(elem_id, 0, f"{readable_name} '{name}' not found in the model.")
        print("✅ [TestSceneLoading] All specified critical elements (cameras, joints) exist.")


if __name__ == '__main__':
    unittest.main(verbosity=2)