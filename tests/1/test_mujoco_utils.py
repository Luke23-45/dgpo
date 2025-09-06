# tests/test_mujoco_utils.py
import unittest
import numpy as np
import mujoco
import warnings

# The functions we are testing
from utils.mujoco_utils import set_joint_qpos_by_name, set_body_position

# A minimal, self-contained MuJoCo model for testing
MINIMAL_XML = """
<mujoco>
  <worldbody>
    <body name="object" pos="0 0 0">
      <joint name="object_joint" type="free"/>
      <geom type="sphere" size="0.1"/>
    </body>
    <body name="target" pos="1 1 1">
      <geom type="sphere" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""

class TestMujocoUtils(unittest.TestCase):
    """
    Tests the robust MuJoCo utility functions.

    This suite uses a minimal, self-contained XML model to verify:
    - Success, failure, and fallback paths for `set_joint_qpos_by_name`.
    - Success, failure, and fallback paths for `set_body_position`.
    - Correct handling of names, IDs, and data writing.
    """

    @classmethod
    def setUpClass(cls):
        """Load the minimal model from the XML string once for all tests."""
        try:
            cls.model = mujoco.MjModel.from_xml_string(MINIMAL_XML)
        except Exception as e:
            cls.fail(f"Failed to load minimal MuJoCo model from string. Error: {e}")

    def setUp(self):
        """Create a fresh MjData object for each test to ensure isolation."""
        self.data = mujoco.MjData(self.model)
        # Store original model positions to check for mutation
        self.original_target_pos = self.model.body_pos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")].copy()

    def test_set_joint_qpos_by_name(self):
        """Test all logical paths of the set_joint_qpos_by_name function."""
        with self.subTest(case="Success - Primary Path (by joint name)"):
            # A free joint has 7 qpos values: [x, y, z, qw, qx, qy, qz]
            target_qpos = [0.1, 0.2, 0.3, 0.707, 0, 0.707, 0]
            result = set_joint_qpos_by_name(self.model, self.data, "object_joint", target_qpos)
            self.assertTrue(result, "Function should return True on success.")
            
            # Verify the data was written correctly to data.qpos
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_joint")
            adr = self.model.jnt_qposadr[jid]
            actual_qpos = self.data.qpos[adr : adr + 7]
            np.testing.assert_allclose(actual_qpos, target_qpos)
            print("\n✅ [TestMujocoUtils] set_joint_qpos_by_name: Primary path works.")

        with self.subTest(case="Success - Fallback Path (heuristic name)"):
            # Pass "target_joint" which doesn't exist; it should find body "target"
            target_pos = [0.4, 0.5, 0.6]
            result = set_joint_qpos_by_name(self.model, self.data, "target_joint", target_pos)
            self.assertTrue(result, "Function should return True on fallback success.")

            # Verify the model itself was mutated
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
            actual_pos = self.model.body_pos[bid]
            np.testing.assert_allclose(actual_pos, target_pos)
            print("✅ [TestMujocoUtils] set_joint_qpos_by_name: Fallback path works.")

        with self.subTest(case="Failure - Name not found"):
            # Suppress the expected warning during this test
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = set_joint_qpos_by_name(self.model, self.data, "nonexistent", [0,0,0])
            self.assertFalse(result, "Function should return False for a nonexistent name.")
            print("✅ [TestMujocoUtils] set_joint_qpos_by_name: Failure case is handled correctly.")

    def test_set_body_position(self):
        """Test all logical paths of the set_body_position function."""
        with self.subTest(case="Success - Primary Path (body with joint)"):
            target_pos = [0.7, 0.8, 0.9]
            result = set_body_position(self.model, self.data, "object", target_pos)
            self.assertTrue(result, "Function should return True on success.")

            # Verify that data.qpos was written to
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_joint")
            adr = self.model.jnt_qposadr[jid]
            actual_pos = self.data.qpos[adr : adr + 3]
            np.testing.assert_allclose(actual_pos, target_pos)
            print("✅ [TestMujocoUtils] set_body_position: Primary path works.")
        
        with self.subTest(case="Success - Fallback Path (body without joint)"):
            target_pos = [-0.1, -0.2, -0.3]
            result = set_body_position(self.model, self.data, "target", target_pos)
            self.assertTrue(result, "Function should return True on fallback success.")

            # Verify that model.body_pos was mutated
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
            actual_pos = self.model.body_pos[bid]
            np.testing.assert_allclose(actual_pos, target_pos)
            print("✅ [TestMujocoUtils] set_body_position: Fallback path works.")

        with self.subTest(case="Failure - Name not found"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = set_body_position(self.model, self.data, "nonexistent", [0,0,0])
            self.assertFalse(result, "Function should return False for a nonexistent name.")
            print("✅ [TestMujocoUtils] set_body_position: Failure case is handled correctly.")


if __name__ == '__main__':
    unittest.main(verbosity=2)