# s1_final_version_agnostic.py
import unittest
import os
import ikpy.chain
from ikpy.link import URDFLink, OriginLink

class TestUrdfForIKSolver(unittest.TestCase):
    """
    Tests the URDF file specifically for its use with our IKSolver,
    using a library-version-agnostic approach.
    """
    URDF_PATH = os.path.join("urdf", "panda_mujoco_kinematics.urdf")

    @classmethod
    def setUpClass(cls):
        """
        Loads the chain. We don't need the active_links_mask here, as we will
        manually filter the links in the tests themselves, just like the IKSolver does.
        """
        cls.assertTrue(os.path.exists(cls.URDF_PATH), f"URDF file not found at: {cls.URDF_PATH}")
        try:
            # We only need to load the full chain once.
            cls.chain = ikpy.chain.Chain.from_urdf_file(
                cls.URDF_PATH,
                base_elements=["link0"]
            )
        except Exception as e:
            cls.fail(f"Failed to load URDF chain. Error: {e}")

    def test_A_chain_was_loaded(self):
        """Verify that the chain object was created successfully."""
        self.assertIsNotNone(self.chain)
        print("\n✅ [TestUrdfForIKSolver] URDF chain loaded successfully.")

    def test_B_chain_structure_is_correct(self):
        """Verify the base link, first physical link, and end-effector link."""
        # Check for ikpy's virtual OriginLink at the start
        self.assertIsInstance(self.chain.links[0], OriginLink)
        self.assertEqual(self.chain.links[0].name, "Base link")

        # The *second* link should be our physical base link from the URDF.
        # This link's name comes from the JOINT that connects to it.
        self.assertEqual(self.chain.links[1].name, "joint1",
                         "The first physical link should correspond to 'joint1'.")

        # The last link in the chain should be the end-effector.
        self.assertEqual(self.chain.links[-1].name, "hand_joint")
        print("✅ [TestUrdfForIKSolver] Chain structure (Base, first joint, EE) is correct.")

    def test_C_active_joint_count_and_names_are_correct(self):
        """
        Manually count and verify the names of the active 'revolute' joints,
        mimicking the IKSolver's logic exactly.
        """
        # --- THIS IS THE CRITICAL FIX ---
        # Manually filter the links to find the active ones, just like IKSolver.
        active_links = [
            link for link in self.chain.links
            if isinstance(link, URDFLink) and getattr(link, "joint_type", "") == "revolute"
        ]
        # -------------------------------

        # 1. Check the count
        self.assertEqual(len(active_links), 7,
                         f"Expected 7 active revolute joints, but found {len(active_links)}.")
        print("✅ [TestUrdfForIKSolver] Found exactly 7 active (revolute) joints.")

        # 2. Check the names
        expected_joint_names = [
            "joint1", "joint2", "joint3", "joint4",
            "joint5", "joint6", "joint7"
        ]
        actual_joint_names = [link.name for link in active_links]
        self.assertListEqual(actual_joint_names, expected_joint_names,
                             "The names of the active joints are not correct or not in the expected order.")
        print("✅ [TestUrdfForIKSolver] Active joint names and order are correct.")


if __name__ == '__main__':
    unittest.main(verbosity=2)