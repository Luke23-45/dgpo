"""
Unit tests for the RLRewardWrapper.

This test suite verifies the reward logic of the RLRewardWrapper by isolating
the reward calculation from the environment's physics simulation. It manually
sets the simulation state (e.g., object and end-effector positions) to
trigger specific reward conditions and then calls the wrapper's internal
reward calculation method directly.

This approach ensures that the tests are deterministic and free from the
unpredictable effects of physics "jiggles" or contact resolution, providing a
robust and precise validation of the reward function.
"""

import unittest

import mujoco
import numpy as np

# Adjust these import paths to match your project's directory structure.
# For example, if your modules are in a 'src' folder, you might need to
# configure your PYTHONPATH or use relative imports.
from envs.panda_env import PandaEnv
from utils.rl_reward_wrapper import RLRewardWrapper


class TestRLRewardWrapper(unittest.TestCase):
    """A comprehensive and robust test suite for the RLRewardWrapper."""

    @classmethod
    def setUpClass(cls):
        """Create the base environment once for all tests."""
        cls.dummy_action = np.zeros(8)
        cls.gripping_action = np.array([0] * 7 + [1.0])

        try:
            cls.base_env = PandaEnv()
        except Exception as e:
            cls.fail(f"Failed to initialize base PandaEnv. Error: {e}")

    def setUp(self):
        """Create a new wrapper and reset the simulation for each test."""
        self.env = RLRewardWrapper(
            self.base_env,
            object_geom_name="object_geom",
            goal_body_name="goal"
        )
        self.model = self.base_env.model
        self.data = self.base_env.data
        self.env.reset()

    def _set_object_pos(self, pos: np.ndarray):
        """
        Robustly set the object's position by finding its free joint qpos address.
        This leaves the object's orientation (quaternion) untouched.
        """
        # Find the geom -> body -> joint -> qpos address
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.env.object_geom_name)
        if geom_id == -1:
            self.fail(f"Geom '{self.env.object_geom_name}' not found in the model.")

        body_id = int(self.model.geom_bodyid[geom_id])
        # Find the first joint of that body
        jnt_addr = int(self.model.body_jntadr[body_id])
        if jnt_addr < 0:
            self.fail(f"Body for geom '{self.env.object_geom_name}' has no joint.")

        # Ensure it's a 'free' joint, which is used for movable objects
        if int(self.model.jnt_type[jnt_addr]) != int(mujoco.mjtJoint.mjJNT_FREE):
            self.fail("Expected a 'free' joint for the object body to set its position.")

        # Get the starting address of this joint's data in the qpos array
        qpos_addr = int(self.model.jnt_qposadr[jnt_addr])
        # A free joint's qpos is [x, y, z, qw, qx, qy, qz]. We only set the x,y,z part.
        self.data.qpos[qpos_addr:qpos_addr+3] = pos

    def _run_reward_calculation(self, action: np.ndarray) -> tuple[float, bool, dict]:
        """
        Runs ONLY the reward calculation logic without stepping the physics.
        This is the key to reliable testing.
        """
        # Synchronize the simulation state after manual position changes.
        mujoco.mj_forward(self.model, self.data)
        # Call the wrapper's internal reward logic directly.
        reward, terminated, info = self.env._calculate_rewards_and_info(action, 0.0, False, {})
        return reward, terminated, info

    def _run_reward_calculation(self, action: np.ndarray) -> tuple[float, bool, dict]:
        """
        Runs ONLY the reward calculation logic without stepping the physics.

        This is the key to reliable testing:
        1. It calls mj_forward() to synchronize the simulation state after
           manual position changes.
        2. It calls the wrapper's internal reward calculation method directly.
        """
        mujoco.mj_forward(self.model, self.data)
        # Call the isolated reward logic with dummy base_reward and terminated status.
        reward, terminated, info = self.env._calculate_rewards_and_info(action, 0.0, False, {})
        return reward, terminated, info

    def test_A_reach_reward(self):
        """Test the R_reach component by moving the object relative to the EE."""
        self.env.reset()
        ee_pos = self.env._safe_site_pos(self.env.ee_site_name)
        self.assertIsNotNone(ee_pos, "End-effector site not found.")

        # Start with the object far away and run once to set the baseline distance.
        self._set_object_pos(ee_pos + np.array([0.3, 0, 0]))
        self._run_reward_calculation(self.dummy_action)

        # Move object closer -> expect positive reward.
        self._set_object_pos(ee_pos + np.array([0.05, 0, 0]))
        _, _, info_closer = self._run_reward_calculation(self.dummy_action)
        self.assertGreater(info_closer["R_reach"], 0, "Moving object closer should yield positive reach reward.")
        print("\n✅ [TestRewardWrapper] Positive reach reward is correct.")

        # Move object farther away -> expect negative reward.
        self._set_object_pos(ee_pos + np.array([0.4, 0, 0]))
        _, _, info_farther = self._run_reward_calculation(self.dummy_action)
        self.assertLess(info_farther["R_reach"], 0, "Moving object farther should yield negative reach reward.")
        print("✅ [TestRewardWrapper] Negative reach reward is correct.")

    def test_B_grasp_reward(self):
        """Test the R_grasp component is awarded once."""
        self.env.reset()
        ee_pos = self.env._safe_site_pos(self.env.ee_site_name)
        self.assertIsNotNone(ee_pos)

        # Place object exactly at the gripper's position.
        self._set_object_pos(ee_pos.copy())
        
        # First grasp attempt should succeed.
        _, _, info = self._run_reward_calculation(self.gripping_action)
        self.assertEqual(info["R_grasp"], self.env.grasp_reward, "A successful grasp should yield the grasp reward.")

        # Subsequent grasp attempts should yield zero reward.
        _, _, info_after = self._run_reward_calculation(self.gripping_action)
        self.assertEqual(info_after["R_grasp"], 0, "Grasp reward should only be given once.")
        print("✅ [TestRewardWrapper] Grasp reward logic is correct.")

    def test_C_lift_and_place_rewards(self):
        """Test R_lift and R_place components by simulating a state transition."""
        self.env.reset()

        ee_pos = self.env._safe_site_pos(self.env.ee_site_name)
        self.assertIsNotNone(ee_pos)

        # --- Grasp: put object exactly at EE so distance < grasp threshold ---
        obj_at_ee = ee_pos.copy()
        self._set_object_pos(obj_at_ee)

        # Temporarily raise the lift threshold so lift doesn't fire during grasp
        original_thresh = self.env.lift_z_threshold
        self.env.lift_z_threshold = float(ee_pos[2] + 0.05)

        _, _, info_grasp = self._run_reward_calculation(self.gripping_action)
        self.assertEqual(info_grasp["R_grasp"], self.env.grasp_reward)
        self.assertEqual(info_grasp["R_lift"], 0.0)  # not lifted yet by construction
        self.assertTrue(self.env._grasp_achieved)

        # --- Lift: restore normal threshold; now is_lifted becomes True -> pay once ---
        self.env.lift_z_threshold = original_thresh  # ~0.45
        _, _, info_lift = self._run_reward_calculation(self.gripping_action)
        self.assertEqual(info_lift["R_lift"], self.env.lift_reward)
        self.assertTrue(self.env._lift_achieved)

        # --- Place shaping: move XY toward goal while Z stays high ---
        goal_pos = self.env._safe_body_pos(self.env.goal_body_name)
        self.assertIsNotNone(goal_pos)
        moved_toward_goal = obj_at_ee.copy()
        moved_toward_goal[:2] = (obj_at_ee[:2] + goal_pos[:2]) / 2.0
        self._set_object_pos(moved_toward_goal)

        _, _, info_place = self._run_reward_calculation(self.gripping_action)
        self.assertGreater(info_place["R_place"], 0.0)

  
    def test_D_success_reward_and_termination(self):
        """Test the R_success component and episode termination."""
        self.env.reset()
        ee_pos = self.env._safe_site_pos(self.env.ee_site_name)
        self.assertIsNotNone(ee_pos)

        # 1. Grasp the object.
        self._set_object_pos(ee_pos.copy())
        self._run_reward_calculation(self.gripping_action)
        
        # 2. Lift the object.
        self._set_object_pos(ee_pos + np.array([0, 0, 0.1]))
        self._run_reward_calculation(self.gripping_action)

        # 3. Place the object successfully on the goal.
        goal_pos = self.env._safe_body_pos(self.env.goal_body_name)
        self.assertIsNotNone(goal_pos)
        success_pos = goal_pos.copy()
        success_pos[2] = 0.41  # Set to table height for a successful placement.
        self._set_object_pos(success_pos)
        
        _, terminated, info = self._run_reward_calculation(self.gripping_action)
        self.assertEqual(info["R_success"], self.env.success_reward, "Achieving the goal should yield the success reward.")
        self.assertTrue(terminated, "The episode should be terminated on success.")
        print("✅ [TestRewardWrapper] Success reward and termination are correct.")

    @classmethod
    def tearDownClass(cls):
        """Clean up the base environment after all tests are done."""
        if hasattr(cls, 'base_env') and cls.base_env is not None:
            cls.base_env.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)