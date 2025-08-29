import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

class PandaEnv(gym.Env):
    """
    Final, robust version. Uses a simplified XML and correctly finds and maps
    the actuators defined in the included panda.xml file.
    """
    def __init__(self, xml_path="envs/panda_pick_place.xml", render_mode="rgb_array"):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.render_mode = render_mode
        self.renderer = mujoco.Renderer(self.model, height=128, width=128)
        self.max_episode_steps = 200
        self.current_step = 0
        
        # --- Robust Actuator Mapping (Your Excellent Idea + Refinement) ---
        # Define the names of the actuators we intend to control.
        # These names come from the original panda.xml file.
        self._actuator_names = [
            "actuator1", "actuator2", "actuator3", "actuator4",
            "actuator5", "actuator6", "actuator7", "actuator8" # actuator8 is the gripper
        ]
        # Get their integer indices in the full actuator list of the model.
        self._actuator_indices = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in self._actuator_names
        ])
        
        # --- Define Spaces Dynamically ---
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
            "end_effector_pose": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        })

        # The action space size is now the number of actuators we found.
        action_dim = len(self._actuator_names)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        print(f"--- PandaEnv Initialized: Controlling {action_dim} actuators. ---")

    def _get_obs(self) -> dict:
        self.renderer.update_scene(self.data, camera="fixed_camera")
        image = self.renderer.render()
        
        # The site is now correctly named "attachment_site" from panda.xml
        ee_pos = self.data.site('attachment_site').xpos.copy()
        ee_rot_mat = self.data.site('attachment_site').xmat.copy().reshape(3, 3)
        ee_quat = R.from_matrix(ee_rot_mat).as_quat()
        ee_pose_7d = np.concatenate([ee_pos, ee_quat])

        return {"image": image, "end_effector_pose": ee_pose_7d}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        mujoco.mj_resetData(self.model, self.data)
        
        # Randomization logic
        cube_x = self.np_random.uniform(0.4, 0.6)
        cube_y = self.np_random.uniform(-0.1, 0.1)
        self.data.joint('object_joint').qpos[:2] = [cube_x, cube_y]
        
        goal_x = self.np_random.uniform(0.4, 0.6)
        goal_y = self.np_random.uniform(0.2, 0.4)
        self.model.body('goal').pos[:2] = [goal_x, goal_y]

        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self.current_step += 1
        
        # Correctly map the 8-dim action to the specific actuator indices
        ctrl_range = self.model.actuator_ctrlrange[self._actuator_indices]
        scaled_action = ctrl_range[:, 0] + 0.5 * (action + 1.0) * (ctrl_range[:, 1] - ctrl_range[:, 0])
        self.data.ctrl[self._actuator_indices] = scaled_action
        
        mujoco.mj_step(self.model, self.data, nstep=5)
        
        obs = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = self.current_step >= self.max_episode_steps
        info = {}
        
        return obs, reward, terminated, truncated, info