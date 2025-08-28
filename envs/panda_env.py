import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class PandaEnv(gym.Env):
    """A clean, basic Gymnasium environment for the Franka Panda arm in MuJoCo."""
    def __init__(self, xml_path, render_mode="rgb_array"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.render_mode = render_mode
        self.renderer = mujoco.Renderer(self.model, height=128, width=128)
        
        self.max_episode_steps = 200
        self.current_step = 0
        
        # Observation space: a dictionary with image and end-effector pose
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
            "end_effector_pose": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32) # x,y,z,qw,qx,qy,qz
        })
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

    def _get_obs(self):
        # Get image
        self.renderer.update_scene(self.data, camera="fixed_camera")
        image = self.renderer.render()
        
        # Get end-effector pose (position and quaternion rotation)
        ee_pos = self.data.site('attachment_site').xpos.copy()
        ee_quat = self.data.site('attachment_site').xmat.copy().flatten() # Note: MuJoCo gives rotation matrix, needs conversion
        
        # For simplicity, we'll just use position for now
        # A full implementation would convert the 3x3 matrix to a quaternion
        # Placeholder for full 7D pose
        ee_pose_7d = np.concatenate([ee_pos, np.array([1.0, 0.0, 0.0, 0.0])])

        return {"image": image, "end_effector_pose": ee_pose_7d}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        mujoco.mj_resetData(self.model, self.data)
        
        # Randomize cube and goal positions
        # ... (add randomization logic as before) ...
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        # Apply action
        ctrl_range = self.model.actuator_ctrlrange
        scaled_action = ctrl_range[:, 0] + 0.5 * (action + 1.0) * (ctrl_range[:, 1] - ctrl_range[:, 0])
        self.data.ctrl[:] = scaled_action
        mujoco.mj_step(self.model, self.data, nstep=5)
        
        obs = self._get_obs()
        reward = 0 # Base environment has no complex reward
        terminated = False
        truncated = self.current_step >= self.max_episode_steps
        info = {}
        
        return obs, reward, terminated, truncated, info