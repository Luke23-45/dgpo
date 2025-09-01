import gymnasium as gym
import numpy as np
import jax
from collections import deque

class SafetyGuidedEnvWrapper(gym.Wrapper):
    """
    DGPO-Foundation Environment Wrapper (v7.1 - Final Annotated Version)

    This wrapper implements the full, final logic for our research. It combines:
    1.  Intermittent, Dynamic OCTO Guidance: It periodically gets a target pose
        from the pre-trained OCTO model to provide a high-level "common sense"
        direction for the agent. This is efficient and avoids slowing down the simulation.
    2.  Shaped Task Curriculum Reward: It provides a dense, step-by-step reward
        signal that teaches the agent the sub-tasks of the main goal (reaching,
        grasping, lifting, and placing).

    This combination provides a rich and stable learning signal for the PPO agent.
    """

    def __init__(self, env: gym.Env, octo_model, instruction: str = "pick up the red block"):
        """
        Initializes the wrapper.

        Args:
            env: The base PandaEnv environment.
            octo_model: The pre-loaded, pre-trained OCTO model.
            instruction: The natural language command for the task.
        """
        super().__init__(env)
        
        # --- Model and Task Setup ---
        self.octo_model = octo_model
        self.instruction = instruction
        # We create the task object once to be reused, which is more efficient.
        self.task = self.octo_model.create_tasks(texts=[self.instruction])
        # A persistent JAX random key for OCTO model sampling.
        self.rng = jax.random.PRNGKey(0)
        
        # --- Observation History Buffers ---
        # OCTO expects a history of 2 observations. We use deques for efficient appending.
        self.image_history = deque(maxlen=2)
        self.proprio_history = deque(maxlen=2)
        
        # --- Intermittent OCTO Guidance Logic ---
        # We will only call the slow OCTO model once every 25 steps.
        self.octo_update_freq = 25
        self._steps_since_octo_update = 0
        self._current_octo_target = None # The XYZ position suggested by OCTO
        
        # --- Reward Calculation State ---
        self._last_ee_dist_to_target = 0

    def reset(self, **kwargs):
        """
        Resets the underlying environment and all wrapper-specific states.
        """
        obs, info = self.env.reset(**kwargs)
        
        # --- Initialize History ---
        # At the start of an episode, we have no past. So, we "pad" the history
        # by duplicating the very first observation.
        self.image_history.append(obs["image_primary"])
        self.image_history.append(obs["image_primary"])
        self.proprio_history.append(obs["proprio"])
        self.proprio_history.append(obs["proprio"])
        
        # --- Get the first OCTO target for the episode ---
        self._update_octo_target()
        
        # Initialize the distance for the guidance reward calculation
        ee_pos = self.env.data.site('attachment_site').xpos
        self._last_ee_dist_to_target = np.linalg.norm(ee_pos - self._current_octo_target)
        
        return self._package_obs(obs), info

    def _update_octo_target(self):
        """
        Calls the OCTO model to get a new target end-effector pose.
        This is the computationally expensive part that we run intermittently.
        """
        # Format the observation history correctly for the model
        octo_obs = self._format_octo_obs()
        
        # Get a new random key for this sampling instance
        self.rng, key = jax.random.split(self.rng)
        
        # Run inference to get the expert's suggested action
        expert_action_raw = self.octo_model.sample_actions(octo_obs, self.task, rng=key)
        
        # Update our current target to the XYZ position of the first predicted action
        self._current_octo_target = expert_action_raw[0, 0, :3]
        
        # Reset the counter
        self._steps_since_octo_update = 0

    def step(self, action):
        """
        Takes an action, steps the environment, and calculates the full reward.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update our observation history buffers with the new observation
        self.image_history.append(obs["image_primary"])
        self.proprio_history.append(obs["proprio"])
        self._steps_since_octo_update += 1

        # Check if it's time to get a new, updated target from OCTO
        if self._steps_since_octo_update >= self.octo_update_freq:
            self._update_octo_target()
        
        # --- Reward Calculation ---
        # Get current positions of key objects from the simulation
        ee_pos = self.env.data.site('attachment_site').xpos
        cube_pos = self.env.data.geom('object_geom').xpos
        goal_pos = self.env.model.body('goal').pos

        # 1. DGPO GUIDANCE REWARD (Dense)
        # Reward for moving closer to the current OCTO target.
        current_dist_to_target = np.linalg.norm(ee_pos - self._current_octo_target)
        R_guidance = self._last_ee_dist_to_target - current_dist_to_target
        self._last_ee_dist_to_target = current_dist_to_target
        
        # 2. TASK CURRICULUM REWARD (Dense and Sparse)
        # Stage A: Reaching for the cube
        dist_ee_to_cube = np.linalg.norm(ee_pos - cube_pos)
        R_reach = 0.1 * np.exp(-10 * dist_ee_to_cube)

        # Stage B: Grasping the cube
        is_gripping = action[7] > 0.5 # Assumes gripper actuator is the 8th action
        R_grasp = 0
        if dist_ee_to_cube < 0.04 and is_gripping:
            R_grasp = 2.0 

        # Stage C: Lifting the cube
        is_lifted = cube_pos[2] > 0.45 # Z-axis of the cube is above the table
        R_lift = 0
        if is_lifted and R_grasp > 0: # Can only get lift reward while grasping
            R_lift = 5.0

        # Stage D: Placing the cube near the goal
        dist_cube_to_goal = np.linalg.norm(cube_pos[:2] - goal_pos[:2]) # XY distance
        R_place = 0
        if is_lifted:
            R_place = 0.1 * np.exp(-10 * dist_cube_to_goal)

        # Stage E: Final Success (Sparse)
        R_success = 0
        if is_lifted and dist_cube_to_goal < 0.05:
            R_success = 50.0 # Large bonus for completing the task
            terminated = True # End the episode upon success
        
        # Combine all task-related rewards
        R_task = R_reach + R_grasp + R_lift + R_place + R_success
        
        # Small penalty for large actions to encourage smoother movements
        R_penalty = -0.001 * np.square(action).sum()

        # --- FINAL REWARD COMBINATION ---
        # These weights can be tuned in the main training script.
        w_guidance = 1.0
        w_task = 1.0
        
        reward = (w_guidance * R_guidance) + (w_task * R_task) + R_penalty
        
        # Store individual reward components in the info dict for logging and debugging
        info.update({
            'R_guidance': R_guidance, 'R_task': R_task, 'R_reach': R_reach,
            'R_grasp': R_grasp, 'R_lift': R_lift, 'R_place': R_place,
            'R_success': R_success
        })
        
        return self._package_obs(obs), reward, terminated, truncated, info

    def _format_octo_obs(self):
        """Formats the observation history correctly for the OCTO model."""
        return {
            "image_primary": np.stack(self.image_history)[np.newaxis, ...],
            "proprio": np.stack(self.proprio_history)[np.newaxis, ...],
            "timestep_pad_mask": np.array([[True, True]]) # We always have a full history of 2
        }
        
    def _package_obs(self, obs):
        """Ensures the observation returned to the agent is in the correct format."""
        # The SB3 agent only needs to see the *current* observation, not the history.
        # The wrapper handles the history internally.
        return obs