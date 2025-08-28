import gymnasium as gym
import numpy as np
import jax

# We assume the OCTO model is loaded once and passed in.
# This avoids reloading the massive model every time we create an environment.
class SafetyGuidedEnvWrapper(gym.Wrapper):
    """
    Environment wrapper that adds a terminal safety reward based on divergence
    from a pre-trained OCTO foundation model. Implements DGPO-Foundation v4.0.
    """
    def __init__(self, env: gym.Env, octo_model, plausibility_weight: float = 1.0, instruction: str = "pick up the red block"):
        super().__init__(env)
        self.octo_model = octo_model
        self.plausibility_weight = plausibility_weight
        self.instruction = instruction
        
        # JAX random key for OCTO sampling
        self.rng = jax.random.PRNGKey(0)

        # Store trajectory data for the current episode
        self._episode_observations = []

    def reset(self, **kwargs):
        """Reset the environment and clear the trajectory buffer."""
        obs, info = self.env.reset(**kwargs)
        self._episode_observations = [obs]
        return obs, info

    def step(self, action):
        """Step the environment and calculate the final reward if the episode ends."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Store the observation that resulted from the action
        self._episode_observations.append(obs)
        
        done = terminated or truncated
        if done:
            # Episode is over, calculate the terminal plausibility reward
            divergence_score = self._calculate_divergence_score()
            
            # This is R_T from our math spec (Equation 3b)
            R_T = -self.plausibility_weight * divergence_score
            
            # Add the terminal reward to the final step's reward
            reward += R_T
            
            # Store the score in the info dict for logging
            info['plausibility_divergence'] = divergence_score
            print(f"Episode End. Divergence: {divergence_score:.4f}, Terminal Reward: {R_T:.4f}")

        return obs, reward, terminated, truncated, info

    def _calculate_divergence_score(self) -> float:
        """
        Calculates the divergence score for the completed episode.
        This function implements Equation 1 from DGPO-Foundation v4.0.
        """
        if len(self._episode_observations) < 2:
            return 0.0 # Not enough data to calculate divergence

        # The states for which OCTO will predict actions are s_0 to s_{T-1}
        states_for_octo = self._episode_observations[:-1]
        
        # --- Prepare Inputs for Batch Inference ---
        # Stack all images into a single batch
        images = np.array([obs['image'] for obs in states_for_octo])
        
        # The OCTO model expects a time dimension. We add it.
        # This assumes a history window of 1 for simplicity here.
        batched_observation = {
            "image_primary": images[:, np.newaxis, ...],
            "timestep_pad_mask": np.full((len(images), 1), True, dtype=bool)
        }
        
        # Create the task instruction
        task = self.octo_model.create_tasks(texts=[self.instruction] * len(images))

        # --- Run Batch Inference with OCTO ---
        # Get expert predictions for the entire trajectory in one go
        expert_actions = self.octo_model.sample_actions(
            batched_observation,
            task,
            rng=self.rng
        )
        # We only care about the first action predicted at each step and its XYZ position
        expert_poses_xyz = expert_actions[:, 0, :3]

        # --- Get Achieved Poses from Simulation ---
        # We need the end-effector pose from the *resulting* states, s_1 to s_T
        achieved_poses_xyz = np.array([obs['end_effector_pose'][:3] for obs in self._episode_observations[1:]])
        
        # --- Calculate Divergence (Equation 1) ---
        squared_errors = np.sum(np.square(achieved_poses_xyz - expert_poses_xyz), axis=-1)
        divergence = np.mean(squared_errors)
        
        return float(divergence)