import gymnasium as gym
import numpy as np
import jax
import cv2 # Import OpenCV for resizing images

class SafetyGuidedEnvWrapper(gym.Wrapper):
    """
    Final, robust version. Correctly formats data for the OCTO model,
    handling history, batching, and image resizing.
    """
    def __init__(self, env: gym.Env, octo_model, plausibility_weight: float = 1.0, instruction: str = "pick up the red block"):
        super().__init__(env)
        self.octo_model = octo_model
        self.plausibility_weight = plausibility_weight
        self.instruction = instruction
        self.rng = jax.random.PRNGKey(0)
        self._episode_observations = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # OCTO requires a history of 2. We pad the initial observation by duplicating it.
        self._episode_observations = [obs, obs]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_observations.append(obs)
        
        done = terminated or truncated
        if done:
            divergence_score = self._calculate_divergence_score()
            R_T = -self.plausibility_weight * divergence_score
            reward += R_T
            info['plausibility_divergence'] = divergence_score
            print(f"\n--- Episode End ---")
            print(f"Divergence Score: {divergence_score:.4f}, Terminal Reward Component: {R_T:.4f}")

        return obs, reward, terminated, truncated, info

    def _calculate_divergence_score(self) -> float:
        if len(self._episode_observations) < 2:
            return 0.0

        # --- CRITICAL DATA FORMATTING ---

        # 1. Prepare image history windows
        # For each step `t`, we need the observation from `t-1` and `t`.
        image_history_windows = []
        for i in range(len(self._episode_observations) - 1):
            # Get images for s_{t-1} and s_t
            img_t_minus_1 = self._episode_observations[i]['image']
            img_t = self._episode_observations[i+1]['image']
            
            # Resize images to the 256x256 that OCTO expects
            img_t_minus_1_resized = cv2.resize(img_t_minus_1, (256, 256))
            img_t_resized = cv2.resize(img_t, (256, 256))
            
            # Stack them to create the (2, H, W, C) history window
            image_history_windows.append(np.stack([img_t_minus_1_resized, img_t_resized]))

        # Create a single batch from all the history windows
        # Shape becomes (num_steps, 2, H, W, C)
        images_batch = np.array(image_history_windows)

        # 2. Prepare the batched observation dictionary for OCTO
        batched_observation = {
            "image_primary": images_batch,
            # We have a full history for all steps, so the mask is all True.
            "timestep_pad_mask": np.full((images_batch.shape[0], 2), True, dtype=bool)
        }
        
        # 3. Create the task instructions for the whole batch
        num_steps = images_batch.shape[0]
        task = self.octo_model.create_tasks(texts=[self.instruction] * num_steps)

        # --- Run Batch Inference with OCTO ---
        expert_actions = self.octo_model.sample_actions(
            batched_observation,
            task,
            rng=self.rng
        )
        expert_poses_xyz = expert_actions[:, 0, :3] # Shape: (num_steps, 3)

        # --- Get Achieved Poses from Simulation ---
        # We need the end-effector pose from the resulting states, s_1 to s_T
        achieved_poses = np.array([obs['end_effector_pose'] for obs in self._episode_observations[1:]])
        achieved_poses_xyz = achieved_poses[:, :3] # Shape: (num_steps, 3)
        
        # --- Calculate Divergence ---
        # Ensure shapes match before calculating error
        if expert_poses_xyz.shape != achieved_poses_xyz.shape:
             print(f"Warning: Mismatched shapes for divergence calculation. Expert: {expert_poses_xyz.shape}, Achieved: {achieved_poses_xyz.shape}")
             return 1.0 # Return a high divergence on error

        squared_errors = np.sum(np.square(achieved_poses_xyz - expert_poses_xyz), axis=-1)
        divergence = np.mean(squared_errors)
        
        return float(divergence)