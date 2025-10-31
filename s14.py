# FILE: eval/run_vidhis.py
# Inside run_vidhis_evaluation function...

    # ... (code for setup, creating dirs, loading models and goal is fine)

    # --- START: ROBUST PATCH 3 (Evaluation Loop) ---
    # Replace the "Initialize Environment" and MPC loop sections with this.

    # --- Initialize Environment ---
    log.info("Initializing environment...")
    # NOTE: You must have a standalone environment creation utility now.
    # We will assume a function `create_panda_env` exists for this example.
    # You will need to move the logic from RLFineTuner._build_single_env into a shared file.
    # For now, let's create it directly.
    env = PandaEnv(**cfg.environment.env_kwargs) # Assumes kwargs are in config
    # Wrap with Gymnasium wrappers if needed
    env = gym.wrappers.TimeLimit(env, max_episode_steps=cfg.environment.max_episode_steps)
    
    obs_horizon = cfg.model.controller.observation_horizon
    # Get the base observation space from the unwrapped environment
    obs_space = env.unwrapped.observation_space if hasattr(env, 'unwrapped') else env.observation_space
    
    obs_history_buffer = ObsHistoryBuffer(n_envs=1, history_len=obs_horizon, obs_space=obs_space)
    log.info("Environment and history buffer initialized.")

    # ... (evaluation loop setup: episode_results, max_steps) ...
    
    for ep_idx in range(cfg.evaluation.num_episodes):
        # ... (episode setup: log message, timers, etc.) ...
        
        obs_dict, info = env.reset(seed=cfg.evaluation.seed + ep_idx)
        # The buffer expects a dict of np.ndarrays, which env.reset provides.
        obs_history_buffer.reset(0, obs_dict)

        while not done and step_count < max_steps:
            # ... (timing) ...

            # 1. Get RAW NumPy observation history from the buffer
            current_hist_obs_dict_np = obs_history_buffer.get_stacked(0)

            # 2. Preprocess the history for the models (NumPy -> Tensor)
            obs_history_batch_tensors = preprocess_obs_history(current_hist_obs_dict_np, cfg, device)
            
            # Prepare planner input (last image from the preprocessed history)
            current_image_tensor = obs_history_batch_tensors['image_primary'][:, -1, ...]
            progress_scalar = torch.tensor([[step_count / max_steps]], device=device, dtype=torch.float32)

            # 3. Planner Inference (this part was mostly correct)
            with torch.no_grad():
                subgoal_img_tensor = planner.sample(
                    current_image=current_image_tensor,
                    goal_image=goal_image_tensor,
                    progress=progress_scalar.squeeze(), # Ensure progress is (B,)
                    num_inference_steps=cfg.inference.planner_inference_steps
                )
            # ... (subgoal saving logic is fine)

            # 4. Controller Inference (Corrected Call)
            with torch.no_grad():
                # Call the correct `sample` method with the correct arguments
                action_trajectory_tensor = controller.sample(
                    obs=obs_history_batch_tensors,
                    subgoal_image=subgoal_img_tensor,
                    guidance_scale=cfg.inference.controller_guidance_scale
                ) # Output shape (1, H_a, A_dim)

            action_trajectory = action_trajectory_tensor[0].cpu().numpy()

            # ... (the rest of the MPC execution loop, rendering, and episode end logic is fine)
            
    # --- END: ROBUST PATCH 3 ---