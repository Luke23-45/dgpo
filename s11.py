# FILE: train_rl.py

# ... (inside RLFineTuner.run method, inside the `for loop_idx ...` loop)

                try:
                    # 1. Get s_t (history *before* this step's observation is added)
                    obs_history_t = self.obs_history.get_batch_stacked()

                    # 2. Update the history buffer for all environments to get s_{t+1}
                    # We append the raw observation list, which contains post-reset obs
                    for i in range(self.n_envs):
                        env_next_obs = {k: v[i] for k, v in next_raw_obs_list.items()}
                        self.obs_history.append(i, env_next_obs)
                    
                    # This is s_{t+1} (containing post-reset obs for any done envs)
                    obs_history_t_plus_1 = self.obs_history.get_batch_stacked()

                    # 3. Add the entire batch of transitions to the replay buffer.
                    # SOTA PATCH 4:
                    # Because `handle_timeout_termination=True` (from Patch 2),
                    # the buffer will automatically look at `infos[i]` when `dones[i]`
                    # is True. It will find `"terminal_observation"` and store
                    # a history-padded version of *that* observation as the
                    # `next_obs`, instead of the one from `obs_history_t_plus_1`.
                    self.replay_buffer.add(
                        obs=obs_history_t,
                        next_obs=obs_history_t_plus_1,
                        action=action,
                        reward=rewards,
                        done=dones,
                        infos=infos,
                    )

                    # 4. Handle Episode Terminations (Reset History Buffers for next loop).
                    # This must happen *after* adding to the buffer.
                    for i in range(self.n_envs):
                        if dones[i]:
                            # SOTA PATCH 4:
                            # The environment has *already* reset. The observation
                            # in `next_raw_obs_list[i]` is the *post-reset* observation.
                            # We MUST reset the history buffer to be in sync with
                            # the environment's *new* state.
                            env_post_reset_obs = {k: v[i] for k, v in next_raw_obs_list.items()}
                            self.obs_history.reset(i, env_post_reset_obs)
                
                except Exception as e:
                    log.exception(f"Error processing vectorized step and adding to buffer: {e}")
                    continue # Skip this entire batch if an error occurs

                # --- Update Timestep Counter ---
# ... (rest of run method)