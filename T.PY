# FILE: train_rl.py (Corrected RLFineTuner.__init__ section)

        log.info("Initializing DictReplayBuffer for dictionary observations...")
        # [CORRECTED] Directly use the correctly imported DictReplayBuffer.
        # The import check is now handled at the top of the file.
        self.replay_buffer = DictReplayBuffer(
            buffer_size=self.cfg.rl_algorithm.buffer_size,
            observation_space=history_obs_space,
            action_space=self.env.action_space,
            device=self.device,
            n_envs=self.n_envs,
            handle_timeout_termination=True,  # Important for PBRS
        )