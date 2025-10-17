# FILE: scripts/pretrain_diffusion.py

# ... (inside the DiffusionPretrainer class)

    def _build_policy(self) -> DiffusionPolicy:
        """Constructs the DiffusionPolicy from the configuration."""
        # Extract proprioception dimension from the dataset's observation space
        # This is a robust way to avoid hardcoding dimensions.
        sample_obs, _ = self.train_loader.dataset[0]
        proprio_dim = sample_obs["proprio"].shape[-1]
        log.info(f"Inferred proprioception dimension: {proprio_dim}")

        scheduler_cfg = NoiseSchedulerConfig(
            beta_start=self.cfg.scheduler.beta_start,
            beta_end=self.cfg.scheduler.beta_end,
            schedule=self.cfg.scheduler.schedule_type,
            timesteps=self.cfg.scheduler.timesteps,
        )

        model_cfg = self.cfg.model
        
        # --- THIS IS THE CORRECTED, FINAL CALL ---
        # It matches the constructor of the state-of-the-art DiffusionPolicy
        # and reads from the simplified, correct Hydra config.
        policy = DiffusionPolicy(
            # Core dimensions
            proprio_dim=proprio_dim,
            H_o=model_cfg.observation_horizon,
            H_a=model_cfg.action_horizon,
            action_dim=model_cfg.action_dim,
            
            # Architectural dimensions from the model config
            image_feat_dim=model_cfg.image_feat_dim,
            d_model=model_cfg.d_model,
            
            # Denoiser-specific hyperparameters
            denoiser_layers=model_cfg.denoiser_layers,
            denoiser_heads=model_cfg.denoiser_heads,
            
            # Scheduler and training parameters
            scheduler_cfg=scheduler_cfg,
            cfg_p_uncond=self.cfg.training.cfg_p_uncond,
            ema_decay=self.cfg.training.ema_decay,
            device=self.device,
        )
        return policy