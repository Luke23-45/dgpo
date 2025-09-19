# FILE: scripts/train_hybrid.py

# ... (inside the run_hybrid_training function) ...

    # OLD, BUGGY FUNCTION:
    # def setup_hybrid_environment(): ...

    # NEW, CORRECTED FUNCTION:
    from stable_baselines3.common.env_util import make_vec_env
    from utils.rl_reward_wrapper import RLRewardWrapper
    from utils.obs_adapters import VecOctoToSB3Adapter # Import the vectorized adapter
    from stable_baselines3.common.vec_env import VecNormalize

    def setup_hybrid_environment():
        """Creates the final, stabilized, vectorized environment for RL fine-tuning."""
        def make_env():
            # Inside here, we ONLY setup the single environment. NO ADAPTERS.
            # It MUST be 'delta' mode for stable RL fine-tuning.
            env = PandaEnv(xml_path=args.xml_path, control_mode='delta')
            env = RLRewardWrapper(
                env,
                grasp_reward=args.grasp_reward,
                lift_reward=args.lift_reward,
                success_reward=args.success_reward,
            )
            return env
        
        # 1. Create the base vectorized environment.
        vec_env = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed)
        
        # 2. Apply the VECTORIZED observation adapter.
        logger.info("Applying Vectorized Observation Adapter.")
        vec_env = VecOctoToSB3Adapter(vec_env)
        
        # 3. Apply VecNormalize for stability. This is the correct final wrapper.
        logger.info("Applying VecNormalize wrapper.")
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        return vec_env

    env = setup_hybrid_environment()
    # Correct the log message to reflect the TRUE control mode.
    logger.info(f"Environment created with control_mode='delta' and stabilization wrappers.")