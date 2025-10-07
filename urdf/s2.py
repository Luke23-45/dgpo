# In scripts/train_hybrid.py

# (All imports and the AdvisedPPO class definition remain the same)
# NO LONGER NEEDED: from stable_baselines3.common.monitor import Monitor
# NO LONGER NEEDED: from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# NO LONGER NEEDED: from run_experiment import RLRewardWrapper, OctoToSB3Adapter

def run_hybrid_training(args: argparse.Namespace):
    """
    Main function to orchestrate the BC-Advised RL training process.
    This version PRESERVES robust setup logic by calling the centralized
    setup_environment function with the explicit Monitor wrapper enabled.
    """
    # --- 1. SETUP AND INITIALIZATION (PRESERVED LOGIC) ---
    run_name = args.run_name or f"advised_ppo_{Path(args.bc_init_dir).name}"
    run_dir = Path(args.output_dir) / run_name
    checkpoints_dir = run_dir / "checkpoints"
    backups_dir = run_dir / "backups"
    
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)
    backups_dir.mkdir(exist_ok=True)

    with (run_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=4)
    logger.info(f"🚀 Starting BC-Advised RL experiment: {run_name}")

    set_random_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device_str = resolve_device(args.device)
    device = torch.device(device_str)

    # --- 2. SETUP ENVIRONMENT (CLEANED UP & CORRECTED) ---
    # We now call the centralized setup_environment function and ensure the
    # Monitor wrapper is added for proper logging.
    env = setup_environment(
        xml_path=args.xml_path,
        seed=args.seed,
        control_mode='absolute', 
        n_envs=args.n_envs,
        add_monitor_wrapper=True, # <-- THE EXPLICIT FIX
        # Pass all other relevant args
        octo_model=None,
        w_plausibility=0.0,
        pos_scale=args.pos_scale,
        rot_scale=getattr(args, 'rot_scale', 1.0),
        div_clip=getattr(args, 'div_clip', 10.0),
        scripted_expert=None,    
        w_guidance=args.w_guidance,
        w_guidance_dense=args.w_guidance_dense,
        guidance_clip=args.guidance_clip,
        grasp_reward=args.grasp_reward,
        lift_reward=args.lift_reward,
        success_reward=args.success_reward,
    )
    logger.info("Environment created using centralized setup_environment with Monitor wrapper.")

    # --- 3. PREPARE INITIAL WEIGHTS and BC ADVISOR (Definitive Fix) ---
    initial_policy_state_dict = None
    try:
        logger.info("Loading BC model to serve as advisor and for weight transfer...")
        bc_model_path = str(Path(args.bc_init_dir) / "checkpoints" / "best_model.pth")
        ckpt = load_bc_checkpoint(bc_model_path, device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

        action_dim = env.action_space.shape[0]
        bc_model = BCNet(n_actions=action_dim).to(device)
        bc_model.load_state_dict(state_dict, strict=False)
        logger.info("BC model loaded to serve as advisor.")

        if transfer_bc_weights:
            logger.info("Creating temporary agent for weight transfer...")
            temp_agent = initialize_ppo_agent(env, run_dir, args.seed, device_str)
            transfer_bc_weights(bc_model, temp_agent)
            
            initial_policy_state_dict = temp_agent.policy.state_dict()
            del temp_agent
            logger.info("Successfully prepared initial weights for RL policy.")
        else:
            logger.warning("`transfer_bc_weights` not available. RL policy will start from scratch.")
    except Exception as e:
        logger.error(f"Failed during BC model loading or weight transfer: {e}", exc_info=True)
        env.close()
        return

    # --- 4. CREATE THE FINAL "BC-ADVISED" AGENT (Definitive Fix) ---
    logger.info("Creating the final BC-Advised PPO agent...")
    epsilon_schedule = {
        "initial": args.eps_initial,
        "final": args.eps_final,
        "decay_steps": args.eps_decay_steps,
    }

    policy_kwargs = {
        "features_extractor_class": BCFeaturesExtractor,
        "net_arch": {"pi": [512, 256], "vf": [512, 256]},
    }
    
    ppo_kwargs = {
        "policy": "MultiInputPolicy",
        "env": env,
        "policy_kwargs": policy_kwargs,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": 10,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "tensorboard_log": str(run_dir / "logs"),
        "seed": args.seed,
        "device": device_str,
        "verbose": 1,
    }

    agent = AdvisedPPO(
        bc_policy_advisor=bc_model,
        epsilon_schedule=epsilon_schedule,
        **ppo_kwargs
    )
    
    if initial_policy_state_dict:
        agent.policy.load_state_dict(initial_policy_state_dict)
        logger.info("Successfully loaded initial weights into the AdvisedPPO policy.")

    if args.freeze_features:
        logger.info("--- FEATURE FREEZING ENABLED ---")
        frozen_keys = 0
        for name, param in agent.policy.named_parameters():
            if 'features_extractor' in name:
                param.requires_grad = False
                frozen_keys += 1
        logger.info(f"Froze {frozen_keys} parameters in the RL policy's feature extractor.")

    logger.info("BC-Advised PPO agent is fully configured and ready for training.")

    # --- 5. SETUP CALLBACKS AND START TRAINING (PRESERVED LOGIC) ---
    callbacks = [
        CheckpointCallback(
            save_freq=max(1, args.save_freq // args.n_envs),
            save_path=str(checkpoints_dir),
            name_prefix="advised_rl_policy"
        ),
        SingleFileBackupCallback(
            save_freq=max(1, 5000 // args.n_envs),
            save_path=str(backups_dir),
            name_prefix="latest_backup"
        )
    ]

    logger.info("\n--- Starting BC-Advised Training Loop ---")
    try:
        agent.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        final_model_path = run_dir / "final_policy.zip"
        agent.save(final_model_path)
        logger.info(f"✅ Training complete. Final policy saved to: {final_model_path}")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user; saving current policy and exiting.")
        agent.save(run_dir / "interrupted_policy.zip")
    except Exception as e:
        logger.exception("Unexpected error during training. Saving state and exiting.", exc_info=True)
        try:
            agent.save(run_dir / "error_policy.zip")
        except Exception:
            logger.exception("Failed to save policy after exception.")
    finally:
        try:
            env.close()
        except Exception:
            logger.exception("Failed to close env cleanly.")