    run_experiment(
        xml_path=args.xml_path,
        bc_model_path=bc_model_path_for_func,
        resume_from=resume_from_for_func,
        run_name=run_name_for_func,
        resume_dir=args.resume_dir, # Pass the new flag
        # ... pass all other args from `args` object ...
        total_timesteps=args.total_timesteps,
        save_freq=args.save_freq,
        seed=args.seed,
        device_arg=args.device,
        n_envs=args.n_envs,
        w_plausibility=args.w_plausibility,
        # ... etc ...
        primary_res=primary_res,
        wrist_res=wrist_res,
    )