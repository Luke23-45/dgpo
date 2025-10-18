        ds = ExpertDataset(
            urdf_path=cfg["urdf_path"],
            env_xml_path=cfg.get("xml_path"),
            base_seed=seed_for_worker,
            max_samples_per_epoch=samples_per_worker,
            skip_on_error=cfg.get("skip_on_error", True),
            
            # 3. Pass the INSTANCE, not the dictionary.
            scripted_cfg=expert_config_instance,
            
            object_size=tuple(np.array(cfg.get("object_size", [0.04,0.04,0.04])).tolist()),
            object_grasp_width=float(cfg.get("grasp_width", 0.6)),
            action_scaling_factor=float(cfg.get("action_scaling_factor", 0.5)),
            warmup=bool(cfg.get("warmup", True)),
            yield_full_obs=True,
        )