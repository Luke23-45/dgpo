# FILE: scripts/pretrain_diffusion.py

# ... (in `main` function) ...

    # Default: user-supplied ExpertTrajectoryDataset usage
    from utils.expert_dataset import ExpertTrajectoryDataset, collate_fn # Make sure collate_fn is imported
    
    demo_path = cfg["dataset"]["path"]
    
    # --- START OF PATCH ---
    # Instantiate the dataset with the required horizon lengths from the config
    train_dataset = ExpertTrajectoryDataset(
        demo_path,
        observation_horizon=cfg["model"]["observation_horizon"],
        action_horizon=cfg["model"]["action_horizon"]
    )
    val_dataset = None # (You can create a validation split in a similar way if needed)
    
    # Pass the NEW collate_fn to the DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["dataset"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["dataset"].get("num_workers", 4)),
        collate_fn=collate_fn, # <-- IMPORTANT
        pin_memory=True,
        drop_last=True
    )
    # --- END OF PATCH ---

    # The rest of the script from here is mostly compatible.
    # The `train()` function already expects a batch of (obs_dict, action_tensor).
    # You will, however, need to update your denoiser and cond_embed_fn to handle
    # the new sequence-based inputs.
    
    # train(cfg, train_dataset, val_dataset) # This call is now incorrect, we pass the loader
    train(cfg, train_loader, val_loader)