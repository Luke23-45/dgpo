# [IN FILE: train/train_semantic_planner.py]

@hydra.main(version_base=None, config_path="../configs", config_name="train_semantic_planner_config")
def main(cfg: DictConfig) -> None:
    """
    Main Entry Point. Sets up environment, loggers, and starts training.
    """
    # 1. Reproducibility
    pl.seed_everything(cfg.seed, workers=True)
    
    logger.info("--- Starting AWSP Training Pipeline (v9.0 SOTA) ---")
    logger.info(f"Working Dir: {os.getcwd()}")
    
    # 2. Logging Setup
    output_dir = Path("/content/drive/MyDrive/pda/logs_awsp/")
    loggers = [TensorBoardLogger(save_dir=".", name="tb_logs")]
    
    if cfg.logging.get("use_wandb", False):
        os.environ["WANDB_MODE"] = cfg.logging.get("wandb_mode", "online")
        if "WANDB_API_KEY" not in os.environ and cfg.logging.get("wandb_mode") != "offline":
            logger.warning("WandB enabled but API Key not found. Switching to offline.")
            os.environ["WANDB_MODE"] = "offline"
            
        wandb_logger = WandbLogger(
            project=cfg.logging.wandb_project,
            name=cfg.logging.run_name,
            save_dir=str(output_dir),
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        loggers.append(wandb_logger)

    # 3. Data & Model
    datamodule = SemanticPlannerDataModule(cfg)
    model = SemanticPlannerLightningModule(cfg)

    # 4. Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="awsp-{epoch:02d}-{val/pos_error_m:.4f}",
        monitor="val/pos_error_m",
        mode="min",
        save_top_k=3,
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    progress_bar = TQDMProgressBar(refresh_rate=10)

    # 5. Trainer
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=cfg.training.max_epochs,
        logger=loggers,
        callbacks=[checkpoint_callback, lr_monitor, progress_bar],
        gradient_clip_val=cfg.training.get("gradient_clip_val", 1.0),
        precision=cfg.training.get("precision", "16-mixed"),
        accumulate_grad_batches=cfg.trainer.get("accumulate_grad_batches", 1),
        log_every_n_steps=10,
        check_val_every_n_epoch=cfg.training.get("check_val_every_n_epoch", 1),
    )

    # 6. Resume Logic (Standard PL)
    # Only set this if you want to resume training a v9 model exactly where it left off.
    resume_path = cfg.training.get("resume_from_checkpoint")
    ckpt_arg = None

    if resume_path and os.path.exists(resume_path):
        logger.info(f"Resuming training from checkpoint: {resume_path}")
        ckpt_arg = resume_path
    else:
        logger.info("No resume checkpoint found. Starting fresh.")

    # 7. Execute
    try:
        logger.info("Starting trainer.fit()...")
        trainer.fit(
            model, 
            datamodule=datamodule,
            ckpt_path=ckpt_arg 
        )
        logger.info(f"Training complete. Best model: {checkpoint_callback.best_model_path}")
    except Exception as e:
        logger.exception(f"Training failed with exception: {e}")
        raise e
    finally:
        for lg in loggers:
            if isinstance(lg, WandbLogger):
                import wandb
                if wandb.run:
                    wandb.finish()
                logger.info("W&B run finalized.")