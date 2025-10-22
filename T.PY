# FILE: scripts/pretrain_diffusion.py

# ... (inside the DiffusionPretrainer class)

    def run(self):
        """The main entry point to start the training process, with SOTA checkpointing."""
        log.info(f"Starting training from epoch {self.start_epoch}...")
        
        # --- START OF SOTA PATCH 2 ---
        current_epoch = self.start_epoch
        try:
            for epoch in range(self.start_epoch, self.cfg.training.epochs + 1):
                current_epoch = epoch # Keep track of the current epoch for exception handling
                
                # --- 1. Train for one epoch ---
                self._train_one_epoch(epoch)
                
                # --- 2. Save high-frequency backup ---
                # This happens after every single epoch.
                self._save_backup_checkpoint(epoch)
                
                # --- 3. Run validation and save "best" / "last" checkpoints periodically ---
                if epoch % self.cfg.logging.val_interval_epochs == 0:
                    self._validate_one_epoch(epoch)
            
            log.info(f"Training completed successfully after {self.cfg.training.epochs} epochs.")

        except KeyboardInterrupt:
            log.warning("Training interrupted by user (KeyboardInterrupt).")
            log.info("Performing a final save of the 'last.pth' checkpoint before exiting...")
            
            # Use the existing global save_checkpoint function for the final "resume" checkpoint
            save_checkpoint(
                state={
                    "epoch": current_epoch, # The epoch that was interrupted
                    "global_step": self.global_step,
                    "policy_state_dict": self.policy.state_dict(),
                    "ema_state_dict": self.policy.ema.state_dict() if self.policy.ema else None,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.lr_scheduler.state_dict(),
                    "best_val_loss": self.best_val_loss,
                    "rng_states": {
                        "torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                        "random": random.getstate(),
                    },
                    "config": OmegaConf.to_container(self.cfg, resolve=True),
                },
                is_best=False, # An interrupted run is never the "best"
                checkpoint_dir=self.output_dir / "checkpoints",
            )
            log.info(f"Final state for interrupted epoch {current_epoch} saved to 'last.pth'.")

        except Exception as e:
            log.exception(f"An unexpected error occurred during training at epoch {current_epoch}.")
            # We don't save on unknown errors, as the state might be corrupt.
            raise
            
        finally:
            elapsed_time = time.time() - self.start_time
            log.info(f"Training process finished. Total runtime: {elapsed_time/3600:.2f} hours.")
            if self.use_wandb:
                wandb.finish()
        # --- END OF SOTA PATCH 2 ---