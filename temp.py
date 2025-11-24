def configure_optimizers(self):
        """
        Robust SOTA Optimizer Configuration with Differential Learning Rates.
        
        Logic:
        1. Filter parameters into 'Decay' (Weights) and 'No Decay' (Biases/Norms).
        2. Split parameters into 'Backbone' (SigLIP) and 'Head' (Planner).
        3. Apply Lower Learning Rate (0.1x) to Backbone to prevent catastrophic forgetting.
        4. Apply Base Learning Rate (1.0x) to Head to learn task dynamics fast.
        """
        # 1. Initialize Parameter Buckets
        head_decay = []
        head_no_decay = []
        backbone_decay = []
        backbone_no_decay = []

        # 2. Define Filtering Rules
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        name_to_module = {n: m for n, m in self.named_modules()}

        for pn, p in self.named_parameters():
            if not p.requires_grad:
                continue # Skip frozen weights (Most of SigLIP)

            # Determine if this is part of the Vision Backbone (SigLIP)
            is_backbone = "vision_backbone" in pn

            # Determine Weight Decay Eligibility
            apply_decay = False
            
            # A. Default: Apply decay to weights in whitelisted modules
            if pn.endswith("weight"):
                # Safe Parent Extraction
                parent_name = pn.rpartition('.')[0]
                if parent_name in name_to_module:
                    parent_mod = name_to_module[parent_name]
                    if isinstance(parent_mod, whitelist_weight_modules):
                        apply_decay = True
                    elif isinstance(parent_mod, blacklist_weight_modules):
                        apply_decay = False
                else:
                    # Fallback: If standard weight, decay it
                    apply_decay = True
            
            # B. Exceptions: Never decay biases
            if pn.endswith('bias'):
                apply_decay = False
            
            # C. Exceptions: Never decay low-dim parameters (scales, 1D embeddings)
            elif p.ndim < 2:
                apply_decay = False

            # D. Exceptions: Special Token Embeddings (Positional, Type, Query)
            elif "spatial_pos_embedding" in pn or "query_token" in pn or "token_type" in pn:
                apply_decay = False

            # 3. Sort into correct Bucket
            if is_backbone:
                if apply_decay: backbone_decay.append(p)
                else: backbone_no_decay.append(p)
            else:
                if apply_decay: head_decay.append(p)
                else: head_no_decay.append(p)

        # 4. logging verification
        if self.trainer.is_global_zero:
            logger.info(f"Optimizer Groups: Head Decay: {len(head_decay)}, Head No-Decay: {len(head_no_decay)}")
            logger.info(f"                  Backbone Decay: {len(backbone_decay)}, Backbone No-Decay: {len(backbone_no_decay)}")

        # 5. Construct Optimizer
        # Head LR = cfg.optimizer.lr (e.g. 1e-4)
        # Backbone LR = cfg.optimizer.lr * 0.1 (e.g. 1e-5)
        base_lr = self.cfg.optimizer.lr
        backbone_lr = base_lr * 0.1
        weight_decay = self.cfg.optimizer.weight_decay

        optimizer = torch.optim.AdamW(
            [
                {"params": head_decay, "lr": base_lr, "weight_decay": weight_decay},
                {"params": head_no_decay, "lr": base_lr, "weight_decay": 0.0},
                {"params": backbone_decay, "lr": backbone_lr, "weight_decay": weight_decay},
                {"params": backbone_no_decay, "lr": backbone_lr, "weight_decay": 0.0},
            ],
            betas=(0.9, 0.999)
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.trainer.estimated_stepping_batches * self.cfg.optimizer.warmup_percentage),
            num_training_steps=self.trainer.estimated_stepping_batches
        )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}