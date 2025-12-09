def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Optional[torch.Tensor]:
        if not batch: return None

        # 1. Forward Pass
        outputs = self.model(batch)
        pred_pose_chunk = outputs['pose_chunk']
        pred_grip_chunk = outputs['gripper_chunk']
        pred_phase_logits = outputs['phase_logits']

        # 2. Ground Truths
        gt_pose_chunk = batch['gt_pose_chunk']
        gt_grip_chunk = batch['gt_grip_chunk']
        gt_phase = batch['gt_phase_label']
        advantage = batch['advantage']

        # ----------------------------------------------------------------------
        # 3. Algorithm Selection (AWR vs BC)
        # ----------------------------------------------------------------------
        # Default to 'awr' if not specified in config to keep backward compatibility
        algo = self.cfg.training.get("algorithm", "awr").lower()

        if algo == "bc":
            # [Pure Behavior Cloning]
            # Ignore advantages completely. Every sample has a weight of 1.0.
            # We construct a tensor of ones on the correct device.
            weights = torch.ones_like(advantage.squeeze(-1))
            
            # (Optional) Log mode once for safety
            if self.trainer.global_step == 0 and batch_idx == 0:
                logger.info(">>> TRAINING MODE: PURE BEHAVIOR CLONING (Weights=1.0) <<<")

        else:
            # [Advantage Weighted Regression - SOTA]
            # Calculate exp(A/tau)
            weights = self._compute_awr_weights(advantage.squeeze(-1))

        # ----------------------------------------------------------------------

        # 4. Compute Trajectory Losses (Weighted)
        
        # A. Pose Loss
        raw_pose_loss = self.pose_criterion(pred_pose_chunk, gt_pose_chunk)
        POSE_SCALE = 10.0 
        pose_loss = (raw_pose_loss.mean(dim=[1, 2]) * weights).mean() * POSE_SCALE

        # B. Gripper Loss
        raw_grip_loss = F.binary_cross_entropy_with_logits(
            pred_grip_chunk, gt_grip_chunk, 
            pos_weight=self.grip_pos_weight, 
            reduction='none'
        ) 
        grip_loss = (raw_grip_loss.mean(dim=[1, 2]) * weights).mean()

        # 5. Phase Loss (Auxiliary - Always Supervised)
        phase_loss = self.phase_criterion(pred_phase_logits, gt_phase)

        # 6. Total Loss
        total_loss = pose_loss + (self.lambda_gripper * grip_loss) + (self.lambda_phase * phase_loss)

        # 7. Logging
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/weights_mean", weights.mean(), on_step=False, on_epoch=True) # Check this! Should be exactly 1.0 for BC
        
        # ... existing accuracy logging ...
        
        return total_loss