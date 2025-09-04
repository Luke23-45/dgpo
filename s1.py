    @torch.inference_mode()
    def _calculate_divergence(self) -> float:
        """
        Calculates a robust divergence between the agent's recorded EE-trajectory
        and OCTO's predicted poses (position + orientation).
        Returns a scalar float divergence (lower = more plausible).
        This function is defensive: on any error or mismatch it logs and returns 0.0.
        """
        # Quick preconditions
        if not getattr(self, "_episode_trajectory", None):
            logger.debug("No episode trajectory recorded; divergence=0.0")
            return 0.0
        if getattr(self, "octo_model", None) is None:
            logger.debug("No OCTO model available; divergence=0.0")
            return 0.0
        logger.info("==========================================================")
        logger.info(">>> ENTERING _calculate_divergence <<<")
        # Subsample trajectory to reduce compute
        stride = max(1, getattr(self, "div_frame_stride", 1))
        trajectory = self._episode_trajectory[::stride]
        if not trajectory:
            logger.debug("Trajectory empty after subsampling; divergence=0.0")
            return 0.0
        logger.info(f"Step 1: Trajectory collected with {len(obs_list)} observations.")
        logger.info("Structure of the FIRST raw observation from the environment (obs_list[0]):")
        self._print_dict_structure(obs_list[0])
        # Required observation keys: prefer internal_full_proprio, fallback to proprio
        required_img_key = "image_primary"
        proprio_key = "internal_full_proprio" if "internal_full_proprio" in trajectory[0]["obs"] else "proprio"
        if required_img_key not in trajectory[0]["obs"] or proprio_key not in trajectory[0]["obs"]:
            logger.error("Missing required keys for divergence: "
                        f"need '{required_img_key}' and '{proprio_key}'; skipping divergence.")
            return 0.0

        # Stack agent poses (should be [N,7] with pos(3)+quat(4))
        try:
            agent_poses = np.stack([t["ee_pose"] for t in trajectory], axis=0).astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to stack agent poses for divergence: {e}")
            return 0.0


        # --- 1. Prepare Inputs using our perfected adapter ---
        try:
            # Get the list of raw environment observations from the trajectory
            obs_list = [t["obs"] for t in trajectory]
            
            # Use the single, correct source of truth to build the entire batch.
            # This function now handles all the complexity of normalization,
            # padding, and creating the final two-level dictionary structure.
            octo_input = build_octo_batch_from_list(obs_list)

        except Exception as e:
            logger.error(f"Failed to prepare OCTO input for divergence using adapter: {e}")
            return 0.0

        # Run OCTO forward (defensive)
        try:
            try:
                if not getattr(self, "_div_schema_checked", False):
                    from utils.validation import validate_against_example_batch
                    validate_against_example_batch(self.octo_model, octo_input)
                    self._div_schema_checked = True
            except Exception:
                pass
            predicted_raw = self.octo_model.sample_actions(octo_input, self.octo_task)
            # Convert to numpy (handle torch/numpy-like returns)
            if hasattr(predicted_raw, "cpu") and hasattr(predicted_raw, "numpy"):
                predicted_raw = predicted_raw.cpu().numpy()
            predicted_raw = np.asarray(predicted_raw)
            # Expect shape (B, T, D) or similar -> take [:,0,:]
            octo_poses = np.asarray(predicted_raw[:, 0, :])
        except Exception as e:
            logger.error(f"OCTO inference failed during divergence calculation: {e}")
            return 0.0

        # Align lengths
        try:
            T = min(agent_poses.shape[0], octo_poses.shape[0])
            if T == 0:
                return 0.0
            agent_poses = agent_poses[:T]
            octo_poses = octo_poses[:T]
        except Exception as e:
            logger.error(f"Failed to align OCTO & agent pose lengths: {e}")
            return 0.0

        # Position MSE (safe numerics)
        try:
            pos_err = agent_poses[:, :3] - octo_poses[:, :3]
            # per-frame squared error then mean
            per_frame_sq = np.sum(pos_err * pos_err, axis=-1)
            pos_mse = float(np.mean(per_frame_sq))
            if not np.isfinite(pos_mse):
                pos_mse = float(np.nan_to_num(pos_mse, nan=0.0, posinf=1e6, neginf=1e6))
        except Exception as e:
            logger.error(f"Position MSE computation failed: {e}")
            pos_mse = 0.0

        # Quaternion distance (1 - |dot|^2) average
        def _safe_normalize_quat(q: np.ndarray) -> np.ndarray:
            q = np.asarray(q, dtype=np.float32)
            if q.ndim == 1:
                q = q[np.newaxis, :]
            norm = np.linalg.norm(q, axis=-1, keepdims=True)
            norm = np.clip(norm, 1e-8, None)
            qn = q / norm
            qn[~np.isfinite(qn)] = 0.0
            return qn

        try:
            q_agent = _safe_normalize_quat(agent_poses[:, 3:])
            q_octo_raw = octo_poses[:, 3:].astype(np.float32)

            # If octo outputs 'wxyz' but we use 'xyzw', convert (configurable)
            if getattr(self, "quat_format", "xyzw") == "wxyz":
                # convert wxyz -> xyzw (move first element to last)
                if q_octo_raw.shape[-1] == 4:
                    q_octo_raw = q_octo_raw[:, [1, 2, 3, 0]]
            q_octo = _safe_normalize_quat(q_octo_raw)

            # dot product per-frame, absolute value (handle antipodal equivalence)
            dot = np.sum(q_agent * q_octo, axis=-1)
            dot = np.clip(np.abs(dot), 0.0, 1.0)
            quat_dist_mean = float(np.mean(1.0 - dot * dot))
            if not np.isfinite(quat_dist_mean):
                quat_dist_mean = float(np.nan_to_num(quat_dist_mean, nan=0.0, posinf=1e6, neginf=1e6))
        except Exception as e:
            logger.error(f"Quaternion divergence computation failed: {e}")
            quat_dist_mean = 0.0

        # Combine using class weights (fall back to sane defaults)
        pos_w = float(getattr(self, "pos_weight", 1.0))
        rot_w = float(getattr(self, "rot_weight", 1.0))
        divergence = pos_w * pos_mse + rot_w * quat_dist_mean

        # Final numeric safety
        divergence = float(np.nan_to_num(divergence, nan=0.0, posinf=1e6, neginf=1e6))
        if not np.isfinite(divergence):
            divergence = 0.0

        logger.debug(f"divergence computed: pos_mse={pos_mse:.6g}, quat={quat_dist_mean:.6g}, total={divergence:.6g}")
        return divergence


