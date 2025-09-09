
    def compute_action(
        self,
        target_pose_7d: np.ndarray,
        current_joint_angles: np.ndarray,
        max_delta: float = 0.1, 
        solution_position_tolerance: float = 0.01,
    ) -> np.ndarray:
        """
        Computes a normalized POSITION action to move towards a target pose.

        This method uses the internal IK solver to find the target joint angles
        and then normalizes them to the action space [-1, 1]. The underlying
        MuJoCo simulation is expected to have a position controller (e.g., PD controller)
        to interpret these commands.

        Returns:
            A NumPy array of shape (7,) representing the normalized target joint positions.
            Returns the current joint positions (a "hold" command) on IK failure.
        """
        n_active = len(self._active_idx)

        # 1. Solve for the target joint configuration.
        target_joint_angles = self._get_target_joint_angles(
            target_pose_7d,
            current_joint_angles,
            solution_position_tolerance,
        )

        # 2. Handle IK failure: command a "hold position" action.
        if target_joint_angles is None:
            logger.warning("IK solver failed. Commanding a hold action (current joint positions).")
            target_joint_angles = current_joint_angles

        # 3. Normalize the target joint angles to the action space [-1, 1].
        # We use the URDF joint limits as the basis for normalization.
        action = np.zeros(n_active, dtype=np.float32)
        for i in range(n_active):
            lo, hi = self._joint_limits[i]
            # Handle infinite limits gracefully.
            if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-6:
                action[i] = np.clip(target_joint_angles[i], -1.0, 1.0)
                continue
            
            # Scale to [0, 1]
            scaled_pos = (target_joint_angles[i] - lo) / (hi - lo)
            # Scale to [-1, 1]
            action[i] = 2.0 * scaled_pos - 1.0

        # Clip to ensure it's strictly within the action space bounds.
        final_action = np.clip(action, -1.0, 1.0)

        # Final safety check for NaN/inf values.
        if not np.all(np.isfinite(final_action)):
            logger.warning("IK produced a non-finite action. Returning zeros to prevent crash.")
            return np.zeros(n_active, dtype=np.float32)

        return final_action
