        elif self._state == "MOVE_TO_PRE_GRASP":
            """
            [FINAL ADAPTIVE TRAJECTORY VERSION]
            Calculates an adaptive duration for the move based on distance and a desired
            cruise velocity. It then generates a smooth, eased trajectory for that
            duration. This makes motion efficient for both short and long distances
            while retaining perfect smoothness and robustness.
            """
            # === STATE ENTRY LOGIC (runs only ONCE on the first step) ===
            if self._wait_counter == 1:
                # 1. Capture start and calculate final end poses.
                self._start_pre_grasp_pos = ee_pos.copy()
                self._start_pre_grasp_orn = R.from_quat(ee_pose_world[3:])
                end_pos = np.array([cube_pos_world[0], cube_pos_world[1], object_top_z + self.cfg.hover_height])
                self._end_pre_grasp_pos = end_pos
                final_aligned_quat = self._calculate_aligned_orientation(object_orn_world, ee_pose_world[3:])
                self._end_pre_grasp_orn = R.from_quat(final_aligned_quat)

                # 2. THIS IS THE KEY: CALCULATE ADAPTIVE DURATION
                total_dist = np.linalg.norm(self._end_pre_grasp_pos - self._start_pre_grasp_pos)
                # Time = Distance / Speed. We need to convert from seconds to steps.
                # Assuming 1 step = 0.01s (100Hz), multiply by 100.
                travel_steps = int((total_dist / self.cfg.cruise_velocity) * 100)
                
                # Add a buffer for acceleration/deceleration to ensure smoothness.
                self._adaptive_duration = travel_steps + self.cfg.accel_decel_buffer_steps
                
                # Ensure the duration is at least a minimum value to avoid jerky tiny moves.
                if self._adaptive_duration < 20:
                    self._adaptive_duration = 20

            # === CONTINUOUS LOGIC (runs EVERY step) ===

            # 1. Calculate progress using the NEW ADAPTIVE duration.
            progress = min(self._wait_counter / self._adaptive_duration, 1.0)

            # 2. Apply cosine easing for smooth acceleration/deceleration.
            eased_progress = 0.5 * (1.0 - np.cos(progress * np.pi))

            # 3. Interpolate position and orientation using the eased progress.
            interp_pos = self._start_pre_grasp_pos + (self._end_pre_grasp_pos - self._start_pre_grasp_pos) * eased_progress
            key_rots = R.from_quat([self._start_pre_grasp_orn.as_quat(), self._end_pre_grasp_orn.as_quat()])
            slerp = Slerp([0, 1], key_rots)
            interp_orn_quat = slerp(eased_progress).as_quat()

            # 4. Set the robot's target pose.
            self._target_pose_7d = np.concatenate([interp_pos, interp_orn_quat])
            self._gripper_action = -1.0

            # === HYBRID TRANSITION LOGIC ===
            
            # The timeout for the state is the greater of our calculated adaptive duration
            # or the globally configured maximum duration, providing a final safety net.
            timeout = max(self._adaptive_duration, self.cfg.move_to_pre_grasp_duration)

            pos_error = np.linalg.norm(ee_pos - self._end_pre_grasp_pos)
            R_current = R.from_quat(ee_pose_world[3:])
            angular_distance = (self._end_pre_grasp_orn.inv() * R_current).magnitude()
            is_at_destination = (pos_error < self.cfg.pos_tolerance) and \
                                (angular_distance < self.cfg.orn_tolerance_rad)
            is_timed_out = self._wait_counter > timeout

            if is_at_destination or is_timed_out:
                if is_timed_out and not is_at_destination:
                    print(f"WARN: MOVE_TO_PRE_GRASP timed out. Pos Error: {pos_error:.3f}m, Orn Error: {angular_distance:.3f}rad")
                
                self.current_grasp_orientation = self._end_pre_grasp_orn.as_quat()
                self._advance_state("PREPARE_GRIPPER")