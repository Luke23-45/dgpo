# FILE: utils/scripted_expert.py

# ... existing code ...
        # Common failure check: Timeout any state to prevent stalling
        self._wait_counter += 1
        if self._wait_counter > self.cfg.failure_timeout_steps:
            self._handle_grasp_failure()
            # Fallback target if failed
            self._target_pose_7d = np.concatenate([ee_pos, self._downward_quat])

        if self._state == "MOVE_TO_PRE_GRASP":
            # Hover safely above object top for approach
            target_pos = np.array([
                cube_pos_world[0],
                cube_pos_world[1],
                object_top_z + self.cfg.hover_height
            ])
            self._target_pose_7d = np.concatenate([target_pos, self._downward_quat])
            self._gripper_action = -1.0  # Open
            if np.linalg.norm(ee_pos - target_pos) < self.cfg.pos_tolerance:
                self._advance_state("DESCEND_TO_GRASP")

        elif self._state == "DESCEND_TO_GRASP":
            # Precise descent to object top (CRITICAL FIX: Use object_top_z, not center_z)
            # Add small offset for finger clearance/attachment site alignment
            grasp_z = object_top_z - self.cfg.grasp_offset_z
            # Apply lateral XY offset to approach from side, reducing visual interpenetration
            descent_xy = cube_pos_world[:2] + self.cfg.descent_xy_offset[:2]
            target_pos = np.array([
                descent_xy[0],
                descent_xy[1],
                grasp_z
            ])
            self._target_pose_7d = np.concatenate([target_pos, self._downward_quat])
            self._gripper_action = -1.0  # Open during descent
            if np.linalg.norm(ee_pos - target_pos) < self.cfg.pos_tolerance:
                self._advance_state("GRASP")

        elif self._state == "GRASP":
            # Hold position and close gripper. Record initial object position.
            self._target_pose_7d = np.concatenate([ee_pos, self._downward_quat])
            self._gripper_action = self.object.grasp_width_normalized  # Close command issued here
            if self._wait_counter > 1:
                self.object_pos_pre_lift = cube_pos_world[2] # Record initial position
                self._advance_state("VERIFY_LIFT")

        elif self._state == "VERIFY_LIFT":
            # New state: attempt a small lift and verify object motion.
            # 1. Target a slightly higher position to initiate the lift attempt.
            target_z = self.object_pos_pre_lift + object_half_height + self.cfg.verify_lift_height
            target_pos = np.array([ee_pos[0], ee_pos[1], target_z])
            self._target_pose_7d = np.concatenate([target_pos, self._downward_quat])
            self._gripper_action = self.object.grasp_width_normalized # Keep closed

            # 2. Check conditions for successful grasp after moving for a few steps.
            object_has_lifted = cube_pos_world[2] > (self.object_pos_pre_lift + self.cfg.verify_lift_height / 2.0)
            
            # 3. Decision logic: Proceed if grasped AND object moved; retry otherwise.
            if self._wait_counter > 20: # Wait 20 steps for physical simulation to resolve lift.
                if is_grasped and object_has_lifted:
                    self._grasp_retry_count = 0  # Reset retries on success
                    self._advance_state("LIFT")
                else:
                    self._handle_grasp_failure()

        elif self._state == "LIFT":
            # Lift object by raising EE (object follows via kinematic offsets)
            # Use current object_top_z (should == ee_z post-grasp) + hover
            current_object_top_z = cube_pos_world[2] + object_half_height
            target_pos = np.array([
                ee_pos[0],
                ee_pos[1],
                current_object_top_z + self.cfg.hover_height
            ])
            self._target_pose_7d = np.concatenate([target_pos, self._downward_quat])
            self._gripper_action = self.object.grasp_width_normalized  # Keep closed
            if self._wait_counter > self.cfg.lift_duration_steps:
                self._advance_state("MOVE_TO_GOAL")

# ... rest of FSM logic remains the same ...