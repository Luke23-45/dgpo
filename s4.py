# FILE: utils/scripted_expert.py

        elif self._state == "PREPARE_PLACE":
            """
            Definitive hover pose calculation.
            """
            if self._wait_counter == 1:
                # Get the full sizes of the object and goal
                goal_size = expert_obs["goal_size_world"]
                object_size = self.object.size

                x_offset = (goal_size[0] - object_size[0]) / 2.0
                y_offset = (goal_size[1] - object_size[1]) / 2.0
                
                # The target XY for the OBJECT is the goal's center offset for edge-alignment.
                object_target_xy = np.array([
                    goal_pos_world[0] - x_offset,
                    goal_pos_world[1] - y_offset
                ])
                
                # The hover height for the OBJECT'S CENTER.
                object_hover_z = self.table_surface_z + object_half_height + self.cfg.hover_height

                # To achieve this, the TCP must be positioned above the object's hover position.
                tcp_hover_pos = np.array([
                    object_target_xy[0],
                    object_target_xy[1],
                    object_hover_z + object_half_height
                ])

                goal_orn_world = expert_obs["goal_orn_world"]
                target_orn = self._calculate_aligned_orientation(
                    goal_orn_world, ee_pose_world[3:]
                )
                
                # Store this definitive 7D hover pose.
                self._final_hover_pose = np.concatenate([tcp_hover_pos, target_orn])
                self._orientation_stable_counter = 0

            self._target_pose_7d = self._final_hover_pose
            self._gripper_action = -1.0

            pos_error = np.linalg.norm(ee_pos - self._final_hover_pose[:3])
            R_current = R.from_quat(ee_pose_world[3:])
            R_target = R.from_quat(self._final_hover_pose[3:])
            angular_distance = (R_target.inv() * R_current).magnitude()
            is_at_pose = (pos_error < self.cfg.pos_tolerance) and (angular_distance < self.cfg.orn_tolerance_rad)

            if is_at_pose: self._orientation_stable_counter += 1
            else: self._orientation_stable_counter = 0
            
            is_stable = self._orientation_stable_counter > 5
            is_timed_out = self._wait_counter > 60

            if is_stable or is_timed_out:
                if is_timed_out: print("WARN: PREPARE_PLACE timed out.")
                self.current_grasp_orientation = self._final_hover_pose[3:].copy()
                self._advance_state("DESCEND_TO_PLACE")

        elif self._state == "DESCEND_TO_PLACE": 
            """
            Descends from the known hover pose to the final placement pose.
            """
            if self._wait_counter == 1:
                # The hover pose is our starting point.
                hover_pos = self._final_hover_pose[:3]
                # The final placement pose is just the hover pose moved down vertically.
                place_pos = hover_pos.copy()
                place_pos[2] -= self.cfg.hover_height
                self._final_place_pose = np.concatenate([place_pos, self.current_grasp_orientation])

            self._target_pose_7d = self._final_place_pose
            self._gripper_action = -1.0 

            if np.linalg.norm(ee_pos - self._final_place_pose[:3]) < self.cfg.pos_tolerance:
                self._advance_state("AWAIT_PLACEMENT_CONTACT")
        
        elif self._state == "AWAIT_PLACEMENT_CONTACT":
            """
            Holds the definitive final placement pose and waits for physical contact.
            """
            # Command the robot to hold the FINAL, correct placement pose.
            self._target_pose_7d = self._final_place_pose
            self._gripper_action = -1.0 

            object_is_supported = abs(expert_obs.get("object_vel", [0,0,0,0,0,0])[2]) < 0.005

            if self._wait_counter > 5 and object_is_supported:
                self._advance_state("RELEASE")
            elif self._wait_counter > 25:
                print("WARN: AWAIT_PLACEMENT_CONTACT timed out. Forcing release.")
                self._advance_state("RELEASE")