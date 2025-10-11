# FILE: utils/scripted_expert.py

# Add a new state variable
class ScriptedExpert:
    def __init__(...):
        # ...
        self._start_lift_object_pos: Optional[np.ndarray] = None # <-- ADD
        self.reset()
    def reset(...):
        # ...
        self._start_lift_object_pos = None # <-- ADD

# Replace the LIFT state
    def get_target_pose(...):
        # ...
        elif self._state == "LIFT":
            """
            Definitive ARC LIFT with self-contained verification to prevent race conditions.
            """
            # === STATE ENTRY LOGIC (runs only on the first step) ===
            if self._wait_counter == 1:
                # 1. Record the object's initial position AT THE START of the lift.
                self._start_lift_object_pos = cube_pos_world.copy()

                # 2. Calculate the kinematically stable "Arc Lift" target.
                start_lift_ee_pos = ee_pos.copy()
                adaptive_h = self.adaptive_hover_height(start_lift_ee_pos[:2], robot_base_pos_world[:2])
                reach_vector_xy = start_lift_ee_pos[:2] - robot_base_pos_world[:2]
                retract_fraction = 0.20
                retract_offset_xy = -reach_vector_xy * retract_fraction
                
                self._lift_target_pos = np.array([
                    start_lift_ee_pos[0] + retract_offset_xy[0],
                    start_lift_ee_pos[1] + retract_offset_xy[1],
                    start_lift_ee_pos[2] + adaptive_h
                ])
                self._lift_target_pos[2] = min(self._lift_target_pos[2], self.cfg.max_lift_height)

            # === CONTINUOUS LOGIC ===
            self._target_pose_7d = np.concatenate([self._lift_target_pos, self.current_grasp_orientation])
            self._gripper_action = -1.0

            # === STABILITY & TRANSITION LOGIC ===
            
            # Use the locally-stored start position for a reliable check.
            is_object_lifted = (cube_pos_world[2] - self._start_lift_object_pos[2]) > self.cfg.verify_lift_height
            is_still_grasped = is_grasped

            if self._wait_counter > 15:
                if not is_still_grasped:
                    print(f"DEBUG: Grasp failure detected in LIFT (is_grasped=False).")
                    self._handle_failure()
                elif is_still_grasped and not is_object_lifted:
                     print(f"DEBUG: Grasp failure detected in LIFT (object not lifted).")
                     self._handle_failure()

            if self._wait_counter > self.cfg.lift_duration_steps:
                if is_still_grasped and is_object_lifted:
                    self._advance_state("MOVE_TO_GOAL")
                else: # If lift duration is over and we failed, it's a failure.
                    print("DEBUG: Grasp failed to lift object by end of LIFT state. Retrying.")
                    self._handle_failure()


       N_SUBSTEPS = 5
        for _ in range(N_SUBSTEPS):
            # ...
            touch_threshold = 0.005
            force_threshold = 1.0    # In Newtons (N)

            # --- START OF THE ROBUST GRASP LOGIC FIX ---
            
            # Condition 1: Is there a command to grip?
            is_gripping_command = gripper_action < 0.1

            # Condition 2: Is there STABLE physical contact on both fingers?
            has_bilateral_contact = (left_touch_val > touch_threshold) and (right_touch_val > touch_threshold)
            
            # Condition 3: Is there sufficient force ON BOTH fingers?
            # This is much stricter and prevents false positives from one-sided contact.
            has_sufficient_bilateral_force = (np.linalg.norm(left_force_vec) > force_threshold) and \
                                             (np.linalg.norm(right_force_vec) > force_threshold)

            # THE NEW, ROBUST FINAL CONDITION:
            # We must have the command, AND we must have contact, AND we must have force.
            # This prevents force from being counted when there is no contact.
            is_grasp_conditions_met = (
                is_gripping_command and 
                has_bilateral_contact and 
                has_sufficient_bilateral_force
            )
            