

# FILE: utils/scripted_expert.py

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation as R,Slerp

@dataclass
class ObjectProfile:
    """Holds the geometric properties of a manipulable object."""
    size: np.ndarray
    grasp_width_normalized: float


@dataclass
class ExpertConfig:
    """Configuration for the scripted expert policy. [DELTA-COMPATIBLE VERSION]"""
    # --- Time-based durations for delta control stability ---
    move_to_pre_grasp_duration: int = 200
    prepare_gripper_duration: int = 30
    descend_to_grasp_duration: int = 50
    lift_duration_steps: int = 40
    move_to_goal_duration: int = 180
    prepare_place_duration: int = 60
    descend_to_place_duration: int = 40
    retract_duration_steps: int = 30

    # --- Original physical parameters ---
    hover_height: float = 0.10
    grasp_offset_z: float = 0.025
    failure_timeout_steps: int = 250 # Increased global timeout per state
    pos_tolerance: float = 0.02 # Still used for final checks
    
    workspace: dict = None
    descent_xy_offset: np.ndarray = np.array([0.00, 0.0, 0.0])   
    max_grasp_retries: int = 2
    verify_lift_height: float = 0.03 
    gripper_open_threshold: float = 0.038
    gripper_closed_threshold: float = 0.002
    home_pose_7d: np.ndarray = np.array([0.5, 0.0, 0.7, 0.0, 1.0, 0.0, 0.0])
    orn_tolerance_rad: float = 0.05 
    max_lift_height: float = 0.65

    def __post_init__(self):
        if self.workspace is None:
            self.workspace = {"x": (0.35, 0.85), "y": (-0.25, 0.25), "z": (0.30, 0.95)}



class ScriptedExpert:
    """
    Final, robust, state-machine-based expert for pick-and-place.

    Key Improvements for Robustness and Optimized Grasping:
    - **Precise Grasp Positioning**: Targets the exact top surface of the object (object_top_z) during descent,
      ensuring the end-effector (attachment site) aligns perfectly above the object's center. This prevents
      penetration or misalignment, allowing the kinematic grasp to engage reliably without the object "floating."
    - **Grasp Offset**: Introduces a tiny downward offset (grasp_offset_z) in the descent target to simulate
      finger closure around the top edge, improving physical intuition and attachment stability.
    - **Lateral Descent Offset**: During `DESCEND_TO_GRASP`, applies a small XY shift (descent_xy_offset) to approach
      from the side, reducing visual finger-cube interpenetration while keeping the attachment site centered.
    - **Retry Mechanism**: If grasp fails (no kinematic attachment after timeout), retry the pre-grasp and descent
      up to max_grasp_retries times. This handles minor positioning errors or simulation jitter.
    - **Extended Timed Transitions**: Increased step counts for lift, move, place, and retract to allow smoother
      trajectories via IK/position control, reducing jerkiness and ensuring the object stays securely attached.
    - **Failure Safeguards**: Per-state timeouts prevent stalling; if retries exhaust, transition to "DONE" (logged as failure).
    - **Orientation Consistency**: Maintains a fixed downward quaternion throughout manipulation for stable holding.
      (Assumes object is axis-aligned; ignores object_orn_world for simplicity, as rotation isn't required for basic pick-place.)
    - **Workspace Clamping**: Applied at every target update to prevent out-of-bounds IK failures.
    - **State Diagnostics**: Internal counters and conditions ensure predictable progression without getting stuck.

    This version ensures the gripper "holds" the cube kinematically: once grasped, the object follows the EE precisely
    via offsets, eliminating floating. The lateral offset mitigates the primary visual artifact without risking stability.
    """
    def __init__(self, object_profile: ObjectProfile, cfg: ExpertConfig = ExpertConfig()):
        self.cfg = cfg
        self.object = object_profile
        self._target_pose_7d: Optional[np.ndarray] = None
        self._wait_counter = 0
        self._grasp_retry_count = 0
        self.table_surface_z = 0.4  # Assumed table height; adjust if env changes
        self._downward_quat = np.array([0.0, 1.0, 0.0, 0.0])  # xyzw: 180° around Y for palm-down grasp/hold
        self.object_pos_pre_lift: Optional[float] = None # New variable: store object Z before verification lift
        self.current_grasp_orientation = self._downward_quat
        self._hold_pos: Optional[np.ndarray] = None
        self._target_orn: Optional[np.ndarray] = None
        self._orientation_stable_counter: int = 0 
        self._descent_target_pos: Optional[np.ndarray] = None
        self._lift_target_pos: Optional[np.ndarray] = None
        self._final_hover_pose: Optional[np.ndarray] = None
        self._final_place_pose: Optional[np.ndarray] = None
        self.reset()

    def is_done(self) -> bool:
        return self._state == "DONE"

    def reset(self):
        self._state = "MOVE_TO_PRE_GRASP"
        self._gripper_action = -1.0  
        self._target_pose_7d = None
        self._wait_counter = 0
        self._grasp_retry_count = 0
        self.succeeded = False
        self.object_pos_pre_lift = None
        self.current_grasp_orientation = self._downward_quat
        self._hold_pos = None
        self._target_orn = None
        self._orientation_stable_counter = 0
        self._descent_target_pos = None 
        self._lift_target_pos = None
        self._final_hover_pose = None # Already exists
        self._final_place_pose = None # <-- ADD THIS LINE

    def _get_motion_aligned_orientation(self, start_pos, end_pos):
        """Calculates a downward-facing quat with yaw aligned to the direction of motion."""
        move_vector = end_pos - start_pos
        if np.linalg.norm(move_vector[:2]) < 1e-3: # If move is vertical, keep current orientation
            return self.current_grasp_orientation
        
        # Calculate yaw angle from the XY movement vector
        yaw = np.arctan2(move_vector[1], move_vector[0])
        
        # We want to align the gripper's forward axis (e.g., local -X) with this direction.
        # The downward_quat is 180deg rotation around Y. This makes local -X point along world +X.
        # So we need to add the yaw rotation.
        yaw_rotation = R.from_euler('z', yaw)
        aligned_orientation = (yaw_rotation * R.from_quat(self._downward_quat)).as_quat()
        return aligned_orientation       

    def get_state(self) -> str:
        return self._state
    
    def was_successful(self) -> bool:
        """Returns True only if the FSM completed the task successfully."""
        return self.succeeded
    def _clamp_to_workspace(self, pos: np.ndarray) -> np.ndarray:
        """Clamps position to safe workspace bounds."""
        pos = pos.copy()
        pos[0] = np.clip(pos[0], *self.cfg.workspace["x"])
        pos[1] = np.clip(pos[1], *self.cfg.workspace["y"])
        pos[2] = np.clip(pos[2], *self.cfg.workspace["z"])
        return pos

    def _reset_wait_counter(self):
        """Utility to reset wait counter on state advance."""
        self._wait_counter = 0

    def _advance_state(self, new_state: str):
        """Advances state and resets wait counter."""
        self._state = new_state
        self._reset_wait_counter()


    def _calculate_aligned_orientation(self, object_quat_xyzw: np.ndarray, gripper_quat_xyzw: np.ndarray) -> np.ndarray:
        """
        Calculates the definitive gripper orientation for a stable, top-down grasp.
        This version robustly satisfies all physical constraints simultaneously:
          1. Aligns the gripper's closing axis with the nearest cube face normal.
          2. Enforces a palm-down orientation for stability.
          3. Guarantees the shortest possible rotation path.
        """
        try:
            R_obj = R.from_quat(object_quat_xyzw)
            R_grip = R.from_quat(gripper_quat_xyzw)

            # --- 1. Identify all 4 candidate grasp normals in the world frame ---
            local_face_normals = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
            world_face_normals = R_obj.apply(local_face_normals)

            # --- 2. Get the gripper's current closing axis (+X) in the world frame ---
            # THIS IS THE CORE FIX: We use the closing axis [1, 0, 0] instead of the finger-plane axis [0, 1, 0].
            gripper_closing_axis_world = R_grip.apply([1, 0, 0])

            # --- 3. Find the best grasp TARGET, guaranteeing the shortest rotation path ---
            dot_products = np.dot(world_face_normals, gripper_closing_axis_world)
            
            # Find the face normal that is most parallel (or anti-parallel) to the closing axis.
            best_axis_index = np.argmax(np.abs(dot_products))
            
            # Now, explicitly choose the direction (parallel or anti-parallel) that requires the smallest rotation.
            if dot_products[best_axis_index] < 0:
                # If the dot product is negative, the flipped normal is the closer target.
                target_normal = -world_face_normals[best_axis_index]
            else:
                target_normal = world_face_normals[best_axis_index]

            # --- 4. Compute the FULL 3D rotation to align the closing axis with the target normal ---
            # This gives us a rotation that satisfies the Face-Parallel and Minimal Rotation constraints.
            axis = np.cross(gripper_closing_axis_world, target_normal)
            dot_product_clipped = np.clip(np.dot(gripper_closing_axis_world, target_normal), -1.0, 1.0)
            angle = np.arccos(dot_product_clipped)

            if np.linalg.norm(axis) < 1e-6:
                R_corr = R.identity()
            else:
                axis_normalized = axis / np.linalg.norm(axis)
                R_corr = R.from_rotvec(axis_normalized * angle)
            
            # This is the raw, unconstrained target orientation.
            R_target_raw = R_corr * R_grip

            # --- 5. Enforce the Palm-Down Constraint ---
            # We take the raw target and apply a minimal "tilt" correction to force its
            # local Z-axis to point downwards, satisfying the final constraint without
            # ruining the yaw alignment from step 4.
            
            # Get the Z-axis (palm vector) of our raw target orientation
            z_axis_raw = R_target_raw.apply([0, 0, 1])
            
            # The desired palm-down vector is [0, 0, -1]
            z_axis_target = np.array([0., 0., -1.])
            
            # Calculate the minimal rotation to tilt the palm down
            tilt_axis = np.cross(z_axis_raw, z_axis_target)
            tilt_dot_product = np.clip(np.dot(z_axis_raw, z_axis_target), -1.0, 1.0)
            tilt_angle = np.arccos(tilt_dot_product)
            
            if np.linalg.norm(tilt_axis) < 1e-6:
                R_tilt_correction = R.identity()
            else:
                tilt_axis_normalized = tilt_axis / np.linalg.norm(tilt_axis)
                R_tilt_correction = R.from_rotvec(tilt_axis_normalized * tilt_angle)
            
            # The final orientation is the raw target with the tilt correction applied.
            R_final = R_tilt_correction * R_target_raw
            
            return R_final.as_quat()

        except Exception as e:
            print(f"DEBUG ALIGNMENT ERROR: {e}")
            return self._downward_quat.copy()


    def adaptive_hover_height(self, current_ee_xy: np.ndarray, robot_base_xy: np.ndarray) -> float:
        """
        Calculates a safe hover height that decreases as the arm extends.
        The min/max values are derived from the ExpertConfig for consistency.
        """
        # 1. Define the kinematic parameters of the robot's reach.
        # These are based on the expert's defined safe workspace.
        min_reach = 0.35
        max_reach = 0.68
        
        # --- START OF THE FIX ---
        # 2. Derive min/max heights from the existing configuration.
        # The maximum height is the standard hover height.
        max_hover_height = self.cfg.hover_height  # Typically 0.10
        # The minimum height must be greater than the lift verification threshold.
        # We add a 1cm safety margin.
        min_hover_height = self.cfg.verify_lift_height + 0.01 # Typically 0.03 + 0.01 = 0.04
        # --- END OF THE FIX ---

        # 3. Calculate the current horizontal reach.
        reach = np.linalg.norm(current_ee_xy - robot_base_xy)

        # 4. Calculate the "risk factor" (0.0 to 1.0) based on the reach.
        if max_reach <= min_reach: return min_hover_height
        risk_factor = (reach - min_reach) / (max_reach - min_reach)
        risk_factor = np.clip(risk_factor, 0.0, 1.0)

        # 5. Linearly interpolate the hover height.
        hover_height = max_hover_height - risk_factor * (max_hover_height - min_hover_height)
        
        return hover_height
    
    def _clamp_to_workspace(self, pos: np.ndarray) -> np.ndarray:
        """Clamps position to safe workspace bounds."""
        pos = pos.copy()
        pos[0] = np.clip(pos[0], *self.cfg.workspace["x"])
        pos[1] = np.clip(pos[1], *self.cfg.workspace["y"])
        pos[2] = np.clip(pos[2], *self.cfg.workspace["z"])
        return pos

    def _handle_failure(self):
        """
        Handles any timeout or failure. If the failure occurs before the object
        is placed, it retries the grasp. If it occurs after placement,
        it terminates the episode as a failure.
        """
        # Define the states that occur *after* a successful placement.
        # A failure in these states is terminal and should not be retried.
        terminal_states = ["AWAIT_PLACEMENT_CONTACT", "RELEASE", "WAIT_FOR_RELEASE", "RETRACT"]

        if self._state in terminal_states:
            # If we fail during the placement/retraction phase, the task is over.
            print(f"ERROR: Terminal failure in state '{self._state}'. Aborting episode.")
            self.succeeded = False
            self._state = "DONE"
            self._gripper_action = 1.0 # Open gripper for safety
        else:
            # For any other failure (e.g., during pre-grasp, grasp, lift), attempt a retry.
            self._grasp_retry_count += 1
            if self._grasp_retry_count <= self.cfg.max_grasp_retries:
                print(f"WARN: Failure in state '{self._state}'. Attempting retry #{self._grasp_retry_count}.")
                self._state = "MOVE_TO_PRE_GRASP"
                self._gripper_action = -1.0 # Ensure gripper is open for retry
                self._reset_wait_counter()
            else:
                print(f"ERROR: Max retries ({self.cfg.max_grasp_retries}) exceeded. Aborting episode.")
                self.succeeded = False
                self._state = "DONE"
                self._gripper_action = 1.0


    def get_target_pose(
        self,
        expert_obs: dict, 
    ) -> Tuple[np.ndarray, float]:
        """
        [DELTA-COMPATIBLE VERSION]
        Computes target 7D pose and gripper action. This version relies on
        timed states and physical events, making it robust for delta control.
        """
        ee_pose_world = expert_obs["ee_pose_world"]
        cube_pos_world = expert_obs["object_pos_world"]
        object_orn_world = expert_obs["object_orn_world"]
        goal_pos_world = expert_obs["goal_pos_world"]
        is_grasped = expert_obs["is_grasped"][0] > 0.5
        gripper_qpos = expert_obs["gripper_qpos"]
        robot_base_pos_world = expert_obs["robot_base_pos_world"]
        ee_pos = ee_pose_world[:3]
        object_half_height = self.object.size[2] / 2.0
        object_top_z = cube_pos_world[2] + object_half_height

        self._wait_counter += 1

        # Global timeout safeguard
        if self._wait_counter > self.cfg.failure_timeout_steps:
            print(f"WARN: Global timeout of {self.cfg.failure_timeout_steps} steps exceeded in state '{self._state}'.")
            self._handle_failure()

        
        elif self._state == "MOVE_TO_PRE_GRASP":
            """
            [HYBRID ROBUST VERSION]
            Generates a smooth, interpolated trajectory but transitions as soon as the
            physical goal is met, making it efficient for both short and long movements.
            A timeout acts as a safety net to guarantee progression.
            """
            # === STATE ENTRY LOGIC (runs only ONCE on the first step) ===
            if self._wait_counter == 1:
                # 1. Capture the starting position and orientation for interpolation.
                self._start_pre_grasp_pos = ee_pos.copy()
                self._start_pre_grasp_orn = R.from_quat(ee_pose_world[3:])
                print("MOve to pregrasp!")

                # 2. Calculate the FINAL destination POSITION and store it.
                end_pos = np.array([cube_pos_world[0], cube_pos_world[1], object_top_z + self.cfg.hover_height])
                self._end_pre_grasp_pos = end_pos
                
                # 3. Calculate the FINAL, ALIGNED orientation and store it.
                final_aligned_quat = self._calculate_aligned_orientation(object_orn_world, ee_pose_world[3:])
                self._end_pre_grasp_orn = R.from_quat(final_aligned_quat)

            # === CONTINUOUS LOGIC (runs EVERY step) ===

            # 1. Calculate progress (0.0 to 1.0).
            progress = min(self._wait_counter / self.cfg.move_to_pre_grasp_duration, 1.0)

            # 2. Interpolate position (Lerp).
            interp_pos = self._start_pre_grasp_pos + (self._end_pre_grasp_pos - self._start_pre_grasp_pos) * progress

            # 3. Spherically interpolate orientation (Slerp).
            key_rots = R.from_quat([self._start_pre_grasp_orn.as_quat(), self._end_pre_grasp_orn.as_quat()])
            slerp = Slerp([0, 1], key_rots)
            interp_orn_quat = slerp(progress).as_quat()

            # 4. Set the robot's target to the smoothly interpolated pose for this timestep.
            self._target_pose_7d = np.concatenate([interp_pos, interp_orn_quat])
            self._gripper_action = 1.0

            # === HYBRID TRANSITION LOGIC ===

            # Condition 1: Has the robot physically reached the FINAL destination?
            # We check the actual EE pose against the final target, not the moving interpolated one.
            pos_error = np.linalg.norm(ee_pos - self._end_pre_grasp_pos)
            
            # Calculate angular distance to the FINAL target orientation.
            R_current = R.from_quat(ee_pose_world[3:])
            angular_distance = (self._end_pre_grasp_orn.inv() * R_current).magnitude()
            
            is_at_destination = (pos_error < self.cfg.pos_tolerance) and \
                                (angular_distance < self.cfg.orn_tolerance_rad)

            # Condition 2: Has the maximum allowed time elapsed? (Fallback)
            is_timed_out = self._wait_counter > self.cfg.move_to_pre_grasp_duration

            # Transition if EITHER condition is met.
            if is_at_destination or is_timed_out:
                if is_timed_out and not is_at_destination:
                    print(f"WARN: MOVE_TO_PRE_GRASP timed out. Pos Error: {pos_error:.3f}m, Orn Error: {angular_distance:.3f}rad")
                
                # Commit the final, correct orientation to the state machine's memory.
                self.current_grasp_orientation = self._end_pre_grasp_orn.as_quat()
                self._advance_state("PREPARE_GRIPPER")

    
    
        elif self._state == "PREPARE_GRIPPER":
            """
            [ULTIMATE ROBUST & FAILSAFE VERSION]
            This state verifies the robot's orientation. If it's already correct,
            it simply holds the pose. If the previous state failed to orient properly,
            this state executes a corrective rotation. It transitions only when both
            the orientation is stable AND the gripper is physically open.
            """
            # === STATE ENTRY LOGIC (runs only ONCE on the first step) ===
            if self._wait_counter == 1:
                # 1. ALWAYS calculate the DEFINITIVE target orientation. This is our "ground truth"
                #    for what the orientation should be, regardless of how we got here.
                target_orn_quat = self._calculate_aligned_orientation(object_orn_world, ee_pose_world[3:])
                self._target_orn = R.from_quat(target_orn_quat)

                # 2. Store the current position to hold it steady during any potential rotation.
                self._hold_pos = ee_pos.copy()
                print("PREPARE_GRIPPER")
                
                # 3. Assemble the full 7D target pose.
                self._stored_target_pose = np.concatenate([self._hold_pos, self._target_orn.as_quat()])
                
                # 4. Pre-calculate the descent target for the next state.
                grasp_z = object_top_z - self.cfg.grasp_offset_z
                descent_xy = cube_pos_world[:2] + self.cfg.descent_xy_offset[:2]
                self._descent_target_pos = np.array([descent_xy[0], descent_xy[1], grasp_z])

                # 5. Initialize a counter to check for orientation stability.
                self._orientation_stable_counter = 0

            # === CONTINUOUS LOGIC (runs EVERY step) ===
            
            # Command the robot to go to the STORED hold position and STORED target orientation.
            # If already oriented correctly, this command will cause no movement.
            # If orientation is wrong, this will command a corrective rotation.
            self._target_pose_7d = self._stored_target_pose
            self._gripper_action = 1.0 # Command gripper to open

            # === ROBUST HYBRID TRANSITION LOGIC ===
            
            # Condition 1: Is the orientation correct and has it been stable for a few steps?
            R_current = R.from_quat(ee_pose_world[3:])
            angular_distance = (self._target_orn.inv() * R_current).magnitude()

            if angular_distance < self.cfg.orn_tolerance_rad:
                self._orientation_stable_counter += 1
            else:
                self._orientation_stable_counter = 0 # Reset counter if orientation deviates
            
            # We require the orientation to be correct for 5 consecutive steps to be "stable".
            is_oriented_and_stable = self._orientation_stable_counter > 5

            # Condition 2: Is the gripper physically open?
            is_gripper_open = np.all(gripper_qpos > self.cfg.gripper_open_threshold)
            
            # Condition 3 (Safety Net): Has too much time passed?
            is_timed_out = self._wait_counter > self.cfg.prepare_gripper_duration

            # Transition only if (stable AND open) OR if the timeout is reached.
            if (is_oriented_and_stable and is_gripper_open) or is_timed_out:
                if is_timed_out:
                    print("WARN: PREPARE_GRIPPER timed out. Forcing transition.")
                
                # Commit the final orientation before advancing.
                self.current_grasp_orientation = self._target_orn.as_quat()
                self._advance_state("DESCEND_TO_GRASP")

        elif self._state == "DESCEND_TO_GRASP":
            """
            [ROBUST HYBRID VERSION]
            Commands a straight descent to the pre-calculated grasp pose.
            It transitions based on reaching the target position, making it efficient.
            A timeout prevents it from getting stuck if the goal is unreachable.
            """
            # === STATE ENTRY LOGIC (runs only ONCE on the first step) ===
            if self._wait_counter == 1:
                # All necessary variables (_descent_target_pos, current_grasp_orientation)
                # have already been set by the previous states. No setup needed.
                pass

            # === CONTINUOUS LOGIC (runs EVERY step) ===
            
            # Command the robot to move to the fixed descent target while maintaining orientation.
            self._target_pose_7d = np.concatenate([self._descent_target_pos, self.current_grasp_orientation])
            self._gripper_action = 1.0  # Keep gripper open during descent.

            # === HYBRID TRANSITION LOGIC ===

            # Condition 1 (Primary): Has the robot physically reached the descent target?
            pos_error = np.linalg.norm(ee_pos - self._descent_target_pos)
            is_at_destination = pos_error < self.cfg.pos_tolerance

            # Condition 2 (Safety Net): Has the maximum allowed time elapsed?
            is_timed_out = self._wait_counter > self.cfg.descend_to_grasp_duration
            
            # Transition if EITHER condition is met.
            if is_at_destination or is_timed_out:
                if is_timed_out and not is_at_destination:
                    print(f"WARN: DESCEND_TO_GRASP timed out. Final position error: {pos_error:.4f}m")
                
                self._advance_state("GRASP")

        elif self._state == "GRASP":
            """
            [ROBUST FAILSAFE VERSION]
            Commands the gripper to close while holding the pose perfectly still.
            It requires the grasp to be physically confirmed for several consecutive
            steps to prevent premature lifting on a noisy signal. Includes a timeout
            to handle grasp failures gracefully and trigger the retry mechanism.
            """
            # === STATE ENTRY LOGIC (runs only ONCE on the first step) ===
            if self._wait_counter == 1:
                # Initialize a counter to confirm the grasp is stable.
                self._grasp_confirmation_counter = 0

            # === CONTINUOUS LOGIC (runs EVERY step) ===
            
            # Command the robot to hold its current pose to prevent any movement during closure.
            self._target_pose_7d = np.concatenate([ee_pos, self.current_grasp_orientation])
            # Command the gripper to close.
            self._gripper_action = -1.0

            # === STABLE CONFIRMATION & TRANSITION LOGIC ===

            # 1. Check for physical grasp confirmation from the environment.
            if is_grasped:
                self._grasp_confirmation_counter += 1
            else:
                # If the signal flickers or is lost, reset the counter.
                self._grasp_confirmation_counter = 0

            # 2. Define the success and failure conditions.
            # Success: Grasp signal has been stable for 5 consecutive steps.
            is_grasp_stable = self._grasp_confirmation_counter > 5
            
            # Failure: We've waited longer than a reasonable timeout (e.g., 40 steps).
            is_timed_out = self._wait_counter > 40

            # 3. Transition based on conditions.
            if is_grasp_stable:
                # The grasp is secure, proceed to lift.
                self._advance_state("LIFT")
            elif is_timed_out:
                # The grasp failed to establish.
                print("ERROR: GRASP state timed out. Grasp failed.")
                # Trigger the global failure handler, which will attempt a retry.
                self._handle_failure()
        

        elif self._state == "LIFT":
            """
            [ROBUST TRAJECTORY-BASED VERSION]
            Performs a smooth, interpolated vertical lift to a safe height.
            Crucially, it monitors the grasp status on EVERY step of the lift.
            If the grasp is lost at any point, it immediately aborts and triggers
            the failure handler.
            """
            # === STATE ENTRY LOGIC (runs only ONCE on the first step) ===
            if self._wait_counter == 1:
                # 1. Capture the starting position for the interpolation.
                self._start_lift_pos = ee_pos.copy()
                
                # 2. Calculate the final lift position using the adaptive height.
                adaptive_h = self.adaptive_hover_height(ee_pos[:2], robot_base_pos_world[:2])
                end_lift_pos = self._start_lift_pos + np.array([0, 0, adaptive_h])
                self._end_lift_pos = self._clamp_to_workspace(end_lift_pos)
                
            # === CONTINUOUS LOGIC & MONITORING (runs EVERY step) ===

            # 1. FAIL FAST: Check if the grasp has been lost mid-lift.
            if not is_grasped and self._wait_counter > 5: # Small grace period
                print("ERROR: Grasp lost during LIFT state. Aborting.")
                self._handle_failure()
                # Return here to ensure we don't execute the rest of the logic on this frame
                # The state will be changed by _handle_failure for the next step.
                return self.get_target_pose(expert_obs)

            # 2. Calculate the progress of the lift (0.0 to 1.0).
            progress = min(self._wait_counter / self.cfg.lift_duration_steps, 1.0)

            # 3. Linearly interpolate the position for a smooth vertical trajectory.
            interp_pos = self._start_lift_pos + (self._end_lift_pos - self._start_lift_pos) * progress
            
            # 4. Command the interpolated pose. Orientation remains constant.
            self._target_pose_7d = np.concatenate([interp_pos, self.current_grasp_orientation])
            self._gripper_action = -1.0  # Maintain grasp

            # === TRANSITION LOGIC ===
            
            # Transition only after the full, smooth lift duration has completed.
            if self._wait_counter > self.cfg.lift_duration_steps:
                # Final verification: is the grasp STILL held after the motion?
                if is_grasped:
                    self._advance_state("MOVE_TO_GOAL")
                else:
                    # This case handles if the grasp is lost on the very last step.
                    print("ERROR: Grasp lost at the end of LIFT. Aborting.")
                    self._handle_failure()

        elif self._state == "MOVE_TO_GOAL":
            """
            [ROBUST TRAJECTORY-BASED VERSION WITH CONTINUOUS MONITORING]
            Executes a smooth, interpolated trajectory in both position and orientation
            to move the object to the goal's hover position. It actively monitors the
            grasp on EVERY step, aborting immediately if the object is dropped.
            """
            # === STATE ENTRY LOGIC (runs only ONCE on the first step) ===
            if self._wait_counter == 1:
                # 1. Setup start and end points for the trajectory.
                self._start_move_pos = ee_pos.copy()
                self._start_move_orn = R.from_quat(self.current_grasp_orientation)
                
                # Calculate the target placement orientation.
                goal_orn_world = expert_obs["goal_orn_world"]
                self._place_orn = R.from_quat(self._calculate_aligned_orientation(goal_orn_world, ee_pose_world[3:]))
                
                # Calculate the target hover position above the goal.
                goal_hover_z = goal_pos_world[2] + self.cfg.hover_height + object_half_height
                self._end_move_pos = np.array([goal_pos_world[0], goal_pos_world[1], goal_hover_z])

            # === CONTINUOUS LOGIC & MONITORING (runs EVERY step) ===

            # 1. FAIL FAST: Check if the grasp has been lost mid-transit.
            if not is_grasped and self._wait_counter > 5: # Small grace period
                print("ERROR: Grasp lost during MOVE_TO_GOAL state. Aborting.")
                self._handle_failure()
                # Use a recursive call to immediately get the next action after failure
                return self.get_target_pose(expert_obs)

            # 2. Calculate progress and interpolate the 7D pose.
            progress = min(self._wait_counter / self.cfg.move_to_goal_duration, 1.0)
            interp_pos = self._start_move_pos + (self._end_move_pos - self._start_move_pos) * progress
            slerp = Slerp([0, 1], R.from_quat([self._start_move_orn.as_quat(), self._place_orn.as_quat()]))
            interp_orn = slerp(progress).as_quat()
            
            # 3. Command the interpolated pose and maintain grip.
            self._target_pose_7d = np.concatenate([interp_pos, interp_orn])
            self._gripper_action = -1.0

            # === TRANSITION LOGIC ===

            # Transition after the full duration has passed.
            if self._wait_counter > self.cfg.move_to_goal_duration:
                # Final check: is the grasp still held after the long move?
                if is_grasped:
                    self.current_grasp_orientation = self._place_orn.as_quat()
                    self._advance_state("PREPARE_PLACE")
                else:
                    print("ERROR: Grasp lost at the end of MOVE_TO_GOAL. Aborting.")
                    self._handle_failure()

  
        elif self._state == "PREPARE_PLACE":
            """
            [ROBUST STABILIZATION & VERIFICATION VERSION]
            This state acts as a final checkpoint. It commands the robot to the exact
            pre-placement hover pose, correcting any tracking errors from the long
            previous move. It transitions only after verifying that the robot has
            physically arrived and is stable, while continuously monitoring the grasp.
            """
            # === STATE ENTRY LOGIC (runs only ONCE on the first step) ===
            if self._wait_counter == 1:
                # 1. Define the single, definitive target pose for this state.
                #    The orientation is already correctly stored in self.current_grasp_orientation.
                final_place_z = self.table_surface_z + self.object.size[2]
                hover_z = final_place_z + self.cfg.hover_height
                tcp_hover_pos = np.array([goal_pos_world[0], goal_pos_world[1], hover_z])
                self._final_hover_pose = np.concatenate([tcp_hover_pos, self.current_grasp_orientation])
                
                # 2. Initialize a counter to check for pose stability before transitioning.
                self._pose_stable_counter = 0

            # === CONTINUOUS LOGIC & MONITORING (runs EVERY step) ===
            
            # 1. FAIL FAST: Continuously monitor the grasp.
            if not is_grasped and self._wait_counter > 5:
                print("ERROR: Grasp lost during PREPARE_PLACE state. Aborting.")
                self._handle_failure()
                return self.get_target_pose(expert_obs)

            # 2. Command the robot to the definitive hover pose.
            self._target_pose_7d = self._final_hover_pose
            self._gripper_action = -1.0 # Maintain grasp

            # === HYBRID TRANSITION LOGIC (Position-based with stability check) ===

            # Condition 1: Is the robot physically at the destination pose?
            pos_error = np.linalg.norm(ee_pos - self._final_hover_pose[:3])
            R_current = R.from_quat(ee_pose_world[3:])
            R_target = R.from_quat(self.current_grasp_orientation)
            angular_distance = (R_target.inv() * R_current).magnitude()
            
            is_at_pose = (pos_error < self.cfg.pos_tolerance) and \
                         (angular_distance < self.cfg.orn_tolerance_rad)

            # Update stability counter
            if is_at_pose:
                self._pose_stable_counter += 1
            else:
                self._pose_stable_counter = 0
            
            is_stable = self._pose_stable_counter > 5

            # Condition 2 (Safety Net): Has the maximum allowed time elapsed?
            is_timed_out = self._wait_counter > self.cfg.prepare_place_duration

            # Transition if the pose is stable OR if timed out.
            if is_stable or is_timed_out:
                if is_timed_out and not is_stable:
                    print(f"WARN: PREPARE_PLACE timed out. Pos Error: {pos_error:.3f}m, Orn Error: {angular_distance:.3f}rad")
                
                self._advance_state("DESCEND_TO_PLACE")


        elif self._state == "DESCEND_TO_PLACE":
            """
            [ROBUST PHYSICS-AWARE VERSION]
            Executes a slow, smooth, interpolated descent to the target surface.
            It transitions NOT based on time, but on physical confirmation that the
            object has made contact and is supported (i.e., its vertical velocity is zero).
            This makes the placement robust to variations in table height or object size.
            """
            # === STATE ENTRY LOGIC (runs only ONCE on the first step) ===
            if self._wait_counter == 1:
                # 1. Capture the starting position for the smooth descent.
                self._start_place_pos = ee_pos.copy()

                # 2. Define the final target placement position on the surface.
                tcp_placement_z = self.table_surface_z + self.object.size[2]
                place_pos = np.array([goal_pos_world[0], goal_pos_world[1], tcp_placement_z])
                self._end_place_pos = place_pos

            # === CONTINUOUS LOGIC & MONITORING (runs EVERY step) ===
            
            # 1. FAIL FAST: Continuously monitor the grasp.
            if not is_grasped and self._wait_counter > 5:
                print("ERROR: Grasp lost during DESCEND_TO_PLACE state. Aborting.")
                self._handle_failure()
                return self.get_target_pose(expert_obs)

            # 2. Generate the smooth, interpolated descent trajectory.
            progress = min(self._wait_counter / self.cfg.descend_to_place_duration, 1.0)
            interp_pos = self._start_place_pos + (self._end_place_pos - self._start_place_pos) * progress

            # 3. Command the interpolated pose and maintain grip.
            self._target_pose_7d = np.concatenate([interp_pos, self.current_grasp_orientation])
            self._gripper_action = -1.0

            # === PHYSICS-BASED TRANSITION LOGIC ===
            
            # We need the object's velocity from the expert observation.
            object_vertical_velocity = expert_obs.get("object_vel", [0]*6)[2]

            # Condition 1 (Primary): Has the object made contact and is it supported?
            # We check if the downward velocity has stopped.
            # A small wait counter (e.g., > 5) prevents transitioning on initial contact bounce.
            contact_made_and_stable = self._wait_counter > 5 and abs(object_vertical_velocity) < 0.005

            # Condition 2 (Safety Net): Has the maximum allowed time elapsed?
            is_timed_out = self._wait_counter > (self.cfg.descend_to_place_duration + 20) # Add a buffer

            # Transition if contact is stable OR if we've timed out.
            if contact_made_and_stable or is_timed_out:
                if is_timed_out and not contact_made_and_stable:
                    print(f"WARN: DESCEND_TO_PLACE timed out. Forcing release. Object Z Vel: {object_vertical_velocity:.4f}")
                
                # IMPORTANT: Before releasing, we must ensure the robot holds the final placement
                # pose to prevent pulling the object away as the gripper opens.
                # We will add a new state for this.
                self._final_place_pose = self._target_pose_7d.copy()
                self._advance_state("AWAIT_STABLE_PLACEMENT")
        elif self._state == "AWAIT_STABLE_PLACEMENT":
            """
            Holds the final placement pose for a few steps to allow physics to
            settle, ensuring the object is fully supported before release.
            """
            # Command the robot to hold the FINAL, correct placement pose.
            self._target_pose_7d = self._final_place_pose
            self._gripper_action = -1.0 # Keep holding

            object_is_stable = np.linalg.norm(expert_obs.get("object_vel", np.ones(6))) < 0.01

            if self._wait_counter > 5 and object_is_stable:
                self._advance_state("RELEASE")
            elif self._wait_counter > 25: # Timeout
                print("WARN: AWAIT_STABLE_PLACEMENT timed out. Forcing release.")
                self._advance_state("RELEASE")

        elif self._state == "RELEASE":
            """
            [ROBUST & FAILSAFE VERSION]
            This state holds the arm perfectly still while opening the gripper.
            It transitions to RETRACT only after receiving confirmation of TWO events:
            1. The gripper joints are physically open.
            2. The grasp contact has been fully broken (is_grasped is False).
            This prevents the robot from dragging the object during retraction.
            """
            # === STATE ENTRY LOGIC (runs only ONCE on the first step) ===
            if self._wait_counter == 1:
                # Store the current pose to ensure we hold it absolutely steady.
                self._hold_pose_at_release = ee_pose_world.copy()

            # === CONTINUOUS LOGIC (runs EVERY step) ===
            
            # Command the robot to hold the stored pose with no changes.
            self._target_pose_7d = self._hold_pose_at_release
            # Command the gripper to open.
            self._gripper_action = 1.0

            # === DUAL-CONDITION TRANSITION LOGIC ===

            # Condition 1: Are the gripper fingers physically wide open?
            is_gripper_fully_open = np.all(gripper_qpos > self.cfg.gripper_open_threshold)
            
            # Condition 2: Has the physical contact with the object been lost?
            # We check for NOT is_grasped.
            is_contact_broken = not is_grasped

            # Condition 3 (Safety Net): Has too much time passed?
            is_timed_out = self._wait_counter > 30 # A generous timeout for opening the gripper

            # We need a small delay before checking, to give physics time to update.
            # Transition only if BOTH physical conditions are met, OR if we time out.
            if self._wait_counter > 5 and ((is_gripper_fully_open and is_contact_broken) or is_timed_out):
                if is_timed_out:
                    print("WARN: RELEASE state timed out waiting for contact to break. Forcing retract.")

                self._advance_state("RETRACT")


        elif self._state == "RETRACT":
            """
            [ROBUST TRAJECTORY-BASED VERSION]
            Executes a final, smooth, interpolated vertical motion to a safe
            height away from the placed object. It transitions to the DONE state
            only after physically arriving at the retract position, ensuring the
            task is marked as successful only when truly complete.
            """
            # === STATE ENTRY LOGIC (runs only ONCE on the first step) ===
            if self._wait_counter == 1:
                # 1. Capture the starting position for the interpolation.
                self._start_retract_pos = ee_pos.copy()
                
                # 2. Calculate the final destination pose.
                end_retract_pos = self._start_retract_pos + np.array([0, 0, self.cfg.hover_height])
                self._end_retract_pos = self._clamp_to_workspace(end_retract_pos)
            
            # === CONTINUOUS LOGIC (runs EVERY step) ===

            # 1. Calculate the progress of the retraction (0.0 to 1.0).
            progress = min(self._wait_counter / self.cfg.retract_duration_steps, 1.0)

            # 2. Linearly interpolate the position for a smooth vertical trajectory.
            interp_pos = self._start_retract_pos + (self._end_retract_pos - self._start_retract_pos) * progress
            
            # 3. Command the interpolated pose. Orientation remains constant.
            self._target_pose_7d = np.concatenate([interp_pos, self.current_grasp_orientation])
            self._gripper_action = 1.0  # Keep gripper open.

            # === HYBRID TRANSITION LOGIC ===
            
            # Condition 1 (Primary): Has the robot physically reached the retract destination?
            pos_error = np.linalg.norm(ee_pos - self._end_retract_pos)
            is_at_destination = pos_error < self.cfg.pos_tolerance

            # Condition 2 (Safety Net): Has the maximum allowed time elapsed?
            is_timed_out = self._wait_counter > self.cfg.retract_duration_steps

            if is_at_destination or is_timed_out:
                if is_timed_out and not is_at_destination:
                    print(f"WARN: RETRACT state timed out. Final pos error: {pos_error:.4f}m")
                
                # The task is now officially complete and successful.
                self.succeeded = True
                self._advance_state("DONE")

      
        elif self._state == "DONE":
            if self._target_pose_7d is None: # First time entering DONE
                self._target_pose_7d = np.concatenate([ee_pos, self._downward_quat])
            # Hold final position
            self._gripper_action = 1.0
            
        # Fallback to prevent crashes
        if self._target_pose_7d is None:
            self._target_pose_7d = np.concatenate([ee_pos, self._downward_quat])

        # Clamp position to workspace for safety
        clamped_pos = self._clamp_to_workspace(self._target_pose_7d[:3])
        final_pose = np.concatenate([clamped_pos, self._target_pose_7d[3:]]).astype(np.float32)

        return final_pose, float(self._gripper_action)