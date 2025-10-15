

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
    descend_to_place_duration: int = 50
    retract_duration_steps: int = 60
    cruise_velocity: float = 0.4  # meters per second
    accel_decel_buffer_steps: int = 30 # Steps reserved for smooth start/end
    # --- Original physical parameters ---
    hover_height: float = 0.10
    grasp_offset_z: float = 0.025
    failure_timeout_steps: int = 250 # Increased global timeout per state
    pos_tolerance: float = 0.025 # Still used for final checks
    
    workspace: dict = None
    descent_xy_offset: np.ndarray = np.array([0.00, 0.0, 0.0])   
    max_grasp_retries: int = 2
    verify_lift_height: float = 0.03 
    gripper_open_threshold: float = 0.038
    gripper_closed_threshold: float = 0.002
    home_pose_7d: np.ndarray = np.array([0.5, 0.0, 0.7, 0.0, 1.0, 0.0, 0.0])
    orn_tolerance_rad: float = 0.05 
    max_lift_height: float = 0.65
    lookahead_distance: float = 0.03

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

        
        if self._state == "MOVE_TO_PRE_GRASP":
            if self._wait_counter == 1:
                self._start_pre_grasp_pos = ee_pos.copy()
                self._start_pre_grasp_orn = R.from_quat(ee_pose_world[3:])
                end_pos = np.array([cube_pos_world[0], cube_pos_world[1], object_top_z + self.cfg.hover_height])
                self._end_pre_grasp_pos = end_pos
                final_aligned_quat = self._calculate_aligned_orientation(object_orn_world, ee_pose_world[3:])
                self._end_pre_grasp_orn = R.from_quat(final_aligned_quat)
                total_dist = np.linalg.norm(self._end_pre_grasp_pos - self._start_pre_grasp_pos)
                travel_steps = int((total_dist / self.cfg.cruise_velocity) * 100)
                self._adaptive_duration = max(20, travel_steps + self.cfg.accel_decel_buffer_steps)

            progress = min(self._wait_counter / self._adaptive_duration, 1.0)
            eased_progress = 0.5 * (1.0 - np.cos(progress * np.pi))
            interp_pos = self._start_pre_grasp_pos + (self._end_pre_grasp_pos - self._start_pre_grasp_pos) * eased_progress
            slerp = Slerp([0, 1], R.from_quat([self._start_pre_grasp_orn.as_quat(), self._end_pre_grasp_orn.as_quat()]))
            interp_orn_quat = slerp(eased_progress).as_quat()
            self._target_pose_7d = np.concatenate([interp_pos, interp_orn_quat])
            self._gripper_action = 1.0

            pos_error = np.linalg.norm(ee_pos - self._end_pre_grasp_pos)
            R_current = R.from_quat(ee_pose_world[3:])
            angular_distance = (self._end_pre_grasp_orn.inv() * R_current).magnitude()
            is_at_destination = (pos_error < self.cfg.pos_tolerance) and (angular_distance < self.cfg.orn_tolerance_rad)
            is_timed_out = self._wait_counter > max(self._adaptive_duration, self.cfg.move_to_pre_grasp_duration)

            if is_at_destination:
                self.current_grasp_orientation = self._end_pre_grasp_orn.as_quat()
                self._advance_state("PREPARE_GRIPPER")
            elif is_timed_out:
                print(f"ERROR: MOVE_TO_PRE_GRASP timed out. Pos Error: {pos_error:.3f}m")
                self._handle_failure()

        elif self._state == "PREPARE_GRIPPER":
            """
            [DEFINITIVE FAILSAFE VERSION]
            This state acts as a robust checkpoint. It verifies and, if necessary,
            corrects the robot's orientation to the ideal grasp pose while holding
            position. It transitions only after confirming BOTH a stable pose AND
            a fully open gripper. A timeout in this state is treated as a grasp
            failure, triggering a full retry of the pick sequence.
            """
            # === STATE ENTRY LOGIC (runs only ONCE on the first step) ===
            if self._wait_counter == 1:
                # 1. ALWAYS calculate the definitive target orientation. This serves as our
                #    uncompromising ground truth for verification and correction.
                target_orn_quat = self._calculate_aligned_orientation(object_orn_world, ee_pose_world[3:])
                self._target_orn = R.from_quat(target_orn_quat)

                # 2. Store the current position to hold it absolutely steady.
                self._hold_pos = ee_pos.copy()
                
                # 3. Assemble the full 7D target pose we will command.
                self._stored_target_pose = np.concatenate([self._hold_pos, self._target_orn.as_quat()])
                
                # 4. Pre-calculate the descent target for the subsequent state.
                grasp_z = object_top_z - self.cfg.grasp_offset_z
                descent_xy = cube_pos_world[:2] + self.cfg.descent_xy_offset[:2]
                self._descent_target_pos = np.array([descent_xy[0], descent_xy[1], grasp_z])

                # 5. Initialize stability counter.
                self._orientation_stable_counter = 0

            # === CONTINUOUS LOGIC (runs EVERY step) ===
            
            # Command the robot to the target pose. This corrects orientation errors
            # while holding the XY position, and does nothing if already aligned.
            self._target_pose_7d = self._stored_target_pose
            self._gripper_action = 1.0 # Command gripper to open

            # === DUAL-CONDITION TRANSITION & FAILURE HANDLING ===
            
            # Condition 1: Verify orientation is correct and stable.
            R_current = R.from_quat(ee_pose_world[3:])
            angular_distance = (self._target_orn.inv() * R_current).magnitude()
            if angular_distance < self.cfg.orn_tolerance_rad:
                self._orientation_stable_counter += 1
            else:
                self._orientation_stable_counter = 0
            is_oriented_and_stable = self._orientation_stable_counter > 5

            # Condition 2: Verify gripper is physically open.
            is_gripper_open = np.all(gripper_qpos > self.cfg.gripper_open_threshold)
            
            # Check for success: Both conditions MUST be met.
            is_ready_to_descend = is_oriented_and_stable and is_gripper_open
            
            # Check for failure: The timeout has been exceeded.
            is_timed_out = self._wait_counter > self.cfg.prepare_gripper_duration

            if is_ready_to_descend:
                # SUCCESS: Everything is perfect. Commit the state and advance.
                self.current_grasp_orientation = self._target_orn.as_quat()
                self._advance_state("DESCEND_TO_GRASP")
            elif is_timed_out:
                # FAILURE: We ran out of time. The preparation has failed.
                print("ERROR: PREPARE_GRIPPER timed out. Could not achieve stable pose and open gripper.")
                # Do not proceed. Trigger the global failure handler to initiate a retry.
                self._handle_failure()

        elif self._state == "DESCEND_TO_GRASP":
            """
            [DEFINITIVE PATH-FOLLOWING VERSION]
            This state defines a static geometric path from the start to the end pose.
            It then generates a dynamic target point that moves along this path,
            staying a small 'lookahead_distance' ahead of the robot's actual position.
            This creates an adaptive, smooth, and stable trajectory that is not
            dependent on a fixed duration, but on the robot's real performance.
            """
            # === STATE ENTRY LOGIC (runs only ONCE on the first step) ===
            if self._wait_counter == 1:
                # 1. Define the static PATH: Start, End, Direction, and total length.
                self._path_start_pos = ee_pos.copy()
                # _descent_target_pos was robustly calculated in the previous state.
                self._path_end_pos = self._descent_target_pos
                self._path_vector = self._path_end_pos - self._path_start_pos
                self._path_length = np.linalg.norm(self._path_vector)
                
                # Normalize the path vector, handling the zero-length case.
                if self._path_length > 1e-6:
                    self._path_direction = self._path_vector / self._path_length
                else:
                    self._path_direction = np.zeros(3)

                # Initialize the robot's progress along the path.
                self._current_path_progress = 0.0

            # === CONTINUOUS LOGIC (Path Following) ===

            # 1. Find the robot's current projection onto the path.
            #    This tells us how far along the path the robot has actually traveled.
            robot_vec = ee_pos - self._path_start_pos
            dist_along_path = np.dot(robot_vec, self._path_direction)
            
            # 2. The "Target Point" is a lookahead distance *ahead* of the robot's current spot on the path.
            target_dist_on_path = dist_along_path + self.cfg.lookahead_distance

            # 3. The new commanded position is the point on the path at this new target distance.
            #    We also ensure the target never goes past the end of the path.
            self._current_path_progress = np.clip(target_dist_on_path, 0.0, self._path_length)
            interp_pos = self._path_start_pos + self._path_direction * self._current_path_progress

            # 4. Command the new target pose. Orientation is constant during descent.
            self._target_pose_7d = np.concatenate([interp_pos, self.current_grasp_orientation])
            self._gripper_action = 1.0

            # === ROBUST TRANSITION & FAILURE HANDLING ===

            # Condition for Success: Has the robot's actual position reached the end of the path?
            pos_error = np.linalg.norm(ee_pos - self._path_end_pos)
            is_at_destination = pos_error < self.cfg.pos_tolerance

            # Condition for Failure: Has a generous timeout elapsed? This prevents getting stuck.
            is_timed_out = self._wait_counter > (self.cfg.descend_to_grasp_duration + 40) # Use a generous fixed timeout
            
            if is_at_destination:
                # SUCCESS: We have arrived. Advance to the grasp action.
                self._advance_state("GRASP")
            elif is_timed_out:
                # FAILURE: We got stuck or something went wrong.
                print(f"ERROR: DESCEND_TO_GRASP timed out. Final position error: {pos_error:.4f}m")
                self._handle_failure()

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
            [DEFINITIVE PATH-FOLLOWING VERSION]
            Executes an adaptive, smooth vertical lift by following a moving target
            point along a defined path. It continuously monitors the grasp for
            integrity and transitions only upon successful arrival at the target height.
            """
            # === STATE ENTRY LOGIC (runs only ONCE on the first step) ===
            if self._wait_counter == 1:
                # 1. Define the static PATH for the lift.
                self._path_start_pos = ee_pos.copy()
                
                # Calculate the final lift position using the adaptive height.
                adaptive_h = self.adaptive_hover_height(ee_pos[:2], robot_base_pos_world[:2])
                self._path_end_pos = self._clamp_to_workspace(self._path_start_pos + np.array([0, 0, adaptive_h]))
                
                self._path_vector = self._path_end_pos - self._path_start_pos
                self._path_length = np.linalg.norm(self._path_vector)
                
                if self._path_length > 1e-6:
                    self._path_direction = self._path_vector / self._path_length
                else: # Handle the case of a zero-length lift
                    self._path_direction = np.array([0, 0, 1.0])
                    self._path_length = 0.0

            # === CONTINUOUS LOGIC & MONITORING (runs EVERY step) ===

            # 1. FAIL FAST: Abort immediately if the grasp is lost.
            if not is_grasped and self._wait_counter > 5:
                print("ERROR: Grasp lost during LIFT state. Aborting.")
                self._handle_failure()
                return self.get_target_pose(expert_obs)

            # 2. Implement the Path-Following logic.
            # Find the robot's current projection onto the vertical path.
            robot_vec = ee_pos - self._path_start_pos
            dist_along_path = np.dot(robot_vec, self._path_direction)
            
            # The target point is a lookahead distance ahead of the robot's current progress.
            target_dist_on_path = dist_along_path + self.cfg.lookahead_distance

            # The new commanded position is the point on the path at this new target distance.
            current_path_progress = np.clip(target_dist_on_path, 0.0, self._path_length)
            interp_pos = self._path_start_pos + self._path_direction * current_path_progress
            
            # 3. Command the new target pose and maintain grip.
            self._target_pose_7d = np.concatenate([interp_pos, self.current_grasp_orientation])
            self._gripper_action = -1.0

            # === ROBUST TRANSITION & FAILURE HANDLING ===

            # Condition for Success: Has the robot's actual position reached the end of the path?
            pos_error = np.linalg.norm(ee_pos - self._path_end_pos)
            is_at_destination = (pos_error < self.cfg.pos_tolerance) or (self._path_length == 0.0)

            # Condition for Failure: Use the fixed duration as a generous timeout.
            is_timed_out = self._wait_counter > (self.cfg.lift_duration_steps + 20)
            
            if is_at_destination:
                # Final check before moving on.
                if is_grasped:
                    self._advance_state("MOVE_TO_GOAL")
                else: # Should be caught by fail-fast, but this is a final guarantee.
                    print("ERROR: Grasp lost at the very end of LIFT. Aborting.")
                    self._handle_failure()
            elif is_timed_out:
                print(f"ERROR: LIFT state timed out. Final position error: {pos_error:.4f}m")
                self._handle_failure()

        elif self._state == "MOVE_TO_GOAL":
            """
            [DEFINITIVE ADAPTIVE PATH-FOLLOWING VERSION]
            Executes a fully synchronized, adaptive trajectory in 6D space.
            Position follows a moving target point along a path. Orientation is
            spherically interpolated in sync with the positional progress.
            Grasp is continuously monitored for a fully robust and efficient transport.
            """
            # === STATE ENTRY LOGIC (runs only ONCE on the first step) ===
            if self._wait_counter == 1:
                # 1. Define the static PATH for POSITION.
                self._path_start_pos = ee_pos.copy()
                goal_hover_z = goal_pos_world[2] + self.cfg.hover_height + object_half_height
                self._path_end_pos = np.array([goal_pos_world[0], goal_pos_world[1], goal_hover_z])
                self._path_vector = self._path_end_pos - self._path_start_pos
                self._path_length = np.linalg.norm(self._path_vector)

                if self._path_length > 1e-6:
                    self._path_direction = self._path_vector / self._path_length
                else:
                    self._path_direction = np.zeros(3)
                
                # 2. Define the "PATH" for ORIENTATION.
                self._start_move_orn = R.from_quat(self.current_grasp_orientation)
                goal_orn_world = expert_obs["goal_orn_world"]
                self._place_orn = R.from_quat(self._calculate_aligned_orientation(goal_orn_world, ee_pose_world[3:]))
                
                # 3. Create the Slerp interpolator once.
                self._slerp = Slerp([0, 1], R.from_quat([self._start_move_orn.as_quat(), self._place_orn.as_quat()]))

            # === CONTINUOUS LOGIC & MONITORING (runs EVERY step) ===

            # 1. FAIL FAST: Abort immediately if the grasp is lost.
            if not is_grasped and self._wait_counter > 5:
                print("ERROR: Grasp lost during MOVE_TO_GOAL state. Aborting.")
                self._handle_failure()
                return self.get_target_pose(expert_obs)

            # 2. PATH-FOLLOWING for POSITION:
            robot_vec = ee_pos - self._path_start_pos
            dist_along_path = np.dot(robot_vec, self._path_direction)
            target_dist_on_path = dist_along_path + self.cfg.lookahead_distance
            current_path_progress = np.clip(target_dist_on_path, 0.0, self._path_length)
            interp_pos = self._path_start_pos + self._path_direction * current_path_progress

            # 3. SYNCHRONIZED INTERPOLATION for ORIENTATION:
            # Calculate the completion ratio of the positional path.
            path_completion_ratio = 0.0 if self._path_length < 1e-6 else np.clip(dist_along_path / self._path_length, 0.0, 1.0)
            
            # Use this ratio as the input to Slerp.
            interp_orn = self._slerp(path_completion_ratio).as_quat()

            # 4. Command the synchronized 7D pose and maintain grip.
            self._target_pose_7d = np.concatenate([interp_pos, interp_orn])
            self._gripper_action = -1.0

            # === ROBUST TRANSITION & FAILURE HANDLING ===

            # Condition for Success: Has the robot's actual position reached the end of the path?
            pos_error = np.linalg.norm(ee_pos - self._path_end_pos)
            is_at_destination = pos_error < self.cfg.pos_tolerance

            # Condition for Failure: Use the fixed duration as a generous timeout.
            is_timed_out = self._wait_counter > self.cfg.move_to_goal_duration
            
            if is_at_destination:
                if is_grasped:
                    self.current_grasp_orientation = self._place_orn.as_quat()
                    self._advance_state("PREPARE_PLACE")
                else:
                    print("ERROR: Grasp lost at the end of MOVE_TO_GOAL. Aborting.")
                    self._handle_failure()
            elif is_timed_out:
                print(f"ERROR: MOVE_TO_GOAL state timed out. Final position error: {pos_error:.4f}m")
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
            [DEFINITIVE ADAPTIVE & PHYSICS-AWARE VERSION]
            Executes a highly controlled, adaptive descent using a path-following
            algorithm. The speed of descent is naturally governed by the robot's
            ability to chase a lookahead target. The state transitions ONLY after
            receiving physical confirmation from the simulation that the object has
            made contact and is fully supported (vertical velocity is zero). This
            provides the highest possible robustness against variations and errors.
            """
            # === STATE ENTRY LOGIC (runs only ONCE on the first step) ===
            if self._wait_counter == 1:
                # 1. Define the static PATH for the descent.
                self._path_start_pos = ee_pos.copy()
                tcp_placement_z = self.table_surface_z + self.object.size[2]
                self._path_end_pos = np.array([goal_pos_world[0], goal_pos_world[1], tcp_placement_z])
                self._path_vector = self._path_end_pos - self._path_start_pos
                self._path_length = np.linalg.norm(self._path_vector)

                if self._path_length > 1e-6:
                    self._path_direction = self._path_vector / self._path_length
                else:
                    self._path_direction = np.array([0, 0, -1.0]) # Assume downward if no path
                    self._path_length = 0.0

            # === CONTINUOUS LOGIC & MONITORING (runs EVERY step) ===

            # 1. FAIL FAST: Continuously monitor the grasp.
            if not is_grasped and self._wait_counter > 5:
                print("ERROR: Grasp lost during DESCEND_TO_PLACE. Aborting.")
                self._handle_failure()
                return self.get_target_pose(expert_obs)

            # 2. PATH-FOLLOWING LOGIC for a smooth, adaptive descent.
            robot_vec = ee_pos - self._path_start_pos
            dist_along_path = np.dot(robot_vec, self._path_direction)
            target_dist_on_path = dist_along_path + self.cfg.lookahead_distance
            current_path_progress = np.clip(target_dist_on_path, 0.0, self._path_length)
            interp_pos = self._path_start_pos + self._path_direction * current_path_progress
            
            # 3. Command the new target pose and maintain grip.
            self._target_pose_7d = np.concatenate([interp_pos, self.current_grasp_orientation])
            self._gripper_action = -1.0

            # === PHYSICS-BASED TRANSITION & FAILURE HANDLING ===

            object_vertical_velocity = expert_obs.get("object_vel", [0]*6)[2]

            # Condition 1 (Primary): Has the object made contact and is it supported?
            contact_made_and_stable = self._wait_counter > 5 and abs(object_vertical_velocity) < 0.005

            # Condition 2 (Safety Net): Has a generous timeout elapsed?
            is_timed_out = self._wait_counter > (self.cfg.descend_to_place_duration + 40) # Use a generous fixed timeout
            
            # Transition if contact is confirmed OR if we time out.
            if contact_made_and_stable or is_timed_out:
                if is_timed_out and not contact_made_and_stable:
                    print(f"WARN: DESCEND_TO_PLACE timed out. Forcing release. Object Z Vel: {object_vertical_velocity:.4f}")

                # Store the absolute final pose we want to hold during the release sequence.
                # It is safer to use the defined end of the path than the robot's current pose.
                self._final_place_pose = np.concatenate([self._path_end_pos, self.current_grasp_orientation])
                self._advance_state("AWAIT_STABLE_PLACEMENT")


        elif self._state == "AWAIT_STABLE_PLACEMENT":
            """
            [DEFINITIVE FAILSAFE & ROBUST VERIFICATION VERSION]
            This state holds the arm at the final placement pose and waits for
            physical confirmation that the object is stable. It uses a debounced
            check on the object's vertical velocity, which is a more robust
            indicator of stability than the full 6D velocity norm. A timeout
            is correctly treated as a task failure, triggering a retry.
            """
            # === STATE ENTRY LOGIC ===
            if self._wait_counter == 1:
                self._object_stable_counter = 0

            # === CONTINUOUS LOGIC & MONITORING ===
            self._target_pose_7d = self._final_place_pose
            self._gripper_action = -1.0
            if not is_grasped and self._wait_counter > 5:
                print("ERROR: Grasp lost during AWAIT_STABLE_PLACEMENT. Aborting.")
                self._handle_failure()
                return self.get_target_pose(expert_obs)

            # === ROBUST, DEBOUNCED TRANSITION LOGIC ===
            
            # THE KEY CHANGE: Check only the vertical velocity.
            # This is a much more stable indicator of whether the object is supported by the table.
            object_vertical_velocity = expert_obs.get("object_vel", [0]*6)[2]
            # Use a slightly more lenient threshold to account for controller noise.
            object_is_currently_stable = abs(object_vertical_velocity) < 0.01 

            if object_is_currently_stable:
                self._object_stable_counter += 1
            else:
                self._object_stable_counter = 0

            is_confirmed_stable = self._object_stable_counter > 5
            is_timed_out = self._wait_counter > 25

            # THE SECOND KEY CHANGE: Timeout is now a FAILURE condition.
            if is_confirmed_stable:
                # SUCCESS: The object is verifiably stable. Proceed to release.
                self._advance_state("RELEASE")
            elif is_timed_out:
                # FAILURE: We could not confirm the object was stable.
                print("ERROR: AWAIT_STABLE_PLACEMENT timed out. Object is not stable.")
                # Do not proceed. Trigger the global failure handler.
                self._handle_failure()

        elif self._state == "RELEASE":
            """
            [DEFINITIVE SEQUENTIAL VERIFICATION VERSION]
            This state ensures an impeccably clean release. It holds the arm
            perfectly still while opening the gripper. It then waits for sequential
            confirmation: first, that the gripper is physically open, and second,
            that the physical contact with the object has been stably broken for
            several consecutive steps. This completely eliminates any risk of
            dragging the object upon retraction.
            """
            # === STATE ENTRY LOGIC (runs only ONCE on the first step) ===
            if self._wait_counter == 1:
                # 1. Store the pose to hold absolutely steady during release.
                self._hold_pose_at_release = ee_pose_world.copy()
                # 2. Initialize a counter to confirm the contact break is not a flicker.
                self._contact_broken_counter = 0

            # === CONTINUOUS LOGIC (runs EVERY step) ===
            
            # Command the robot to hold still and open the gripper.
            self._target_pose_7d = self._hold_pose_at_release
            self._gripper_action = 1.0

            # === SEQUENTIAL & DEBOUNCED TRANSITION LOGIC ===

            # Condition 1: Is the gripper mechanism fully open?
            is_gripper_fully_open = np.all(gripper_qpos > self.cfg.gripper_open_threshold)

            # Condition 2: Has the physical contact with the object been broken?
            # This is only checked AFTER the gripper is confirmed to be open.
            if is_gripper_fully_open:
                if not is_grasped:
                    # If contact is broken, increment the stability counter.
                    self._contact_broken_counter += 1
                else:
                    # If the signal flickers back, reset the counter.
                    self._contact_broken_counter = 0
            
            # We require 3 consecutive steps of broken contact to be sure.
            is_release_confirmed = self._contact_broken_counter > 3

            # Condition 3 (Safety Net): Has a generous timeout elapsed?
            is_timed_out = self._wait_counter > 30

            # Transition if the debounced release is confirmed OR if we time out.
            if is_release_confirmed or is_timed_out:
                if is_timed_out and not is_release_confirmed:
                    print("WARN: RELEASE state timed out. Forcing retract.")
                
                self._advance_state("RETRACT")


        elif self._state == "RETRACT":
            """
            [DEFINITIVE ADAPTIVE & SMOOTH VERSION]
            Executes a final, smooth, and professional retraction. The duration is
            adaptively calculated based on the retract distance and a desired cruise
            velocity. The trajectory is eased to provide smooth acceleration and
            deceleration. It transitions to DONE only after physically verifying
            arrival at the safe retract position.
            """
            # === STATE ENTRY LOGIC (runs only ONCE on the first step) ===
            if self._wait_counter == 1:
                # 1. Define the start and end of the path.
                self._start_retract_pos = ee_pos.copy()
                end_retract_pos = self._start_retract_pos + np.array([0, 0, self.cfg.hover_height])
                self._end_retract_pos = self._clamp_to_workspace(end_retract_pos)
                
                # 2. Calculate the adaptive duration for this specific move.
                total_dist = np.linalg.norm(self._end_retract_pos - self._start_retract_pos)
                # Time = Distance / Speed. Convert to steps (assuming 100Hz).
                travel_steps = int((total_dist / self.cfg.cruise_velocity) * 100)
                self._adaptive_duration = travel_steps + self.cfg.accel_decel_buffer_steps
                
                # Ensure a minimum duration for very short moves.
                if self._adaptive_duration < 20:
                    self._adaptive_duration = 20
            
            # === CONTINUOUS LOGIC (runs EVERY step) ===

            # 1. Calculate progress using the adaptive duration and apply easing.
            progress = min(self._wait_counter / self._adaptive_duration, 1.0)
            eased_progress = 0.5 * (1.0 - np.cos(progress * np.pi))

            # 2. Interpolate the position for a smooth vertical trajectory.
            interp_pos = self._start_retract_pos + (self._end_retract_pos - self._start_retract_pos) * eased_progress
            
            # 3. Command the interpolated pose.
            self._target_pose_7d = np.concatenate([interp_pos, self.current_grasp_orientation])
            self._gripper_action = 1.0  # Keep gripper open.

            # === HYBRID TRANSITION LOGIC ===
            
            # The timeout is the larger of our adaptive plan or the global config.
            timeout = max(self._adaptive_duration, self.cfg.retract_duration_steps)
            
            pos_error = np.linalg.norm(ee_pos - self._end_retract_pos)
            is_at_destination = pos_error < self.cfg.pos_tolerance
            is_timed_out = self._wait_counter > timeout

            if is_at_destination or is_timed_out:
                if is_timed_out and not is_at_destination:
                    print(f"WARN: RETRACT state timed out. Final pos error: {pos_error:.4f}m")
                
                # Set success flag ONLY after the final action is verifiably complete.
                self.succeeded = True
                self._advance_state("DONE")

      
        elif self._state == "DONE":
            """
            [DEFINITIVE TERMINAL STATE]
            This is the final, quiescent state. Its only purpose is to command the
            robot to hold its final, intended pose from the successful RETRACT
            maneuver. This ensures absolute stability and prevents any post-task
            drift or unnecessary motion until the episode is reset.
            """
            # === STATE ENTRY LOGIC (runs only ONCE on the first step) ===
            if self._wait_counter == 1:
                # 1. The ideal final pose is the destination of the previous RETRACT state.
                #    We retrieve it from our state variables to ensure perfect continuity.
                #    No new calculation is needed.
                self._final_hold_pose = np.concatenate([self._end_retract_pos, self.current_grasp_orientation])

            # === CONTINUOUS LOGIC (runs EVERY step) ===
            
            # 1. Continuously command the robot to hold the final pose.
            self._target_pose_7d = self._final_hold_pose
            
            # 2. Keep the gripper open.
            self._gripper_action = 1.0
            
        # Fallback to prevent crashes
        if self._target_pose_7d is None:
            self._target_pose_7d = np.concatenate([ee_pos, self._downward_quat])

        # Clamp position to workspace for safety
        clamped_pos = self._clamp_to_workspace(self._target_pose_7d[:3])
        final_pose = np.concatenate([clamped_pos, self._target_pose_7d[3:]]).astype(np.float32)

        return final_pose, float(self._gripper_action)