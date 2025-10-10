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
    """Configuration for the scripted expert policy."""
    hover_height: float = 0.10
    pos_tolerance: float = 0.015
    workspace: dict = None
    grasp_offset_z: float = 0.025  
    descent_xy_offset: np.ndarray = np.array([0.00, 0.0, 0.0])   
    max_grasp_retries: int = 2      # Number of retry attempts if grasp fails
    lift_duration_steps: int = 40   # Increased for smoother, more stable lift
    move_duration_steps: int = 60   # Ensure ample time for horizontal moves
    place_duration_steps: int = 25  # Controlled descent for placement
    retract_duration_steps: int = 20
    failure_timeout_steps: int = 150 # Timeout per state to prevent infinite loops
    verify_lift_height: float = 0.03 
    gripper_open_threshold: float = 0.038  # Considered "open" if joint position is > 0.038
    gripper_closed_threshold: float = 0.002 # Considered "closed" if joint position is < 0.002
    home_pose_7d: np.ndarray = np.array([0.5, 0.0, 0.7, 0.0, 1.0, 0.0, 0.0])
    orn_tolerance_rad: float = 0.05 


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

    def _handle_grasp_failure(self):
        """Handles grasp failure by retrying or failing gracefully."""
        self._grasp_retry_count += 1
        if self._grasp_retry_count <= self.cfg.max_grasp_retries:
            # Retry: Go back to pre-grasp
            self._state = "MOVE_TO_PRE_GRASP"
            self._gripper_action = -1.0  # Ensure open for retry
            self._reset_wait_counter()
        else:
            # Exhaust retries: Fail to done
            self.succeeded = False
            self._state = "DONE"
            self._gripper_action = -1.0


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


    def get_target_pose(
        self,
        expert_obs: dict, 
    ) -> Tuple[np.ndarray, float]:
        """
        Computes target 7D pose (pos + xyzw quat) and normalized gripper action [-1, 1].
        Advances state machine based on conditions.
        """

        ee_pose_world = expert_obs["ee_pose_world"]
        cube_pos_world = expert_obs["object_pos_world"]
        object_orn_world = expert_obs["object_orn_world"]
        goal_pos_world = expert_obs["goal_pos_world"]
        is_grasped = expert_obs["is_grasped"][0] > 0.5
        gripper_qpos = expert_obs["gripper_qpos"]

        ee_pos = ee_pose_world[:3]
        object_half_height = self.object.size[2] / 2.0
        object_center_z = cube_pos_world[2]
        object_top_z = object_center_z + object_half_height
        object_vel = expert_obs.get("object_vel", np.zeros(6))


        self._wait_counter += 1
        timeout_steps = 0
        
        # States that follow a trajectory for a fixed duration.
        # The timeout should be slightly longer than their intended duration.
        if self._state == "LIFT":
            timeout_steps = self.cfg.lift_duration_steps + 20
        elif self._state == "MOVE_TO_GOAL":
            timeout_steps = self.cfg.move_duration_steps + 30 # Extra generous
        elif self._state == "MOVE_HOVER_FINAL":
            timeout_steps = self.cfg.retract_duration_steps + 20
        
        # States that are event-driven (e.g., waiting for grasp).
        # These get a fixed, generous timeout.
        elif self._state in ["GRASP", "RELEASE", "WAIT_FOR_RELEASE"]:
            timeout_steps = 40
            
        # All other states are position-based. Their timeout depends on
        # how far the robot has to travel.
        else:
            temp_target_pos = self._target_pose_7d[:3] if self._target_pose_7d is not None else ee_pos
            dist_to_target = np.linalg.norm(ee_pos - temp_target_pos)
            timeout_steps = 40 + int(dist_to_target * 250) # The original adaptive logic

        # 2. Now, check if the single, unified timeout has been exceeded.
        if self._wait_counter > timeout_steps:
            print(f"WARN: Timeout of {timeout_steps} steps exceeded in state '{self._state}'. Handling grasp failure.")
            self._handle_grasp_failure()
            
        if self._state == "MOVE_TO_PRE_GRASP":

            target_pos = np.array([
                cube_pos_world[0],
                cube_pos_world[1],
                object_top_z + self.cfg.hover_height
            ])
      
            # 2. Get the gripper's CURRENT orientation from the expert observation.
            current_ee_orientation = ee_pose_world[3:] # This is the xyzw quaternion
            
            # 3. Command the robot to move to the target_pos but KEEP its current orientation.
            self._target_pose_7d = np.concatenate([target_pos, current_ee_orientation])
            self._gripper_action = -1.0  
            
            # 4. The transition condition remains the same.
            if np.linalg.norm(ee_pos - target_pos) < self.cfg.pos_tolerance:
                self._gripper_action = 1.0 
                # NOW that we are in position, the next state (PREPARE_GRIPPER) will handle the re-orientation.
                self._advance_state("PREPARE_GRIPPER")

        elif self._state == "PREPARE_GRIPPER":
            """
            NEW, ROBUST STATE: Orients the gripper for grasping and opens it.
            This version fixes instability, race conditions, and drifting targets.
            """
            # === STATE ENTRY LOGIC (runs only on the first step) ===
            if self._wait_counter == 1:
                # --- FIX for Smoking Guns #1 (Moving Target) & #2 (Position Jitter) ---
                self._hold_pos = ee_pos.copy()
                self._target_orn = self._calculate_aligned_orientation(
                    object_orn_world,
                    ee_pose_world[3:]
                )
                self._orientation_stable_counter = 0

                # --- FIX for Smoking Gun #4 (Drifting Target) ---
                # Capture the object's position now, so the descent has a fixed target.
                grasp_z = object_top_z - self.cfg.grasp_offset_z
                descent_xy = cube_pos_world[:2] + self.cfg.descent_xy_offset[:2]
                self._descent_target_pos = np.array([descent_xy[0], descent_xy[1], grasp_z])

            # === CONTINUOUS LOGIC (runs every step) ===
            
            # Command the robot to go to the STORED hold position and STORED target orientation.
            self._target_pose_7d = np.concatenate([self._hold_pos, self._target_orn])
            self._gripper_action = 1.0 # Command to open

            # === ROBUST TRANSITION LOGIC ===
            
            # --- FIX for Smoking Gun #2 (Chattering Transition) ---
            # We now require BOTH orientation stability AND an open gripper to transition.
            
            # Condition 1: Is the orientation correct and stable?
            current_ee_quat = ee_pose_world[3:]
            R_current = R.from_quat(current_ee_quat)
            R_target = R.from_quat(self._target_orn)
            angular_distance = (R_target.inv() * R_current).magnitude()

            if angular_distance < self.cfg.orn_tolerance_rad:
                self._orientation_stable_counter += 1
            else:
                self._orientation_stable_counter = 0
            
            is_oriented_and_stable = self._orientation_stable_counter > 5

            # Condition 2: Is the gripper physically open?
            is_gripper_open = np.all(gripper_qpos > self.cfg.gripper_open_threshold)
            
            # Condition 3: Timeout safety net.
            is_timed_out = self._wait_counter > 40

            # Transition only if stable AND open, OR if timed out.
            if (is_oriented_and_stable and is_gripper_open) or is_timed_out:
                if is_timed_out:
                    print("WARN: PREPARE_GRIPPER timed out.")
                self.current_grasp_orientation = self._target_orn.copy()
                self._advance_state("DESCEND_TO_GRASP")
      
        elif self._state == "DESCEND_TO_GRASP":
            """
            NEW, ROBUST STATE: Descends to the pre-calculated grasp position.
            """
            # Use the stable, fixed target position calculated in the previous state.
            target_pos = self._descent_target_pos
            
            # --- FIX for Smoking Gun #5 (Unnecessary Command) ---
            # Maintain the orientation and hold the gripper open, no need to resend `1.0`.
            self._target_pose_7d = np.concatenate([target_pos, self.current_grasp_orientation])
            self._gripper_action = 1.0  # Keep open

            # Standard position-based transition.
            distance_to_target = np.linalg.norm(ee_pos - target_pos)
            if distance_to_target < self.cfg.pos_tolerance or self._wait_counter > 30:
                self._advance_state("GRASP") 

        elif self._state == "GRASP":
            # Hold position and close gripper. Record initial object position.
            self._target_pose_7d = np.concatenate([ee_pos, self.current_grasp_orientation]) 
            self._gripper_action = -1.0
            if is_grasped:
                self.object_pos_pre_lift = cube_pos_world[2] 
                self._advance_state("LIFT")
            elif self._wait_counter > 30:
                self._handle_grasp_failure()

        elif self._state == "LIFT":
            # --- START OF FIX: Use a fixed, object-centric lift target ---
            
            # === STATE ENTRY LOGIC (runs only on the first step) ===
            if self._wait_counter == 1:
                # Calculate the lift target position ONCE and store it.
                # Base the XY on the cube's position at the start of the lift for a perfect vertical trajectory.
                object_top_z_start = self.object_pos_pre_lift + object_half_height
                self._lift_target_pos = np.array([
                    cube_pos_world[0], 
                    cube_pos_world[1], 
                    object_top_z_start + self.cfg.hover_height
                ])

            
            # Command the robot to move to the STORED target position.
            self._target_pose_7d = np.concatenate([self._lift_target_pos, self.current_grasp_orientation])
            self._gripper_action = -1.0 # Keep gripper closed
            # --- END OF FIX ---

            # The failure checks remain the same and are already robust.
            is_object_lifted = (cube_pos_world[2] - self.object_pos_pre_lift) > self.cfg.verify_lift_height
            is_still_grasped = is_grasped

            if self._wait_counter > 15:
                if not is_object_lifted or not is_still_grasped:
                    print(f"DEBUG: Grasp failure detected in LIFT. Lifted: {is_object_lifted}, Grasped: {is_still_grasped}")
                    self._handle_grasp_failure()
            
            # The success transition now correctly checks against the fixed target.
            if np.linalg.norm(ee_pos - self._lift_target_pos) < self.cfg.pos_tolerance:
                self._advance_state("MOVE_TO_GOAL")

        elif self._state == "MOVE_TO_GOAL":
            # --- START OF FULL REPLACEMENT ---
            """
            NEW, ROBUST STATE: Moves to the goal while simultaneously re-orienting.
            This creates a single, smooth, efficient trajectory using SLERP for rotation.
            """
            # === STATE ENTRY LOGIC (runs only on the first step) ===
            if self._wait_counter == 1:
                # 1. Store the starting position and orientation for interpolation.
                self._start_move_pos = ee_pos.copy()
                self._start_move_orn = R.from_quat(self.current_grasp_orientation)

                # 2. Calculate the FINAL target placement orientation.
                goal_orn_world = expert_obs["goal_orn_world"]
                # We use the goal's orientation as the "object" and the current EE orn as the "gripper".
                self._place_orn = R.from_quat(self._calculate_aligned_orientation(
                    goal_orn_world, ee_pose_world[3:]
                ))
            
            # === CONTINUOUS LOGIC (runs every step) ===

            # 1. Calculate the target position (this remains the same).
            goal_hover_z = goal_pos_world[2] + self.cfg.hover_height + object_half_height
            target_pos = np.array([
                goal_pos_world[0],
                goal_pos_world[1],
                goal_hover_z
            ])

            # 2. Interpolate both position and orientation over the duration of the move.
            progress = min(self._wait_counter / self.cfg.move_duration_steps, 1.0)

            # Linear interpolation for position
            interp_pos = self._start_move_pos + (target_pos - self._start_move_pos) * progress
            
            key_times = [0, 1]
            key_rots = R.from_quat([self._start_move_orn.as_quat(), self._place_orn.as_quat()])
            
            # 2. Create the Slerp object.
            slerp = Slerp(key_times, key_rots)
            
            # 3. Call the Slerp object with the current progress to get the interpolated rotation.
            interp_rotation = slerp(progress)
            interp_orn = interp_rotation.as_quat()

            self._target_pose_7d = np.concatenate([interp_pos, interp_orn])
            self._gripper_action = -1.0

            pos_error = np.linalg.norm(ee_pos - target_pos)
            
            R_current = R.from_quat(ee_pose_world[3:])
            angular_distance = (self._place_orn.inv() * R_current).magnitude()
            
            is_at_destination = (pos_error < self.cfg.pos_tolerance) and \
                                (angular_distance < self.cfg.orn_tolerance_rad)

            # Condition 2: Has the maximum allowed time elapsed? (Fallback)
            is_timed_out = self._wait_counter > self.cfg.move_duration_steps

            if is_at_destination or is_timed_out:
                # Before advancing, commit the final placement orientation to the main state.
                self.current_grasp_orientation = self._place_orn.as_quat()
                # We can now transition directly to placing the object.
                self._advance_state("PREPARE_PLACE")

        elif self._state == "PREPARE_PLACE":
            """
            NEW, ROBUST STATE: Corrects both position and orientation to a precise
            hovering pose above the goal before the final descent.
            """
            # === STATE ENTRY LOGIC (runs only on the first step) ===
            if self._wait_counter == 1:
                # 1. Define the single, ideal 7D pre-placement pose for the END-EFFECTOR.
                
                # --- START OF CENTERING FIX ---
                
                # The final desired position for the OBJECT'S CENTER.
                object_final_pos = np.array([
                    goal_pos_world[0],
                    goal_pos_world[1],
                    self.table_surface_z + object_half_height # Z-pos of object center on table
                ])

                # To achieve this, the TCP must be positioned above the object's final center
                # by half the object's height. This is our hover target for the TCP.
                tcp_hover_pos = np.array([
                    object_final_pos[0],
                    object_final_pos[1],
                    object_final_pos[2] + object_half_height + self.cfg.hover_height
                ])
                # --- END OF CENTERING FIX ---

                # The target orientation is the one calculated for placement.
                goal_orn_world = expert_obs["goal_orn_world"]
                target_orn = self._calculate_aligned_orientation(
                    goal_orn_world, ee_pose_world[3:]
                )
                
                # 2. Store this fixed 7D pose and reset the stability counter.
                self._final_hover_pose = np.concatenate([tcp_hover_pos, target_orn])
                self._orientation_stable_counter = 0

            # === CONTINUOUS LOGIC (runs every step) ===
            
            # Command the robot to go to the STORED final hover pose.
            self._target_pose_7d = self._final_hover_pose
            self._gripper_action = -1.0  # Keep gripper closed

            # === ROBUST TRANSITION LOGIC ===
            
            # Condition 1: Is the robot at the target pose and stable?
            pos_error = np.linalg.norm(ee_pos - self._final_hover_pose[:3])
            
            R_current = R.from_quat(ee_pose_world[3:])
            R_target = R.from_quat(self._final_hover_pose[3:])
            angular_distance = (R_target.inv() * R_current).magnitude()
            
            is_at_pose = (pos_error < self.cfg.pos_tolerance) and \
                         (angular_distance < self.cfg.orn_tolerance_rad)

            if is_at_pose:
                self._orientation_stable_counter += 1
            else:
                self._orientation_stable_counter = 0
            
            is_stable = self._orientation_stable_counter > 5

            # Condition 2: Timeout safety net.
            is_timed_out = self._wait_counter > 60

            if is_stable or is_timed_out:
                if is_timed_out:
                    print("WARN: PREPARE_PLACE timed out.")
                # Commit the final, correct orientation before descending.
                self.current_grasp_orientation = self._final_hover_pose[3:].copy()
                self._advance_state("DESCEND_TO_PLACE")

        elif self._state == "DESCEND_TO_PLACE": 
            # --- START OF FINAL, EDGE-ALIGNED PLACEMENT LOGIC ---
            
            # Get the full sizes of the object and goal
            goal_size = expert_obs["goal_size_world"]
            object_size = self.object.size

            # Calculate the required offset to align the "bottom-left" (min-X, min-Y) edges
            x_offset = (goal_size[0] - object_size[0]) / 2.0
            y_offset = (goal_size[1] - object_size[1]) / 2.0
            
            # The target for the OBJECT'S CENTER is the goal's center minus the offsets.
            # (Assuming +X is right, +Y is forward)
            object_target_xy = np.array([
                goal_pos_world[0] - x_offset,
                goal_pos_world[1] - y_offset
            ])

            # The Z-height for the object's CENTER when it rests on the table.
            object_center_on_table_z = self.table_surface_z + object_half_height

            # To place the object correctly, the TCP (end-effector) must be positioned
            # above the object's center by half the object's height.
            tcp_placement_z = object_center_on_table_z + object_half_height

            # Construct the final target position FOR THE END-EFFECTOR.
            target_pos = np.array([
                object_target_xy[0],
                object_target_xy[1],
                tcp_placement_z
            ])

            # Use the final, correct orientation from the previous state.
            self._target_pose_7d = np.concatenate([target_pos, self.current_grasp_orientation])
            self._gripper_action = -1.0 # Keep gripper closed

            if np.linalg.norm(ee_pos - target_pos) < self.cfg.pos_tolerance:
                self._advance_state("AWAIT_PLACEMENT_CONTACT")  
        
        elif self._state == "AWAIT_PLACEMENT_CONTACT":
            """
            NEW, ROBUST STATE: Holds the final placement pose and waits for physical
            confirmation that the object is supported by the table before releasing.
            """
            object_center_on_table_z = self.table_surface_z + object_half_height

            # To place the object correctly, the TCP (end-effector) must be positioned
            # above the object's center by half the object's height.
            tcp_placement_z = object_center_on_table_z + object_half_height

            # Construct the final target position FOR THE END-EFFECTOR.
            target_pos = np.array([
                goal_pos_world[0],
                goal_pos_world[1],
                tcp_placement_z
            ])
            # 1. Command the robot to hold the final placement pose.
            # We use the pose from the previous state's target for stability.
            self._target_pose_7d = np.concatenate([target_pos, self.current_grasp_orientation])
            self._gripper_action = -1.0 # Keep holding

            # 2. Check for physical confirmation of placement.
            # When the object is supported by the table, the force on the gripper will drop.
            # We check if the object's upward velocity has stopped, indicating contact.
            object_is_supported = abs(expert_obs.get("object_vel", [0,0,0,0,0,0])[2]) < 0.005

            # We wait a few steps for physics to stabilize AND for the object to be supported.
            if self._wait_counter > 5 and object_is_supported:
                self._advance_state("RELEASE")
            
            # Fallback timeout
            elif self._wait_counter > 25:
                print("WARN: AWAIT_PLACEMENT_CONTACT timed out. Forcing release.")
                self._advance_state("RELEASE")

        elif self._state == "RELEASE":
            # Hold position and open gripper (object stays via physics)
            self._target_pose_7d = np.concatenate([ee_pos, self.current_grasp_orientation]) 
            self._gripper_action = 1.0  # Open
            
            # Advance after brief hold
            is_gripper_physically_open = np.all(gripper_qpos > self.cfg.gripper_open_threshold)
            
            # --- START OF FIX ---
            # Transition to WAIT_FOR_RELEASE, not DONE.
            if is_gripper_physically_open or self._wait_counter > 20:
                self._advance_state("WAIT_FOR_RELEASE")
            # --- END OF FIX ---

        elif self._state == "WAIT_FOR_RELEASE":
            # This state's logic is already correct and will now be executed.
            self._target_pose_7d = np.concatenate([ee_pos, self.current_grasp_orientation])
            self._gripper_action = 1.0  

            contact_is_lost = not is_grasped
            object_is_stable = np.linalg.norm(object_vel) < 0.01

            if self._wait_counter > 5 and contact_is_lost and object_is_stable:
                print("DEBUG: Release confirmed. Advancing to RETRACT.")
                self._advance_state("RETRACT")
            
            elif self._wait_counter > 30:
                print("DEBUG: WAIT_FOR_RELEASE timed out. Forcing RETRACT.")
                self._advance_state("RETRACT")


        elif self._state == "RETRACT":
            # --- START OF FULL REPLACEMENT ---
            """
            NEW, ROBUST STATE: Performs a single, controlled vertical lift to a
            safe height and then completes the task.
            """
            # === STATE ENTRY LOGIC (runs only on the first step) ===
            if self._wait_counter == 1:
                # Calculate the retract position ONCE, based on the position at the start of the retract.
                self._retract_pos = np.array([ee_pos[0], ee_pos[1], ee_pos[2] + self.cfg.hover_height])

            # === CONTINUOUS LOGIC (runs every step) ===
            
            # Command the robot to move to the STORED fixed retract position.
            self._target_pose_7d = np.concatenate([self._retract_pos, self.current_grasp_orientation])
            self._gripper_action = 1.0  # Keep gripper open

            # === TRANSITION LOGIC (position-based with timeout) ===
            
            # Transition when the end-effector reaches the target retract position.
            if np.linalg.norm(ee_pos - self._retract_pos) < self.cfg.pos_tolerance:
                self.succeeded = True
                self._advance_state("DONE")
    
  

        elif self._state == "DONE":
            # Hold final position
            self._target_pose_7d = np.concatenate([ee_pos, self._downward_quat])
            self._gripper_action = 1.0

        # Fallback: Ensure target is always valid
        if self._target_pose_7d is None:
            self._target_pose_7d = np.concatenate([ee_pos, self._downward_quat])

        # Clamp position to workspace for safety (prevents IK blowups)
        clamped_pos = self._clamp_to_workspace(self._target_pose_7d[:3])
        final_pose = np.concatenate([clamped_pos, self._target_pose_7d[3:]]).astype(np.float32)

        return final_pose, float(self._gripper_action)