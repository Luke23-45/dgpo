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
    move_to_pre_grasp_duration: int = 80
    prepare_gripper_duration: int = 30
    descend_to_grasp_duration: int = 50
    lift_duration_steps: int = 40
    move_to_goal_duration: int = 90
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

        # --- STATE MACHINE RE-ARCHITECTED FOR DELTA CONTROL ---

        if self._state == "MOVE_TO_PRE_GRASP":
            if self._wait_counter == 1: # On state entry
                target_pos = np.array([cube_pos_world[0], cube_pos_world[1], object_top_z + self.cfg.hover_height])
                self._stored_target_pose = np.concatenate([target_pos, self._downward_quat])
            
            self._target_pose_7d = self._stored_target_pose
            self._gripper_action = 1.0  # Open gripper in preparation

            if self._wait_counter > self.cfg.move_to_pre_grasp_duration:
                self._advance_state("PREPARE_GRIPPER")

        elif self._state == "PREPARE_GRIPPER":
            if self._wait_counter == 1:
                # Keep the position from the previous state, but calculate the correct grasp orientation
                pre_grasp_pos = self._target_pose_7d[:3].copy()
                self._target_orn = self._calculate_aligned_orientation(object_orn_world, ee_pose_world[3:])
                self._stored_target_pose = np.concatenate([pre_grasp_pos, self._target_orn])
                
                # Store the fixed descent target to avoid chasing a moving block
                grasp_z = object_top_z - self.cfg.grasp_offset_z
                descent_xy = cube_pos_world[:2] + self.cfg.descent_xy_offset[:2]
                self._descent_target_pos = np.array([descent_xy[0], descent_xy[1], grasp_z])
            
            self._target_pose_7d = self._stored_target_pose
            self._gripper_action = 1.0 # Command open

            # Wait for orientation to settle and gripper to physically open
            is_gripper_open = np.all(gripper_qpos > self.cfg.gripper_open_threshold)
            if self._wait_counter > self.cfg.prepare_gripper_duration and is_gripper_open:
                self.current_grasp_orientation = self._target_orn.copy()
                self._advance_state("DESCEND_TO_GRASP")

        elif self._state == "DESCEND_TO_GRASP":
            self._target_pose_7d = np.concatenate([self._descent_target_pos, self.current_grasp_orientation])
            self._gripper_action = 1.0  # Keep open

            if self._wait_counter > self.cfg.descend_to_grasp_duration:
                self._advance_state("GRASP")

        elif self._state == "GRASP":
            self._target_pose_7d = np.concatenate([ee_pos, self.current_grasp_orientation]) 
            self._gripper_action = -1.0 # Command close

            if is_grasped:
                self._advance_state("LIFT")
        
        elif self._state == "LIFT":
            if self._wait_counter == 1:
                adaptive_h = self.adaptive_hover_height(ee_pos[:2], robot_base_pos_world[:2])
                lift_pos = ee_pos.copy() + np.array([0, 0, adaptive_h])
                self._stored_target_pose = np.concatenate([lift_pos, self.current_grasp_orientation])

            self._target_pose_7d = self._stored_target_pose
            self._gripper_action = -1.0

            if self._wait_counter > self.cfg.lift_duration_steps:
                if is_grasped: # Verify grasp is held after lift
                    self._advance_state("MOVE_TO_GOAL")
                else:
                    self._handle_failure()

        elif self._state == "MOVE_TO_GOAL":
            if self._wait_counter == 1:
                self._start_move_pos = ee_pos.copy()
                self._start_move_orn = R.from_quat(self.current_grasp_orientation)
                goal_orn_world = expert_obs["goal_orn_world"]
                self._place_orn = R.from_quat(self._calculate_aligned_orientation(goal_orn_world, ee_pose_world[3:]))
                
                goal_hover_z = goal_pos_world[2] + self.cfg.hover_height + object_half_height
                self._end_move_pos = np.array([goal_pos_world[0], goal_pos_world[1], goal_hover_z])
            
            # Interpolate position and orientation over the duration
            progress = min(self._wait_counter / self.cfg.move_to_goal_duration, 1.0)
            interp_pos = self._start_move_pos + (self._end_move_pos - self._start_move_pos) * progress
            slerp = Slerp([0, 1], R.from_quat([self._start_move_orn.as_quat(), self._place_orn.as_quat()]))
            interp_orn = slerp(progress).as_quat()
            self._target_pose_7d = np.concatenate([interp_pos, interp_orn])
            self._gripper_action = -1.0
            
            if self._wait_counter > self.cfg.move_to_goal_duration:
                self.current_grasp_orientation = self._place_orn.as_quat()
                self._advance_state("PREPARE_PLACE")

        elif self._state == "PREPARE_PLACE":
            if self._wait_counter == 1:
                goal_size = expert_obs["goal_size_world"]
                final_place_z = self.table_surface_z + self.object.size[2]
                hover_z = final_place_z + self.cfg.hover_height
                tcp_hover_pos = np.array([goal_pos_world[0], goal_pos_world[1], hover_z])
                self._stored_target_pose = np.concatenate([tcp_hover_pos, self.current_grasp_orientation])
                
            self._target_pose_7d = self._stored_target_pose
            self._gripper_action = -1.0
            
            if self._wait_counter > self.cfg.prepare_place_duration:
                self._advance_state("DESCEND_TO_PLACE")
                
        elif self._state == "DESCEND_TO_PLACE":
            if self._wait_counter == 1:
                tcp_placement_z = self.table_surface_z + self.object.size[2]
                place_pos = np.array([goal_pos_world[0], goal_pos_world[1], tcp_placement_z])
                self._stored_target_pose = np.concatenate([place_pos, self.current_grasp_orientation])

            self._target_pose_7d = self._stored_target_pose
            self._gripper_action = -1.0
            
            if self._wait_counter > self.cfg.descend_to_place_duration:
                self._advance_state("RELEASE")
        
        elif self._state == "RELEASE":
            self._target_pose_7d = np.concatenate([ee_pos, self.current_grasp_orientation]) 
            self._gripper_action = 1.0  # Command open
            
            is_gripper_physically_open = np.all(gripper_qpos > self.cfg.gripper_open_threshold)
            if self._wait_counter > 5 and is_gripper_physically_open:
                self._advance_state("RETRACT")

        elif self._state == "RETRACT":
            if self._wait_counter == 1:
                retract_pos = np.array([ee_pos[0], ee_pos[1], ee_pos[2] + self.cfg.hover_height])
                self._stored_target_pose = np.concatenate([retract_pos, self.current_grasp_orientation])
            
            self._target_pose_7d = self._stored_target_pose
            self._gripper_action = 1.0 # Keep open

            if self._wait_counter > self.cfg.retract_duration_steps:
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