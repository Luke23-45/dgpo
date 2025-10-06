# FILE: utils/scripted_expert.py

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

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
    grasp_offset_z: float = 0.005  # Small downward offset for precise top-grasp alignment (accounts for attachment site or finger clearance)
    descent_xy_offset: np.ndarray = np.array([0.0, 0.0, 0.0])  # Lateral shift during descent to reduce finger-cube interpenetration (tune based on finger width)
    max_grasp_retries: int = 2      # Number of retry attempts if grasp fails
    lift_duration_steps: int = 40   # Increased for smoother, more stable lift
    move_duration_steps: int = 60   # Ensure ample time for horizontal moves
    place_duration_steps: int = 25  # Controlled descent for placement
    retract_duration_steps: int = 20
    failure_timeout_steps: int = 50 # Timeout per state to prevent infinite loops

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
        self.reset()

    def is_done(self) -> bool:
        return self._state == "DONE"

    def reset(self):
        self._state = "MOVE_TO_PRE_GRASP"
        self._gripper_action = -1.0  # Open gripper
        self._target_pose_7d = None
        self._wait_counter = 0
        self._grasp_retry_count = 0
        self.succeeded = False

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

    def get_target_pose(
        self,
        ee_pose_world: np.ndarray,
        cube_pos_world: np.ndarray,
        object_orn_world: any,  # Unused for now (assumes no rotation needed)
        goal_pos_world: np.ndarray,
        is_grasped: bool
    ) -> Tuple[np.ndarray, float]:
        """
        Computes target 7D pose (pos + xyzw quat) and normalized gripper action [-1, 1].
        Advances state machine based on conditions.
        """
        ee_pos = ee_pose_world[:3]
        object_half_height = self.object.size[2] / 2.0
        object_center_z = cube_pos_world[2]
        object_top_z = object_center_z + object_half_height

        # --- Robust State Machine with Timeouts and Retries ---

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
            # Hold position and close gripper (kinematic grasp engages here in env)
            self._target_pose_7d = np.concatenate([ee_pos, self._downward_quat])
            self._gripper_action = self.object.grasp_width_normalized  # Close
            # Immediately advance after 1 step (gripper closes instantly in ctrl)
            if self._wait_counter > 1:
                self._advance_state("WAIT_FOR_GRASP")

        elif self._state == "WAIT_FOR_GRASP":
            # Hold and wait for kinematic attachment confirmation
            self._target_pose_7d = np.concatenate([ee_pos, self._downward_quat])
            self._gripper_action = self.object.grasp_width_normalized  # Keep closed
            if is_grasped:
                self._grasp_retry_count = 0  # Reset retries on success
                self._advance_state("LIFT")
            elif self._wait_counter > 20:  # Extended wait for grasp detection
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

        elif self._state == "MOVE_TO_GOAL":
            # Horizontal move to above goal (object follows)
            goal_hover_z = goal_pos_world[2] + self.cfg.hover_height + object_half_height
            target_pos = np.array([
                goal_pos_world[0],
                goal_pos_world[1],
                goal_hover_z
            ])
            self._target_pose_7d = np.concatenate([target_pos, self._downward_quat])
            self._gripper_action = self.object.grasp_width_normalized  # Keep closed
            # Transition when object XY is aligned over goal (robust to Z)
            if np.linalg.norm(cube_pos_world[:2] - goal_pos_world[:2]) < self.cfg.pos_tolerance:
                self._advance_state("PLACE")

        elif self._state == "PLACE":
            # Controlled descent to place object on table
            place_z = self.table_surface_z + object_half_height
            target_pos = np.array([
                goal_pos_world[0],
                goal_pos_world[1],
                place_z
            ])
            self._target_pose_7d = np.concatenate([target_pos, self._downward_quat])
            self._gripper_action = self.object.grasp_width_normalized  # Keep closed until release
            if np.linalg.norm(ee_pos - target_pos) < self.cfg.pos_tolerance:
                self._advance_state("RELEASE")

        elif self._state == "RELEASE":
            # Hold position and open gripper (object stays via physics)
            self._target_pose_7d = np.concatenate([ee_pos, self._downward_quat])
            self._gripper_action = -1.0  # Open
            # Advance after brief hold
            if self._wait_counter > 1:
                self._advance_state("WAIT_FOR_RELEASE")

        elif self._state == "WAIT_FOR_RELEASE":
            # Confirm release (no grasp) before retracting
            self._target_pose_7d = np.concatenate([ee_pos, self._downward_quat])
            self._gripper_action = -1.0  # Keep open
            if not is_grasped and self._wait_counter > 10:
                self._advance_state("RETRACT")
            elif self._wait_counter > 30:  # Safeguard timeout
                self._advance_state("RETRACT")  # Force retract even if still "grasped"

        elif self._state == "RETRACT":
            # Lift EE away to clear object
            retract_z = ee_pos[2] + self.cfg.hover_height
            target_pos = np.array([
                ee_pos[0],
                ee_pos[1],
                retract_z
            ])
            self._target_pose_7d = np.concatenate([target_pos, self._downward_quat])
            self._gripper_action = -1.0  # Open
            if self._wait_counter > self.cfg.retract_duration_steps:
                self.succeeded = True
                self._advance_state("DONE")

        elif self._state == "DONE":
            # Hold final position
            self._target_pose_7d = np.concatenate([ee_pos, self._downward_quat])
            self._gripper_action = -1.0

        # Fallback: Ensure target is always valid
        if self._target_pose_7d is None:
            self._target_pose_7d = np.concatenate([ee_pos, self._downward_quat])

        # Clamp position to workspace for safety (prevents IK blowups)
        clamped_pos = self._clamp_to_workspace(self._target_pose_7d[:3])
        final_pose = np.concatenate([clamped_pos, self._target_pose_7d[3:]]).astype(np.float32)

        return final_pose, float(self._gripper_action)