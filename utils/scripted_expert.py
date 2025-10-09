# FILE: utils/scripted_expert.py
# REPLACE THE EXISTING DATACLASSES AND CLASS WITH THIS ENTIRE BLOCK

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
    """Configuration for the new physics-aware scripted expert policy."""
    # --- Motion Durations (in simulation steps) ---
    pre_grasp_duration: int = 80
    descend_duration: int = 50
    align_duration: int = 25
    secure_grasp_duration: int = 20
    verify_lift_duration: int = 30
    lift_duration: int = 50
    move_duration: int = 100
    place_duration: int = 40
    release_duration: int = 20
    retract_duration: int = 40

    # --- Tolerances and Offsets ---
    hover_height: float = 0.10
    verify_lift_height: float = 0.03
    
    # --- Grasping Parameters ---
    max_grasp_retries: int = 2
    
    # --- Workspace ---
    workspace: dict = None

    def __post_init__(self):
        if self.workspace is None:
            self.workspace = {"x": (0.35, 0.85), "y": (-0.25, 0.25), "z": (0.30, 0.95)}

class ScriptedExpert:
    """
    An advanced, physics-aware, state-machine-based expert for pick-and-place.
    This version uses timed transitions and tactile feedback to create robust,
    physically plausible trajectories.
    """
    def __init__(self, object_profile: ObjectProfile, cfg: ExpertConfig = ExpertConfig()):
        self.cfg = cfg
        self.object = object_profile
        self._target_pose_7d: Optional[np.ndarray] = None
        self._downward_quat = np.array([0.0, 1.0, 0.0, 0.0])  # xyzw
        self.table_surface_z = 0.4
        self.reset()

    def reset(self):
        self._state = "MOVE_TO_PRE_GRASP"
        self._gripper_action = -1.0  # Start with gripper open
        self._wait_counter = 0
        self._grasp_retry_count = 0
        self.succeeded = False
        self._last_proprio = None

    def get_state(self) -> str: return self._state
    def is_done(self) -> bool: return self._state == "DONE"
    def was_successful(self) -> bool: return self.succeeded
    
    def _advance_state(self, new_state: str):
        self._state = new_state
        self._wait_counter = 0
        
    def _handle_grasp_failure(self):
        self._grasp_retry_count += 1
        if self._grasp_retry_count <= self.cfg.max_grasp_retries:
            self._advance_state("MOVE_TO_PRE_GRASP")
            self._gripper_action = -1.0  # Ensure gripper is open for retry
        else:
            self.succeeded = False
            self._advance_state("DONE")

    def get_target_pose(
        self,
        ee_pose_world: np.ndarray,
        object_pos_world: np.ndarray,
        proprio: np.ndarray,
        goal_pos_world: np.ndarray,
        is_grasped: bool # Raw contact flag from env
    ) -> Tuple[np.ndarray, float]:
        self._last_proprio = proprio
        ee_pos = ee_pose_world[:3]
        object_half_height = self.object.size[2] / 2.0
        object_top_z = object_pos_world[2] + object_half_height
        self._wait_counter += 1

        # Continuous check: if we are supposed to be holding the object but lose contact, fail.
        if self._state in ["LIFT", "MOVE_TO_GOAL", "PLACE"] and not is_grasped:
            self._handle_grasp_failure()

        # --- Physics-Aware State Machine with Timed Transitions ---
        if self._state == "MOVE_TO_PRE_GRASP":
            target_pos = np.array([object_pos_world[0], object_pos_world[1], object_top_z + self.cfg.hover_height])
            self._target_pose_7d = np.concatenate([target_pos, self._downward_quat])
            self._gripper_action = -1.0
            if self._wait_counter > self.cfg.pre_grasp_duration:
                self._advance_state("DESCEND_TO_OBJECT")

        elif self._state == "DESCEND_TO_OBJECT":
            target_pos = np.array([object_pos_world[0], object_pos_world[1], object_top_z + 0.01])
            self._target_pose_7d = np.concatenate([target_pos, self._downward_quat])
            self._gripper_action = -1.0
            if self._wait_counter > self.cfg.descend_duration:
                self._advance_state("ALIGN_GRASP")

        elif self._state == "ALIGN_GRASP":
            self._target_pose_7d = np.concatenate([ee_pos, self._downward_quat])
            self._gripper_action = 0.8  # Command a gentle, partial close
            if is_grasped or self._wait_counter > self.cfg.align_duration:
                self._advance_state("SECURE_GRASP")

        elif self._state == "SECURE_GRASP":
            self._target_pose_7d = np.concatenate([ee_pos, self._downward_quat])
            self._gripper_action = 1.0  # Command a full squeeze
            if self._wait_counter > self.cfg.secure_grasp_duration:
                self._advance_state("VERIFY_LIFT")
        
        elif self._state == "VERIFY_LIFT":
            target_pos = ee_pos + np.array([0, 0, self.cfg.verify_lift_height])
            self._target_pose_7d = np.concatenate([target_pos, self._downward_quat])
            self._gripper_action = 1.0
            if self._wait_counter > self.cfg.verify_lift_duration:
                object_has_lifted = object_pos_world[2] > (self.table_surface_z + object_half_height + 0.01)
                if is_grasped and object_has_lifted:
                    self._advance_state("LIFT")
                else:
                    self._handle_grasp_failure()

        elif self._state == "LIFT":
            target_z = self.table_surface_z + self.object.size[2] + self.cfg.hover_height
            target_pos = np.array([ee_pos[0], ee_pos[1], target_z])
            self._target_pose_7d = np.concatenate([target_pos, self._downward_quat])
            self._gripper_action = 1.0
            if self._wait_counter > self.cfg.lift_duration:
                self._advance_state("MOVE_TO_GOAL")

        elif self._state == "MOVE_TO_GOAL":
            target_pos = np.array([goal_pos_world[0], goal_pos_world[1], ee_pos[2]])
            self._target_pose_7d = np.concatenate([target_pos, self._downward_quat])
            self._gripper_action = 1.0
            if self._wait_counter > self.cfg.move_duration:
                self._advance_state("PLACE")

        elif self._state == "PLACE":
            target_pos = np.array([goal_pos_world[0], goal_pos_world[1], self.table_surface_z + object_half_height + 0.01])
            self._target_pose_7d = np.concatenate([target_pos, self._downward_quat])
            self._gripper_action = 1.0
            if self._wait_counter > self.cfg.place_duration:
                self._advance_state("RELEASE")

        elif self._state == "RELEASE":
            self._target_pose_7d = np.concatenate([ee_pos, self._downward_quat])
            self._gripper_action = -1.0
            if self._wait_counter > self.cfg.release_duration:
                self._advance_state("RETRACT")

        elif self._state == "RETRACT":
            target_pos = ee_pos + np.array([0, 0, self.cfg.hover_height])
            self._target_pose_7d = np.concatenate([target_pos, self._downward_quat])
            self._gripper_action = -1.0
            if self._wait_counter > self.cfg.retract_duration:
                self.succeeded = True
                self._advance_state("DONE")
                
        elif self._state == "DONE":
            self._target_pose_7d = np.concatenate([ee_pos, self._downward_quat])
            self._gripper_action = -1.0

        if self._target_pose_7d is None:
            self._target_pose_7d = np.concatenate([ee_pos, self._downward_quat])

        # Clamp position to workspace for safety
        clamped_pos = self._target_pose_7d[:3].copy()
        clamped_pos[0] = np.clip(clamped_pos[0], *self.cfg.workspace["x"])
        clamped_pos[1] = np.clip(clamped_pos[1], *self.cfg.workspace["y"])
        clamped_pos[2] = np.clip(clamped_pos[2], *self.cfg.workspace["z"])
        final_pose = np.concatenate([clamped_pos, self._target_pose_7d[3:]]).astype(np.float32)

        return final_pose, float(self._gripper_action)