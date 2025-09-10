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
    workspace:    dict  = None

    def __post_init__(self):
        if self.workspace is None:
            self.workspace = {"x": (0.35, 0.85), "y": (-0.25, 0.25), "z": (0.30, 0.95)}

class ScriptedExpert:
    """
    A final, robust, state-machine-based expert. This version uses timed actions
    for dynamic movements (lift, move) to ensure it never gets stuck, and
    position-based checks for static goals (pre-grasp, place).
    """
    def __init__(self, object_profile: ObjectProfile, cfg: ExpertConfig = ExpertConfig()):
        self.cfg = cfg
        self.object = object_profile
        self._target_pose_7d: Optional[np.ndarray] = None
        self._wait_counter = 0
        self.table_surface_z = 0.4
        self.reset()

    def is_done(self) -> bool:
        return self._state == "DONE"

    def reset(self):
        self._state = "MOVE_TO_PRE_GRASP"
        self._gripper_action = -1.0
        self._target_pose_7d = None
        self._wait_counter = 0

    def get_state(self) -> str:
        return self._state
    
    def _clamp_to_workspace(self, pos: np.ndarray) -> np.ndarray:
        pos = pos.copy()
        pos[0] = np.clip(pos[0], *self.cfg.workspace["x"])
        pos[1] = np.clip(pos[1], *self.cfg.workspace["y"])
        pos[2] = np.clip(pos[2], *self.cfg.workspace["z"])
        return pos

    def get_target_pose(
        self,
        ee_pose_world: np.ndarray,
        cube_pos_world: np.ndarray,
        goal_pos_world: np.ndarray,
        is_grasped: bool 
    ) -> Tuple[np.ndarray, float]:
        
        ee_pos = ee_pose_world[:3]
        downward_quat = np.array([0.0, 1.0, 0.0, 0.0])
        object_half_height = self.object.size[2] / 2.0
        object_top_z = cube_pos_world[2] + object_half_height

        def advance_state(new_state):
            self._state = new_state
            self._wait_counter = 0 # Reset wait counter on every state change

        # --- Final, Robust State Machine ---

        if self._state == "MOVE_TO_PRE_GRASP":
            target_pos = np.array([cube_pos_world[0], cube_pos_world[1], object_top_z + self.cfg.hover_height])
            self._target_pose_7d = np.concatenate([target_pos, downward_quat])
            self._gripper_action = -1.0
            if np.linalg.norm(ee_pos - target_pos) < self.cfg.pos_tolerance:
                advance_state("DESCEND_TO_GRASP")

        elif self._state == "DESCEND_TO_GRASP":
            target_pos = np.array([cube_pos_world[0], cube_pos_world[1], cube_pos_world[2]])
            self._target_pose_7d = np.concatenate([target_pos, downward_quat])
            self._gripper_action = -1.0
            if np.linalg.norm(ee_pos - target_pos) < self.cfg.pos_tolerance:
                advance_state("GRASP")

        elif self._state == "GRASP":
            self._target_pose_7d = ee_pose_world.copy()
            self._gripper_action = self.object.grasp_width_normalized
            advance_state("WAIT_FOR_GRASP")

        elif self._state == "WAIT_FOR_GRASP":
            self._target_pose_7d = ee_pose_world.copy()
            self._gripper_action = self.object.grasp_width_normalized
            self._wait_counter += 1
            if is_grasped and self._wait_counter > 5:
                advance_state("LIFT")
            elif not is_grasped and self._wait_counter > 20:
                advance_state("DONE")

        elif self._state == "LIFT":
            target_pos = np.array([ee_pos[0], ee_pos[1], object_top_z + self.cfg.hover_height])
            self._target_pose_7d = np.concatenate([target_pos, downward_quat])
            self._gripper_action = self.object.grasp_width_normalized
            self._wait_counter += 1
            if self._wait_counter > 30: # Lift for 30 steps
                advance_state("MOVE_TO_GOAL")

        elif self._state == "MOVE_TO_GOAL":
            target_pos = goal_pos_world + np.array([0, 0, self.cfg.hover_height + object_half_height])
            self._target_pose_7d = np.concatenate([target_pos, downward_quat])
            self._gripper_action = self.object.grasp_width_normalized
            # Transition once the CUBE is aligned over the goal, which is robust.
            if np.linalg.norm(cube_pos_world[:2] - goal_pos_world[:2]) < self.cfg.pos_tolerance:
                advance_state("PLACE")

        elif self._state == "PLACE":
            target_z = self.table_surface_z + object_half_height
            target_pos = np.array([goal_pos_world[0], goal_pos_world[1], target_z])
            self._target_pose_7d = np.concatenate([target_pos, downward_quat])
            self._gripper_action = self.object.grasp_width_normalized
            if np.linalg.norm(ee_pos - target_pos) < self.cfg.pos_tolerance:
                advance_state("RELEASE")
        
        elif self._state == "RELEASE":
            self._target_pose_7d = ee_pose_world.copy()
            self._gripper_action = -1.0
            advance_state("WAIT_FOR_RELEASE")

        elif self._state == "WAIT_FOR_RELEASE":
            self._target_pose_7d = ee_pose_world.copy()
            self._gripper_action = -1.0
            self._wait_counter += 1
            if not is_grasped and self._wait_counter > 10:
                advance_state("RETRACT")

        elif self._state == "RETRACT":
            target_pos = ee_pose_world[:3] + np.array([0, 0, self.cfg.hover_height])
            self._target_pose_7d = np.concatenate([target_pos, downward_quat])
            self._gripper_action = -1.0
            self._wait_counter += 1
            if self._wait_counter > 15:
                advance_state("DONE")
            
        elif self._state == "DONE":
            self._target_pose_7d = ee_pose_world.copy()

        if self._target_pose_7d is None:
            self._target_pose_7d = ee_pose_world.copy()
        
        clamped_pos = self._clamp_to_workspace(self._target_pose_7d[:3])
        final_pose = np.concatenate([clamped_pos, self._target_pose_7d[3:]])
        
        return final_pose.astype(np.float32), float(self._gripper_action)