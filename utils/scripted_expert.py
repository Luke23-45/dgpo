# In file: utils/scripted_expert.py
import numpy as np
from dataclasses import dataclass
from typing import Tuple

def _normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32); norm = np.linalg.norm(q)
    if norm < 1e-6: return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    return q / norm

@dataclass
class ExpertConfig:
    hover_height: float = 0.10
    grasp_depth:  float = 0.015
    pos_tolerance: float = 0.01
    workspace:    dict  = None
    def __post_init__(self):
        if self.workspace is None:
            self.workspace = {"x": (0.35, 0.85), "y": (-0.25, 0.25), "z": (0.40, 0.95)}

class ScriptedExpert:
    def __init__(self, cfg: ExpertConfig = ExpertConfig()):
        self.cfg = cfg
        self._wait_counter = 0
        self.reset()

    @property
    def done(self) -> bool:
        return self._state == "DONE"
    def is_done(self) -> bool:
        """Returns True if the expert has completed its task."""
        return self._state == "DONE"
    def reset(self):
        self._state = "MOVE_TO_PRE_GRASP"
        self._gripper_action = -1.0
        self._wait_counter = 0
    def get_state(self) -> str:
        """Returns the current state of the expert's state machine."""
        return self._state
    
    def _is_in_workspace(self, pos: np.ndarray) -> bool:
        return (self.cfg.workspace["x"][0] <= pos[0] <= self.cfg.workspace["x"][1] and
                self.cfg.workspace["y"][0] <= pos[1] <= self.cfg.workspace["y"][1] and
                self.cfg.workspace["z"][0] <= pos[2] <= self.cfg.workspace["z"][1])

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
        goal_pos_world: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        ee_pos = ee_pose_world[:3].astype(np.float32)
        ultimate_goal_pos = None

        if not self._is_in_workspace(ee_pos):
            safe_x, safe_y = np.mean(self.cfg.workspace["x"]), np.mean(self.cfg.workspace["y"])
            ultimate_goal_pos = self._clamp_to_workspace(np.array([safe_x, safe_y, ee_pos[2]]))
        else:
            if self._state == "MOVE_TO_PRE_GRASP":
                ultimate_goal_pos = cube_pos_world + np.array([0, 0, self.cfg.hover_height])
                self._gripper_action = -1.0  # Open gripper
                if np.linalg.norm(ee_pos - ultimate_goal_pos) < self.cfg.pos_tolerance:
                    self._state = "DESCEND_TO_GRASP"
            
            elif self._state == "DESCEND_TO_GRASP":
                ultimate_goal_pos = cube_pos_world + np.array([0, 0, self.cfg.grasp_depth])
                self._gripper_action = -1.0  # Keep gripper open
                if np.linalg.norm(ee_pos - ultimate_goal_pos) < self.cfg.pos_tolerance:
                    self._state = "GRASP"
            
            elif self._state == "GRASP":
                # Command the gripper to close but DON'T move the arm
                ultimate_goal_pos = ee_pos
                self._gripper_action = 1.0
                self._state = "WAIT_FOR_GRASP"  # <-- NEW: Transition to a wait state
                self._wait_counter = 0
            
            elif self._state == "WAIT_FOR_GRASP":
                # Wait for a few timesteps to ensure the gripper is fully closed
                ultimate_goal_pos = ee_pos
                self._gripper_action = 1.0
                self._wait_counter += 1
                if self._wait_counter > 5:  # Wait for 5 steps (can be tuned)
                    self._state = "LIFT"

            elif self._state == "LIFT":
                ultimate_goal_pos = ee_pos.copy()
                target_lift_z = goal_pos_world[2] + self.cfg.hover_height
                ultimate_goal_pos[2] = target_lift_z
                self._gripper_action = 1.0
                if ee_pos[2] > target_lift_z - 0.015:
                    self._state = "MOVE_TO_GOAL"
            
            elif self._state == "MOVE_TO_GOAL":
                ultimate_goal_pos = goal_pos_world + np.array([0, 0, self.cfg.hover_height])
                self._gripper_action = 1.0
                if np.linalg.norm(ee_pos - ultimate_goal_pos) < self.cfg.pos_tolerance:
                    self._state = "PLACE"
            
            elif self._state == "PLACE":
                ultimate_goal_pos = goal_pos_world
                self._gripper_action = 1.0  # Keep closed while moving into place
                if np.linalg.norm(ee_pos - ultimate_goal_pos) < self.cfg.pos_tolerance:
                    self._state = "RELEASE" # <-- NEW: Transition to a release state
            
            elif self._state == "RELEASE":
                # Command the gripper to open but DON'T move the arm
                ultimate_goal_pos = ee_pos
                self._gripper_action = -1.0
                self._state = "WAIT_FOR_RELEASE" # <-- NEW: Transition to a wait state
                self._wait_counter = 0
            
            elif self._state == "WAIT_FOR_RELEASE":
                # Wait for a few timesteps to ensure the gripper is open and object is released
                ultimate_goal_pos = ee_pos
                self._gripper_action = -1.0
                self._wait_counter += 1
                if self._wait_counter > 5: # Wait for 5 steps
                    self._state = "RETRACT"

            elif self._state == "RETRACT":
                retreat_pos = goal_pos_world + np.array([0, 0, self.cfg.hover_height + 0.05])
                ultimate_goal_pos = retreat_pos
                self._gripper_action = -1.0
                if np.linalg.norm(ee_pos - retreat_pos) < self.cfg.pos_tolerance:
                    self._state = "DONE"
            
            else: # DONE state
                ultimate_goal_pos = ee_pos
                self._gripper_action = -1.0
        clamped_goal_pos = self._clamp_to_workspace(ultimate_goal_pos)
        downward_q_xyzw = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        final_pose = np.concatenate([clamped_goal_pos, _normalize_quat_xyzw(downward_q_xyzw)])
        
        return final_pose.astype(np.float32), float(self._gripper_action)