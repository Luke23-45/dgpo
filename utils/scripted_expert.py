# FILE: utils/scripted_expert.py

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

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
            self.workspace = {"x": (0.35, 0.85), "y": (-0.25, 0.25), "z": (0.30, 0.95)}

class ScriptedExpert:
    def __init__(self, cfg: ExpertConfig = ExpertConfig()):
        self.cfg = cfg
        self._target_pose_7d: Optional[np.ndarray] = None
        self._settle_counter = 0
        self._wait_counter = 0
        self.table_surface_z = 0.4
        self.reset()

    def is_done(self) -> bool:
        return self._state == "DONE"

    def reset(self):
        self._state = "MOVE_TO_PRE_GRASP"
        self._gripper_action = -1.0
        self._target_pose_7d = None
        self._settle_counter = 0
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

        def has_settled(target_pos):
            pos_error = np.linalg.norm(ee_pos - target_pos)
            if pos_error < self.cfg.pos_tolerance:
                self._settle_counter += 1
            else:
                self._settle_counter = 0
            return self._settle_counter >= 5

        def advance_state(new_state):
            self._state = new_state
            self._target_pose_7d = None
            self._settle_counter = 0

        # --- State Machine with Dynamic Targeting Fix ---

        if self._state == "MOVE_TO_PRE_GRASP":
            # Target is static, calculated once.
            if self._target_pose_7d is None:
                target_pos = cube_pos_world + np.array([0, 0, self.cfg.hover_height])
                self._target_pose_7d = np.concatenate([target_pos, downward_quat])
            self._gripper_action = -1.0
            if has_settled(self._target_pose_7d[:3]):
                advance_state("DESCEND_TO_GRASP")

        elif self._state == "DESCEND_TO_GRASP":
            # Target is static, calculated once.
            if self._target_pose_7d is None:
                target_z = self.table_surface_z + self.cfg.grasp_depth
                target_pos = np.array([cube_pos_world[0], cube_pos_world[1], target_z])
                self._target_pose_7d = np.concatenate([target_pos, downward_quat])
            self._gripper_action = -1.0
            if has_settled(self._target_pose_7d[:3]):
                advance_state("GRASP")

        elif self._state == "GRASP":
            # Target is static, calculated once.
            if self._target_pose_7d is None:
                target_z = self.table_surface_z + self.cfg.grasp_depth
                target_pos = np.array([cube_pos_world[0], cube_pos_world[1], target_z])
                self._target_pose_7d = np.concatenate([target_pos, downward_quat])
            self._gripper_action = 1.0
            advance_state("WAIT_FOR_GRASP"); self._wait_counter = 0

        elif self._state == "WAIT_FOR_GRASP":
            # Target is dynamic to help the weld.
            target_pos = cube_pos_world + np.array([0, 0, self.cfg.grasp_depth])
            self._target_pose_7d = np.concatenate([target_pos, downward_quat])
            self._gripper_action = 1.0
            self._wait_counter += 1
            if is_grasped and self._wait_counter > 5:
                advance_state("LIFT")
            elif not is_grasped and self._wait_counter > 20:
                 advance_state("DONE")

        elif self._state == "LIFT":
            # This logic is correct: set the target only once.
            if self._target_pose_7d is None:
                target_pos = ee_pose_world[:3] + np.array([0, 0, self.cfg.hover_height])
                self._target_pose_7d = np.concatenate([target_pos, downward_quat])
            
            self._gripper_action = 1.0
            
            # =========================================================================
            # VVVVVV                  INSTRUMENTATION BLOCK                    VVVVVV
            # =========================================================================
            target_pos = self._target_pose_7d[:3]
            pos_error = np.linalg.norm(ee_pos - target_pos)
            
            print("\n--- [EXPERT DEBUG | LIFT STATE] ---")
            print(f"  - Current EE Pose:  {np.round(ee_pos, 4)}")
            print(f"  - Target Lift Pose: {np.round(target_pos, 4)}")
            print(f"  - Calculated 3D Error: {pos_error:.6f}")
            print(f"  - Tolerance:             {self.cfg.pos_tolerance:.6f}")

            if pos_error < self.cfg.pos_tolerance:
                self._settle_counter += 1
                print(f"  - Error is WITHIN tolerance. Settle counter: {self._settle_counter}/5")
            else:
                self._settle_counter = 0
                print(f"  - ❌ Error is OUTSIDE tolerance. Settle counter RESET to 0.")

            if self._settle_counter >= 5:
                print("  - ✅ SETTLED! Advancing to MOVE_TO_GOAL state.")
                advance_state("MOVE_TO_GOAL")
            else:
                print("  - Waiting to settle...")
            print("-------------------------------------\n")
        
        elif self._state == "MOVE_TO_GOAL":
            # =========================================================================
            # VVVVVV                   THE COOPERATIVE FIX (2/3)                 VVVVVV
            # =========================================================================
            # The target is also DYNAMIC, guiding the held cube towards the goal.
            target_pos = goal_pos_world + np.array([0, 0, self.cfg.hover_height])
            self._target_pose_7d = np.concatenate([target_pos, downward_quat])
            self._gripper_action = 1.0
            # The transition condition is now based on the CUBE's xy position. This is
            # the most robust check for arrival.
            xy_error = np.linalg.norm(cube_pos_world[:2] - goal_pos_world[:2])
            if xy_error < self.cfg.pos_tolerance:
                self._settle_counter += 1
            else:
                self._settle_counter = 0
            if self._settle_counter >= 5:
                advance_state("PLACE")

        elif self._state == "PLACE":
            # =========================================================================
            # VVVVVV                   THE COOPERATIVE FIX (3/3)                 VVVVVV
            # =========================================================================
            # The target for placement must also be DYNAMIC to avoid fighting the weld.
            target_z = self.table_surface_z + (self.cfg.grasp_depth * 1.5)
            target_pos = np.array([goal_pos_world[0], goal_pos_world[1], target_z])
            self._target_pose_7d = np.concatenate([target_pos, downward_quat])
            self._gripper_action = 1.0
            if has_settled(self._target_pose_7d[:3]):
                advance_state("RELEASE")
        
        elif self._state == "RELEASE":
            # This state is brief and static is fine.
            if self._target_pose_7d is None: self._target_pose_7d = ee_pose_world.copy()
            self._gripper_action = -1.0
            advance_state("WAIT_FOR_RELEASE"); self._wait_counter = 0

        elif self._state == "WAIT_FOR_RELEASE":
            if self._target_pose_7d is None: self._target_pose_7d = ee_pose_world.copy()
            self._gripper_action = -1.0
            self._wait_counter += 1
            if self._wait_counter > 5:
                advance_state("RETRACT")

        elif self._state == "RETRACT":
            # This state is after the object is released, so static is fine.
            if self._target_pose_7d is None:
                # Base retraction on the goal, not the EE's potentially drifted position.
                target_pos = goal_pos_world + np.array([0, 0, self.cfg.hover_height + 0.05])
                self._target_pose_7d = np.concatenate([target_pos, downward_quat])
            self._gripper_action = -1.0
            if has_settled(self._target_pose_7d[:3]):
                advance_state("DONE")

        elif self._state == "DONE":
            if self._target_pose_7d is None: self._target_pose_7d = ee_pose_world.copy()
            self._gripper_action = -1.0

        if self._target_pose_7d is None: self._target_pose_7d = ee_pose_world.copy()
        clamped_pos = self._clamp_to_workspace(self._target_pose_7d[:3])
        final_pose = np.concatenate([clamped_pos, self._target_pose_7d[3:]])
        
        return final_pose.astype(np.float32), float(self._gripper_action)