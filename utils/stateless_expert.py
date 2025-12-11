# FILE: utils/stateless_expert.py
"""
Stateless Expert Critic for DGPO-Foundation

Unlike ScriptedExpert (stateful FSM), this class provides a "critic" view:
Given the current observation, what would be the ideal EE target pose?

This is used for divergence reward calculation in DGPO training,
where the robot follows the POLICY (not the expert) but we want to
measure how far the policy's achieved pose is from the expert's ideal.

Key Design:
- NO internal state machine
- NO wait counters or timeouts
- Pure function: obs -> ideal_target_pose
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple


class StatelessExpertCritic:
    """
    Computes ideal target EE pose based on task phase detection from observation.
    
    Task Phases (inferred from observation, not tracked internally):
    1. APPROACH: Object not grasped, EE above object -> target = above object
    2. DESCEND: Object not grasped, EE at hover height -> target = grasp position
    3. LIFT: Object grasped, object low -> target = lift position
    4. TRANSPORT: Object grasped, object high -> target = above goal
    5. PLACE: Object grasped, EE above goal -> target = place position
    """
    
    def __init__(
        self,
        hover_height: float = 0.15,
        grasp_offset_z: float = 0.02,
        lift_height: float = 0.20,
        place_height: float = 0.05,
    ):
        self.hover_height = hover_height
        self.grasp_offset_z = grasp_offset_z
        self.lift_height = lift_height
        self.place_height = place_height
        
        # Standard downward gripper orientation
        self._downward_quat = R.from_euler('xyz', [180, 0, 0], degrees=True).as_quat()
    
    def get_ideal_pose(self, obs: dict) -> Tuple[np.ndarray, float]:
        """
        Computes ideal target pose and gripper command based on current observation.
        
        Args:
            obs: Expert observation dict with keys:
                - ee_pose_world: (7,) current EE pose
                - object_pos_world: (3,) object position
                - goal_pos_world: (3,) goal position
                - is_grasped: (1,) bool-like, is object grasped
        
        Returns:
            target_pose: (7,) [x, y, z, qx, qy, qz, qw]
            gripper_cmd: float, -1 = close, +1 = open
        """
        ee_pos = obs['ee_pose_world'][:3]
        obj_pos = obs['object_pos_world']
        goal_pos = obs['goal_pos_world']
        is_grasped = bool(obs['is_grasped'][0] > 0.5)
        
        # Detect phase from geometric relations
        ee_above_obj = np.linalg.norm(ee_pos[:2] - obj_pos[:2]) < 0.05
        ee_above_goal = np.linalg.norm(ee_pos[:2] - goal_pos[:2]) < 0.05
        obj_lifted = obj_pos[2] > 0.50  # Table height ~0.42, lifted > 0.50
        
        # Phase detection and target computation
        if not is_grasped:
            # Phase 1/2: Approach/Descend to object
            if not ee_above_obj or ee_pos[2] > obj_pos[2] + self.hover_height - 0.02:
                # APPROACH: Move to hover above object
                target_pos = np.array([
                    obj_pos[0],
                    obj_pos[1],
                    obj_pos[2] + self.hover_height
                ])
                gripper_cmd = 1.0  # Open
            else:
                # DESCEND: Move down to grasp
                target_pos = np.array([
                    obj_pos[0],
                    obj_pos[1],
                    obj_pos[2] + self.grasp_offset_z
                ])
                if ee_pos[2] > (obj_pos[2] + 0.02):
                    gripper_cmd = 1.0 # Still descending -> OPEN
                else:
                    gripper_cmd = -1.0 # At bottom -> CLOSE
        else:
            # Object is grasped
            if not obj_lifted:
                # LIFT: Lift the object
                target_pos = np.array([
                    obj_pos[0],
                    obj_pos[1],
                    obj_pos[2] + self.lift_height
                ])
                gripper_cmd = -1.0  # Keep closed
            elif not ee_above_goal:
                # TRANSPORT: Move to above goal
                target_pos = np.array([
                    goal_pos[0],
                    goal_pos[1],
                    goal_pos[2] + self.hover_height + 0.05  # Higher for clearance
                ])
                gripper_cmd = -1.0  # Keep closed
            else:
                # PLACE: Descend to place
                target_pos = np.array([
                    goal_pos[0],
                    goal_pos[1],
                    goal_pos[2] + self.place_height
                ])
                # Release if close enough
                if ee_pos[2] < goal_pos[2] + self.place_height + 0.02:
                    gripper_cmd = 1.0  # Open to release
                else:
                    gripper_cmd = -1.0  # Keep closed during descent
        
        target_pose = np.concatenate([target_pos, self._downward_quat]).astype(np.float32)
        return target_pose, float(gripper_cmd)
    
    def reset(self):
        """No-op for API compatibility. This class is stateless."""
        pass
