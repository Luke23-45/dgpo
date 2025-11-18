# FILE: utils/reward_functions.py
# (Definitive, SOTA, Production-Grade, Geometrically-Aware Version 3.1 - Robust Patch)

"""
Reward Functions Module.

This module defines the "Ground Truth" success metrics for the robotic task.
It serves as the authoritative source of signal for the Advantage Calculator.

Design Philosophy (SOTA):
1.  **Geometric Invariance**: Rewards physical correctness (e.g., "Is the gripper pointing down?")
    rather than strict value matching (e.g., "Is the quaternion exactly [0,1,0,0]?"),
    allowing for valid solutions under domain randomization (like object yaw).
2.  **Phase-Based Logic**: Recognizes distinct task phases (Reach, Grasp, Transport)
    to apply context-appropriate shaping without conflicting gradients.
3.  **NumPy Optimization**: Uses raw vector math instead of heavy libraries (Scipy)
    inside loops to ensure high-throughput processing of massive datasets.
4.  **Robust Data Access**: Handles partial/corrupt observation dictionaries gracefully
    via defensive getter methods.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np

# Setup a logger for the module
logger = logging.getLogger(__name__)


@dataclass
class ProprioceptionIndices:
    """
    Defines the slice indices for a 22-dim proprioception vector (Panda standard).
    Vector Layout: [qpos(7), qvel(7), torque(7) - OR - touch/force(8)]
    This maps specifically to the panda_env.py construction:
    [qpos(7), qvel(7), touch(2), force(6)]
    """
    QPOS_START: int = 0
    QPOS_END: int = 7
    QVEL_START: int = 7
    QVEL_END: int = 14
    TOUCH_START: int = 14
    TOUCH_END: int = 16
    FORCE_START: int = 16
    FORCE_END: int = 22

INDICES = ProprioceptionIndices()


@dataclass
class RewardConfig:
    """
    Configuration for Reward Shaping.
    """
    # --- Phase 1: Reaching & Alignment ---
    reach_dist_weight: float = 2.0          # Cost per meter of distance to object
    reach_align_weight: float = 1.0         # Reward for pointing gripper at object
    reach_palm_down_weight: float = 0.5     # Reward for keeping gripper vertical (Palm Down)
    pre_grasp_stability_weight: float = 1.5 # Bonus for slowing down near object

    # --- Phase 2: Grasping ---
    grasp_event_bonus: float = 5.0          # Transition reward for establishing contact
    grasp_slip_penalty: float = -20.0       # Penalty for dropping object after lift

    # --- Phase 3: Transporting & Placing ---
    transport_dist_weight: float = 3.0      # Cost per meter of distance to goal
    placement_event_bonus: float = 50.0     # Huge sparse reward for completion
    placement_stability_weight: float = 2.0 # Bonus for stable approach to goal

    # --- Global Costs ---
    time_penalty: float = 0.02              # Encourages speed
    control_penalty: float = 0.001          # Penalizes high velocity (smoothness)

    # --- Physical Constants ---
    table_height: float = 0.40
    lift_threshold: float = 0.03            # How high above table counts as "lifted"
    goal_dist_threshold: float = 0.04       # Distance to count as "placed"
    stability_vel_threshold: float = 0.05   # Velocity considered "stopped"
    
    # Geometric Vectors (in Gripper Frame)
    local_approach_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))


# --- Optimized NumPy Math Helpers (No Scipy Overhead) ---

def quat_apply(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """
    Rotates vector `vec` by quaternion `quat` [x, y, z, w].
    Pure NumPy implementation for speed.
    """
    x, y, z, w = quat
    vx, vy, vz = vec
    
    t0 = 2.0 * (w * vx + y * vz - z * vy)
    t1 = 2.0 * (w * vy + z * vx - x * vz)
    t2 = 2.0 * (w * vz + x * vy - y * vx)
    
    return np.array([
        vx + w * t0 + y * t2 - z * t1,
        vy + w * t1 + z * t0 - x * t2,
        vz + w * t2 + x * t1 - y * t0
    ])


def _safe_get(d: Dict, key: str, default: np.ndarray) -> np.ndarray:
    """
    Robustly retrieves a numpy array from a dictionary.
    Handles cases where key is missing OR value is explicitly None.
    """
    val = d.get(key)
    if val is None:
        return default
    return val


def calculate_rewards_for_episode(
    episode_obs_list: List[Dict],
    config: RewardConfig = RewardConfig()
) -> np.ndarray:
    """
    [SOTA V3.1] Calculates rewards for an entire episode.
    
    Includes defensive checks against missing/None data modalities.
    """
    rewards = []
    num_steps = len(episode_obs_list)
    
    # World Vectors
    world_down = np.array([0.0, 0.0, -1.0])

    for t in range(num_steps):
        current_obs = episode_obs_list[t]
        previous_obs = episode_obs_list[t - 1] if t > 0 else None
        reward_t = 0.0

        # --- 1. Safe State Extraction ---
        
        # Safe extraction of EE Pose
        ee_pose = _safe_get(current_obs, 'ee_pose_world', np.zeros(7, dtype=np.float32))
        ee_pos, ee_quat = ee_pose[:3], ee_pose[3:]
        
        # Normalize quaternion to prevent drift errors
        norm = np.linalg.norm(ee_quat)
        if norm > 1e-6:
            ee_quat = ee_quat / norm

        obj_pos = _safe_get(current_obs, 'object_pos_world', np.zeros(3, dtype=np.float32))
        goal_pos = _safe_get(current_obs, 'goal_pos_world', np.zeros(3, dtype=np.float32))
        
        # Safe extraction of Grasp State
        # Crash Fix: Handle explicitly None values if loader defaults failed
        is_grasped_arr = _safe_get(current_obs, 'is_grasped', np.array([0.0]))
        is_grasped = bool(is_grasped_arr[0] > 0.5)
        
        # Safe extraction of Proprioception
        proprio = _safe_get(current_obs, 'proprio', np.zeros(22, dtype=np.float32))
        
        # Ensure proprio has enough elements before slicing
        if len(proprio) >= INDICES.QVEL_END:
            ee_velocity = np.linalg.norm(proprio[INDICES.QVEL_START:INDICES.QVEL_END])
        else:
            ee_velocity = 0.0

        # Historical Context
        if previous_obs:
            prev_grasped_arr = _safe_get(previous_obs, 'is_grasped', np.array([0.0]))
            prev_is_grasped = bool(prev_grasped_arr[0] > 0.5)
        else:
            prev_is_grasped = False

        # --- 2. Logic Branching (Phase-Based) ---

        # PHASE: REACH (Not Holding)
        if not is_grasped and not prev_is_grasped:
            # A. Distance to Object
            dist = np.linalg.norm(ee_pos - obj_pos)
            reward_t -= config.reach_dist_weight * dist
            
            # B. Alignment
            target_vec = obj_pos - ee_pos
            dist_norm = np.linalg.norm(target_vec)
            if dist_norm > 1e-4:
                target_vec /= dist_norm
                curr_approach = quat_apply(ee_quat, config.local_approach_axis)
                alignment = np.dot(curr_approach, target_vec)
                reward_t += config.reach_align_weight * max(0.0, alignment)

            # C. Palm Down Constraint
            curr_approach = quat_apply(ee_quat, config.local_approach_axis)
            palm_down_score = np.dot(curr_approach, world_down)
            if palm_down_score > 0:
                 reward_t += config.reach_palm_down_weight * palm_down_score

            # D. Pre-Grasp Stability
            if dist < 0.05 and ee_velocity < config.stability_vel_threshold:
                reward_t += config.pre_grasp_stability_weight

        # PHASE: TRANSPORT (Holding)
        elif is_grasped:
            # A. Grasp Success Bonus
            if not prev_is_grasped:
                reward_t += config.grasp_event_bonus
            
            # B. Transport Distance
            dist = np.linalg.norm(obj_pos - goal_pos)
            reward_t -= config.transport_dist_weight * dist
            
            # C. Placement Stability
            if dist < config.goal_dist_threshold and ee_velocity < config.stability_vel_threshold:
                reward_t += config.placement_stability_weight

        # PHASE: PLACEMENT (Just Released)
        elif not is_grasped and prev_is_grasped:
            # Fallback to current obj pos if previous unavailable
            prev_obj = _safe_get(previous_obs, 'object_pos_world', obj_pos)
            final_dist = np.linalg.norm(prev_obj - goal_pos)
            
            if final_dist < config.goal_dist_threshold:
                reward_t += config.placement_event_bonus

        # --- 3. Failure Handling: Slip Detection ---
        if prev_is_grasped and not is_grasped:
            prev_obj = _safe_get(previous_obs, 'object_pos_world', np.zeros(3))
            prev_obj_z = prev_obj[2]
            if prev_obj_z > (config.table_height + 0.05): 
                reward_t += config.grasp_slip_penalty

        # --- 4. Regularization ---
        reward_t -= config.time_penalty
        reward_t -= config.control_penalty * (ee_velocity ** 2)

        rewards.append(reward_t)

    return np.array(rewards, dtype=np.float32)