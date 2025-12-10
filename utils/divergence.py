# FILE: utils/divergence.py
"""
DGPO-Foundation: Expert Divergence Calculation Utilities

This module provides functions to calculate the divergence between:
1. The policy's achieved EE pose (from env.get_ee_pose())
2. The expert's recommended EE pose (from ScriptedExpert.get_target_pose())

The divergence is used as a DENSE per-step reward signal to guide the policy
towards expert-like behavior, solving the credit assignment problem.
"""

import numpy as np
from typing import Tuple


def compute_step_divergence(
    achieved_ee_pose: np.ndarray,
    expert_ee_pose: np.ndarray,
    position_weight: float = 1.0,
    orientation_weight: float = 0.1,
) -> float:
    """
    Computes the divergence between achieved and expert EE poses at a single timestep.
    
    Args:
        achieved_ee_pose: (7,) array [x, y, z, qx, qy, qz, qw] - what the robot achieved
        expert_ee_pose: (7,) array [x, y, z, qx, qy, qz, qw] - what the expert wanted
        position_weight: Weight for position error (Euclidean distance in meters)
        orientation_weight: Weight for orientation error (radians)
    
    Returns:
        Scalar divergence score (lower is better, always >= 0)
    """
    achieved_ee_pose = np.asarray(achieved_ee_pose, dtype=np.float64).ravel()
    expert_ee_pose = np.asarray(expert_ee_pose, dtype=np.float64).ravel()
    
    if achieved_ee_pose.shape[0] < 7 or expert_ee_pose.shape[0] < 7:
        raise ValueError(f"EE poses must be 7D, got {achieved_ee_pose.shape} and {expert_ee_pose.shape}")
    
    # 1. Position Error (Euclidean)
    pos_error = np.linalg.norm(achieved_ee_pose[:3] - expert_ee_pose[:3])
    
    # 2. Orientation Error (Quaternion geodesic distance)
    quat_achieved = achieved_ee_pose[3:7]
    quat_expert = expert_ee_pose[3:7]
    
    # Normalize quaternions for safety
    quat_achieved = quat_achieved / (np.linalg.norm(quat_achieved) + 1e-8)
    quat_expert = quat_expert / (np.linalg.norm(quat_expert) + 1e-8)
    
    # Quaternion dot product (accounts for double-cover)
    dot_product = np.abs(np.dot(quat_achieved, quat_expert))
    dot_product = np.clip(dot_product, 0.0, 1.0)
    
    # Angular error in radians (geodesic distance on SO(3))
    orn_error = 2.0 * np.arccos(dot_product)
    
    # 3. Weighted sum
    divergence = position_weight * pos_error + orientation_weight * orn_error
    
    return float(divergence)


def compute_gripper_divergence(
    achieved_gripper: float,
    expert_gripper: float,
) -> float:
    """
    Computes the divergence between achieved and expert gripper states.
    
    Args:
        achieved_gripper: Normalized gripper position [-1, 1] (open to closed)
        expert_gripper: Expert's commanded gripper [-1, 1]
    
    Returns:
        Absolute difference (0 to 2)
    """
    return abs(float(achieved_gripper) - float(expert_gripper))


def compute_trajectory_divergence(
    achieved_poses: np.ndarray,
    expert_poses: np.ndarray,
    position_weight: float = 1.0,
    orientation_weight: float = 0.1,
) -> Tuple[float, np.ndarray]:
    """
    Computes mean and per-step divergence over a trajectory.
    
    Args:
        achieved_poses: (T, 7) array of achieved EE poses
        expert_poses: (T, 7) array of expert EE poses
    
    Returns:
        mean_divergence: Scalar mean divergence over trajectory
        per_step_divergence: (T,) array of per-step divergences
    """
    T = min(len(achieved_poses), len(expert_poses))
    per_step = np.zeros(T)
    
    for t in range(T):
        per_step[t] = compute_step_divergence(
            achieved_poses[t], expert_poses[t],
            position_weight, orientation_weight
        )
    
    return float(np.mean(per_step)), per_step
