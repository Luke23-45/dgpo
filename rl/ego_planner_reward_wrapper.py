# FILE: rl/ego_planner_reward_wrapper.py
# (State-of-the-Art, V3 - Definitive Synthesis for Ego-Planner)

"""
The definitive, state-of-the-art reward wrapper for the Ego-Planner model.

This "V3" implementation is a synthesis of two design philosophies. It adopts the
architecturally-consistent, bi-modal reward structure (Pre-Grasp / Post-Grasp)
that is required by the Ego-Planner's static-plan architecture. It then enhances
this structure with the superior, technically-advanced components from modern
robotics research, including velocity-based penalties and more detailed state
representations.

This version explicitly REJECTS overly complex concepts like multi-phase blending,
curriculum learning, and HER, which are architecturally mismatched with the
Ego-Planner. The result is a reward function that is simple where it needs to be
(in its high-level structure) and sophisticated where it matters (in its low-level
physics and penalties).

Key SOTA Features of this Definitive Version:
  - **Architectural Consistency**: A clean, bi-modal potential function that
    switches based on the `is_grasped` state, providing an unambiguous signal
    to the static-plan Ego-Planner.
  - **Stable, Dimension-Separated Potentials**: Uses robust `1 / (1 + k*x)`
    potentials and treats XY and Z dimensions separately in the post-grasp
    phase to prevent Z-axis bias and allow for fine-tuned control.
  - **Advanced Physics-Based Penalties**: Incorporates quadratic penalties on
    both action and velocity to encourage smooth, stable motion. It also
    includes a robust drop penalty and a contact penalty.
  - **Comprehensive State Extraction**: A dedicated `PhysicalState` dataclass
    pre-computes all necessary metrics, including velocities, distances, and
    orientation errors, for a clean and efficient reward calculation.
  - **Rich Diagnostics**: The `info` dictionary is heavily populated with
    diagnostic values for every component of the reward, providing maximum
    observability for debugging and analysis during training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import gymnasium as gym
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Dict as TypingDict, List

from envs.panda_env import PandaEnv

# Setup a logger for this module
log = logging.getLogger(__name__)


# ==============================================================================
# SECTION 1: CONFIGURATION & STATE REPRESENTATION (SYNTHESIZED V3)
# ==============================================================================

@dataclass
class EgoPlannerRewardConfig:
    """
    The definitive, tunable configuration for the synthesized V3 Reward Wrapper.
    """
    # --- Potential Function Coefficients (k-values) ---
    k_reach_xy: float = 5.0       # Weight for EE-to-Object XY distance
    k_reach_z_hover: float = 4.0  # Weight for achieving correct hover height over object
    k_orient: float = 3.0       # Weight for gripper palm-down alignment
    k_lift: float = 6.0         # Weight for lifting the object after grasp
    k_move_xy: float = 7.0        # Weight for Object-to-Goal XY distance
    k_move_z: float = 4.0         # Weight for maintaining hover height while moving

    # --- Sparse Event Bonuses ---
    grasp_bonus: float = 15.0     # Large, decisive bonus for the first grasp event
    success_bonus: float = 200.0    # The final "jackpot" reward for task completion

    # --- Continuous Penalties (from Advanced Research) ---
    action_penalty: float = 0.002  # Quadratic penalty on action magnitude for smoothness
    velocity_penalty: float = 0.001 # Quadratic penalty on EE and object velocity
    contact_penalty: float = 2.0    # Penalty for robot arm colliding with the table
    drop_penalty: float = 30.0      # Large penalty for dropping the object incorrectly

    # --- Potential-Based Reward Shaping (PBRS) ---
    gamma: float = 0.99  # MUST match the discount factor of the RL algorithm

    # --- Physical & Task Thresholds ---
    hover_height: float = 0.08       # Ideal height above object/goal for hovering
    lift_height_thresh: float = 0.05 # Min height to be considered "lifted"
    goal_pos_thresh: float = 0.03    # Final 3D position tolerance for success
    stable_velocity_thresh: float = 0.02 # Max object velocity for successful placement


@dataclass
class PhysicalState:
    """
    A comprehensive, pre-computed snapshot of the environment's physical state,
    synthesized from the best of both previous versions.
    """
    # Poses
    ee_pos: np.ndarray
    object_pos: np.ndarray
    goal_pos: np.ndarray
    
    # Velocities
    ee_velocity_magnitude: float
    object_velocity_magnitude: float

    # Grasp/Lift State
    is_physically_grasped: bool
    object_lift_relative: float
    was_lifted_flag: bool
    
    # Pre-computed Errors for Potentials
    dist_ee_obj_xy: float
    dist_ee_obj_z_abs: float  # Absolute Z distance for hover
    dist_obj_goal_xy: float
    dist_obj_goal_z_abs: float  # Absolute Z distance for move
    dist_obj_goal_3d: float
    angle_grasp_alignment: float # Radians from perfect palm-down


# ==============================================================================
# SECTION 2: THE DEFINITIVE EGO-PLANNER REWARD WRAPPER (V3)
# ==============================================================================

class EgoPlannerRewardWrapper(gym.Wrapper):
    """
    The definitive V3 reward wrapper, implementing a bi-modal PBRS structure
    with advanced, physics-based penalties, architecturally consistent with the
    Ego-Planner model.
    """

    def __init__(self, env: PandaEnv, cfg: EgoPlannerRewardConfig = EgoPlannerRewardConfig()):
        super().__init__(env)
        self.env: PandaEnv
        self.cfg = cfg

        # Internal state, reset at the beginning of each episode
        self._last_potential: float = 0.0
        self._initial_object_z: float = 0.0
        self._was_lifted_flag: bool = False
        self._given_grasp_bonus: bool = False
        
        # Cache MuJoCo geom IDs for efficient penalty calculation
        self._cache_geom_ids()

    def reset(self, **kwargs) -> tuple[dict, dict]:
        obs, info = self.env.reset(**kwargs)

        self._given_grasp_bonus = False
        self._was_lifted_flag = False
        self._initial_object_z = obs['object_pos_world'][2]

        state = self._extract_state(obs)
        self._last_potential = self._calculate_potential(state)

        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        obs, _, terminated, truncated, info = self.env.step(action)

        state = self._extract_state(obs)

        if not self._was_lifted_flag and state.object_lift_relative > self.cfg.lift_height_thresh:
            self._was_lifted_flag = True
        state = replace(state, was_lifted_flag=self._was_lifted_flag)

        new_potential = self._calculate_potential(state)
        dense_reward = self.cfg.gamma * new_potential - self._last_potential
        
        sparse_reward, is_success = self._calculate_sparse_reward(state)
        penalties = self._calculate_penalties(state, action)
        
        total_reward = dense_reward + sparse_reward + penalties

        self._last_potential = new_potential
        
        if is_success:
            terminated = True
            info['is_success'] = True

        info.update({
            'reward_total': total_reward,
            'r_dense_shaped': dense_reward,
            'r_sparse_event': sparse_reward,
            'r_penalty_total': penalties,
            'potential_current': new_potential,
        })
        
        return obs, total_reward, terminated, truncated, info

    # ==========================================================================
    # Private Helper Methods for Reward Calculation (Synthesized V3 Logic)
    # ==========================================================================

    def _extract_state(self, obs: dict) -> PhysicalState:
        """Extracts and computes the full V3 state representation."""
        ee_pose = obs['ee_pose_world']
        obj_pos = obs['object_pos_world']
        goal_pos = obs['goal_pos_world']
        
        return PhysicalState(
            ee_pos=ee_pose[:3],
            object_pos=obj_pos,
            goal_pos=goal_pos,
            ee_velocity_magnitude=np.linalg.norm(obs['ee_vel']),
            object_velocity_magnitude=np.linalg.norm(obs['object_vel']),
            is_physically_grasped=obs.get('is_grasped', np.array([0.0]))[0] > 0.5,
            object_lift_relative=obj_pos[2] - self._initial_object_z,
            was_lifted_flag=self._was_lifted_flag,
            dist_ee_obj_xy=np.linalg.norm(ee_pose[:2] - obj_pos[:2]),
            dist_ee_obj_z_abs=abs(ee_pose[2] - (obj_pos[2] + self.cfg.hover_height)),
            dist_obj_goal_xy=np.linalg.norm(obj_pos[:2] - goal_pos[:2]),
            dist_obj_goal_z_abs=abs(obj_pos[2] - (goal_pos[2] + self.cfg.hover_height)),
            dist_obj_goal_3d=np.linalg.norm(obj_pos - goal_pos),
            angle_grasp_alignment=self._calculate_palm_down_angle_error(ee_pose[3:])
        )

    def _calculate_potential(self, state: PhysicalState) -> float:
        """Calculates the bi-modal potential, synthesizing the best of V1 and V2."""
        if not state.is_physically_grasped:
            # --- PRE-GRASP STAGE: Guide EE to object ---
            pot_reach_xy = 1 / (1 + self.cfg.k_reach_xy * state.dist_ee_obj_xy)
            pot_hover_z = 1 / (1 + self.cfg.k_reach_z_hover * state.dist_ee_obj_z_abs)
            pot_orient = (1 - (state.angle_grasp_alignment / np.pi)) ** 2
            
            return pot_reach_xy + pot_hover_z + pot_orient
        else:
            # --- POST-GRASP STAGE: Guide object to goal ---
            lift_progress = np.clip(state.object_lift_relative / self.cfg.lift_height_thresh, 0.0, 1.0)
            pot_lift = lift_progress ** 2  # Quadratic form from V2 to encourage full lift

            pot_move_xy = 1 / (1 + self.cfg.k_move_xy * state.dist_obj_goal_xy)
            pot_move_z = 1 / (1 + self.cfg.k_move_z * state.dist_obj_goal_z_abs) # Separate Z from V2
            
            return (self.cfg.k_lift * pot_lift +
                    self.cfg.k_move_xy * pot_move_xy +
                    self.cfg.k_move_z * pot_move_z)

    def _calculate_sparse_reward(self, state: PhysicalState) -> tuple[float, bool]:
        """Calculates decisive, event-based bonuses."""
        sparse_reward = 0.0
        
        if state.is_physically_grasped and not self._given_grasp_bonus:
            sparse_reward += self.cfg.grasp_bonus
            self._given_grasp_bonus = True

        is_pos_success = state.dist_obj_goal_3d < self.cfg.goal_pos_thresh
        is_released = not state.is_physically_grasped and state.was_lifted_flag
        is_stable = state.object_velocity_magnitude < self.cfg.stable_velocity_thresh
        
        is_success = is_pos_success and is_released and is_stable
        if is_success:
            sparse_reward += self.cfg.success_bonus
            
        return sparse_reward, is_success

# In FILE: rl/ego_planner_reward_wrapper.py
# In CLASS: EgoPlannerRewardWrapper

    def _calculate_penalties(self, state: PhysicalState, action: np.ndarray) -> float:
        """Calculates advanced, physics-based penalties from V2."""
        total_penalty = 0.0

        total_penalty -= self.cfg.action_penalty * np.linalg.norm(action) ** 2
        total_penalty -= self.cfg.velocity_penalty * (state.ee_velocity_magnitude**2 + state.object_velocity_magnitude**2)
        
        # Contact penalty for arm-table collisions
        if self.table_collision_geom_id != -1:
            
            # --- START OF THE DEFINITIVE PATCH 3 ---
            data = self.env.unwrapped.data # Get the unwrapped data object

            for i in range(data.ncon):
                contact = data.contact[i]
            # --- END OF THE DEFINITIVE PATCH 3 ---

                g1_is_arm = contact.geom1 in self.robot_collision_geom_ids
                g2_is_arm = contact.geom2 in self.robot_collision_geom_ids
                g1_is_table = contact.geom1 == self.table_collision_geom_id
                g2_is_table = contact.geom2 == self.table_collision_geom_id
                if (g1_is_arm and g2_is_table) or (g2_is_arm and g1_is_table):
                    total_penalty -= self.cfg.contact_penalty
                    break

        is_dropped = state.was_lifted_flag and not state.is_physically_grasped and \
                     state.dist_obj_goal_3d > self.cfg.goal_pos_thresh
        if is_dropped:
            total_penalty -= self.cfg.drop_penalty
            self._was_lifted_flag = False

        return total_penalty

    # ==========================================================================
    # Utility Methods
    # ==========================================================================

# In FILE: rl/ego_planner_reward_wrapper.py
# In CLASS: EgoPlannerRewardWrapper

    def _cache_geom_ids(self):
        """Caches MuJoCo geom IDs for efficient collision checking."""
        try:

            self.table_collision_geom_id = mujoco.mj_name2id(self.env.unwrapped.model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")
            self.robot_collision_geom_ids = self._find_robot_geoms()
            log.info(f"Reward Wrapper V3 initialized with {len(self.robot_collision_geom_ids)} robot geoms.")
            # --- END OF THE DEFINITIVE PATCH 1 ---
        except ValueError as e:
            log.error(f"Geom caching failed: {e}. Disabling contact penalties.")
            self.table_collision_geom_id = -1
            self.robot_collision_geom_ids = []



    def _find_robot_geoms(self) -> List[int]:
        """Finds robot arm geoms, excluding the hand and fingers."""
        geoms = []
        exclude_bodies = {"hand", "left_finger", "right_finger"}
        
        # --- START OF THE DEFINITIVE PATCH 2 ---
        model = self.env.unwrapped.model # Get the unwrapped model once
        
        for i in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            body_id = model.geom_bodyid[i]
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if name and 'link' in name and body_name not in exclude_bodies:
                geoms.append(i)
        return geoms
        # --- END OF THE DEFINITIVE PATCH 2 ---

    @staticmethod
    def _calculate_palm_down_angle_error(ee_orn_xyzw: np.ndarray) -> float:
        """Calculates the angle (radians) between the gripper's palm (+Z) and world down (-Z)."""
        try:
            palm_vec_world = R.from_quat(ee_orn_xyzw).apply([0, 0, 1])
            dot_product = np.clip(np.dot(palm_vec_world, [0, 0, -1.0]), -1.0, 1.0)
            return np.arccos(dot_product)
        except Exception:
            return np.pi