# FILE: rl_reward_wrapper_v2.py
# (State-of-the-Art, V2 - Corrected Potentials, Simplified Blending, Scaled)

"""
An advanced, state-of-the-art reward wrapper (Version 2) for complex robotic
manipulation tasks, specifically addressing issues identified from analyzing
reward logs (e.g., near-zero dense rewards despite task progress).

Key Improvements in V2:
  - Robust Potential Functions: Replaced vanishing exponentials with more stable
    forms (e.g., 1 / (1 + k*dist)) providing better gradients.
  - Simplified Blending Logic: Uses Gaussian functions based on direct physical
    state measurements (distances, grasp state, lift height) for smoother and
    more predictable stage transitions, removing complex inter-potential dependencies.
  - Corrected Lift Potential: Uses a linear clamped potential for lifting.
  - Master Potential Scaling: Introduces `potential_scale` for better control
    over the magnitude of the shaped reward.
  - Enhanced Diagnostics: Includes blending weights and raw errors in the info dict.
  - Curriculum Adjustment: Anneals `potential_scale` for smoother curriculum control.
  - Improved Clarity: Added comments and minor refactoring.
"""

import gymnasium as gym
import numpy as np
from dataclasses import dataclass, replace, field
import mujoco
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Dict as TypingDict
import logging
from envs.panda_env import PandaEnv
log = logging.getLogger(__name__)

# --- Configuration (V2) ---



@dataclass
class AdvancedRewardConfig:
    """Configuration for the V2 blended reward system."""
    # --- Potential Function Coefficients (k-values) ---
    # These scale the contribution of each state component to the total potential.
    # Tuned based on analysis - aim for components to be roughly 0-1 before scaling.

    # 1. Reach Stage (Moving EE above object)
    k_reach_pos_xy: float = 3.0    # Reward closeness in XY plane
    k_reach_pos_z: float = 2.0     # Reward closeness to hover height
    k_reach_orn: float = 1.5       # Reward aligning gripper palm-down

    # 2. Grasp Stage (Descending and Closing Gripper)
    k_grasp_pos_z: float = 3.0     # Reward descending towards object top
    k_grasp_align: float = 1.0     # Reward maintaining XY alignment during descent
    # Grasp force potential removed - use sparse bonus + physical check

    # 3. Lift Stage (Raising the object)
    k_lift: float = 5.0            # Reward increasing object height

    # 4. Move Stage (Transporting object towards goal)
    k_move_pos_xy: float = 4.0     # Reward reducing XY distance to goal
    k_move_pos_z: float = 1.0      # Reward maintaining a safe transport height
    k_move_orn: float = 1.0        # Reward aligning object orientation with goal

    # 5. Place Stage (Lowering object onto goal)
    k_place_pos_z: float = 4.0     # Reward descending object towards goal Z
    k_place_align_xy: float = 2.0  # Reward maintaining XY alignment during placement
    k_place_orn_final: float = 1.5 # Reward final object orientation alignment
    k_retract: float = 2.5  

    # --- Master Scale & Discount ---
    # Scales the entire potential function *before* calculating the shaped reward.
    # Helps balance dense vs. sparse rewards. Should be tuned based on algorithm needs.
    potential_scale: float = 5.0
    # IMPORTANT: Gamma must match the RL algorithm's discount factor for correct PBRS.
    gamma: float = 0.99

    # --- Penalty Coefficients ---
    action_penalty: float = 0.001
    jerk_penalty: float = 0.005
    contact_penalty: float = 2.0
    drop_penalty: float = 20.0
    instability_penalty: float = 0.2    # Penalty on object velocity magnitude while grasped

    # --- Sparse Bonuses ---
    grasp_bonus: float = 10.0
    lift_bonus: float = 15.0
    place_bonus: float = 25.0 # Given when object is *near* goal and stable, before release
    success_bonus: float = 100.0 # Given on final successful placement

    # --- Physical & Task Thresholds ---
    hover_height: float = 0.05        # Ideal height above object/goal for hovering
    grasp_descend_height: float = 0.005 # How far EE should be above obj top for grasp
    lift_height_thresh: float = 0.04  # Min height delta to be considered "lifted"
    place_dist_thresh: float = 0.03   # Pos distance (3D) to trigger near-goal checks/bonus
    goal_pos_thresh: float = 0.02     # Final position tolerance (3D) for success
    goal_orn_thresh: float = 0.2      # Final orientation tolerance (radians) for success
    stable_velocity_thresh: float = 0.01 # Max object velocity magnitude to be considered stable

    # --- Blending Sigmas (Control smoothness of stage transitions) ---
    # Smaller sigma = sharper transition
    sigma_dist_xy: float = 0.03       # For XY distance-based blending
    sigma_dist_z: float = 0.02        # For Z distance-based blending
    sigma_lift: float = 0.02          # For lift-based blending
    RETRACT_HOME_POS = np.array([0.4, 0.0, 0.7])


@dataclass
class CurriculumConfig:
    """Configuration for annealing reward parameters (V2)."""
    total_episodes: int = 2000 # Should match RL total training episodes/iterations

    # Anneal potential scale (starts at initial_potential_scale, ends at factor * initial)
    potential_scale_anneal_end_factor: float = 0.5

    # Anneal goal tolerances (start at initial thresholds, end at factor * initial)
    goal_thresh_anneal_end_factor: float = 0.5


# --- State Representation (V2 - added fields) ---

@dataclass
class PhysicalState:
    """Comprehensive snapshot of physical state (V2)."""
    # Poses (pos, orn_xyzw)
    ee_pos: np.ndarray             # (3,) End effector position
    ee_orn_xyzw: np.ndarray        # (4,) End effector orientation
    object_pos: np.ndarray         # (3,) Object position
    object_orn_xyzw: np.ndarray    # (4,) Object orientation
    goal_pos: np.ndarray           # (3,) Goal position
    goal_orn_xyzw: np.ndarray      # (4,) Goal orientation

    # Grasp State
    is_physically_grasped: bool    # True based on env's grasp detection/force sensors
    gripper_width: float           # Current distance between fingertips

    # Physics & Task Progress
    object_vel_linear: np.ndarray  # (3,)
    object_vel_angular: np.ndarray # (3,)
    object_z_relative_to_table: float # Object Z pos - Table Z pos
    object_lift_relative: float    # Current object Z - initial object Z at reset
    was_lifted_flag: bool          # Flag if object has ever been lifted past threshold

    # Pre-computed distances & angles for efficiency
    dist_ee_obj_xy: float          # Horizontal distance EE <-> Object
    dist_ee_obj_3d: float          # 3D distance EE <-> Object
    dist_obj_goal_xy: float        # Horizontal distance Object <-> Goal
    dist_obj_goal_3d: float        # 3D distance Object <-> Goal
    angle_grasp_alignment: float   # Angle between EE palm and world down (-Z)
    angle_obj_goal_alignment: float # Angle between Object orientation and Goal orientation
    dist_ee_home_3d: float  

    # For penalty calculation
    last_action: np.ndarray = field(default_factory=lambda: np.zeros(8)) # Store last action


# --- Gaussian Blending Function ---
def gaussian_blend(x, mu, sigma):
    """Gaussian function for smooth blending, peaks at 1 when x=mu."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# --- The Wrapper (V2) ---

class AdvancedRewardWrapper(gym.Wrapper):
    """
    Applies V2 reward: corrected potentials, simplified blending, scaling.
    """

    def __init__(
        self,
        env: PandaEnv, # Type hint for clarity
        reward_cfg: AdvancedRewardConfig = AdvancedRewardConfig(),
        curriculum_cfg: CurriculumConfig = CurriculumConfig()
    ):
        super().__init__(env)
        # --- Type Hinting ---
        self.env: PandaEnv
        self.action_space: gym.spaces.Box
        self.observation_space: gym.spaces.Dict

        self.reward_cfg = reward_cfg
        self.curriculum_cfg = curriculum_cfg
        self._initial_reward_cfg = replace(reward_cfg) # Store initial for annealing

        # --- Internal state variables ---
        self._last_potential: float = 0.0
        self._last_action: np.ndarray = np.zeros(self.action_space.shape)
        self._initial_object_z: float = 0.0
        self._was_lifted_flag: bool = False
        self._current_episode: int = 0

        # One-time bonus flags per episode
        self._given_grasp_bonus = False
        self._given_lift_bonus = False
        self._given_place_bonus = False
        self.RETRACT_HOME_POS = np.array([0.4, 0.0, 0.7])

        # --- Cache environment constants for efficiency ---
        try:
            table_geom_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")
            self.TABLE_Z = self.env.data.geom_xpos[table_geom_id][2]
            # Object start Z depends on table Z and object half-height
            obj_geom_id = self.env.object_geom_id
            obj_half_height = self.env.model.geom_size[obj_geom_id][2]
            self.OBJECT_START_Z_ON_TABLE = self.TABLE_Z + obj_half_height
            log.info(f"RewardWrapperV2: Cached TABLE_Z={self.TABLE_Z:.4f}, OBJECT_START_Z_ON_TABLE={self.OBJECT_START_Z_ON_TABLE:.4f}")
        except Exception as e:
            log.exception(f"RewardWrapperV2: Failed to cache env constants: {e}. Using defaults.")
            self.TABLE_Z = 0.4 # Default fallback
            self.OBJECT_START_Z_ON_TABLE = 0.42 # Default fallback

        # Cache geom IDs for collision checking
        self.robot_collision_geom_ids: list[int] = self._get_robot_collision_geoms()
        self.table_collision_geom_id = table_geom_id

        log.info(f"AdvancedRewardWrapper initialized with {len(self.robot_collision_geom_ids)} robot collision geoms.")


    def _get_robot_collision_geoms(self) -> list[int]:
        """Identifies robot geoms used for collision penalties (excluding hand/fingers)."""
        geom_ids = []
        excluded_bodies = {"hand", "left_finger", "right_finger"} # Bodies to exclude
        excluded_geom_names = {"finger"} # Geom names to exclude (e.g., finger_0)

        for i in range(self.env.model.ngeom):
            name = mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            body_id = self.env.model.geom_bodyid[i]
            body_name = mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_BODY, body_id)

            is_excluded_body = body_name in excluded_bodies
            is_excluded_name = any(ex_name in name for ex_name in excluded_geom_names if name)

            # Include if it belongs to the robot (e.g., starts with 'link') AND is not excluded
            if name and name.startswith('link') and not is_excluded_body and not is_excluded_name:
                 geom_ids.append(i)
        return geom_ids


    def reset(self, **kwargs) -> tuple[dict, dict]:
        # Reset curriculum first
        self._current_episode += 1
        self._update_curriculum() # Apply annealed parameters for this episode

        obs, info = self.env.reset(**kwargs)

        # Reset internal state based on initial observation
        state = self._extract_state(obs, self._last_action) # Pass zero action initially
        self._initial_object_z = state.object_pos[2]
        self._last_potential = self._calculate_potential(state)[0] # (potential, diags)
        self._last_action = np.zeros(self.action_space.shape)
        self._was_lifted_flag = False

        # Reset one-time bonus flags
        self._given_grasp_bonus = False
        self._given_lift_bonus = False
        self._given_place_bonus = False

        log.debug(f"Episode {self._current_episode} reset. Initial potential: {self._last_potential:.4f}")
        # Add current config to info for logging
        info['reward_cfg'] = self.reward_cfg_dict()
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        obs, _, terminated, truncated, info = self.env.step(action)

        # --- Extract current physical state ---
        # Include the action just taken for penalty calculations
        state = self._extract_state(obs, self._last_action)

        # --- Update task progress flags ---
        # Update _was_lifted_flag *before* calculating rewards for this step
        if not self._was_lifted_flag and state.object_lift_relative > self.reward_cfg.lift_height_thresh:
            self._was_lifted_flag = True
        # Update state dataclass with the potentially updated flag
        state = replace(state, was_lifted_flag=self._was_lifted_flag)


        # --- 1. Dense Reward from Potential Shaping (PBRS) ---
        # R_dense = potential_scale * [ gamma * Phi(s_{t+1}) - Phi(s_t) ]
        new_potential, dense_diags = self._calculate_potential(state)
        dense_reward = self.reward_cfg.potential_scale * (
            self.reward_cfg.gamma * new_potential - self._last_potential
        )
        self._last_potential = new_potential # Update potential for next step

        # --- 2. Sparse Event-Based Rewards ---
        sparse_reward, is_success, sparse_diags = self._calculate_sparse_reward(state)

        # --- 3. Penalties for Regularization ---
        penalties, penalty_diags = self._calculate_penalties(state, action)

        # --- 4. Total Reward ---
        total_reward = dense_reward + sparse_reward + penalties

        # --- Update state for next step ---
        self._last_action = action.copy() # Store action for next step's jerk calc

        # --- Termination override on success ---
        if is_success:
            terminated = True
            info['is_success'] = True # Add success flag for logging/evaluation

        # --- Populate info dictionary for rich diagnostics ---
        info.update({
            'reward_total': total_reward,
            'r_dense_shaped': dense_reward,
            'r_sparse_event': sparse_reward,
            'r_penalty_total': penalties,
            'potential_total': new_potential, # Current potential Phi(s_t+1)
            'is_grasped': state.is_physically_grasped,
            'object_lift': state.object_lift_relative,
            'was_lifted': state.was_lifted_flag,
            'dist_ee_obj_3d': state.dist_ee_obj_3d,
            'dist_obj_goal_3d': state.dist_obj_goal_3d,
            'angle_obj_goal_align': state.angle_obj_goal_alignment,
            **dense_diags,
            **sparse_diags,
            **penalty_diags,
        })

        return obs, total_reward, terminated, truncated, info

    def _extract_state(self, obs: dict, last_action: np.ndarray) -> PhysicalState:
        """Extracts and computes V2 state variables from the obs dict."""
        ee_pose = obs['ee_pose_world']
        obj_pos = obs['object_pos_world']
        goal_pos = obs['goal_pos_world']
        obj_vel_6d = obs['object_vel'] # Linear (0:3), Angular (3:6)
        is_grasped = obs.get('is_grasped', np.array([0.0]))[0] > 0.5 # Handle potential missing key

        # Gripper width calculation (example, adjust indices if needed)
        # Assumes gripper_qpos holds the state of the two finger joints
        gripper_qpos = obs.get('gripper_qpos', np.array([0.04, 0.04])) # Default to open
        # Simple approximation: width is sum of joint positions
        # A more accurate calculation might involve forward kinematics if needed
        gripper_width = max(0.0, gripper_qpos[0] + gripper_qpos[1])

        # Ensure quaternions are valid (normalized or identity)
        ee_orn = self._safe_quat(ee_pose[3:])
        obj_orn = self._safe_quat(obs['object_orn_world'])
        goal_orn = self._safe_quat(obs['goal_orn_world'])

        # Calculate Z relative to table
        obj_z_rel_table = obj_pos[2] - self.TABLE_Z

        # Calculate Z relative to initial start (handle potential init issues)
        obj_lift_rel = obj_pos[2] - self._initial_object_z if hasattr(self, '_initial_object_z') else 0.0

        # Precompute distances and angles
        dist_ee_obj_xy = np.linalg.norm(ee_pose[:2] - obj_pos[:2])
        dist_ee_obj_3d = np.linalg.norm(ee_pose[:3] - obj_pos[:3])
        dist_obj_goal_xy = np.linalg.norm(obj_pos[:2] - goal_pos[:2])
        dist_obj_goal_3d = np.linalg.norm(obj_pos[:3] - goal_pos[:3])

        # Angle: EE Palm (local +Z) vs World Down (-Z)
        ee_palm_vec_world = R.from_quat(ee_orn).apply([0, 0, 1])
        world_down_vec = np.array([0, 0, -1.0])
        # Use dot product -> angle = acos(dot), range [0, pi]
        dot_grasp = np.dot(ee_palm_vec_world, world_down_vec)
        angle_grasp = np.arccos(np.clip(dot_grasp, -1.0, 1.0)) # Angle error from perfect palm-down

        # Angle: Object Orientation vs Goal Orientation
        angle_obj_goal = self._angular_distance(obj_orn, goal_orn)
        dist_ee_home_3d = np.linalg.norm(ee_pose[:3] - self.RETRACT_HOME_POS)

        return PhysicalState(
            ee_pos=ee_pose[:3], ee_orn_xyzw=ee_orn,
            object_pos=obj_pos, object_orn_xyzw=obj_orn,
            goal_pos=goal_pos, goal_orn_xyzw=goal_orn,
            is_physically_grasped=is_grasped,
            gripper_width=gripper_width,
            object_vel_linear=obj_vel_6d[:3], object_vel_angular=obj_vel_6d[3:],
            object_z_relative_to_table=obj_z_rel_table,
            object_lift_relative=obj_lift_rel,
            was_lifted_flag=self._was_lifted_flag, # Get from internal state
            dist_ee_obj_xy=dist_ee_obj_xy, dist_ee_obj_3d=dist_ee_obj_3d,
            dist_obj_goal_xy=dist_obj_goal_xy, dist_obj_goal_3d=dist_obj_goal_3d,
            angle_grasp_alignment=angle_grasp,
            angle_obj_goal_alignment=angle_obj_goal,
            dist_ee_home_3d=dist_ee_home_3d,
            last_action=last_action
        )


    def _calculate_potential(
        self, state: PhysicalState
    ) -> Tuple[float, TypingDict[str, float]]:
        """Calculates V2 potential with robust 3-phase blending logic."""

        cfg = self.reward_cfg # Local alias for brevity
        sigma_xy = cfg.sigma_dist_xy
        sigma_z = cfg.sigma_dist_z
        sigma_lift = cfg.sigma_lift

        # --- 1. Define Task Phases (Mutually Exclusive) ---
        is_pre_grasp_phase = not state.is_physically_grasped and not state.was_lifted_flag
        is_manipulation_phase = state.is_physically_grasped
        is_post_release_phase = not state.is_physically_grasped and state.was_lifted_flag

        # --- 2. Individual Potential Components (Range ~[0, 1]) ---

        # Reach: XY distance to object, Z to hover, and orientation
        pot_reach_xy = 1.0 / (1.0 + 10.0 * state.dist_ee_obj_xy)
        hover_z_target = state.object_pos[2] + cfg.hover_height
        pot_reach_z = 1.0 / (1.0 + 20.0 * abs(state.ee_pos[2] - hover_z_target))
        pot_reach_orn = max(0.0, 1.0 - state.angle_grasp_alignment / (np.pi / 2))

        # Grasp: Z to descend and maintaining XY alignment
        grasp_z_target = state.object_pos[2] + cfg.grasp_descend_height
        pot_grasp_z = 1.0 / (1.0 + 30.0 * abs(state.ee_pos[2] - grasp_z_target))
        pot_grasp_align = pot_reach_xy # Reuse XY potential

        # Lift: Relative height of object
        pot_lift = np.clip(state.object_lift_relative / (cfg.lift_height_thresh + 0.02), 0.0, 1.0)

        # Move: XY to goal, Z at hover, and object orientation
        pot_move_xy = 1.0 / (1.0 + 5.0 * state.dist_obj_goal_xy)
        move_z_target = self.TABLE_Z + (self._initial_object_z - self.TABLE_Z) + cfg.hover_height
        pot_move_z = gaussian_blend(abs(state.object_pos[2] - move_z_target), mu=0, sigma=sigma_z * 2)
        pot_move_orn = max(0.0, 1.0 - state.angle_obj_goal_alignment / (np.pi / 4))

        # Place: Z to goal, XY alignment, and final orientation
        place_z_target = self.OBJECT_START_Z_ON_TABLE
        pot_place_z = 1.0 / (1.0 + 40.0 * abs(state.object_pos[2] - place_z_target))
        pot_place_align_xy = pot_move_xy
        pot_place_orn_final = pot_move_orn

        # Retract: Distance to a safe home position (NEW)
        pot_retract = np.exp(-3.0 * state.dist_ee_home_3d)

        # --- 3. Blending Weights (Based on 3-Phase Logic) ---
        
        # Pre-Grasp Phase Weights
        w_reach = is_pre_grasp_phase * gaussian_blend(state.dist_ee_obj_xy, mu=0, sigma=sigma_xy*1.5)
        w_grasp = is_pre_grasp_phase * gaussian_blend(state.dist_ee_obj_xy, mu=0, sigma=sigma_xy)
        
        # Manipulation Phase Weights
        w_lift = is_manipulation_phase * gaussian_blend(state.object_lift_relative, mu=cfg.lift_height_thresh+0.01, sigma=sigma_lift)
        w_move = is_manipulation_phase * (1.0 - gaussian_blend(state.dist_obj_goal_xy, mu=0, sigma=sigma_xy*2))
        w_place = is_manipulation_phase * gaussian_blend(state.dist_obj_goal_xy, mu=0, sigma=sigma_xy*1.5)
        
        # Post-Release Phase Weight
        w_retract = is_post_release_phase

        # --- 4. Calculate Total Potential ---
        potential = (
            # Phase 1
            cfg.k_reach_pos_xy * pot_reach_xy * w_reach +
            cfg.k_reach_pos_z * pot_reach_z * w_reach +
            cfg.k_reach_orn * pot_reach_orn * w_reach +
            cfg.k_grasp_pos_z * pot_grasp_z * w_grasp +
            cfg.k_grasp_align * pot_grasp_align * w_grasp +

            # Phase 2
            cfg.k_lift * pot_lift * w_lift +
            cfg.k_move_pos_xy * pot_move_xy * w_move +
            cfg.k_move_pos_z * pot_move_z * w_move +
            cfg.k_move_orn * pot_move_orn * w_move +
            cfg.k_place_pos_z * pot_place_z * w_place +
            cfg.k_place_align_xy * pot_place_align_xy * w_place +
            cfg.k_place_orn_final * pot_place_orn_final * w_place +

            # Phase 3 (NEW)
            cfg.k_retract * pot_retract * w_retract
        )

        # --- 5. Diagnostics ---
        diagnostics = {
            'pot_reach_xy': pot_reach_xy, 'pot_reach_z': pot_reach_z, 'pot_reach_orn': pot_reach_orn,
            'pot_grasp_z': pot_grasp_z, 'pot_grasp_align': pot_grasp_align,
            'pot_lift': pot_lift,
            'pot_move_xy': pot_move_xy, 'pot_move_z': pot_move_z, 'pot_move_orn': pot_move_orn,
            'pot_place_z': pot_place_z, 'pot_place_align_xy': pot_place_align_xy, 'pot_place_orn_final': pot_place_orn_final,
            'pot_retract': pot_retract, # NEW
            'w_reach': w_reach, 'w_grasp': w_grasp, 'w_lift': w_lift, 'w_move': w_move, 'w_place': w_place,
            'w_retract': w_retract, # NEW
            'err_angle_grasp': state.angle_grasp_alignment,
            'err_angle_obj_goal': state.angle_obj_goal_alignment,
            'diag_is_pre_grasp': float(is_pre_grasp_phase), # NEW DIAGNOSTIC
            'diag_is_manipulation': float(is_manipulation_phase), # NEW DIAGNOSTIC
            'diag_is_post_release': float(is_post_release_phase), # NEW DIAGNOSTIC
        }

        return potential, diagnostics

    def _calculate_sparse_reward(
        self, state: PhysicalState
    ) -> Tuple[float, bool, TypingDict[str, float]]:
        """Calculates V2 event bonuses and checks success."""
        sparse_reward = 0.0
        diagnostics = {}
        cfg = self.reward_cfg

        # Grasp Bonus: Given ONCE when grasp is first detected.
        if state.is_physically_grasped and not self._given_grasp_bonus:
            sparse_reward += cfg.grasp_bonus
            self._given_grasp_bonus = True
            diagnostics['r_sparse_grasp'] = cfg.grasp_bonus

        # Lift Bonus: Given ONCE when object crosses lift threshold.
        if state.object_lift_relative > cfg.lift_height_thresh and not self._given_lift_bonus:
            sparse_reward += cfg.lift_bonus
            self._given_lift_bonus = True
            diagnostics['r_sparse_lift'] = cfg.lift_bonus

        # Place Bonus: Given ONCE when object is near goal AND stable before release.
        obj_velocity_mag = np.linalg.norm(state.object_vel_linear) + np.linalg.norm(state.object_vel_angular)
        is_stable_near_goal = (state.dist_obj_goal_3d < cfg.place_dist_thresh and
                               obj_velocity_mag < cfg.stable_velocity_thresh)

        if is_stable_near_goal and state.is_physically_grasped and not self._given_place_bonus:
             sparse_reward += cfg.place_bonus
             self._given_place_bonus = True
             diagnostics['r_sparse_place'] = cfg.place_bonus

        # Final Success Condition:
        is_pos_success = state.dist_obj_goal_3d < cfg.goal_pos_thresh
        # is_orn_success = state.angle_obj_goal_alignment < cfg.goal_orn_thresh
        is_released_and_lifted = not state.is_physically_grasped and state.was_lifted_flag
        # Check if object is stable on the table (low Z velocity)
        is_stable_on_table = abs(state.object_vel_linear[2]) < cfg.stable_velocity_thresh \
                             and state.object_z_relative_to_table < 0.01 # Close to table

        # is_success = (is_pos_success and is_orn_success and
        #               is_released_and_lifted and is_stable_on_table)
        is_success = (is_pos_success and
                      is_released_and_lifted and is_stable_on_table)
        if is_success:
            sparse_reward += cfg.success_bonus
            diagnostics['r_sparse_success'] = cfg.success_bonus

        # Diagnostic flags
        diagnostics['diag_is_pos_success'] = float(is_pos_success)
        diagnostics['diag_is_released_lifted'] = float(is_released_and_lifted)
        diagnostics['diag_is_stable_on_table'] = float(is_stable_on_table)


        return sparse_reward, is_success, diagnostics

    def _calculate_penalties(
        self, state: PhysicalState, current_action: np.ndarray
    ) -> Tuple[float, TypingDict[str, float]]:
        """Calculates V2 penalties."""
        diagnostics = {}
        cfg = self.reward_cfg

        # Action Norm Penalty (encourage smaller actions)
        action_penalty = -cfg.action_penalty * np.sum(np.square(current_action))
        diagnostics['r_penalty_action'] = action_penalty

        # Jerk Penalty (encourage smoother actions)
        jerk = current_action - state.last_action # Use state.last_action
        jerk_penalty = -cfg.jerk_penalty * np.sum(np.square(jerk))
        diagnostics['r_penalty_jerk'] = jerk_penalty

        # Collision Penalty (Robot links hitting table)
        contact_penalty = 0.0
        if hasattr(self.env, 'data'): # Check if env has mujoco data
            for i in range(self.env.data.ncon):
                contact = self.env.data.contact[i]
                geom1_is_robot = contact.geom1 in self.robot_collision_geom_ids
                geom2_is_robot = contact.geom2 in self.robot_collision_geom_ids
                geom1_is_table = contact.geom1 == self.table_collision_geom_id
                geom2_is_table = contact.geom2 == self.table_collision_geom_id

                if (geom1_is_robot and geom2_is_table) or (geom2_is_robot and geom1_is_table):
                    contact_penalty = -cfg.contact_penalty
                    diagnostics['r_penalty_contact'] = contact_penalty
                    break # Only apply penalty once per step

        # Drop Penalty (if object was lifted and is now not grasped and not near goal)
        drop_penalty = 0.0
        # Check if dropped far from goal
        dropped_far = state.was_lifted_flag and not state.is_physically_grasped \
                      and state.dist_obj_goal_3d > cfg.place_dist_thresh * 1.5
        # Check if dropped while still high up (prevents penalizing successful placement)
        dropped_high = state.object_z_relative_to_table > 0.01

        if dropped_far and dropped_high:
            drop_penalty = -cfg.drop_penalty
            diagnostics['r_penalty_drop'] = drop_penalty
            self._was_lifted_flag = False # Reset lift flag if dropped badly

        # Instability Penalty (Object shaking while grasped)
        instability_penalty = 0.0
        if state.is_physically_grasped:
            obj_velocity_mag = np.linalg.norm(state.object_vel_linear) + np.linalg.norm(state.object_vel_angular)
            # Only penalize if velocity exceeds a threshold, allows slow movements
            if obj_velocity_mag > cfg.stable_velocity_thresh * 5:
                 instability_penalty = -cfg.instability_penalty * (obj_velocity_mag - cfg.stable_velocity_thresh*5)
                 diagnostics['r_penalty_instability'] = instability_penalty

        total_penalty = (
            action_penalty + jerk_penalty + contact_penalty +
            drop_penalty + instability_penalty
        )

        return total_penalty, diagnostics

    def _update_curriculum(self):
        """Anneals V2 reward parameters based on episode count."""
        # Calculate curriculum progress (0.0 to 1.0)
        progress = 0.0
        if self.curriculum_cfg.total_episodes > 0:
             progress = min(1.0, self._current_episode / self.curriculum_cfg.total_episodes)

        # Anneal potential_scale
        initial_scale = self._initial_reward_cfg.potential_scale
        final_scale = initial_scale * self.curriculum_cfg.potential_scale_anneal_end_factor
        self.reward_cfg.potential_scale = initial_scale + (final_scale - initial_scale) * progress

        # Anneal goal tolerances (make them stricter)
        initial_pos_thresh = self._initial_reward_cfg.goal_pos_thresh
        final_pos_thresh = initial_pos_thresh * self.curriculum_cfg.goal_thresh_anneal_end_factor
        self.reward_cfg.goal_pos_thresh = initial_pos_thresh + (final_pos_thresh - initial_pos_thresh) * progress

        initial_orn_thresh = self._initial_reward_cfg.goal_orn_thresh
        final_orn_thresh = initial_orn_thresh * self.curriculum_cfg.goal_thresh_anneal_end_factor
        self.reward_cfg.goal_orn_thresh = initial_orn_thresh + (final_orn_thresh - initial_orn_thresh) * progress

        # Log annealed values periodically
        if self._current_episode % 100 == 0: # Log every 100 episodes
            log.info(f"Curriculum Ep {self._current_episode}/{self.curriculum_cfg.total_episodes} (Progress: {progress:.2f}):")
            log.info(f"  potential_scale: {self.reward_cfg.potential_scale:.4f}")
            log.info(f"  goal_pos_thresh: {self.reward_cfg.goal_pos_thresh:.4f}")
            log.info(f"  goal_orn_thresh: {self.reward_cfg.goal_orn_thresh:.4f}")

    # --- Static Helper Methods ---
    @staticmethod
    def _angular_distance(q1_xyzw: np.ndarray, q2_xyzw: np.ndarray) -> float:
        """Calculates angular distance (radians) between two quaternions."""
        try:
            # Ensure they are ndarrays
            q1 = np.asarray(q1_xyzw)
            q2 = np.asarray(q2_xyzw)
            # Normalize inputs defensively
            q1 /= np.linalg.norm(q1)
            q2 /= np.linalg.norm(q2)
            dot_product = np.clip(np.dot(q1, q2), -1.0, 1.0)
            # Angle is 2 * acos(|dot|)
            angle = 2.0 * np.arccos(abs(dot_product))
            return angle
        except (ValueError, TypeError, ZeroDivisionError):
             log.warning(f"Error calculating angular distance for q1={q1_xyzw}, q2={q2_xyzw}. Returning pi.", exc_info=True)
             return np.pi # Max distance on error

    @staticmethod
    def _safe_quat(q_xyzw: np.ndarray) -> np.ndarray:
        """Normalizes a quaternion or returns identity if norm is near zero."""
        q = np.asarray(q_xyzw)
        norm = np.linalg.norm(q)
        if norm < 1e-6:
            # log.warning(f"Input quaternion norm is near zero: {q}. Returning identity.")
            return np.array([0.0, 0.0, 0.0, 1.0]) # Identity: [x, y, z, w]
        return q / norm

    def reward_cfg_dict(self) -> dict:
         """Returns the current reward config as a dictionary for logging."""
         # Use vars() for dataclasses
         return vars(self.reward_cfg)
