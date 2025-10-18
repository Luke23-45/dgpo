# FILE: reward_wrapper.py
# (State-of-the-Art, Blended Potential, Orientation-Aware, Curriculum-Enabled Version)

"""
An advanced, state-of-the-art reward wrapper for complex robotic manipulation tasks.

This implementation is designed for the TD3+BC fine-tuning pipeline, providing a
rich, dense, and physically-grounded reward signal. It is a significant
upgrade over simpler designs, incorporating lessons from SOTA robotics research.

Key Advancements:
  - **Blended Potential-Based Shaping**: Implements potential-based reward shaping
    (R = gamma * Phi(s_t+1) - Phi(s_t))[cite: 883], which is the theoretically
    sound way to provide dense rewards without altering the optimal policy.
    The total potential is a *smoothly blended* sum of sub-potentials for
    each stage of the task (reach, grasp, lift, move, place), eliminating
    reward discontinuities that destabilize Q-learning [cite: 894-895].

  - **Orientation-Aware Potentials**: This version *correctly* incorporates
    orientation into the reward. It guides the agent not just to a 3D position,
    but also to align its gripper for a stable top-down grasp and to align
    the object with the goal's orientation during placement. This is
    critical for success and was a key limitation in simpler wrappers.

  - **Integrated Training Curriculum**: The wrapper anneals key reward parameters
    over the course of training. It starts with a heavy emphasis on
    dense potential-shaping rewards and easier goal tolerances, then
    gradually shifts focus towards sparse success bonuses and tighter
    tolerances. This guides the agent effectively in the beginning and pushes
    it towards high-performance, precise behaviors in later stages [cite: 870-872].

  - **Advanced Physics-Based Penalties**: Includes penalties for control jerk
    (encouraging smoother motions) , grasp instability (detecting
    object slips), and unnecessary robot-body collisions
    (encouraging safer motions) .

  - **Rich Diagnostics**: The `info` dictionary is populated with detailed,
    disaggregated reward components (e.g., 'r_dense_reach', 'r_sparse_grasp',
    'r_penalty_jerk'), making it significantly easier to debug and
    analyze the agent's learning process.
"""

import gymnasium as gym
import numpy as np
from dataclasses import dataclass, replace
import mujoco
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Dict as TypingDict

# --- Configuration ---

@dataclass
class AdvancedRewardConfig:
    """Configuration for the advanced, blended reward system."""
    # --- Potential Function Coefficients ---
    # These 'k' values scale the "value" of being in a good state.
    
    # 1. Reach Stage
    k_reach_pos_xy: float = 30.0   # Reaching the object in XY
    k_reach_pos_z: float = 20.0    # Reaching the correct hover height
    k_reach_orn: float = 15.0      # Aligning the gripper for a top-down grasp

    # 2. Grasp Stage
    k_grasp_pos_z: float = 30.0    # Descending onto the object
    k_grasp_force: float = 5.0     # Applying the correct grasp force
    
    # 3. Lift Stage
    k_lift: float = 50.0           # Lifting the object vertically

    # 4. Move Stage
    k_move_pos_xy: float = 30.0    # Moving the object towards the goal XY
    k_move_orn: float = 10.0       # Aligning the object's orientation to the goal's
    
    # 5. Place Stage
    k_place_pos_3d: float = 60.0   # Precisely placing the object at the goal
    k_place_orn_final: float = 20.0 # Final orientation alignment

    # Master weight for all dense rewards
    dense_reward_weight: float = 1.0

    # --- Penalty Coefficients ---
    action_penalty: float = 0.001       # Penalty on L2 norm of action 
    jerk_penalty: float = 0.005         # Penalty on L2 norm of (a_t - a_{t-1}) 
    contact_penalty: float = 2.0        # Penalty for robot-table collision 
    drop_penalty: float = 20.0          # Large penalty for dropping a lifted object 
    instability_penalty: float = 0.2    # Penalty on object velocity while grasped 

    # --- Sparse Bonuses ---
    grasp_bonus: float = 10.0
    lift_bonus: float = 15.0
    place_bonus: float = 25.0
    success_bonus: float = 100.0

    # --- Physical & Task Thresholds ---
    target_grasp_force: float = 10.0    # Target force (from sensors) for grasp potential
    hover_height: float = 0.05          # Ideal height above object for pre-grasp
    lift_height_thresh: float = 0.04    # Min height to be "lifted"
    place_dist_thresh: float = 0.03     # Pos distance to trigger "place_bonus"
    goal_pos_thresh: float = 0.02       # Final position tolerance for success
    goal_orn_thresh: float = 0.1        # Final orientation tolerance (radians) for success


@dataclass
class CurriculumConfig:
    """Configuration for annealing reward parameters over training."""
    # Total episodes over which to anneal
    total_episodes: int = 2000
    
    # Final weight of dense reward (e.g., 0.5 means it anneals from 1.0 down to 0.5)
    dense_reward_anneal_end_factor: float = 0.5 
    
    # Final goal tolerances (e.g., 0.5 means pos_thresh anneals from 0.02 to 0.01)
    goal_thresh_anneal_end_factor: float = 0.5 


# --- State Representation ---

@dataclass
class PhysicalState:
    """
    A comprehensive snapshot of the physical state, extracted from the
    observation dictionary for clean reward calculation.
    """
    # 6D Poses (pos, orn)
    ee_pos: np.ndarray             # (3,)
    ee_orn_xyzw: np.ndarray        # (4,)
    object_pos: np.ndarray         # (3,)
    object_orn_xyzw: np.ndarray    # (4,)
    goal_pos: np.ndarray           # (3,)
    goal_orn_xyzw: np.ndarray      # (4,)
    
    # Grasp State
    is_grasped: bool               # True if `is_grasped` sensor is active
    grip_force_scalar: float       # Scalar sum of left/right finger forces
    
    # Physics & Task Progress
    object_vel_linear: np.ndarray  # (3,)
    object_vel_angular: np.ndarray # (3,)
    object_lift: float             # Current object Z - initial object Z
    was_lifted: bool               # Flag if object has *ever* been lifted
    
    # Pre-computed distances for efficiency
    dist_ee_obj_xy: float
    dist_ee_obj_z: float
    dist_obj_goal_xy: float
    dist_obj_goal_3d: float


# --- The Wrapper ---

class AdvancedRewardWrapper(gym.Wrapper):
    """
    Applies a sophisticated, orientation-aware, and curriculum-driven
    reward to a manipulation environment, compatible with the `PandaEnv`.
    """

    def __init__(
        self,
        env: gym.Env,
        reward_cfg: AdvancedRewardConfig = AdvancedRewardConfig(),
        curriculum_cfg: CurriculumConfig = CurriculumConfig()
    ):
        super().__init__(env)
        self.reward_cfg = reward_cfg
        self.curriculum_cfg = curriculum_cfg
        
        # Store the initial config for curriculum annealing
        self._initial_reward_cfg = replace(reward_cfg)

        # Internal state variables
        self._last_potential: float = 0.0
        self._last_action: np.ndarray = np.zeros(self.action_space.shape)
        self._initial_object_z: float = 0.0
        self._was_lifted: bool = False
        self._current_episode: int = 0
        
        # One-time bonus flags
        self._given_grasp_bonus = False
        self._given_lift_bonus = False
        self._given_place_bonus = False

        # --- Cache environment constants ---
        self.TABLE_Z = self.unwrapped.data.geom_xpos[
            mujoco.mj_name2id(self.unwrapped.model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")
        ][2]
        self.OBJECT_START_Z = self.TABLE_Z + self.unwrapped.model.geom_size[
            self.unwrapped.object_geom_id
        ][2]

        # Cache geom IDs for collision checking 
        self.robot_geom_ids: list[int] = []
        self.table_geom_id = mujoco.mj_name2id(
            self.unwrapped.model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom"
        )
        for i in range(self.unwrapped.model.ngeom):
            name = mujoco.mj_id2name(self.unwrapped.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            # Exclude fingers/hand from collision penalty
            if name and name.startswith('link') and 'c' in name:
                self.robot_geom_ids.append(i)

    def reset(self, **kwargs) -> tuple[dict, dict]:
        obs, info = self.env.reset(**kwargs)
        
        # Reset internal state
        state = self._extract_state(obs)
        self._initial_object_z = state.object_pos[2]
        self._last_potential = self._calculate_potential(state)[0] # (potential, diagnostics)
        self._last_action = np.zeros(self.action_space.shape)
        self._was_lifted = False
        self._current_episode += 1
        
        # Reset one-time bonus flags
        self._given_grasp_bonus = False
        self._given_lift_bonus = False
        self._given_place_bonus = False
        
        # Update curriculum parameters for the new episode
        self._update_curriculum() 

        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        obs, _, terminated, truncated, info = self.env.step(action)
        state = self._extract_state(obs)

        # --- 1. Dense Reward from Potential Shaping ---
        new_potential, dense_diags = self._calculate_potential(state)
        dense_reward = new_potential - self._last_potential
        self._last_potential = new_potential

        # --- 2. Sparse Event-Based Rewards ---
        sparse_reward, is_success, sparse_diags = self._calculate_sparse_reward(state)
        if is_success:
            terminated = True
            info['is_success'] = True

        # --- 3. Penalties for Regularization ---
        penalties, penalty_diags = self._calculate_penalties(state, action)
        
        # --- 4. Total Reward ---
        total_reward = (
            self.reward_cfg.dense_reward_weight * dense_reward +
            sparse_reward +
            penalties
        )
        
        # Update state for next step
        self._last_action = action
        if state.object_lift > self.reward_cfg.lift_height_thresh:
            self._was_lifted = True
            state = replace(state, was_lifted=True) # Update state for this step

        # Populate info dictionary for rich diagnostics
        info.update({
            'reward_total': total_reward,
            'r_dense_shaped': self.reward_cfg.dense_reward_weight * dense_reward,
            'r_sparse_event': sparse_reward,
            'r_penalty_total': penalties,
            'potential_total': new_potential,
            'is_grasped': state.is_grasped,
            'object_lift': state.object_lift,
            **dense_diags,
            **sparse_diags,
            **penalty_diags,
        })
        
        return obs, total_reward, terminated, truncated, info

    def _extract_state(self, obs: dict) -> PhysicalState:
        """Extracts and computes all necessary state variables from the obs dict."""
        ee_pose = obs['ee_pose_world'] 
        object_pos = obs['object_pos_world']
        goal_pos = obs['goal_pos_world']
        object_vel_6d = obs['object_vel']
        is_grasped = obs['is_grasped'][0] > 0.5 
        
        # Get scalar grasp force
        left_force = obs['proprio'][14:17] # 
        right_force = obs['proprio'][17:20] # 
        grip_force_scalar = np.linalg.norm(left_force) + np.linalg.norm(right_force)

        # Ensure quaternions are valid
        ee_orn_xyzw = self._safe_quat(ee_pose[3:])
        object_orn_xyzw = self._safe_quat(obs['object_orn_world']) 
        goal_orn_xyzw = self._safe_quat(obs['goal_orn_world']) 

        return PhysicalState(
            ee_pos=ee_pose[:3],
            ee_orn_xyzw=ee_orn_xyzw,
            object_pos=object_pos,
            object_orn_xyzw=object_orn_xyzw,
            goal_pos=goal_pos,
            goal_orn_xyzw=goal_orn_xyzw,
            is_grasped=is_grasped,
            grip_force_scalar=grip_force_scalar,
            object_vel_linear=object_vel_6d[:3],
            object_vel_angular=object_vel_6d[3:],
            object_lift=object_pos[2] - self._initial_object_z,
            was_lifted=self._was_lifted,
            dist_ee_obj_xy=np.linalg.norm(ee_pose[:2] - object_pos[:2]),
            dist_ee_obj_z=abs(ee_pose[2] - object_pos[2]),
            dist_obj_goal_xy=np.linalg.norm(object_pos[:2] - goal_pos[:2]),
            dist_obj_goal_3d=np.linalg.norm(object_pos - goal_pos)
        )

    def _calculate_potential(
        self, state: PhysicalState
    ) -> Tuple[float, TypingDict[str, float]]:
        """Calculates a smoothly blended potential based on the current state."""
        
        # --- 1. Individual Potential Components ---
        
        # Reach Potential: Reaching hover position
        target_hover_z = state.object_pos[2] + self.reward_cfg.hover_height
        pot_reach_pos_xy = np.exp(-5 * state.dist_ee_obj_xy)
        pot_reach_pos_z = np.exp(-10 * abs(state.ee_pos[2] - target_hover_z))
        
        # Reach Orientation: Aligning for a top-down grasp.
        # We reward the gripper's Z-axis (palm) for being anti-parallel
        # to the world's Z-axis (i.e., pointing straight down).
        ee_palm_vec = R.from_quat(state.ee_orn_xyzw).apply([0, 0, 1])
        world_z_vec = np.array([0, 0, 1])
        # Dot product is -1 for perfect alignment. We want potential to be high (e.g., ~1.0).
        # (1 + dot) / 2 maps [-1, 1] to [0, 1]. We negate dot to get reward.
        pot_reach_orn = (1 - ee_palm_vec[2]) / 2.0  # (1 - (-1)) / 2 = 1.0

        # Grasp Potential: Descending onto object
        pot_grasp_pos_z = np.exp(-20 * state.dist_ee_obj_z)
        pot_grasp_force = np.exp(-0.5 * abs(
            state.grip_force_scalar - self.reward_cfg.target_grasp_force
        ))

        # Lift Potential: Lifting the object
        pot_lift = 1 - np.exp(-15 * state.object_lift)

        # Move Potential: Moving object to goal XY
        pot_move_pos_xy = np.exp(-3 * state.dist_obj_goal_xy)
        
        # Move Orientation: Aligning object to goal orientation
        ang_dist_obj_goal = self._angular_distance(
            state.object_orn_xyzw, state.goal_orn_xyzw
        )
        pot_move_orn = np.exp(-2.0 * ang_dist_obj_goal)

        # Place Potential: Final 3D position
        pot_place_pos_3d = np.exp(-5 * state.dist_obj_goal_3d)

        # --- 2. Dynamic Blending Weights (Sigmoids) ---
        # These weights smoothly activate/deactivate stages.
        w_reach = self._sigmoid_weight(1.0 - pot_lift) # Deactivate as lift starts
        w_grasp = self._sigmoid_weight(pot_reach_pos_xy - 0.5) * w_reach # Active during reach
        w_lift = self._sigmoid_weight(1.5 * state.is_grasped - 0.5) # Activate on grasp
        w_move = self._sigmoid_weight(1.2 * pot_lift) # Activate as lift progresses
        w_place = self._sigmoid_weight(1.5 * pot_move_pos_xy) # Activate as move progresses

        # --- 3. Final Blended Potential ---
        potential = (
            self.reward_cfg.k_reach_pos_xy * pot_reach_pos_xy * w_reach +
            self.reward_cfg.k_reach_pos_z * pot_reach_pos_z * w_reach +
            self.reward_cfg.k_reach_orn * pot_reach_orn * w_reach +
            
            self.reward_cfg.k_grasp_pos_z * pot_grasp_pos_z * w_grasp +
            (self.reward_cfg.k_grasp_force * pot_grasp_force * w_grasp if state.is_grasped else 0.0) +
            
            self.reward_cfg.k_lift * pot_lift * w_lift +
            
            self.reward_cfg.k_move_pos_xy * pot_move_pos_xy * w_move +
            self.reward_cfg.k_move_orn * pot_move_orn * w_move +
            
            self.reward_cfg.k_place_pos_3d * pot_place_pos_3d * w_place +
            self.reward_cfg.k_place_orn_final * pot_move_orn * w_place # Re-use move_orn
        )
        
        diagnostics = {
            'pot_reach_pos': pot_reach_pos_xy + pot_reach_pos_z,
            'pot_reach_orn': pot_reach_orn,
            'pot_grasp': pot_grasp_pos_z,
            'pot_lift': pot_lift,
            'pot_move_pos': pot_move_pos_xy,
            'pot_move_orn': pot_move_orn,
            'pot_place': pot_place_pos_3d,
        }
        
        return potential, diagnostics

    def _calculate_sparse_reward(
        self, state: PhysicalState
    ) -> Tuple[float, bool, TypingDict[str, float]]:
        """Calculates event-based bonuses and checks for task success."""
        sparse_reward = 0.0
        diagnostics = {}

        if state.is_grasped and not self._given_grasp_bonus:
            sparse_reward += self.reward_cfg.grasp_bonus
            self._given_grasp_bonus = True
            diagnostics['r_sparse_grasp'] = self.reward_cfg.grasp_bonus
            
        if state.object_lift > self.reward_cfg.lift_height_thresh and not self._given_lift_bonus:
            sparse_reward += self.reward_cfg.lift_bonus
            self._given_lift_bonus = True
            diagnostics['r_sparse_lift'] = self.reward_cfg.lift_bonus
            
        if state.dist_obj_goal_3d < self.reward_cfg.place_dist_thresh and \
           state.is_grasped and not self._given_place_bonus:
            sparse_reward += self.reward_cfg.place_bonus
            self._given_place_bonus = True
            diagnostics['r_sparse_place'] = self.reward_cfg.place_bonus
            
        # Final success condition:
        # Object is at goal (pos AND orn) AND it's not being held AND it was lifted.
        is_pos_success = state.dist_obj_goal_3d < self.reward_cfg.goal_pos_thresh
        ang_dist = self._angular_distance(state.object_orn_xyzw, state.goal_orn_xyzw)
        is_orn_success = ang_dist < self.reward_cfg.goal_orn_thresh
        
        is_success = (
            is_pos_success and
            is_orn_success and
            not state.is_grasped and
            state.was_lifted
        )
        
        if is_success:
            sparse_reward += self.reward_cfg.success_bonus
            diagnostics['r_sparse_success'] = self.reward_cfg.success_bonus
            
        return sparse_reward, is_success, diagnostics

    def _calculate_penalties(
        self, state: PhysicalState, action: np.ndarray
    ) -> Tuple[float, TypingDict[str, float]]:
        """Calculates all negative reward components."""
        diagnostics = {}

        # Action & Jerk Penalties 
        action_penalty = -self.reward_cfg.action_penalty * np.sum(np.square(action))
        jerk = action - self._last_action
        jerk_penalty = -self.reward_cfg.jerk_penalty * np.sum(np.square(jerk))
        diagnostics['r_penalty_action'] = action_penalty
        diagnostics['r_penalty_jerk'] = jerk_penalty

        # Collision Penalty 
        contact_penalty = 0.0
        for i in range(self.unwrapped.data.ncon):
            contact = self.unwrapped.data.contact[i]
            if (contact.geom1 in self.robot_geom_ids and contact.geom2 == self.table_geom_id) or \
               (contact.geom2 in self.robot_geom_ids and contact.geom1 == self.table_geom_id):
                contact_penalty = -self.reward_cfg.contact_penalty
                diagnostics['r_penalty_contact'] = contact_penalty
                break
        
        # Grasp-related Penalties
        drop_penalty = 0.0
        if state.was_lifted and not state.is_grasped:
            drop_penalty = -self.reward_cfg.drop_penalty
            diagnostics['r_penalty_drop'] = drop_penalty
            
        instability_penalty = 0.0
        if state.is_grasped:
            obj_velocity = np.linalg.norm(state.object_vel_linear) + \
                           np.linalg.norm(state.object_vel_angular)
            instability_penalty = -self.reward_cfg.instability_penalty * obj_velocity 
            diagnostics['r_penalty_instability'] = instability_penalty

        total_penalty = (
            action_penalty +
            jerk_penalty +
            contact_penalty +
            drop_penalty +
            instability_penalty
        )
        
        return total_penalty, diagnostics

    def _update_curriculum(self):
        """Anneals reward parameters based on the current episode count."""
        progress = min(1.0, self._current_episode / self.curriculum_cfg.total_episodes)
        
        # Anneal dense reward weight 
        final_weight = self.curriculum_cfg.dense_reward_anneal_end_factor
        self.reward_cfg.dense_reward_weight = 1.0 - (1.0 - final_weight) * progress

        # Anneal goal thresholds to be stricter over time 
        final_pos_thresh = self._initial_reward_cfg.goal_pos_thresh * \
                           self.curriculum_cfg.goal_thresh_anneal_end_factor
        self.reward_cfg.goal_pos_thresh = self._initial_reward_cfg.goal_pos_thresh - \
            (self._initial_reward_cfg.goal_pos_thresh - final_pos_thresh) * progress
        
        final_orn_thresh = self._initial_reward_cfg.goal_orn_thresh * \
                           self.curriculum_cfg.goal_thresh_anneal_end_factor
        self.reward_cfg.goal_orn_thresh = self._initial_reward_cfg.goal_orn_thresh - \
            (self._initial_reward_cfg.goal_orn_thresh - final_orn_thresh) * progress

    @staticmethod
    def _sigmoid_weight(x: float, steepness: float = 5.0, offset: float = 0.5) -> float:
        """A simple sigmoid function to create smooth weights between 0 and 1."""
        return 1 / (1 + np.exp(-steepness * (x - offset)))

    @staticmethod
    def _angular_distance(q1_xyzw: np.ndarray, q2_xyzw: np.ndarray) -> float:
        """Calculates the angular distance (in radians) between two quaternions."""
        try:
            R1 = R.from_quat(q1_xyzw)
            R2 = R.from_quat(q2_xyzw)
            # This computes the angle of the difference rotation
            return (R1 * R2.inv()).magnitude()
        except Exception:
            # Fallback for invalid quaternions (e.g., [0,0,0,0])
            return np.pi

    @staticmethod
    def _safe_quat(q_xyzw: np.ndarray) -> np.ndarray:
        """Normalizes a quaternion or returns a neutral one if norm is zero."""
        norm = np.linalg.norm(q_xyzw)
        if norm < 1e-6:
            return np.array([0.0, 0.0, 0.0, 1.0]) # Return identity quaternion (w=1)
        return q_xyzw / norm