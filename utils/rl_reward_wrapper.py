# utils/rl_reward_wrapper.py
"""
A SOTA-aligned, staged, potential-based reward wrapper for the Panda pick-and-place task.
"""

import gymnasium as gym
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
import mujoco

# --- Configuration and State Enums ---

class RewardStage(Enum):
    """Defines the mutually exclusive stages of the pick-and-place task."""
    APPROACH_OBJECT = auto()
    LIFT_OBJECT = auto()
    MOVE_TO_GOAL = auto()
    PLACE_OBJECT = auto()

@dataclass
class RewardConfig:
    """Hyperparameters for the reward function."""
    # Potential scaling factors
    k_reach_xy: float = 20.0
    k_reach_z: float = 10.0
    k_lift: float = 30.0
    k_move: float = 20.0
    k_place: float = 30.0
    
    # Sparse milestone bonuses
    approach_bonus: float = 2.5
    grasp_bonus: float = 10.0
    lift_bonus: float = 5.0
    place_bonus: float = 15.0
    success_bonus: float = 50.0

    # Safety and efficiency penalties
    action_penalty_coef: float = 0.001
    contact_penalty: float = 2.0
    time_penalty: float = 0.01

    # Thresholds for state transitions
    hover_height: float = 0.08
    approach_dist_thresh: float = 0.02
    lift_height_thresh: float = 0.05
    place_dist_thresh: float = 0.03
    goal_dist_thresh: float = 0.02

@dataclass
class PhysicalState:
    """A clean container for physical state extracted from the observation."""
    ee_pos: np.ndarray
    object_pos: np.ndarray
    goal_pos: np.ndarray
    is_grasped: bool
    dist_ee_obj: float
    dist_ee_obj_xy: float
    dist_ee_obj_z: float
    dist_obj_goal_xy: float
    dist_obj_goal: float
    object_height: float

class RLRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, config: RewardConfig = RewardConfig()):
        super().__init__(env)
        self.config = config
        
        # Internal state for reward calculation
        self._reward_stage: RewardStage = RewardStage.APPROACH_OBJECT
        self._last_potential: float = 0.0
        
        # Flags to ensure bonuses are given only once per episode
        self._given_approach_bonus: bool = False
        self._given_grasp_bonus: bool = False
        self._given_lift_bonus: bool = False
        self._given_place_bonus: bool = False

        # Get body IDs for contact checking (robustness)
        self.robot_geom_ids = []
        for i in range(env.unwrapped.model.ngeom):
            geom_name = mujoco.mj_id2name(env.unwrapped.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            # Exclude fingers and hand from collision penalty
            if geom_name and 'finger' not in geom_name and 'hand' not in geom_name and 'attachment' not in geom_name:
                self.robot_geom_ids.append(i)
        
        self.table_geom_id = mujoco.mj_name2id(env.unwrapped.model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")


    def _extract_state(self, obs: dict) -> PhysicalState:
        """Extracts and computes all necessary physical quantities from the observation dict."""
        ee_pos = obs['ee_pose_world'][:3]
        object_pos = obs['object_pos_world']
        goal_pos = obs['goal_pos_world']
        
        return PhysicalState(
            ee_pos=ee_pos,
            object_pos=object_pos,
            goal_pos=goal_pos,
            is_grasped=obs['is_grasped'][0] > 0.5,
            dist_ee_obj=np.linalg.norm(ee_pos - object_pos),
            dist_ee_obj_xy=np.linalg.norm(ee_pos[:2] - object_pos[:2]),
            dist_ee_obj_z=abs(ee_pos[2] - object_pos[2]),
            dist_obj_goal_xy=np.linalg.norm(object_pos[:2] - goal_pos[:2]),
            dist_obj_goal=np.linalg.norm(object_pos - goal_pos),
            object_height=object_pos[2]
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Reset internal state
        self._reward_stage = RewardStage.APPROACH_OBJECT
        self._given_approach_bonus = False
        self._given_grasp_bonus = False
        self._given_lift_bonus = False
        self._given_place_bonus = False
        
        # Initialize potential
        state = self._extract_state(obs)
        self._last_potential = self._calculate_potential(state)
        
        return obs, info

    def _calculate_potential(self, state: PhysicalState) -> float:
        """Calculates the potential based on the current reward stage."""
        if self._reward_stage == RewardStage.APPROACH_OBJECT:
            target_z = state.object_pos[2] + self.config.hover_height
            return -self.config.k_reach_xy * state.dist_ee_obj_xy - self.config.k_reach_z * abs(state.ee_pos[2] - target_z)
        
        elif self._reward_stage == RewardStage.LIFT_OBJECT:
            return self.config.k_lift * state.object_height
            
        elif self._reward_stage == RewardStage.MOVE_TO_GOAL:
            return -self.config.k_move * state.dist_obj_goal_xy

        elif self._reward_stage == RewardStage.PLACE_OBJECT:
            return -self.config.k_place * state.dist_obj_goal
        
        return 0.0

    def _update_stage(self, state: PhysicalState) -> bool:
        """Updates the reward stage and re-initializes potential if changed."""
        initial_stage = self._reward_stage
        
        # Transitions are ordered by task progression
        if self._reward_stage == RewardStage.APPROACH_OBJECT and state.is_grasped:
            self._reward_stage = RewardStage.LIFT_OBJECT
        
        elif self._reward_stage == RewardStage.LIFT_OBJECT and state.object_height > (self.env.unwrapped.OBJECT_Z_HEIGHT + self.config.lift_height_thresh):
            self._reward_stage = RewardStage.MOVE_TO_GOAL
            
        elif self._reward_stage == RewardStage.MOVE_TO_GOAL and state.dist_obj_goal_xy < self.config.place_dist_thresh:
            self._reward_stage = RewardStage.PLACE_OBJECT
            
        # Check if stage has changed
        if initial_stage != self._reward_stage:
            # Re-initialize potential to prevent reward spikes from stage changes
            self._last_potential = self._calculate_potential(state)
            return True
        return False
        
    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        state = self._extract_state(obs)
        
        # Update stage and re-initialize potential if it changes
        self._update_stage(state)
        
        # --- Calculate Reward Components ---
        
        # 1. Potential-based dense reward
        new_potential = self._calculate_potential(state)
        dense_reward = new_potential - self._last_potential
        self._last_potential = new_potential
        
        # 2. Sparse milestone bonuses
        sparse_reward = 0.0
        
        # Approach bonus
        if not self._given_approach_bonus and state.dist_ee_obj < self.config.approach_dist_thresh:
            sparse_reward += self.config.approach_bonus
            self._given_approach_bonus = True

        # Grasp bonus
        if not self._given_grasp_bonus and state.is_grasped:
            sparse_reward += self.config.grasp_bonus
            self._given_grasp_bonus = True

        # Lift bonus
        if not self._given_lift_bonus and self._reward_stage in [RewardStage.MOVE_TO_GOAL, RewardStage.PLACE_OBJECT]:
            sparse_reward += self.config.lift_bonus
            self._given_lift_bonus = True
        
        # Place bonus
        if not self._given_place_bonus and self._reward_stage == RewardStage.PLACE_OBJECT and state.dist_obj_goal < self.config.goal_dist_thresh:
            sparse_reward += self.config.place_bonus
            self._given_place_bonus = True
            
        # Success bonus (terminal)
        is_success = (self._reward_stage == RewardStage.PLACE_OBJECT and 
                      state.dist_obj_goal < self.config.goal_dist_thresh and
                      not state.is_grasped)
        if is_success:
            sparse_reward += self.config.success_bonus
            terminated = True
            info['is_success'] = True

        # 3. Penalties
        action_penalty = -self.config.action_penalty_coef * np.sum(np.square(action))
        
        # Check for illegal contacts
        contact_penalty = 0.0
        for i in range(self.env.unwrapped.data.ncon):
            contact = self.env.unwrapped.data.contact[i]
            if (contact.geom1 in self.robot_geom_ids and contact.geom2 == self.table_geom_id) or \
               (contact.geom2 in self.robot_geom_ids and contact.geom1 == self.table_geom_id):
                contact_penalty = -self.config.contact_penalty
                break
        
        time_penalty = -self.config.time_penalty

        # Total reward
        total_reward = dense_reward + sparse_reward + action_penalty + contact_penalty + time_penalty

        # Log reward components for debugging and analysis
        info['reward_stage'] = self._reward_stage.name
        info['r_dense'] = dense_reward
        info['r_sparse'] = sparse_reward
        info['r_action_penalty'] = action_penalty
        info['r_contact_penalty'] = contact_penalty

        return obs, total_reward, terminated, truncated, info