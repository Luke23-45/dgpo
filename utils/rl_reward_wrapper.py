# FILE: reward_wrapper.py
# (State-of-the-Art, Blended Potential, Curriculum-Enabled Version)

"""
An advanced, state-of-the-art reward wrapper for complex robotic manipulation tasks.

This wrapper moves beyond discrete, staged rewards and implements a sophisticated,
smoothly blended reward system with a built-in training curriculum. This approach
provides a more stable and informative learning signal, accelerating training and
leading to more robust policies.

Key Advancements:
  - **Blended Potential Function**: Instead of switching between reward functions
    at different task stages, this wrapper calculates a weighted sum of all
    potential components at every step. The weights are dynamically calculated
    using smooth sigmoid functions based on the current physical state, eliminating
    reward discontinuities and providing a continuous gradient for the policy to learn from.

  - **Integrated Training Curriculum**: The wrapper anneals key reward parameters
    over the course of training. It starts with a heavy emphasis on dense potential-shaping
    rewards and gradually shifts focus towards sparse success bonuses and tighter
    goal tolerances. This guides the agent effectively in the beginning and pushes
    it towards high-performance, precise behaviors in later stages.

  - **Advanced Physics-Based Penalties**: Includes penalties for control jerk (encouraging
    smoother motions), grasp instability (detecting object slips), and unnecessary
    contact, leading to more efficient and safer policies.

  - **Rich Diagnostics**: The `info` dictionary is populated with detailed,
    disaggregated reward components, making it significantly easier to debug and
    analyze the agent's learning process.
"""

import gymnasium as gym
import numpy as np
from dataclasses import dataclass,  replace 
import mujoco

# --- Configuration ---

@dataclass
class AdvancedRewardConfig:
    """Configuration for the advanced, blended reward system."""
    # Potential Function Coefficients (how much to value each sub-goal)
    k_reach: float = 30.0  # Approaching the object
    k_grasp: float = 15.0  # Aligning gripper and applying force
    k_lift: float = 40.0   # Lifting the object vertically
    k_move: float = 30.0   # Moving the object towards the goal XY
    k_place: float = 50.0  # Precisely placing the object at the goal
    dense_reward_weight: float = 1.0

    # Penalty Coefficients
    action_penalty: float = 0.001
    jerk_penalty: float = 0.005      # Penalty for non-smooth actions
    contact_penalty: float = 2.0       # Penalty for robot-table collision
    drop_penalty: float = 20.0       # Penalty for dropping a lifted object
    instability_penalty: float = 0.2 # Penalty for object velocity/wobble

    # Sparse Bonuses
    grasp_bonus: float = 10.0
    lift_bonus: float = 15.0
    place_bonus: float = 25.0
    success_bonus: float = 100.0

    # Physical & Task Thresholds
    target_force: float = 10.0
    hover_height: float = 0.05
    lift_height_thresh: float = 0.04
    place_dist_thresh: float = 0.03
    goal_dist_thresh: float = 0.02

@dataclass
class CurriculumConfig:
    """Configuration for annealing reward parameters over training."""
    total_episodes: int = 2000
    dense_reward_anneal_end_factor: float = 0.5  # Final weight of dense reward
    goal_thresh_anneal_end_factor: float = 0.5   # Final goal distance is 50% of initial
    

# --- State Representation ---

@dataclass
class PhysicalState:
    """A comprehensive snapshot of the physical state relevant for reward calculation."""
    ee_pos: np.ndarray
    object_pos: np.ndarray
    goal_pos: np.ndarray
    is_grasped: bool
    grip_force: float
    object_vel: np.ndarray
    dist_ee_obj_xy: float
    dist_ee_obj_z: float
    dist_obj_goal: float
    object_height: float
    object_lift: float  # Height relative to starting position

# --- The Wrapper ---

class AdvancedRewardWrapper(gym.Wrapper):
    """Applies a sophisticated, curriculum-driven reward to a manipulation environment."""

    def __init__(self, env: gym.Env, reward_cfg: AdvancedRewardConfig = AdvancedRewardConfig(), curriculum_cfg: CurriculumConfig = CurriculumConfig()):
        super().__init__(env)
        self.reward_cfg = reward_cfg
        self.curriculum_cfg = curriculum_cfg
        self._initial_reward_cfg = replace(reward_cfg)

        self._last_potential: float = 0.0
        self._last_action: np.ndarray = np.zeros(self.action_space.shape)
        self._initial_object_z: float = 0.0
        self._was_lifted: bool = False
        self._current_episode: int = 0
        self._given_grasp_bonus = False
        self._given_lift_bonus = False
        self._given_place_bonus = False
        # Identify robot geoms to detect collisions
        self.robot_geom_ids: list[int] = []
        for i in range(env.unwrapped.model.ngeom):
            name = mujoco.mj_id2name(env.unwrapped.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and 'finger' not in name and 'hand' not in name and 'attachment' not in name:
                self.robot_geom_ids.append(i)
        self.table_geom_id = mujoco.mj_name2id(env.unwrapped.model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")

    def reset(self, **kwargs) -> tuple[dict, dict]:
        obs, info = self.env.reset(**kwargs)
        state = self._extract_state(obs)

        self._initial_object_z = state.object_pos[2]
        self._last_potential = self._calculate_potential(state)
        self._last_action = np.zeros(self.action_space.shape)
        self._was_lifted = False
        self._current_episode += 1
        self._given_grasp_bonus = False
        self._given_lift_bonus = False
        self._given_place_bonus = False
        self._update_curriculum()

        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        obs, _, terminated, truncated, info = self.env.step(action)
        state = self._extract_state(obs)

        # --- 1. Dense Reward from Potential Shaping ---
        new_potential = self._calculate_potential(state)
        dense_reward = new_potential - self._last_potential
        self._last_potential = new_potential

        # --- 2. Sparse Event-Based Rewards ---
        sparse_reward, is_success = self._calculate_sparse_reward(state)
        if is_success:
            terminated = True
            info['is_success'] = True

        # --- 3. Penalties for Regularization ---
        penalties = self._calculate_penalties(state, action)
        
        # --- 4. Total Reward ---
        total_reward = dense_reward + sparse_reward + penalties
        
        # Update state for next step
        self._last_action = action
        if state.object_lift > self.reward_cfg.lift_height_thresh:
            self._was_lifted = True

        # Populate info dictionary for diagnostics
        info.update({
            'reward_total': total_reward, 'r_dense': dense_reward, 'r_sparse': sparse_reward,
            'r_penalty': penalties, 'potential': new_potential,
            'is_grasped': state.is_grasped, 'object_lift': state.object_lift,
        })
        return obs, total_reward, terminated, truncated, info

    def _extract_state(self, obs: dict) -> PhysicalState:
        ee_pos = obs['ee_pose_world'][:3]
        object_pos = obs['object_pos_world']
        return PhysicalState(
            ee_pos=ee_pos, object_pos=object_pos, goal_pos=obs['goal_pos_world'],
            is_grasped=obs['is_grasped'][0] > 0.5, grip_force=obs.get('grip_force', 0.0),
            object_vel=obs.get('object_vel', np.zeros(3)),
            dist_ee_obj_xy=np.linalg.norm(ee_pos[:2] - object_pos[:2]),
            dist_ee_obj_z=abs(ee_pos[2] - object_pos[2]),
            dist_obj_goal=np.linalg.norm(object_pos - obs['goal_pos_world']),
            object_height=object_pos[2],
            object_lift=object_pos[2] - self._initial_object_z
        )

    def _calculate_potential(self, state: PhysicalState) -> float:
        """Calculates a smoothly blended potential based on the current state."""
        # --- Individual Potential Components ---
        # Reach Potential: higher when closer to the object in XY and at a hover Z.
        target_z = state.object_pos[2] + self.reward_cfg.hover_height
        reach_potential = np.exp(-5 * state.dist_ee_obj_xy) + np.exp(-10 * abs(state.ee_pos[2] - target_z))

        # Grasp Potential: higher when aligned in Z and applying correct force.
        grasp_potential = np.exp(-20 * state.dist_ee_obj_z)
        if state.is_grasped:
             grasp_potential += np.exp(-0.5 * abs(state.grip_force - self.reward_cfg.target_force))

        # Lift Potential: higher when the object is lifted higher.
        lift_potential = 1 - np.exp(-15 * state.object_lift)

        # Move Potential: higher when the object is closer to the goal in XY.
        move_potential = np.exp(-3 * np.linalg.norm(state.object_pos[:2] - state.goal_pos[:2]))

        # Place Potential: higher when the object is closer to the goal in 3D.
        place_potential = np.exp(-5 * state.dist_obj_goal)

        # --- Dynamic Blending Weights ---
        # Weights change smoothly based on task progress.
        w_reach = self._sigmoid_weight(1 - lift_potential)
        w_grasp = self._sigmoid_weight(reach_potential - 0.5)
        w_lift = self._sigmoid_weight(1.5 * grasp_potential if state.is_grasped else 0)
        w_move = self._sigmoid_weight(1.2 * lift_potential)
        w_place = self._sigmoid_weight(1.5 * move_potential)

        # --- Final Blended Potential ---
        total_potential = (
            w_reach * self.reward_cfg.k_reach * reach_potential +
            w_grasp * self.reward_cfg.k_grasp * grasp_potential +
            w_lift  * self.reward_cfg.k_lift  * lift_potential +
            w_move  * self.reward_cfg.k_move  * move_potential +
            w_place * self.reward_cfg.k_place * place_potential
        )
        return total_potential * self.reward_cfg.dense_reward_weight

# FILE: reward_wrapper.py

    def _calculate_sparse_reward(self, state: PhysicalState) -> tuple[float, bool]:
        """Calculates event-based bonuses and checks for task success."""
        sparse_reward = 0.0
        
        if state.is_grasped and not self._given_grasp_bonus:
            sparse_reward += self.reward_cfg.grasp_bonus
            self._given_grasp_bonus = True
            
        if state.object_lift > self.reward_cfg.lift_height_thresh and not self._given_lift_bonus:
            sparse_reward += self.reward_cfg.lift_bonus
            self._given_lift_bonus = True
            
        if state.dist_obj_goal < self.reward_cfg.place_dist_thresh and state.is_grasped and not self._given_place_bonus:
            sparse_reward += self.reward_cfg.place_bonus
            self._given_place_bonus = True
            
        # Final success condition
        is_success = (state.dist_obj_goal < self.reward_cfg.goal_dist_thresh and not state.is_grasped and self._was_lifted)
        if is_success:
            sparse_reward += self.reward_cfg.success_bonus
            
        return sparse_reward, is_success

    def _calculate_penalties(self, state: PhysicalState, action: np.ndarray) -> float:
        """Calculates all negative reward components."""
        # Action & Jerk Penalties
        action_penalty = -self.reward_cfg.action_penalty * np.sum(np.square(action))
        jerk = action - self._last_action
        jerk_penalty = -self.reward_cfg.jerk_penalty * np.sum(np.square(jerk))

        # Collision Penalty
        contact_penalty = 0.0
        for i in range(self.env.unwrapped.data.ncon):
            contact = self.env.unwrapped.data.contact[i]
            if (contact.geom1 in self.robot_geom_ids and contact.geom2 == self.table_geom_id) or \
               (contact.geom2 in self.robot_geom_ids and contact.geom1 == self.table_geom_id):
                contact_penalty = -self.reward_cfg.contact_penalty
                break
        
        # Grasp-related Penalties
        drop_penalty = -self.reward_cfg.drop_penalty if self._was_lifted and not state.is_grasped else 0.0
        instability_penalty = -self.reward_cfg.instability_penalty * np.linalg.norm(state.object_vel) if state.is_grasped else 0.0

        return action_penalty + jerk_penalty + contact_penalty + drop_penalty + instability_penalty

    def _update_curriculum(self):
        """Anneals reward parameters based on the current episode count."""
        progress = min(1.0, self._current_episode / self.curriculum_cfg.total_episodes)
        
        # Anneal dense reward weight
        final_weight = self.curriculum_cfg.dense_reward_anneal_end_factor
        self.reward_cfg.dense_reward_weight = 1.0 - (1.0 - final_weight) * progress

        # Anneal goal thresholds to be stricter over time
        final_goal_thresh = self._initial_reward_cfg.goal_dist_thresh * self.curriculum_cfg.goal_thresh_anneal_end_factor
        
        # This line calculates the new, stricter threshold for the current step
        self.reward_cfg.goal_dist_thresh = self._initial_reward_cfg.goal_dist_thresh - (self._initial_reward_cfg.goal_dist_thresh - final_goal_thresh) * progress

    @staticmethod
    def _sigmoid_weight(x: float, steepness: float = 5.0, offset: float = 0.5) -> float:
        """A simple sigmoid function to create smooth weights between 0 and 1."""
        return 1 / (1 + np.exp(-steepness * (x - offset)))