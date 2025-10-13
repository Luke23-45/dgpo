import gymnasium as gym
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
import mujoco

class RewardStage(Enum):
    APPROACH_OBJECT = auto()
    LIFT_OBJECT = auto()
    MOVE_TO_GOAL = auto()
    PLACE_OBJECT = auto()

@dataclass
class RewardConfig:
    k_reach_xy: float = 20.0
    k_reach_z: float = 10.0
    k_lift: float = 30.0
    k_move: float = 20.0
    k_place: float = 30.0
    k_force: float = 5.0
    target_force: float = 10.0
    k_vel: float = 0.05
    max_vel: float = 0.5
    k_stab: float = 0.1  # Stability
    approach_bonus: float = 2.5
    grasp_bonus: float = 10.0
    lift_bonus: float = 5.0
    place_bonus: float = 15.0
    success_bonus: float = 50.0
    action_penalty_coef: float = 0.001
    contact_penalty: float = 2.0
    time_penalty: float = 0.01
    hover_height: float = 0.08
    approach_dist_thresh: float = 0.02
    lift_height_thresh: float = 0.05
    place_dist_thresh: float = 0.03
    goal_dist_thresh: float = 0.02

@dataclass
class PhysicalState:
    ee_pos: np.ndarray
    object_pos: np.ndarray
    goal_pos: np.ndarray
    is_grasped: bool
    grip_force: float
    dist_ee_obj: float
    dist_ee_obj_xy: float
    dist_ee_obj_z: float
    dist_obj_goal_xy: float
    dist_obj_goal: float
    object_height: float
    object_vel: np.ndarray  # For stability

class RLRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, config: RewardConfig = RewardConfig(), total_eps=1000):
        super().__init__(env)
        self.config = config
        self._reward_stage = RewardStage.APPROACH_OBJECT
        self._last_potential = 0.0
        self._given_approach_bonus = False
        self._given_grasp_bonus = False
        self._given_lift_bonus = False
        self._given_place_bonus = False

        for i in range(env.unwrapped.model.ngeom):
            name = mujoco.mj_id2name(env.unwrapped.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and 'finger' not in name and 'hand' not in name and 'attachment' not in name:
                self.robot_geom_ids.append(i)
        self.table_geom_id = mujoco.mj_name2id(env.unwrapped.model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")
        self.total_eps = total_eps
        self.current_ep = 0
        self.w_dense = 1.0  # Anneal to 0.5
        self._initial_object_z = 0.0

    def _extract_state(self, obs: dict) -> PhysicalState:
        ee_pos = obs['ee_pose_world'][:3]
        object_pos = obs['object_pos_world']
        goal_pos = obs['goal_pos_world']
        grip_force = obs.get('grip_force', 0.0)
        object_vel = obs.get('object_vel', np.zeros(3))
        return PhysicalState(
            ee_pos=ee_pos, object_pos=object_pos, goal_pos=goal_pos,
            is_grasped=obs['is_grasped'][0] > 0.5, grip_force=grip_force,
            dist_ee_obj=np.linalg.norm(ee_pos - object_pos),
            dist_ee_obj_xy=np.linalg.norm(ee_pos[:2] - object_pos[:2]),
            dist_ee_obj_z=abs(ee_pos[2] - object_pos[2]),
            dist_obj_goal_xy=np.linalg.norm(object_pos[:2] - goal_pos[:2]),
            dist_obj_goal=np.linalg.norm(object_pos - goal_pos),
            object_height=object_pos[2], object_vel=object_vel
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reward_stage = RewardStage.APPROACH_OBJECT
        self._given_approach_bonus = False
        self._given_grasp_bonus = False
        self._given_lift_bonus = False
        self._given_place_bonus = False
        self._initial_object_z = state.object_pos[2] 
        self._last_potential = self._calculate_potential(state)
        state = self._extract_state(obs)
        self._last_potential = self._calculate_potential(state)
        self.current_ep += 1
        anneal = max(0.5, 1.0 - (self.current_ep / self.total_eps))
        self.w_dense = anneal
        return obs, info

    def _calculate_potential(self, state: PhysicalState) -> float:
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
        initial_stage = self._reward_stage
        if self._reward_stage == RewardStage.APPROACH_OBJECT and state.is_grasped:
            self._reward_stage = RewardStage.LIFT_OBJECT
        elif self._reward_stage == RewardStage.LIFT_OBJECT and state.object_height > (self._initial_object_z + self.config.lift_height_thresh):
            self._reward_stage = RewardStage.MOVE_TO_GOAL

        elif self._reward_stage == RewardStage.MOVE_TO_GOAL and state.dist_obj_goal_xy < self.config.place_dist_thresh:
            self._reward_stage = RewardStage.PLACE_OBJECT
        if initial_stage != self._reward_stage:
            self._last_potential = self._calculate_potential(state)
            return True
        return False

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        state = self._extract_state(obs)
        self._update_stage(state)
        new_potential = self._calculate_potential(state)
        dense_reward = (new_potential - self._last_potential) * self.w_dense
        self._last_potential = new_potential
        sparse_reward = 0.0
        if not self._given_approach_bonus and state.dist_ee_obj < self.config.approach_dist_thresh:
            sparse_reward += self.config.approach_bonus
            self._given_approach_bonus = True
        if not self._given_grasp_bonus and state.is_grasped:
            sparse_reward += self.config.grasp_bonus
            self._given_grasp_bonus = True
        if not self._given_lift_bonus and self._reward_stage in [RewardStage.MOVE_TO_GOAL, RewardStage.PLACE_OBJECT]:
            sparse_reward += self.config.lift_bonus
            self._given_lift_bonus = True
        if not self._given_place_bonus and self._reward_stage == RewardStage.PLACE_OBJECT and state.dist_obj_goal < self.config.goal_dist_thresh:
            sparse_reward += self.config.place_bonus
            self._given_place_bonus = True
        is_success = (self._reward_stage == RewardStage.PLACE_OBJECT and state.dist_obj_goal < self.config.goal_dist_thresh and not state.is_grasped)
        if is_success:
            sparse_reward += self.config.success_bonus
            terminated = True
            info['is_success'] = True
        action_penalty = -self.config.action_penalty_coef * np.sum(np.square(action))
        contact_penalty = 0.0
        for i in range(self.env.unwrapped.data.ncon):
            contact = self.env.unwrapped.data.contact[i]
            if (contact.geom1 in self.robot_geom_ids and contact.geom2 == self.table_geom_id) or (contact.geom2 in self.robot_geom_ids and contact.geom1 == self.table_geom_id):
                contact_penalty -= self.config.contact_penalty
                break
        force_penalty = -self.config.k_force * abs(state.grip_force - self.config.target_force) if state.is_grasped else 0.0
        vel_penalty = -self.config.k_vel * max(0, np.linalg.norm(state.object_vel) - self.config.max_vel) if state.is_grasped else 0.0
        stab_penalty = -self.config.k_stab * np.linalg.norm(state.object_vel) if state.is_grasped else 0.0
        time_penalty = -self.config.time_penalty
        total_reward = dense_reward + sparse_reward + action_penalty + contact_penalty + force_penalty + vel_penalty + stab_penalty + time_penalty
        info['reward_stage'] = self._reward_stage.name
        info['r_dense'] = dense_reward
        info['r_sparse'] = sparse_reward
        info['r_action_penalty'] = action_penalty
        info['r_contact_penalty'] = contact_penalty
        info['r_force'] = force_penalty
        info['r_vel'] = vel_penalty
        info['r_stab'] = stab_penalty
        return obs, total_reward, terminated, truncated, info

    def compute_reward_(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info) -> np.ndarray:
        # position error
        d_pos = np.linalg.norm(achieved_goal[:, :3] - desired_goal[:, :3], axis=-1)

        # attempt to extract grasped flag robustly from info
        # info may be: list of dicts OR dict of arrays OR None
        if info is None:
            grasped = np.zeros(len(d_pos), dtype=np.float32)
        elif isinstance(info, dict):
            # dict of arrays
            g = info.get('is_grasped', None)
            if g is None:
                grasped = np.zeros(len(d_pos), dtype=np.float32)
            else:
                grasped = np.array(g).astype(np.float32)
        else:
            # assume list/iterable of dict-like
            grasped = np.array([i.get('is_grasped', 0) if isinstance(i, dict) else 0 for i in info], dtype=np.float32)

        # dense component (normalized; tune coefficients as needed)
        r_dense = -d_pos - 0.5 * np.abs(achieved_goal[:, 3:] - desired_goal[:, 3:]).mean(axis=-1)

        # bonus for grasp (vectorized)
        r_bonus = 10.0 * grasped

        return r_dense + r_bonus
    
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info) -> np.ndarray:
        """
        Vectorized binary success reward:
        returns 0.0 for success (within goal_dist_thresh), -1.0 for failure.
        """
        # Euclidean distance per sample (works for shape (N, D))
        distance = np.linalg.norm(achieved_goal[:, :3] - desired_goal[:, :3], axis=-1)
        success = distance < self.config.goal_dist_thresh
        # return float array shaped (N,)
        return (success.astype(np.float32) * 0.0) + (~success).astype(np.float32) * -1.0

    

