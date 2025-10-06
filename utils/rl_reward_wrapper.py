"""
Robust RLRewardWrapper for Panda pick-and-place tasks.

This Gymnasium wrapper provides a dense, multi-stage reward function suitable
for reinforcement learning on a pick-and-place task. It is designed to be
robust against common simulation issues like missing MuJoCo elements or
unstable physics.

The reward is composed of the following components:
- R_reach: Dense reward for moving the end-effector towards the object.
- R_grasp: Sparse reward for successfully grasping the object.
- R_lift: Sparse reward for lifting the grasped object off the table.
- R_place: Dense reward for moving the lifted object towards the goal.
- R_success: Large sparse reward for placing the object at the goal.
- R_penalty: Small penalty on the magnitude of actions to encourage efficiency.

The wrapper is designed for testability by separating the reward calculation
logic from the environment's physics step.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, List
import gymnasium as gym
import mujoco
import numpy as np
import torch 
from utils.obs_adapters import build_octo_batch_from_list
import jax
from utils.scripted_expert import ScriptedExpert

logger = logging.getLogger(__name__)


def _safe_norm(vec: np.ndarray) -> float:
    """
    Calculates the L2 norm of a vector with numerical stability.

    If the norm calculation fails or results in a non-finite number (inf, nan),
    it safely returns 0.0.

    Args:
        vec: The input numpy array.

    Returns:
        The L2 norm as a float, or 0.0 if an error occurs.
    """
    try:
        val = float(np.linalg.norm(vec))
        if not np.isfinite(val):
            return 0.0
        return val
    except (ValueError, TypeError):
        return 0.0


class RLRewardWrapper(gym.Wrapper):
    """
    A dense reward wrapper for a pick-and-place task.

    This wrapper computes a multi-component reward and handles episode state
    tracking (e.g., whether an object has been grasped or lifted).

    Args:
        env: The Gymnasium environment to wrap.
        ee_site_name: Name of the end-effector site in the MuJoCo model.
        object_geom_name: Name of the object's geometry.
        goal_body_name: Name of the goal's body.
        gripper_action_index: Index of the gripper control in the action vector.
        reach_scale: Scaling factor for the reach reward component.
        place_scale: Scaling factor for the place reward component.
        grasp_reward: One-time reward for a successful grasp.
        lift_reward: One-time reward for lifting the object.
        success_reward: One-time reward for completing the task.
        lift_z_threshold: Z-coordinate threshold to consider the object "lifted".
        grasp_distance_threshold: Max distance between EE and object for a grasp.
        success_distance_threshold: Max distance between object and goal for success.
        gripper_threshold: Action value threshold to consider the gripper "closing".
        action_penalty: Scaling factor for the L2 action penalty.
        base_reward_weight: Weight to mix in the environment's base reward.
    """
    def __init__(
        self,
        env: gym.Env,
        *,
        scripted_expert: Optional[ScriptedExpert] = None,
        w_guidance: float = 0.1,
        w_guidance_dense: float = 5.0, 
        guidance_clip: float = 1.0,
        ee_site_name: str = "attachment_site",
        object_geom_name: str = "object_geom",
        goal_body_name: str = "goal",
        gripper_action_index: int = 7,

        reach_scale_3d: float = 5.0,
        reach_scale_z: float = 10.0,
        gripper_timing_bonus: float = 2.5,
        grasp_reward: float = 50.0,
        lift_reward: float = 100.0,
        success_reward: float = 250.0,
        place_scale: float = 10.0, 


        lift_z_threshold: float = 0.45,
        grasp_distance_threshold: float = 0.04,
        success_distance_threshold: float = 0.05,
        gripper_threshold: float = 0.5,
        action_penalty: float = 0.001,
        base_reward_weight: float = 0.0,
        octo_model: Optional[Any] = None,
        w_plausibility: float = 0.1,
        quat_format: str = "xyzw", 
        pos_scale: float = 0.05,
        rot_scale: float = 1.0,
        div_clip: float = 10.0,
        div_frame_stride: int = 1,
    ):
        super().__init__(env)

        # Names for MuJoCo elements
        self.ee_site_name = ee_site_name
        self.object_geom_name = object_geom_name
        self.goal_body_name = goal_body_name

        self.gripper_action_index = int(gripper_action_index)

        # Reward parameters
        self.lift_z_threshold = float(lift_z_threshold)
        self.grasp_distance_threshold = float(grasp_distance_threshold)
        self.success_distance_threshold = float(success_distance_threshold)
        self.gripper_threshold = float(gripper_threshold)
        self.action_penalty = float(action_penalty)
        self.base_reward_weight = float(base_reward_weight)
        self.reach_scale_3d = float(reach_scale_3d)
        self.reach_scale_z = float(reach_scale_z)
        self.gripper_timing_bonus = float(gripper_timing_bonus)
        
        # This line was missing from the original file, we re-add it.
        self.place_scale = float(place_scale)

        # Keep these lines:
        self.grasp_reward = float(grasp_reward)
        self.lift_reward = float(lift_reward)
        self.success_reward = float(success_reward)

        # INSERT these new lines for state tracking:
        self.z_reach_threshold = 0.05
        self._gripper_timing_bonus_achieved = False
        # Internal state tracking for one-time rewards and dense shaping
        self._last_dist_ee_to_cube: float = 0.0
        self._last_dist_cube_to_goal: float = 0.0
        self._grasp_achieved: bool = False
        self._lift_achieved: bool = False
        self._last_dist_ee_to_cube_z: float = 0.0

        # Cache for one-time warnings to avoid log spam
        self._warned = {"ee": False, "geom": False, "goal": False, "gripper": False}
        self.octo_model = octo_model
        self.scripted_expert = scripted_expert
        self.w_plausibility = w_plausibility
        self.w_guidance = w_guidance     
        self.w_guidance_dense = w_guidance_dense    
        self.guidance_clip = guidance_clip   
        self.div_clip = div_clip
        self.quat_format = quat_format
        self.pos_weight = 1.0 / (pos_scale**2) if pos_scale > 1e-6 else 1.0
        self.rot_weight = rot_scale
        if self.octo_model:
            self.octo_task = None
        
        self._episode_trajectory: List[Dict[str, np.ndarray]] = []
        self.div_frame_stride = max(1, div_frame_stride)
        self._rng = jax.random.PRNGKey(0)


    
    def _safe_site_pos(self, name: str) -> Optional[np.ndarray]:
        """Safely get a site's position using the core MuJoCo API."""
        try:
            idx = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if idx == -1:
                raise ValueError(f"Site '{name}' not found.")
            pos = self.env.data.site_xpos[idx]
            return np.array(pos, dtype=np.float64)
        except (ValueError, IndexError):
            if not self._warned["ee"]:
                logger.warning(f"[RewardWrapper] MuJoCo site '{name}' not found.")
                self._warned["ee"] = True
            return None

    def _safe_geom_pos(self, name: str) -> Optional[np.ndarray]:
        """Safely get a geom's position using the core MuJoCo API."""
        try:
            idx = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if idx == -1:
                raise ValueError(f"Geom '{name}' not found.")
            pos = self.env.data.geom_xpos[idx]
            return np.array(pos, dtype=np.float64)
        except (ValueError, IndexError):
            if not self._warned["geom"]:
                logger.warning(f"[RewardWrapper] MuJoCo geom '{name}' not found.")
                self._warned["geom"] = True
            return None

    def _safe_body_pos(self, name: str) -> Optional[np.ndarray]:
        """Safely get a body's position using the core MuJoCo API."""
        try:
            idx = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if idx == -1:
                raise ValueError(f"Body '{name}' not found.")
            pos = self.env.data.xpos[idx]
            return np.array(pos, dtype=np.float64)
        except (ValueError, IndexError):
            if not self._warned["goal"]:
                logger.warning(f"[RewardWrapper] MuJoCo body '{name}' not found.")
                self._warned["goal"] = True
            return None

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Any, dict[str, Any]]:
        """Resets the environment and the wrapper's internal state."""
        obs, info = self.env.reset(seed=seed, options=options)
        if self.octo_model:
            instruction = None
            if "language_instruction" in info:
                instruction = info["language_instruction"]
            elif "language_instruction" in obs:
                instruction = obs["language_instruction"]
            
            if instruction is not None:
                # Ensure instruction is a plain string
                if hasattr(instruction, 'decode'):
                    instruction = instruction.decode('utf-8')
                self.octo_task = self.octo_model.create_tasks(texts=[str(instruction)])
                logger.debug(f"RewardWrapper task updated to: '{instruction}'")
            elif self.octo_task is None:
                default_instruction = "pick up the red block"
                self.octo_task = self.octo_model.create_tasks(texts=[default_instruction])
                logger.warning(f"No instruction found in reset(), using default: '{default_instruction}'")
        # Reset internal state flags
        self._grasp_achieved = False
        self._lift_achieved = False
        self._gripper_timing_bonus_achieved = False
        # Recalculate initial distances for dense reward shaping
        ee_pos = self._safe_site_pos(self.ee_site_name)
        cube_pos = self._safe_geom_pos(self.object_geom_name)
        goal_pos = self._safe_body_pos(self.goal_body_name)

        if ee_pos is not None and cube_pos is not None:
            self._last_dist_ee_to_cube = _safe_norm(ee_pos - cube_pos)
            self._last_dist_ee_to_cube_z = abs(ee_pos[2] - cube_pos[2])
        else:
            self._last_dist_ee_to_cube = 0.0

        if cube_pos is not None and goal_pos is not None:
            self._last_dist_cube_to_goal = _safe_norm(cube_pos[:2] - goal_pos[:2])
        else:
            self._last_dist_cube_to_goal = 0.0
        self._episode_trajectory = []
        return obs, info

    def _calculate_rewards_and_info(
        self, obs: Dict[str, Any], action: np.ndarray, base_reward: float, terminated: bool, info: Dict[str, Any]
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Calculates all reward components based on the current simulation state.
        This method is separated from `step` to allow for direct testing of the
        reward logic without running the physics simulation.
        """
        try:
            if hasattr(action, "cpu") and hasattr(action, "numpy"):
                a_np = action.cpu().numpy().ravel()
            else:
                a_np = np.asarray(action, dtype=float).ravel()
        except Exception:
            a_np = np.zeros_like(self.action_space.sample(), dtype=float)

        ee_pos = self._safe_site_pos(self.ee_site_name)
        cube_pos = self._safe_geom_pos(self.object_geom_name)
        goal_pos = self._safe_body_pos(self.goal_body_name)

        if ee_pos is None or cube_pos is None or goal_pos is None:
            R_penalty = -self.action_penalty * float(np.sum(np.square(a_np)))
            reward_total = self.base_reward_weight * float(base_reward) + R_penalty
            info.update({
                "R_reach": 0.0, "R_grasp": 0.0, "R_lift": 0.0, "R_place": 0.0, 
                "R_success": 0.0, "R_penalty": R_penalty, "warning": "degraded_reward"
            })
            return reward_total, terminated, info
        #--------------------
        # -- R_reach: Dense reward for moving EE to the cube --
        dist_ee_to_cube = _safe_norm(ee_pos - cube_pos)
        dist_ee_to_cube_z = abs(ee_pos[2] - cube_pos[2])
        gripper_action_command = a_np[self.gripper_action_index]
        # Ensure is_grasped is a boolean for clean logic
        is_grasped = obs.get("is_grasped", np.array([0.0]))[0] > 0.5

        # --- 1. Two-Stage Reach Reward (R_reach) ---
        R_reach = 0.0
        if dist_ee_to_cube < self.z_reach_threshold:
            # Stage 2: Very close. Reward descending onto the cube (Z-axis only).
            R_reach = (self._last_dist_ee_to_cube_z - dist_ee_to_cube_z) * self.reach_scale_z
        else:
            # Stage 1: Far away. Reward reducing the 3D distance.
            R_reach = (self._last_dist_ee_to_cube - dist_ee_to_cube) * self.reach_scale_3d
            self._gripper_timing_bonus_achieved = False
        
        # Update distance trackers for the next step
        self._last_dist_ee_to_cube = dist_ee_to_cube
        self._last_dist_ee_to_cube_z = dist_ee_to_cube_z

        # --- 2. One-Time Gripper Timing Bonus (R_gripper_timing) ---
        R_gripper_timing = 0.0
        is_closing = gripper_action_command > self.gripper_threshold
        if not self._gripper_timing_bonus_achieved and is_closing and dist_ee_to_cube < self.grasp_distance_threshold:
            R_gripper_timing = self.gripper_timing_bonus
            self._gripper_timing_bonus_achieved = True

        # --- 3. Milestone Grasp Reward (R_grasp) ---
        R_grasp = 0.0
        if is_grasped and not self._grasp_achieved:
            R_grasp = self.grasp_reward
            self._grasp_achieved = True

        # --- 4. Milestone Lift Reward (R_lift) ---
        is_lifted = cube_pos[2] > self.lift_z_threshold
        R_lift = 0.0
        if self._grasp_achieved and is_lifted and not self._lift_achieved:
            R_lift = self.lift_reward
            self._lift_achieved = True
        
        # --- 5. Place Reward (R_place) ---
        dist_cube_to_goal = _safe_norm(cube_pos[:2] - goal_pos[:2])
        R_place = 0.0
        if self._lift_achieved: # Start giving this reward only after lifting is confirmed
            R_place = (self._last_dist_cube_to_goal - dist_cube_to_goal) * self.place_scale
        self._last_dist_cube_to_goal = dist_cube_to_goal
        
        # --- 6. Final Success Reward (R_success) ---
        R_success = 0.0
        is_opening = gripper_action_command < -self.gripper_threshold
        # Success is: having lifted, being near the goal, and commanding the gripper to open.
        if self._lift_achieved and dist_cube_to_goal < self.success_distance_threshold and is_opening:
            R_success = self.success_reward
            terminated = True
            info["is_success"] = True

        # --- 7. Action Penalty (R_penalty) ---
        R_penalty = -self.action_penalty * float(np.sum(np.square(a_np)))
        

        R_guidance_dense = 0.0
        if self.scripted_expert and self.w_guidance_dense > 0.0:
            try:
                ideal_pose, _ = self.scripted_expert.get_target_pose(
                    obs['ee_pose_world'],
                    obs['object_pos_world'],
                    obs['object_orn_world'],
                    obs['goal_pos_world'],
                    obs['is_grasped'][0]
                )
                
                agent_pos = ee_pos # Current EE position
                ideal_pos = ideal_pose[:3]

                # Use an exponential reward based on distance error
                # This rewards being close and has a max value of w_guidance_dense
                pos_error = _safe_norm(agent_pos - ideal_pos)
                R_guidance_dense = np.exp(-20.0 * pos_error) * self.w_guidance_dense
                info['guidance_pos_error'] = float(pos_error)

            except Exception as e:
                logger.warning(f"Could not compute dense guidance reward: {e}")
        # -- Total Reward --
        R_proximity = 0.0      
        R_proximity = 10.0 * np.exp(-20.0 * dist_ee_to_cube)

        reward_total = (
            R_proximity + R_reach + R_gripper_timing + R_grasp + R_lift + R_place + R_success + R_penalty + 
            R_guidance_dense + 
            self.base_reward_weight * float(base_reward)
        )
        reward_total = float(np.nan_to_num(reward_total))

        # -- Update info dictionary for logging/debugging --
        info.update({
            "R_proximity": float(R_proximity),
            "R_reach": float(R_reach),
            "R_reach": float(R_reach),
            "R_grasp": float(R_grasp),
            "R_gripper_timing": float(R_gripper_timing),
            "R_lift": float(R_lift),
            "R_place": float(R_place),
            "R_success": float(R_success),
            "R_penalty": float(R_penalty),
            "dist_ee_to_cube": float(dist_ee_to_cube),
            "dist_cube_to_goal": float(dist_cube_to_goal),
            "is_lifted": bool(is_lifted),
            "gripper_value": float(gripper_action_command),
        })

        return reward_total, terminated, info

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Executes one step in the environment, computes the custom reward,
        and returns the results.
        """
        # Clip action to avoid MuJoCo instability from extreme values
        if hasattr(self.env.action_space, "low"):
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

        # 1. Run the physics simulation in the underlying environment
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        if self.octo_model and hasattr(self.env.unwrapped, 'get_ee_pose'):
            # Collect data for OCTO
            self._episode_trajectory.append({"obs": obs, "ee_pose": self.env.unwrapped.get_ee_pose()})
        elif self.scripted_expert and hasattr(self.env.unwrapped, 'get_expert_obs'):
            # Collect rich data for ScriptedExpert
            self._episode_trajectory.append(self.env.unwrapped.get_expert_obs())
        # 2. Calculate our custom rewards on the new state
        reward_total, terminated, info = self._calculate_rewards_and_info(
            obs=obs,  # Pass the new obs dictionary
            action=action, base_reward=base_reward, terminated=terminated, info=info
        )

        if self.octo_model and (terminated or truncated):
            divergence_mse = self._calculate_divergence()
            # Clip the raw divergence score before applying weight
            clipped_divergence = np.clip(divergence_mse, 0.0, self.div_clip)
            R_T = -self.w_plausibility * clipped_divergence
            
            reward_total += R_T
            info['R_T_divergence'] = float(R_T)
            info['divergence_raw'] = float(divergence_mse) # Telemetry
            info['divergence_clipped'] = float(clipped_divergence) # Telemetry
            info['divergence_enabled'] = True
            info['divergence_steps'] = len(self._episode_trajectory)

        if self.scripted_expert and (terminated or truncated):
            divergence = self._calculate_scripted_divergence()
            clipped_divergence = np.clip(divergence, 0.0, self.guidance_clip)
            R_T_guidance = -self.w_guidance * clipped_divergence
            reward_total += R_T_guidance
            info['R_T_guidance'] = R_T_guidance
            info['guidance_divergence_raw'] = divergence          
      
      
      
        reward_total = float(np.nan_to_num(reward_total))
        return obs, reward_total, terminated, truncated, info
    def _print_dict_structure(self, d, indent=0):
        """Helper to recursively print the structure of a dictionary for debugging."""
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info(f"{'  ' * indent}{key} (dict):")
                self._print_dict_structure(value, indent + 1)
            elif hasattr(value, 'shape') and hasattr(value, 'dtype'):
                logger.info(f"{'  ' * indent}{key}: shape={value.shape}, dtype={value.dtype}")
            else:
                logger.info(f"{'  ' * indent}{key}: {type(value)}")

    def _normalize_bt(self, batch: dict) -> dict:
        """
        Ensure all arrays are shaped (B, T, ...) with B=1, not (T, 1, ...).
        This is a critical fix to align the trajectory batch with the OCTO model's
        single-task (B=1) expectation.
        """
        out = {}
        for k, v in batch.items():
            if k == "pad_mask_dict" and isinstance(v, dict):
                inner = {}
                for kk, vv in v.items():
                    arr = np.asarray(vv)
                    if arr.ndim >= 2 and arr.shape[0] != 1 and arr.shape[1] == 1:
                        inner[kk] = np.swapaxes(arr, 0, 1)  # (T,1,...) -> (1,T,...)
                    else:
                        inner[kk] = arr
                out[k] = inner
            else:
                arr = np.asarray(v)
                if arr.ndim >= 2 and arr.shape[0] != 1 and arr.shape[1] == 1:
                    out[k] = np.swapaxes(arr, 0, 1) # (T,1,...) -> (1,T,...)
                else:
                    out[k] = arr
        return out

    def _calculate_scripted_divergence(self) -> float:
        """
        Calculates the divergence between the agent's trajectory and the ideal path
        of the stateless ScriptedExpert. Returns Mean Squared Positional Error.
        """
        if not self._episode_trajectory:
            return 0.0

        temp_expert = ScriptedExpert(self.scripted_expert.object)
        temp_expert.reset()
        
        total_sq_error = 0.0
        for expert_obs_at_step in self._episode_trajectory:
            agent_pose = expert_obs_at_step['ee_pose_world']
            
            ideal_pose, _ = temp_expert.get_target_pose(
                expert_obs_at_step['ee_pose_world'],
                expert_obs_at_step['object_pos_world'],
                expert_obs_at_step['object_orn_world'],
                expert_obs_at_step['goal_pos_world'],
                expert_obs_at_step['is_grasped']
            )
            
            total_sq_error += np.sum(np.square(agent_pose[:3] - ideal_pose[:3]))

        return total_sq_error / len(self._episode_trajectory) if self._episode_trajectory else 0.0
    @torch.inference_mode()
    def _calculate_divergence(self) -> float:
        """
        Calculates a robust divergence between the agent's recorded EE-trajectory
        and OCTO's predicted poses (position + orientation).
        Returns a scalar float divergence (lower = more plausible).
        This function is defensive: on any error or mismatch it logs and returns 0.0.
        """
        # Quick preconditions
        if not getattr(self, "_episode_trajectory", None):
            logger.debug("No episode trajectory recorded; divergence=0.0")
            return 0.0
        if getattr(self, "octo_model", None) is None:
            logger.debug("No OCTO model available; divergence=0.0")
            return 0.0
        # logger.info("==========================================================")
        # logger.info(">>> ENTERING _calculate_divergence <<<")
        # # Subsample trajectory to reduce compute
        stride = max(1, getattr(self, "div_frame_stride", 1))
        trajectory = self._episode_trajectory[::stride]
        if not trajectory:
            logger.debug("Trajectory empty after subsampling; divergence=0.0")
            return 0.0
        MAX_HORIZON = 8  # or better: octo_model.config.max_horizon if exposed
        if len(trajectory) > MAX_HORIZON:
            trajectory = trajectory[-MAX_HORIZON:]
        # Required observation keys: prefer internal_full_proprio, fallback to proprio
        required_img_key = "image_primary"
        proprio_key = "internal_full_proprio" if "internal_full_proprio" in trajectory[0]["obs"] else "proprio"
        if required_img_key not in trajectory[0]["obs"] or proprio_key not in trajectory[0]["obs"]:
            logger.error("Missing required keys for divergence: "
                        f"need '{required_img_key}' and '{proprio_key}'; skipping divergence.")
            return 0.0

        # Stack agent poses (should be [N,7] with pos(3)+quat(4))
        try:
            agent_poses = np.stack([t["ee_pose"] for t in trajectory], axis=0).astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to stack agent poses for divergence: {e}")
            return 0.0



        try:
            # Get the list of raw environment observations from the trajectory
            obs_list = [t["obs"] for t in trajectory]
            
            # Use the single, correct source of truth to build the entire batch.
            octo_input_batch = build_octo_batch_from_list(obs_list)

            if not octo_input_batch:
                logger.warning("Observation adapter returned an empty batch for divergence check.")
                return 0.0

        except Exception as e:
            logger.error(f"Failed to prepare OCTO input for divergence using adapter: {e}")
            return 0.0
        

        # logger.info(f"Step 1: Trajectory collected with {len(obs_list)} observations.")
        # logger.info("Structure of the FIRST raw observation from the environment (obs_list[0]):")
        # self._print_dict_structure(obs_list[0])
        # logger.info("Step 2: Batch created by `build_octo_batch_from_list`.")
        # logger.info("Final structure of `octo_input_batch` being passed to model:")
        # self._print_dict_structure(octo_input_batch)
        # Run OCTO forward (defensive)
        try:
            try:
                if not getattr(self, "_div_schema_checked", False):
                    from utils.validation import validate_against_example_batch
                    validate_against_example_batch(self.octo_model, octo_input_batch)
                    self._div_schema_checked = True
            except Exception:
                pass
            octo_input_batch = self._normalize_bt(octo_input_batch)
            
            assert octo_input_batch['image_primary'].shape[0] == 1, \
                f"Batch normalization failed! Shape is {octo_input_batch['image_primary'].shape}"
            self._rng, subkey = jax.random.split(self._rng)
            predicted_raw = self.octo_model.sample_actions(octo_input_batch, self.octo_task,  rng=subkey)
            # Convert to numpy (handle torch/numpy-like returns)
            if hasattr(predicted_raw, "cpu") and hasattr(predicted_raw, "numpy"):
                predicted_raw = predicted_raw.cpu().numpy()
            
            predicted_raw = np.asarray(predicted_raw)
            if predicted_raw.ndim == 3: 
                octo_poses = predicted_raw[:, 0, :]
            elif predicted_raw.ndim == 2: 
                octo_poses = predicted_raw
            else:
                logger.error(f"Unexpected action shape from OCTO model: {predicted_raw.shape}. Skipping divergence.")
                return 0.0
        except Exception as e:
            logger.error(f"OCTO inference failed during divergence calculation: {e}")
            logger.error("!!! OCTO INFERENCE FAILED !!!")
            logger.error(f"    ERROR TYPE: {type(e).__name__}")
            logger.error(f"    ERROR MESSAGE: {e}")
            logger.info("==========================================================")
            return 0.0

        # Align lengths
        try:
            T = min(agent_poses.shape[0], octo_poses.shape[0])
            if T == 0:
                return 0.0
            agent_poses = agent_poses[:T]
            octo_poses = octo_poses[:T]
        except Exception as e:
            logger.error(f"Failed to align OCTO & agent pose lengths: {e}")
            return 0.0

        # Position MSE (safe numerics)
        try:
            pos_err = agent_poses[:, :3] - octo_poses[:, :3]
            # per-frame squared error then mean
            per_frame_sq = np.sum(pos_err * pos_err, axis=-1)
            pos_mse = float(np.mean(per_frame_sq))
            if not np.isfinite(pos_mse):
                pos_mse = float(np.nan_to_num(pos_mse, nan=0.0, posinf=1e6, neginf=1e6))
        except Exception as e:
            logger.error(f"Position MSE computation failed: {e}")
            pos_mse = 0.0

        # Quaternion distance (1 - |dot|^2) average
        def _safe_normalize_quat(q: np.ndarray) -> np.ndarray:
            q = np.asarray(q, dtype=np.float32)
            if q.ndim == 1:
                q = q[np.newaxis, :]
            norm = np.linalg.norm(q, axis=-1, keepdims=True)
            norm = np.clip(norm, 1e-8, None)
            qn = q / norm
            qn[~np.isfinite(qn)] = 0.0
            return qn

        try:
            q_agent = _safe_normalize_quat(agent_poses[:, 3:])
            q_octo_raw = octo_poses[:, 3:].astype(np.float32)

            # If octo outputs 'wxyz' but we use 'xyzw', convert (configurable)
            if getattr(self, "quat_format", "xyzw") == "wxyz":
                # convert wxyz -> xyzw (move first element to last)
                if q_octo_raw.shape[-1] == 4:
                    q_octo_raw = q_octo_raw[:, [1, 2, 3, 0]]
            q_octo = _safe_normalize_quat(q_octo_raw)

            # dot product per-frame, absolute value (handle antipodal equivalence)
            dot = np.sum(q_agent * q_octo, axis=-1)
            dot = np.clip(np.abs(dot), 0.0, 1.0)
            quat_dist_mean = float(np.mean(1.0 - dot * dot))
            if not np.isfinite(quat_dist_mean):
                quat_dist_mean = float(np.nan_to_num(quat_dist_mean, nan=0.0, posinf=1e6, neginf=1e6))
        except Exception as e:
            logger.error(f"Quaternion divergence computation failed: {e}")
            quat_dist_mean = 0.0

        # Combine using class weights (fall back to sane defaults)
        pos_w = float(getattr(self, "pos_weight", 1.0))
        rot_w = float(getattr(self, "rot_weight", 1.0))
        divergence = pos_w * pos_mse + rot_w * quat_dist_mean

        # Final numeric safety
        divergence = float(np.nan_to_num(divergence, nan=0.0, posinf=1e6, neginf=1e6))
        if not np.isfinite(divergence):
            divergence = 0.0

        logger.debug(f"divergence computed: pos_mse={pos_mse:.6g}, quat={quat_dist_mean:.6g}, total={divergence:.6g}")
        return divergence
