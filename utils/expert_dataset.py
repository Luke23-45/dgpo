# utils/expert_dataset.py
"""
ExpertDataset (robust, production-ready)

This IterableDataset generates (observation, expert_action) samples on the fly by:
 - resetting a MuJoCo environment (PandaEnv),
 - querying a pre-trained foundation model (OCTO) for an end-effector target,
 - converting that to joint commands via IKSolver.

Design notes:
 - Heavy objects (OctoModel, PandaEnv, IKSolver) are initialized lazily in each worker's iterator
   to avoid pickling / cross-process issues.
 - RNG: a base_seed can be provided for reproducibility. Per-worker seeds are derived deterministically.
 - Works with DataLoader(num_workers=0) and also tolerates num_workers>0 (each worker gets own resources).
 - By default the dataset yields CPU tensors and leaves device placement to the training loop.
   If you pass move_to_device=True and device=..., tensors will be moved inside the dataset.
"""
from __future__ import annotations
from scipy.spatial.transform import Rotation as R

import logging
import time
from typing import Dict, Optional, Tuple, Iterator

import numpy as np
import jax
import torch
from torch.utils.data import IterableDataset, get_worker_info
import mujoco
# Project imports (adjust if your package layout differs)
from octo.model.octo_model import OctoModel
from envs.panda_env import PandaEnv
from utils.ik_solver import IKSolver
from utils.scripted_expert import ScriptedExpert, ExpertConfig,ObjectProfile
from utils.obs_adapters import build_octo_observation

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OCTO_POSTPROCESS_CONFIG = {
    # If the predicted z-coordinate is negative (a common failure mode) and the
    # current end-effector is well above the table, flip the sign.
    "Z_FLIP_THRESHOLD_LOW": 0.0,
    "Z_FLIP_THRESHOLD_HIGH": 0.1,
    
    # A hard reachability limit to reject nonsensical OCTO predictions.
    # This is the max distance from the robot base to the target end-effector position.
    "REACHABILITY_LIMIT": 0.85, # in meters

    # Heuristics to determine when the gripper should close.
    # Closes if the XY distance to the cube is less than this threshold AND
    # the Z height is below the cube's top plus a small margin.
    "GRIPPER_CLOSE_XY_THRESHOLD": 0.04, # in meters
    "GRIPPER_CLOSE_Z_MARGIN": 0.03, # in meters
}

class ExpertDataset(IterableDataset):
    def __init__(
        self,
        urdf_path: str,
        instruction: str = "pick up the red block",
        *,
        object_size: Tuple[float, float, float] = (0.04, 0.04, 0.04),
        object_grasp_width: float = 0.6,
        octo_model_name: str = "hf://rail-berkeley/octo-small-1.5",
        env_xml_path: Optional[str] = None,
        base_seed: Optional[int] = None,
        max_samples_per_epoch: Optional[int] = None,
        skip_on_error: bool = True,
        warmup: bool = False,
        use_octo: bool = True, # Flag to enable/disable OCTO
        scripted_cfg: ExpertConfig = ExpertConfig(), # Config for our fallback expert
        yield_full_obs: bool = False,
        action_scaling_factor: float = 0.05
    ) -> None:
        """
        Args:
          urdf_path: path to panda.urdf used by IKSolver.
          instruction: natural-language instruction for OCTO.
          octo_model_name: identifier used by OctoModel.load_pretrained(...)
          env_xml_path: optional MuJoCo xml for PandaEnv (if None, PandaEnv default is used)
          base_seed: optional int seed for deterministic sampling; if None uses non-deterministic seed.
          device: torch.device to place tensors on (only used if move_to_device=True).
          move_to_device: if True, dataset will move each sample to `device` before yielding.
          octo_pad_mask: pad mask passed to OCTO; default [[False, True]] (first frame valid).
          max_samples_per_epoch: if set, each iterator yields at most this many samples then stops.
          skip_on_error: if True, dataset will skip samples that raise exceptions (recommended).
          warmup: if True, calls a small warmup inference after initialization to reduce first-sample latency.
        """
        super().__init__()
        self.urdf_path = urdf_path
        self.instruction = instruction
        self.octo_model_name = octo_model_name
        self.env_xml_path = env_xml_path
        self.base_seed = int(base_seed) if base_seed is not None else None
        self.max_samples_per_epoch = int(max_samples_per_epoch) if max_samples_per_epoch is not None else None
        self.skip_on_error = bool(skip_on_error)
        self.warmup = bool(warmup)
        self.object_profile = ObjectProfile(
            size=np.array(object_size, dtype=np.float32),
            grasp_width_normalized=object_grasp_width
        )
        # Worker-local attributes (initialized in __iter__)
        self._worker_state_initialized = False
        self.use_octo = bool(use_octo)
        self._env = None
        self._ik_solver = None
        self._task = None
        self._jax_key = None
        self._samples_yielded = 0
        self.skipped_samples = 0
        self.scripted_cfg = scripted_cfg
        self._scripted_expert: Optional[ScriptedExpert] = None # Add placeholder
        self._episode_buffer: list = []
        self.yield_full_obs = yield_full_obs
        self.action_scaling_factor = action_scaling_factor
        logger.info("ExpertDataset created (lazy initialization).")

    # ----------------------------
    # Worker-state initialization
    # ----------------------------
    def _init_worker_state(self) -> None:
        """Initialize heavy objects per worker. Safe to call multiple times (idempotent)."""

        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        seed = (self.base_seed if self.base_seed is not None else int(time.time() * 1e6))
        worker_seed = (seed + worker_id) & 0x7FFFFFFF
        
        logger.info(f"[worker {worker_id}] Initializing components with seed {worker_seed}...")
        if self.use_octo:
            self._octo_model = OctoModel.load_pretrained(self.octo_model_name)
            self._task = self._octo_model.create_tasks(texts=[self.instruction])
            self._jax_key = jax.random.PRNGKey(worker_seed)
        
        self._env = PandaEnv(xml_path=self.env_xml_path, control_mode='absolute')
        self._env.set_object_size(self.object_profile.size)
        self._env.reset(seed=worker_seed)
        
        self._ik_solver = IKSolver(urdf_path=self.urdf_path)
        self._scripted_expert = ScriptedExpert(object_profile=self.object_profile, cfg=self.scripted_cfg)

        
        if self.warmup and self.use_octo:
            logger.info(f"[worker {worker_id}] Performing OCTO model warmup...")
            try:
                dummy_obs = {
                    "image_primary": np.zeros((256, 256, 3), dtype=np.uint8),
                    "proprio": np.zeros(22, dtype=np.float32), 
                    "task_completed": np.array([0.0], dtype=np.float32),
                }
                # Use our robust adapter to build the final OCTO-compliant observation.
                octo_obs = build_octo_observation(dummy_obs)
                
                # Perform one sample action call to trigger JIT compilation.
                self._octo_model.sample_actions(octo_obs, self._task, rng=self._jax_key)
                logger.info(f"[worker {worker_id}] OCTO model warmup successful.")
            except Exception as e:
                # If warmup fails, it's not a fatal error. We should log it but continue.
                logger.warning(f"[worker {worker_id}] OCTO model warmup failed: {e}")

        self._worker_state_initialized = True
        self._samples_yielded = 0
        logger.info(f"[worker {worker_id}] Worker state initialized.")


    def _align_action_dim(self, action: np.ndarray) -> np.ndarray:
        """Clamp/pad/truncate IK output to match environment action dimension."""
        action = np.asarray(action, dtype=np.float32).ravel()
        desired = int(self._env.action_space.shape[0])
        if action.size < desired:
            pad = np.zeros(desired - action.size, dtype=np.float32)
            action = np.concatenate([action, pad])
        elif action.size > desired:
            action = action[:desired]
        return action

    # ----------------------------
    # Sample generation core
    # ----------------------------

    def _generate_one(self, current_obs: Dict) -> Tuple[Dict, np.ndarray]:
        """
        Generates a single (observation, action) sample.

        It first attempts to get a valid action from the OCTO model. If the OCTO
        prediction is invalid (unreachable, non-finite), it falls back to the
        deterministic ScriptedExpert to guarantee a high-quality sample.
        """
        # 1. Reset env and get a rich observation with ground-truth data
        obs = current_obs
        # self._scripted_expert.reset() # Reset expert state for each new sample
        pose_world = None
        gripper_action = -1.0 # Default to open
        expert_source = "scripted" # Assume scripted unless OCTO succeeds

        # 2. Try to get a pose from OCTO if enabled
        if self.use_octo:
            try:
                # Build the compliant observation for OCTO
                octo_obs = build_octo_observation(obs)
                
                # Query the model
                self._jax_key, key = jax.random.split(self._jax_key)
                raw_action = self._octo_model.sample_actions(octo_obs, self._task, rng=key)
                
                # --- Start Post-Processing and Validation ---
                # This logic is copied from our debug script
                # --- Start Post-Processing and Validation ---
                candidate_pose = np.array(raw_action[0, 0, :7], dtype=np.float32)
                
                # Z-flip heuristic
                if (candidate_pose[2] < OCTO_POSTPROCESS_CONFIG["Z_FLIP_THRESHOLD_LOW"]
                    and obs["ee_pose_world"][2] > OCTO_POSTPROCESS_CONFIG["Z_FLIP_THRESHOLD_HIGH"]):
                    candidate_pose[2] *= -1.0
                
                # Normalize quaternion
                q = candidate_pose[3:7]
                qn = np.linalg.norm(q)
                if qn > 1e-6:
                    candidate_pose[3:7] = q / qn

                # Workspace clamp (using the same workspace as the scripted expert is a good choice)
                cfg = self.scripted_cfg
                for i, ax in enumerate(("x", "y", "z")):
                    lo, hi = cfg.workspace[ax]
                    candidate_pose[i] = np.clip(candidate_pose[i], lo, hi)

                # Reachability check
                base_pos, _ = self._env.get_base_pose()
                dist_from_base = np.linalg.norm(candidate_pose[:3] - base_pos)
                if (np.all(np.isfinite(candidate_pose)) and 
                    dist_from_base <= OCTO_POSTPROCESS_CONFIG["REACHABILITY_LIMIT"]):
                    # SUCCESS! The OCTO pose is valid.
                    pose_world = candidate_pose
                    expert_source = "octo"

                    # Simple gripper heuristic for OCTO
                    xy_dist_to_cube = np.linalg.norm(pose_world[:2] - obs["object_pos_world"][:2])
                    z_pos_relative_to_cube = pose_world[2] - obs["object_pos_world"][2]

                    is_near_cube = xy_dist_to_cube < OCTO_POSTPROCESS_CONFIG["GRIPPER_CLOSE_XY_THRESHOLD"]
                    is_low_enough = z_pos_relative_to_cube < OCTO_POSTPROCESS_CONFIG["GRIPPER_CLOSE_Z_MARGIN"]
                    
                    gripper_action = 1.0 if (is_near_cube and is_low_enough) else -1.0
                else:
                    logger.debug(f"OCTO pose rejected (dist: {dist_from_base:.2f}m). Falling back.")

            except Exception as e:
                logger.warning(f"OCTO inference failed: {e}. Falling back to ScriptedExpert.")

        # 3. If OCTO failed or was disabled, use the ScriptedExpert
        if pose_world is None:
            pose_world, gripper_action = self._scripted_expert.get_target_pose(
                obs["ee_pose_world"],
                obs["object_pos_world"],
                obs["proprio"], # Pass the full proprio vector
                obs["goal_pos_world"],
                obs["is_grasped"][0] > 0.5, # Pass as a boolean
            )
                
        # 4. Convert the final valid world pose to a joint action via IK
        # (This part is the same for both experts)
        base_pos, base_quat = self._env.get_base_pose()
        R_world_base = R.from_quat(base_quat)
        R_base_world = R_world_base.inv()
        pos_in_base = R_base_world.apply(pose_world[:3] - base_pos)
        rot_in_base = R_base_world * R.from_quat(pose_world[3:7])
        target_pose_base = np.concatenate([pos_in_base, rot_in_base.as_quat()]).astype(np.float32)

        current_joints = obs["internal_full_proprio"][:7]
        # compute_action now correctly returns a 7-DOF arm action

        
        absolute_arm_action = self._ik_solver.compute_action(target_pose_base, current_joints)
        
        # 5. Combine to get the full absolute action for the simulation step
        absolute_final_action = np.concatenate([absolute_arm_action, [gripper_action]])
        absolute_final_action = self._align_action_dim(absolute_final_action)
        
        # Un-normalize the absolute action to get the target physical joint positions
        arm_ctrl_range = self._env.model.actuator_ctrlrange[:7]
        arm_lo, arm_hi = arm_ctrl_range[:, 0], arm_ctrl_range[:, 1]
        physical_target_qpos = arm_lo + 0.5 * (absolute_arm_action + 1.0) * (arm_hi - arm_lo)
        
        # Get current physical joint positions from the observation
        current_physical_qpos = current_obs["proprio"][:7]
        
        # Calculate the required physical delta
        required_physical_delta = physical_target_qpos - current_physical_qpos
        
        # Normalize the delta to get the final delta action
        delta_arm_action = required_physical_delta / self.action_scaling_factor
        
        # Combine with gripper action to form the final 8D delta action
        delta_final_action = np.concatenate([delta_arm_action, [gripper_action]])
        delta_final_action = np.clip(delta_final_action, -1.0, 1.0)
        # --- END OF NEW CONVERSION LOGIC ---

        obs["expert_source"] = 1 if expert_source == "octo" else 0
        
        return obs, absolute_final_action, delta_final_action
  

    def __iter__(self) -> Iterator[Tuple[Dict, np.ndarray]]:
        self._init_worker_state()
        self._episode_buffer.clear()
        samples_this_epoch = 0
        
        # --- START OF IMPROVEMENT ---
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 20 # Raise an error after this many failed attempts
        # --- END OF IMPROVEMENT ---

        while True:
            if self.max_samples_per_epoch is not None and samples_this_epoch >= self.max_samples_per_epoch:
                return

            if not self._episode_buffer:
                try:
                    temp_trajectory = []
                    self._scripted_expert.reset()
                    obs, _ = self._env.reset()

                    for _ in range(self._env.max_episode_steps):
                        # The call to _generate_one is now correct
                        policy_obs, sim_action, data_action = self._generate_one(obs)

                        # 2. The data balancing logic remains the same.
                        #    It decides whether to keep the current (obs, action) pair.
                        keep_sample = False
                        ee_velocity = np.linalg.norm(obs['proprio'][7:14])
                        if ee_velocity < 0.1:
                            keep_sample = True
                        elif np.random.uniform() < 0.1:
                            keep_sample = True
                        
                        if keep_sample:
                            if not self.use_octo:
                                policy_obs["expert_fsm_state"] = self._scripted_expert.get_state()
                            # 3. CRITICAL: Append the correct action (data_action) to the buffer.
                            #    This is the delta action intended for the RL agent.
                            temp_trajectory.append((policy_obs, data_action))
                        
                        # 4. CRITICAL: Step the internal simulation with the correct action (sim_action).
                        #    This is the absolute action required by the IK-driven expert.
                        obs, _, terminated, truncated, _ = self._env.step(sim_action)
                        
                        # 5. The stop condition is also updated slightly for clarity.
                        is_done = (not self.use_octo and self._scripted_expert.is_done()) or terminated or truncated
                        if is_done:
                            break
                    
                    is_successful_trajectory = False
                    if self.use_octo:
                        final_obs = obs
                        object_pos = final_obs['object_pos_world']
                        goal_pos = final_obs['goal_pos_world']
                        object_lifted = object_pos[2] > (self._env.OBJECT_Z_HEIGHT + 0.03)
                        object_near_goal = np.linalg.norm(object_pos[:2] - goal_pos[:2]) < 0.05
                        if object_lifted and object_near_goal:
                            is_successful_trajectory = True
                    else:
                        is_successful_trajectory = self._scripted_expert.was_successful()

                    if is_successful_trajectory:
                        self._episode_buffer.extend(temp_trajectory)
                        consecutive_failures = 0
                    else:
                        expert_type = "OCTO" if self.use_octo else f"ScriptedExpert (state={self._scripted_expert.get_state()})"
                        logger.debug(f"Expert ({expert_type}) did not complete trajectory successfully, discarding.")

                except Exception as exc:
                    if self.skip_on_error:
                        logger.warning(f"Skipped trajectory generation due to error: {exc}", exc_info=True)
                        self._episode_buffer.clear()
                        # --- START OF IMPROVEMENT ---
                        consecutive_failures += 1 
                        # Also check here in case of repeated crashes
                        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                             raise RuntimeError(
                                f"ExpertDataset crashed {MAX_CONSECUTIVE_FAILURES} times in a row. "
                                f"Please check the error logs above for the root cause."
                            ) from exc
                        # --- END OF IMPROVEMENT ---
                        continue
                    else:
                        raise

            if not self._episode_buffer:
                continue

            obs_from_buffer, action_to_yield = self._episode_buffer.pop(0)
            
            if self.yield_full_obs:
                yield obs_from_buffer, action_to_yield
            else:
                obs_for_policy = {
                    "image_primary": obs_from_buffer["image_primary"],
                    "proprio": obs_from_buffer["proprio"],
                }
                yield obs_for_policy, action_to_yield
            samples_this_epoch += 1

    def get_stats(self) -> Dict:
        """Return simple stats about the dataset/worker (samples yielded so far)."""
        return {"samples_yielded": int(self._samples_yielded)}

    def __len__(self) -> int:
        """Only return a length if max_samples_per_epoch provided; otherwise raise TypeError (infinite)."""
        if self.max_samples_per_epoch is None:
            raise TypeError("ExpertDataset is an iterable (infinite) dataset; length undefined.")
        return int(self.max_samples_per_epoch)


# ----------------------------
# Quick standalone smoke test
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke test ExpertDataset (small batch).")
    parser.add_argument("--samples", type=int, default=4, help="Number of samples to fetch (per worker).")
    parser.add_argument("--device", type=str, default="cpu", help="Device to move data to for the test.")
    parser.add_argument("--move", action="store_true", help="Move tensors to device inside dataset.")
    parser.add_argument("--warmup", action="store_true", help="Warmup OCTO on worker init (reduces first-sample latency).")
    args = parser.parse_args()

    # Configure logging to console
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(ch)

    URDF_PATH = "urdf/panda_mujoco_kinematics.urdf"
    DEVICE = torch.device(args.device)
    instruction = "pick up the red block from the table"
    ds = ExpertDataset(urdf_path=URDF_PATH, instruction=instruction,
                       base_seed=1234, device=DEVICE, move_to_device=args.move,
                       max_samples_per_epoch=args.samples, warmup=args.warmup)

    loader = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=0)  # recommended: num_workers=0
    it = iter(loader)
    batch_obs, batch_actions = next(it)

    print("Batch image shape (B,C,H,W):", batch_obs["image_primary"].shape)
    print("Batch proprio shape (B,14):", batch_obs["proprio"].shape)
    print("Batch actions shape (B, action_dim):", batch_actions.shape)
    print("Dtype:", batch_obs["image_primary"].dtype, batch_actions.dtype)
    print("Device image:", batch_obs["image_primary"].device)
    print("Sample stats:", ds.get_stats())
    logger.info("ExpertDataset smoke test complete.")


#python -m utils.expert_dataset --samples 2 --warmup
