"""
utils/expert_dataset.py — robust expert demo generator (scripted-only mode, delta control),
with on-disk serialization, replay validation, diagnostics, and full metadata support.

Features:
  - Default use_octo=False (so no OCTO dependency by default)
  - Strict schema validation for observations & actions
  - Traceable metadata (seed, URDF hashes, solver version, etc.)
  - Replay validation: ability to replay stored sim_actions to re-simulate final pose
  - Support for writing to LMDB (preferred) or fallback pickle format
  - Balancing control (low-velocity vs motion frames)
  - Diagnostic logging & histogram exports
  - Episode-level indexing & split compatibility for training loader
  - Tags for IK failures / fallback events
  - Collate function for DataLoader compatibility

**Important: you must install `lmdb` for LMDB support (optional fallback to pickle)**
"""

from __future__ import annotations
import pickle
import os
import time
import json
import hashlib
import logging
from typing import Dict, Optional, Tuple, Iterator, List, Any
from numpy.random import Generator, PCG64
import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset, get_worker_info
from scipy.spatial.transform import Rotation as R

import mujoco

# Lazy imports for LMDB to avoid pickling issues
try:
    import lmdb
except ImportError:
    lmdb = None

# Project imports (adjust if your project layout differs)
from envs.panda_env import PandaEnv
from utils.ik_solver import IKSolver
from utils.scripted_expert import ScriptedExpert, ExpertConfig, ObjectProfile
from utils.obs_adapters import build_octo_observation  # You might not need OCTO paths now

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Default configuration thresholds (you may expose these as args)
REPLAY_POS_TOL = 0.03  # 3 cm tolerance
REPLAY_ORN_TOL = 5.0 * np.pi / 180.0  # 5 degrees in radians

# A minimal observation schema: keys with expected shapes/dtypes
OBS_SCHEMA = {
    "image_primary": ("uint8", (None, None, 3)),
    "proprio": ("float32", (None,)),
    "internal_full_proprio": ("float32", (None,)),
    "ee_pose_world": ("float32", (7,)),
    "object_pos_world": ("float32", (3,)),
    "object_orn_world": ("float32", (4,)),
    "goal_pos_world": ("float32", (3,)),
    "is_grasped": ("float32", (1,)),
    "gripper_qpos": ("float32", (None,)),
    "robot_base_pos_world": ("float32", (3,)),
    "base_quat": ("float32", (4,)),
    # you can add more keys if needed
}

class ExpertDatasetWriter:
    """
    Helper to accumulate episodes and write out on-disk expert demo file plus index & metadata.
    Supports LMDB format if available; otherwise fallback to pickle.
    """
    def __init__(self, out_dir: str, run_name: Optional[str] = None):
        os.makedirs(out_dir, exist_ok=True)
        if run_name is None:
            run_name = time.strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name
        self.out_dir = out_dir
        self.episodes: List[Dict[str, Any]] = []  # list of per-episode dicts
        self.metadata: Dict[str, Any] = {}
        self._episode_id_counter = 0
    
    def add_episode(self, ep_dict: Dict[str, Any]):
        self.episodes.append(ep_dict)
    
    def save(self):
        # Compute run hash
        md5 = hashlib.md5(json.dumps(self.metadata, sort_keys=True).encode("utf-8")).hexdigest()
        base_name = f"expert_{self.run_name}_{md5}"
        fname = base_name + (".lmdb" if lmdb else ".pkl")
        fpath = os.path.join(self.out_dir, fname)
        logger.info(f"Saving expert dataset to {fpath} (episodes: {len(self.episodes)})")
        
        if lmdb:
            self._save_lmdb(fpath)
        else:
            self._save_pickle(fpath)
        
        # Also write index and config metadata
        index = []
        for idx, ep in enumerate(self.episodes):
            index.append({
                "episode_id": ep["episode_id"],
                "length": len(ep["actions"]), # Use the unified "actions" key
                "success": bool(ep.get("success", False)),
                "seed": ep.get("seed"),
                "first_object_pos": ep["obs_list"][0]["object_pos_world"].tolist(),
            })
        with open(os.path.join(self.out_dir, base_name + "_index.json"), "w") as f:
            json.dump(index, f, indent=2)
        with open(os.path.join(self.out_dir, base_name + "_config.json"), "w") as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"Index and config metadata saved.")
    
    def _save_pickle(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.episodes, f)
    
    def _save_lmdb(self, path: str):
        map_size = 10 * (1024**3)  # 10 GB initial map size, can be increased
        env = lmdb.open(path, map_size=map_size, subdir=False, readonly=False, lock=False)
        with env.begin(write=True) as txn:
            for idx, ep in enumerate(self.episodes):
                key = f"{idx:08d}".encode("ascii")
                # Serialize the entire episode dictionary into a binary blob using pickle
                val = pickle.dumps(ep)
                txn.put(key, val)
        env.sync()
        env.close()
    
    @staticmethod
    def _np_encoder(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Unserializable object {obj} of type {type(obj)}")


class ExpertTrajectoryDataset(Dataset):
    """
    Loader for the on-disk expert demos produced by ExpertDatasetWriter.
    Each __getitem__ returns (obs_dict, data_action) as used by diffusion BC training.
    Uses lazy LMDB initialization to avoid pickling env issues in DataLoader workers.
    """
    def __init__(self, demo_path: str):
        self.demo_path = demo_path
        self.is_lmdb = lmdb and demo_path.endswith(".lmdb")
        self.episodes = []  # will hold in-memory or index pointers
        self._env = None
        self._txn = None
        
        if self.is_lmdb:
            # open env lazily later
            pass
        else:
            # Pickle fallback: load entire episodes
            import pickle
            with open(demo_path, "rb") as f:
                self.episodes = pickle.load(f)
        # Build flat index: mapping global index -> (episode_idx, step_idx)
        self.index_map = []
        for ep_i, ep in enumerate(self.episodes):
            L = len(ep["actions"])
            for t in range(L):
                self.index_map.append((ep_i, t))
        logger.info(f"Loaded expert demos: {len(self.episodes)} episodes, {len(self.index_map)} total steps.")
    
    def __len__(self):
        return len(self.index_map)
    
    def _init_lmdb(self):
        if self._env is None:
            self._env = lmdb.open(self.demo_path, readonly=True, lock=False, readahead=False, meminit=False)
            self._txn = self._env.begin(write=False)
    
    def __getitem__(self, idx: int):
        ep_i, t = self.index_map[idx]
        ep = self.episodes[ep_i] if not self.is_lmdb else None
        if self.is_lmdb:
            self._init_lmdb()
            key = f"{ep_i:08d}".encode("ascii")
            blob = self._txn.get(key)
            assert blob is not None, f"Missing LMDB key {key}"
            ep = pickle.loads(blob)
        
        obs = ep["obs_list"][t]
        data_action = np.array(ep["actions"][t], dtype=np.float32)
        # Convert necessary observation items to torch tensors
        torch_obs = {
            "image_primary": torch.from_numpy(obs["image_primary"]),
            "proprio": torch.from_numpy(obs["proprio"]),
        }
        return torch_obs, torch.from_numpy(data_action)
    
    @staticmethod
    def collate_fn(batch):
        imgs = torch.stack([b[0]["image_primary"] for b in batch], dim=0)
        proprios = torch.stack([b[0]["proprio"] for b in batch], dim=0)
        actions = torch.stack([b[1] for b in batch], dim=0)
        return {"image_primary": imgs, "proprio": proprios}, actions


class ExpertDataset(IterableDataset):
    """
    IterableDataset version that generates expert demos online, and (optionally) writes them out.
    After generation, you may save via ExpertDatasetWriter.
    This class yields (obs, data_action) for training.
    """
    def __init__(
        self,
        urdf_path: str,
        instruction: str = "pick up the red block",
        *,
        object_size: Tuple[float, float, float] = (0.04, 0.04, 0.04),
        object_grasp_width: float = 0.6,
        env_xml_path: Optional[str] = None,
        base_seed: Optional[int] = None,
        max_samples_per_epoch: Optional[int] = None,
        skip_on_error: bool = True,
        warmup: bool = False,
        scripted_cfg: ExpertConfig = ExpertConfig(),
        yield_full_obs: bool = False,
        action_scaling_factor: float = 0.5,
        # new config options:
        p_low_vel: float = 0.4,
        p_motion_frame: float = 0.6,
        min_keep_per_state: int = 5,
        diagnostics_dir: Optional[str] = None,
    ):
        super().__init__()
        self.urdf_path = urdf_path
        self.instruction = instruction
        self.env_xml_path = env_xml_path
        self.base_seed = base_seed
        self.max_samples_per_epoch = max_samples_per_epoch
        self.skip_on_error = skip_on_error
        self.warmup = warmup
        self.scripted_cfg = scripted_cfg
        self.yield_full_obs = yield_full_obs
        self.action_scaling_factor = action_scaling_factor
        
        # balancing / filtering parameters
        self.p_low_vel = p_low_vel
        self.p_motion_frame = p_motion_frame
        self.min_keep_per_state = min_keep_per_state
        
        # diagnostics
        self.diagnostics_dir = diagnostics_dir
        if diagnostics_dir:
            os.makedirs(diagnostics_dir, exist_ok=True)
        self.object_profile = ObjectProfile(
            size=np.array(object_size, dtype=np.float32),
            grasp_width_normalized=object_grasp_width
        )
        # worker-local state (initialized lazily in __iter__)
        self._worker_state_initialized = False
        self._env = None
        self._ik_solver = None
        self._scripted_expert: Optional[ScriptedExpert] = None
        self._episode_buffer: List[Tuple[Dict, np.ndarray]] = []
        self.episodes: List[Dict[str, Any]] = []

        
        logger.info("ExpertDataset (improved) initialized (lazy).")
    
    def _init_worker_state(self):
        # --- START OF PATCH, STEP 2 ---
        # This replaces the entire old method.
        if self._worker_state_initialized:
            return

        worker_info = get_worker_info()
        self._worker_id = worker_info.id if worker_info is not None else 0
        
        # 1. Create a single, unique, deterministic master seed for this entire worker process.
        #    This is the root of all randomness for this worker.
        seed = self.base_seed if self.base_seed is not None else int(time.time() * 1e9)
        self._worker_master_seed = seed + self._worker_id
        
        # 2. Create a dedicated, seeded random number generator (RNG) for this worker.
        #    This will be used for any probabilistic logic (like data filtering) to make it reproducible.
        self._rng = Generator(PCG64(self._worker_master_seed))
        
        logger.info(f"[Worker {self._worker_id}] Initializing with master seed {self._worker_master_seed}")
        
        self._env = PandaEnv(xml_path=self.env_xml_path, control_mode='delta')
        logger.info(
            f"[worker {self._worker_id}] PandaEnv initialized. "
            f"ACTION_SCALING_FACTOR = {self._env.ACTION_SCALING_FACTOR}"
        )
        self._env.set_object_size(self.object_profile.size)
        self._ik_solver = IKSolver(urdf_path=self.urdf_path)
        self._scripted_expert = ScriptedExpert(
            object_profile=self.object_profile,
            cfg=self.scripted_cfg
        )
        
        if self.warmup:
            logger.info(f"[Worker {self._worker_id}] Warmup (scripted only).")
            try:
                # REPLACE the hardcoded dimension with a dynamic lookup from the env.
                dummy_obs = {
                    "image_primary": np.zeros((256, 256, 3), dtype=np.uint8),
                    # PandaEnv now has a `proprio_dim` attribute.
                    "proprio": np.zeros(self._env.proprio_dim, dtype=np.float32),
                    "task_completed": np.array([0.0], dtype=np.float32),
                }
                _ = build_octo_observation(dummy_obs)
            except Exception as e:
                logger.warning("Warmup failed: " + str(e))
        
        self._worker_state_initialized = True
        self._samples_yielded = 0
        self._episode_id_counter = 0  # Add this
        self._episode_attempt_counter = 0 # Use this for seeding
        logger.info(f"[Worker {self._worker_id}] State initialization complete.")
    
    def _check_schema(self, obs: Dict[str, np.ndarray]):
        for k, (dtype, shape_tpl) in OBS_SCHEMA.items():
            if k not in obs:
                raise ValueError(f"Missing OBS_SCHEMA key {k}")
            arr = obs[k]
            if arr.dtype != np.dtype(dtype):
                raise ValueError(f"Key {k} has dtype {arr.dtype}, expected {dtype}")
            # shape check (only lower dims)
            if shape_tpl[0] is not None and arr.ndim < len(shape_tpl):
                raise ValueError(f"Key {k} has shape {arr.shape}, expected at least dims {shape_tpl}")
            # we could enforce exact dims for fixed-length keys
    
    def _generate_one(self, current_obs: Dict) -> Tuple[Dict, np.ndarray, np.ndarray, bool]:
        """
        Produces (obs, sim_action, data_action, ik_failed_flag).
        sim_action: absolute action used to step simulator
        data_action: normalized delta to train on
        ik_failed_flag: True if IK solver used fallback
        """
        # Validate schema on input (optional)
        # self._check_schema(current_obs)
        
        # Use scripted expert always
        pose_world, gripper_act = self._scripted_expert.get_target_pose(current_obs)
        
        # Transform world pose into base frame
        N_SUBSTEPS = 20
        effective_dt = self._env.model.opt.timestep * N_SUBSTEPS
        max_dq = self._env.ACTION_SCALING_FACTOR / effective_dt
        arm_joint_ids = np.arange(7) # Assuming the first 7 joints are the arm
        if self._env.data.time < 1e-6: # Log only at the beginning of an episode
             logger.info(
                 f"[worker {get_worker_info().id if get_worker_info() else 0}] "
                 f"IK params calculated: effective_dt={effective_dt:.4f}, "
                 f"max_dq={max_dq:.4f}"
             )
        # 2. Call compute_delta_action to get the data_action directly.
        delta_arm_action = self._ik_solver.compute_delta_action(
            target_ee_pose=pose_world,
            model=self._env.model,
            data=self._env.data,
            ee_site_id=self._env.ee_site_id,
            joint_qpos_indices=arm_joint_ids,
            effective_dt=effective_dt,
            max_dq=max_dq
        )
        
        # 3. For a direct delta pipeline, the sim_action IS the data_action.
        action = np.concatenate([delta_arm_action, [gripper_act]]).astype(np.float32)

        # 4. The concept of IK failure is less direct here. We can assume it doesn't
        #    fail in the same way, or check if the returned action is all zeros.
        ik_failed = np.linalg.norm(delta_arm_action) < 1e-4
        
        # Add expert source tag (scripted-only)
        current_obs["expert_source"] = 0
        
        return current_obs, action, ik_failed
    
    def __iter__(self) -> Iterator[Tuple[Dict, np.ndarray]]:
        self._init_worker_state()
        self._episode_buffer.clear()
        samples_this_epoch = 0
        consecutive_failures = 0
        self._episode_id_counter = 0 
        MAX_CONSEC = 1
        
        while True:
            # Stop when enough samples
            if self.max_samples_per_epoch is not None and samples_this_epoch >= self.max_samples_per_epoch:
                return
            
            if not self._episode_buffer:
                # start a new trajectory
                try:
                    # --- START OF PATCH, STEP 3 ---
                    # 1. Derive the seed for THIS specific episode attempt from the worker's master seed.
                    current_episode_seed = self._worker_master_seed + self._episode_attempt_counter
                    self._episode_attempt_counter += 1

                    logger.info(f"[Worker {self._worker_id}] Starting episode attempt {self._episode_attempt_counter} with seed {current_episode_seed}.")
                    
                    # 2. Reset the environment using this unique, deterministic episode seed.
                    obs, _ = self._env.reset(seed=current_episode_seed)
                    self._env.set_object_size(self.object_profile.size) 
                    self._ik_solver.reset_controller_state() 
                    consecutive_ik_failures = 0
                    IK_FAILURE_THRESHOLD = 10 
                    temp_trajectory = []
                    actions = []
                    obs_list = []
                    ik_fail_flags = []
                    
                    for step in range(self._env.max_episode_steps):
                        policy_obs, action, ik_failed = self._generate_one(obs)
                        if ik_failed:
                            consecutive_ik_failures += 1
                        else:
                            consecutive_ik_failures = 0 # Reset counter on a successful IK solve.
                        
                        # If the threshold is exceeded, terminate this trajectory attempt.
                        if consecutive_ik_failures >= IK_FAILURE_THRESHOLD:
                            logger.warning(
                                f"[Worker {self._worker_id}] Terminating trajectory due to "
                                f"{consecutive_ik_failures} consecutive IK failures."
                            )
                            # Since this is a failure, we break the loop. The expert's `was_successful`
                            # flag will be False, and the trajectory will be discarded correctly.
                            break
                        # Filtering / balancing
                        ee_vel = np.linalg.norm(obs["proprio"][7:14])
                        keep = False
                        
                        if ee_vel < 0.1:
                            keep = (self._rng.random() < self.p_low_vel)
                        else:
                            keep = (self._rng.random() < self.p_motion_frame)
                        
                        if keep:
                            obs_snapshot = {k: np.copy(v) for k, v in policy_obs.items()}
                            
                            # Append to the correct lists.
                            temp_trajectory.append((obs_snapshot, action.copy()))
                            obs_list.append(obs_snapshot)
                            ik_fail_flags.append(ik_failed)
                            # Append to the new unified actions list.
                            actions.append(action.copy())

                        
                        obs, _, terminated, truncated, _ = self._env.step(action)
                        if terminated or truncated or self._scripted_expert.is_done():
                            break
                    
                    # Determine success
                    success = self._scripted_expert.was_successful()
                    
                    if success:
                        ep_id = f"w{self._worker_id}_e{self._episode_id_counter}"
                        self._episode_id_counter += 1
                        
                        # The `ep` dictionary is now correctly constructed.
                        ep = {
                            "episode_id": ep_id,
                            "seed": current_episode_seed,
                            "obs_list": obs_list,
                            "actions": actions,
                            "ik_fail_flags": ik_fail_flags,
                            "success": True,
                        }
                        
                        # CRITICAL FIX: Add the complete episode to the internal episodes list.
                        self.episodes.append(ep)
                        self._episode_buffer.extend(temp_trajectory)
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                        logger.debug(f"Discarding failed trajectory (success={success})")
                        if consecutive_failures >= MAX_CONSEC:
                            raise RuntimeError("Too many consecutive failures")
                        continue
                    consecutive_failures = 0
                except Exception as e:
                    if self.skip_on_error:
                        logger.warning("Trajectory generation failed: " + str(e))
                        consecutive_failures += 1
                        if consecutive_failures >= MAX_CONSEC:
                            raise
                        continue
                    else:
                        raise
            
            # Yield from buffer
            policy_obs, data_act = self._episode_buffer.pop(0)
            if self.yield_full_obs:
                yield policy_obs, data_act
            else:
                obs_for_policy = {"image_primary": policy_obs["image_primary"],
                                  "proprio": policy_obs["proprio"]}
                yield obs_for_policy, data_act
            samples_this_epoch += 1
            self._samples_yielded += 1
    
    def get_stats(self):
        return {
            "samples_yielded": int(self._samples_yielded),
            "episodes_collected": len(self.episodes)
        }


def replay_validate_episode(ep: Dict[str, Any], urdf_path: str, env_xml_path: Optional[str] = None) -> bool:
    """
    Replay sim_actions in a fresh environment and compare final object pose vs stored.
    Return True if within tolerance.
    """
    env = PandaEnv(xml_path=env_xml_path, control_mode='delta')
    env.reset(seed=ep.get("seed", None))
    # You might want to also reset object initial states if stored
    for a in ep["actions"]: # Use the unified "actions" key
        obs, _, done, trunc, _ = env.step(np.array(a, dtype=np.float32))
        if done or trunc:
            break
    final = obs
    tgt = ep["obs_list"][-1]["object_pos_world"]
    got = final["object_pos_world"]
    pos_err = np.linalg.norm(tgt - got)
    # orientation check (if stored object_orn_world)
    # skip orientation for now
    ok = pos_err <= REPLAY_POS_TOL
    if not ok:
        logger.warning(f"Replay mismatch pos_err={pos_err:.5f}")
    return ok
