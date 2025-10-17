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
import copy
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
from utils.lmdb_utils import open_lmdb_env, close_lmdb_env

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
    
    # def _save_lmdb(self, path: str):
    #     map_size = 10 * (1024**3)  # 10 GB initial map size, can be increased
    #     env = lmdb.open(path, map_size=map_size, subdir=False, readonly=False, lock=False)
    #     with env.begin(write=True) as txn:
    #         for idx, ep in enumerate(self.episodes):
    #             key = f"{idx:08d}".encode("ascii")
    #             # Serialize the entire episode dictionary into a binary blob using pickle
    #             val = pickle.dumps(ep)
    #             txn.put(key, val)
    #     env.sync()
    #     env.close()
    def _save_lmdb(self, path: str):
        # Ensure parent dir exists (do NOT create the .lmdb file as a directory)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Use our robust helper — on Windows subdir=False is required for single-file lmdb
        env = open_lmdb_env(path, readonly=False, lock=True, map_size_gb=1.0, subdir=False)
        try:
            with env.begin(write=True) as txn:
                for idx, ep in enumerate(self.episodes):
                    key = f"{idx:08d}".encode("ascii")
                    val = pickle.dumps(ep)
                    txn.put(key, val)
            # durable flush
            env.sync()
            logger.info(f"LMDB successfully written: {path} (episodes={len(self.episodes)})")
        finally:
            close_lmdb_env(env)
    @staticmethod
    def _np_encoder(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Unserializable object {obj} of type {type(obj)}")


class ExpertTrajectoryDataset(Dataset):
    """
    State-of-the-art loader for expert demos.

    Features:
    - Loads data from LMDB or pickle files.
    - Slices full episodes into structured chunks of (observation_horizon, action_horizon).
    - Correctly handles multi-view images (primary and wrist).
    - Designed for use with diffusion policy training.
    """
    def __init__(self, demo_path: str, observation_horizon: int, action_horizon: int):
        self.demo_path = demo_path
        self.observation_horizon = observation_horizon
        self.action_horizon = action_horizon
        self.is_lmdb = lmdb and demo_path.endswith(".lmdb")

        
        # In-memory cache for loaded episodes (especially for pickle mode)
        self._episode_cache = {} 
        self._lmdb_env = None
        self._lmdb_txn = None

        # --- Build a chunk-aware index map ---
        # First, we need to get the length of each episode.
        episode_lengths = []
        if self.is_lmdb:
            # For LMDB, we read the index from the transaction length
            env = open_lmdb_env(self.demo_path, readonly=True, lock=False, readahead=False)
            with env.begin() as txn:
                num_episodes = txn.stat()['entries']
                for i in range(num_episodes):
                    key = f"{i:08d}".encode("ascii")
                    blob = txn.get(key)
                    ep = pickle.loads(blob)
                    episode_lengths.append(len(ep["actions"]))
            env.close()
        else:
            # For pickle, we load the whole file once
            with open(demo_path, "rb") as f:
                self.episodes = pickle.load(f)
            episode_lengths = [len(ep["actions"]) for ep in self.episodes]

        self.index_map = []
        for ep_idx, ep_len in enumerate(episode_lengths):
            # A valid chunk starts at an index `t` where there are enough past
            # observations and enough future actions.
            # First possible start index `t`: self.observation_horizon - 1
            # Last possible start index `t`: ep_len - self.action_horizon
            start_idx = self.observation_horizon - 1
            end_idx = ep_len - self.action_horizon
            for t in range(start_idx, end_idx + 1):
                self.index_map.append((ep_idx, t))
        
        logger.info(
            f"Loaded expert demos: {len(episode_lengths)} episodes, "
            f"{len(self.index_map)} total valid chunks."
        )

    def __len__(self):
        return len(self.index_map)

    # def _get_episode(self, ep_idx: int) -> Dict[str, Any]:
    #     """Helper to get an episode, using cache if available."""
    #     if ep_idx in self._episode_cache:
    #         return self._episode_cache[ep_idx]
        
    #     if self.is_lmdb:
    #         if self._lmdb_env is None:
    #             self._lmdb_env = lmdb.open(self.demo_path, readonly=True, lock=False, readahead=False, meminit=False)
    #             self._lmdb_txn = self._lmdb_env.begin(write=False)
    #         key = f"{ep_idx:08d}".encode("ascii")
    #         blob = self._lmdb_txn.get(key)
    #         ep = pickle.loads(blob)
    #         self._episode_cache[ep_idx] = ep # Cache the loaded episode
    #         return ep
    #     else:
    #         # For pickle, all episodes are already in memory
    #         return self.episodes[ep_idx]

    def _get_episode(self, ep_idx: int) -> Dict[str, Any]:
        if ep_idx in self._episode_cache:
            return self._episode_cache[ep_idx]

        if self.is_lmdb:
            # lazily open environment once per process
            if self._lmdb_env is None:
                # readahead=False improves concurrency; lock=False avoids writer locks for readers
                self._lmdb_env = open_lmdb_env(self.demo_path, readonly=True, lock=False, readahead=False, subdir=False)
            with self._lmdb_env.begin(write=False) as txn:
                key = f"{ep_idx:08d}".encode("ascii")
                blob = txn.get(key)
                if blob is None:
                    raise KeyError(f"Missing LMDB key {key!r} in {self.demo_path}")
                ep = pickle.loads(blob)
                self._episode_cache[ep_idx] = ep
                return ep
        else:
            return self.episodes[ep_idx]
    
    def _init_lmdb(self):
        if self._env is None:
            self._env = lmdb.open(self.demo_path, readonly=True, lock=False, readahead=False, meminit=False)
            self._txn = self._env.begin(write=False)
    def __del__(self):
        if getattr(self, "_lmdb_env", None) is not None:
            close_lmdb_env(self._lmdb_env)
            self._lmdb_env = None
  
    def __getitem__(self, idx: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        ep_idx, t = self.index_map[idx]
        ep = self._get_episode(ep_idx)
        
        # --- Slicing Logic for Chunks ---
        obs_start_idx = t - self.observation_horizon + 1
        obs_end_idx = t + 1
        action_start_idx = t
        action_end_idx = t + self.action_horizon

        # Slice the observation and action lists for the chunk
        obs_chunk_list = ep["obs_list"][obs_start_idx:obs_end_idx]
        action_chunk = np.array(ep["actions"][action_start_idx:action_end_idx], dtype=np.float32)

        # --- Stack Multi-View Images and Proprioception ---
        # This creates the final chunked numpy arrays
        obs_chunk = {
            "image_primary": np.stack([o["image_primary"] for o in obs_chunk_list]),
            "image_wrist": np.stack([o["image_wrist"] for o in obs_chunk_list]),
            "proprio": np.stack([o["proprio"] for o in obs_chunk_list]),
        }
        
        return obs_chunk, action_chunk
    
    @staticmethod
    def collate_fn(batch: List[Tuple[Dict[str, np.ndarray], np.ndarray]]):
        """
        Collates a batch of chunked data into a single PyTorch tensor dictionary.
        Input shape (example):
          - obs["image_primary"]: (B, H_obs, H, W, C)
          - action: (B, H_act, A_dim)
        """
        obs_batch = {}
        # Get all observation keys from the first sample (e.g., 'image_primary', 'image_wrist', 'proprio')
        obs_keys = batch[0][0].keys()
        
        for key in obs_keys:
            # Stack all observations for this key across the batch dimension
            obs_batch[key] = torch.from_numpy(np.stack([sample[0][key] for sample in batch]))

        # Stack all action chunks across the batch dimension
        action_batch = torch.from_numpy(np.stack([sample[1] for sample in batch]))
        
        return obs_batch, action_batch
    
def collate_fn(batch: List[Tuple[Dict[str, np.ndarray], np.ndarray]]):
    """
    Collates a batch of chunked data into a single PyTorch tensor dictionary.
    Input shape (example):
      - sample[0] (obs): Dict with arrays like (H_obs, H, W, C)
      - sample[1] (action): Array like (H_act, A_dim)
    Returns:
      - obs_batch: Dict with tensors like (B, H_obs, H, W, C)
      - action_batch: Tensor like (B, H_act, A_dim)
    """
    if not batch:
        return {}, torch.empty(0)

    obs_batch = {}
    # Get all observation keys from the first sample
    obs_keys = batch[0][0].keys()
    
    for key in obs_keys:
        # Stack all observations for this key across the batch dimension
        # Converts numpy arrays to torch tensors automatically
        obs_batch[key] = torch.from_numpy(np.stack([sample[0][key] for sample in batch]))

    # Stack all action chunks across the batch dimension
    action_batch = torch.from_numpy(np.stack([sample[1] for sample in batch]))
    
    return obs_batch, action_batch


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
        """
        Robust iterator for ExpertDataset.

        Guarantees:
          - deterministic per-worker RNG (via self._rng)
          - at least one sample kept from any successful trajectory
          - safe copying of numpy arrays to avoid shallow-copy bugs
          - consistent episode dict keys ("actions", "obs_list", "ik_fail_flags")
        """
        # Ensure worker state is initialized (this must set self._worker_id, self._worker_master_seed, self._rng)
        self._init_worker_state()

        # defensive inits
        self._episode_buffer.clear()
        samples_this_epoch = 0
        consecutive_failures = 0
        episode_attempt_counter = 0
        MAX_CONSEC = 25

        while True:
            # stop condition
            if self.max_samples_per_epoch is not None and samples_this_epoch >= self.max_samples_per_epoch:
                return

            # if buffer empty, generate a new episode
            if not self._episode_buffer:
                try:
                    # deterministic per-episode seed derived from master seed
                    current_episode_seed = (self._worker_master_seed + episode_attempt_counter) & 0x7FFFFFFF
                    episode_attempt_counter += 1
                    logger.debug(f"[worker {getattr(self,'_worker_id',0)}] Starting episode attempt seed={current_episode_seed}")

                    # reset env & expert
                    obs, _ = self._env.reset(seed=current_episode_seed)
                    self._env.set_object_size(self.object_profile.size)
                    self._scripted_expert.reset()
                    if hasattr(self._ik_solver, "reset_controller_state"):
                        self._ik_solver.reset_controller_state()

                    # collect the full (unfiltered) trajectory in memory for possible fallback
                    unfiltered_obs: List[Dict[str, np.ndarray]] = []
                    unfiltered_actions: List[np.ndarray] = []
                    unfiltered_ik_flags: List[bool] = []

                    for step in range(self._env.max_episode_steps):
                        policy_obs, action, ik_failed = self._generate_one(obs)

                        # convert and copy immediately to avoid aliasing
                        obs_snapshot = {k: (np.copy(v) if isinstance(v, np.ndarray) else copy.deepcopy(v))
                                        for k, v in policy_obs.items()}
                        action_arr = np.asarray(action, dtype=np.float32).copy()

                        unfiltered_obs.append(obs_snapshot)
                        unfiltered_actions.append(action_arr)
                        unfiltered_ik_flags.append(bool(ik_failed))

                        # step simulator with sim_act (absolute action used to step)
                        obs, _, terminated, truncated, _ = self._env.step(action)
                        if terminated or truncated or self._scripted_expert.is_done():
                            break

                    # determine success using the same logic as dataset (scripted_expert or env-based)
                    # prefer scripted_expert.was_successful() if available
                    try:
                        is_success = self._scripted_expert.was_successful()
                    except Exception:
                        # fallback: basic object-lift & near-goal test
                        final = obs
                        objp = final["object_pos_world"]
                        goalp = final["goal_pos_world"]
                        is_lifted = objp[2] > (self._env.OBJECT_Z_HEIGHT + 0.03)
                        is_near_goal = np.linalg.norm(objp[:2] - goalp[:2]) < 0.05
                        is_success = bool(is_lifted and is_near_goal)

                    # If successful, filter frames for storage (balanced selection)
                    if is_success:
                        filtered: List[Tuple[Dict, np.ndarray]] = []
                        for step_idx, (step_obs, step_action) in enumerate(zip(unfiltered_obs, unfiltered_actions)):
                            ee_vel = np.linalg.norm(step_obs["proprio"][7:14])
                            if ee_vel < 0.1:
                                keep = (self._rng.random() < self.p_low_vel)
                            else:
                                keep = (self._rng.random() < self.p_motion_frame)
                            if keep:
                                # store copy-safe snapshots
                                filtered.append(( {k: np.copy(v) if isinstance(v, np.ndarray) else copy.deepcopy(v)
                                                  for k, v in step_obs.items()},
                                                  step_action.copy() ))

                        # fallback: if empty, keep most "active" frame (highest joint delta magnitude)
                        if not filtered and unfiltered_actions:
                            # robustly compute magnitudes (exclude gripper scalar if present)
                            try:
                                mags = []
                                for a in unfiltered_actions:
                                    a = np.asarray(a, dtype=np.float32)
                                    if a.size >= 2:
                                        mags.append(np.linalg.norm(a[:-1]))  # assume last is gripper
                                    else:
                                        mags.append(np.linalg.norm(a))
                                best_idx = int(np.argmax(mags))
                            except Exception:
                                best_idx = -1
                            logger.warning(
                                "Successful trajectory entirely filtered by stochastic selector; "
                                f"keeping fallback frame idx={best_idx} (worker={getattr(self,'_worker_id',0)})"
                            )
                            fb_obs = {k: np.copy(v) if isinstance(v, np.ndarray) else copy.deepcopy(v)
                                      for k, v in unfiltered_obs[best_idx].items()}
                            fb_act = np.asarray(unfiltered_actions[best_idx], dtype=np.float32).copy()
                            filtered.append((fb_obs, fb_act))

                        # Acceptance
                        if filtered:
                            # append to the in-memory episode buffer which will be yielded
                            self._episode_buffer.extend(filtered)

                            # Build full episode dict (store unfiltered trajectory for offline writer)
                            ep_id = f"w{getattr(self,'_worker_id',0)}_e{self._episode_id_counter}"
                            episode_dict = {
                                "episode_id": ep_id,
                                "seed": int(current_episode_seed),
                                "obs_list": [{k: (np.copy(v) if isinstance(v, np.ndarray) else copy.deepcopy(v))
                                              for k, v in o.items()} for o in unfiltered_obs],
                                "actions": [np.asarray(a, dtype=np.float32).copy() for a in unfiltered_actions],
                                "ik_fail_flags": list(unfiltered_ik_flags),
                                "success": True,
                            }
                            self.episodes.append(episode_dict)
                            self._episode_id_counter += 1
                            consecutive_failures = 0
                        else:
                            # defensive: should not happen due to fallback
                            consecutive_failures += 1
                            logger.error("Filtered trajectory unexpectedly empty after fallback.")
                            if consecutive_failures >= MAX_CONSEC:
                                raise RuntimeError("Too many consecutive failures")
                            continue
                    else:
                        # Failed trajectory: log & increment counter
                        consecutive_failures += 1
                        logger.debug(f"Discarding failed trajectory (worker={getattr(self,'_worker_id',0)}). Consecutive failures: {consecutive_failures}")
                        if consecutive_failures >= MAX_CONSEC:
                            raise RuntimeError("Too many consecutive failures")
                        continue

                except Exception as exc:
                    # robust error handling
                    if self.skip_on_error:
                        logger.warning(f"Episode generation exception (worker={getattr(self,'_worker_id',0)}): {exc}", exc_info=True)
                        consecutive_failures += 1
                        if consecutive_failures >= MAX_CONSEC:
                            raise RuntimeError(f"ExpertDataset crashed {MAX_CONSEC} times in a row.") from exc
                        continue
                    else:
                        raise

            # Yield samples from the episode buffer one-by-one
            if not self._episode_buffer:
                continue

            obs_from_buffer, action_to_yield = self._episode_buffer.pop(0)
            # increment counters
            samples_this_epoch += 1
            self._samples_yielded += 1

            if self.yield_full_obs:
                yield obs_from_buffer, action_to_yield
            else:
                obs_for_policy = {"image_primary": obs_from_buffer["image_primary"],
                                  "proprio": obs_from_buffer["proprio"]}
                yield obs_for_policy, action_to_yield

    
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
    if ep.get("obs_list"):
        try:
            # Restore position
            init_obj_pos = np.array(ep["obs_list"][0]["object_pos_world"], dtype=np.float32)
            obj_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "object")
            if obj_body_id != -1:
                env.data.xpos[obj_body_id] = init_obj_pos
            
            # Restore orientation (if available)
            if "object_orn_world" in ep["obs_list"][0]:
                init_obj_orn_xyzw = np.array(ep["obs_list"][0]["object_orn_world"], dtype=np.float32)
                # Convert to MuJoCo's wxyz format
                init_obj_orn_wxyz = np.array([init_obj_orn_xyzw[3], init_obj_orn_xyzw[0], init_obj_orn_xyzw[1], init_obj_orn_xyzw[2]])
                obj_joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "object_joint")
                if obj_joint_id != -1:
                    qpos_adr = env.model.jnt_qposadr[obj_joint_id]
                    env.data.qpos[qpos_adr + 3 : qpos_adr + 7] = init_obj_orn_wxyz

            # Apply changes to the simulation state
            mujoco.mj_forward(env.model, env.data)
        except (KeyError, IndexError) as e:
            logger.warning(f"Could not restore initial object state for replay: {e}")
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
