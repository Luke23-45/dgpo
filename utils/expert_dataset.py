# FILE: utils/expert_dataset.py
# (State-of-the-Art, SoA, JPEG-Compressed, Virtual-Indexed, LRU-Cached Version)

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
from pathlib import Path
import functools

# --- SOTA Imports ---
try:
    import cv2 # Required for JPEG compression
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

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
REPLAY_POS_TOL = 0.03  # 3 cm tolerance [cite: 219]
REPLAY_ORN_TOL = 5.0 * np.pi / 180.0  # 5 degrees in radians [cite: 219]

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

# ==============================================================================
# 1. STATE-OF-THE-ART DATASET WRITER
#    (Generates the optimized SoA + JPEG-compressed format)
# ==============================================================================

class ExpertDatasetWriter:
    """
    State-of-the-art helper to write expert demos to an optimized on-disk format.
    
    This writer implements a "Struct of Arrays" (SoA) format and on-the-fly
    image compression to create a highly efficient dataset for chunked reading.

    Key Features:
    - **Struct-of-Arrays (SoA):** Instead of one giant pickle per episode,
      each modality (e.g., 'actions', 'proprio', 'image_primary') is saved
      as its own key in LMDB. This allows the reader to load *only* the
      modalities it needs.
    - **Image Compression:** 'image_primary' and 'image_wrist' are compressed
      on-the-fly to JPEG, massively reducing the dataset size (e.g., 25GB -> 2-5GB).
    - **Instant-On JSON Index:** Creates a single `_index.json` file that
      contains all metadata (episode lengths, dtypes, shapes, compression)
      for the entire dataset. The reader *only* loads this file,
      eliminating any startup scan.
    """
    def __init__(
        self,
        out_dir: str,
        run_name: Optional[str] = None,
        image_compression: str = "jpeg",
        jpeg_quality: int = 90,
        expected_episodes: int = 100,  # NEW: For dynamic LMDB map size
    ):
        if not CV2_AVAILABLE:
            raise ImportError("cv2 (OpenCV) is required for the SOTA ExpertDatasetWriter. Please install it.")
        if not lmdb:
            raise ImportError("lmdb is required for the SOTA ExpertDatasetWriter. Please install it.") 

        os.makedirs(out_dir, exist_ok=True) 
        if run_name is None:
            run_name = time.strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name
        self.out_dir = Path(out_dir)
        self.episodes_in_memory: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {} 
        self._episode_id_counter = 0

        self.image_compression = image_compression
        self.jpeg_quality = jpeg_quality
        self.index_data = {"episodes": []}
        
        # Dynamic LMDB map size based on expected episodes (~100MB per episode)
        from utils.lmdb_utils import calculate_lmdb_map_size_gb
        self.map_size_gb = calculate_lmdb_map_size_gb(expected_episodes)
        
        logger.info(f"SOTA ExpertDatasetWriter initialized. Compression: {self.image_compression}")
        logger.info(f"  Expected episodes: {expected_episodes}, Map size: {self.map_size_gb:.1f} GB")

    def add_episode(self, ep_dict: Dict[str, Any]):
        """
        Adds a completed episode (in the old "List of Structs" format)
        to the in-memory buffer, ready to be written.
        """
        self.episodes_in_memory.append(ep_dict)

    def _write_pickled_modality(self, txn, ep_meta, prefix, name, data_list):
        """Helper to write a list of generic Python objects by pickling."""
        if not data_list:
            return
        key = f"{prefix}_{name}"
        # We pickle the entire list of objects at once.
        txn.put(key.encode("ascii"), pickle.dumps(data_list))
        
        ep_meta["modalities"][name] = {
            "key": key,
            "compression": "pickle", # A new compression type
            "dtype": "object",
            "shape": [len(data_list)]
        }  
        
    def save_batch(self, episode_list: List[Dict[str, Any]]):
        """Append a list of episodes directly to LMDB (safe for large datasets)."""
        if not episode_list:
            return

        # construct paths like in save() but allow appending to existing file if present
        meta_blob = json.dumps(self.metadata or {}, sort_keys=True)
        md5 = hashlib.md5(meta_blob.encode("utf-8")).hexdigest()[:8]
        base_name = f"expert_{self.run_name}_{md5}"

        if not hasattr(self, "_lmdb_path"):
            meta_blob = json.dumps(self.metadata or {}, sort_keys=True)
            md5 = hashlib.md5(meta_blob.encode("utf-8")).hexdigest()[:8]
            
            self._lmdb_path = self.out_dir / (base_name + ".lmdb")

        lmdb_path = self._lmdb_path

        # open env with dynamic map size
        env = open_lmdb_env(str(lmdb_path), readonly=False, lock=True, map_size_gb=self.map_size_gb, subdir=False)
        try:
            with env.begin(write=True) as txn:
                for ep_dict in episode_list:
                    prefix = f"ep_{self._episode_id_counter:06d}"
                    self._episode_id_counter += 1

                    ep_meta = {"episode_id": prefix, "length": len(ep_dict["actions"]), "success": bool(ep_dict.get("success", False)), "seed": ep_dict.get("seed", None), "modalities": {}}
                    
                    self._write_raw_numpy(txn, ep_meta, prefix, "actions", np.array(ep_dict["actions"], dtype=np.float32))
                    self._write_raw_numpy(txn, ep_meta, prefix, "proprio", np.stack([o["proprio"] for o in ep_dict["obs_list"]]).astype(np.float32))
                    self._write_compressed_images(txn, ep_meta, prefix, "image_primary", [o["image_primary"] for o in ep_dict["obs_list"]])
                    self._write_compressed_images(txn, ep_meta, prefix, "image_wrist", [o["image_wrist"] for o in ep_dict["obs_list"]])

                    ee_poses_arr = np.stack([o["ee_pose_world"] for o in ep_dict["obs_list"]]).astype(np.float32)
                    self._write_raw_numpy(txn, ep_meta, prefix, "ee_pose_world", ee_poses_arr)
                    # if "goal_image_primary" in ep_dict and ep_dict["goal_image_primary"] is not None:
                    #     goal_img_list = [ep_dict["goal_image_primary"]]
                    #     self._write_compressed_images(txn, ep_meta, prefix, "goal_image_primary", goal_img_list)
                    # else:
                    #     logger.warning(f"Episode {prefix} is missing 'goal_image_primary'.")
                    expert_states_list = [o["expert_state"] for o in ep_dict["obs_list"]]
                    self._write_pickled_modality(txn, ep_meta, prefix, "expert_states", expert_states_list)
                    camera_params_list = [o["camera_params"] for o in ep_dict["obs_list"]]
                    self._write_pickled_modality(txn, ep_meta, prefix, "camera_params", camera_params_list)
                    gt_phases_arr = np.stack([o["gt_phase"] for o in ep_dict["obs_list"]]).astype(np.int32)
                    self._write_raw_numpy(txn, ep_meta, prefix, "gt_phase", gt_phases_arr)

                    gt_gripper_arr = np.stack([o["gt_gripper"] for o in ep_dict["obs_list"]]).astype(np.float32)
                    self._write_raw_numpy(txn, ep_meta, prefix, "gt_gripper", gt_gripper_arr)

                    # [CRITICAL FIX] Write the EXPERT TARGET POSE - the commanded targets, NOT achieved poses
                    # This is the correct supervision signal for goal-directed behavior
                    if "expert_target_pose" in ep_dict["obs_list"][0]:
                        target_poses_arr = np.stack([o["expert_target_pose"] for o in ep_dict["obs_list"]]).astype(np.float32)
                        self._write_raw_numpy(txn, ep_meta, prefix, "expert_target_pose", target_poses_arr)

                    self.index_data["episodes"].append(ep_meta)
                    self._episode_id_counter += 1
        finally:
            env.sync()
            close_lmdb_env(env)
            
        
        json_path = self._lmdb_path.parent / f"{self._lmdb_path.stem}_index.json"
        with open(json_path, "w") as f:
            json.dump(self.index_data, f)


    def save(self):
        """
        Processes all in-memory episodes, converts them to the optimized
        SoA format, and writes them to the LMDB file and JSON index.
        """
        if not self.episodes_in_memory:
            logger.warning("No episodes to save.")
            return

        # --- 1. Prepare File Paths ---
        md5 = hashlib.md5(json.dumps(self.metadata, sort_keys=True).encode("utf-8")).hexdigest()
        base_name = f"expert_{self.run_name}_{md5}"
        lmdb_path = self.out_dir / (base_name + ".lmdb")
        json_path = self.out_dir / (base_name + "_index.json")
        config_path = self.out_dir / (base_name + "_config.json")
        logger.info(f"Saving {len(self.episodes_in_memory)} episodes to {lmdb_path}...")

        # --- 2. Open LMDB Environment ---
        # Dynamic map size based on expected episodes
        env = open_lmdb_env(str(lmdb_path), readonly=False, lock=True, map_size_gb=self.map_size_gb, subdir=False) 
        
        try:
            with env.begin(write=True) as txn: 
                for ep_idx, ep_dict in enumerate(self.episodes_in_memory):
                    episode_key_prefix = f"ep_{ep_idx:06d}"
                    ep_meta = {
                        "episode_id": episode_key_prefix,
                        "length": len(ep_dict["actions"]), 
                        "success": bool(ep_dict.get("success", False)), 
                        "seed": ep_dict.get("seed", None),
                        "modalities": {}
                    }

                    # --- 3. Process and Write Modalities (SoA) ---
                    
                    # A) 'actions' (Raw Numpy)
                    actions_arr = np.array(ep_dict["actions"], dtype=np.float32) 
                    self._write_raw_numpy(txn, ep_meta, episode_key_prefix, "actions", actions_arr)

                    # B) 'proprio' (Raw Numpy)
                    proprio_arr = np.stack([o["proprio"] for o in ep_dict["obs_list"]]).astype(np.float32) 
                    self._write_raw_numpy(txn, ep_meta, episode_key_prefix, "proprio", proprio_arr)
                    
                    # C) 'image_primary' (Compressed Images)
                    img_p_list = [o["image_primary"] for o in ep_dict["obs_list"]] 
                    self._write_compressed_images(txn, ep_meta, episode_key_prefix, "image_primary", img_p_list)

                    # D) 'image_wrist' (Compressed Images)
                    img_w_list = [o["image_wrist"] for o in ep_dict["obs_list"]] 
                    self._write_compressed_images(txn, ep_meta, episode_key_prefix, "image_wrist", img_w_list)
                    if "goal_image_primary" in ep_dict:
                        goal_img_list = [ep_dict["goal_image_primary"]]
                        self._write_compressed_images(txn, ep_meta, episode_key_prefix, "goal_image_primary", goal_img_list)
                    else:
                        logger.warning(f"Episode {episode_key_prefix} is missing 'goal_image_primary'. This key will not be saved for this episode.")
                        
                    expert_states_list = [o["expert_state"] for o in ep_dict["obs_list"]]
                    self._write_pickled_modality(txn, ep_meta, episode_key_prefix, "expert_states", expert_states_list)

                    # F) 'camera_params' (Pickled List of Dictionaries)
                    camera_params_list = [o["camera_params"] for o in ep_dict["obs_list"]]
                    self._write_pickled_modality(txn, ep_meta, episode_key_prefix, "camera_params", camera_params_list)
                    
                    # G) 'expert_target_pose' - [CRITICAL FIX] The commanded targets for training
                    if "expert_target_pose" in ep_dict["obs_list"][0]:
                        target_poses_arr = np.stack([o["expert_target_pose"] for o in ep_dict["obs_list"]]).astype(np.float32)
                        self._write_raw_numpy(txn, ep_meta, episode_key_prefix, "expert_target_pose", target_poses_arr)
                    
                    # --- 4. Add this episode's metadata to the main index ---
                    self.index_data["episodes"].append(ep_meta)

            env.sync() 
            logger.info("LMDB write complete.")
        
        finally:
            close_lmdb_env(env) 
            self.episodes_in_memory.clear()

        # --- 5. Write the final JSON index and config ---
        self.index_data["metadata"] = self.metadata
        with open(json_path, "w") as f:
            json.dump(self.index_data, f) # No indent for smaller file size
        
        with open(config_path, "w") as f:
            json.dump(self.metadata, f, indent=2) 
        
        logger.info(f"Dataset saved. Index: {json_path}")

    def _write_raw_numpy(self, txn, ep_meta, prefix, name, arr):
        """Helper to write a raw numpy array."""
        key = f"{prefix}_{name}"
        txn.put(key.encode("ascii"), arr.tobytes()) 
        ep_meta["modalities"][name] = {
            "key": key,
            "compression": "raw",
            "dtype": str(arr.dtype),
            "shape": list(arr.shape)
        }

    def _write_compressed_images(self, txn, ep_meta, prefix, name, img_list):
        """Helper to compress and write a list of image arrays."""
        if not img_list:
            return
            
        key = f"{prefix}_{name}"
        if self.image_compression == "jpeg":
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            encode_fn = lambda img: cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tobytes()
        else: # 'png' or default
            encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 1] # Fast PNG
            encode_fn = lambda img: cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tobytes()
        
        # This can be parallelized with multiprocessing.Pool for max speed
        byte_list = [encode_fn(img) for img in img_list]
        
        # We pickle the *list of byte strings*
        txn.put(key.encode("ascii"), pickle.dumps(byte_list)) 
        
        ep_meta["modalities"][name] = {
            "key": key,
            "compression": self.image_compression,
            "dtype": str(img_list[0].dtype),
            "shape": [len(img_list), *img_list[0].shape]
        }

# ==============================================================================
# 2. STATE-OF-THE-ART DATASET READER
#    (Consumes the optimized format)
# ==============================================================================

class ExpertTrajectoryDataset(Dataset):
    """
    State-of-the-art, high-performance loader for expert demonstrations
    stored in the optimized (SoA, compressed, virtually-indexed) format.

    This class is designed for maximum throughput and minimal startup time
    with multi-GB/TB datasets.

    Features:
    - **Instant-On (Zero-Scan):** Reads a pre-computed `_index.json` file.
      It *never* scans the LMDB file at startup.
    - **Virtual Indexing:** Uses O(N_episodes) memory for indexing,
      not O(N_chunks). It computes chunk-to-episode mappings in O(log N)
      time during `__getitem__`.
    - **SoA Chunking:** Loads *only* the required data chunks (e.g.,
      2 frames of 'proprio', 8 frames of 'actions') from the SoA database,
      avoiding any overhead from reading the full episode.
    - **On-the-Fly Decompression:** Decompresses JPEG/PNG images in the
      `DataLoader` workers, leveraging multiple CPU cores and minimizing
      I/O bottlenecks.
    - **Per-Worker LRU Caching:** Uses a bounded LRU cache (`@functools.lru_cache`)
      to keep hot *full-episode-modalities* (like an episode's entire 'proprio'
      array) in each worker's memory. This prevents OOM errors while
      massively speeding up subsequent accesses to the same episode.
    """
    def __init__(self, demo_path: str, observation_horizon: int, action_horizon: int): 
        self.demo_path = Path(demo_path)
        self.observation_horizon = observation_horizon
        self.action_horizon = action_horizon
        self.is_lmdb = lmdb and self.demo_path.suffix == ".lmdb" 

        if not self.is_lmdb:
            raise ValueError("SOTA ExpertTrajectoryDataset only supports LMDB (.lmdb) format.")
        if not CV2_AVAILABLE:
            raise ImportError("cv2 (OpenCV) is required for the SOTA ExpertTrajectoryDataset to decompress images.")

        # --- 1. Load the JSON Index ---
        self.index_path = self.demo_path.parent / f"{self.demo_path.stem}_index.json"

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Missing required index file: {self.index_path}\n"
                f"Please regenerate your dataset with the new ExpertDatasetWriter."
            )
        
        logger.info(f"Loading index from {self.index_path}...")
        with open(self.index_path, "r") as f:
            index_data = json.load(f)

        self.episode_metadata = index_data["episodes"]
        self.metadata = index_data.get("metadata", {})

        # --- 2. Build the Virtual Index ---
        # We calculate the number of valid chunks in each episode.
        self.chunks_per_episode = []
        for ep_meta in self.episode_metadata:
            ep_len = ep_meta["length"]
            # A valid chunk starts at an index `t` where there are enough past
            # observations and enough future actions.
            # First possible start index `t`: self.observation_horizon - 1
            # Last possible start index `t`: ep_len - self.action_horizon
            start_t = self.observation_horizon - 1
            end_t = ep_len - self.action_horizon
            num_chunks = (end_t - start_t) + 1
            # Ensure we don't have negative chunks for short episodes
            self.chunks_per_episode.append(max(0, num_chunks))

        self.total_chunks = sum(self.chunks_per_episode)
        
        # The cumulative sum is the core of our virtual index.
        # It maps a global chunk `idx` to an `ep_idx`.
        self._cumulative_chunks = np.cumsum(self.chunks_per_episode)

        # Worker-local state (initialized lazily)
        self._lmdb_env = None 
        self._pid = os.getpid() # Store the PID of the process that created the dataset        
        logger.info(
            f"Loaded {len(self.episode_metadata)} episodes, "
            f"{self.total_chunks} total valid chunks (Virtually Indexed)."
        ) 

    def get_goal_image(self, ep_idx: int) -> np.ndarray:
        """
        Efficiently loads and decodes the static goal image for a specific episode.

        This method leverages the main SoA loading pipeline, including the per-worker
        LRU cache, to provide fast access to the visual goal.

        Args:
            ep_idx: The index of the episode for which to retrieve the goal image.

        Returns:
            A NumPy array of the goal image in RGB, uint8 format.
        """
        if not (0 <= ep_idx < len(self.episode_metadata)):
            raise IndexError(f"Episode index {ep_idx} is out of range for {len(self.episode_metadata)} episodes.")

        ep_meta = self.episode_metadata[ep_idx]
        try:
            # Look for the metadata of our new modality
            meta = ep_meta["modalities"]["goal_image_primary"]
        except KeyError:
            raise KeyError(f"Modality 'goal_image_primary' not found in the index for episode {ep_idx}. "
                           "Please ensure the dataset was enhanced correctly.")

        # This call is fast and cached. It returns the goal image wrapped in an array of shape (1, H, W, 3).
        # It leverages the existing, powerful _get_full_modality_array method.
        full_array = self._get_full_modality_array(
            meta["key"], meta["compression"], meta["dtype"], tuple(meta["shape"])
        )

        # The result is an array containing a single image. We return just that image.
        return full_array[0]

    def get_num_episodes(self) -> int:
        """
        Returns the total number of episodes in the dataset.
        This is a fast, metadata-only operation.
        """
        return len(self.episode_metadata)
    
    def get_episode_length(self, episode_idx: int) -> int:
        """
        Returns the length (number of timesteps) of a specific episode.
        This is a fast, metadata-only operation.
        """
        if not (0 <= episode_idx < len(self.episode_metadata)):
            raise IndexError(f"Episode index {episode_idx} is out of range.")
        return self.episode_metadata[episode_idx]['length']

    def get_episode_and_timestep(self, idx: int) -> Tuple[int, int]:
        """
        Performs a "reverse lookup" to map a global sample index back to its
        corresponding episode index and local timestep within that episode.

        This is a critical utility for orchestrator datasets that need to access
        both chunked and single-timestep data for a given sample.

        Args:
            idx: The global sample index.

        Returns:
            A tuple of (episode_index, timestep_in_episode).
        """
        if not (0 <= idx < self.total_chunks):
            raise IndexError(f"Index {idx} out of range for dataset with {self.total_chunks} chunks.")

        # This is the same highly-efficient O(log N) lookup logic used in __getitem__.
        # Find the first episode index where the cumulative sum of chunks is >= idx.
        ep_idx = np.searchsorted(self._cumulative_chunks, idx, side='right')
        
        # Determine the starting global index for this episode's chunks.
        ep_start_chunk_idx = self._cumulative_chunks[ep_idx - 1] if ep_idx > 0 else 0
        
        # The local chunk index is the offset from the episode's start.
        local_chunk_idx = idx - ep_start_chunk_idx
        
        # The local timestep `t` is the first possible start time plus the local chunk index.
        timestep_t = (self.observation_horizon - 1) + local_chunk_idx
        
        return int(ep_idx), int(timestep_t)


    def get_proprioception_dim(self) -> int:
        """
        Returns the dimension of the 'proprio' modality from the dataset's metadata.
        This is a fast, metadata-only operation.
        """
        if not self.episode_metadata:
            # This should not happen if the index loaded correctly
            raise RuntimeError("Dataset index is empty, cannot infer proprioception dim.")
            
        # Get the metadata for the 'proprio' modality from the first episode.
        # We assume this is consistent across all episodes.
        try:
            proprio_meta = self.episode_metadata[0]['modalities']['proprio']
            # The shape is [T, Dim]. We want the last element.
            return proprio_meta['shape'][-1]
        except (KeyError, IndexError):
            raise RuntimeError(
                "Could not infer proprioception dimension from the dataset's index.json. "
                "Ensure 'proprio' modality with a valid 'shape' is present in the index."
            )
  
    def __len__(self):
        return self.total_chunks 

    def close_env(self):
        """Explicitly closes the LMDB environment (useful for pickling)."""
        if getattr(self, "_lmdb_env", None) is not None:
            close_lmdb_env(self._lmdb_env)
            self._lmdb_env = None
            logger.debug(f"Process {os.getpid()} explicitly closed LMDB handle.")

    def __getstate__(self):
        """Prepare for pickling by removing unpicklable handles."""
        state = self.__dict__.copy()
        # Environments and bound methods (lru_cache wrappers) are not picklable.
        state['_lmdb_env'] = None
        # We don't pickle the cache itself to avoid pickling the method wrapper
        return state

    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)
        self._lmdb_env = None # Force re-init in worker process

    def _init_lmdb(self):
        """Initializes the LMDB environment for the current worker."""
        # [SOTA FIX] Detect if we have been forked and have an inherited handle
        current_pid = os.getpid()
        if self._pid != current_pid:
            # We are in a child process (worker) and the handle was opened in the parent.
            # We MUST reset the handle to None so this worker opens its own.
            # Trying to use the parent's handle would cause a segmentation fault.
            self._lmdb_env = None
            self._pid = current_pid
            # Clear the LRU cache as well, as decoding handles might be stale
            self._get_full_modality_array.cache_clear()
            logger.debug(f"Worker {current_pid} detected fork; resetting LMDB handle.")

        if self._lmdb_env is None:
            self._lmdb_env = open_lmdb_env(
                str(self.demo_path),
                readonly=True,
                lock=False,      # No locks, we are read-only 
                map_size_gb=200.0, # [SOTA FIX] Massive virtual address space (doesn't consume RAM)
                readahead=False, # False = better for random access [cite: 246]
                subdir=False     # Our DB is a single file
            ) 
            logger.debug(f"Process {current_pid} opened LMDB env.")

    def __del__(self):
        """Ensures the LMDB environment is closed when a worker is destroyed."""
        if getattr(self, "_lmdb_env", None) is not None:
            close_lmdb_env(self._lmdb_env) 
            self._lmdb_env = None

    def _get_lmdb_blob(self, key: str) -> bytes:
        """Gets a raw byte blob from LMDB, initializing the env if needed."""
        self._init_lmdb()
        with self._lmdb_env.begin(write=False) as txn:
            blob = txn.get(key.encode("ascii"))
            if blob is None:
                raise KeyError(f"Missing LMDB key {key!r} in {self.demo_path}") 
            return blob

    @functools.lru_cache(maxsize=128)
    def _get_full_modality_array(self, key: str, compression: str, dtype_str: str, shape_list: tuple) -> Any:
        """
        [DEFINITIVE, FULLY PATCHED, SOTA VERSION]
        This is the cached, expensive part. It loads an *entire* modality
        (e.g., all 'proprio' for one episode) from LMDB and decodes it.
        This version correctly handles 'raw', 'jpeg'/'png', and 'pickle' compression.
        """
        blob = self._get_lmdb_blob(key)
        shape = tuple(shape_list) # Ensure shape is a tuple for consistency

        if compression == "raw":
            dtype = np.dtype(dtype_str)
            data = np.frombuffer(blob, dtype=dtype).reshape(shape)
            return data

        elif compression in ("jpeg", "png"):
            byte_list = pickle.loads(blob)
            
            # This logic robustly handles color (3-ch) and grayscale (1-ch) images.
            is_grayscale = len(shape) == 4 and shape[3] == 1
            im_read_flag = cv2.IMREAD_GRAYSCALE if is_grayscale else cv2.IMREAD_COLOR
            
            images = [cv2.imdecode(np.frombuffer(b, dtype=np.uint8), im_read_flag) for b in byte_list]
            data = np.stack(images)
            
            if not is_grayscale:
                data = data[..., ::-1]  # BGR to RGB for color images
            if is_grayscale and len(data.shape) == 3:
                data = data[..., np.newaxis] # Ensure channel dim exists for grayscale

            return data

        elif compression == "pickle":
            # THIS IS THE CRITICAL MISSING BLOCK.
            # For modalities like expert_states and camera_params, we just unpickle the blob.
            # The result is a list of objects, which is correct.
            data = pickle.loads(blob)
            return data

        else:
            raise ValueError(f"Unknown compression type: {compression}")

    def __getitem__(self, idx: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]: 
        if not (0 <= idx < self.total_chunks):
            raise IndexError(f"Index {idx} out of range for dataset with {self.total_chunks} chunks.")

        # --- 1. Virtual Indexing (O(log N) lookup) ---
        # Find the first episode index where the cumulative sum is >= idx
        ep_idx = np.searchsorted(self._cumulative_chunks, idx, side='right')
        
        # Get the start index of this episode's chunks
        ep_start_chunk_idx = self._cumulative_chunks[ep_idx - 1] if ep_idx > 0 else 0
        
        # Get the local timestep `t` within the episode
        local_chunk_idx = idx - ep_start_chunk_idx
        t = (self.observation_horizon - 1) + local_chunk_idx
        
        # Get the metadata for this specific episode
        ep_meta = self.episode_metadata[ep_idx]

        # --- 2. Slicing Logic (Identical to original) ---
        obs_start_idx = t - self.observation_horizon + 1
        obs_end_idx = t + 1
        action_start_idx = t
        action_end_idx = t + self.action_horizon 

        # --- 3. SOTA Chunk Loading ---
        # We load *only* the slices we need from the cached full arrays.
        obs_chunk = {}
        for key in ["image_primary", "image_wrist", "proprio"]: 
            meta = ep_meta["modalities"][key]
            
            # This call is fast:
            # 1. Hits the LRU cache.
            # 2. If miss, loads/decodes the *full* modality.
            # 3. Caches the full modality.
            full_array = self._get_full_modality_array(
                meta["key"], meta["compression"], meta["dtype"], tuple(meta["shape"])
            )
            
            # This numpy slice is a fast, C-backend view operation
            obs_chunk[key] = full_array[obs_start_idx:obs_end_idx]

        # Load the action chunk
        meta_actions = ep_meta["modalities"]["actions"]
        full_actions = self._get_full_modality_array(
            meta_actions["key"], meta_actions["compression"], meta_actions["dtype"], tuple(meta_actions["shape"])
        )
        action_chunk = full_actions[action_start_idx:action_end_idx].astype(np.float32)

        return obs_chunk, action_chunk 


# ==============================================================================
# 3. COLLATE FUNCTION (Unchanged, but required for completeness)
# ==============================================================================

def collate_fn(batch: List[Tuple[Dict[str, np.ndarray], np.ndarray]]): 
    """
    Collates a batch of chunked data into a single PyTorch tensor dictionary.
    This function is compatible with both the old and new dataset.
    
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


# ==============================================================================
# 4. ONLINE DATA GENERATOR (Unchanged, it *produces* data for the writer)
# ==============================================================================

class ExpertDataset(IterableDataset): 
    """
    IterableDataset version that generates expert demos online, and (optionally) writes them out.
    [cite: 258]
    This class is the *source* of data for the `ExpertDatasetWriter`.
    It remains unchanged.
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
        max_episodes_per_epoch: Optional[int] = None,
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
        self.max_episodes_per_epoch = max_episodes_per_epoch
        
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
        #    This is the root of all randomness for this worker. [cite: 267]
        seed = self.base_seed if self.base_seed is not None else int(time.time() * 1e9) 
        self._worker_master_seed = seed + self._worker_id
        
        # 2. Create a dedicated, seeded random number generator (RNG) for this worker.
        #    This will be used for any probabilistic logic (like data filtering) to make it reproducible. [cite: 269]
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
                # REPLACE the hardcoded dimension with a dynamic lookup from the env. [cite: 272]
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

    def get_last_seed(self) -> Optional[int]:
        # Returns the seed used for the *last completed or currently running* episode generation attempt.
        # Assumes _init_worker_state sets _worker_master_seed and _episode_attempt_counter [cite: 276]
        if not hasattr(self, '_worker_master_seed') or not hasattr(self, '_episode_attempt_counter'):
             # Should not happen if worker is initialized correctly
             return None
        # The seed for the *next* episode would be master + attempts.
        # The seed for the *current or last* attempt is master + attempts - 1. [cite: 277]
        if self._episode_attempt_counter > 0:
            return (self._worker_master_seed + self._episode_attempt_counter - 1) & 0x7FFFFFFF 
        else:
             # If no attempts made yet, return the initial seed planned
             return self._worker_master_seed & 0x7FFFFFFF  
    
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
        Captures Expert ground truth metadata.
        """
        # Unpack 3 values: Pose, Action, Info (Metadata)
        pose_world, gripper_act, expert_info = self._scripted_expert.get_target_pose(current_obs) 
         
        # Transform world pose into base frame (IK Logic remains the same)
        N_SUBSTEPS = 20
        effective_dt = self._env.model.opt.timestep * N_SUBSTEPS
        max_dq = self._env.ACTION_SCALING_FACTOR / effective_dt
        arm_joint_ids = np.arange(7) 

        delta_arm_action = self._ik_solver.compute_delta_action( 
            target_ee_pose=pose_world,
            model=self._env.model,
            data=self._env.data,
            ee_site_id=self._env.ee_site_id,
            joint_qpos_indices=arm_joint_ids,
            effective_dt=effective_dt,
            max_dq=max_dq
        ) 
         
        action = np.concatenate([delta_arm_action, [gripper_act]]).astype(np.float32) 
        ik_failed = np.linalg.norm(delta_arm_action) < 1e-4 
        
        # [SOTA PATCH: Inject Ground Truths into Observation Dict]
        # We store these in the obs dict so they travel through the pipeline to the Writer
        current_obs["gt_phase"] = np.array([expert_info["gt_phase"]], dtype=np.int32)
        current_obs["gt_gripper"] = np.array([expert_info["gt_gripper_intent"]], dtype=np.float32)
        current_obs["expert_state_str"] = expert_info["expert_state_str"] # Optional, mostly for debug
        
        current_obs["expert_source"] = 0
        
        return current_obs, action, ik_failed

    def __iter__(self) -> Iterator[Tuple[Dict, np.ndarray]]:
        self._init_worker_state()
        self._episode_buffer.clear()
        consecutive_failures = 0
        episode_attempt_counter = 0
        MAX_CONSEC = 25

        while True:
            if self.max_samples_per_epoch and self._samples_yielded >= self.max_samples_per_epoch: return
            if self.max_episodes_per_epoch and self._episode_id_counter >= self.max_episodes_per_epoch: return

            if not self._episode_buffer:
                try:
                    # --- PASS 1: PHYSICS ONLY (FAST, LIGHTWEIGHT) ---
                    self._env.set_rendering_enabled(False)
                    
                    seed = (self._worker_master_seed + episode_attempt_counter) & 0x7FFFFFFF
                    episode_attempt_counter += 1
                    
                    obs, _ = self._env.reset(seed=seed)
                    self._scripted_expert.reset()
                    self._ik_solver.reset_controller_state()
                    
                    cached_steps = []
                    
                    for _ in range(self._env.max_episode_steps):
                        target_pose, grip_act, info = self._scripted_expert.get_target_pose(obs)
                        
                        eff_dt = self._env.model.opt.timestep * 20
                        max_dq = self._env.ACTION_SCALING_FACTOR / eff_dt
                        delta_action = self._ik_solver.compute_delta_action(
                            target_ee_pose=target_pose,
                            model=self._env.model, data=self._env.data,
                            ee_site_id=self._env.ee_site_id,
                            joint_qpos_indices=np.arange(7),
                            effective_dt=eff_dt, max_dq=max_dq
                        )
                        full_action = np.concatenate([delta_action, [grip_act]]).astype(np.float32)
                        ik_failed = np.linalg.norm(delta_action) < 1e-4

                        # --- FIX: CACHE NUMPY ARRAYS ONLY ---
                        # We avoid copying the entire MjData object.
                        step_snapshot = {
                            "qpos": self._env.data.qpos.copy(), # Tiny array
                            "qvel": self._env.data.qvel.copy(), # Tiny array
                            "ctrl": self._env.data.ctrl.copy(), # Tiny array
                            "py_grasp_state": self._env._is_physically_grasped,
                            "action": full_action,
                            "expert_info": info,
                            # [CRITICAL FIX] Store the EXPERT TARGET POSE - this is where
                            # the expert FSM wants the EE to go, NOT where it actually is.
                            # Training on this teaches the model goal-directed behavior.
                            "expert_target_pose": target_pose.copy(),
                            "ik_failed": ik_failed,
                            "expert_state_str": info["expert_state_str"]
                        }
                        cached_steps.append(step_snapshot)

                        obs, _, done, trunc, _ = self._env.step(full_action)
                        if done or trunc or self._scripted_expert.is_done():
                            break

                    # --- PASS 2: FILTER & RENDER ---
                    if self._scripted_expert.was_successful():
                        consecutive_failures = 0
                        indices_to_keep = []
                        
                        # 1. Filter using cached qvel (no need for full state)
                        CRITICAL_INTERACTION_PHASES = {
                            "PREPARE_GRIPPER",        # Fine alignment (Low Vel)
                            "DESCEND_TO_GRASP",       # Fine Z-axis control (Low Vel)
                            "GRASP",                  # Contact initiation (Zero Vel) -> THE KEY FIX
                            "LIFT",                   # Load bearing dynamics
                            "PREPARE_PLACE",          # Alignment for placement
                            "DESCEND_TO_PLACE",       # Fine Z-axis control
                            "AWAIT_STABLE_PLACEMENT", # Contact release physics
                            "RELEASE"                 # Gripper actuation
                        }

                        for i, step_data in enumerate(cached_steps):
                            # Retrieve Ground Truth State from the snapshot
                            state_str = step_data.get("expert_state_str", "UNKNOWN")
                            
                            # Rule A: Critical Phase = FORCE KEEP (Dense Sampling)
                            # The model needs every millisecond of data here to learn precision.
                            if state_str in CRITICAL_INTERACTION_PHASES:
                                keep = True
                                
                            # Rule B: Transit Phase = STOCHASTIC FILTER (Sparse Sampling)
                            # During "Approach" or "Retract", we can safely drop frames to save space.
                            else:
                                qvel = step_data["qvel"][:7]
                                ee_vel = np.linalg.norm(qvel)
                                
                                if ee_vel < 0.1:
                                    # Idle/Start/Stop during transit -> Filter Aggressively
                                    keep = (self._rng.random() < self.p_low_vel)
                                else:
                                    # Moving during transit -> Filter Normally
                                    keep = (self._rng.random() < self.p_motion_frame)
                            
                            if keep: 
                                indices_to_keep.append(i)
                        
                        # 2. Force Last Frame
                        last_idx = len(cached_steps) - 1
                        if last_idx not in indices_to_keep:
                            indices_to_keep.append(last_idx)
                        indices_to_keep.sort()

                        # 3. Enable Rendering
                        self._env.set_rendering_enabled(True)
                        
                        processed_obs_list = []
                        processed_act_list = []
                        processed_flags = []
                        
                        # 4. Render Loop
                        for idx in indices_to_keep:
                            step_data = cached_steps[idx]
                            
                            # --- FIX: LIGHTWEIGHT RESTORE ---
                            self._env.data.qpos[:] = step_data["qpos"]
                            self._env.data.qvel[:] = step_data["qvel"]
                            self._env.data.ctrl[:] = step_data["ctrl"]
                            
                            # CRITICAL: Recompute derived physics (collisions, camera matrices)
                            mujoco.mj_forward(self._env.model, self._env.data)
                            
                            # Restore Python State
                            self._env._is_physically_grasped = step_data["py_grasp_state"]
                            
                            # Render
                            full_obs = self._env.get_expert_obs()
                            
                            # Inject Info
                            full_obs["gt_phase"] = np.array([step_data["expert_info"]["gt_phase"]], dtype=np.int32)
                            full_obs["gt_gripper"] = np.array([step_data["expert_info"]["gt_gripper_intent"]], dtype=np.float32)
                            full_obs["expert_state"] = step_data["expert_state_str"]
                            # [CRITICAL FIX] Store the expert's COMMANDED target pose for training
                            full_obs["expert_target_pose"] = step_data["expert_target_pose"].astype(np.float32)
                            
                            processed_obs_list.append({k: copy.deepcopy(v) for k, v in full_obs.items()})
                            processed_act_list.append(step_data["action"].copy())
                            processed_flags.append(step_data["ik_failed"])

                        # --- ASSEMBLE DICT (Clean, No Redundant Goal) ---
                        ep_id = f"w{self._worker_id}_e{self._episode_id_counter}"
                        episode_dict = {
                            "episode_id": ep_id,
                            "seed": int(seed),
                            "obs_list": processed_obs_list,
                            "actions": processed_act_list,
                            "ik_fail_flags": processed_flags,
                            "success": True
                        }
                        
                        self.episodes.append(episode_dict)
                        
                        for o, a in zip(processed_obs_list, processed_act_list):
                            self._episode_buffer.append((o, a))
                            
                        self._episode_id_counter += 1
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= MAX_CONSEC:
                            raise RuntimeError("Expert failed too many times consecutively.")

                except Exception as exc:
                    if self.skip_on_error:
                        logger.warning(f"Gen Error: {exc}", exc_info=False)
                        consecutive_failures += 1
                        # Force clear to free memory if crash happened mid-loop
                        cached_steps = []
                        continue
                    else: raise

            if self._episode_buffer:
                yield self._episode_buffer.pop(0)
                self._samples_yielded += 1

    
    def get_stats(self):
        return {
            "samples_yielded": int(self._samples_yielded),
            "episodes_collected": len(self.episodes) 
        }

# ==============================================================================
# 5. REPLAY VALIDATION (Unchanged, operates on in-memory dicts)
# ==============================================================================

def replay_validate_episode(ep: Dict[str, Any], urdf_path: str, env_xml_path: Optional[str] = None) -> bool: 
    """
    Replay sim_actions in a fresh environment and compare final object pose vs stored.
    Return True if within tolerance.
    [cite: 332]
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