# FILE: utils/dataset_enhancer.py
#
# Definitive, SOTA, Multiprocessed Dataset Enhancement Pipeline for ViP-C.
# This script transforms the "raw enhanced" dataset from Phase 1 into the final,
# "training-ready" dataset by adding the crucial `task_phases` and
# `subgoal_heatmaps` modalities.
#
""""""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import pickle
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lmdb
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import cv2

# It is assumed this script is run from the project's root directory
# or that the project's root is in the Python path.
from utils.expert_dataset import ExpertTrajectoryDataset

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dataset_enhancer")

# This mapping is the canonical source of truth for converting the expert's
# internal FSM state into the TaskPhase enum for the ViP-C Planner.
EXPERT_STATE_TO_TASK_PHASE = {
    "MOVE_TO_PRE_GRASP": 0,
    "PREPARE_GRIPPER": 0,
    "DESCEND_TO_GRASP": 0,
    "GRASP": 1,
    "LIFT": 2,
    "MOVE_TO_GOAL": 2,
    "PREPARE_PLACE": 3,
    "DESCEND_TO_PLACE": 3,
    "AWAIT_STABLE_PLACEMENT": 3,
    "RELEASE": 4,
    "RETRACT": 4,
    "DONE": -1,  # Terminal state, not used for training
}
def _compress_heatmaps_to_png_list(heatmaps_stack: np.ndarray) -> List[bytes]:
    """Quantizes and compresses a stack of float32 heatmaps."""
    # Normalize from [0.0, 1.0] to [0, 255] and convert to uint8
    heatmaps_uint8 = (np.clip(heatmaps_stack, 0.0, 1.0) * 255).astype(np.uint8)
    
    # Use cv2 to encode each heatmap as a PNG in memory
    # PNG is lossless, so no information is lost in this compression.
    # Level 1 compression is the fastest.
    encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 1]
    
    compressed_list = []
    for i in range(heatmaps_uint8.shape[0]):
        result, encoded_image = cv2.imencode(".png", heatmaps_uint8[i], encode_param)
        if not result:
            raise RuntimeError("Failed to encode heatmap to PNG.")
        compressed_list.append(encoded_image.tobytes())
        
    return compressed_list

# --- Core Geometric and Data Generation Utilities ---

def _project_world_to_pixel(
    world_pos: np.ndarray,
    camera_params: Dict[str, Any],
) -> Tuple[int, int]:
    """
    Performs a precise 3D world coordinate to 2D pixel coordinate projection.

    This function implements the standard computer vision pipeline:
    World Space -> Camera Space -> Clip Space -> NDC Space -> Pixel Space.
    It uses the exact camera parameters saved for the specific timestep.

    Args:
        world_pos: The 3D point in world coordinates (shape: [3,]).
        camera_params: A dictionary containing the camera's pose and intrinsics.

    Returns:
        A tuple (u, v) representing the pixel coordinates.
    """
    cam_pos = camera_params["pos"]
    cam_quat_xyzw = camera_params["quat_xyzw"]
    fovy_deg = camera_params["fovy"]
    height = camera_params["height"]
    width = camera_params["width"]

    # 1. World to Camera Space Transformation
    R_wc = R.from_quat(cam_quat_xyzw).as_matrix()
    R_cw = R_wc.T
    t_cw = -R_cw @ cam_pos
    point_in_cam_space = R_cw @ world_pos + t_cw

    # 2. Camera to Clip Space (Perspective Projection)
    # MuJoCo cameras look along their -Z axis.
    if point_in_cam_space[2] >= 0:
        return -1, -1  # Point is behind the camera

    fovy_rad = np.deg2rad(fovy_deg)
    focal_length_y = height / (2 * np.tan(fovy_rad / 2))
    focal_length_x = focal_length_y # Assume square pixels

    # 3. Perspective Divide (Clip to NDC) & NDC to Pixel
    # Invert camera Z because it's negative
    z = -point_in_cam_space[2]
    u = (point_in_cam_space[0] * focal_length_x) / z + (width / 2)
    v = (-point_in_cam_space[1] * focal_length_y) / z + (height / 2) # Y is inverted in pixel space

    return int(round(u)), int(round(v))


def _generate_heatmap(
    center_uv: Tuple[int, int],
    size: Tuple[int, int],
    sigma: float,
) -> np.ndarray:
    """
    Generates a 2D Gaussian heatmap in a fully vectorized manner.

    Args:
        center_uv: The (u, v) pixel coordinate for the center of the peak.
        size: The (height, width) of the desired heatmap.
        sigma: The standard deviation of the Gaussian in pixels.

    Returns:
        A 2D numpy array of shape `size` with the rendered Gaussian.
    """
    height, width = size
    u, v = center_uv

    if not (0 <= u < width and 0 <= v < height):
        return np.zeros(size, dtype=np.float32)

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    d_sq = (x - u)**2 + (y - v)**2
    heatmap = np.exp(-d_sq / (2 * sigma**2))
    
    return heatmap.astype(np.float32)


# --- Worker Function for Multiprocessing ---

def process_episode_wrapper(args: Dict[str, Any]) -> Tuple[int, Dict[str, list]]:
    """
    A simple wrapper to allow `_enhance_one_episode` to be called by a
    multiprocessing pool. It unpacks arguments for clarity.
    """
    episode_idx = args["episode_idx"]
    source_db_path = args["source_db_path"]
    heatmap_config = args["heatmap_config"]

    # Each worker gets its own reader instance to avoid sharing DB handles.
    reader = ExpertTrajectoryDataset(
        demo_path=source_db_path,
        observation_horizon=1, # We only need to read full episodes, so horizons are minimal.
        action_horizon=1,
    )

    return _enhance_one_episode(reader, episode_idx, heatmap_config)


# In FILE: utils/dataset_enhancer.py

def _enhance_one_episode(
    reader: ExpertTrajectoryDataset,
    episode_idx: int,
    heatmap_config: Dict[str, Any],
) -> Tuple[int, Dict[str, list]]:
    """
    [DEFINITIVE, VERIFIABLE VERSION]
    The core logic for processing a single episode.
    """
    # --- START OF VERIFIABLE PATCH ---
    print(f"--- DEBUG: Now processing episode {episode_idx} with the FINAL, PATCHED function. ---")
    # --- END OF VERIFIABLE PATCH ---
    
    ep_meta = reader.episode_metadata[episode_idx]
    ep_len = ep_meta["length"]

    def get_modality_with_hashable_args(modality_name: str):
        """Helper to prepare args and call the reader, ensuring hashable types."""
        meta = ep_meta["modalities"][modality_name]
        meta_args = {
            "key": meta["key"],
            "compression": meta["compression"],
            "dtype_str": meta["dtype"],
            "shape_list": tuple(meta["shape"]) # The critical conversion
        }
        return reader._get_full_modality_array(**meta_args)

    all_expert_states = get_modality_with_hashable_args("expert_states")
    all_cam_params = get_modality_with_hashable_args("camera_params")
    all_ee_poses = get_modality_with_hashable_args("ee_pose_world")

    task_phases = np.array(
        [EXPERT_STATE_TO_TASK_PHASE.get(state, -1) for state in all_expert_states],
        dtype=np.int32,
    )

    subgoal_heatmaps = []
    for t in range(ep_len):
        current_phase = task_phases[t]
        target_keyframe_idx = -1

        if current_phase == 0:  # APPROACHING_OBJECT
            future_indices = np.where(task_phases[t:] == 1)[0]
            if len(future_indices) > 0:
                target_keyframe_idx = t + future_indices[0]
        elif current_phase == 2: # TRANSPORTING_OBJECT_TO_GOAL
            future_indices = np.where(task_phases[t:] == 3)[0]
            if len(future_indices) > 0:
                target_keyframe_idx = t + future_indices[0]
        elif current_phase == 1: # EXECUTING_GRASP
            target_keyframe_idx = t
        elif current_phase == 3: # PLACING_OBJECT
             future_indices = np.where(task_phases[t:] == 4)[0]
             if len(future_indices) > 0:
                target_keyframe_idx = t + future_indices[0]
        elif current_phase == 4: # RELEASING_AND_RETRACTING
            target_keyframe_idx = ep_len - 1

        if target_keyframe_idx != -1:
            target_world_pos = all_ee_poses[target_keyframe_idx, :3]
            current_cam_params = all_cam_params[t]
            pixel_coord = _project_world_to_pixel(target_world_pos, current_cam_params)
            heatmap = _generate_heatmap(
                pixel_coord,
                size=(heatmap_config["height"], heatmap_config["width"]),
                sigma=heatmap_config["sigma"],
            )
            subgoal_heatmaps.append(heatmap)
        else:
            subgoal_heatmaps.append(
                np.zeros((heatmap_config["height"], heatmap_config["width"]), dtype=np.float32)
            )

    compressed_heatmaps = _compress_heatmaps_to_png_list(np.stack(subgoal_heatmaps))

    new_modalities = {
        "task_phases": task_phases,
        "subgoal_heatmaps_compressed": compressed_heatmaps,
    }
    
    return episode_idx, new_modalities

# --- Main Orchestration Script ---

def main(args):
    """
    The main orchestration function. Manages I/O, multiprocessing, and
    the final dataset assembly.
    """
    logger.info("Starting ViP-C Dataset Enhancement Pipeline.")
    source_path = Path(args.source_db)
    dest_path = Path(args.dest_db)

    if not source_path.exists():
        logger.error(f"Source dataset not found at: {source_path}")
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and not args.overwrite:
        logger.error(f"Destination path {dest_path} already exists. Use --overwrite to replace.")
        return

    logger.info("Locating and preparing source and destination paths...")

    # Robustly find the source index file based on the LMDB stem name
    source_index_path = source_path.parent / f"{source_path.stem}_index.json"
    if not source_index_path.exists():
        logger.error(f"FATAL: Corresponding index file not found for source DB!")
        logger.error(f"Looked for: {source_index_path}")
        return

    # Define destination paths
    dest_index_path = dest_path.parent / f"{dest_path.stem}_index.json"

    logger.info(f"  - Source DB:    {source_path}")
    logger.info(f"  - Source Index: {source_index_path}")
    logger.info(f"  - Dest DB:      {dest_path}")
    logger.info(f"  - Dest Index:   {dest_index_path}")

    # Copy BOTH the database and its index file to the destination
    logger.info(f"Copying original dataset files...")
    shutil.copy(source_path, dest_path)
    shutil.copy(source_index_path, dest_index_path)
    logger.info("Copy complete. Now enhancing the new dataset.")

    # Load the COPIED index file for modification
    with open(dest_index_path, "r") as f:
        source_index_data = json.load(f)
    
    num_episodes = len(source_index_data["episodes"])
    logger.info(f"Found {num_episodes} episodes to process.")

    # Prepare arguments for each worker process.
    worker_args = [
        {
            "episode_idx": i,
            "source_db_path": str(dest_path), # Workers read from the copied DB
            "heatmap_config": {"height": 56, "width": 56, "sigma": args.sigma},
        }
        for i in range(num_episodes)
    ]

    # Open the destination database for writing the new modalities.
    dest_env = lmdb.open(str(dest_path), map_size=int(1.8 * 1024**3), subdir=False, readonly=False, lock=True)
    
    try:
        if args.num_workers > 0:
            logger.info(f"Starting parallel processing with {args.num_workers} workers.")
            with mp.Pool(processes=args.num_workers) as pool:
                with tqdm(total=num_episodes, desc="Enhancing Episodes") as pbar:
                    # Use imap_unordered for efficient progress bar updates.
                    for ep_idx, new_modalities in pool.imap_unordered(process_episode_wrapper, worker_args):
                        with dest_env.begin(write=True) as txn:
                            ep_prefix = f"ep_{ep_idx:06d}"
                            
                            # Write task_phases (raw numpy)
                            phases_key = f"{ep_prefix}_task_phases"
                            txn.put(phases_key.encode('ascii'), new_modalities["task_phases"].tobytes())

                            # Write subgoal_heatmaps (pickled list of compressed arrays could be more efficient,
                            # but raw numpy is simpler and fine for this size).
                            heatmaps_key = f"{ep_prefix}_subgoal_heatmaps"
                            # Correctly access the compressed data and use pickle for serialization.
                            txn.put(heatmaps_key.encode('ascii'), pickle.dumps(new_modalities["subgoal_heatmaps_compressed"]))
                        pbar.update(1)
        else:
            logger.info("Starting sequential processing (single-threaded).")
            for arg_set in tqdm(worker_args, desc="Enhancing Episodes"):
                 ep_idx, new_modalities = process_episode_wrapper(arg_set)
                 with dest_env.begin(write=True) as txn:
                    ep_prefix = f"ep_{ep_idx:06d}"
                    phases_key = f"{ep_prefix}_task_phases"
                    txn.put(phases_key.encode('ascii'), new_modalities["task_phases"].tobytes())
                    heatmaps_key = f"{ep_prefix}_subgoal_heatmaps"
                    # The data is already a list of byte strings, so we just pickle it.
                    # This now mirrors the logic from the parallel processing block.
                    txn.put(heatmaps_key.encode('ascii'), pickle.dumps(new_modalities["subgoal_heatmaps_compressed"]))

        logger.info("All episodes processed. Updating the final index file...")

        # Update the index file with the new modality metadata.
        for i in range(num_episodes):
            ep_prefix = f"ep_{i:06d}"
            source_index_data["episodes"][i]["modalities"]["task_phases"] = {
                "key": f"{ep_prefix}_task_phases", "compression": "raw",
                "dtype": "int32", "shape": [source_index_data["episodes"][i]["length"]]
            }

            source_index_data["episodes"][i]["modalities"]["subgoal_heatmaps"] = {
                "key": f"{ep_prefix}_subgoal_heatmaps",
                "compression": "png", # SOTA compression format
                "dtype": "uint8",     # SOTA quantized dtype
                "shape": [source_index_data["episodes"][i]["length"], 56, 56, 1] # Add channel dim
            }

        final_index_path = dest_path.parent / f"{dest_path.stem}_index.json"
        
        with open(final_index_path, "w") as f:
            # --- [DELETE THIS INCORRECT LINE] ---
            # json.dump(index_data, f, indent=2) # Renamed variable for clarity
            # --- [END DELETE] ---

            # --- [REPLACE WITH THE CORRECT VARIABLE NAME] ---
            json.dump(source_index_data, f, indent=2)
            # --- [END REPLACE] ---
        
        logger.info(f"Final enhanced index saved to {final_index_path}.")

    finally:
        dest_env.sync()
        dest_env.close()
        logger.info("LMDB environment closed.")

    logger.info("Dataset Enhancement Pipeline finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SOTA Offline Dataset Enhancement Pipeline for ViP-C.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--source-db",
        required=True,
        type=str,
        help="Path to the 'raw enhanced' LMDB dataset from Phase 1.",
    )
    parser.add_argument(
        "--dest-db",
        required=True,
        type=str,
        help="Path to write the new, final, 'training-ready' LMDB dataset.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, mp.cpu_count() - 2),
        help="Number of parallel worker processes to use. Set to 0 for single-threaded debug mode.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=5.0,
        help="Standard deviation (in pixels) for the Gaussian subgoal heatmaps.",
    )
    parser.add_argument(
        "--map-size-gb",
        type=float,
        default=100.0,
        help="LMDB virtual map size in gigabytes. Should be larger than the final dataset size.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, allows overwriting an existing destination database.",
    )

    start_time = time.time()
    main(parser.parse_args())
    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds.")




# python -m utils.dataset_enhancer --source-db "C:\Users\Hellx\Documents\Programming\python\Project\redhot\data\validation\expert_expert_run_validation_dataset_10_episodes\expert_expert_run_validation_dataset_10_episodes.lmdb" --dest-db "data\validation_\validation_dataset.lmdb" --num-workers 0 --overwrite

#python -m utils.dataset_enhancer --source-db "C:\Users\Hellx\Documents\Programming\python\Project\redhot\data\training_temp\expert_expert_run_validation_dataset_80_episodes\expert_expert_run_validation_dataset_80_episodes.lmdb" --dest-db "data\training\training_dataset.lmdb" --num-workers 0 --overwrite
