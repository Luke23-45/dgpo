# FILE: scripts/visualize_dataset.py
# (UPDATED for State-of-the-Art SoA Dataset Format)

import argparse
import os
import sys
import random
import pickle
import json
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm import tqdm
import functools

# --- SOTA Imports ---
try:
    import cv2
except ImportError:
    print("Error: OpenCV (cv2) package not found. Please install it with 'pip install opencv-python'")
    sys.exit(1)

try:
    import lmdb
except ImportError:
    lmdb = None
    print("Error: LMDB package not found. Please install it with 'pip install lmdb'")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# --- Patched LMDB Helpers (keep for consistency) ---
def open_lmdb_env(path: str, readonly: bool, lock: bool, readahead: bool = False, subdir: bool = False, map_size_gb: float = 1.0):
    map_size = int(map_size_gb * (1024**3))
    return lmdb.open(path, readonly=readonly, lock=lock, readahead=readahead, subdir=subdir, map_size=map_size, meminit=False)

def close_lmdb_env(env):
    if env is not None:
        env.close()

# ==============================================================================
# 1. NEW SOTA EPISODE LOADER (Replaces the old one)
# ==============================================================================
class SoAEpisodeLoader:
    """
    Handles loading and reconstructing full episodes from the SOTA
    "Struct of Arrays" (SoA) LMDB + JSON index format.
    """
    def __init__(self, demo_path: str):
        self.demo_path = Path(demo_path)
        self._lmdb_env = None
        
        if not self.demo_path.exists():
            raise FileNotFoundError(f"Demo path not found: {self.demo_path}")

        # --- 1. Load the JSON Index ---
        index_path = self.demo_path.parent / f"{self.demo_path.stem}_index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Required index file not found: {index_path}")
        
        logger.info(f"Loading SOTA index from {index_path}...")
        with open(index_path, 'r') as f:
            self.index_data = json.load(f)
            
        self.episode_metadata = self.index_data["episodes"]
        logger.info(f"Indexed {len(self.episode_metadata)} episodes.")
        
        # --- 2. Open LMDB Handle ---
        self._lmdb_env = open_lmdb_env(str(self.demo_path), readonly=True, lock=False, readahead=False, subdir=False)
        if self._lmdb_env is None:
            raise RuntimeError("LMDB environment could not be opened.")

    def __len__(self):
        return len(self.episode_metadata)

    def _get_lmdb_blob(self, key: str) -> bytes:
        """Gets a raw byte blob from LMDB."""
        with self._lmdb_env.begin(write=False) as txn:
            blob = txn.get(key.encode("ascii"))
            if blob is None:
                raise KeyError(f"Missing LMDB key {key!r}")
            return blob

    @functools.lru_cache(maxsize=32) # Cache full modality arrays to speed up reconstruction
    def _get_full_modality_array(self, key: str, compression: str, dtype_str: str, shape_tuple: tuple) -> np.ndarray:
        """Loads and decodes a single, full modality array for one episode."""
        blob = self._get_lmdb_blob(key)
        dtype = np.dtype(dtype_str)
        # The shape is already a tuple, no change needed here
        shape = shape_tuple

        if compression == "raw":
            data = np.frombuffer(blob, dtype=dtype).reshape(shape)
        elif compression in ("jpeg", "png"):
            byte_list = pickle.loads(blob)
            images = [cv2.imdecode(np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_COLOR) for b in byte_list]
            data = np.stack(images)

        else:
            raise ValueError(f"Unknown compression type: {compression}")
        return data

    def get_episode(self, index: int) -> Dict[str, Any]:
        """
        Retrieves and reconstructs a single episode from the SoA format
        back into the legacy "List of Structs" (LoS) format for rendering.
        """
        if not (0 <= index < len(self)):
            raise IndexError(f"Episode index {index} out of bounds.")
            
        ep_meta = self.episode_metadata[index]
        episode_len = ep_meta["length"]

        # --- Reconstruct all modalities ---
        modalities = {}
        for name, meta in ep_meta["modalities"].items():
            
            # --- START OF PATCH 2 ---
            # Convert the 'shape' list from the JSON into a tuple before passing it
            # to the lru_cached function. This makes the arguments hashable.
            shape_as_tuple = tuple(meta["shape"])
            # --- END OF PATCH 2 ---

            modalities[name] = self._get_full_modality_array(
                meta["key"], meta["compression"], meta["dtype"], shape_as_tuple # Pass the tuple here
            )
            
        # --- Rebuild the 'obs_list' from the arrays ---
        obs_list = []
        for t in range(episode_len):
            obs_step = {
                # Add other observation keys here as needed
                "image_primary": modalities["image_primary"][t],
                "image_wrist": modalities["image_wrist"][t],
                "proprio": modalities["proprio"][t],
                # Add mock data for keys not stored in SoA format if renderer needs them
                "ee_pose_world": np.zeros(7), 
                "object_pos_world": np.zeros(3)
            }
            obs_list.append(obs_step)

        # --- Reconstruct the final episode dictionary ---
        reconstructed_ep = {
            "episode_id": ep_meta.get("episode_id", f"ep{index}"),
            "seed": ep_meta.get("seed"),
            "success": ep_meta.get("success"),
            "obs_list": obs_list,
            "actions": modalities.get("actions", [])
        }
        return reconstructed_ep

    def close(self):
        close_lmdb_env(self._lmdb_env)

# ==============================================================================
# 2. RENDERER (Unchanged, as it consumes the LoS format we reconstruct)
# ==============================================================================
class ExpertVideoRenderer:
    """Renders diagnostic videos from episode data."""

    def __init__(self, fps: int = 10):
        self.fps = fps

    def render_episode(self, ep: Dict[str, Any], output_path: Path):
        """Renders one episode to a video file."""
        obs_list = ep.get("obs_list", [])
        actions = ep.get("actions", [])
        
        if not obs_list:
            logger.warning(f"Episode {ep.get('episode_id', 'Unknown')} is empty. Skipping.")
            return

        H, W, _ = obs_list[0]["image_primary"].shape
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use a common codec
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (W, H))
        if not out.isOpened():
             logger.error(f"Could not open VideoWriter at {output_path}.")
             return

        episode_id = ep.get('episode_id', 'N/A')
        seed = ep.get('seed', 'N/A')
        is_success = ep.get('success', False)
        
        for t in tqdm(range(len(obs_list)), desc=f"Rendering {episode_id}", leave=False):
            frame_rgb = obs_list[t]["image_primary"].copy()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            text_color = (0, 255, 0) if is_success else (0, 0, 255)
            cv2.putText(frame_bgr, f"ID: {episode_id} | Seed: {seed}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame_bgr, f"Success: {is_success}", (W - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, cv2.LINE_AA)
            cv2.putText(frame_bgr, f"Step: {t}/{len(obs_list) - 1}", (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            out.write(frame_bgr)

        out.release()
        logger.info(f"Successfully rendered video to {output_path}")

# ==============================================================================
# 3. MAIN SCRIPT (Main change is to use the new loader)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Render diagnostic videos from SOTA expert datasets.")
    parser.add_argument("--demo-path", type=str, required=True, help="Path to the .lmdb SOTA demo file.")
    # ... (the rest of your argparse is perfect)
    parser.add_argument("--output-dir", type=str, default="videos/inspection", help="Directory to save output videos.")
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of random episodes to render.")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second for the output video.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for episode selection.")
    
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = None
    try:
        # --- USE THE NEW SoA LOADER ---
        logger.info(f"Loading SOTA dataset index from {args.demo_path}")
        loader = SoAEpisodeLoader(args.demo_path)
        
        total_episodes = len(loader)
        if total_episodes == 0:
            logger.error("Dataset contains zero episodes. Exiting.")
            return

        renderer = ExpertVideoRenderer(fps=args.fps)
        
        indices_to_render = random.sample(range(total_episodes), min(args.num_episodes, total_episodes))
        logger.info(f"Selected {len(indices_to_render)} episodes for rendering: {indices_to_render}")

        for ep_idx in indices_to_render:
            try:
                # The loader reconstructs the episode into the format the renderer expects
                ep = loader.get_episode(ep_idx)
                
                ep_id_str = str(ep.get('episode_id', f'ep{ep_idx}')).replace("/", "_").replace("\\", "_")
                # Use .mp4 for better compatibility
                output_path = output_dir / f"{ep_id_str}_render.mp4"
                
                renderer.render_episode(ep, output_path)
            
            except Exception as e:
                logger.error(f"Failed to render episode {ep_idx}: {e}", exc_info=True)

    except Exception as e:
        logger.critical(f"Fatal error during data loading or indexing: {e}", exc_info=True)
    finally:
        if loader:
            loader.close()
        
    logger.info("Video generation complete.")

if __name__ == "__main__":
    main()

"""
python -m s13 --demo-path   --output-dir videos/inspection --num-episodes 5 --fps 15 --seed 42


tests.datasets.visualize_dataset

"""