# FILE: scripts/visualize_dataset.py

import argparse
import os
import sys
import random
import pickle
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm import tqdm

# Try importing CV2 for video writing
try:
    import cv2
except ImportError:
    print("Error: OpenCV (cv2) package not found. Please install it with 'pip install opencv-python'")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# --- LMDB Setup ---
try:
    import lmdb
except ImportError:
    lmdb = None
    logger.warning("LMDB not installed. Only pickle files will be readable.")

# Minimal LMDB helper stubs (assuming they were external utils/lmdb_utils.py)
def open_lmdb_env(path: str, readonly: bool, lock: bool, readahead: bool = False, subdir: bool = False, map_size_gb: float = 1.0):
    if lmdb is None: return None
    map_size = int(map_size_gb * (1024**3))
    # Note: readahead=False is important for streaming stability
    return lmdb.open(path, readonly=readonly, lock=lock, readahead=readahead, subdir=subdir, map_size=map_size, meminit=False)

def close_lmdb_env(env):
    if env is not None:
        env.close()
# ------------------

class EpisodeLoader:
    """Handles loading and indexing episodes from LMDB or pickle."""
    def __init__(self, demo_path: str):
        self.demo_path = Path(demo_path)
        self.is_lmdb = lmdb and self.demo_path.suffix == ".lmdb"
        self._lmdb_env = None
        self._episode_keys: List[str] = []
        self._pickle_data: Optional[List[Dict]] = None

        self._load_index()

    def _load_index(self):
        """Determines total number of episodes and sets up necessary handles."""
        if not self.demo_path.exists():
            raise FileNotFoundError(f"Demo path not found: {self.demo_path}")

        if self.is_lmdb:
            self._lmdb_env = open_lmdb_env(str(self.demo_path), readonly=True, lock=False, readahead=False, subdir=False)
            if self._lmdb_env is None:
                raise RuntimeError("LMDB environment could not be opened.")
                
            try:
                with self._lmdb_env.begin() as txn:
                    num_episodes = txn.stat()['entries']
                    # *** FIX: Use 10-digit padding (010d) to match the merging script's output ***
                    self._episode_keys = [f"{i:010d}" for i in range(num_episodes)] 
                logger.info(f"Indexed {num_episodes} episodes from LMDB.")
            except Exception as e:
                close_lmdb_env(self._lmdb_env)
                raise IOError(f"Failed to read LMDB index: {e}")
        else:
            # Pickle mode: load all data into memory
            logger.warning("Loading entire dataset via pickle. May consume significant memory.")
            with open(self.demo_path, "rb") as f:
                self._pickle_data = pickle.load(f)
            self._episode_keys = [str(i) for i in range(len(self._pickle_data))]
            logger.info(f"Loaded {len(self._pickle_data)} episodes from pickle.")

    def __len__(self):
        return len(self._episode_keys)

    def get_episode(self, index: int) -> Dict[str, Any]:
        """Retrieves and deserializes a single episode."""
        if index < 0 or index >= len(self):
            raise IndexError(f"Episode index {index} out of bounds.")
        
        if self.is_lmdb:
            key_str = self._episode_keys[index]
            key = key_str.encode("ascii")
            
            if self._lmdb_env is None:
                 raise RuntimeError("LMDB environment is closed or not initialized.")
                 
            with self._lmdb_env.begin(write=False) as txn:
                blob = txn.get(key)
                if blob is None:
                    # Original error trace pointed here
                    raise KeyError(f"Missing LMDB key {key_str!r} at index {index}")
                try:
                    ep = pickle.loads(blob)
                    return ep
                except pickle.UnpicklingError as e:
                    raise IOError(f"Failed to unpickle key {key_str}: {e}")
        else:
            return self._pickle_data[index]

    def close(self):
        close_lmdb_env(self._lmdb_env)


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

        # Determine video dimensions from the first frame
        H, W, C = obs_list[0]["image_primary"].shape
        
        # Setup VideoWriter
        # Use XVID codec (widely compatible)
        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (W, H))
        if not out.isOpened():
            # If XVID fails, try MP4V (H.264)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (W, H))
            if not out.isOpened():
                 logger.error(f"Could not open VideoWriter at {output_path}. Check codec installation (XVID/MP4V).")
                 return

        episode_id = ep.get('episode_id', 'N/A')
        seed = ep.get('seed', 'N/A')
        is_success = ep.get('success', False)
        
        for t in tqdm(range(len(obs_list)), desc=f"Rendering {episode_id}", leave=False):
            obs = obs_list[t]
            action = actions[t] if t < len(actions) else np.zeros(1) # Handle potential off-by-one or trailing obs
            
            frame = obs["image_primary"].copy()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # --- Annotation ---
            text_color = (0, 255, 0) # Green
            if not is_success:
                 text_color = (0, 0, 255) # Red

            # 1. Episode/Global Metadata
            cv2.putText(frame_bgr, f"ID: {episode_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame_bgr, f"Seed: {seed}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame_bgr, f"Success: {is_success}", (W - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, cv2.LINE_AA)

            # 2. Step Metadata
            cv2.putText(frame_bgr, f"Step: {t}/{len(obs_list) - 1}", (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # 3. State/Action Diagnostics 
            try:
                ee_pos = obs['ee_pose_world'][:3]
                obj_pos = obs['object_pos_world']
                action_norm = np.linalg.norm(action)
                
                cv2.putText(frame_bgr, f"EE Pos: {ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}", 
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame_bgr, f"Obj Pos: {obj_pos[0]:.2f}, {obj_pos[1]:.2f}, {obj_pos[2]:.2f}", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame_bgr, f"Action Norm: {action_norm:.4f}", 
                            (W - 150, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 255), 1, cv2.LINE_AA)
            except KeyError as e:
                cv2.putText(frame_bgr, f"Missing state key: {e}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            
            out.write(frame_bgr)

        out.release()
        logger.info(f"Successfully rendered video to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Render diagnostic videos from expert LMDB/Pickle datasets.")
    parser.add_argument("--demo-path", type=str, required=True, help="Path to the .lmdb or .pkl demo file.")
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
        logger.info(f"Loading index from {args.demo_path}")
        loader = EpisodeLoader(args.demo_path)
        
        total_episodes = len(loader)
        if total_episodes == 0:
            logger.error("Dataset contains zero episodes. Exiting.")
            return

        renderer = ExpertVideoRenderer(fps=args.fps)
        
        # Select indices
        if total_episodes <= args.num_episodes:
            indices_to_render = list(range(total_episodes))
        else:
            indices_to_render = random.sample(range(total_episodes), args.num_episodes)
        
        logger.info(f"Selected {len(indices_to_render)} episodes for rendering: {indices_to_render}")

        for ep_idx in indices_to_render:
            try:
                ep = loader.get_episode(ep_idx)
                
                # Determine output filename
                ep_id_str = str(ep.get('episode_id', f'ep{ep_idx}')).replace("/", "_").replace("\\", "_")
                output_path = output_dir / f"{ep_id_str}_render.avi"
                
                renderer.render_episode(ep, output_path)
            
            except Exception as e:
                logger.error(f"Failed to render episode {ep_idx}: {e}", exc_info=True)
                continue

    except Exception as e:
        logger.critical(f"Fatal error during data loading or indexing: {e}", exc_info=True)
    finally:
        if loader:
            loader.close()
        
    logger.info("Video generation complete.")

if __name__ == "__main__":
    main()


"""
python -m s13 --demo-path data\training\final_merged_dataset.lmdb --output-dir videos\inspection --num-episodes 5 --fps 15 --seed 42

"""