# FILE: scripts/visualize_dataset.py
# (FINAL SOTA Version for SoA Datasets, with Raw Frame Saving)

import argparse
import os
import sys
import random
import pickle
import json
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any
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

# ==============================================================================
# 1. SOTA EPISODE LOADER (Reads the new SoA format)
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

        # --- Load the required JSON Index ---
        index_path = self.demo_path.parent / f"{self.demo_path.stem}_index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Required index file not found: {index_path}")
        
        logger.info(f"Loading SOTA index from {index_path}...")
        with open(index_path, 'r') as f:
            self.index_data = json.load(f)
            
        self.episode_metadata = self.index_data["episodes"]
        logger.info(f"Indexed {len(self.episode_metadata)} episodes.")
        
        # --- Open LMDB Handle ---
        self._lmdb_env = lmdb.open(str(self.demo_path), readonly=True, lock=False, readahead=False, subdir=False, meminit=False)
        if self._lmdb_env is None:
            raise RuntimeError("LMDB environment could not be opened.")

    def __len__(self):
        return len(self.episode_metadata)

    def _get_lmdb_blob(self, key: str) -> bytes:
        with self._lmdb_env.begin(write=False) as txn:
            blob = txn.get(key.encode("ascii"))
            if blob is None:
                raise KeyError(f"Missing LMDB key {key!r}")
            return blob

    @functools.lru_cache(maxsize=32)
    def _get_full_modality_array(self, key: str, compression: str, dtype_str: str, shape_tuple: tuple) -> np.ndarray:
        blob = self._get_lmdb_blob(key)
        dtype = np.dtype(dtype_str)
        shape = shape_tuple

        if compression == "raw":
            return np.frombuffer(blob, dtype=dtype).reshape(shape)
        elif compression in ("jpeg", "png"):
            byte_list = pickle.loads(blob)
            images = [
                cv2.cvtColor(
                    cv2.imdecode(np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB
                ) for b in byte_list
            ]
            return np.stack(images)
        else:
            raise ValueError(f"Unknown compression type: {compression}")

    def get_episode(self, index: int) -> Dict[str, Any]:
        ep_meta = self.episode_metadata[index]
        episode_len = ep_meta["length"]
        modalities = {}
        for name, meta in ep_meta["modalities"].items():
            shape_as_tuple = tuple(meta["shape"])
            modalities[name] = self._get_full_modality_array(
                meta["key"], meta["compression"], meta["dtype"], shape_as_tuple
            )
            
        obs_list = []
        for t in range(episode_len):
            obs_step = {
                "image_primary": modalities["image_primary"][t],
                "image_wrist": modalities["image_wrist"][t],
                "proprio": modalities["proprio"][t],
            }
            obs_list.append(obs_step)

        reconstructed_ep = {
            "episode_id": ep_meta.get("episode_id", f"ep{index}"),
            "seed": ep_meta.get("seed"),
            "success": ep_meta.get("success"),
            "obs_list": obs_list,
            "actions": modalities.get("actions", [])
        }
        return reconstructed_ep

    def close(self):
        if self._lmdb_env:
            self._lmdb_env.close()

# ==============================================================================
# 2. RENDERER (Unchanged, it correctly consumes the reconstructed format)
# ==============================================================================
class ExpertVideoRenderer:
    def __init__(self, fps: int = 10):
        self.fps = fps

    def render_episode(self, ep: Dict[str, Any], output_path: Path):
        obs_list = ep.get("obs_list", [])
        if not obs_list: return

        H, W, _ = obs_list[0]["image_primary"].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
# 3. MAIN SCRIPT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Render diagnostic videos from SOTA expert datasets.")
    parser.add_argument("demo_path", type=str, help="Path to the .lmdb SOTA demo file.")
    parser.add_argument("--output-dir", type=str, default="videos/inspection", help="Directory to save output videos and frames.")
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of random episodes to render.")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second for the output video.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for episode selection.")
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="If set, saves the first, middle, and last frame of each rendered episode as a PNG file."
    )
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = None
    try:
        logger.info(f"Loading SOTA dataset index from {args.demo_path}")
        loader = SoAEpisodeLoader(args.demo_path)
        
        renderer = ExpertVideoRenderer(fps=args.fps)
        indices_to_render = random.sample(range(len(loader)), min(args.num_episodes, len(loader)))
        logger.info(f"Selected {len(indices_to_render)} episodes for rendering: {indices_to_render}")

        for ep_idx in indices_to_render:
            try:
                ep = loader.get_episode(ep_idx)
                ep_id_str = str(ep.get('episode_id', f'ep{ep_idx}'))
                output_path = output_dir / f"{ep_id_str}_render.mp4"
                
                renderer.render_episode(ep, output_path)

                if args.save_frames and ep.get("obs_list"):
                    frame_dir = output_dir / "raw_frames"
                    frame_dir.mkdir(exist_ok=True)
                    obs_list = ep["obs_list"]
                    indices_to_save = [0, len(obs_list) // 2, len(obs_list) - 1]
                    
                    for frame_idx in sorted(list(set(indices_to_save))):
                        frame_rgb = obs_list[frame_idx]["image_primary"]
                        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                        frame_path = frame_dir / f"{ep_id_str}_frame_{frame_idx:03d}.png"
                        cv2.imwrite(str(frame_path), frame_bgr)
                        logger.info(f"Saved raw frame to {frame_path}")
            
            except Exception as e:
                logger.error(f"Failed to process episode {ep_idx}: {e}", exc_info=True)

    except Exception as e:
        logger.critical(f"Fatal error during data loading or indexing: {e}", exc_info=True)
    finally:
        if loader:
            loader.close()
        
    logger.info("Video generation complete.")

if __name__ == "__main__":
    main()


"""
python -m s15 data\training\sota_dataset\expert_training_run_99914b93.lmdb --output-dir videos\inspection --num-episodes 5 --fps 15 --seed 42 --save-frames

"""