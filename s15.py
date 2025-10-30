# FILE: scripts/enhance_sota_dataset.py
# SOTA "In-Place" Dataset Enhancement Script

import argparse
import logging
import json
from pathlib import Path
import numpy as np
import pickle
import cv2
from tqdm import tqdm
import shutil
import os
# --- Project Imports ---
import sys
try:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from utils.lmdb_utils import open_lmdb_env
except ImportError as e:
    print(f"Error importing project modules: {e}.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("enhance_sota_dataset")


def main(args):
    dataset_path = Path(args.dataset_path).resolve()
    index_path = dataset_path.parent / f"{dataset_path.stem}_index.json"

    # --- SOTA Pre-flight Checks ---
    if not dataset_path.is_file():
        logger.error(f"FATAL: Input LMDB file not found: {dataset_path}")
        return
    if not index_path.is_file():
        logger.error(f"FATAL: Corresponding index file not found: {index_path}")
        logger.error("This script enhances an existing SOTA-formatted dataset. Ensure the _index.json file is present.")
        return

    logger.info(f"Starting enhancement process for dataset: {dataset_path}")
    logger.info(f"Using index file: {index_path}")
    
    # --- SOTA Backup Strategy ---
    backup_path = index_path.with_suffix(".json.bak")
    if not backup_path.exists():
        logger.info(f"Creating a backup of the original index file at: {backup_path}")
        shutil.copy(index_path, backup_path)
    else:
        logger.info(f"Backup file already exists at: {backup_path}. No new backup will be created.")

    # --- Main Enhancement Logic ---
    try:
        with open(index_path, 'r') as f:
            index_data = json.load(f)

        env = open_lmdb_env(str(dataset_path), readonly=False, lock=True, map_size_gb=19)

        with env.begin(write=True) as txn:
            episodes_to_enhance = [ep for ep in index_data["episodes"] if "goal_image_primary" not in ep["modalities"]]
            
            if not episodes_to_enhance:
                logger.info("All episodes in the index already appear to be enhanced. No action taken.")
                env.close()
                return

            logger.info(f"Found {len(episodes_to_enhance)} episodes to enhance...")
            for ep_meta in tqdm(episodes_to_enhance, desc="Enhancing Episodes"):
                # ... (The core logic inside the loop is correct and remains unchanged) ...
                ep_id_str = ep_meta["episode_id"]
                try:
                    img_meta = ep_meta["modalities"]["image_primary"]
                    img_key = img_meta["key"].encode('ascii')
                    ep_len = ep_meta["length"]
                except KeyError:
                    logger.warning(f"Episode {ep_id_str} missing 'image_primary' metadata. Cannot enhance.")
                    continue

                img_blob = txn.get(img_key)
                if not img_blob:
                    logger.warning(f"Image data for key {img_key.decode()} not found. Skipping.")
                    continue
                
                byte_list = pickle.loads(img_blob)
                if ep_len == 0 or len(byte_list) < ep_len:
                    logger.warning(f"Episode {ep_id_str} has inconsistent length. Skipping.")
                    continue

                last_frame_bytes = byte_list[ep_len - 1]
                img_bgr = cv2.imdecode(np.frombuffer(last_frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    logger.warning(f"Failed to decode last image for episode {ep_id_str}. Skipping.")
                    continue
                goal_image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                goal_key_str = f"{ep_id_str}_goal_image_primary"
                goal_key = goal_key_str.encode('ascii')
                
                _, goal_bytes_encoded = cv2.imencode(".jpg", cv2.cvtColor(goal_image_rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
                goal_blob_to_write = pickle.dumps([goal_bytes_encoded.tobytes()])
                txn.put(goal_key, goal_blob_to_write)

                ep_meta["modalities"]["goal_image_primary"] = {
                    "key": goal_key_str,
                    "compression": "jpeg",
                    "dtype": str(goal_image_rgb.dtype),
                    "shape": [1, *goal_image_rgb.shape]
                }
        
        # --- SOTA Atomic Write Strategy ---
        # Write the fully modified index to a temporary file first.
        temp_index_path = index_path.with_suffix(".json.tmp")
        with open(temp_index_path, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        # Atomically replace the old index with the new one.
        os.replace(temp_index_path, index_path)

        logger.info("Enhancement complete! The original index file has been atomically updated.")
        logger.info(f"A backup of the original index is available at: {backup_path}")

    except Exception as e:
        logger.exception("An unhandled error occurred during enhancement.")
    finally:
        if 'env' in locals() and env:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhance an SOTA-formatted dataset by adding a goal image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("dataset_path", type=str, help="Path to the SOTA .lmdb file (e.g., 'data/expert_data.lmdb').")
    parser.add_argument("--jpeg-quality", type=int, default=100, help="JPEG quality for the new goal image.")
    args = parser.parse_args()
    main(args)



"""
python -m s15 "/content/drive/MyDrive/pda/data/validation/sota_dataset/expert_validation_run_99914b93.lmdb"
"""