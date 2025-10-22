# FILE: scripts/validate_transformation.py
# (A Data Auditor to Verify the Legacy-to-SOTA Dataset Transformation)

import argparse
import pickle
import logging
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

import lmdb
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import json
# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Loader for OLD "List of Structs" (LoS) Format ---
class LegacyEpisodeLoader:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.env = lmdb.open(str(db_path), readonly=True, lock=False, readahead=False, subdir=False)
        with self.env.begin() as txn:
            # Assumes keys are sequential, zero-padded strings
            self.keys = sorted([key for key, _ in txn.cursor()])
        logger.info(f"[Legacy Loader] Indexed {len(self.keys)} episodes.")

    def __len__(self):
        return len(self.keys)

    def get_episode(self, index: int) -> dict:
        key = self.keys[index]
        with self.env.begin() as txn:
            blob = txn.get(key)
        if blob is None:
            raise KeyError(f"Key {key.decode()} not found in legacy DB.")
        return pickle.loads(blob)

    def close(self):
        self.env.close()

# --- Loader for NEW "Struct of Arrays" (SoA) Format ---
# This is a simplified version of the loader from visualize_dataset.py
# We are borrowing its tested logic.
class SoAEpisodeLoader:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.env = lmdb.open(str(db_path), readonly=True, lock=False, readahead=False, subdir=False)
        
        index_path = self.db_path.parent / f"{self.db_path.stem}_index.json"
        with open(index_path, 'r') as f:
            self.index_data = json.load(f)
        self.episode_metadata = self.index_data["episodes"]
        logger.info(f"[SOTA Loader] Indexed {len(self.episode_metadata)} episodes from JSON.")

    def __len__(self):
        return len(self.episode_metadata)

    def _get_modality(self, key, compression, dtype, shape) -> np.ndarray:
        with self.env.begin() as txn:
            blob = txn.get(key.encode('ascii'))
        if blob is None: raise KeyError(f"Key {key} not found.")

        if compression == 'raw':
            return np.frombuffer(blob, dtype=np.dtype(dtype)).reshape(shape)
        elif compression in ('jpeg', 'png'):
            byte_list = pickle.loads(blob)
            images = [cv2.cvtColor(cv2.imdecode(np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for b in byte_list]
            return np.stack(images)
        raise ValueError(f"Unknown compression {compression}")

    def get_episode(self, index: int) -> dict:
        ep_meta = self.episode_metadata[index]
        modalities = {}
        for name, meta in ep_meta["modalities"].items():
            modalities[name] = self._get_modality(meta['key'], meta['compression'], meta['dtype'], meta['shape'])
        
        obs_list = []
        for t in range(ep_meta["length"]):
            obs_list.append({
                "image_primary": modalities["image_primary"][t],
                "image_wrist": modalities["image_wrist"][t],
                "proprio": modalities["proprio"][t],
            })
        
        return {
            "episode_id": ep_meta.get("episode_id"),
            "seed": ep_meta.get("seed"),
            "success": ep_meta.get("success"),
            "obs_list": obs_list,
            "actions": modalities.get("actions")
        }

    def close(self):
        self.env.close()


def compare_episodes(legacy_ep: dict, sota_ep: dict, psnr_thresh: float, ssim_thresh: float) -> bool:
    """Compares two reconstructed episodes and returns True if they match."""
    is_match = True
    
    # Check 1: Metadata
    if len(legacy_ep['actions']) != len(sota_ep['actions']):
        logger.error(f"  ❌ Mismatch: Action length ({len(legacy_ep['actions'])} vs {len(sota_ep['actions'])})")
        is_match = False
    if legacy_ep.get('success') != sota_ep.get('success'):
        logger.error(f"  ❌ Mismatch: Success flag ({legacy_ep.get('success')} vs {sota_ep.get('success')})")
        is_match = False

    # Check 2: Numerical Data
    if not np.allclose(legacy_ep['actions'], sota_ep['actions'], atol=1e-6):
        err = np.max(np.abs(np.array(legacy_ep['actions']) - np.array(sota_ep['actions'])))
        logger.warning(f"  ⚠️ Mismatch: Actions do not match. Max error: {err:.8f}")
        is_match = False # This could be a warning or an error depending on strictness

    legacy_proprio = np.stack([o['proprio'] for o in legacy_ep['obs_list']])
    sota_proprio = np.stack([o['proprio'] for o in sota_ep['obs_list']])
    
    if not np.allclose(legacy_proprio, sota_proprio, atol=1e-6):
        err = np.max(np.abs(legacy_proprio - sota_proprio))
        logger.warning(f"  ⚠️ Mismatch: Proprioception data does not match. Max error: {err:.8f}")
        is_match = False

    # Check 3: Image Data
    for t in range(len(legacy_ep['obs_list'])):
        img_legacy = legacy_ep['obs_list'][t]['image_primary']
        img_sota = sota_ep['obs_list'][t]['image_primary']
        
        psnr_val = psnr(img_legacy, img_sota, data_range=255)
        ssim_val = ssim(img_legacy, img_sota, channel_axis=-1, data_range=255)
        
        if psnr_val < psnr_thresh or ssim_val < ssim_thresh:
            logger.warning(f"  ⚠️ Image Similarity Low at t={t}: PSNR={psnr_val:.2f} (thresh>{psnr_thresh}), SSIM={ssim_val:.3f} (thresh>{ssim_thresh})")
            is_match = False
    
    return is_match

def main():
    parser = argparse.ArgumentParser(description="Audits the transformation from a legacy to a SOTA dataset.")
    parser.add_argument("legacy_path", type=str, help="Path to the legacy .lmdb file.")
    parser.add_argument("sota_path", type=str, help="Path to the new SOTA .lmdb file.")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of random episodes to check.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random sampling.")
    parser.add_argument("--psnr-thresh", type=float, default=35.0, help="Minimum acceptable PSNR for images.")
    parser.add_argument("--ssim-thresh", type=float, default=0.95, help="Minimum acceptable SSIM for images.")
    args = parser.parse_args()

    random.seed(args.seed)

    logger.info("--- Initializing Data Loaders ---")
    legacy_loader = LegacyEpisodeLoader(args.legacy_path)
    sota_loader = SoAEpisodeLoader(args.sota_path)

    # --- Completeness Check ---
    if len(legacy_loader) != len(sota_loader):
        logger.critical(f"❌ FATAL: Episode count mismatch! Legacy has {len(legacy_loader)}, SOTA has {len(sota_loader)}.")
        return

    logger.info(f"✅ Completeness PASSED: Both datasets have {len(legacy_loader)} episodes.")

    # --- Integrity Check on a Random Sample ---
    num_to_check = min(args.num_episodes, len(legacy_loader))
    indices_to_check = random.sample(range(len(legacy_loader)), num_to_check)
    
    logger.info(f"\n--- Checking Integrity of {num_to_check} Random Episodes ---")
    
    mismatched_episodes = 0
    for ep_idx in tqdm(indices_to_check, desc="Auditing Episodes"):
        legacy_ep = legacy_loader.get_episode(ep_idx)
        sota_ep = sota_loader.get_episode(ep_idx)
        
        logger.debug(f"Comparing episode index {ep_idx}...")
        if not compare_episodes(legacy_ep, sota_ep, args.psnr_thresh, args.ssim_thresh):
            logger.error(f"  ❌ Mismatch found in episode index {ep_idx}!")
            mismatched_episodes += 1
            
    legacy_loader.close()
    sota_loader.close()

    # --- Final Report ---
    logger.info("\n--- Audit Complete ---")
    if mismatched_episodes == 0:
        logger.info(f"✅ SUCCESS: All {num_to_check} sampled episodes were a perfect match.")
        logger.info("The transformation appears to be successful and high-fidelity.")
    else:
        logger.error(f"❌ FAILURE: Found mismatches in {mismatched_episodes} out of {num_to_check} sampled episodes.")
        logger.error("The transformation introduced errors. Please review the warnings above.")

if __name__ == "__main__":
    main()

"""
python -m s12 data\validation\expert_20251019_011015_6000_samples.lmdb data\validation\sota_dataset\expert_validation_run_99914b93.lmdb
"""