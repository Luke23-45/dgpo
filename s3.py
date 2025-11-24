# FILE: scripts/verify_dataset_integrity.py
# (SOTA Robust Validator for Strategist Datasets)

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from collections import defaultdict

import lmdb
import numpy as np
from tqdm import tqdm

# --- Project Imports ---
# Ensure we can import the actual Reader class to test compatibility
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from utils.expert_dataset import ExpertTrajectoryDataset
except ImportError:
    print("CRITICAL: Could not import utils.expert_dataset. Check python path.")
    sys.exit(1)

# --- Logging Setup ---
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

logging.basicConfig(level=logging.INFO, format=f'{Colors.HEADER}%(asctime)s [%(levelname)s] %(message)s{Colors.ENDC}')
log = logging.getLogger("DataValidator")

class DatasetValidator:
    def __init__(self, dataset_path: str):
        self.path = Path(dataset_path)
        self.index_path = self.path.parent / f"{self.path.stem}_index.json"
        self.errors = []
        self.warnings = []
        self.stats = defaultdict(list)
        
        # Specific SOTA keys required for Strategist v9.0
        self.REQUIRED_KEYS = [
            "image_primary", "proprio", "actions", "ee_pose_world", 
            "gt_phase", "gt_gripper", "advantages"
        ]

    def _log_err(self, msg):
        self.errors.append(msg)
        log.error(f"{Colors.FAIL}{msg}{Colors.ENDC}")

    def _log_warn(self, msg):
        self.warnings.append(msg)
        log.warning(f"{Colors.WARNING}{msg}{Colors.ENDC}")

    def check_filesystem(self):
        log.info(f"--- Phase 1: Filesystem & Metadata Check ---")
        
        if not self.path.exists():
            self._log_err(f"LMDB file not found: {self.path}")
            return False
        
        if not self.index_path.exists():
            self._log_err(f"Index JSON not found: {self.index_path}")
            return False
            
        try:
            with open(self.index_path, 'r') as f:
                self.index = json.load(f)
            self.num_episodes = len(self.index['episodes'])
            log.info(f"Metadata loaded. Total Episodes claimed: {Colors.BOLD}{self.num_episodes}{Colors.ENDC}")
        except Exception as e:
            self._log_err(f"Failed to parse JSON index: {e}")
            return False
            
        return True

    def check_lmdb_structure(self):
        """Checks if keys in JSON actually exist in LMDB without decoding data."""
        log.info(f"--- Phase 2: LMDB Key Consistency Scan ---")
        
        env = lmdb.open(str(self.path), readonly=True, lock=False, readahead=False,subdir=False)
        missing_keys = 0
        checked_keys = 0
        
        try:
            with env.begin() as txn:
                cursor = txn.cursor()
                # Create a set of all keys in DB for O(1) lookup
                # Note: For massive DBs (TB+), iteration might be better, but for <50GB this is fine
                log.info("Mapping LMDB keys...")
                db_keys = set(key.decode('ascii') for key, _ in cursor)
                
                for i, ep_meta in enumerate(tqdm(self.index['episodes'], desc="Verifying Keys")):
                    ep_id = ep_meta.get('episode_id', f'ep_{i:06d}')
                    
                    for mod_name, mod_meta in ep_meta['modalities'].items():
                        key = mod_meta['key']
                        checked_keys += 1
                        if key not in db_keys:
                            self._log_err(f"Episode {ep_id}: Missing key in LMDB '{key}' (Modality: {mod_name})")
                            missing_keys += 1
        finally:
            env.close()
            
        if missing_keys == 0:
            log.info(f"{Colors.OKGREEN}Structure Clean. Verified {checked_keys} keys.{Colors.ENDC}")
            return True
        return False

    def deep_content_inspection(self, sample_ratio=1.0):
        """
        Uses the actual Reader class to load data, decode images, and check numerical validity.
        """
        log.info(f"--- Phase 3: Deep Content Inspection (Checking {sample_ratio*100:.0f}% of data) ---")
        
        try:
            # Initialize SOTA Reader
            ds = ExpertTrajectoryDataset(
                demo_path=str(self.path),
                observation_horizon=1,
                action_horizon=1
            )
        except Exception as e:
            self._log_err(f"CRITICAL: Reader class failed to initialize: {e}")
            traceback.print_exc()
            return False

        total_frames = 0
        nan_detected = False
        black_images = 0
        
        # Sampling logic
        indices = np.arange(len(ds))
        if sample_ratio < 1.0:
            np.random.shuffle(indices)
            indices = indices[:int(len(indices) * sample_ratio)]
            
        log.info(f"Inspecting {len(indices)} frames...")

        # Stats containers
        phases_found = set()
        adv_sum = 0.0
        adv_count = 0
        
        pbar = tqdm(indices, desc="Deep Scan")
        for idx in pbar:
            try:
                # This calls __getitem__, triggering decompression and slicing
                # It returns (obs_chunk, action_chunk)
                sample = ds[idx]
                
                if sample is None:
                    self._log_err(f"Index {idx} returned None from dataset!")
                    continue
                    
                obs, action = sample
                
                # 1. Check NaNs in Action
                if np.isnan(action).any():
                    self._log_err(f"NaN found in ACTIONS at global index {idx}")
                    nan_detected = True
                
                # 2. Check Proprioception
                proprio = obs.get('proprio')
                if proprio is not None:
                    if np.isnan(proprio).any():
                        self._log_err(f"NaN found in PROPRIO at global index {idx}")
                        nan_detected = True
                    # Physics sanity check (e.g., qpos shouldn't be 1000.0)
                    if np.max(np.abs(proprio)) > 100.0:
                        self._log_warn(f"Suspiciously high value in proprio at {idx}: {np.max(np.abs(proprio))}")

                # 3. Check Images (Visual Sanity)
                img = obs.get('image_primary')
                if img is not None:
                    if img.shape != (1, 256, 256, 3): # Assuming horizon=1
                        self._log_err(f"Unexpected image shape at {idx}: {img.shape}")
                    
                    # Check for "Black Screen Bug" (all zeros)
                    if np.mean(img) < 1.0:
                        black_images += 1
                
                # 4. Recover Episode Metadata to check Aux labels
                # Map global idx back to episode to check advantage stats
                ep_idx, t = ds.get_episode_and_timestep(idx)
                ep_meta = ds.episode_metadata[ep_idx]
                
                # Check required SOTA keys exist in meta
                for req in self.REQUIRED_KEYS:
                    if req not in ep_meta['modalities']:
                        # Only warn once per key per run to avoid spam
                        if f"missing_{req}" not in self.stats:
                            self._log_err(f"Episode {ep_idx} missing required modality '{req}'")
                            self.stats[f"missing_{req}"] = True

                # 5. Check Advantage Statistics (if available)
                # Access raw array directly via reader helper to avoid overhead
                if "advantages" in ep_meta['modalities']:
                    adv_mod = ep_meta['modalities']['advantages']
                    adv_val = ds._get_full_modality_array(
                        adv_mod['key'], adv_mod['compression'], adv_mod['dtype'], tuple(adv_mod['shape'])
                    )[t]
                    adv_sum += adv_val
                    adv_count += 1
                
                # 6. Check Phase
                if "gt_phase" in ep_meta['modalities']:
                    ph_mod = ep_meta['modalities']['gt_phase']
                    ph_val = ds._get_full_modality_array(
                        ph_mod['key'], ph_mod['compression'], ph_mod['dtype'], tuple(ph_mod['shape'])
                    )[t]
                    phases_found.add(int(ph_val))

                total_frames += 1

            except Exception as e:
                self._log_err(f"Crash reading index {idx}: {e}")
                if total_frames < 5: traceback.print_exc() # Print first few stacks
                
        # --- Final Report ---
        print("\n" + "="*60)
        print(f"{Colors.BOLD}DATASET INTEGRITY REPORT{Colors.ENDC}")
        print("="*60)
        
        status = f"{Colors.OKGREEN}PASS{Colors.ENDC}"
        
        if self.errors:
            status = f"{Colors.FAIL}FAIL{Colors.ENDC}"
            print(f"Errors Found: {len(self.errors)}")
            for e in self.errors[:10]: print(f" - {e}")
            if len(self.errors) > 10: print(" ... (and more)")
        else:
            print(f"Errors Found: 0")

        if self.warnings:
            print(f"Warnings: {len(self.warnings)}")
            for w in self.warnings[:5]: print(f" - {w}")

        print("-" * 30)
        print(f"Total Episodes: {self.num_episodes}")
        print(f"Total Frames Scanned: {total_frames}")
        print(f"Nan/Inf Detected: {nan_detected}")
        
        if black_images > 0:
            print(f"{Colors.FAIL}Black (Zero) Images: {black_images} ({black_images/total_frames*100:.2f}%){Colors.ENDC}")
        else:
            print(f"Black Images: 0 (OK)")

        print(f"Phases Found: {sorted(list(phases_found))} (Should be [0, 1, 2, 3, 4])")
        
        if adv_count > 0:
            avg_adv = adv_sum / adv_count
            print(f"Mean Advantage: {avg_adv:.4f} (Should be close to 0.0)")
        else:
            print(f"{Colors.WARNING}No Advantage values found! (Run advantage_calculator.py){Colors.ENDC}")

        print("="*60)
        return len(self.errors) == 0

def main():
    parser = argparse.ArgumentParser(description="SOTA Dataset Integrity Validator")
    parser.add_argument("--dataset", type=str, required=True, help="Path to .lmdb file")
    parser.add_argument("--samples", type=str, default="all", help="'all' or number of samples to check")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    
    validator = DatasetValidator(dataset_path)
    
    # 1. Basic Checks
    if not validator.check_filesystem():
        sys.exit(1)
        
    if not validator.check_lmdb_structure():
        sys.exit(1)
        
    # 2. Content Checks
    # Determine sample ratio
    if args.samples.lower() == "all":
        ratio = 1.0
    else:
        # Load index to get total count for ratio calc
        with open(validator.index_path, 'r') as f:
            idx_data = json.load(f)
            total_eps = len(idx_data['episodes'])
            # Estimate total frames (approx 200 per ep)
            est_frames = total_eps * 200 
            ratio = min(1.0, int(args.samples) / est_frames)

    valid = validator.deep_content_inspection(sample_ratio=ratio)
    
    if not valid:
        sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main()

#python -m s3 --dataset "C:\Users\Hellx\Documents\Programming\python\Project\redhot\data\final_training_set\training_set.lmdb" --samples all