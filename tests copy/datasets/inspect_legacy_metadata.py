# FILE: scripts/inspect_legacy_metadata.py

import argparse
import pickle
import logging
from pathlib import Path
from tqdm import tqdm
import lmdb
from collections import Counter

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def inspect_legacy_db(db_path: str):
    """
    Scans a legacy (List-of-Structs) LMDB and reports on the presence
    and values of specific metadata keys like 'success' and 'seed'.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        logger.critical(f"Database not found at: {db_path}")
        return

    logger.info(f"--- 🔍 Inspecting Legacy DB Metadata: {db_path} ---")

    try:
        env = lmdb.open(str(db_path), readonly=True, lock=False, readahead=False, subdir=False)
    except lmdb.Error as e:
        logger.critical(f"Failed to open LMDB environment. Error: {e}")
        return

    success_values = Counter()
    has_seed_count = 0
    missing_success_key = 0
    missing_seed_key = 0
    total_episodes = 0

    with env.begin() as txn:
        total_entries = txn.stat()['entries']
        logger.info(f"Found {total_entries} total entries (episodes) to scan.")
        
        cursor = txn.cursor()
        
        for key, blob in tqdm(cursor, total=total_entries, desc="Scanning Episodes"):
            total_episodes += 1
            try:
                ep_dict = pickle.loads(blob)
                
                # --- Check for 'success' key ---
                if 'success' in ep_dict:
                    success_values[ep_dict['success']] += 1
                else:
                    missing_success_key += 1

                # --- Check for 'seed' key ---
                if 'seed' in ep_dict:
                    has_seed_count += 1
                else:
                    missing_seed_key += 1

            except pickle.UnpicklingError:
                logger.warning(f"Could not unpickle data for key: {key.decode()}, skipping.")
                continue
    
    env.close()

    # --- Print Final Report ---
    print("\n" + "="*50)
    logger.info("--- 🕵️‍♂️ Legacy Metadata Audit Report ---")
    print("="*50)
    logger.info(f"Total Episodes Scanned: {total_episodes}")
    
    print("-" * 25)
    logger.info("Analysis of 'success' flag:")
    if missing_success_key > 0:
        logger.warning(f"  - Episodes MISSING the 'success' key: {missing_success_key}")
    for value, count in success_values.items():
        logger.info(f"  - Episodes with 'success' == {value}: {count}")
    
    print("-" * 25)
    logger.info("Analysis of 'seed' key:")
    logger.info(f"  - Episodes containing a 'seed' key: {has_seed_count}")
    if missing_seed_key > 0:
        logger.warning(f"  - Episodes MISSING the 'seed' key: {missing_seed_key}")
    
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspects metadata keys in a legacy LMDB dataset.")
    parser.add_argument("db_path", type=str, help="Path to the legacy .lmdb file.")
    args = parser.parse_args()
    
    inspect_legacy_db(args.db_path)

"""
python -m s13 --demo-path data\training\final_merged_dataset.lmdb --output-dir videos\inspection --num-episodes 5 --fps 15 --seed 42

"""
"""
python -m s14 --demo-path data\training\final_merged_dataset.lmdb  --output-dir videos/inspection --num-episodes 5 --fps 15 --seed 42


tests.datasets.visualize_dataset

"""