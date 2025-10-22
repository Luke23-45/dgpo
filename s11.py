import argparse
import json
import logging
from pathlib import Path
import sys

# --- LMDB Imports and Helpers (for legacy fallback) ---
try:
    import lmdb
except ImportError:
    lmdb = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def open_lmdb_env(path: str, readonly: bool = True, lock: bool = False):
    if not lmdb:
        raise ImportError("LMDB package not found. Please install with 'pip install lmdb'")
    # Use a large map size for safety, but it won't be used if readonly.
    map_size = int(1 * (1024**3)) # 1 GB
    return lmdb.open(path, readonly=readonly, lock=lock, readahead=False, map_size=map_size, meminit=False)

def close_lmdb_env(env):
    if env is not None:
        env.close()

def count_episodes_sota(lmdb_path: Path) -> int:
    """Counts episodes by reading the SOTA JSON index file."""
    
    # --- START OF FIX ---
    # Correctly construct the index path name
    # e.g., 'file.lmdb' -> stem is 'file' -> 'file_index.json'
    stem = lmdb_path.stem
    index_path = lmdb_path.parent / f"{stem}_index.json"
    # --- END OF FIX ---

    if not index_path.exists():
        # Raise a more specific error for clarity
        raise FileNotFoundError(f"SOTA index file not found at expected path: {index_path}")
        
    with open(index_path, 'r') as f:
        index_data = json.load(f)
        
    return len(index_data.get("episodes", []))

def count_episodes_legacy(lmdb_path: Path) -> int:
    """Counts episodes by reading the number of keys in a legacy LMDB file."""
    env = None
    try:
        env = open_lmdb_env(str(lmdb_path))
        with env.begin(write=False) as txn:
            # txn.stat() is the most efficient way to get the number of entries.
            stats = txn.stat()
            return stats['entries']
    finally:
        if env:
            close_lmdb_env(env)

def main():
    parser = argparse.ArgumentParser(
        description="Count the total number of episodes in a legacy or SOTA LMDB dataset."
    )
    parser.add_argument(
        "lmdb_path",
        type=str,
        help="Path to the .lmdb file."
    )
    args = parser.parse_args()

    db_path = Path(args.lmdb_path)

    if not db_path.is_file():
        logger.error(f"Error: File not found at '{db_path}'")
        sys.exit(1)

    try:
        # --- Attempt 1: Try the fast SOTA method first ---
        count = count_episodes_sota(db_path)
        logger.info(f"Detected SOTA dataset with pre-computed index.")
        print(f"\nTotal episodes: {count}\n")

    except FileNotFoundError:
        # --- Attempt 2: Fall back to the legacy method ---
        logger.info("SOTA index file not found. Falling back to legacy scan method...")
        try:
            count = count_episodes_legacy(db_path)
            logger.info("Scan complete.")
            print(f"\nTotal episodes: {count}\n")
        except Exception as e:
            logger.error(f"Failed to scan LMDB file as a legacy dataset: {e}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

"""

python -m s11 data/training/final_merged_dataset.lmdb
"""