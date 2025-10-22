# FILE: scripts/split_sota_dataset.py

import argparse
import json
import math
import lmdb
import pickle
import shutil
from pathlib import Path
from tqdm import tqdm

def open_lmdb_env(path, map_size_gb=4.0, readonly=False):
    """
    Opens LMDB safely across multiple environments on Windows.
    """
    flags = {
        'map_size': int(map_size_gb * 1024**3),
        'subdir': False,
        'lock': False,
        'readahead': False,
        'meminit': False,
        'readonly': readonly
    }

    # Prevent Windows mmap reuse issues:
    # readonly=True ensures LMDB opens without write locking.
    # writemap=False prevents persistent mmap memory.
    return lmdb.open(str(path), writemap=False, **flags)



def split_dataset(src_path, output_dir, episodes_per_shard=50, map_size_gb=4.0):
    src_path = Path(src_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load the master index ---
    index_path = src_path.parent / f"{src_path.stem}_index.json"
    with open(index_path, "r") as f:
        master_index = json.load(f)

    episodes = master_index["episodes"]
    num_episodes = len(episodes)
    num_shards = math.ceil(num_episodes / episodes_per_shard)
    print(f"[INFO] Splitting {num_episodes} episodes into {num_shards} shards...")

    # --- Open source LMDB ---
    src_env = open_lmdb_env(src_path, map_size_gb, readonly=True)

    total_written = 0

    for shard_idx in range(num_shards):
        start = shard_idx * episodes_per_shard
        end = min(start + episodes_per_shard, num_episodes)
        shard_eps = episodes[start:end]

        shard_dir = output_dir / f"shard_{shard_idx:03d}"
        shard_dir.mkdir(exist_ok=True)

        shard_lmdb_path = shard_dir / f"{src_path.stem}_part{shard_idx:03d}.lmdb"
        shard_index_path = shard_dir / f"{src_path.stem}_part{shard_idx:03d}_index.json"

        dst_env = open_lmdb_env(shard_lmdb_path, map_size_gb)

        shard_index = {"episodes": [], "total_episodes": len(shard_eps)}

        with dst_env.begin(write=True) as dst_txn, src_env.begin(write=False) as src_txn:
            for ep_meta in tqdm(shard_eps, desc=f"Shard {shard_idx:03d}", leave=False):
                ep_entry = ep_meta.copy()
                modalities = ep_entry["modalities"]

                # copy each modality key blob
                for mod_name, mod_meta in modalities.items():
                    key = mod_meta["key"].encode("ascii")
                    blob = src_txn.get(key)
                    if blob is None:
                        raise KeyError(f"Missing key {key.decode()} in source LMDB.")
                    dst_txn.put(key, blob)

                shard_index["episodes"].append(ep_entry)
                total_written += 1

        with open(shard_index_path, "w") as f:
            json.dump(shard_index, f, indent=2)

        dst_env.close()
        print(f"[INFO] Wrote {len(shard_eps)} episodes to {shard_lmdb_path}")

    src_env.close()
    print(f"[SUCCESS] Split complete. Total {total_written} episodes processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a large SoA LMDB dataset into smaller shards.")
    parser.add_argument("src_path", type=str, help="Path to the source SOTA .lmdb dataset")
    parser.add_argument("output_dir", type=str, help="Directory to store output shards")
    parser.add_argument("--episodes-per-shard", type=int, default=50, help="Number of episodes per shard")
    parser.add_argument("--map-size-gb", type=float, default=5.0, help="LMDB map size (GB) for writing")
    args = parser.parse_args()

    split_dataset(args.src_path, args.output_dir, args.episodes_per_shard, args.map_size_gb)


"""
python -m s15 data\training\sota_dataset\expert_training_run_99914b93.lmdb data\split --episodes-per-shard 100

"""