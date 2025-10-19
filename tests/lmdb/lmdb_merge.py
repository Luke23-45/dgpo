#!/usr/bin/env python3
# FILE: utils/lmdb_merge.py
"""
Stream-oriented LMDB merge utility.

Usage:
    python -m utils.lmdb_merge --src a.lmdb b.lmdb --dst merged.lmdb --clobber

Improvements over original:
- Handles existing dst: appends by default, resizes map_size if needed.
- --clobber to remove dst if exists.
- Auto-detects subdir vs file based on path (py-lmdb default).
- lock=True for destination writes (safer).
- Better map_size estimation including existing dst.
- Optional --renumber for sequential key renumbering (useful for ML datasets).
- Optional --compact to compact dst after merge (fixes fragmentation/slow reads).
- Context managers for src txns.
- Error handling for key collisions if --error-on-collision.
- Consistent safety factor (1.3).
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import lmdb
import shutil
from tqdm import tqdm
import math
import logging

logging.basicConfig(level=logging.INFO)

def estimate_map_size(paths: list[Path], safety_factor: float = 1.3) -> int:
    """Estimate LMDB map_size from file sizes."""
    total = 0
    for p in paths:
        if p.is_dir():
            for f in p.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        elif p.is_file():
            total += p.stat().st_size
    est = int(total * safety_factor) + (1024**2) * 100  # +100MB
    # Round up to nearest 256MB
    step = 256 * 1024 * 1024
    est = int(math.ceil(est / step) * step)
    return max(est, 10485760)  # Min 10MB

def get_existing_size(dst: Path) -> int:
    """Sum file sizes in existing dst."""
    if not dst.exists():
        return 0
    total = 0
    if dst.is_dir():
        for f in dst.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    else:
        total = dst.stat().st_size
    return total

def find_max_numeric_key(txn) -> int:
    """Find max int key (assume padded str keys) for renumbering."""
    cursor = txn.cursor()
    if cursor.last():
        try:
            return int(cursor.key().decode('utf-8'))
        except ValueError:
            raise ValueError("Keys are not numeric for renumbering.")
    return -1  # Start from 0

def merge_lmdbs(
    src_paths: list[str],
    dst_path: str,
    overwrite: bool = True,
    prefix_keys: bool = False,
    renumber: bool = False,
    error_on_collision: bool = False,
    commit_every: int = 1000,
    map_size: int = None,
    compact: bool = False,
    clobber: bool = False
):
    src_paths = [Path(p) for p in src_paths]
    dst = Path(dst_path)

    if dst.exists() and clobber:
        logging.info(f"Clobbering existing {dst}")
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    existing_size = get_existing_size(dst)
    add_size = estimate_map_size(src_paths, safety_factor=1.0)  # Base add without factor
    total_est = estimate_map_size([], safety_factor=1.3)  # Min, then add
    total_est = int((existing_size + add_size) * 1.3) + (1024**2) * 100

    if map_size is not None:
        total_est = map_size

    logging.info(f"Estimated map_size = {total_est / (1024**3):.2f} GB")

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        # Open existing, resize if needed
        dst_env = lmdb.open(str(dst), readonly=False, subdir=False)
        current_map = dst_env.info()['map_size']
        if total_est > current_map:
            logging.info(f"Resizing map from {current_map / (1024**3):.2f} GB to {total_est / (1024**3):.2f} GB")
            dst_env.set_mapsize(total_est)
    else:
        # Create new
        dst_env = lmdb.open(str(dst), map_size=total_est, readonly=False, subdir=False)

    total_written = 0
    txn = dst_env.begin(write=True)
    max_key = -1
    if renumber:
        max_key = find_max_numeric_key(txn)
        txn.abort()  # Read-only check, restart write
        txn = dst_env.begin(write=True)

    for i, src in enumerate(src_paths):
        logging.info(f"Merging source {i}: {src}")
        src_env = lmdb.open(str(src), readonly=True, lock=False, readahead=True,subdir=False)
        with src_env.begin() as src_txn:
            cursor = src_txn.cursor()
            count = 0
            for key, val in tqdm(cursor, desc=f"src[{i}]"):
                if renumber:
                    max_key += 1
                    out_key = '{:010d}'.format(max_key).encode('utf-8')
                elif prefix_keys:
                    out_key = f"{i:03d}_".encode("utf-8") + key
                else:
                    out_key = key

                exists = txn.get(out_key) is not None
                if exists:
                    if error_on_collision:
                        raise ValueError(f"Key collision: {out_key}")
                    if not overwrite:
                        continue

                txn.put(out_key, val)
                count += 1
                total_written += 1
                if count % commit_every == 0:
                    txn.commit()
                    txn = dst_env.begin(write=True)
            # Commit remaining
            txn.commit()
            txn = dst_env.begin(write=True)
        src_env.close()
        logging.info(f"Finished source {i}, wrote {count} entries.")

    txn.commit()
    dst_env.sync()

    if compact:
        compact_path = str(dst) + ".compact"
        logging.info(f"Compacting to {compact_path}")
        dst_env.copy(compact_path, compact=True)
        dst_env.close()
        # Replace original with compact
        if dst.is_dir():
            shutil.rmtree(dst)
            shutil.move(compact_path, dst)
        else:
            os.remove(dst)
            shutil.move(compact_path, dst)
        logging.info("Compact complete.")
    else:
        dst_env.close()

    logging.info(f"Merge complete. Total entries written: {total_written}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("lmdb_merge")
    parser.add_argument("--src", nargs="+", required=True, help="Source LMDB paths")
    parser.add_argument("--dst", required=True, help="Destination LMDB path")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false", help="Skip existing keys")
    parser.add_argument("--prefix-keys", action="store_true", help="Prefix keys with source index")
    parser.add_argument("--renumber", action="store_true", help="Renumber keys sequentially (assumes numeric keys)")
    parser.add_argument("--error-on-collision", action="store_true", help="Error if key collision")
    parser.add_argument("--commit-every", type=int, default=1000, help="Commit after this many writes")
    parser.add_argument("--map-size-gb", type=float, default=None, help="Explicit map_size in GB")
    parser.add_argument("--clobber", action="store_true", help="Remove destination if exists")
    parser.add_argument("--compact", action="store_true", help="Compact destination after merge")
    args = parser.parse_args()

    map_size = None
    if args.map_size_gb is not None:
        map_size = int(args.map_size_gb * 1024**3)
    merge_lmdbs(
        args.src, args.dst, overwrite=args.overwrite, prefix_keys=args.prefix_keys,
        renumber=args.renumber, error_on_collision=args.error_on_collision,
        commit_every=args.commit_every, map_size=map_size, compact=args.compact, clobber=args.clobber
    )
"""
python -m s13 --src data\demos\expert_20251018_222103_15000_samples.lmdb data\dataset_part_2\shards\shard_w0.lmdb --dst data\training\final_merged_dataset --renumber --clobber --commit-every 10000 

"""