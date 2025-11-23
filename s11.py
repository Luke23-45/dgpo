# FILE: scripts/benchmark_storage.py

import os
import shutil
import logging
import sys
import time
from pathlib import Path
import numpy as np

# --- Project Imports ---
# Ensure project root is in path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.expert_dataset import ExpertDataset, ExpertDatasetWriter
from utils.scripted_expert import ExpertConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("Benchmark")

def get_dir_size(path_obj):
    """Calculates total size of a directory in Bytes."""
    total = 0
    for p in path_obj.rglob('*'):
        if p.is_file():
            total += p.stat().st_size
    return total

def main():
    # 1. Configuration
    TEST_EPISODES = 5  # Small number to test
    TEMP_DIR = Path("temp_storage_benchmark")
    
    # Clean up previous runs
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir()

    log.info(f"--- Starting Storage Benchmark ({TEST_EPISODES} Episodes) ---")

    # 2. Initialize Generator
    # Use the exact same settings as production (JPEG compression, etc.)
    expert_cfg = ExpertConfig()
    
    dataset = ExpertDataset(
        urdf_path="urdf/panda_mujoco_kinematics.urdf",
        env_xml_path="envs/panda_pick_place.xml",
        # We control the loop manually, so max_samples doesn't strictly matter here
        # but we set it to ensure it runs.
        max_episodes_per_epoch=TEST_EPISODES,
        scripted_cfg=expert_cfg,
        action_scaling_factor=0.5,
        warmup=False
    )

    writer = ExpertDatasetWriter(
        out_dir=str(TEMP_DIR),
        run_name="benchmark",
        image_compression="jpeg", # CRITICAL: Matches production setting
        jpeg_quality=90
    )

    # 3. Generate Data
    log.info("Generating data...")
    start_time = time.time()
    
    # Run the iterator until we have enough episodes
    for _ in dataset:
        if len(dataset.episodes) >= TEST_EPISODES:
            break
            
    # Save to Disk
    log.info("Writing to disk...")
    writer.save_batch(dataset.episodes)
    writer.save() # Finalize index
    
    duration = time.time() - start_time

    # 4. Measure
    # Find the .lmdb file (it might be named based on hash)
    lmdb_files = list(TEMP_DIR.glob("*.lmdb"))
    if not lmdb_files:
        log.error("No LMDB file generated!")
        return

    lmdb_file = lmdb_files[0]
    file_size_bytes = lmdb_file.stat().st_size
    
    # 5. Analysis
    size_mb = file_size_bytes / (1024 * 1024)
    avg_mb_per_ep = size_mb / TEST_EPISODES
    
    projected_500_mb = avg_mb_per_ep * 500
    projected_500_gb = projected_500_mb / 1024

    print("\n" + "="*50)
    print(f" BENCHMARK RESULTS")
    print("="*50)
    print(f"Time taken:         {duration:.2f} seconds")
    print(f"Total File Size:    {size_mb:.2f} MB (for {TEST_EPISODES} eps)")
    print(f"Average Episode:    {avg_mb_per_ep:.2f} MB / episode")
    print("-" * 50)
    print(f"PROJECTION FOR 500 EPISODES:")
    print(f"Estimated Size:     {projected_500_gb:.2f} GB")
    print("="*50)
    
    # Recommendation
    recommended_map_size_gb = int(projected_500_gb * 2) + 1 # 2x safety factor
    print(f"Recommended LMDB 'map_size':  {recommended_map_size_gb} GB (or just use 1024 GB safe limit)")
    print("="*50)

    # Cleanup
    shutil.rmtree(TEMP_DIR)
    log.info("\nCleaned up temporary files.")

if __name__ == "__main__":
    main()