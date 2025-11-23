# FILE: scripts/bootstrap_training_data.py
# (SOTA Data Factory for Colab)

"""
Data Bootstrapper.

This script automates the creation of a large, fresh, training-ready dataset
on the local VM disk. It chains the raw generation and the advantage calculation steps.

Workflow:
1.  **Generate**: Runs `generate_dataset` to create raw episodes with explicit GT labels.
2.  **Compute**: Runs `advantage_calculator` to inject AWR weights.
3.  **Verify**: Checks the health of the new data.
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BOOTSTRAP] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("Bootstrap")

def run_command(cmd_list):
    """Runs a shell command and streams output."""
    cmd_str = " ".join(cmd_list)
    log.info(f"Running: {cmd_str}")
    try:
        subprocess.run(cmd_list, check=True)
    except subprocess.CalledProcessError as e:
        log.error(f"Command failed with exit code {e.returncode}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=260, help="Number of episodes to generate")
    parser.add_argument("--seed", type=int, default=105132456731, help="Random seed for generation")
    parser.add_argument("--workers", type=int, default=2, help="CPU workers for generation")
    parser.add_argument("--output_dir", type=str, default="/content/fresh_data", help="Where to store the final LMDB")
    args = parser.parse_args()

    # Paths
    base_dir = Path(args.output_dir)
    raw_dir = base_dir / "raw_shards"
    final_lmdb_dir = base_dir / "final_training_set"
    
    # Clean start
    if base_dir.exists():
        log.warning(f"Cleaning existing directory: {base_dir}")
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # STEP 1: GENERATE RAW DATA (SHARDED)
    # ==========================================
    log.info(">>> STEP 1: Generating Raw Trajectories...")
    
    # We use subprocess to call your existing robust script
    # num_episodes logic overrides num_samples logic
    gen_cmd = [
        sys.executable, "-m", "scripts.generate_dataset",
        "--config", "configs/gen_dataset_config.yaml",
        "--out_dir", str(raw_dir),
    ]
    
    # We pass overrides via command line arguments that Hydra/argparse accepts
    # Note: Your generate_dataset uses custom parsing + config loading. 
    # We assume we modify the yaml or pass overrides.
    # Since generate_dataset loads config internally, let's write a temp config 
    # or rely on CLI overrides if supported. 
    # Based on your code, it uses `num_workers` from config.
    
    # Let's construct a temporary config override file for generation
    temp_config_path = "configs/temp_gen_config.yaml"
    with open("configs/gen_dataset_config.yaml", "r") as f:
        base_config = f.read()
    
    # Inject new settings
    with open(temp_config_path, "w") as f:
        f.write(base_config)
        f.write(f"\nnum_episodes: {args.episodes}")
        f.write(f"\nnum_workers: {args.workers}")
        f.write(f"\nseed: {args.seed}")
        # Ensure output path matches
        f.write(f"\noutput_dir: {raw_dir}")

    gen_cmd = [
        sys.executable, "-m", "scripts.generate_dataset",
        "--config", temp_config_path,
        "--out_dir", str(raw_dir)
    ]
    
    run_command(gen_cmd)

    # Identify the merged output from Step 1
    # generate_dataset creates `expert_{run}_{num}_episodes.lmdb` inside out_dir
    # We find it dynamically.
    generated_dbs = list(raw_dir.glob("*_episodes.lmdb"))
    if not generated_dbs:
        generated_dbs = list(raw_dir.glob("*.lmdb")) # Fallback
    
    if not generated_dbs:
        log.error("Could not find generated LMDB in output directory.")
        sys.exit(1)
        
    raw_lmdb_path = generated_dbs[0]
    log.info(f"Raw Dataset located: {raw_lmdb_path}")

    # ==========================================
    # STEP 2: CALCULATE ADVANTAGES (Processing)
    # ==========================================
    log.info(">>> STEP 2: Calculating Advantages & Injecting Weights...")
    
    # Final destination
    training_ready_lmdb = final_lmdb_dir / "training.lmdb"
    
    adv_cmd = [
        sys.executable, "-m", "utils.advantage_calculator",
        "--source-db", str(raw_lmdb_path),
        "--dest-db", str(training_ready_lmdb),
        "--gamma", "0.99",
        "--overwrite"
    ]
    
    run_command(adv_cmd)

    # ==========================================
    # STEP 3: VERIFICATION
    # ==========================================
    log.info(">>> STEP 3: Verifying Data Integrity...")
    
    verify_cmd = [
        sys.executable, "-m", "scripts.verify_dataset_integrity",
        "--dataset", str(training_ready_lmdb),
        "--samples", "500"
    ]
    
    run_command(verify_cmd)
    
    log.info("="*50)
    log.info("BOOTSTRAP COMPLETE.")
    log.info(f"New Training Data is ready at: {training_ready_lmdb}")
    log.info("You can now update your training config path.")
    log.info("="*50)

if __name__ == "__main__":
    main()