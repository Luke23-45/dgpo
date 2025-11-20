
import time
import torch
import sys
import os
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.semantic_planner_dataset import SemanticPlannerDataset

def verify_performance():
    # Path to the dataset - adjust if necessary
    dataset_path = "C:/Users/Hellx/Documents/Programming/python/Project/redhot/data/training_ready/training.lmdb"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Please update the path in the script.")
        return

    print(f"Initializing dataset from {dataset_path}...")
    try:
        dataset = SemanticPlannerDataset(dataset_path=dataset_path, use_aug=False)
    except Exception as e:
        print(f"Failed to initialize dataset: {e}")
        return

    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        print("Dataset is empty.")
        return

    # Warmup
    print("Warming up (loading 1 sample)...")
    _ = dataset[0]

    # Benchmark
    num_samples = 50
    print(f"Benchmarking loading {num_samples} samples...")
    
    start_time = time.time()
    for i in range(num_samples):
        _ = dataset[i % len(dataset)]
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_samples
    
    print(f"Total time: {total_time:.4f}s")
    print(f"Average time per sample: {avg_time:.4f}s")
    print(f"Throughput: {1/avg_time:.2f} samples/s")
    
    # Verification of shapes
    sample = dataset[0]
    print("\nSample shapes:")
    for k, v in sample.items():
        print(f"  {k}: {v.shape}")

if __name__ == "__main__":
    verify_performance()
