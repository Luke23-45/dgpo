
import torch
import numpy as np
import time
import logging
from torch.utils.data import Dataset
import psutil
import os

logger = logging.getLogger(__name__)

class GpuCachedDataset(Dataset):
    """
    [SOTA OPTIMIZATION] All-in-VRAM Dataset Wrapper.
    
    This wrapper pre-loads an entire PyTorch Dataset into a static GPU tensor dictionary.
    
    Benefits:
    1. Zero CPU RAM usage for data (after initialization).
    2. Zero inter-process communication overhead (num_workers=0).
    3. Zero PCIe transfer latency during training (data is local to GPU).
    
    Requirements:
    - Sufficient VRAM to hold the entire dataset (approx 3.6GB for 8k samples in uint8).
    - Dataset must return dictionaries of numpy arrays or tensors.
    """
    def __init__(self, dataset: Dataset, device: torch.device):
        self.dataset = dataset
        self.device = device
        self.length = len(dataset)
        self.cache = {}
        
        logger.info(f"⚡ [GpuCache] Initializing VRAM Cache for {self.length} samples on {device}...")
        
        # 1. Analyze first item to determine schema and shapes
        try:
            first_item = dataset[0]
        except Exception as e:
            raise RuntimeError(f"[GpuCache] Failed to load first item from dataset: {e}")

        # Define specific dtypes for optimal memory usage
        # We enforce uint8 for images to save 75% memory
        keys_to_cache = {
            "image_primary": torch.uint8,
            "image_wrist": torch.uint8,
            "goal_image_primary": torch.uint8,
            "proprio": torch.float32,
            "actions": torch.float32,
            "expert_target_pose": torch.float32,
            "gt_phase": torch.long,
            "gt_gripper": torch.float32
        }
        
        # 2. Pre-allocate GPU Tensors (Contiguous block)
        # This prevents fragmentation and OOM by reserving the big chunk immediately.
        try:
            for key, target_dtype in keys_to_cache.items():
                if key in first_item:
                    val = first_item[key]
                    # Handle if value is tensor or numpy
                    src_shape = val.shape if hasattr(val, 'shape') else ()
                    
                    # Allocate final shape: (N, ...)
                    final_shape = (self.length, *src_shape)
                    
                    # Calculate estimated size for log
                    elem_size = torch.tensor([], dtype=target_dtype).element_size()
                    total_bytes = np.prod(final_shape) * elem_size
                    logger.info(f"   Allocating '{key}': {final_shape} ({total_bytes / 1024**3:.2f} GB)")
                    
                    # Allocation
                    self.cache[key] = torch.empty(final_shape, dtype=target_dtype, device=device)
                    
        except torch.cuda.OutOfMemoryError:
            logger.error("🛑 [GpuCache] OOM Error during pre-allocation! Your VRAM is too full.")
            torch.cuda.empty_cache()
            raise

        # 3. Fill the Cache (Iterative Scan)
        # We loop through the dataset on CPU and copy to GPU slot-by-slot.
        # This keeps CPU RAM usage extremely low (only 1 item in memory at a time).
        logger.info("   Start filling cache (this may take ~30-60s)...")
        start_time = time.time()
        
        try:
            for i in range(self.length):
                item = dataset[i]
                
                for key in self.cache:
                    if key in item:
                        val = item[key]
                        
                        # Convert to Tensor
                        if isinstance(val, np.ndarray):
                            val = torch.from_numpy(val)
                        elif not isinstance(val, torch.Tensor):
                             # Scalars or lists
                             val = torch.tensor(val)
                        
                        # Handle Image Uint8 Conversion
                        # If the dataset returns floats (0-1), we must scale back to 0-255 uint8
                        if "image" in key and val.dtype == torch.float32:
                            val = (val * 255.0).to(torch.uint8)
                        
                        # Direct Copy to GPU slot
                        # (non_blocking=True helps if pinned, but here we just want speed)
                        self.cache[key][i] = val.to(device, non_blocking=True)
                
                if (i + 1) % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    logger.info(f"   Cached {i + 1}/{self.length} samples... ({rate:.1f} it/s)")
                    
        except Exception as e:
            logger.error(f"   [GpuCache] Failed during cache filling at index {i}: {e}")
            raise

        load_time = time.time() - start_time
        logger.info(f"✅ [GpuCache] Complete! Loaded {self.length} samples in {load_time:.1f}s.")
        
        # 4. Cleanup Source
        # Now that data is on GPU, we can close the LMDB/File handles of the source dataset.
        if hasattr(dataset, "close_env"):
            logger.info("   Closing source dataset LMDB environment to free CPU RAM.")
            dataset.close_env()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Extremely fast lookup. Returns a dict of GPU tensors.
        # Note: These are already on device, so 'to(device)' in training loop becomes a no-op.
        return {k: v[idx] for k, v in self.cache.items()}
