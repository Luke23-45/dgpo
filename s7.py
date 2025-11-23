# FILE: test_data_consistency.py
# (The Truth-Teller: Dataset vs Environment Verification)

import logging
import sys
import numpy as np
import torch
import hydra
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from omegaconf import DictConfig

# --- Project Imports ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from envs.panda_env import PandaEnv
from utils.semantic_planner_dataset import SemanticPlannerDataset as SemanticPlanningDataset # Adjust import if your dataset file is named differently

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("DATA_TEST")

class DataConsistencyChecker:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cpu") # CPU for analysis
        
        log.info("==================================================")
        log.info("   📊 DATA CONSISTENCY VERIFICATION")
        log.info("==================================================")

    def analyze_dataset_statistics(self):
        """
        Loops through the actual training dataset to calculate Ground Truth statistics.
        """
        log.info("\n[PHASE 1] Analyzing Training Dataset (Ground Truth)...")
        
        # 1. Init Dataset
        try:
            # We assume the config structure matches your training yaml
            dataset = SemanticPlanningDataset(
                root_dir=self.cfg.dataset.root_dir,
                tasks=self.cfg.dataset.tasks,
                # Add any other args your dataset needs from cfg
                transform=None # We want raw values first to check normalization logic
            )
            loader = DataLoader(dataset, batch_size=1, shuffle=True)
            log.info(f"   > Dataset Loaded. Size: {len(dataset)}")
        except Exception as e:
            log.error(f"   ❌ Failed to load Dataset: {e}")
            log.error("      Check 'cfg.dataset.root_dir' in your config.")
            return None, None

        # 2. Iterate and Accumulate
        proprio_accumulator = []
        img_min = 1000
        img_max = -1000
        
        num_samples = min(100, len(dataset)) # Check 100 samples
        log.info(f"   > Sampling {num_samples} items to calculate Mean/Std...")

        for i in range(num_samples):
            sample = dataset[i]
            
            # Handle keys
            # v9.0 might use 'curr_proprio' or 'proprio'
            p = sample.get('curr_proprio', sample.get('proprio', None))
            if p is None:
                log.error("   ❌ Could not find 'proprio' key in dataset sample!")
                return None, None
                
            proprio_accumulator.append(p.numpy())
            
            # Check Image Stats
            img = sample.get('curr_image', sample.get('image', None))
            if img is not None:
                img_min = min(img_min, img.min().item())
                img_max = max(img_max, img.max().item())

        # 3. Calculate Stats
        proprio_stack = np.stack(proprio_accumulator) # (N, Dim)
        
        gt_mean = np.mean(proprio_stack, axis=0)
        gt_std = np.std(proprio_stack, axis=0)
        
        # Avoid division by zero
        gt_std[gt_std < 1e-5] = 1.0 

        log.info(f"   > Dataset Proprio Range: Min={np.min(proprio_stack):.3f}, Max={np.max(proprio_stack):.3f}")
        log.info(f"   > Dataset Image Range:   Min={img_min:.3f}, Max={img_max:.3f}")
        
        if img_max > 2.0:
            log.info("   ℹ️  Dataset Images are [0-255] (Raw).")
        elif img_min < -0.5:
            log.info("   ℹ️  Dataset Images are [-1, 1] (Normalized).")
        else:
            log.info("   ℹ️  Dataset Images are [0, 1] (ToTensor).")

        return gt_mean, gt_std

    def check_environment_match(self, gt_mean, gt_std):
        """
        Compares the Dataset Truth against the Environment's raw output.
        """
        log.info("\n[PHASE 2] Checking Environment Output...")
        
        # 1. Init Env
        try:
            xml_path = self.cfg.get("xml_path", "envs/panda_pick_place.xml")
            env = PandaEnv(xml_path=xml_path, control_mode='delta', render_mode="rgb_array")
            obs, _ = env.reset()
        except Exception as e:
            log.error(f"   ❌ Failed to load Environment: {e}")
            return

        # 2. Get Raw Env Data
        raw_proprio = obs['proprio'] # Numpy array
        raw_img = obs['image_primary'] # (H, W, 3) uint8
        
        log.info(f"   > Env Raw Proprio: {raw_proprio[:5]}... (Shape: {raw_proprio.shape})")
        
        # 3. Check for Mismatch
        # Apply the calculated normalization
        normalized_proprio = (raw_proprio - gt_mean) / gt_std
        
        log.info("\n[PHASE 3] THE VERDICT")
        log.info("-" * 30)
        
        # --- PROPRIO CHECK ---
        is_proprio_mismatched = False
        if np.abs(np.mean(raw_proprio)) > 0.5 and np.max(np.abs(normalized_proprio)) < 3.0:
            # Case: Raw is large (e.g. radians), Normalized is small (-1 to 1).
            # This means the Model EXPECTS Normalized, but Env produces Raw.
            log.warning("🚨 PROPRIO MISMATCH DETECTED!")
            log.info(f"   Dataset expects values around: 0.0")
            log.info(f"   Environment is giving:         {np.mean(raw_proprio):.3f}")
            is_proprio_mismatched = True
        else:
            log.info("✅ Proprioception looks roughly consistent.")

        # --- IMAGE CHECK ---
        # Standard Eval Transform
        tf = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        processed_img = tf(Image.fromarray(raw_img))
        
        log.info(f"   > Env Image (After Transform): Min={processed_img.min():.2f}, Max={processed_img.max():.2f}")
        
        # 4. GENERATE FIX
        if is_proprio_mismatched:
            log.info("\n" + "="*60)
            log.info("🛠️  SOLUTION: COPY THIS INTO 'evaluate_semantic_planner.py'")
            log.info("="*60)
            
            # Format arrays for copy-paste
            str_mean = np.array2string(gt_mean, separator=', ', precision=4).replace('\n', '')
            str_std = np.array2string(gt_std, separator=', ', precision=4).replace('\n', '')
            
            print(f"\n# PASTE THIS AT THE TOP OF YOUR SCRIPT:")
            print(f"PROPRIO_MEAN = np.array({str_mean})")
            print(f"PROPRIO_STD  = np.array({str_std})")
            print("\n" + "="*60)
        else:
            log.info("\n✅ Data looks consistent. If robot fails, check IK or Physics PID gains.")

@hydra.main(version_base=None, config_path="./configs", config_name="train_semantic_planner_config") 
# NOTE: Using TRAIN config here to ensure we load the dataset parameters correctly
def main(cfg: DictConfig):
    checker = DataConsistencyChecker(cfg)
    mean, std = checker.analyze_dataset_statistics()
    
    if mean is not None:
        checker.check_environment_match(mean, std)

if __name__ == "__main__":
    main()