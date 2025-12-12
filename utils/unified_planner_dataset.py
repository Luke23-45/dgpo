# FILE: utils/unified_planner_dataset.py
# State-of-the-Art Dataset for UnifiedDiffusionPlanner
# Key feature: Computes DELTA actions at training time

"""
UnifiedPlannerDataset: High-performance dataset for diffusion-based manipulation.

Key Features:
1. Uses ee_pose_world (achieved poses) NOT interpolated waypoints
2. Computes delta actions at training time: delta = target[t+k] - current[t]
3. Action chunking with K=8 future steps
4. Goal-conditioned with initial, current, goal frames
5. Efficient caching via ExpertTrajectoryDataset

This dataset directly addresses the "hovering bug" by:
- Training on achieved poses, not commanded waypoints
- Using delta action representation (more robust to distribution shift)
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.spatial.transform import Rotation as R

log = logging.getLogger(__name__)


# =============================================================================
# 1. DELTA POSE COMPUTATION UTILITIES
# =============================================================================

def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions (xyzw format).
    
    Args:
        q1, q2: Quaternions in [x, y, z, w] format
        
    Returns:
        Product quaternion [x, y, z, w]
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Compute conjugate of quaternion [x, y, z, w]."""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def compute_delta_pose(current_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
    """
    Compute delta pose from current to target.
    
    Delta = Target - Current (for position)
    Delta_quat = Current_inv * Target (for rotation)
    
    Args:
        current_pose: [x, y, z, qx, qy, qz, qw]
        target_pose: [x, y, z, qx, qy, qz, qw]
        
    Returns:
        delta_pose: [dx, dy, dz, dqx, dqy, dqz, dqw]
    """
    # Position delta (simple subtraction)
    delta_pos = target_pose[:3] - current_pose[:3]
    
    # Rotation delta: delta_q = q_current^-1 * q_target
    q_current = current_pose[3:7]
    q_target = target_pose[3:7]
    
    # Normalize quaternions
    q_current = q_current / (np.linalg.norm(q_current) + 1e-8)
    q_target = q_target / (np.linalg.norm(q_target) + 1e-8)
    
    q_current_inv = quaternion_conjugate(q_current)
    delta_quat = quaternion_multiply(q_current_inv, q_target)
    
    # Normalize result
    delta_quat = delta_quat / (np.linalg.norm(delta_quat) + 1e-8)
    
    # Ensure positive w for consistent representation
    if delta_quat[3] < 0:
        delta_quat = -delta_quat
    
    return np.concatenate([delta_pos, delta_quat]).astype(np.float32)


def apply_delta_pose(current_pose: np.ndarray, delta_pose: np.ndarray) -> np.ndarray:
    """
    Apply delta pose to get target pose.
    
    Args:
        current_pose: [x, y, z, qx, qy, qz, qw]
        delta_pose: [dx, dy, dz, dqx, dqy, dqz, dqw]
        
    Returns:
        target_pose: [x, y, z, qx, qy, qz, qw]
    """
    # Position: target = current + delta
    target_pos = current_pose[:3] + delta_pose[:3]
    
    # Rotation: target = current * delta
    q_current = current_pose[3:7]
    delta_quat = delta_pose[3:7]
    
    q_current = q_current / (np.linalg.norm(q_current) + 1e-8)
    delta_quat = delta_quat / (np.linalg.norm(delta_quat) + 1e-8)
    
    q_target = quaternion_multiply(q_current, delta_quat)
    q_target = q_target / (np.linalg.norm(q_target) + 1e-8)
    
    return np.concatenate([target_pos, q_target]).astype(np.float32)


# =============================================================================
# 2. MAIN DATASET CLASS
# =============================================================================

class UnifiedPlannerDataset(Dataset):
    """
    High-performance dataset for UnifiedDiffusionPlanner.
    
    Key differences from SemanticPlannerDataset:
    1. Uses ee_pose_world (achieved poses) instead of expert_target_pose
    2. Computes DELTA actions at training time
    3. Returns prev/curr/goal images + proprio + delta action chunks
    
    This directly fixes the "hovering bug" by training on achieved poses
    with delta action representation.
    """
    
    def __init__(
        self,
        dataset_path: str,
        action_chunk_size: int = 8,
        image_size: int = 224,
        use_augmentation: bool = False
    ):
        """
        Args:
            dataset_path: Path to LMDB dataset
            action_chunk_size: Number of future steps to predict (K)
            image_size: Size to resize images to
            use_augmentation: Whether to apply data augmentation
        """
        super().__init__()
        self.action_chunk_size = action_chunk_size
        self.use_augmentation = use_augmentation
        
        log.info(f"Initializing UnifiedPlannerDataset with K={action_chunk_size}")
        
        # Import expert dataset for efficient data loading
        from utils.expert_dataset import ExpertTrajectoryDataset
        
        # Create underlying reader
        self.expert_reader = ExpertTrajectoryDataset(
            demo_path=dataset_path,
            observation_horizon=2,  # We need t-1, t for prev/curr
            action_horizon=action_chunk_size
        )
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), 
                             interpolation=transforms.InterpolationMode.BICUBIC,
                             antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
        ])
        
        # Augmentation (optional)
        if use_augmentation:
            self.augment = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomGrayscale(p=0.05)
            ])
        else:
            self.augment = None
        
        # Build sample index: (episode_idx, timestep_t)
        self.samples: List[Tuple[int, int]] = []
        for ep_idx, meta in enumerate(self.expert_reader.episode_metadata):
            ep_len = meta.get("episode_len", meta.get("length"))
            if ep_len is None:
                continue
            
            # Need at least: 1 prev step + current + action_chunk_size future steps
            min_start = 1
            max_start = ep_len - action_chunk_size - 1
            
            for t in range(min_start, max_start):
                self.samples.append((ep_idx, t))
        
        log.info(f"Found {len(self.samples)} valid samples from {len(self.expert_reader.episode_metadata)} episodes")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _get_modality(self, ep_idx: int, modality_name: str) -> np.ndarray:
        """Helper to retrieve a full modality array for an episode."""
        ep_meta = self.expert_reader.episode_metadata[ep_idx]
        mod_meta = ep_meta["modalities"].get(modality_name)
        if mod_meta is None:
            raise KeyError(f"Modality '{modality_name}' not found in episode {ep_idx}")
        
        return self.expert_reader._get_full_modality_array(
            mod_meta["key"],
            mod_meta["compression"],
            mod_meta["dtype"],
            tuple(mod_meta["shape"])
        )
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get a training sample.
        
        Returns:
            Dictionary with:
                - prev_image: (3, 224, 224)
                - curr_image: (3, 224, 224)
                - goal_image: (3, 224, 224)
                - curr_proprio: (proprio_dim,)
                - gt_delta_actions: (K, 8) - delta pose (7) + gripper (1)
        """
        if not (0 <= idx < len(self)):
            raise IndexError(f"Index {idx} out of range")
        
        try:
            ep_idx, t = self.samples[idx]
            
            # 1. Get images
            images = self._get_modality(ep_idx, "image_primary")
            
            prev_image_np = images[t - 1]
            curr_image_np = images[t]
            goal_image_np = self.expert_reader.get_goal_image(ep_idx)
            
            # 2. Get proprioception
            proprio = self._get_modality(ep_idx, "proprio")
            curr_proprio_np = proprio[t]
            
            # 3. Get poses - USE ee_pose_world (achieved), NOT expert_target_pose!
            # This is the CRITICAL FIX for the hovering bug
            poses = self._get_modality(ep_idx, "ee_pose_world")
            current_pose = poses[t]  # Current EE pose
            
            # 4. Get gripper states (saved as gt_gripper by save_batch)
            try:
                grippers = self._get_modality(ep_idx, "gt_gripper")
            except KeyError:
                try:
                    grippers = self._get_modality(ep_idx, "gripper_state")
                except KeyError:
                    # Fallback: extract from actions
                    actions = self._get_modality(ep_idx, "actions")
                    grippers = actions[:, -1:]  # Last element is gripper
            
            # 5. Get gt_phase for phase prediction (if available)
            try:
                gt_phases = self._get_modality(ep_idx, "gt_phase")
                gt_phase_t = int(gt_phases[t])
            except KeyError:
                gt_phase_t = None  # Phase not available
            
            # 6. Compute delta actions for chunk
            delta_actions_list = []
            for k in range(self.action_chunk_size):
                future_t = t + k + 1
                if future_t < len(poses):
                    future_pose = poses[future_t]
                    future_gripper = grippers[future_t] if future_t < len(grippers) else grippers[-1]
                else:
                    # Padding: use last pose
                    future_pose = poses[-1]
                    future_gripper = grippers[-1]
                
                # Compute delta from CURRENT pose (not previous delta!)
                delta_pose = compute_delta_pose(current_pose, future_pose)
                
                # Handle gripper
                if isinstance(future_gripper, np.ndarray):
                    gripper_val = float(future_gripper.flatten()[0])
                else:
                    gripper_val = float(future_gripper)
                
                # Combine: 7D delta pose + 1D gripper
                delta_action = np.append(delta_pose, gripper_val)
                delta_actions_list.append(delta_action)
            
            gt_delta_actions = np.stack(delta_actions_list, axis=0).astype(np.float32)
            
            # 6. Transform images
            prev_pil = Image.fromarray(prev_image_np)
            curr_pil = Image.fromarray(curr_image_np)
            goal_pil = Image.fromarray(goal_image_np)
            
            if self.augment is not None:
                # Apply same augmentation to all for consistency
                seed = random.randint(0, 2**32)
                
                random.seed(seed)
                torch.manual_seed(seed)
                prev_pil = self.augment(prev_pil)
                
                random.seed(seed)
                torch.manual_seed(seed)
                curr_pil = self.augment(curr_pil)
                
                random.seed(seed)
                torch.manual_seed(seed)
                goal_pil = self.augment(goal_pil)
            
            prev_image = self.transform(prev_pil)
            curr_image = self.transform(curr_pil)
            goal_image = self.transform(goal_pil)
            
            result = {
                'prev_image': prev_image,
                'curr_image': curr_image,
                'goal_image': goal_image,
                'curr_proprio': torch.from_numpy(curr_proprio_np.copy()).float(),
                'gt_delta_actions': torch.from_numpy(gt_delta_actions).float(),
            }
            
            # Add gt_phase if available (for phase prediction auxiliary loss)
            if gt_phase_t is not None:
                result['gt_phase'] = torch.tensor(gt_phase_t, dtype=torch.long)
            
            return result
            
        except Exception as e:
            log.warning(f"Error loading sample {idx}: {e}")
            return None
    
    def get_action_dim(self) -> int:
        """Get action dimension (7 pose + 1 gripper = 8)."""
        return 8
    
    def get_proprio_dim(self) -> int:
        """Get proprioception dimension from dataset."""
        return self.expert_reader.get_proprioception_dim()


# =============================================================================
# 3. COLLATE FUNCTION
# =============================================================================

def unified_planner_collate_fn(batch: List[Optional[Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    """
    Collate function that handles None samples gracefully.
    """
    # Filter out None samples
    batch = [s for s in batch if s is not None]
    
    if len(batch) == 0:
        raise RuntimeError("All samples in batch are None")
    
    result = {
        'prev_image': torch.stack([s['prev_image'] for s in batch]),
        'curr_image': torch.stack([s['curr_image'] for s in batch]),
        'goal_image': torch.stack([s['goal_image'] for s in batch]),
        'curr_proprio': torch.stack([s['curr_proprio'] for s in batch]),
        'gt_delta_actions': torch.stack([s['gt_delta_actions'] for s in batch]),
    }
    
    # Add gt_phase if all samples have it (for phase prediction auxiliary loss)
    if all('gt_phase' in s for s in batch):
        result['gt_phase'] = torch.stack([s['gt_phase'] for s in batch])
    
    return result


# =============================================================================
# 4. UNIT TEST
# =============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    
    # Test delta pose computation
    log.info("Testing delta pose computation...")
    
    current = np.array([0.3, 0.4, 0.2, 0.0, 0.0, 0.0, 1.0])  # Identity rotation
    target = np.array([0.35, 0.45, 0.15, 0.0, 0.0, 0.1, 0.995])  # Small delta
    
    delta = compute_delta_pose(current, target)
    log.info(f"  Current: {current[:3]}")
    log.info(f"  Target: {target[:3]}")
    log.info(f"  Delta pos: {delta[:3]}")
    
    # Verify inverse
    reconstructed = apply_delta_pose(current, delta)
    pos_error = np.linalg.norm(reconstructed[:3] - target[:3])
    log.info(f"  Reconstruction error: {pos_error:.6f}m")
    assert pos_error < 1e-5, f"Reconstruction error too large: {pos_error}"
    
    log.info("Delta pose tests passed!")
    
    # Test dataset if path exists
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='Path to LMDB dataset')
    args = parser.parse_args()
    
    if args.dataset:
        log.info(f"\nTesting dataset loading from {args.dataset}...")
        
        dataset = UnifiedPlannerDataset(
            dataset_path=args.dataset,
            action_chunk_size=8
        )
        
        log.info(f"Dataset size: {len(dataset)}")
        log.info(f"Action dim: {dataset.get_action_dim()}")
        log.info(f"Proprio dim: {dataset.get_proprio_dim()}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            log.info(f"Sample keys: {sample.keys()}")
            log.info(f"  prev_image: {sample['prev_image'].shape}")
            log.info(f"  curr_image: {sample['curr_image'].shape}")
            log.info(f"  goal_image: {sample['goal_image'].shape}")
            log.info(f"  curr_proprio: {sample['curr_proprio'].shape}")
            log.info(f"  gt_delta_actions: {sample['gt_delta_actions'].shape}")
            
            # Test dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=4,
                shuffle=True,
                num_workers=0,
                collate_fn=unified_planner_collate_fn
            )
            
            batch = next(iter(dataloader))
            log.info(f"\nBatch shapes:")
            for k, v in batch.items():
                log.info(f"  {k}: {v.shape}")
            
            log.info("\nDataset test passed!")
