#!/usr/bin/env python3
#!/pretrain_policy
"""
Robust dataset generation + behavioral cloning pretraining script.

Features:
 - Deterministic OCTO sampling using a persistent JAX PRNGKey (split per sample).
 - Memory-safe streaming mode using numpy.memmap for large NUM_SAMPLES.
 - Retry logic for occasional IK/OCTO failures.
 - Pretraining supports either in-memory arrays or file-backed memmap arrays.
 - Proper device handling for SB3 policy (move batches to policy device).
 - Save dataset artifacts to disk (images.npy, proprio.npy, actions.npy) when streaming.
"""

from __future__ import annotations

import os
import math
import time
import logging
from typing import Dict, Any, Optional, Tuple
from scipy.spatial.transform import Rotation as R
import numpy as np
import jax
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from numpy.lib.format import open_memmap
# Project imports
from octo.model.octo_model import OctoModel
from envs.panda_env import PandaEnv
from utils.ik_solver import IKSolver
from stable_baselines3 import PPO
import gc
from utils.obs_adapters import octo_batch_from_env_obs
from utils.validation import validate_against_example_batch # <-- ADD THIS

logger = logging.getLogger("pretrain")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# -------------------------
# Small helper dataset for file-backed arrays
# -------------------------
class NumpyMemmapDataset(Dataset):
    """
    Reads image/proprio/action arrays from disk-backed .npy files (memmap).
    - images on disk are HWC uint8; we return CHW float32 normalized to [0,1].
    - proprio saved float32.
    - actions saved float32.
    """
    def __init__(self, images_path: str, proprio_path: str, actions_path: str):
        self.images = np.load(images_path, mmap_mode="r")
        self.proprio = np.load(proprio_path, mmap_mode="r")
        self.actions = np.load(actions_path, mmap_mode="r")
        assert len(self.images) == len(self.proprio) == len(self.actions)
        self._len = len(self.actions)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        img = self.images[idx]  # HWC uint8 or float
        # convert to CHW float32 normalized to [0,1]
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)
            if img.max() > 2.0:
                img = img / 255.0
        # convert HWC -> CHW
        img = np.transpose(img, (2, 0, 1)).copy()

        proprio = self.proprio[idx].astype(np.float32)
        action = self.actions[idx].astype(np.float32)
        return torch.from_numpy(img), torch.from_numpy(proprio), torch.from_numpy(action)
  
  
    def close(self):
            """Explicitly close the underlying memory-map file handles."""
            for arr in (self.images, self.proprio, self.actions):
                if hasattr(arr, '_mmap'):
                    try:
                        arr._mmap.close()
                    except Exception:
                        pass # Ignore errors on close
            self.images = self.proprio = self.actions = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        # Best-effort cleanup when the object is garbage collected
        try:
            self.close()
        except Exception:
            pass

# -------------------------
# Core: Dataset generation
# -------------------------
def generate_synthetic_dataset(
    num_samples: int,
    *,
    urdf_path: str = "urdf/panda_mujoco_kinematics.urdf",
    octo_model_name: str = "hf://rail-berkeley/octo-small-1.5",
    output_dir: str = "synthetic_dataset",
    stream_to_disk_threshold: int = 2000,
    retry_limit: int = 3,
    jax_seed: int = 0,
    deterministic_env_seed: Optional[int] = None,
    # --- New robustness arguments ---
    quat_format: str = "xyzw",
    write_placeholders_on_fail: bool = True,
) -> Dict[str, Any]:
    """
    Robust synthetic dataset generator.
    Key improvements:
      - Automatic detection of CHW vs HWC image layout from env sample.
      - Unambiguous image dtype conversion: memmap stores uint8 HWC.
      - Quaternion order normalization (configurable via quat_format).
      - Action-dim inference with safer fallback + error if unknown.
      - Configurable behavior on persistent sample failure.
      - Sanity-check logs to aid debugging.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Generating synthetic dataset: %d samples (output_dir=%s)", num_samples, output_dir)

    try:
        octo_model = OctoModel.load_pretrained(octo_model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load OctoModel '{octo_model_name}': {e}. "
                           "If offline, supply a locally cached model or check network access.")

    ik_solver = IKSolver(urdf_path=urdf_path)
    env = PandaEnv()
    obs, _ = env.reset()

    # --- 1. Robustly detect image layout & dtype ---
    sample_img = np.asarray(obs["image_primary"])
    if sample_img.ndim != 3:
        env.close()
        raise RuntimeError(f"Unexpected image shape from env: {sample_img.shape}")
    
    # If first dim is small (1 or 3), assume CHW. Otherwise, assume HWC.
    if sample_img.shape[0] in (1, 3):
        H, W, C = sample_img.shape[1], sample_img.shape[2], sample_img.shape[0]
    else:
        H, W, C = sample_img.shape[0], sample_img.shape[1], sample_img.shape[2]
    logger.info(f"Detected sample image layout: shape={sample_img.shape}, HWC=({H},{W},{C}), dtype={sample_img.dtype}")

    sample_proprio = np.asarray(obs["proprio"])
    proprio_dim = sample_proprio.shape[-1]
    logger.info(f"Detected proprio_dim={proprio_dim}")

    stream_to_disk = (num_samples > stream_to_disk_threshold)
    logger.info(f"stream_to_disk={stream_to_disk} (threshold={stream_to_disk_threshold})")

    # --- 2. Prepare memmaps with safer action_dim inference ---
    if stream_to_disk:
        images_path = os.path.join(output_dir, "images.npy")
        proprio_path = os.path.join(output_dir, "proprio.npy")
        actions_path = os.path.join(output_dir, "actions.npy")
        images_mm  = open_memmap(images_path,  mode="w+", dtype=np.uint8,   shape=(num_samples, H, W, C))
        proprio_mm = open_memmap(proprio_path, mode="w+", dtype=np.float32, shape=(num_samples, proprio_dim))
        
        try:
            cur_q = np.asarray(sample_proprio[:7], dtype=np.float32)
            # Use a safe, neutral target pose for inference
            sample_target_pose = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
            sample_action = ik_solver.compute_action(sample_target_pose, cur_q)
            action_dim = np.asarray(sample_action).shape[0]
        except Exception as e:
            logger.warning(f"Unable to infer action_dim via IK sample: {e}. Falling back to action_dim=8.")
            action_dim = 8

        actions_mm = open_memmap(actions_path, mode="w+", dtype=np.float32, shape=(num_samples, action_dim))
        logger.info(f"Memmap output prepared (action_dim={action_dim})")
    else:
        images_list, proprio_list, actions_list = [], [], []

    task = octo_model.create_tasks(texts=["pick up the red block"])
    rng = jax.random.PRNGKey(jax_seed)

    # --- 3. Define robust helper functions for data conversion ---
    def _image_to_hwc_uint8(img_np: np.ndarray) -> np.ndarray:
        a = np.asarray(img_np)
        if a.shape[0] in (1, 3): # CHW -> HWC
            a = np.transpose(a, (1, 2, 0))
        if a.dtype == np.uint8:
            return a
        # Handle float to uint8 conversion safely
        if a.max() <= 1.0 and a.min() >= 0.0: # Range [0, 1]
            a = a * 255.0
        return np.clip(a, 0, 255).round().astype(np.uint8)

    def _ensure_xyzw(quat: np.ndarray, fmt: str) -> np.ndarray:
        q = np.asarray(quat)
        if fmt == "xyzw":
            return q
        elif fmt == "wxyz":
            return np.array([q[1], q[2], q[3], q[0]]) # Convert wxyz -> xyzw
        raise ValueError("quat_format must be 'xyzw' or 'wxyz'")

    # --- 4. Main Generation Loop ---
    i_written = 0
    total_failures = 0
    max_total_failures = max(100, num_samples // 2) # Safeguard against infinite loops
    pbar = tqdm(total=num_samples, desc="Generating samples")
    
    while i_written < num_samples:
        if total_failures > max_total_failures:
            env.close()
            pbar.close()
            raise RuntimeError(f"Exceeded max total failures ({max_total_failures}). Aborting.")

        attempt_this_sample = 0
        while attempt_this_sample < retry_limit:
            try:
                seed = deterministic_env_seed + i_written if deterministic_env_seed is not None else None
                obs, _ = env.reset(seed=seed)
                
                # We create a batch of B=1 and a history of T=2 (by duplicating the step)
                octo_obs = octo_batch_from_env_obs(obs, B=1, T=2, task_completed_dim=4)

                # Some OCTO versions expect observations nested under a top-level key
                octo_input = {"observations": octo_obs}
                
                # Add the global pad mask that some OCTO checkpoints require
                octo_input["timestep_pad_mask"] = np.array([[True, True]])
                if i_written == 0: # Only validate on the very first sample
                    is_valid = validate_against_example_batch(octo_model, octo_input)
                    if not is_valid:
                        raise RuntimeError("OCTO input validation failed. Check env and adapter. Aborting.")
                # Now, sample the action using the correctly formatted input
                assert octo_model is not None, "OCTO model is None, cannot sample actions."
                rng, key = jax.random.split(rng)
                action_raw = np.asarray(octo_model.sample_actions(octo_input, task, rng=key))



                target_pose_world_7d = action_raw.reshape(-1, action_raw.shape[-1])[0]

                base_pos, base_quat = env.get_base_pose()
                base_quat_xyzw = _ensure_xyzw(np.asarray(base_quat), quat_format)
                target_quat_xyzw = _ensure_xyzw(target_pose_world_7d[3:], quat_format)

                R_world_base = R.from_quat(base_quat_xyzw)
                R_world_target = R.from_quat(target_quat_xyzw)
                R_base_world = R_world_base.inv()
                
                pos_target_base = R_base_world.apply(target_pose_world_7d[:3] - np.asarray(base_pos))
                R_target_base = R_base_world * R_world_target
                quat_target_base_xyzw = R_target_base.as_quat()
                
                target_pose_base_7d = np.concatenate([pos_target_base, quat_target_base_xyzw]).astype(np.float32)
                
                current_q = np.asarray(obs["proprio"][:7], dtype=np.float32)
                expert_action = np.asarray(ik_solver.compute_action(target_pose_base_7d, current_q), dtype=np.float32)

                # --- Save the valid sample ---
                img_to_save = _image_to_hwc_uint8(image_np)
                proprio_to_save = np.asarray(obs["proprio"], dtype=np.float32)
                
                if stream_to_disk:
                    images_mm[i_written] = img_to_save
                    proprio_mm[i_written] = proprio_to_save
                    # Pad/truncate expert action to fit memmap shape
                    action_to_save = np.zeros(actions_mm.shape[1], dtype=np.float32)
                    slice_len = min(expert_action.shape[0], action_to_save.shape[0])
                    action_to_save[:slice_len] = expert_action[:slice_len]
                    actions_mm[i_written] = action_to_save
                else:
                    images_list.append(img_to_save)
                    proprio_list.append(proprio_to_save)
                    actions_list.append(expert_action)

                i_written += 1
                pbar.update(1)
                break # Success, break inner retry loop

            except Exception as e:
                attempt_this_sample += 1
                total_failures += 1
                if attempt_this_sample >= retry_limit:
                    logger.error(f"Exceeded retry limit for sample idx {i_written}. Reason: {e}")
                    if stream_to_disk and write_placeholders_on_fail:
                        logger.warning("Writing zero placeholders to maintain dataset integrity.")
                        images_mm[i_written] = np.zeros((H, W, C), dtype=np.uint8)
                        proprio_mm[i_written] = np.zeros((proprio_dim,), dtype=np.float32)
                        actions_mm[i_written] = np.zeros((actions_mm.shape[1],), dtype=np.float32)
                        i_written += 1
                        pbar.update(1)
                    break # Move to the next sample index

    pbar.close()
    logger.info("Data generation complete. Written %d samples.", i_written)
    
    # --- 5. Finalize and Return ---
    if stream_to_disk:
        images_mm.flush()
        proprio_mm.flush()
        actions_mm.flush()
        env.close()
        return {"mode": "disk", "images_path": images_path, "proprio_path": proprio_path, "actions_path": actions_path, "n": i_written}
    else:
        env.close()
        # In-memory images are stored as HWC uint8, convert to CHW float for PyTorch
        image_arr_hwc = np.stack(images_list, axis=0)
        image_arr_chw = np.transpose(image_arr_hwc.astype(np.float32) / 255.0, (0, 3, 1, 2))
        return {
            "mode": "memory",
            "obs": {"image_primary": torch.from_numpy(image_arr_chw), "proprio": torch.from_numpy(np.stack(proprio_list))},
            "actions": torch.from_numpy(np.stack(actions_list)),
        }
# -------------------------
# Pretraining (Behavioral Cloning)
# -------------------------


def pretrain_policy_from_dataset(
    dataset_obj: Dict[str, Any],
    *,
    batch_size: int = 128,
    lr: float = 1e-4,
    epochs: int = 10,
    save_path: str = "policy_pretrained_bc.zip",
):
    """
    Robust BC pretraining with corrected action prediction and resource management.
    """
    env = PandaEnv()
    agent = PPO("MultiInputPolicy", env, verbose=0)
    policy = agent.policy
    policy_device = next(policy.parameters()).device
    logger.info(f"Pretraining BC policy for {epochs} epochs | batch={batch_size} | lr={lr} | device={policy_device}")

    dataset = None
    loader = None
    try:
        # --- 1. Dataset and DataLoader Setup ---
        if dataset_obj["mode"] == "memory":
            dataset = TensorDataset(dataset_obj["obs"]["image_primary"], dataset_obj["obs"]["proprio"], dataset_obj["actions"])
        else:
            dataset = NumpyMemmapDataset(dataset_obj["images_path"], dataset_obj["proprio_path"], dataset_obj["actions_path"])

        if len(dataset) == 0:
            logger.error("Dataset is empty. Cannot start pretraining.")
            return None

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(policy_device.type == "cuda"))
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()

        # --- 2. Dry Run with Corrected Action Prediction ---
        b_images, b_proprio, b_actions = next(iter(loader))
        batch_obs = {"image_primary": b_images.to(policy_device), "proprio": b_proprio.to(policy_device)}
        
        with torch.no_grad():
            dist = policy.get_distribution(batch_obs)
            if hasattr(dist, "distribution") and hasattr(dist.distribution, "mean"):
                predicted_actions = dist.distribution.mean
            else:
                predicted_actions = policy._predict(batch_obs, deterministic=True)
        
        action_dim_policy = predicted_actions.shape[-1]
        action_dim_data = b_actions.shape[-1]
        
        if action_dim_policy != action_dim_data:
            raise RuntimeError(f"Action-dimension mismatch: Policy outputs dim {action_dim_policy} but dataset has dim {action_dim_data}.")
        logger.info("Dry run successful: Policy and dataset shapes are compatible.")

        # --- 3. Main Training Loop with Corrected Action Prediction ---
        for epoch in range(epochs):
            policy.train()
            total_loss, n_batches = 0.0, 0
            pbar_desc = f"BC Epoch {epoch+1}/{epochs}"
            with tqdm(loader, desc=pbar_desc) as pbar:
                for b_images, b_proprio, b_actions in pbar:
                    batch_obs = {"image_primary": b_images.to(policy_device), "proprio": b_proprio.to(policy_device)}
                    batch_actions = b_actions.to(policy_device)

                    dist = policy.get_distribution(batch_obs)
                    if hasattr(dist, "distribution") and hasattr(dist.distribution, "mean"):
                        predicted_actions = dist.distribution.mean
                    else:
                        predicted_actions = policy._predict(batch_obs, deterministic=True)
                    
                    loss = loss_fn(predicted_actions, batch_actions)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    n_batches += 1
                    pbar.set_postfix(loss=f"{loss.item():.6f}")

            avg_loss = total_loss / n_batches if n_batches > 0 else float("nan")
            logger.info(f"Epoch {epoch+1}/{epochs} average loss: {avg_loss:.6f}")

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        agent.save(save_path)
        logger.info(f"Pretrained policy saved to {save_path}")
        return save_path

    finally:
        # --- 4. Robust Resource Cleanup ---
        try:
            env.close()
        except Exception:
            pass
        # Important on Windows: explicitly close memmap handles
        if hasattr(dataset, "close"):
            try:
                dataset.close()
            except Exception:
                pass
        # Help Python's garbage collector release resources
        del loader, dataset, agent, policy
        gc.collect()

# -------------------------
# CLI-friendly main
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic OCTO+IK dataset and pretrain BC policy.")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--urdf_path", type=str, default="urdf/panda_mujoco_kinematics.urdf")
    parser.add_argument("--dataset_dir", type=str, default="synthetic_dataset")
    parser.add_argument("--pretrained_out", type=str, default="policy_pretrained_bc.zip")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--jax_seed", type=int, default=0)
    parser.add_argument("--deterministic_env_seed", type=int, default=None)
    args = parser.parse_args()

    ds = generate_synthetic_dataset(
        args.num_samples,
        urdf_path=args.urdf_path,
        output_dir=args.dataset_dir,
        jax_seed=args.jax_seed,
        deterministic_env_seed=args.deterministic_env_seed,
        quat_format=args.quat_format,  # Add this line
    )

    pretrain_policy_from_dataset(
        ds,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        save_path=args.pretrained_out,
    )

    logger.info("All done.")
