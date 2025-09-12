#!/usr/bin/env python3
# pretrain_policy.py
"""
Unified pretraining entry point (ExpertDataset + BCTrainer + optional SB3 export)

This script consolidates Phase-1 components into a single, robust CLI:

  1) Builds an ExpertDataset that generates OCTO/IK expert actions on-the-fly.
  2) Trains a BCNet policy via BCTrainer (preferred) or a safe built-in fallback loop.
  3) Saves a .pth checkpoint of BCNet (supervised BC weights).
  4) Optionally instantiates an SB3 PPO agent, transfers compatible weights, and saves .zip.

Hardened points:
 - Removes dataset args your ExpertDataset doesn’t support.
 - Float32 coercion for obs/actions to avoid uint8/float64 drift.
 - Infers action_dim directly from dataset batch before model instantiation.
 - Asserts SB3 env action_dim == BC action_dim for safe export.
"""

from __future__ import annotations

import os
import sys
import gc
import time
import json
import random
import logging
import argparse
from typing import Dict, Any, Optional, Tuple
from torch.utils.data._utils.collate import default_collate
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.obs_adapters import OctoToSB3Adapter 
from models.registry import get_model
import cv2
from pathlib import Path  
import subprocess        

# ---------------------------
# Project imports (Phase-1)
# ---------------------------
from utils.expert_dataset import ExpertDataset
from models.bc_policy import BCNet
from models.custom_sb3_extractor import BCFeaturesExtractor
from utils.paths import resolve_path
from pathlib import Path
from training.bc_trainer import BCTrainer
from utils.model_utils import sanity_forward_shape

# DELETE the old `create_loss_aware_collate_fn` function and ADD THIS CLASS in its place.
# It should be at the top level of the script, not inside any function.

class LossAwareCollate:
    """
    A picklable collate function implemented as a class.
    It wraps the base float32 collate and formats the action tensor 
    correctly for the specified loss function.
    """
    def __init__(self, loss_type: str):
        self.loss_type = loss_type
        # The base collate function is now a member
        self.base_collate = _float32_collate

    def __call__(self, batch):
        # First, do the standard collation (numpy -> torch, float32, etc.)
        obs_batch, act_batch = self.base_collate(batch)

        # Now, perform the final transformation based on the loss type stored during init
        if self.loss_type == "ce":
            # For CrossEntropy, target must be Long and 1D (B,)
            if act_batch.ndim == 2 and act_batch.shape[1] == 1:
                act_batch = act_batch.squeeze(1)
            if act_batch.is_floating_point():
                act_batch = act_batch.long()
        else:
            # For regression losses, ensure it's a float tensor
            if not act_batch.is_floating_point():
                act_batch = act_batch.float()
        
        return obs_batch, act_batch

def _maybe_import_sb3():
    try:
        from stable_baselines3 import PPO
        from envs.panda_env import PandaEnv
        from utils.transfer_bc_to_ppo import transfer_bc_weights
        return PPO, PandaEnv, transfer_bc_weights
    except Exception:
        return None, None, None


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("pretrain_policy")

# ---------------------------
# Utilities
# ---------------------------

def get_git_commit_hash() -> Optional[str]:
    """Tries to get the current git commit hash."""
    try:
        # Check if we are in a git repository
        subprocess.check_output(['git', 'rev-parse', '--is-inside-work-tree'], stderr=subprocess.STDOUT)
        # Get the commit hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        return commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
def set_global_seed(seed: int) -> None:
    """Deterministic-ish setup."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _to_float32_tree(x):
    """Recursively convert all torch.Tensors to float32, except for uint8 images."""
    if isinstance(x, torch.Tensor):
        # Keep uint8 images as-is, convert everything else
        return x if x.dtype == torch.uint8 else x.float()
    if isinstance(x, dict):
        return {k: _to_float32_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_float32_tree(v) for v in x)
    return x

def _float32_collate(batch):
    """
    A custom collate_fn that takes a batch of (obs, action) from the dataset,
    converts them to tensors, coerces dtypes to float32, and then batches them.
    """
    # The dataset can yield (numpy_obs_dict, numpy_action)
    obs_list, act_list = zip(*batch)

    # First, ensure everything is a torch.Tensor
    torch_obs_list = []
    for obs_np in obs_list:
        torch_obs = {}
        for key, val in obs_np.items():
            if isinstance(val, dict):
                torch_obs[key] = {k: torch.from_numpy(np.asarray(v)) for k, v in val.items()}
            else:
                torch_obs[key] = torch.from_numpy(np.asarray(val))
        torch_obs_list.append(torch_obs)
    torch_act_list = [torch.from_numpy(np.asarray(a)) for a in act_list]

    # Now, apply the float32 conversion and default batching
    obs_batch = default_collate([_to_float32_tree(o) for o in torch_obs_list])
    act_batch = default_collate([_to_float32_tree(a) for a in torch_act_list])

    return obs_batch, act_batch


# ADD THIS NEW FUNCTION IN pretrain_policy.py

def create_loss_aware_collate_fn(loss_type: str):
    """
    Creates a collate function that wraps the base float32 collate and then
    formats the action tensor correctly for the specified loss function.
    """
    base_collate = _float32_collate

    def loss_aware_collate(batch):
        # First, do the standard collation (numpy -> torch, float32, etc.)
        obs_batch, act_batch = base_collate(batch)

        # Now, perform the final transformation based on the loss
        if loss_type == "ce":
            # For CrossEntropy, target must be Long and 1D (B,)
            if act_batch.ndim == 2 and act_batch.shape[1] == 1:
                act_batch = act_batch.squeeze(1)
            if act_batch.is_floating_point():
                act_batch = act_batch.long()
        else:
            # For regression losses, ensure it's a float tensor
            if not act_batch.is_floating_point():
                act_batch = act_batch.float()
        
        return obs_batch, act_batch

    return loss_aware_collate



def _build_dataloaders(
    *,
    epoch: int,
    object_size: Tuple[float, ...],
    object_grasp_width: float,
    use_octo: bool,
    # Keep all existing args
    urdf_path: str,
    instruction: str,
    octo_model_name: str,
    env_xml_path: Optional[str],
    base_seed: Optional[int],
    warmup: bool,
    samples_per_epoch: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    loss_type: str,
    val_split_ratio: float,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Constructs train and validation dataloaders by splitting the ExpertDataset.
    """
    if not (0.0 <= val_split_ratio < 1.0):
        raise ValueError(f"val_split_ratio must be between 0.0 and 1.0, but got {val_split_ratio}")
    epoch_seed = base_seed + epoch

    ds = ExpertDataset(
        object_size=object_size,
        object_grasp_width=object_grasp_width,
        use_octo=use_octo,
        urdf_path=urdf_path,
        instruction=instruction,
        octo_model_name=octo_model_name,
        env_xml_path=env_xml_path,
        base_seed=epoch_seed,
        max_samples_per_epoch=int(samples_per_epoch),
        skip_on_error=True,
        warmup=warmup,
    )

    val_loader = None
    collate_fn = LossAwareCollate(loss_type=loss_type)
    if val_split_ratio > 0.0:
        # Since ExpertDataset is an IterableDataset, we cannot use traditional splitting.
        # Instead, we create two separate datasets with different seeds and sample counts.
        num_val_samples = int(samples_per_epoch * val_split_ratio)
        num_train_samples = samples_per_epoch - num_val_samples
        
        logger.info(f"Creating train/val split: {num_train_samples} train samples, {num_val_samples} val samples per epoch.")

        # Reconfigure the main dataset for training
        ds.max_samples_per_epoch = num_train_samples
        
        # Create a new dataset instance for validation
        val_ds = ExpertDataset(
            object_size=object_size,
            object_grasp_width=object_grasp_width,
            use_octo=use_octo,
            urdf_path=urdf_path,
            instruction=instruction,
            octo_model_name=octo_model_name,
            env_xml_path=env_xml_path,
            base_seed=epoch_seed + 9999, # Use a different seed for validation data
            max_samples_per_epoch=num_val_samples,
            skip_on_error=True,
            warmup=warmup,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False, # Don't drop last for validation
            collate_fn=collate_fn,
        )
    
    train_loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_fn,
    )
    
    return train_loader, val_loader


def _instantiate_bcnet(
    *,
    model_name: str, # <-- New argument
    device: torch.device,
    action_dim: int,
    model_ctor_overrides: Optional[str] = None,
    resume_from_path: Optional[str] = None,
) -> Tuple[nn.Module, Optional[Dict[str, Any]]]:
    """
    Instantiates a model from the registry and optionally loads weights.
    """
    logger.info(f"Instantiating model '{model_name}' from the registry...")
    
    # 1. Get the model class from the registry
    model_class = get_model(model_name)
    
    # 2. Prepare constructor arguments
    ctor_kwargs: Dict[str, Any] = {}
    if model_ctor_overrides:
        try:
            ctor_kwargs = json.loads(model_ctor_overrides)
        except Exception as e:
            raise ValueError(f"--model-ctor-overrides must be valid JSON: {e}")

    ctor_kwargs.setdefault("n_actions", action_dim)
    
    # 3. Instantiate the model
    model = model_class(**ctor_kwargs).to(device)
    logger.info(f"Model created with kwargs: {ctor_kwargs}")

    # 4. Load checkpoint weights (logic remains the same)
    checkpoint = None
    if resume_from_path:
        if not os.path.exists(resume_from_path):
            logger.warning(f"Resume path not found: '{resume_from_path}'. Starting from scratch.")
        else:
            logger.info(f"Loading checkpoint to resume training from: {resume_from_path}")
            checkpoint = torch.load(resume_from_path, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)

            # 2. Now, load the extracted state_dict into the model,
            #    passing strict=False to this function.
            model.load_state_dict(state_dict, strict=False)
            # --- END OF FIX ---

            logger.info("Successfully loaded model weights from checkpoint (non-strict).")
            
    return model, checkpoint



def _export_to_sb3_zip(
    *,
    bc_state_dict: Dict[str, torch.Tensor],
    save_path_zip: str,
    ppo_kwargs_json: Optional[str],
    expected_action_dim: int,
    env_xml_path: Optional[str],
) -> None:
    """
    Create an SB3 PPO agent, transfer BC weights, and save a .zip archive.
    Asserts env action_dim == expected_action_dim before saving.
    """
    PPO, PandaEnv, transfer_bc_weights = _maybe_import_sb3()
    if PPO is None or PandaEnv is None or transfer_bc_weights is None:
        raise RuntimeError("SB3 export requested but stable_baselines3/PandaEnv/transfer function are unavailable.")

    logger.info("Creating base PandaEnv for SB3 export...")
    env = PandaEnv(xml_path=env_xml_path)
    
    logger.info("Applying OctoToSB3Adapter to make the environment SB3-compatible...")
    env = OctoToSB3Adapter(env)

    # Guard: action dim must match
    env_act_dim = int(env.action_space.shape[0])
    if env_act_dim != expected_action_dim:
        raise RuntimeError(
            f"Action dim mismatch: env {env_act_dim} != BC {expected_action_dim}. "
            "Ensure dataset/BCNet/env use the same joint ordering and dimension."
        )

    policy_kwargs = {
        "features_extractor_class": BCFeaturesExtractor,
        "net_arch": {
            "pi": [512, 256],  # Policy network hidden layers
            "vf": [512, 256],  # Value network hidden layers
        }
    }
    logger.info(f"Forcing PPO policy_kwargs to match BCNet head: {policy_kwargs}")

    # --- 3. Parse OPTIONAL User-Provided PPO Arguments from command line ---
    user_ppo_kwargs = {}
    if ppo_kwargs_json:
        try:
            user_ppo_kwargs = json.loads(ppo_kwargs_json)
            assert isinstance(user_ppo_kwargs, dict)
        except Exception as e:
            raise ValueError(f"--ppo-kwargs must be a valid JSON object string; got: {ppo_kwargs_json}") from e

    # --- 4. Create the PPO Agent with the Correct, Merged Configuration ---
    agent = PPO(
        "MultiInputPolicy",
        env,
        verbose=0,
        policy_kwargs=policy_kwargs,  # Pass the required architecture here
        **user_ppo_kwargs              # Pass any other user args (like n_steps) here
    )



    # Transfer compatible weights
    report = transfer_bc_weights(bc_state_dict, agent, allow_shape_only_fallback=False, verbose=True)
    if not report.get("loaded_ok", False):
        logger.warning("transfer_bc_weights did not report loaded_ok=True. Review the transfer report above.")

    # Save .zip
    os.makedirs(os.path.dirname(save_path_zip) or ".", exist_ok=True)
    agent.save(save_path_zip)
    logger.info("Saved SB3 PPO agent to %s", save_path_zip)

    # Cleanup
    try:
        env.close()
    except Exception:
        pass
    del agent
    gc.collect()


# ---------------------------
# CLI
# ---------------------------

# In pretrain_policy.py, in the Utilities section

import logging
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch

# Set up a basic logger for demonstration purposes.
# In a real application, this would likely be configured elsewhere.


# --- Constants for Visualization ---
# Using constants makes the code cleaner and easier to modify.
VIS_MAX_IMAGES = 4
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_COLOR_BGR = (255, 255, 255)  # White in BGR
LINE_TYPE = 1
TEXT_START_X = 10
PROPRIO_TEXT_Y = 20
ACTION_TEXT_Y = 40

def _prepare_image_for_save(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a single image tensor into a saveable BGR numpy array.

    This robust version correctly handles multiple data formats:
    - uint8 tensors in [0, 255] range.
    - float tensors in [0, 1] range.
    - float tensors in [-1, 1] range.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input must be a torch.Tensor, but got {type(tensor)}")

    # 1. Move to CPU and convert to NumPy array
    img_np = tensor.cpu().numpy()

    # 2. Handle tensor format (CHW vs HWC)
    if img_np.ndim == 3 and img_np.shape[0] in {1, 3, 4}:
        img_hwc = np.transpose(img_np, (1, 2, 0))
    else:
        img_hwc = img_np

    # 3. Robust Normalization to [0, 255] uint8 range
    # Check if the image is already in the correct uint8 format
    if img_hwc.dtype == np.uint8:
        img_uint8 = img_hwc
    # Handle float tensors
    elif img_hwc.dtype == np.float32 or img_hwc.dtype == np.float64:
        # Check for [-1, 1] range
        if img_hwc.min() < 0:
            img_scaled = (img_hwc * 127.5 + 127.5)
        # Assume [0, 1] range
        else:
            img_scaled = (img_hwc * 255)
        img_uint8 = np.clip(img_scaled, 0, 255).astype(np.uint8)
    else:
        raise TypeError(f"Unsupported numpy dtype: {img_hwc.dtype}")

    # 4. Ensure image is 3-channel BGR for OpenCV
    if img_uint8.ndim == 2:
        return cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
    elif img_uint8.shape[2] == 1:
        return cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
    elif img_uint8.shape[2] == 3:
        return cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    elif img_uint8.shape[2] == 4:
        return cv2.cvtColor(img_uint8, cv2.COLOR_RGBA2BGR)
    else:
        raise ValueError(f"Unsupported number of channels: {img_uint8.shape[2]}")

def save_debug_batch_visualization(
    batch: Tuple[Dict[str, torch.Tensor], torch.Tensor],
    run_dir: Path,
    filename_prefix: str = "debug_batch"
) -> None:
    """
    Saves a visual and numerical representation of a training batch.

    This function is robust and handles:
    - Missing image or proprioception data in the batch.
    - Both CHW and HWC image tensor formats.
    - Float image tensors in [-1, 1] or [0, 1] ranges.
    - Grayscale, RGB, and RGBA images.
    - Missing OpenCV dependency.

    Args:
        batch: A tuple containing (observations, actions).
               Observations is a dictionary of tensors.
        run_dir: The directory to save the debug files in (using pathlib.Path).
        filename_prefix: The base name for the output .png and .txt files.
    """
    try:
        obs_batch, act_batch = batch
    except (ValueError, TypeError) as e:
        logger.error(f"Failed to unpack batch. Expected a tuple of (dict, tensor), but got {type(batch)}. Error: {e}")
        return

    # --- Part 1: Save the Visual PNG ---
    try:
        import cv2
        _create_visual_artifact(obs_batch, act_batch, run_dir, filename_prefix)
    except ImportError:
        logger.warning("OpenCV not found ('pip install opencv-python'). Skipping image visualization.")
    except Exception as e:
        logger.error(f"Failed to create debug image visualization: {e}", exc_info=True)

    # --- Part 2: Save the Numerical TXT ---
    try:
        _create_numerical_artifact(obs_batch, act_batch, run_dir, filename_prefix)
    except Exception as e:
        logger.error(f"Failed to create debug text dump: {e}", exc_info=True)


def _create_visual_artifact(
    obs_batch: Dict[str, torch.Tensor],
    act_batch: torch.Tensor,
    run_dir: Path,
    filename_prefix: str
) -> None:
    """Helper to generate and save the visual PNG artifact."""
    img_primary_batch = obs_batch.get("image_primary")
    if img_primary_batch is None:
        logger.info("No 'image_primary' found in batch. Skipping image visualization.")
        return

    img_wrist_batch = obs_batch.get("image_wrist")
    num_samples_to_show = min(img_primary_batch.shape[0], VIS_MAX_IMAGES)
    vis_strips = []

    for i in range(num_samples_to_show):
        img_primary_bgr = _prepare_image_for_save(img_primary_batch[i])

        if img_wrist_batch is not None:
            img_wrist_bgr = _prepare_image_for_save(img_wrist_batch[i])
            # Resize wrist cam to match primary cam for consistent layout
            h, w, _ = img_primary_bgr.shape
            img_wrist_resized = cv2.resize(img_wrist_bgr, (w, h))
            combined_img = np.concatenate((img_primary_bgr, img_wrist_resized), axis=1)
        else:
            combined_img = img_primary_bgr

        # Add proprioception and action text overlays
        proprio_tensor = obs_batch.get("proprio")
        if proprio_tensor is not None:
            proprio_text = "Proprio: " + np.array2string(
                proprio_tensor[i].cpu().numpy(), precision=2, separator=','
            )
            cv2.putText(combined_img, proprio_text, (TEXT_START_X, PROPRIO_TEXT_Y), FONT, FONT_SCALE, FONT_COLOR_BGR, LINE_TYPE)

        action_text = "Action:  " + np.array2string(
            act_batch[i].cpu().numpy(), precision=2, separator=','
        )
        cv2.putText(combined_img, action_text, (TEXT_START_X, ACTION_TEXT_Y), FONT, FONT_SCALE, FONT_COLOR_BGR, LINE_TYPE)

        vis_strips.append(combined_img)

    final_vis = np.concatenate(vis_strips, axis=0)
    output_path = run_dir / f"{filename_prefix}.png"
    cv2.imwrite(str(output_path), final_vis)
    logger.info(f"✅ Saved debug batch visualization to: {output_path}")


def _create_numerical_artifact(
    obs_batch: Dict[str, torch.Tensor],
    act_batch: torch.Tensor,
    run_dir: Path,
    filename_prefix: str
) -> None:
    """Helper to generate and save the numerical text artifact."""
    output_path = run_dir / f"{filename_prefix}.txt"
    num_samples = act_batch.shape[0]

    with open(output_path, "w") as f:
        f.write(f"--- Debug Batch Dump: {num_samples} samples ---\n\n")

        for i in range(num_samples):
            f.write(f"--- Sample #{i} ---\n")

            # Log shape and stats for all tensors in the observation
            for key, value in obs_batch.items():
                if isinstance(value, torch.Tensor):
                    tensor = value[i].cpu().numpy()
                    f.write(f"  Obs['{key}']:\n")
                    f.write(f"    - Shape: {tensor.shape}\n")
                    f.write(f"    - Dtype: {tensor.dtype}\n")
                    f.write(f"    - Min:   {np.min(tensor):.4f}\n")
                    f.write(f"    - Max:   {np.max(tensor):.4f}\n")
                    f.write(f"    - Mean:  {np.mean(tensor):.4f}\n")
                    # Only print full array if it's small
                    if np.prod(tensor.shape) <= 20:
                        f.write(f"    - Value: {np.array2string(tensor, precision=4)}\n")

            # Log the action tensor
            action_tensor = act_batch[i].cpu().numpy()
            f.write(f"  Action:\n")
            f.write(f"    - Shape: {action_tensor.shape}\n")
            f.write(f"    - Dtype: {action_tensor.dtype}\n")
            f.write(f"    - Value: {np.array2string(action_tensor, precision=4)}\n\n")

    logger.info(f"✅ Saved debug batch numerical data to: {output_path}")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified BC pretraining (ExpertDataset + BCTrainer) with optional SB3 export.")
    # Dataset / OCTO / IK
    run_group = p.add_mutually_exclusive_group(required=True)
    p.add_argument("--urdf_path", type=str, required=True, help="URDF path for IKSolver.")
    p.add_argument("--instruction", type=str, default="pick up the red block", help="Natural language instruction for OCTO.")
    p.add_argument("--octo_model_name", type=str, default="hf://rail-berkeley/octo-small-1.5", help="Octo model identifier.")
    p.add_argument("--env_xml_path", type=str, default=None, help="Optional MuJoCo XML for PandaEnv.")
    p.add_argument("--base_seed", type=int, default=1234, help="Base RNG seed for dataset/model.")
    p.add_argument("--warmup", action="store_true", help="Run one warmup OCTO inference per worker to reduce first-sample latency.")
    p.add_argument("--samples_per_epoch", type=int, default=4096, help="Finite samples per epoch (ExpertDataset.max_samples_per_epoch).")

    # Training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0, help="Prefer 0 for Mujoco stability.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", help="Enable mixed precision (CUDA only).")

    # Model / PPO knobs
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model-ctor-overrides", type=str, default=None,
                   help='JSON dict with kwargs for BCNet constructor. Example: \'{"image_channels":3}\'')
    p.add_argument("--expect_action_dim", type=int, default=None,
                   help="Optional guard: assert dataset action_dim equals this value.")
    p.add_argument("--output-dir", type=str, default="artifacts", 
                   help="Base directory for saving all run artifacts.")
    run_group.add_argument("--run_name", type=str,
                         help="Name for a NEW run. A directory will be created in --output_dir.")
    run_group.add_argument("--resume_dir", type=str,
                         help="Path to an existing run directory to RESUME training.")
    p.add_argument("--resume-from", type=str, default=None, choices=["latest", "best", "error"],
                   help="Explicitly choose which checkpoint to resume from within --resume_dir: "
                        "'latest' (end-of-epoch), 'best' (best validation), or 'error' (emergency save). "
                        "If not set, uses automatic fallback.")
    # Artifacts
    p.add_argument("--artifact", type=str, default="both", choices=("pth", "zip", "both"),
                   help="Which artifacts to produce.")

    p.add_argument("--ppo-kwargs", type=str, default=None,
                   help='JSON dict of kwargs to pass to PPO(...). Example: \'{"n_steps":2048,"batch_size":256}\'')
    p.add_argument("--loss_type", type=str, default="mse", choices=["mse", "l1", "ce"],
                   help="Loss function to use for training.")
    p.add_argument("--early_stop_patience", type=int, default=0,
                   help="Number of epochs with no validation improvement to wait before stopping. 0 to disable.")
    p.add_argument("--val_split_ratio", type=float, default=0.1, 
                   help="Fraction of samples_per_epoch to use for validation (e.g., 0.1 for 10%). Set to 0 to disable validation.")
    p.add_argument("--model_name", type=str, default="bc_net_v1",
                   help="Name of the model architecture to use from the registry.")
    p.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "none"],
                      help="Which logger to use for metrics.")
    p.add_argument("--debug_dump_batch", action="store_true",
                   help="If set, saves a visualization of the first training batch to 'debug_batch.png'.")
    p.add_argument("--use-octo", action="store_true",
                  help="If set, use the OCTO model as the expert. Defaults to the scripted expert.")
    p.add_argument("--object-size", type=str, default="0.04,0.04,0.04",
                  help="Size of the object (x,y,z) as a comma-separated string.")
    p.add_argument("--object-grasp-width", type=float, default=0.6,
                  help="Normalized gripper width for grasping the object (1.0 = fully closed).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.base_seed)
    if args.resume_dir:
        # --- RESUME MODE ---
        logger.info(f"Resuming training from directory: {args.resume_dir}")

        run_dir = Path(args.resume_dir)
        run_name = run_dir.name
        checkpoints_dir = run_dir / "checkpoints"
        logs_dir = run_dir / "logs"
        config_path = run_dir / "config.json"

        error_ckpt = checkpoints_dir / "resume_on_error.pth"
        latest_ckpt = checkpoints_dir / "resume_checkpoint.pth"
        best_ckpt = checkpoints_dir / "best_model.pth"
        
        resume_checkpoint_path = None

        if args.resume_from:
            # --- 1. EXPLICIT USER CHOICE ---
            logger.info(f"Attempting to resume from explicitly requested checkpoint: '{args.resume_from}'")
            if args.resume_from == "latest" and latest_ckpt.is_file():
                resume_checkpoint_path = latest_ckpt
            elif args.resume_from == "best" and best_ckpt.is_file():
                resume_checkpoint_path = best_ckpt
            elif args.resume_from == "error" and error_ckpt.is_file():
                resume_checkpoint_path = error_ckpt
            
            if resume_checkpoint_path is None:
                 logger.critical(f"Resume failed: Explicitly requested checkpoint '{args.resume_from}' not found in {checkpoints_dir}")
                 sys.exit(1)

        else:
            # --- 2. AUTOMATIC FALLBACK (NO KEYWORD PROVIDED) ---
            # Priority: Emergency Save > Latest Epoch > Best Model
            logger.info("No specific checkpoint requested. Using automatic fallback priority.")
            if error_ckpt.is_file():
                resume_checkpoint_path = error_ckpt
                logger.info(f"Found emergency checkpoint '{error_ckpt.name}', will resume.")
            elif latest_ckpt.is_file():
                resume_checkpoint_path = latest_ckpt
                logger.info(f"Found standard resume checkpoint '{latest_ckpt.name}', will resume.")
            elif best_ckpt.is_file():
                resume_checkpoint_path = best_ckpt
                logger.info(f"No standard resume checkpoint found. Falling back to best model '{best_ckpt.name}'.")

        if not config_path.is_file():
            logger.critical(f"Resume failed: config.json not found in {run_dir}")
            sys.exit(1)
        if resume_checkpoint_path is None:
            logger.critical(f"Resume failed: No valid checkpoint (.pth) found in {checkpoints_dir}")
            sys.exit(1)

        # Load the ORIGINAL config, but override a few key values from the new command
        current_cli_args = parse_args()
        
        # 2. Load the configuration from the saved JSON file.
        with config_path.open("r") as f:
            saved_config_dict = json.load(f)
            
        # 3. Update the current args Namespace with the saved values.
        #    This preserves any new arguments from the current script version
        #    while loading the old settings for existing arguments.
        args_dict = vars(current_cli_args) # Get a dictionary of all current args and defaults
        args_dict.update(saved_config_dict) # Update it with the saved values
        args = argparse.Namespace(**args_dict) # Create the final, complete args object

        # 5. Surgically override any arguments that are *meant* to be changed on resume,
        #    like the number of epochs. We get these from `current_cli_args`.
        args.epochs = current_cli_args.epochs
        args.resume_dir = current_cli_args.resume_dir # Ensure resume_dir is correctly set
        args.resume_from = current_cli_args.resume_from # And the resume_from keyword
        # You could add other overridable args here, e.g., args.lr = current_cli_args.lr

        # This tells the rest of the script which checkpoint file to load
        args.resume_from_path_internal = str(resume_checkpoint_path)
    else:
        run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
        run_dir = Path(args.output_dir) / run_name
        # Create the full directory structure
        checkpoints_dir = run_dir / "checkpoints"
        logs_dir = run_dir / "logs"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        args.resume_from_path_internal = None 
        logger.info(f"All artifacts for this run will be saved in: {run_dir}")

    # Save the input configuration
    config_path = run_dir / "config.json"
    with config_path.open("w") as f:
        json.dump(vars(args), f, indent=4)
    logger.info(f"Run configuration saved to: {config_path}")

    # Instantiate the logger with the correct sub-directory
    if args.logger == "tensorboard":
        from utils.loggers import TensorBoardLogger
        metrics_logger = TensorBoardLogger(log_dir=str(logs_dir)) # Pass logs subdir
    else:
        from utils.loggers import NullLogger
        metrics_logger = NullLogger()

    args.urdf_path = resolve_path(args.urdf_path)
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # --- 2. DATA PROBING: Infer dimensions from a sample batch ---
    # Bug 2.2 & 2.3 Fix: Use a dedicated, lightweight loader for probing.
    # We create a temporary dataloader by calling the constructor directly.
    logger.info("Creating a temporary dataloader to infer model dimensions...")
    peek_ds = ExpertDataset(
        urdf_path=args.urdf_path,
        instruction=args.instruction,
        octo_model_name=args.octo_model_name,
        object_size=[float(d) for d in args.object_size.split(',')],
        object_grasp_width=args.object_grasp_width,
        use_octo=args.use_octo,
        env_xml_path=args.env_xml_path,
        base_seed=args.base_seed,
        max_samples_per_epoch=args.batch_size, # Only need one batch
        skip_on_error=True,
        warmup=False, # No warmup needed for a single batch
    )
    peek_loader = DataLoader(
        peek_ds,
        batch_size=args.batch_size,
        num_workers=0, # Use 0 workers for quick, clean probing
        collate_fn=_float32_collate, # Use the base collate for probing
    )


    try:
        first_batch_obs, first_batch_act = next(iter(peek_loader))
    except StopIteration:
        logger.critical("The dataloader is empty. Cannot proceed.")
        sys.exit(1)
    
    # Derive everything from this single batch
    obs_small = {k: v[:2].to(device) for k, v in first_batch_obs.items() if isinstance(v, torch.Tensor)}
    action_dim = first_batch_act.shape[1]

    # Perform the debug dump if requested
    if args.debug_dump_batch:
        # The batch is already in torch.Tensor format from the collate_fn
        save_debug_batch_visualization((first_batch_obs, first_batch_act), run_dir)
    
    del peek_ds, peek_loader
    gc.collect()
    logger.info(f"Action dimension inferred from data: {action_dim}")

    if args.expect_action_dim is not None and int(args.expect_action_dim) != action_dim:
        raise ValueError(f"Expected action dim {args.expect_action_dim}, but inferred {action_dim} from data.")

    # --- 3. COMPONENT ASSEMBLY: Create Model, Optimizer, and Loss ---
    logger.info("Instantiating model and optimizer...")
    model, checkpoint = _instantiate_bcnet(
        model_name=args.model_name,
        device=device,
        action_dim=action_dim,
        model_ctor_overrides=args.model_ctor_overrides,
        resume_from_path=args.resume_from_path_internal,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    logger.info(f"Using loss function: {args.loss_type.upper()}")
    if args.loss_type == "mse":
        loss_fn = torch.nn.MSELoss()
    elif args.loss_type == "l1":
        loss_fn = torch.nn.L1Loss()
    elif args.loss_type == "ce":
        loss_fn = torch.nn.CrossEntropyLoss()
    is_classification_loss = (args.loss_type == "ce")

    # --- 4. PRE-FLIGHT CHECK: Sanity check the model's forward pass ---
    sanity_forward_shape(model, obs_small, action_dim)

    # --- 5. DATA PIPELINE: Create the main dataloader for 
    logger.info("Preparing arguments for the dynamic dataloader factory...")
    dataloader_args = {
        "object_size": [float(d) for d in args.object_size.split(',')],
        "object_grasp_width": args.object_grasp_width,
        "use_octo": args.use_octo,
        "urdf_path": args.urdf_path,
        "instruction": args.instruction,
        "octo_model_name": args.octo_model_name,
        "env_xml_path": args.env_xml_path,
        "base_seed": args.base_seed,
        "warmup": args.warmup,
        "samples_per_epoch": args.samples_per_epoch,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": (device.type == "cuda"),
        "loss_type": args.loss_type,
        "val_split_ratio": args.val_split_ratio,
    }

# --- END OF NEW BLOCK ---
    # --- 6. ENGINE ASSEMBLY: Instantiate the BCTrainer ---
    logger.info("Instantiating BCTrainer engine...")
    trainer = BCTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        # Bug 2.5 Fix: Pass the required is_classification_loss flag
        is_classification_loss=is_classification_loss,
        metrics_logger=metrics_logger,
        device=device,
        use_amp=args.amp,
        grad_clip_norm=args.grad_clip_norm,
        early_stop_patience=args.early_stop_patience,
        seed=args.base_seed,
    )
    if args.resume_from_path_internal:
        trainer.load_checkpoint(args.resume_from_path_internal)
    # --- 7. LAUNCH TRAINING ---
    trainer.fit(
        build_dataloaders_fn=_build_dataloaders,
        dataloader_args=dataloader_args,
        epochs=args.epochs,
        run_dir=run_dir,
    )

    best_model_path = run_dir / "checkpoints" / "best_model.pth"
    
    if best_model_path.is_file():
        logger.info("Loading best model weights for final metadata and export.")
        # Load the checkpoint to get the best loss value for metadata
        best_ckpt = torch.load(best_model_path, map_location="cpu")
        best_val_loss = best_ckpt.get("best_loss")
        
        # Load the state_dict into the model for SB3 export
        model.load_state_dict(best_ckpt["model_state_dict"], strict=False)
        
        # --- Create and save the final metadata.json file ---
        metadata = {
            "run_name": run_name,
            "model_type": "bc",
            "model_architecture": args.model_name,
            "model_config": json.loads(args.model_ctor_overrides or "{}"),
            "training_metrics": {
                "best_validation_loss": best_val_loss,
                "total_epochs_trained": best_ckpt.get("epoch"),
            },
            "artifact_paths": {
                "best_checkpoint": str((checkpoints_dir / "best_model.pth").relative_to(run_dir)),
                "final_checkpoint": str((checkpoints_dir / "final_model.pth").relative_to(run_dir)),
                "tensorboard_logs": str(logs_dir.relative_to(run_dir)),
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_hash": get_git_commit_hash(),
        }
        metadata_path = run_dir / "metadata.json"
        with metadata_path.open("w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Final metadata saved to: {metadata_path}")
        
        # --- SB3 Export (if requested) ---
        if args.artifact in ("zip", "both"):
            logger.info("Proceeding with SB3 agent export...")
            try:
                sb3_save_path = run_dir / "ppo_policy_pretrained.zip"
                _export_to_sb3_zip(
                    bc_state_dict=model.state_dict(),
                    save_path_zip=str(sb3_save_path),
                    ppo_kwargs_json=args.ppo_kwargs,
                    expected_action_dim=action_dim,
                    env_xml_path=args.env_xml_path,
                )
            except Exception as e:
                logger.error(f"Failed to export to SB3 format. Error: {e}", exc_info=True)

    else:
        logger.warning("No 'best_model.pth' found. Skipping metadata generation and SB3 export.")

    # --- 9. CLEANUP ---
    del model, trainer
    gc.collect()
    logger.info("Pretraining script finished.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
    except Exception as e:
        logger.exception("Fatal error in pretraining: %s", e)
        sys.exit(1)
