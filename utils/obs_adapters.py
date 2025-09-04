# In file: utils/obs_adapters.py

import numpy as np
from typing import Dict, Any, List

def build_octo_observation(
    env_obs: Dict[str, Any],
    history_len: int = 2
) -> Dict[str, np.ndarray]:
    """
    Builds a single-timestep observation dictionary that is strictly compliant
    with the octo-small-1.5 model's expected schema.

    This is the single source of truth for creating OCTO observations.

    Args:
        env_obs: A single observation dictionary from the PandaEnv.
        history_len: The length of the time dimension (T). Should be 2 for
                     ExpertDataset and 1 for RLRewardWrapper's divergence check.

    Returns:
        A dictionary ready to be passed to `octo_model.sample_actions`.
    """
    T = int(max(1, history_len))
    B = 1  # This adapter always creates a batch of 1
    octo_obs = {}

    # --- 1. Filter and Process Core Modalities ---

    # image_primary (ensure HWC)
    if "image_primary" in env_obs:
        img = np.asarray(env_obs["image_primary"])
        if img.ndim == 3 and img.shape[0] in (1, 3): # CHW -> HWC
            img = np.transpose(img, (1, 2, 0))
        # Add Batch and Time dimensions
        octo_obs["image_primary"] = np.repeat(img[np.newaxis, np.newaxis, ...], T, axis=1)

    # image_wrist (ensure HWC)
    if "image_wrist" in env_obs:
        wrist = np.asarray(env_obs["image_wrist"])
        if wrist.ndim == 3 and wrist.shape[0] in (1, 3): # CHW -> HWC
            wrist = np.transpose(wrist, (1, 2, 0))
        # Add Batch and Time dimensions
        octo_obs["image_wrist"] = np.repeat(wrist[np.newaxis, np.newaxis, ...], T, axis=1)

    # timestep (B, T)
    t0 = int(np.asarray(env_obs.get("timestep", 0), dtype=np.int32).reshape(()))
    octo_obs["timestep"] = np.full((B, T), t0, dtype=np.int32)

    
    task_completed_scalar = float(np.asarray(env_obs.get("task_completed", 0.0)).item())
    tc_vec = np.zeros(4, dtype=np.float32)
    tc_vec[0] = task_completed_scalar
    tc_vec_reshaped = tc_vec.reshape(1, 1, -1)
    octo_obs["task_completed"] = np.tile(tc_vec_reshaped, (B, T, 1))
    # --- 2. Build Padding Masks for ONLY the keys we included ---
    pad = np.ones((B, T), dtype=bool)
    
    # Create nested pad mask dict for all keys we've added so far
    # This automatically adapts if, for example, image_wrist is missing.
    nested_pad = {k: pad for k in octo_obs if k != "pad_mask_dict"}
    octo_obs["pad_mask_dict"] = nested_pad
    
    # --- 3. Add the required legacy key ---
    octo_obs["timestep_pad_mask"] = pad

    # --- 4. Explicitly do NOT add proprio, internal_full_proprio, or task_completed ---
    
    return octo_obs


def build_octo_batch_from_list(
    obs_list: List[Dict[str, Any]],
    model_example_batch: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """
    A more advanced adapter for the RLRewardWrapper. It takes a LIST of
    observations and creates a true batch for efficient inference.
    
    This is a placeholder for now, as it's more complex. For your immediate
    needs, the single-observation adapter is sufficient.
    """
    # For now, let's just loop and stack the single-observation adapter.
    # A more optimized version would stack first, then add B,T dims.
    octo_obs_list = [build_octo_observation(obs, history_len=1) for obs in obs_list]
    
    # Collate the list of dictionaries into a single dictionary of stacked arrays
    batch = {}
    if not octo_obs_list:
        return batch
        
    for key in octo_obs_list[0]:
        if isinstance(octo_obs_list[0][key], dict):
            # Handle nested pad_mask_dict
            batch[key] = {}
            for sub_key in octo_obs_list[0][key]:
                batch[key][sub_key] = np.vstack([d[key][sub_key] for d in octo_obs_list])
        else:
            batch[key] = np.vstack([d[key] for d in octo_obs_list])
            
    return batch