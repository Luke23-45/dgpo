# utils/obs_adapters.py
import numpy as np
from typing import Dict, Any

def octo_batch_from_env_obs(
    obs: Dict[str, Any],
    *,
    B: int = 1,
    T: int = 1,
    task_completed_dim: int = 4,
) -> Dict[str, np.ndarray]:
    """
    Convert a single-timestep env obs into OCTO's batched (B,T,...) dict.
    Ensures:
      - image_primary: (B,T,H,W,C) uint8 or float in [0,1]
      - image_wrist:   (B,T,Hw,Ww,C)
      - proprio:       (B,T,D)
      - task_completed:(B,T,task_completed_dim)
      - pad_mask_dict: nested dict with keys -> (B,T) boolean
      - timestep_pad_mask: (B,T) boolean (legacy)
    """
    # fetch base entries (with fallbacks)
    img_p = np.asarray(obs.get("image_primary"))
    img_w = np.asarray(obs.get("image_wrist", np.zeros((128,128,3), dtype=np.uint8)))
    proprio = np.asarray(obs.get("proprio", obs.get("internal_full_proprio", np.zeros((14,), dtype=np.float32))), dtype=np.float32)
    timestep = np.asarray(obs.get("timestep", np.array([0], dtype=np.int32)), dtype=np.int32)
    task_completed = np.asarray(obs.get("task_completed", np.zeros((1,), dtype=np.float32)), dtype=np.float32)

    # normalize task_completed to desired length
    if task_completed.ndim == 0:
        task_completed = np.array([float(task_completed)], dtype=np.float32)
    if task_completed.size == 1 and task_completed_dim > 1:
        tc = np.zeros((task_completed_dim,), dtype=np.float32)
        tc[0] = float(task_completed[0])
        task_completed = tc
    elif task_completed.size != task_completed_dim:
        # if it's larger or different, try to reshape or truncate/pad
        arr = np.zeros((task_completed_dim,), dtype=np.float32)
        arr[:min(task_completed.size, task_completed_dim)] = task_completed.ravel()[:task_completed_dim]
        task_completed = arr

    # helper to add (B,T,...) dims
    def bt(x):
        x = np.asarray(x)
        return np.broadcast_to(x, (B, T) + x.shape)

    octo = {
        "image_primary": bt(img_p),   # (B,T,H,W,C)
        "image_wrist":   bt(img_w),
        "proprio":       bt(proprio),
        "task_completed":bt(task_completed),
        "timestep":      bt(timestep),
    }

    # Build nested pad_mask_dict (preferred by OCTO)
    mask = np.ones((B, T), dtype=bool)
    octo["pad_mask_dict"] = {
        "image_primary": mask,
        "image_wrist":   mask,
        "proprio":       mask,
        "timestep":      mask,
        "task_completed":mask,
    }

    # legacy alias
    octo["timestep_pad_mask"] = mask

    return octo
