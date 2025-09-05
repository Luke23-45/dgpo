# utils/model_utils.py
from __future__ import annotations
import logging
from typing import Any, Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

def extract_pred_actions(model_out: Any) -> torch.Tensor:
    """
    Robustly extracts predicted actions from a model's output, which could be
    a raw tensor or a dictionary containing a standard action key.
    """
    if isinstance(model_out, torch.Tensor):
        pred = model_out
    elif isinstance(model_out, dict):
        pred = None
        for key in ("action", "mu", "mean", "logits", "out"):
            if key in model_out and isinstance(model_out[key], torch.Tensor):
                pred = model_out[key]
                break
        if pred is None:
            raise RuntimeError(f"Could not find a valid action tensor in model output. Keys: {list(model_out.keys())}")
    else:
        raise TypeError(f"Unsupported model output type: {type(model_out)}")

    if pred.ndim == 3 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    if pred.ndim != 2:
        raise ValueError(f"Expected a 2D action tensor (B, A), but got shape {pred.shape} after processing.")
    return pred

def sanity_forward_shape(model: nn.Module, obs_small: Dict[str, Any], action_dim: int) -> None:
    """
    Runs a small forward pass to ensure the model produces a tensor of the
    expected shape (Batch, ActionDim) before starting the main training loop.
    
    Note: Requires a recursive _to_device helper if obs_small is on CPU.
    """
    logger.info("Performing a sanity check on the model's forward pass shape...")
    model.eval()
    device = next(model.parameters()).device
    
    # Helper to move data to the correct device
    def _to_device_recursive(obj, dev):
        if torch.is_tensor(obj):
            return obj.to(dev)
        if isinstance(obj, dict):
            return {k: _to_device_recursive(v, dev) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_to_device_recursive(x, dev) for x in obj)
        return obj

    obs_small_device = _to_device_recursive(obs_small, device)
    
    with torch.no_grad():
        out = model(obs_small_device)
        pred = extract_pred_actions(out).float()

    if pred.ndim != 2 or pred.shape[1] != action_dim:
        raise RuntimeError(
            f"Model forward pass produced an incorrect shape. "
            f"Expected: (Batch, {action_dim}), but got: {tuple(pred.shape)}."
        )
    logger.info(f"✅ Sanity forward OK | Output shape: {tuple(pred.shape)}")
    model.train()