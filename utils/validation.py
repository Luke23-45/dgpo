# utils/validation.py
import numpy as np
from typing import Dict, Any, Tuple, Mapping, List

RUNTIME_TOPLEVEL_EXCLUDE = {"pad_mask_dict", "timestep_pad_mask"}

def _obs_view(batch: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    Returns a dict-like view of 'observation' tensors and a label indicating format.
    - If batch has a nested 'observations' dict, use that.
    - Otherwise, treat the batch as flat and filter out known runtime-only keys.
    """
    if isinstance(batch, Mapping) and "observations" in batch and isinstance(batch["observations"], Mapping):
        return dict(batch["observations"]), "nested"
    # flat view: everything except runtime masks
    flat = {k: v for k, v in batch.items() if k not in RUNTIME_TOPLEVEL_EXCLUDE}
    return flat, "flat"

def _post_bt(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Safely return the shape after batch/time dims. For <2-D arrays, return ()."""
    if len(shape) >= 2:
        return tuple(shape[2:])
    return ()

def _shape(x: Any) -> Tuple[int, ...]:
    try:
        return tuple(np.asarray(x).shape)
    except Exception:
        return ()

def validate_against_example_batch(octo_model: Any, octo_input: Dict[str, Any]) -> bool:
    """
    Compare octo_input against octo_model.example_batch, tolerant to flat or nested formats.
    Prints a concise, actionable report. Returns True if compatible, False otherwise.
    """
    # 1) Pull example_batch
    try:
        ex_batch = getattr(octo_model, "example_batch")
    except Exception:
        print("INFO: No example_batch available on the model. Skipping validation.")
        return True
    if ex_batch is None:
        print("INFO: example_batch is None on the model. Skipping validation.")
        return True

    # 2) Get observation views
    ex_obs, ex_fmt = _obs_view(ex_batch)
    in_obs, in_fmt = _obs_view(octo_input)

    if not isinstance(ex_obs, dict) or not isinstance(in_obs, dict) or not ex_obs or not in_obs:
        print("INFO: Unable to derive comparable observation dicts from inputs. Skipping.")
        return True

    # 3) Key comparison
    ex_keys = set(ex_obs.keys())
    in_keys = set(in_obs.keys())
    mismatches: List[str] = []

    extra_keys = sorted(in_keys - ex_keys)
    missing_keys = sorted(ex_keys - in_keys)

    if extra_keys:
        mismatches.append(f"Extra keys (ignored at train-time): {extra_keys}")
    if missing_keys:
        mismatches.append(f"Missing keys (present in example): {missing_keys}")

    # 4) Shape comparison (post batch/time dims)
    common = sorted(ex_keys & in_keys)
    for key in common:
        s_ex = _shape(ex_obs[key])
        s_in = _shape(in_obs[key])
        pbt_ex, pbt_in = _post_bt(s_ex), _post_bt(s_in)

        # If both have >=2 dims, compare post-(B,T); else compare full shapes.
        if len(s_ex) >= 2 and len(s_in) >= 2:
            if pbt_ex != pbt_in:
                mismatches.append(
                    f"Shape mismatch for '{key}': example {s_ex} vs input {s_in} (compare tail {pbt_ex} vs {pbt_in})"
                )
        else:
            if s_ex != s_in:
                mismatches.append(
                    f"Shape mismatch for '{key}': example {s_ex} vs input {s_in}"
                )

    # 5) Report
    print(f"Example format: {ex_fmt}; Input format: {in_fmt}")
    if not mismatches:
        print("✅ OK: Input appears schema-compatible with example_batch.")
        return True

    print("❌ Schema differences detected:")
    for m in mismatches:
        print("  -", m)
    return False
