# utils/validation.py
import numpy as np
from typing import Dict, Any

def validate_against_example_batch(octo_model: Any, octo_input: Dict[str, Any]) -> bool:
    """
    Prints differences between a generated OCTO input and the model's example_batch.
    Returns True if compatible, False otherwise.
    """
    try:
        # The example_batch is the "golden reference"
        ex_batch = octo_model.example_batch
    except Exception:
        print("INFO: No example_batch available on the model. Skipping validation.")
        return True

    # Both the input and the example batch are nested under "observations"
    if "observations" not in ex_batch or "observations" not in octo_input:
        print("ERROR: 'observations' key missing from input or example_batch.")
        return False

    ex_obs = ex_batch["observations"]
    actual_obs = octo_input["observations"]

    mismatches = []
    ex_keys = set(ex_obs.keys())
    actual_keys = set(actual_obs.keys())

    # Check for key differences
    if extra_keys := sorted(actual_keys - ex_keys):
        mismatches.append(f"Extra keys in generated obs: {extra_keys}")
    if missing_keys := sorted(ex_keys - actual_keys):
        mismatches.append(f"Missing keys in generated obs: {missing_keys}")

    # Check shapes for common keys
    for key in sorted(ex_keys & actual_keys):
        shape_ex = np.asarray(ex_obs[key]).shape
        shape_actual = np.asarray(actual_obs[key]).shape
        
        # We only care that the dimensions *after* the batch and time dims match
        if shape_ex[2:] != shape_actual[2:]:
            mismatches.append(
                f"Shape mismatch for '{key}': example is {shape_ex}, generated is {shape_actual}"
            )

    if not mismatches:
        print("✅ OK: Generated OCTO input matches the model's example_batch schema.")
        return True
    else:
        print("❌ ERROR: Generated OCTO input has schema mismatches:")
        for m in mismatches:
            print(f"  - {m}")
        return False