# utils/transfer_bc_weights.py
"""
Robust weight-transfer utility: copy compatible weights from a pre-trained BCNet
into a Stable-Baselines3 PPO policy.  This implementation:

 - Loads BC state dict (or accepts a BCNet module).
 - Attempts prioritized mapping:
     1) Prefix mapping (cnn -> features_extractor.cnn, proprio_mlp -> features_extractor.mlp)
     2) Explicit head mapping (head.* -> policy_net / action_net variants)
     3) Suffix-based mapping when unambiguous
 - Avoids dangerous shape-only fallbacks by default (opt-in).
 - Ensures tensors are moved and cast to the policy device/dtype before loading.
 - Produces a detailed report dictionary.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Optional

import torch
from models.bc_policy import BCNet
from stable_baselines3 import PPO

logger = logging.getLogger(__name__)


def _shape_str(t: torch.Tensor) -> str:
    return "x".join(str(s) for s in t.shape)


def _infer_policy_device_and_dtype(ppo_agent: PPO) -> Tuple[torch.device, torch.dtype]:
    # Try to inspect one parameter of the policy to infer device/dtype
    for _, p in ppo_agent.policy.named_parameters():
        return p.device, p.dtype
    # fallback
    return torch.device("cpu"), torch.float32


def _maybe_move_and_cast(tensor: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if tensor.device != device or tensor.dtype != dtype:
        return tensor.to(device=device, dtype=dtype)
    return tensor


def transfer_bc_weights(
    bc_model_or_state: Any,
    ppo_agent: PPO,
    allow_shape_only_fallback: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Transfer matching parameters from bc_model (or a state_dict) to ppo_agent.policy.

    Args:
        bc_model_or_state: either a BCNet instance, or a state_dict-like mapping.
        ppo_agent: the SB3 PPO object whose policy weights should be updated.
        allow_shape_only_fallback: if True, a last-resort mapping by shape (single candidate) is allowed.
        verbose: whether to log a detailed report.

    Returns:
        report: dict with transfer statistics and lists
    """
    # --- Get BC state dict ---
    if isinstance(bc_model_or_state, BCNet):
        bc_sd = bc_model_or_state.state_dict()
    elif isinstance(bc_model_or_state, dict):
        bc_sd = bc_model_or_state
    else:
        # attempt to extract model.state_dict() if possible
        try:
            bc_sd = bc_model_or_state.state_dict()
        except Exception as e:
            raise ValueError("Unsupported bc_model_or_state type; provide BCNet instance or state_dict") from e

    # --- PPO policy state dict ---
    ppo_sd = ppo_agent.policy.state_dict()

    # Working copy of PPO state dict we will modify (clone tensors)
    new_ppo_sd = OrderedDict()
    for k, v in ppo_sd.items():
        new_ppo_sd[k] = v.clone() if isinstance(v, torch.Tensor) else v

    # Device/dtype where policy expects params
    policy_device, policy_dtype = _infer_policy_device_and_dtype(ppo_agent)
    if verbose:
        logger.info("Policy device=%s dtype=%s", policy_device, policy_dtype)

    # --- Mapping strategies ---
    # prefix_map: common BC module prefixes -> PPO policy prefixes
    prefix_map = {
        "cnn.": "features_extractor.cnn.",
    }

    # explicit head mapping (BCNet head indexing -> likely PPO names)
    # We'll attempt several plausible PPO targets and only pick one that exists in ppo_sd.
    explicit_map = {
        # Proprio MLP: BCNet's Linear layers are now at indices 0 and 3.
        "proprio_mlp.0.weight": ["features_extractor.proprio_mlp.0.weight"],
        "proprio_mlp.0.bias":   ["features_extractor.proprio_mlp.0.bias"],
        "proprio_mlp.3.weight": ["features_extractor.proprio_mlp.2.weight"], # Map BC idx 3 -> PPO idx 2
        "proprio_mlp.3.bias":   ["features_extractor.proprio_mlp.2.bias"],

        # Head MLP: BCNet's Linear layers are now at indices 0, 3, and 6.
        # Shared layers are correctly mapped to BOTH actor (policy_net) and critic (value_net).
        "head.0.weight": ["mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"],
        "head.0.bias":   ["mlp_extractor.policy_net.0.bias", "mlp_extractor.value_net.0.bias"],
        "head.3.weight": ["mlp_extractor.policy_net.2.weight", "mlp_extractor.value_net.2.weight"],
        "head.3.bias":   ["mlp_extractor.policy_net.2.bias", "mlp_extractor.value_net.2.bias"],

        # Final action layer is mapped ONLY to the actor's action_net.
        "head.6.weight": ["action_net.weight"],
        "head.6.bias":   ["action_net.bias"],
    }

    report = {
        "transferred": [],  # tuples of (bc_key, ppo_key)
        "skipped_shape_mismatch": [],
        "ambiguous": [],
        "not_found": [],
        "bc_total": len(bc_sd),
        "ppo_total": len(ppo_sd),
    }

    # Helper: find first existing candidate name from list
    def _first_existing(candidates: List[str]) -> List[str]:
        """ Helper: find ALL existing candidate names from a list. """
        return [c for c in candidates if c in ppo_sd]

    # --- Strategy 1: Prefix mapping ---
    handled_bc_keys = set()
    for bc_key, bc_tensor in bc_sd.items():
        matched = False
        for src_prefix, tgt_prefix in prefix_map.items():
            if bc_key.startswith(src_prefix):
                candidate = tgt_prefix + bc_key[len(src_prefix):]
                if candidate in ppo_sd:
                    tgt_tensor = ppo_sd[candidate]
                    if tgt_tensor.shape == bc_tensor.shape:
                        new_ppo_sd[candidate] = _maybe_move_and_cast(bc_tensor.clone(), policy_device, policy_dtype)
                        report["transferred"].append((bc_key, candidate))
                    else:
                        report["skipped_shape_mismatch"].append((bc_key, candidate, _shape_str(bc_tensor), _shape_str(tgt_tensor)))
                    matched = True
                else:
                    report["not_found"].append(candidate)
                    matched = True
                break
        if matched:
            handled_bc_keys.add(bc_key)


    # --- Strategy 2: Explicit head mapping ---
    for bc_key, ppo_candidates in explicit_map.items():
        if bc_key in handled_bc_keys: continue
        if bc_key not in bc_sd: continue
        
        bc_tensor = bc_sd[bc_key]
        chosen_targets = _first_existing(ppo_candidates)

        if not chosen_targets:
            report["not_found"].append(bc_key)
        else:
            for target_key in chosen_targets:
                # This logic is correct for copying the tensor to one or more targets
                tgt_tensor = ppo_sd[target_key]
                if tgt_tensor.shape == bc_tensor.shape:
                    new_ppo_sd[target_key] = _maybe_move_and_cast(bc_tensor.clone(), policy_device, policy_dtype)
                    report["transferred"].append((bc_key, target_key))
                else:
                    report["skipped_shape_mismatch"].append((bc_key, target_key, _shape_str(bc_tensor), _shape_str(tgt_tensor)))
        
        handled_bc_keys.add(bc_key)

    # --- Strategy 3: Suffix-based unambiguous mapping for remaining keys ---
    # Build reverse index: suffix -> list of (ppo_key, shape)
    suffix_index_last2: Dict[str, List[str]] = {}
    suffix_index_last1: Dict[str, List[str]] = {}
    for ppo_k in ppo_sd:
        tokens = ppo_k.split(".")
        if len(tokens) >= 2:
            suffix_index_last2.setdefault(".".join(tokens[-2:]), []).append(ppo_k)
        suffix_index_last1.setdefault(tokens[-1], []).append(ppo_k)

    for bc_key, bc_tensor in bc_sd.items():
        if bc_key in handled_bc_keys:
            continue

        bc_tokens = bc_key.split(".")
        
        # First, try matching the more specific last-2-token suffix.
        suffix2 = ".".join(bc_tokens[-2:]) if len(bc_tokens) >= 2 else None
        if suffix2:
            candidates_last2 = suffix_index_last2.get(suffix2, [])
            exact_last2 = [c for c in candidates_last2 if ppo_sd[c].shape == bc_tensor.shape]
            if len(exact_last2) == 1:
                chosen = exact_last2[0]
                new_ppo_sd[chosen] = _maybe_move_and_cast(bc_tensor.clone(), policy_device, policy_dtype)
                report["transferred"].append((bc_key, chosen))
                handled_bc_keys.add(bc_key)
                continue
            elif len(exact_last2) > 1:
                report["ambiguous"].append((bc_key, exact_last2))
                handled_bc_keys.add(bc_key)
                continue

        # If last-2-token suffix fails, FALLBACK to the less specific last-1-token suffix.
        suffix1 = bc_tokens[-1]
        candidates_last1 = suffix_index_last1.get(suffix1, [])
        exact_last1 = [c for c in candidates_last1 if ppo_sd[c].shape == bc_tensor.shape]

        if len(exact_last1) == 1:
            chosen = exact_last1[0]
            new_ppo_sd[chosen] = _maybe_move_and_cast(bc_tensor.clone(), policy_device, policy_dtype)
            report["transferred"].append((bc_key, chosen))
            handled_bc_keys.add(bc_key)
            continue
        elif len(exact_last1) > 1:
            report["ambiguous"].append((bc_key, exact_last1))
            handled_bc_keys.add(bc_key)
            continue

    # --- Optional Strategy 4: shape-only fallback (dangerous) ---
    if allow_shape_only_fallback:
        # For bc keys not yet handled, try to find unique PPO param with same shape
        for bc_key, bc_tensor in bc_sd.items():
            if bc_key in handled_bc_keys:
                continue
            same_shape = [k for k, v in ppo_sd.items() if v.shape == bc_tensor.shape]
            if len(same_shape) == 1:
                chosen = same_shape[0]
                new_ppo_sd[chosen] = _maybe_move_and_cast(bc_tensor.clone(), policy_device, policy_dtype)
                report["transferred"].append((bc_key, chosen))
                handled_bc_keys.add(bc_key)
            elif len(same_shape) > 1:
                report["ambiguous"].append((bc_key, same_shape))
                handled_bc_keys.add(bc_key)
            else:
                report["not_found"].append(bc_key)
                handled_bc_keys.add(bc_key)

    # Any remaining bc keys -> mark as not_found
    for bc_key in bc_sd.keys():
        if bc_key not in handled_bc_keys:
            report["not_found"].append(bc_key)

    # --- Load into PPO policy ---
    try:
        ppo_agent.policy.load_state_dict(new_ppo_sd, strict=False)
        loaded_ok = True
    except Exception as e:
        logger.exception("ppo_agent.policy.load_state_dict failed; will attempt final strict=False fallback: %s", e)
        try:
            ppo_agent.policy.load_state_dict(new_ppo_sd, strict=False)
            loaded_ok = True
        except Exception as ee:
            logger.exception("Final load_state_dict(strict=False) also failed: %s", ee)
            loaded_ok = False

    report["loaded_ok"] = loaded_ok

    # --- Summary logging ---
    if verbose:
        logger.info("---- BC -> PPO transfer report ----")
        logger.info("BC params total: %d", report["bc_total"])
        logger.info("PPO params total: %d", report["ppo_total"])
        logger.info("Transferred pairs: %d", len(report["transferred"]))
        if report["transferred"]:
            logger.info("  Examples (first 8): %s", report["transferred"][:8])
        if report["skipped_shape_mismatch"]:
            logger.warning("  Skipped shape mismatch: %d entries (examples): %s", len(report["skipped_shape_mismatch"]), report["skipped_shape_mismatch"][:8])
        if report["ambiguous"]:
            logger.warning("  Ambiguous matches: %d (examples): %s", len(report["ambiguous"]), report["ambiguous"][:8])
        if report["not_found"]:
            logger.info("  Not found / unmatched BC params: %d (examples): %s", len(report["not_found"]), report["not_found"][:8])
        logger.info("Loaded into PPO policy: %s", loaded_ok)
        logger.info("---- end transfer report ----")

    return report
