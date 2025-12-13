#!/usr/bin/env python3
"""
transfer_bc_to_unified.py - Transfer weights from trained BC model to UnifiedDiffusionPlanner.

This script transfers compatible weights from a trained SemanticPlanner (BC model)
to the UnifiedDiffusionPlanner for faster convergence.

Transferable Components:
- SigLIP vision backbone (fine-tuned top layers)
- Proprio encoder
- Spatial positional embeddings
- Token type embeddings (partial - first 4 types)
- Fusion transformer (if dimension matches)

Non-Transferable (different architecture):
- Trajectory/Gripper heads (BC predicts poses, UDP predicts noise)
- Action-related heads (diffusion head is new)

Usage:
    python scripts/transfer_bc_to_unified.py \
        --bc_checkpoint notes/checkpoints/bc_backup_epoch_082.ckpt \
        --output checkpoints/unified_planner_initialized.ckpt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from collections import OrderedDict

import torch

# Project imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.semantic_planner import SemanticPlanner, SemanticPlannerConfig
from models.unified_diffusion_planner import UnifiedDiffusionPlanner, UnifiedDiffusionConfig

log = logging.getLogger(__name__)

# =============================================================================
# WEIGHT MAPPING
# =============================================================================

# Map BC (SemanticPlanner) keys to UnifiedDiffusionPlanner keys
# Format: (bc_prefix, udp_prefix, requires_reshape)
WEIGHT_MAPPINGS = [
    # SigLIP backbone - EXACT MATCH
    ("model.vision_backbone.", "vision_encoder.vision_backbone.", False),
    
    # Proprio encoder - EXACT MATCH
    ("model.proprio_encoder.", "vision_encoder.proprio_encoder.", False),
    
    # Spatial positional embedding - EXACT MATCH
    ("model.spatial_pos_embedding", "vision_encoder.spatial_pos_emb", False),
    
    # Token type embeddings - PARTIAL (BC has 7 types, UDP has 4)
    # Types: 0=prev, 1=curr, 2=goal, 3=proprio - these match
    ("model.token_type_embeddings.", "vision_encoder.token_type_emb.", "token_type"),
    
    # Phase head - if dimensions match
    ("model.phase_head.", "phase_head.", False),
]


def load_bc_checkpoint(path: str) -> dict:
    """Load BC model checkpoint."""
    log.info(f"Loading BC checkpoint from {path}")
    ckpt = torch.load(path, map_location='cpu',  weights_only=False)
    
    # Handle Lightning checkpoint format
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    
    log.info(f"  Found {len(state_dict)} keys in checkpoint")
    return state_dict


def create_empty_unified_model(bc_checkpoint: dict = None) -> UnifiedDiffusionPlanner:
    """Create a new UnifiedDiffusionPlanner with default config."""
    config = UnifiedDiffusionConfig(
        # Match BC settings where applicable
        proprio_dim=22,
        vision_feature_dim=768,
        fusion_layers=4,
        fusion_heads=8,
        num_task_phases=5,
    )
    log.info("Creating UnifiedDiffusionPlanner with default config")
    model = UnifiedDiffusionPlanner(config)
    return model


def transfer_weights(
    bc_state_dict: dict,
    udp_model: UnifiedDiffusionPlanner,
    strict: bool = False
) -> tuple[int, int, list[str]]:
    """
    Transfer compatible weights from BC to UnifiedDiffusionPlanner.
    
    Returns:
        (transferred_count, skipped_count, skipped_keys)
    """
    udp_state_dict = udp_model.state_dict()
    
    transferred = 0
    skipped = 0
    skipped_keys = []
    matched_keys = set()
    
    log.info("Transferring weights...")
    
    for bc_key, bc_tensor in bc_state_dict.items():
        matched = False
        
        for bc_prefix, udp_prefix, special in WEIGHT_MAPPINGS:
            if bc_key.startswith(bc_prefix):
                # Compute target key
                udp_key = bc_key.replace(bc_prefix, udp_prefix)
                
                # Handle special cases
                if special == "token_type":
                    # Only transfer first 4 token types (0-3)
                    if udp_key in udp_state_dict:
                        udp_shape = udp_state_dict[udp_key].shape
                        bc_tensor_trimmed = bc_tensor[:min(4, bc_tensor.shape[0])]
                        if bc_tensor_trimmed.shape[0] == udp_shape[0]:
                            udp_state_dict[udp_key] = bc_tensor_trimmed
                            log.info(f"  ✓ {bc_key} -> {udp_key} (trimmed to {udp_shape})")
                            transferred += 1
                            matched = True
                            matched_keys.add(udp_key)
                        else:
                            skipped_keys.append(f"{bc_key} (shape mismatch)")
                            skipped += 1
                            matched = True
                else:
                    # Direct transfer
                    if udp_key in udp_state_dict:
                        if bc_tensor.shape == udp_state_dict[udp_key].shape:
                            udp_state_dict[udp_key] = bc_tensor
                            log.debug(f"  ✓ {bc_key} -> {udp_key}")
                            transferred += 1
                            matched = True
                            matched_keys.add(udp_key)
                        else:
                            skipped_keys.append(f"{bc_key} (shape {bc_tensor.shape} != {udp_state_dict[udp_key].shape})")
                            skipped += 1
                            matched = True
                
                break
        
        if not matched:
            skipped += 1
    
    # Load the modified state dict
    udp_model.load_state_dict(udp_state_dict)
    
    # Log summary
    total_udp_keys = len(udp_state_dict)
    log.info(f"\n=== Transfer Summary ===")
    log.info(f"  BC keys:        {len(bc_state_dict)}")
    log.info(f"  UDP keys:       {total_udp_keys}")
    log.info(f"  Transferred:    {transferred}")
    log.info(f"  Skipped:        {skipped}")
    log.info(f"  New (random):   {total_udp_keys - len(matched_keys)}")
    
    return transferred, skipped, skipped_keys


def main(args):
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # 1. Load BC checkpoint
    bc_state_dict = load_bc_checkpoint(args.bc_checkpoint)
    
    # 2. Create empty UDO model
    udp_model = create_empty_unified_model()
    
    # 3. Transfer weights
    transferred, skipped, skipped_keys = transfer_weights(bc_state_dict, udp_model)
    
    # 4. Save initialized model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as Lightning-compatible checkpoint
    checkpoint = {
        'state_dict': udp_model.state_dict(),
        'hyper_parameters': {
            'model_config': udp_model.cfg.__dict__
        },
        'transferred_from': str(args.bc_checkpoint),
        'transferred_count': transferred,
    }
    
    torch.save(checkpoint, output_path)
    log.info(f"\n✓ Saved initialized model to: {output_path}")
    
    # Report skipped keys if verbose
    if args.verbose and skipped_keys:
        log.info("\nSkipped keys:")
        for key in skipped_keys[:20]:  # Limit output
            log.info(f"  - {key}")
        if len(skipped_keys) > 20:
            log.info(f"  ... and {len(skipped_keys) - 20} more")
    
    log.info("\n=== Next Steps ===")
    log.info("1. Update your training config to load this checkpoint")
    log.info("2. Run training with a reduced learning rate (e.g., 5e-5)")
    log.info("3. The diffusion head will learn from scratch, vision encoder is pre-initialized")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer BC weights to UnifiedDiffusionPlanner')
    parser.add_argument('--bc_checkpoint', type=str, required=True,
                        help='Path to BC (SemanticPlanner) checkpoint')
    parser.add_argument('--output', type=str, default='checkpoints/unified_planner_initialized.ckpt',
                        help='Output path for initialized model')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed transfer info')
    
    args = parser.parse_args()
    main(args)
