
import sys
import argparse
from pathlib import Path
import logging

# Add root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate.evaluate_unified_planner_auto import main as original_main, HybridEvaluator
from models.unified_diffusion_planner import UnifiedDiffusionPlanner, UnifiedDiffusionConfig
import torch

# Mock Policy Loader
def mock_load_policy(self, checkpoint_path):
    print(f"Mocking policy load for: {checkpoint_path}")
    cfg = UnifiedDiffusionConfig()
    self.model = UnifiedDiffusionPlanner(cfg).to(self.device)
    self.model.eval()

# Patch the method
HybridEvaluator._load_policy = mock_load_policy

if __name__ == "__main__":
    # Remove --checkpoint if present to avoid argparse errors if we wanted to hardcode it, 
    # but original_main expects it. We will pass a dummy path and let the mock handle it.
    original_main()
