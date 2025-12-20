import argparse
import lmdb
import numpy as np
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Optimizer")

def evaluate_config(advantages, temp, max_w):
    """
    Returns metrics for a specific (Temp, MaxW) configuration.
    """
    # Calculate weights
    raw_weights = np.exp(advantages / temp)
    
    # Calculate Clipping stats
    clipped_weights = np.clip(raw_weights, 0, max_w)
    clip_mask = raw_weights > (max_w """