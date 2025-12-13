# FILE: utils/lmdb_utils.py
"""
Robust cross-platform LMDB utility module.

This module provides safe, configurable wrappers for opening and managing LMDB
environments in both single-file and directory modes.

Features:
- Works reliably on Windows (requires subdir=False for .lmdb files)
- Prevents locking errors during concurrent read access
- Provides detailed error messages for diagnostics
- Automatically creates directories if needed
"""

import os
import lmdb
import logging

logger = logging.getLogger("lmdb_utils")


# =============================================================================
# DYNAMIC MAP SIZE CALCULATION
# =============================================================================

# Constants for size estimation (bytes per episode)
BYTES_PER_EPISODE_ESTIMATE = 100 * 1024 * 1024  # ~100 MB per episode (conservative)
MIN_MAP_SIZE_GB = 1.0  # Minimum 1 GB
MAX_MAP_SIZE_GB = 100.0  # Maximum 100 GB (reasonable limit)
SAFETY_FACTOR = 1.2  # 50% extra for safety margin


def calculate_lmdb_map_size_gb(
    num_episodes: int,
    bytes_per_episode: int = BYTES_PER_EPISODE_ESTIMATE,
    safety_factor: float = SAFETY_FACTOR,
    min_size_gb: float = MIN_MAP_SIZE_GB,
    max_size_gb: float = MAX_MAP_SIZE_GB,
) -> float:
    """
    Calculate the appropriate LMDB map size based on expected number of episodes.
    
    This function dynamically computes the map size to avoid:
    - MDB_MAP_FULL errors from undersized databases
    - Wasted disk space from oversized allocations
    
    Args:
        num_episodes: Expected number of episodes to store
        bytes_per_episode: Estimated bytes per episode (~100MB typical for robot demos)
        safety_factor: Multiplier for safety margin (1.5 = 50% extra)
        min_size_gb: Minimum map size in GB
        max_size_gb: Maximum map size in GB
        
    Returns:
        Recommended map size in GB
        
    Example:
        >>> calculate_lmdb_map_size_gb(100)  # 100 episodes
        15.0  # Returns ~15 GB (100 ep * 100MB * 1.5 safety)
    """
    # Calculate raw estimate
    raw_bytes = num_episodes * bytes_per_episode * safety_factor
    raw_gb = raw_bytes / (1024 ** 3)
    
    # Clamp to bounds
    clamped_gb = max(min_size_gb, min(raw_gb, max_size_gb))
    
    # Round up to nearest 0.5 GB for cleaner values
    rounded_gb = round(clamped_gb * 2) / 2
    
    logger.debug(f"LMDB map size: {num_episodes} episodes -> {rounded_gb:.1f} GB")
    
    return rounded_gb


def calculate_lmdb_map_size_bytes(num_episodes: int, **kwargs) -> int:
    """
    Calculate LMDB map size in bytes (for direct use with lmdb.open).
    
    Args:
        num_episodes: Expected number of episodes
        **kwargs: Additional args passed to calculate_lmdb_map_size_gb
        
    Returns:
        Map size in bytes
    """
    size_gb = calculate_lmdb_map_size_gb(num_episodes, **kwargs)
    return int(size_gb * 1024 ** 3)

def open_lmdb_env(
    path: str,
    readonly: bool = False,
    lock: bool = False,
    map_size_gb: float = 1.0,
    readahead: bool = True,
    subdir: bool = None,
) -> lmdb.Environment:
    """
    Safely open an LMDB environment with robust cross-platform behavior.

    Args:
        path (str): Path to the LMDB file or directory.
        readonly (bool): Whether to open in read-only mode.
        lock (bool): Enable LMDB locking (disable for concurrent readers).
        map_size_gb (float): Maximum database map size in gigabytes.
        readahead (bool): Enable OS read-ahead caching.
        subdir (bool or None): Whether the path is a directory.
            If None, it is inferred automatically.

    Returns:
        lmdb.Environment: An open LMDB environment ready for use.
    """
    path = os.path.normpath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # --- Auto-detect single-file vs directory mode ---
    if subdir is None:
        # If path ends with .lmdb or .mdb and is a file, we assume single-file mode
        if os.path.splitext(path)[1].lower() in [".lmdb", ".mdb"]:
            subdir = False
        else:
            subdir = True

    # --- Validate existence when readonly ---
    if readonly and not os.path.exists(path):
        raise FileNotFoundError(
            f"LMDB path not found: {path}. Expected an existing "
            f"{'directory' if subdir else 'file'}."
        )

    # --- Attempt to open environment ---
    try:
        env = lmdb.open(
            path,
            subdir=subdir,
            readonly=readonly,
            lock=lock,
            readahead=readahead,
            map_size=int(map_size_gb * 1e9),
            meminit=False,
            max_dbs=1,
        )
        logger.info(
            f" Opened LMDB at {path} | mode={'RO' if readonly else 'RW'} | "
            f"{'dir' if subdir else 'file'}-mode"
        )
        return env
    except lmdb.Error as e:
        logger.error(f" Failed to open LMDB at {path}: {e}")
        raise


def close_lmdb_env(env: lmdb.Environment):
    """Safely closes an LMDB environment."""
    if env is None:
        return
    try:
        env.close()
        logger.info(" LMDB environment closed successfully.")
    except Exception as e:
        logger.warning(f" LMDB close warning: {e}")
