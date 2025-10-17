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
