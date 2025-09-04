# In utils/paths.py

import os
from pathlib import Path

def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    # Assumes this file is in 'utils' which is one level down from the root.
    return Path(__file__).resolve().parent.parent

def resolve_path(path_str: str) -> str:
    """
    Resolves a given path string. If it's an absolute path, it's returned
    as is. If it's a relative path, it's resolved relative to the project root.
    """
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    
    project_root = get_project_root()
    resolved_path = (project_root / path).resolve()
    
    if not resolved_path.exists():
        # As a fallback, check relative to the current working directory
        cwd_path = (Path.cwd() / path).resolve()
        if cwd_path.exists():
            return str(cwd_path)
    
    return str(resolved_path)