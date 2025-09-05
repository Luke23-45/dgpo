# models/registry.py
from typing import Dict, Type
import torch.nn as nn

# The registry is a simple dictionary mapping model names to model classes
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}

def register_model(name: str):
    """A decorator to register a new model class."""
    def decorator(cls: Type[nn.Module]):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered.")
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def get_model(name: str) -> Type[nn.Module]:
    """Retrieves a model class from the registry."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]