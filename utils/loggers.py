# utils/loggers.py
from __future__ import annotations
import os
from typing import Dict, Any

# A simple protocol for our loggers
class BaseLogger:
    def log_metrics(self, data: Dict[str, Any], step: int):
        raise NotImplementedError
    
    def save_config(self, config: Dict[str, Any]):
        raise NotImplementedError

    def finish(self):
        pass # Optional cleanup

class TensorBoardLogger(BaseLogger):
    def __init__(self, log_dir: str):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError("TensorBoardLogger requires `tensorboard` to be installed. Run `pip install tensorboard`.")
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_metrics(self, data: Dict[str, Any], step: int):
        for key, val in data.items():
            self.writer.add_scalar(key, val, step)

    def save_config(self, config: Dict[str, Any]):
        # TensorBoard can log hyperparameters and text for config
        self.writer.add_hparams(
            {k: v for k, v in config.items() if isinstance(v, (str, bool, int, float))},
            {} # No metrics to link here, just saving the config
        )
        # Also save the full config as text
        config_str = "```\n" + "\n".join([f"{k}: {v}" for k, v in config.items()]) + "\n```"
        self.writer.add_text("config", config_str)

    def finish(self):
        self.writer.close()

# A null logger for when we don't want to log
class NullLogger(BaseLogger):
    def log_metrics(self, data: Dict[str, Any], step: int): pass
    def save_config(self, config: Dict[str, Any]): pass