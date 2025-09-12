# models/bc_policy.py
"""
BCNet: Robust Multi-Modal Behavioral Cloning Network for the Panda Robot.

This version is hardened for training / transfer:
 - Flexible observation unpacking (common dict keys accepted).
 - Explicit normalization mode (no heuristic guessing unless chosen).
 - Configurable final activation for action outputs (tanh or linear).
 - AdaptiveAvgPool for fixed CNN output size.
 - Robust load_state_dict_flexible that works across PyTorch versions.
 - Validation helpers: validate_against_env and rescale_action_from_tanh.
 - Clear logging and helpful error messages.

Usage notes:
 - Instantiate with correct `proprio_dim`, `image_channels` and `pooled_spatial`
   matching the BC training configuration.
 - If saving/loading across devices, prefer to load checkpoints to CPU first
   then call `load_state_dict_flexible`.
"""

from __future__ import annotations
from typing import Dict, Tuple, Union, Any, Optional, List

import logging
import math
from .registry import register_model
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

TensorOrDict = Union[torch.Tensor, Dict[str, torch.Tensor]]
ObsType = Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]

@register_model("bc_net_v1")
class BCNet(nn.Module):
    def __init__(
        self,
        n_actions: int,
        *,
        proprio_dim: int = 14,
        image_channels: int = 3,
        pooled_spatial: Tuple[int, int] = (8, 8),
        action_activation: Optional[str] = "tanh",  # "tanh" or None for linear
        normalize_mode: str = "-1,1",  # "-1,1" or "0,1" or "raw"
        dropout_p: float = 0.3,
    ):
        super().__init__()

        # Basic config
        self.n_actions = int(n_actions)
        self.proprio_dim = int(proprio_dim)
        self.image_channels = int(image_channels)
        self.pooled_spatial = pooled_spatial
        self.action_activation = action_activation
        if normalize_mode not in ("-1,1", "0,1", "raw"):
            raise ValueError("normalize_mode must be one of ('-1,1','0,1','raw')")
        self.normalize_mode = normalize_mode
        self.dropout_p = dropout_p

        # Convolutional feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(self.image_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(self.pooled_spatial),
            nn.Flatten(),
        )
        self.cnn_out_channels = 64
        self.cnn_output_dim = self.cnn_out_channels * self.pooled_spatial[0] * self.pooled_spatial[1]

        # Proprio MLP
        self.proprio_mlp = nn.Sequential(
            nn.Linear(self.proprio_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p), # <-- ADD THIS
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p), # <-- AND THIS
        )
        self.proprio_output_dim = 128

        # Fusion head
        fusion_in = self.cnn_output_dim + self.proprio_output_dim
        head_layers: List[nn.Module] = [
            nn.Linear(fusion_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p), # <-- ADD THIS
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p), # <-- AND THIS
            nn.Linear(256, self.n_actions),
        ]
        if self.action_activation == "tanh":
            head_layers.append(nn.Tanh())
        self.head = nn.Sequential(*head_layers)

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights for Conv2d and Linear modules (Kaiming for weights, zeros for biases)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -------------------------
    # Observation helpers
    # -------------------------
    def _unpack_obs(self, obs: ObsType) -> Tuple[torch.Tensor, torch.Tensor]:
        """Accepts multiple common key names and returns (image, proprio)."""
        if isinstance(obs, dict):
            img_keys = ["image_primary", "image", "image_rgb", "pixels", "rgb"]
            prop_keys = ["proprio", "proprioceptive", "state", "vector_obs", "robot_state"]
            img = None
            proprio = None
            for k in img_keys:
                if k in obs:
                    img = obs[k]
                    break
            for k in prop_keys:
                if k in obs:
                    proprio = obs[k]
                    break
            if img is None or proprio is None:
                raise ValueError(
                    "Observation dict must contain image and proprio. "
                    f"Tried image keys: {img_keys}, proprio keys: {prop_keys}."
                )
            return img, proprio
        elif isinstance(obs, (tuple, list)) and len(obs) == 2:
            return obs[0], obs[1]
        else:
            raise ValueError("Unsupported observation format. Provide dict or (image, proprio) tuple/list.")

    # -------------------------
    # Image normalization
    # -------------------------
    def _normalize_image_tensor(self, img: torch.Tensor) -> torch.Tensor:
        """
        Normalize image tensor according to self.normalize_mode.

        Accepts uint8 or float. Accepts CHW (B,C,H,W) or HWC (B,H,W,C) and returns CHW float32.
        """
        if not torch.is_tensor(img):
            raise TypeError("Image must be a torch.Tensor.")

        # If channels-last HWC, move to CHW first for consistent max checks
        if img.ndim == 4 and img.shape[-1] == self.image_channels and (img.shape[1] != self.image_channels):
            img = img.permute(0, 3, 1, 2).contiguous()

        # Convert dtype and normalize according to mode
        if self.normalize_mode == "raw":
            img = img.to(torch.float32)
            return img

        # If uint8, common case
        if img.dtype == torch.uint8:
            img = img.to(torch.float32)
            if self.normalize_mode == "-1,1":
                img = img / 127.5 - 1.0
            else:  # "0,1"
                img = img / 255.0
            return img

        # float dtype: decide by normalize_mode
        img = img.to(torch.float32)
        if self.normalize_mode == "-1,1":
            # assume input either in [-1,1], [0,1], or [0,255]
            maxval = float(img.max().item())
            if maxval > 2.0:  # likely [0,255]
                img = img / 127.5 - 1.0
            else:
                # already roughly in [-1,1] or [0,1]; if in [0,1], shift
                if img.min().item() >= 0.0 and maxval <= 1.0:
                    img = img * 2.0 - 1.0
            return img
        else:  # "0,1"
            maxval = float(img.max().item())
            if maxval > 2.0:
                img = img / 255.0
            # if already in [-1,1] map to [0,1]
            if img.min().item() < 0.0:
                img = (img + 1.0) / 2.0
            return img

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, obs: ObsType) -> torch.Tensor:
        """
        Forward pass.

        Args:
            obs: dict with keys for image and proprio (flexible), or (image, proprio) tuple/list.

        Returns:
            actions: Tensor (B, n_actions) with values in [-1,1] if action_activation=='tanh', else linear.
        """
        img, proprio = self._unpack_obs(obs)

        if not torch.is_tensor(proprio):
            raise TypeError("Proprio must be a torch.Tensor.")
        if proprio.ndim != 2 or proprio.shape[1] != self.proprio_dim:
            raise ValueError(f"Proprio tensor must be shape (B, {self.proprio_dim}). Got {tuple(proprio.shape)}")

        img = self._normalize_image_tensor(img)
        if img.ndim != 4:
            raise ValueError(f"Image tensor must be 4D (B,C,H,W). Got shape {tuple(img.shape)}")
        if img.shape[1] != self.image_channels:
            raise ValueError(f"Image channel mismatch. Expected {self.image_channels}, got {img.shape[1]}")

        img_feat = self.cnn(img)  # (B, cnn_output_dim)
        prop_feat = self.proprio_mlp(proprio)  # (B, proprio_output_dim)

        fused = torch.cat([img_feat, prop_feat], dim=1)  # (B, fusion_in)
        out = self.head(fused)  # (B, n_actions)
        return out

    # -------------------------
    # Flexible checkpoint loader
    # -------------------------
    def load_state_dict_flexible(self, ckpt: Any, strict: bool = False) -> Dict[str, Any]:
        """
        Robust loader that accepts:
          - raw state_dict
          - checkpoint dict containing 'model_state_dict' (or similar)
        Returns a report: {'missing_keys': [], 'unexpected_keys': [], 'loaded': bool}
        """
        if ckpt is None:
            raise ValueError("Checkpoint is None.")
        if isinstance(ckpt, dict) and ("model_state_dict" in ckpt):
            state_dict = ckpt["model_state_dict"]
        else:
            # Best-effort: assume ckpt is a raw state dict or object with state_dict()
            if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                state_dict = ckpt
            else:
                try:
                    state_dict = ckpt.state_dict()
                except Exception:
                    state_dict = ckpt  # leave as-is; let load_state_dict raise if incompatible

        report: Dict[str, Any] = {"missing_keys": [], "unexpected_keys": [], "loaded": False}
        try:
            res = self.load_state_dict(state_dict, strict=strict)
            # PyTorch may return a NamedTuple with .missing_keys/.unexpected_keys or None
            if res is None:
                report["missing_keys"] = []
                report["unexpected_keys"] = []
            else:
                report["missing_keys"] = list(getattr(res, "missing_keys", []))
                report["unexpected_keys"] = list(getattr(res, "unexpected_keys", []))
            report["loaded"] = True
        except Exception as e:
            logger.exception("Failed to load state_dict into BCNet: %s", e)
            report["loaded"] = False
            # Do not re-raise here; caller may want to inspect report
        logger.info("BCNet.load_state_dict_flexible loaded=%s missing=%d unexpected=%d",
                    report["loaded"], len(report["missing_keys"]), len(report["unexpected_keys"]))
        return report

    # -------------------------
    # Validation and helpers
    # -------------------------
    def validate_against_shapes(self, img_hw: Tuple[int, int] = (256, 256), batch_size: int = 1) -> None:
        """Quick forward pass to ensure shapes line up (raises on mismatch)."""
        img = torch.randn(batch_size, self.image_channels, img_hw[0], img_hw[1], dtype=torch.float32)
        proprio = torch.randn(batch_size, self.proprio_dim, dtype=torch.float32)
        out = self({"image_primary": img, "proprio": proprio})
        if out.shape != (batch_size, self.n_actions):
            raise RuntimeError(f"BCNet forward output shape {out.shape} != (B, {self.n_actions})")
        logger.info("BCNet.validate_against_shapes OK: img %s -> out %s", img.shape, out.shape)

    @staticmethod
    def rescale_action_from_tanh(tanh_out: torch.Tensor, action_space) -> torch.Tensor:
        """
        Map tanh outputs in [-1,1] to the environment action_space (numpy Box).
        action_space expected to have .low and .high numpy arrays.
        """
        if not torch.is_tensor(tanh_out):
            raise TypeError("tanh_out must be a torch.Tensor.")
        low = torch.from_numpy(action_space.low).to(tanh_out.device, dtype=tanh_out.dtype)
        high = torch.from_numpy(action_space.high).to(tanh_out.device, dtype=tanh_out.dtype)
        scaled = (tanh_out + 1.0) / 2.0
        return low + scaled * (high - low)

    def __repr__(self) -> str:
        return (
            f"<BCNet n_actions={self.n_actions} proprio_dim={self.proprio_dim} "
            f"image_ch={self.image_channels} pooled={self.pooled_spatial} act={self.action_activation} norm={self.normalize_mode}>"
        )


# -------------------------
# Smoke tests (run as script)
# -------------------------
if __name__ == "__main__":
    import numpy as np
    def run_smoke_test(batch_size=4, img_size=256):
        model = BCNet(n_actions=8, proprio_dim=14, image_channels=3, pooled_spatial=(8, 8))
        # 1) float normalized [-1,1] case
        imgs = torch.randn(batch_size, 3, img_size, img_size, dtype=torch.float32) * 0.5
        propr = torch.randn(batch_size, 14, dtype=torch.float32)
        out = model({"image_primary": imgs, "proprio": propr})
        assert out.shape == (batch_size, 8)
        print("✅ Float-normalized CHW test passed.")

        # 2) uint8 HWC
        imgs_uint8 = (np.random.rand(batch_size, img_size, img_size, 3) * 255).astype(np.uint8)
        imgs_t = torch.from_numpy(imgs_uint8)
        propr2 = torch.randn(batch_size, 14, dtype=torch.float32)
        out2 = model({"image_primary": imgs_t, "proprio": propr2})
        assert out2.shape == (batch_size, 8)
        print("✅ Uint8 HWC test passed.")

        # 3) validate_against_shapes
        model.validate_against_shapes(img_hw=(img_size, img_size))
        print("✅ validate_against_shapes passed.")

    run_smoke_test()
