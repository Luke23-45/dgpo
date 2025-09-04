# In new file: models/custom_sb3_extractor.py
from typing import Dict
import gymnasium as gym
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class BCFeaturesExtractor(BaseFeaturesExtractor):
    """
    A custom feature extractor for Stable Baselines 3 that perfectly mirrors
    the architecture of the BCNet model's feature extractors.

    This version combines the best features of both proposed implementations:
    1. It robustly infers the proprioception dimension from the observation space.
    2. It uses a clean, dedicated helper method for image normalization.
    """
    def __init__(self, observation_space: gym.spaces.Dict):
        # CNN output (64 * 8 * 8 = 4096) + Proprio MLP output (128) = 4224
        features_dim = 4224
        super().__init__(observation_space, features_dim)

        # --- Architecture is an EXACT COPY of BCNet's modules ---
        
        # 1. Visual Feature Extractor (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
        )

        # 2. Proprioception Feature Extractor (MLP)
        # BEST PRACTICE: Infer the dimension from the observation space
        proprio_dim = observation_space["proprio"].shape[0]
        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )

    def _normalize_image(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalizes images from uint8 [0, 255] to float32 [-1, 1].
        """
        if img_tensor.dtype != torch.uint8:
            # If already float, we assume it's in [0, 1] and scale to [-1, 1]
            # This is more robust than assuming it's already in the correct range.
            return img_tensor * 2.0 - 1.0
            
        return img_tensor.float() / 127.5 - 1.0

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Takes the observation dictionary and produces the final feature vector.
        """
        # --- Image processing ---
        # SB3 automatically handles HWC -> CHW, so we expect (B, C, H, W) here.
        image_obs = observations["image_primary"]
        normalized_image = self._normalize_image(image_obs)
        image_features = self.cnn(normalized_image)

        # --- Proprioception processing ---
        proprio_obs = observations["proprio"]
        proprio_features = self.proprio_mlp(proprio_obs)

        # --- Concatenate to form the final feature vector ---
        return torch.cat([image_features, proprio_features], dim=1)