# FILE: models/diffusion_policy.py
"""
State-of-the-art diffusion policy (Transformer-conditioned) for multi-view,
temporally-chunked robotic manipulation data.

Design notes:
 - Input conditioning: multi-view image sequences (primary & wrist) and proprio
   sequence. Shapes expected by API are documented in each method.
 - Action representation: sequences of joint-deltas (H_a x D_a). Policy predicts
   noise over flattened action sequence, or returns sampled denoised actions.
 - Denoiser: Transformer encoder that accepts tokenized action-sequence tokens
   and cross-attends / conditions on observation tokens (via concatenation).
 - Scheduler: flexible noise schedule (linear / cosine).
 - EMA teacher: optional Exponential Moving Average copy for stable sampling.
 - Logging & diagnostics: shaped returns, device-aware.

Important: This implementation favors clarity, robustness, and modularity.
Swap the VisionEncoder or Transformer components with heavier backbones later.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import math
import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# -------------------------
# Utilities
# -------------------------
def exists(x):
    return x is not None

def default(val, d):
    return val if exists(val) else d

# small stable projection helper
def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    return x / (x.norm(p=2, dim=dim, keepdim=True).clamp(min=eps))


# -------------------------
# Noise scheduler
# -------------------------
@dataclass
class NoiseSchedulerConfig:
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: str = "linear"  # or "cosine"
    timesteps: int = 100


class NoiseScheduler:
    """
    Implements forward diffusion schedule and helper alphas for DDPM/DDIM sampling.

    - q_sample: sample x_t given x_0 and noise eps
    - predict_x0_from_xt: closed-form
    - ddim_step: one DDIM reverse step (deterministic/stochastic with eta)
    """
    def __init__(self, cfg: NoiseSchedulerConfig):
        self.cfg = cfg
        self.T = int(cfg.timesteps)
        if cfg.schedule == "linear":
            self.betas = torch.linspace(cfg.beta_start, cfg.beta_end, self.T)
        elif cfg.schedule == "cosine":
            # cosine schedule ala Nichol & Dhariwal
            timesteps = self.T
            s = 0.008
            steps = torch.arange(timesteps + 1, dtype=torch.float64)
            alphas_cumprod = torch.cos(((steps / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = (alphas_cumprod / alphas_cumprod[0]).float()
            self.betas = (1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])).clamp(min=1e-6)
            self.betas = self.betas.float()
        else:
            raise ValueError("Unknown schedule: " + str(cfg.schedule))

        self.betas = self.betas.clone()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=self.alphas_cumprod.dtype), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

    def q_sample(self, x0: torch.Tensor, t: torch.LongTensor, noise: torch.Tensor):
        """
        Sample x_t given x0: x_t = sqrt(alpha_cumprod[t]) * x0 + sqrt(1 - alpha_cumprod[t]) * noise
        x0: (B, D)
        t: (B,) long
        noise: (B, D)
        """
        assert x0.shape == noise.shape
        device = x0.device
        acp = self.sqrt_alphas_cumprod.to(device)[t].unsqueeze(-1)
        scm = self.sqrt_one_minus_alphas_cumprod.to(device)[t].unsqueeze(-1)
        return acp * x0 + scm * noise

    def get_alpha_terms(self, t: torch.LongTensor, device=None):
        device = device or t.device
        return {
            "alpha": self.alphas.to(device)[t].unsqueeze(-1),
            "alpha_cumprod": self.alphas_cumprod.to(device)[t].unsqueeze(-1),
            "sqrt_alpha_cumprod": self.sqrt_alphas_cumprod.to(device)[t].unsqueeze(-1),
            "sqrt_one_minus_alpha_cumprod": self.sqrt_one_minus_alphas_cumprod.to(device)[t].unsqueeze(-1),
            "beta": self.betas.to(device)[t].unsqueeze(-1)
        }

    def predict_x0_from_eps(self, xt: torch.Tensor, t: torch.LongTensor, eps: torch.Tensor):
        terms = self.get_alpha_terms(t, device=xt.device)
        sqrt_alpha_cumprod = terms["sqrt_alpha_cumprod"]
        sqrt_one_minus_alpha_cumprod = terms["sqrt_one_minus_alpha_cumprod"]
        x0_pred = (xt - sqrt_one_minus_alpha_cumprod * eps) / sqrt_alpha_cumprod
        return x0_pred

    def ddim_step(self, xt: torch.Tensor, t: int, t_next: int, eps: torch.Tensor, eta: float = 0.0):
        """
        One DDIM step taking xt -> x_{t_next} deterministically if eta=0.
        xt: (B, D)
        t, t_next: scalars (int timestep indices)
        eps: predicted noise at time t (B, D)
        Returns x_{t_next}.
        """
        # use terms computed as tensors
        device = xt.device
        alpha_t = self.alphas_cumprod[t]
        alpha_t_next = self.alphas_cumprod[t_next]

        sqrt_alpha_t = math.sqrt(alpha_t)
        sqrt_alpha_t_next = math.sqrt(alpha_t_next)
        sqrt_one_minus_alpha_t = math.sqrt(max(1.0 - alpha_t, 1e-20))

        # predict x0
        x0_pred = (xt - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t

        # direction pointing to xt
        sigma_t = eta * math.sqrt((1 - alpha_t_next) / (1 - alpha_t)) * math.sqrt(1 - alpha_t / alpha_t_next)
        # compute the mean predicted xt_next (no noise)
        dir_xt = math.sqrt(1 - alpha_t_next) * eps
        x_next_mean = sqrt_alpha_t_next * x0_pred + dir_xt * (math.sqrt(1 - alpha_t_next))
        if sigma_t == 0.0:
            return x_next_mean
        else:
            noise = torch.randn_like(xt)
            return x_next_mean + sigma_t * noise


# -------------------------
# Vision encoder (simple, replaceable)
# -------------------------
class VisionEncoder(nn.Module):
    """
    Lightweight CNN-based image encoder.

    Input: images shaped (B, H_o, C, H_img, W_img)
    Output: features shaped (B, H_o, D_v)
    """
    def __init__(self, in_channels: int = 3, features_dim: int = 64):
        super().__init__()
        self.features_dim = features_dim
        # Basic conv stack - easy to replace with stronger backbone (ResNet/ViT)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=3, padding=3),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(64, features_dim, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H_o, C, H_img, W_img)
        returns (B, H_o, D_v)
        """
        B, H_o = x.shape[0], x.shape[1]
        # collapse batch and time for CNN forward
        x = x.view(B * H_o, x.shape[2], x.shape[3], x.shape[4])
        feat = self.conv(x).view(B, H_o, -1)  # (B, H_o, features_dim)
        return feat


# -------------------------
# Temporal encoder (small Transformer)
# -------------------------
class TemporalTransformer(nn.Module):
    """
    Encodes sequence of observational tokens (B, H_o, D_cond) -> (B, H_o, D_model)
    Uses standard TransformerEncoder layers.
    """
    def __init__(self, d_input: int, d_model: int = 256, n_layers: int = 3, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_input, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq: (B, H_o, D_input)
        returns: (B, H_o, D_model)
        """
        x = self.input_proj(seq)
        x = self.transformer(x)
        x = self.out_proj(x)
        return x


# -------------------------
# Denoiser (Transformer-based, conditioned)
# -------------------------
class TransformerDenoiser(nn.Module):
    """
    Denoiser that takes the noisy action sequence (B, H_a, D_a) and sequence
    conditioning tokens (B, H_o, D_cond_model) and predicts noise for each
    action token: output (B, H_a, D_a).

    Implementation: Project action tokens and cond tokens to a common embedding
    space, concatenate them (cond first, then action tokens), run through a
    Transformer encoder, and finally project action token positions back to
    D_a noise predictions.
    """
    def __init__(self, action_dim: int, action_horizon: int, cond_dim: int, d_model: int = 256, n_layers: int = 6, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.cond_dim = cond_dim
        self.d_model = d_model

        # token projections
        self.act_proj = nn.Linear(action_dim, d_model)
        self.cond_proj = nn.Linear(cond_dim, d_model)

        # positional embeddings for total sequence (cond_len + action_len)
        self.pos_emb = nn.Parameter(torch.randn(1, 1024, d_model) * 0.02)  # support up to 1024 tokens (safe)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, action_dim)

        # small time embedding for diffusion timestep
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, noisy_actions: torch.Tensor, cond_seq: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        noisy_actions: (B, H_a, D_a)
        cond_seq: (B, H_o, D_cond)  --> already produced by temporal encoder
        timesteps: (B,) long or float timesteps
        returns: predicted noise (B, H_a, D_a)
        """
        B = noisy_actions.shape[0]
        device = noisy_actions.device
        H_o = cond_seq.shape[1]
        H_a = noisy_actions.shape[1]
        assert H_a == self.action_horizon, f"Expected H_a={self.action_horizon}, got {H_a}"

        # project tokens
        act_tokens = self.act_proj(noisy_actions)  # (B, H_a, d_model)
        cond_tokens = self.cond_proj(cond_seq)     # (B, H_o, d_model)

        # time embedding and add to both token sets (broadcast)
        t_emb = self.time_mlp(timesteps.float().unsqueeze(-1))  # (B, d_model)
        t_emb = t_emb.unsqueeze(1)  # (B, 1, d_model)
        act_tokens = act_tokens + t_emb
        cond_tokens = cond_tokens + t_emb

        # concat cond then actions
        seq = torch.cat([cond_tokens, act_tokens], dim=1)  # (B, H_o + H_a, d_model)
        seq_len = seq.shape[1]
        # add positional embeddings (slice)
        if seq_len > self.pos_emb.shape[1]:
            raise ValueError("Sequence too long for positional embeddings; increase pos_emb length.")
        seq = seq + self.pos_emb[:, :seq_len, :].to(device)

        # transformer
        out = self.transformer(seq)  # (B, H_o + H_a, d_model)

        # take the action token positions and project back to action dim
        action_out = out[:, H_o:, :]  # (B, H_a, d_model)
        noise_pred = self.out_proj(action_out)  # (B, H_a, D_a)
        return noise_pred


# -------------------------
# EMA helper
# -------------------------
class EMA:
    """
    Exponential Moving Average for model weights.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.collected = False

    def update(self, model: nn.Module):
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema_model.state_dict().items():
                if v.dtype.is_floating_point:
                    v.copy_(v * self.decay + msd[k].to(v.device) * (1.0 - self.decay))
                else:
                    v.copy_(msd[k].to(v.device))

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, sd):
        self.ema_model.load_state_dict(sd)


# -------------------------
# High-level DiffusionPolicy class
# -------------------------
class DiffusionPolicy(nn.Module):
    """
    Top-level diffusion policy wrapper.

    Constructor args:
      - image_channels: channels per image (usually 3)
      - image_feat_dim: per-image feature dim output by vision encoder
      - proprio_dim: dims of proprio vector per timestep (D_p)
      - H_o: observation horizon
      - H_a: action horizon
      - action_dim: per-step action dimension (D_a)
      - scheduler_cfg: NoiseSchedulerConfig instance
      - model dims: d_model, n_layers, ...
      - ema_decay: if provided, create EMA teacher

    Key API:
      - compute_loss(actions, obs_dict) -> (loss, diagnostics)
          actions: (B, H_a, D_a)
          obs_dict: keys 'image_primary', 'image_wrist', 'proprio'
            image tensors expected as FloatTensor / uint8 normalized to [0,255]
            with shape (B, H_o, H, W, C) (matches ExpertTrajectoryDataset collate)
      - sample(obs_dict, steps=None, eta=0.0, use_ema=True) -> sampled_actions (B, H_a, D_a)
      - save(path) / load(path)
      - log_prob_proxy(actions, obs_dict) -> approximate log-prob (not exact)
    """
    def __init__(
        self,
        *,
        image_channels: int = 3,
        image_feat_dim: int = 64,
        proprio_dim: int = 22,
        H_o: int = 2,
        H_a: int = 8,
        action_dim: int = 8,
        scheduler_cfg: NoiseSchedulerConfig = NoiseSchedulerConfig(),
        d_model: int = 256,
        denoiser_layers: int = 6,
        denoiser_heads: int = 8,
        denoiser_dropout: float = 0.1,
        ema_decay: Optional[float] = 0.999,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.device = default(device, torch.device("cpu"))
        # store dims
        self.image_channels = image_channels
        self.image_feat_dim = image_feat_dim
        self.proprio_dim = proprio_dim
        self.H_o = H_o
        self.H_a = H_a
        self.action_dim = action_dim
        self.d_model = d_model

        # components
        self.vision = VisionEncoder(in_channels=image_channels, features_dim=image_feat_dim)
        # cond dimension per timestep: primary + wrist + proprio
        cond_dim = (2 * image_feat_dim) + proprio_dim
        self.temporal = TemporalTransformer(d_input=cond_dim, d_model=d_model, n_layers=3, n_heads=4, dropout=0.1)

        self.denoiser = TransformerDenoiser(action_dim=action_dim, action_horizon=H_a, cond_dim=d_model, d_model=d_model, n_layers=denoiser_layers, n_heads=denoiser_heads, dropout=denoiser_dropout)

        # noise scheduler
        self.scheduler_cfg = scheduler_cfg
        self.scheduler = NoiseScheduler(scheduler_cfg)

        # EMA teacher
        self.ema = EMA(self, decay=ema_decay) if ema_decay is not None and ema_decay > 0.0 else None

        # to device
        self.to(self.device)

        logger.info(f"DiffusionPolicy initialized: H_o={H_o}, H_a={H_a}, action_dim={action_dim}, device={self.device}")

    # -------------------------
    # Conditioning helpers
    # -------------------------
    def _prepare_images(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Ensure images are float tensors normalized to [-1,1] or [0,1] as required.
        img_tensor shape expectations (B, H_o, H, W, C) with channel-last (numpy-style).
        We'll convert to (B, H_o, C, H, W) and float32 / normalize to [0,1].
        """
        if img_tensor is None:
            raise ValueError("Image tensor is required.")
        if not torch.is_tensor(img_tensor):
            img_tensor = torch.as_tensor(img_tensor)
        # input may be uint8 or float
        img = img_tensor.float()
        # channel-last -> channel-first
        if img.dim() == 5:  # (B, H_o, H, W, C)
            img = img.permute(0, 1, 4, 2, 3).contiguous()
        elif img.dim() == 4 and img.shape[1] in (1, 3):  # (B, C, H, W) - single timestep
            # expand time dimension
            img = img.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected image tensor shape {img.shape}. Expected (B,H_o,H,W,C).")
        img = img / 255.0
        return img.to(self.device)

    def _cond_embed(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Create sequence conditioning tokens of shape (B, H_o, d_model).
        obs keys:
          - image_primary: (B, H_o, H, W, C) (numpy-style)
          - image_wrist: (B, H_o, H, W, C)
          - proprio: (B, H_o, D_p)
        """
        # validate
        if "image_primary" not in obs or "image_wrist" not in obs or "proprio" not in obs:
            raise KeyError("Observation dict must contain keys 'image_primary','image_wrist','proprio'")

        img_p = self._prepare_images(obs["image_primary"])  # (B,H_o,C,H,W)
        img_w = self._prepare_images(obs["image_wrist"])
        proprio = torch.as_tensor(obs["proprio"], dtype=torch.float32, device=self.device)
        if proprio.dim() == 2:
            # shape (B, D_p) -> replicate across H_o
            proprio = proprio.unsqueeze(1).repeat(1, self.H_o, 1)
        # ensure shapes
        B = proprio.shape[0]
        assert proprio.shape[1] == self.H_o, f"Expected proprio horizon {self.H_o}, got {proprio.shape[1]}"

        # vision features
        feat_p = self.vision(img_p)  # (B, H_o, D_v)
        feat_w = self.vision(img_w)  # (B, H_o, D_v)

        # concat
        cond_seq = torch.cat([feat_p, feat_w, proprio.to(self.device)], dim=-1)  # (B, H_o, D_cond)
        cond_emb = self.temporal(cond_seq)  # (B, H_o, d_model)
        return cond_emb

    # -------------------------
    # Loss / forward (training)
    # -------------------------
    def compute_loss(self, actions: torch.Tensor, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute diffusion MSE loss for a batch.

        actions: torch.Tensor shaped (B, H_a, D_a) or (B, H_a*D_a)
        obs: dict with keys image_primary, image_wrist, proprio (see _cond_embed)

        Returns: (loss, diagnostics)
        """
        # to tensor
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        # normalize shape to (B, H_a, D_a)
        if actions.dim() == 2:
            B = actions.shape[0]
            actions = actions.view(B, self.H_a, self.action_dim)
        elif actions.dim() == 3:
            pass
        else:
            raise ValueError("actions must be shape (B, H_a, D_a) or (B, H_a*D_a)")

        B = actions.shape[0]
        device = actions.device

        # flatten for noise sampling and scheduler shape; but we keep (B,H_a,D_a) for denoiser
        x0 = actions.to(self.device)

        # sample random noise
        eps = torch.randn_like(x0, device=self.device)
        # sample random t for each sample
        t = torch.randint(0, self.scheduler.T, (B,), dtype=torch.long, device=self.device)
        # sample x_t
        xt = self.scheduler.q_sample(x0=x0.view(B, -1), t=t, noise=eps.view(B, -1)).view_as(x0)

        # condition
        cond_seq = self._cond_embed(obs)  # (B, H_o, d_model)

        # denoiser predicts eps
        eps_pred = self.denoiser(noisy_actions=xt, cond_seq=cond_seq, timesteps=t)

        loss = F.mse_loss(eps_pred, eps)

        diagnostics = {
            "loss": loss.item(),
            "t_mean": float(t.float().mean().item()),
        }

        # update EMA teacher if present
        if self.ema is not None and self.training:
            self.ema.update(self)

        return loss, diagnostics

    def forward(self, actions: torch.Tensor, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        alias to compute_loss for convenience in training loops.
        """
        return self.compute_loss(actions, obs)

    # -------------------------
    # Sampling / inference
    # -------------------------
    @torch.no_grad()
    def sample(self, obs: Dict[str, torch.Tensor], steps: Optional[int] = None, eta: float = 0.0, use_ema: bool = True, return_intermediates: bool = False) -> torch.Tensor:
        """
        Sample an action sequence conditioned on obs.

        obs: dict with keys same as _cond_embed
        steps: number of diffusion steps to run (<= scheduler.T). If None, uses scheduler.T.
        eta: DDIM stochasticity parameter (0.0 deterministic)
        use_ema: whether to use EMA teacher for denoiser (if available)
        return_intermediates: if True return list of intermediate x_t for debugging

        Returns: sampled_actions (B, H_a, D_a)
        """
        self.eval()
        # choose model for sampling
        model = None
        if use_ema and self.ema is not None:
            # use ema model for stability
            model = self.ema.ema_model
        else:
            model = self

        # prepare conditioning
        cond_seq = self._cond_embed(obs)  # (B, H_o, d_model)
        B = cond_seq.shape[0]
        device = cond_seq.device

        T = self.scheduler.T if steps is None else int(min(steps, self.scheduler.T))
        # build timesteps list for ddim (descending)
        timesteps = list(range(self.scheduler.T - 1, -1, -max(1, self.scheduler.T // T)))
        # initial sample x_T ~ N(0, I)
        x_t = torch.randn((B, self.H_a, self.action_dim), device=device)

        intermediates = []
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((B,), t, dtype=torch.long, device=device)
            eps_pred = model.denoiser(noisy_actions=x_t, cond_seq=cond_seq, timesteps=t_tensor)
            # if using ddim deterministic step
            t_next = timesteps[i + 1] if (i + 1) < len(timesteps) else 0
            x_t = self.scheduler.ddim_step(xt=x_t.view(B, -1), t=t, t_next=t_next, eps=eps_pred.view(B, -1), eta=eta)
            x_t = x_t.view(B, self.H_a, self.action_dim)
            if return_intermediates:
                intermediates.append(x_t.clone())

        # final predicted x0
        # ensure shape
        sampled = x_t
        self.train()  # reset training flag (no state changes done)
        if return_intermediates:
            return sampled, intermediates
        return sampled

    # -------------------------
    # Log-prob proxy (approximate)
    # -------------------------
    def log_prob_proxy(self, actions: torch.Tensor, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Approximate log-probability of actions under the learned policy.
        This is NOT an exact log-prob for diffusion models; it's a practical proxy:
        - Run a single-step denoising forward: compute predicted noise `eps_pred` at t=1..T,
          reconstruct x0 and compute gaussian log-likelihood under predicted residual variance.
        - Here we use a heuristic: assume unit variance and compute negative MSE as a proxy.
        Returns (B,) approximate log-prob (higher is better).
        """
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        # compute conditioning
        cond_seq = self._cond_embed(obs)
        B = actions.shape[0]
        # use mid timestep T//2 as proxy
        t = torch.full((B,), self.scheduler.T // 2, dtype=torch.long, device=self.device)
        xt = self.scheduler.q_sample(actions, t=t, noise=torch.randn_like(actions, device=self.device))
        eps_pred = self.denoiser(noisy_actions=xt, cond_seq=cond_seq, timesteps=t)
        mse = F.mse_loss(eps_pred, (xt - actions) / self.scheduler.sqrt_one_minus_alphas_cumprod[t].to(self.device).unsqueeze(-1), reduction='none')
        # reduce per sample
        per_sample_mse = mse.view(mse.shape[0], -1).mean(dim=1)
        logp_proxy = -per_sample_mse  # higher better
        return logp_proxy

    # -------------------------
    # Save / load utilities
    # -------------------------
    def save(self, path: str):
        """
        Save policy weights and optional EMA state.
        """
        payload = {
            "model_state": self.state_dict(),
            "scheduler_cfg": self.scheduler_cfg.__dict__,
        }
        if self.ema is not None:
            payload["ema_state"] = self.ema.state_dict()
            payload["ema_decay"] = self.ema.decay
        torch.save(payload, path)
        logger.info(f"Saved DiffusionPolicy to {path}")

    def load(self, path: str, map_location: Optional[str] = None):
        payload = torch.load(path, map_location=map_location)
        self.load_state_dict(payload["model_state"])
        if "ema_state" in payload and self.ema is not None:
            self.ema.load_state_dict(payload["ema_state"])
        logger.info(f"Loaded DiffusionPolicy from {path}")
