"""
diffusion_policy.py

Modular conditional diffusion policy for continuous-action robotic control.

Key features:
- PyTorch-based denoiser (plug-in backbone)
- Pluggable noise scheduler (linear / cosine)
- EMA teacher utility (enable/disable)
- Sampling: ddpm-step loop, plus deterministic one-step/low-step option (for RL exploitation)
- Approximate log-probability (Gaussian proxy) to integrate with PPO (see docstring)
- Clear type hints, docstrings, and extension points

Usage (high-level):
    policy = DiffusionPolicy(observation_dim=obs_dim, action_dim=act_dim, denoiser=MyDenoiser(), scheduler='linear')
    # pretraining: call policy.compute_loss(batch_actions, cond)
    # sampling (stochastic): a = policy.sample(cond, steps=50)
    # deterministic exploitation: a_det = policy.sample_deterministic(cond)  # uses fewer steps or DDIM-like path
    # approximate log-prob for PPO: lp = policy.log_prob_approx(a, cond)

Notes about log_prob_approx:
- Exact log-prob of diffusion policies is non-trivial. For integration into PPO (which needs
  a density or surrogate for the importance ratio), we provide a well-grounded *Gaussian proxy*:
  assume the denoiser produces a mean action mu(cond) and an (implicit) variance sigma^2 (derived
  from the noise schedule). Then we approximate log p(a|cond) ~ -0.5 * ||(a - mu)/sigma||^2 + const.
- This is a pragmatic choice used in several hybrid works. If you require exact likelihoods,
  consider an energy-based or flows-based wrapper (left as extension point).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Scheduler utilities
# ----------------------------
@dataclass
class NoiseScheduler:
    """
    Noise schedule handler.

    Attributes:
        betas: Tensor of shape (T,) with β_t.
        alphas: Tensor α_t = 1 - β_t.
        alpha_bars: Tensor of cumulative product \bar{α}_t.
        T: number of diffusion steps.
    """
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    T: int

    @staticmethod
    def linear(beta_start: float, beta_end: float, T: int, device: torch.device) -> "NoiseScheduler":
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return NoiseScheduler(betas=betas, alphas=alphas, alpha_bars=alpha_bars, T=T)

    @staticmethod
    def cosine(T: int, device: torch.device, s: float = 0.008) -> "NoiseScheduler":
        # Cosine schedule from Nichol & Dhariwal
        ts = torch.arange(0, T + 1, device=device, dtype=torch.float64) / T
        alphas_cum = torch.cos(((ts + s) / (1 + s)) * math.pi / 2) ** 2
        alpha_bars = alphas_cum[1:] / alphas_cum[0]
        # derive betas from alpha_bars
        alphas = torch.zeros_like(alpha_bars)
        betas = torch.zeros_like(alpha_bars)
        alphas[0] = alpha_bars[0]
        betas[0] = 1.0 - alphas[0]
        for t in range(1, T):
            alphas[t] = alpha_bars[t] / alpha_bars[t - 1]
            betas[t] = 1.0 - alphas[t]
        return NoiseScheduler(betas=betas.float(), alphas=alphas.float(), alpha_bars=alpha_bars.float(), T=T)


# ----------------------------
# EMA helper
# ----------------------------
class EMA:
    """
    Simple Exponential Moving Average for model weights.

    Typical usage:
        ema = EMA(model, decay=0.9999)
        ema.update(model)  # call after each optimizer.step()
        ema.store(); ema.copy_to(model);  # to evaluate ema weights
        ema.restore()
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999, device: Optional[torch.device] = None):
        self.decay = decay
        self.model = model
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self.device = device

        # initialize shadow to model params
        for name, p in model.state_dict().items():
            self.shadow[name] = p.detach().clone().to(device) if device is not None else p.detach().clone()

    def update(self, model: nn.Module):
        with torch.no_grad():
            for name, p in model.state_dict().items():
                assert name in self.shadow
                self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def store(self):
        self.backup = {k: v.clone() for k, v in self.model.state_dict().items()}

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow)

    def restore(self):
        if self.backup:
            self.model.load_state_dict(self.backup)
            self.backup = {}


# ----------------------------
# Base denoiser interface
# ----------------------------
class BaseDenoiser(nn.Module):
    """
    Minimal base interface that any denoiser backbone should implement.

    Must implement forward(x_t, t, cond) -> predicted_noise (epsilon).
    - x_t: noisy actions (B, action_dim)
    - t: integer timesteps (B,) or scalar
    - cond: conditioning data (dict or tensor). Typical: {'image': ..., 'proprio': ...}
    """
    def __init__(self):
        super().__init__()

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: Any) -> torch.Tensor:
        raise NotImplementedError("Implement forward(x_t, t, cond) returning predicted noise epsilon")


# ----------------------------
# Simple MLP denoiser (reference backbone)
# ----------------------------
class SimpleMLPDenoiser(BaseDenoiser):
    """
    A simple MLP denoiser for low-dimensional actions. Useful as a baseline and unit-testable.
    """
    def __init__(self, action_dim: int, cond_dim: int, hidden: int = 512, n_layers: int = 3):
        super().__init__()
        layers = []
        in_dim = action_dim + cond_dim + 1  # +1 for timestep scalar
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond is expected as (B, cond_dim)
        # t may be scalar or tensor; map to normalized scalar and concat
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x_t.shape[0])
        t_norm = (t.float() / 1e3).unsqueeze(-1)  # coarse normalization
        inp = torch.cat([x_t, cond, t_norm], dim=-1)
        return self.net(inp)


# ----------------------------
# Diffusion policy
# ----------------------------
class DiffusionPolicy(nn.Module):
    """
    DiffusionPolicy: wraps a denoiser + scheduler to provide training loss and sampling APIs.

    Args:
        denoiser: nn.Module implementing BaseDenoiser.forward(x_t, t, cond) -> predicted noise
        action_dim: int
        cond_embed_fn: Callable that maps conditioning dict -> tensor (B, cond_dim)
        scheduler: NoiseScheduler instance
        device: torch.device
        ema_decay: optional float; if provided, EMA object is created
    """
    def __init__(
        self,
        denoiser: BaseDenoiser,
        action_dim: int,
        cond_embed_fn: Callable[[Any], torch.Tensor],
        scheduler: NoiseScheduler,
        device: Optional[torch.device] = None,
        ema_decay: Optional[float] = None,
    ):
        super().__init__()
        self.denoiser = denoiser.to(device) if device is not None else denoiser
        self.action_dim = action_dim
        self.cond_embed_fn = cond_embed_fn
        self.scheduler = scheduler
        self.device = device or torch.device("cpu")
        self.register_buffer("_betas", scheduler.betas)
        self.register_buffer("_alphas", scheduler.alphas)
        self.register_buffer("_alpha_bars", scheduler.alpha_bars)
        self.T = scheduler.T

        self.ema: Optional[EMA] = None
        if ema_decay is not None and ema_decay > 0.0:
            self.ema = EMA(self, decay=ema_decay, device=device)

    # ----------------------
    # Training forward / loss
    # ----------------------
    def forward(self, a0: torch.Tensor, cond: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the diffusion training loss (standard denoising objective).

        Args:
            a0: clean actions tensor (B, action_dim), expected normalized (e.g., [-1,1]).
            cond: conditioning input passed to cond_embed_fn

        Returns:
            loss (mean over batch), per-sample loss tensor (B,)
        """
        B = a0.shape[0]
        device = self.device
        # Sample random timesteps
        t = torch.randint(0, self.T, (B,), device=device).long()
        betas = self._betas.to(device)[t]  # (B,)
        alpha_bars = self._alpha_bars.to(device)[t]  # (B,)
        sqrt_alpha_bars = torch.sqrt(alpha_bars).unsqueeze(-1)  # (B,1)
        sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bars).unsqueeze(-1)

        noise = torch.randn_like(a0, device=device)
        x_t = sqrt_alpha_bars * a0 + sqrt_one_minus_ab * noise  # noisy action
        cond_embed = self.cond_embed_fn(cond).to(device)  # (B, cond_dim)
        eps_pred = self.denoiser(x_t, t, cond_embed)
        per_sample = F.mse_loss(eps_pred, noise, reduction="none").mean(dim=-1)
        loss = per_sample.mean()
        return loss, per_sample.detach()

    # ----------------------
    # Sampling utilities
    # ----------------------
    @torch.no_grad()
    def sample(self, cond: Any, steps: int = 50, batch_size: Optional[int] = None, clipped: bool = True) -> torch.Tensor:
        """
        Stochastic sampling (DDPM-like) from p(a|cond).

        Args:
            cond: conditioning input (accepted by cond_embed_fn)
            steps: number of diffusion steps to run (<= self.T). Lower steps = faster, less fidelity.
            batch_size: optional; if cond already batched, ignore
            clipped: whether to clip final action in [-1,1]

        Returns:
            a0: sampled action tensor (B, action_dim)
        """
        device = self.device
        cond_embed = self.cond_embed_fn(cond).to(device)
        if isinstance(cond_embed, torch.Tensor):
            B = cond_embed.shape[0]
        else:
            raise ValueError("cond_embed_fn must return a Tensor shaped (B, cond_dim)")
        # Setup timesteps to iterate (linearly spaced over T)
        # Choose t_indices descending from T-1 -> 0 with 'steps' number
        if steps >= self.T:
            t_indices = torch.arange(self.T - 1, -1, -1, device=device)
        else:
            t_indices = torch.round(torch.linspace(self.T - 1, 0, steps)).long().to(device)

        x_t = torch.randn((B, self.action_dim), device=device)  # start from prior
        for t in t_indices:
            t_tensor = torch.full((B,), int(t.item()), device=device, dtype=torch.long)
            alpha_t = self._alphas[t].to(device)
            alpha_bar_t = self._alpha_bars[t].to(device)
            sqrt_recip_alpha = (1.0 / torch.sqrt(alpha_t)).to(device)
            # predict noise
            eps_pred = self.denoiser(x_t, t_tensor, cond_embed)
            # compute model mean μ_θ
            coef = (1 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
            mu = sqrt_recip_alpha * (x_t - coef * eps_pred)
            # variance for diffusion step (simple choice)
            if t > 0:
                beta_t = self._betas[t].to(device)
                sigma = torch.sqrt(beta_t).to(device)
                noise = torch.randn_like(x_t) * sigma
                x_prev = mu + noise
            else:
                x_prev = mu
            x_t = x_prev
        a0 = x_t
        if clipped:
            a0 = torch.clamp(a0, -1.0, 1.0)
        return a0.detach()

    @torch.no_grad()
    def sample_deterministic(self, cond: Any, steps: int = 10) -> torch.Tensor:
        """
        Deterministic sampling path (DDIM-ish): set noise to zero when possible.
        Useful for exploitation in RL where we want low-variance actions.

        Args:
            cond: conditioning input
            steps: number of steps (small, e.g., 1 or 10). If steps==1, this behaves like a single-step
                   denoising using a deterministic map (approx).
        """
        device = self.device
        cond_embed = self.cond_embed_fn(cond).to(device)
        B = cond_embed.shape[0]
        if steps >= self.T:
            t_indices = torch.arange(self.T - 1, -1, -1, device=device)
        else:
            t_indices = torch.round(torch.linspace(self.T - 1, 0, steps)).long().to(device)

        x_t = torch.randn((B, self.action_dim), device=device) * 0.001  # small init noise
        for t in t_indices:
            t_tensor = torch.full((B,), int(t.item()), device=device, dtype=torch.long)
            eps_pred = self.denoiser(x_t, t_tensor, cond_embed)
            sqrt_recip_alpha = 1.0 / torch.sqrt(self._alphas[int(t)].to(device))
            coef = (1 - self._alphas[int(t)].to(device)) / torch.sqrt(1.0 - self._alpha_bars[int(t)].to(device))
            mu = sqrt_recip_alpha * (x_t - coef * eps_pred)
            x_t = mu  # deterministic (no noise)
        a0 = torch.clamp(x_t, -1.0, 1.0)
        return a0.detach()

    # ----------------------
    # Approximate log-probability
    # ----------------------
    def log_prob_approx(self, a: torch.Tensor, cond: Any, sigma_floor: float = 1e-3) -> torch.Tensor:
        """
        Approximate log p(a | cond) by a Gaussian proxy centered at the one-step deterministic
        denoised action mu(cond), with variance derived from the scheduler's final step.

        Rationale:
          - exact diffusion marginal likelihood is intractable to compute efficiently for large models
          - using the final denoiser as a mean estimator and the scheduler-derived variance is
            a pragmatic approximation often used for RL integration (see comments in repo).

        Args:
            a: (B, action_dim)
            cond: conditioning input
            sigma_floor: minimal std to avoid numerical issues

        Returns:
            logp: (B,) approximated log probability (natural log)
        """
        device = self.device
        mu = self.sample_deterministic(cond, steps=1).to(device)  # one-step denoised mean
        # approximate variance: use average beta at t=0 or effective noise at final step
        beta_0 = float(self._betas[0].item())
        sigma = max(sigma_floor, math.sqrt(beta_0))
        var = sigma * sigma
        # compute Gaussian log-prob (elementwise)
        diff = (a.to(device) - mu).view(a.shape[0], -1)
        logp = -0.5 * (diff * diff).sum(dim=-1) / var
        # add normalization const
        D = a.shape[-1]
        logp = logp - 0.5 * D * math.log(2 * math.pi * var)
        return logp

    # ----------------------
    # EMA helpers
    # ----------------------
    def maybe_update_ema(self):
        if self.ema is not None:
            self.ema.update(self)

    def state_dict_for_save(self) -> Dict[str, Any]:
        # Save denoiser params and scheduler hyperparams
        return {
            "denoiser_state": self.denoiser.state_dict(),
            "action_dim": self.action_dim,
            "scheduler_T": self.T,
            "betas": self._betas.cpu().numpy(),
        }

    def load_state_dict_from_checkpoint(self, ckpt: Dict[str, Any]):
        self.denoiser.load_state_dict(ckpt["denoiser_state"])
