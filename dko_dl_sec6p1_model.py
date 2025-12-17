#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DKO Sec.6.1 — Deep Distributional Koopman (DL-DKO)

Model = Encoder + (implicit) Koopman matrix A(s) + Decoder
- Encoder: learned pointwise MLP h_hat(x) -> R^n, aggregated by empirical expectation:
    s(mu) = E_{x~mu}[h_hat(x)] approx (1/K) sum_i h_hat(x_i)
  For circle angle theta, we embed x = (cos theta, sin theta) to enforce periodicity.

- Koopman latent dynamics: s_{t+1} ≈ A s_t, where A is solved on-the-fly
  via differentiable ridge least squares given latent pairs.

- Decoder: conditional density on circle as Mixture of von Mises:
    p(theta | s) = sum_{l=1}^L w_l(s) * vM(theta; mu_l(s), kappa_l(s))

Training typically uses NLL losses, which correspond to minimizing KL( empirical || model ).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def angle_to_unit(theta: torch.Tensor) -> torch.Tensor:
    """theta: (...,) in radians -> (...,2) = [cos, sin]."""
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)


class Sine(nn.Module):
    def __init__(self, omega0: float = 1.0):
        super().__init__()
        self.omega0 = float(omega0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega0 * x)


def siren_init_(layer: nn.Linear, is_first: bool, omega0: float):
    """SIREN initialization for stable sine networks."""
    with torch.no_grad():
        in_dim = layer.weight.size(1)
        if is_first:
            bound = 1.0 / in_dim
        else:
            bound = math.sqrt(6.0 / in_dim) / omega0
        layer.weight.uniform_(-bound, bound)
        if layer.bias is not None:
            layer.bias.uniform_(-bound, bound)


class PointMLP(nn.Module):
    """
    Pointwise MLP: R^2 -> R^n (circle input is [cos(theta), sin(theta)]).
    Default uses sine activations (SIREN-style) to approximate Fourier-like spans.
    """
    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 12,
        width: int = 64,
        depth: int = 3,
        activation: str = "sine",
        omega0: float = 10.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert depth >= 1
        self.activation_name = activation.lower()
        self.omega0 = float(omega0)

        layers = []
        last = in_dim
        for d in range(depth):
            lin = nn.Linear(last, width)
            layers.append(lin)
            if self.activation_name == "sine":
                layers.append(Sine(omega0=self.omega0))
            elif self.activation_name == "gelu":
                layers.append(nn.GELU())
            elif self.activation_name == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation={activation}. Choose from sine|gelu|tanh.")
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = width

        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

        if self.activation_name == "sine":
            # Apply SIREN init on all internal linear layers (excluding last projection).
            linears = [m for m in self.net.modules() if isinstance(m, nn.Linear)]
            for i, lin in enumerate(linears[:-1]):
                siren_init_(lin, is_first=(i == 0), omega0=self.omega0)
            # Last layer: small init helps stabilize early training.
            with torch.no_grad():
                linears[-1].weight.uniform_(-1e-3, 1e-3)
                if linears[-1].bias is not None:
                    linears[-1].bias.zero_()

    def forward(self, x2: torch.Tensor) -> torch.Tensor:
        return self.net(x2)


class MeanEncoder(nn.Module):
    """
    Encoder for empirical measures on S^1 given by K samples (angles in radians).
    Input: theta of shape (B, K) or (K,)
    Output: s of shape (B, n) or (n,)
    """
    def __init__(self, n: int = 12, width: int = 64, depth: int = 3, activation: str = "sine", omega0: float = 10.0):
        super().__init__()
        self.n = int(n)
        self.point_mlp = PointMLP(in_dim=2, out_dim=self.n, width=width, depth=depth,
                                  activation=activation, omega0=omega0)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.dim() == 1:
            x2 = angle_to_unit(theta)  # (K,2)
            h = self.point_mlp(x2)     # (K,n)
            return h.mean(dim=0)       # (n,)
        if theta.dim() == 2:
            B, K = theta.shape
            x2 = angle_to_unit(theta.reshape(-1))           # (B*K,2)
            h = self.point_mlp(x2).reshape(B, K, self.n)    # (B,K,n)
            return h.mean(dim=1)                            # (B,n)
        raise ValueError(f"theta must have dim 1 or 2, got {theta.shape}")


def solve_A_ridge(
    S0: torch.Tensor,
    S1: torch.Tensor,
    ridge: float = 1e-6,
) -> torch.Tensor:
    """Differentiable ridge least squares for A in: S1 ≈ A S0.

    Inputs
      S0: (N, n) latent at time t
      S1: (N, n) latent at time t+1

    Returns
      A: (n, n) such that A = (S1^T S0) (S0^T S0 + ridge I)^{-1}

    Notes
    - Uses torch.linalg.solve (preferred over explicit inverse).
    - Fully differentiable w.r.t. S0 and S1.
    """
    if S0.dim() != 2 or S1.dim() != 2:
        raise ValueError(f"S0 and S1 must be 2D (N,n). Got {S0.shape}, {S1.shape}")
    if S0.shape != S1.shape:
        raise ValueError(f"Shape mismatch: {S0.shape} vs {S1.shape}")
    N, n = S0.shape
    # Gram and cross-cov
    G = S0.T @ S0  # (n,n)
    C = S1.T @ S0  # (n,n)
    if ridge is None or ridge <= 0:
        A = torch.linalg.solve(G, C.T).T  # C * G^{-1}
        return A
    I = torch.eye(n, device=S0.device, dtype=S0.dtype)
    A = torch.linalg.solve(G + float(ridge) * I, C.T).T
    return A


class VonMisesMixtureDecoder(nn.Module):
    """
    Conditional density on circle using a mixture of von Mises distributions.

    Given s in R^n, outputs parameters for L components:
      weights w (softmax), means mu in (-pi,pi] (via atan2), concentrations kappa>0 (softplus)
    """
    def __init__(self, n: int = 12, L: int = 8, hidden: int = 64, depth: int = 2, activation: str = "gelu", kappa_min: float = 1e-3):
        super().__init__()
        self.n = int(n)
        self.L = int(L)
        self.kappa_min = float(kappa_min)

        acts = {"gelu": nn.GELU, "tanh": nn.Tanh, "relu": nn.ReLU}
        if activation.lower() not in acts:
            raise ValueError(f"decoder activation must be one of {list(acts)}")
        Act = acts[activation.lower()]

        layers = []
        last = self.n
        for _ in range(depth):
            layers.append(nn.Linear(last, hidden))
            layers.append(Act())
            last = hidden
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()

        self.head_logits = nn.Linear(last, self.L)
        self.head_mean2 = nn.Linear(last, 2 * self.L)
        self.head_kappa = nn.Linear(last, self.L)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _params(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if s.dim() == 1:
            s = s.unsqueeze(0)
        h = self.trunk(s)
        log_w = F.log_softmax(self.head_logits(h), dim=-1)      # (B,L)

        mean2 = self.head_mean2(h).reshape(-1, self.L, 2)       # (B,L,2)
        mean2 = mean2 / mean2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        mu = torch.atan2(mean2[..., 1], mean2[..., 0])          # (B,L)

        kappa = F.softplus(self.head_kappa(h)) + self.kappa_min
        return log_w, mu, kappa

    def log_prob(self, theta: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
            squeeze_out = True
        elif theta.dim() == 2:
            squeeze_out = False
        else:
            raise ValueError("theta must be (K,) or (B,K)")

        if s.dim() == 1:
            s = s.unsqueeze(0)

        B, K = theta.shape
        if s.shape[0] != B:
            raise ValueError(f"Batch mismatch: theta batch {B}, s batch {s.shape[0]}")

        log_w, mu, kappa = self._params(s)

        theta_b = theta.unsqueeze(1).expand(B, self.L, K)       # (B,L,K)
        mu_b = mu.unsqueeze(-1)                                  # (B,L,1)
        kappa_b = kappa.unsqueeze(-1)                            # (B,L,1)

        vm = torch.distributions.VonMises(loc=mu_b, concentration=kappa_b)
        log_comp = vm.log_prob(theta_b)                          # (B,L,K)

        log_mix = torch.logsumexp(log_w.unsqueeze(-1) + log_comp, dim=1)  # (B,K)
        return log_mix.squeeze(0) if squeeze_out else log_mix

    def nll(self, theta: torch.Tensor, s: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        nll = -self.log_prob(theta, s)
        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum()
        if reduction == "none":
            return nll
        raise ValueError("reduction must be mean|sum|none")

    @torch.no_grad()
    def sample(self, s: torch.Tensor, num_samples: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if s.dim() == 1:
            s = s.unsqueeze(0)
            squeeze_out = True
        else:
            squeeze_out = False

        log_w, mu, kappa = self._params(s)
        w = torch.exp(log_w)
        cat = torch.distributions.Categorical(probs=w)
        idx = cat.sample((num_samples,), generator=generator).T  # (B,num_samples)
        mu_sel = mu.gather(1, idx)
        kappa_sel = kappa.gather(1, idx)

        vm = torch.distributions.VonMises(loc=mu_sel, concentration=kappa_sel)
        th = vm.sample(generator=generator).remainder(2.0 * math.pi)
        return th.squeeze(0) if squeeze_out else th


@dataclass
class DLDKOConfig:
    n: int = 12
    enc_width: int = 64
    enc_depth: int = 3
    enc_activation: str = "sine"
    enc_omega0: float = 10.0
    dec_L: int = 8
    dec_hidden: int = 64
    dec_depth: int = 2
    dec_activation: str = "gelu"


class DLDKOModel(nn.Module):
    def __init__(self, cfg: DLDKOConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = MeanEncoder(n=cfg.n, width=cfg.enc_width, depth=cfg.enc_depth,
                                   activation=cfg.enc_activation, omega0=cfg.enc_omega0)
        self.decoder = VonMisesMixtureDecoder(n=cfg.n, L=cfg.dec_L, hidden=cfg.dec_hidden,
                                              depth=cfg.dec_depth, activation=cfg.dec_activation)

    def encode(self, theta: torch.Tensor) -> torch.Tensor:
        return self.encoder(theta)

    @staticmethod
    def apply_A(s: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Apply latent linear map. s: (B,n) or (n,), A: (n,n)."""
        if s.dim() == 1:
            return (A @ s).contiguous()
        return (s @ A.T).contiguous()

    @staticmethod
    def rollout_latent(s0: torch.Tensor, A: torch.Tensor, H: int) -> torch.Tensor:
        """Rollout s_{k+1}=A s_k for k=0..H-1.

        Returns: s_pred of shape (B,H,n) if s0 is (B,n).
        """
        if H < 1:
            raise ValueError("H must be >= 1")
        s = s0
        outs = []
        for _ in range(H):
            s = DLDKOModel.apply_A(s, A)
            outs.append(s)
        return torch.stack(outs, dim=1)

    def decode_nll(self, theta: torch.Tensor, s: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        return self.decoder.nll(theta, s, reduction=reduction)

    def forward(self, theta0: torch.Tensor, theta1: torch.Tensor):
        """Encode a one-step pair. (A is solved outside, in the training loop.)"""
        s0 = self.encode(theta0)
        s1 = self.encode(theta1)
        return s0, s1
