#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DKO Sec.6.1 — Visualize DL-DKO (trajectory dataset)

Plots:
1) Spectrum of learned A vs analytic eigenvalues.
2) Density comparison figure showing:
   pi, dec(enc(pi)), mu, dec(enc(mu)), dec(A enc(pi))
"""

from __future__ import annotations

import os
import math
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from scipy.stats import gaussian_kde

from dko_dl_sec6p1_model import DLDKOModel, DLDKOConfig, solve_A_ridge


def analytic_lambdas(k_list):
    k = np.asarray(k_list, dtype=int)
    return 1j * (1.0 - np.exp(1j * k)) / k


def empirical_density_hist(theta: np.ndarray, smooth_bins: int = 6, bins: int = 512):
    """Empirical density via histogram (optionally smoothed).

    This is fast but will look "blocky" compared to KDE.
    """
    hist, edges = np.histogram(theta, bins=bins, range=(0.0, 2.0 * np.pi), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    y = hist
    if smooth_bins and smooth_bins > 1:
        w = int(smooth_bins)
        ypad = np.r_[y[-w:], y, y[:w]]
        kernel = np.ones(2 * w + 1, dtype=float)
        kernel /= kernel.sum()
        ys = np.convolve(ypad, kernel, mode="same")[w:-w]
        y = ys
    return centers, y


def empirical_density_kde(theta: np.ndarray, grid: np.ndarray, bw_method=None):
    """Empirical density via *circular* KDE on [0, 2π).

    We periodize by duplicating samples at ±2π to avoid boundary artifacts.
    Returns density evaluated on the provided grid.
    """
    theta = np.asarray(theta, dtype=np.float64)
    theta = np.mod(theta, 2.0 * np.pi)
    theta_ext = np.concatenate([theta, theta - 2.0 * np.pi, theta + 2.0 * np.pi])
    kde = gaussian_kde(theta_ext, bw_method=bw_method)
    return kde(grid)


@torch.no_grad()
def model_density_on_grid(model: DLDKOModel, s: torch.Tensor, grid: torch.Tensor) -> np.ndarray:
    logp = model.decoder.log_prob(grid, s)
    return torch.exp(logp).detach().cpu().numpy()


@torch.no_grad()
def estimate_A_from_npz(
    model: DLDKOModel,
    X_traj: np.ndarray,
    indices: np.ndarray,
    time_index: int,
    horizon_pairs: int,
    ridge: float,
    device: torch.device,
    max_pairs: int = 20000,
) -> np.ndarray:
    """Estimate a single A by aggregating latent pairs over chosen indices."""
    L, K, m = X_traj.shape
    H = min(int(horizon_pairs), L - 1)
    t0 = int(time_index)
    t0 = max(0, min(t0, L - (H + 1)))

    S0s, S1s = [], []
    n_pairs = 0
    for j in indices:
        theta_seq = X_traj[t0:t0 + H + 1, :, j].astype(np.float32)  # (H+1,K)
        th = torch.from_numpy(theta_seq).to(device)
        s = model.encode(th)  # (H+1,n)
        s0 = s[:-1, :]
        s1 = s[1:, :]
        S0s.append(s0)
        S1s.append(s1)
        n_pairs += s0.shape[0]
        if n_pairs >= max_pairs:
            break
    if n_pairs == 0:
        raise RuntimeError("No pairs to estimate A.")
    S0 = torch.cat(S0s, dim=0)[:max_pairs]
    S1 = torch.cat(S1s, dim=0)[:max_pairs]
    A = solve_A_ridge(S0, S1, ridge=ridge)
    return A.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data_npz", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--kmax", type=int, default=12)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_val", type=int, default=1000)
    ap.add_argument("--n_test", type=int, default=2000)

    ap.add_argument("--time_index", type=int, default=0)
    ap.add_argument("--plot_examples", type=int, default=6)
    ap.add_argument("--grid_size", type=int, default=512)
    ap.add_argument("--ridge", type=float, default=None,
                    help="Ridge used to estimate A if checkpoint lacks A_global. Defaults to ckpt['ridge'] or 1e-6.")
    ap.add_argument("--A_pairs", type=int, default=3,
                    help="How many adjacent pairs per trajectory to use when estimating A (if needed).")

    # Empirical density rendering (controls whether the empirical curves look smooth)
    ap.add_argument("--empirical", type=str, default="kde", choices=["kde", "hist"],
                    help="How to plot empirical densities: 'kde' (smooth, circular) or 'hist' (binned).")
    ap.add_argument("--kde_bw", type=float, default=None,
                    help="Bandwidth for gaussian_kde (bw_method). Default None lets scipy choose.")
    ap.add_argument("--smooth_bins", type=int, default=6)
    ap.add_argument("--hist_bins", type=int, default=512)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg_dict = ckpt.get("cfg", None)
    if cfg_dict is None:
        raise ValueError("Checkpoint missing 'cfg'.")
    cfg = DLDKOConfig(**cfg_dict)

    model = DLDKOModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Data (needed to estimate A if not stored in checkpoint)
    data = np.load(args.data_npz, allow_pickle=True)
    if "X_traj" in data:
        X_traj = data["X_traj"]
    else:
        X_init = data["X_init"]; X_next = data["X_next"]
        X_traj = np.stack([X_init, X_next], axis=0)

    L, K, m = X_traj.shape

    rng = np.random.default_rng(args.split_seed)
    perm = rng.permutation(m)
    train_idx = perm[:args.n_train]
    val_idx = perm[args.n_train:args.n_train + args.n_val]
    test_idx = perm[args.n_train + args.n_val : args.n_train + args.n_val + args.n_test]

    # Obtain A for spectrum/inference.
    ridge = args.ridge
    if ridge is None:
        ridge = float(ckpt.get("ridge", 1e-6))

    if "A_global" in ckpt:
        A = np.asarray(ckpt["A_global"], dtype=np.float64)
    else:
        # Fallback: estimate A from train split.
        A = estimate_A_from_npz(
            model,
            X_traj=X_traj,
            indices=train_idx,
            time_index=args.time_index,
            horizon_pairs=args.A_pairs,
            ridge=float(ridge),
            device=device,
            max_pairs=20000,
        )

    eigs = np.linalg.eigvals(A)
    np.savetxt(os.path.join(args.out_dir, "eig_A_all.txt"), np.c_[eigs.real, eigs.imag],
               fmt="%.10f", header="Re\tIm")

    ks = [0] + list(range(1, args.kmax + 1)) + list(range(-1, -args.kmax - 1, -1))
    lam = np.zeros(len(ks), dtype=np.complex128)
    lam[0] = 1.0 + 0.0j
    lam[1:] = analytic_lambdas(ks[1:])

    fig = plt.figure(figsize=(6.2, 5.6), dpi=140)
    ax = plt.gca()
    tt = np.linspace(0, 2*np.pi, 512)
    ax.plot(np.cos(tt), np.sin(tt), alpha=0.5, linewidth=1.0)
    ax.scatter(eigs.real, eigs.imag, s=26, label="eig(A) — learned")
    ax.scatter(lam.real, lam.imag, s=60, marker="x", label=f"analytic λ_k (0, ±1..±{args.kmax})")
    ax.axhline(0, lw=0.8); ax.axvline(0, lw=0.8)
    ax.set_aspect("equal", "box"); ax.grid(True, ls="--", alpha=0.3)
    ax.set_xlabel("Re"); ax.set_ylabel("Im")
    ax.set_title(f"DL-DKO Sec.6.1 — spectrum of learned A (n={cfg.n}, Lmix={cfg.dec_L})")
    ax.legend(loc="best")
    spectrum_png = os.path.join(args.out_dir, "spectrum_A.png")
    plt.tight_layout(); plt.savefig(spectrum_png); plt.close(fig)

    t0 = int(args.time_index)
    if t0 < 0 or t0 + 1 >= L:
        raise ValueError(f"time_index must satisfy 0 <= t <= L-2. Got t={t0}, L={L}")

    n_ex = min(args.plot_examples, len(test_idx))

    # Use a *non-duplicated* circular grid (avoid having both 0 and 2π on the same plot).
    grid_np = np.linspace(0.0, 2.0 * math.pi, int(args.grid_size), endpoint=False)
    grid = torch.from_numpy(grid_np.astype(np.float32)).to(device)

    for ii in range(n_ex):
        j = int(test_idx[ii])

        th_pi = X_traj[t0, :, j].astype(np.float64)
        th_mu = X_traj[t0 + 1, :, j].astype(np.float64)

        if args.empirical == "kde":
            c_pi, d_pi = grid_np, empirical_density_kde(th_pi, grid_np, bw_method=args.kde_bw)
            c_mu, d_mu = grid_np, empirical_density_kde(th_mu, grid_np, bw_method=args.kde_bw)
        else:
            c_pi, d_pi = empirical_density_hist(th_pi, smooth_bins=args.smooth_bins, bins=args.hist_bins)
            c_mu, d_mu = empirical_density_hist(th_mu, smooth_bins=args.smooth_bins, bins=args.hist_bins)

        th_pi_t = torch.from_numpy(X_traj[t0, :, j].astype(np.float32)).to(device)
        th_mu_t = torch.from_numpy(X_traj[t0 + 1, :, j].astype(np.float32)).to(device)

        s_pi = model.encode(th_pi_t)
        s_mu = model.encode(th_mu_t)
        A_t = torch.from_numpy(A.astype(np.float32)).to(device)
        s_mu_pred = model.apply_A(s_pi, A_t)

        d_dec_pi = model_density_on_grid(model, s_pi, grid)
        d_dec_mu = model_density_on_grid(model, s_mu, grid)
        d_dec_pred = model_density_on_grid(model, s_mu_pred, grid)

        fig = plt.figure(figsize=(7.8, 3.9), dpi=140)
        ax = plt.gca()

        ax.plot(c_pi, d_pi, label="pi (empirical)", linewidth=1.5)
        ax.plot(grid_np, d_dec_pi, label="dec(enc(pi))", linewidth=1.5)

        ax.plot(c_mu, d_mu, label="mu (empirical)", linewidth=1.5)
        ax.plot(grid_np, d_dec_mu, label="dec(enc(mu))", linewidth=1.5)

        ax.plot(grid_np, d_dec_pred, "--", label="dec(A enc(pi)) (pred)", linewidth=1.5)

        ax.set_xlabel("theta"); ax.set_ylabel("density")
        ax.set_title(f"Test traj j={j} at t={t0} -> t+1")
        ax.grid(True, ls="--", alpha=0.3)
        ax.legend(loc="best")

        outp = os.path.join(args.out_dir, f"density_compare_ex{ii:02d}_traj{j}_t{t0}.png")
        plt.tight_layout(); plt.savefig(outp); plt.close(fig)

    with open(os.path.join(args.out_dir, "viz_report.json"), "w") as f:
        json.dump(
            dict(
                ckpt=args.ckpt,
                data_npz=args.data_npz,
                traj_shape=dict(L=int(L), K=int(K), m=int(m)),
                time_index=t0,
                n=int(cfg.n),
                mix_components=int(cfg.dec_L),
                kmax=int(args.kmax),
                spectrum_png=spectrum_png,
                examples=int(n_ex),
            ),
            f, indent=2
        )

    print(f"[saved] {spectrum_png}")
    print(f"[saved] eig_A_all.txt")
    if n_ex > 0:
        print(f"[saved] density_compare_*.png (n={n_ex})")


if __name__ == "__main__":
    main()
