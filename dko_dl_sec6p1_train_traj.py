#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DKO Sec.6.1 — Train DL-DKO with trajectory multi-step loss (A solved by differentiable ridge-LS)

Data:
  X_traj: (L, K, m) angles in radians

Loss (two-term, multi-step):
  L_lin  = mean_k w_k * MSE( A^k s_t , s_{t+k} )
  L_pred = mean_k w_k * NLL( theta_{t+k} | A^k s_t )

Key change:
  - A is NOT a trainable parameter.
  - At each optimization step, we solve A on the current batch by ridge least squares,
    using torch.linalg.solve (differentiable) with robust fallbacks.

Extra:
  - Logs CPU/GPU utilization (GPU via nvidia-smi).
  - Every --log_every steps: estimate A_global on val split and save spectrum plot
    (learned eigs vs analytic eigenvalues), so you can track spectral evolution.

"""

from __future__ import annotations

import os
import json
import math
import time
import argparse
import subprocess
from dataclasses import asdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dko_dl_sec6p1_model import DLDKOModel, DLDKOConfig


# -------------------------
# Optional CPU utilization
# -------------------------
try:
    import psutil  # type: ignore
except Exception:
    psutil = None


# -------------------------
# Dataset
# -------------------------
class TrajMeasuresDataset(Dataset):
    """Trajectory dataset: returns theta_traj of shape (L, K)."""

    def __init__(self, X_traj: np.ndarray, indices: np.ndarray):
        assert X_traj.ndim == 3  # (L,K,m)
        self.L, self.K, self.m = X_traj.shape
        # (N, L, K)
        self.X = X_traj[:, :, indices].transpose(2, 0, 1).astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


def split_indices(m: int, seed: int, n_train: int, n_val: int, n_test: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(m)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:n_train + n_val + n_test]
    return train_idx, val_idx, test_idx


def horizon_schedule(epoch: int, epochs: int, h_start: int, h_max: int, warmup_epochs: int) -> int:
    if h_max <= h_start:
        return int(h_max)
    if epoch <= warmup_epochs:
        return int(h_start)
    prog = (epoch - warmup_epochs) / max(1, (epochs - warmup_epochs))
    H = h_start + int(round(prog * (h_max - h_start)))
    return int(max(h_start, min(h_max, H)))


def select_score(val_metrics: dict, select_metric: str) -> float:
    if select_metric == "val_loss":
        return float(val_metrics["loss"])
    if select_metric == "val_pred":
        return float(val_metrics["pred"])
    if select_metric == "val_lin":
        return float(val_metrics["lin"])
    if select_metric == "val_pred+lin":
        return float(val_metrics["pred"] + val_metrics["lin"])
    raise ValueError(f"Unknown select_metric={select_metric}")


def print_device_banner(device: torch.device, amp: bool):
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[device] Using GPU: {name} | capability={cap} | total_mem={mem_gb:.1f} GB")
        print(f"[amp] AMP enabled: {bool(amp)} (torch.amp.autocast/GradScaler)")
    else:
        print("[device] Using CPU (no CUDA detected).")
        print("[amp] AMP enabled: False (CPU)")


# -------------------------
# Utilization logging
# -------------------------
def _nvidia_smi_util() -> tuple[float | None, float | None]:
    """
    Returns (gpu_util_percent, gpu_mem_percent_used) from nvidia-smi (best-effort).
    """
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        # If multiple GPUs, take first line.
        line = out.splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            return None, None
        gpu_util = float(parts[0])
        mem_used = float(parts[1])
        mem_total = float(parts[2])
        mem_pct = 100.0 * mem_used / max(1.0, mem_total)
        return gpu_util, mem_pct
    except Exception:
        return None, None


def get_utilization(device: torch.device) -> tuple[float | None, float | None, float | None]:
    """
    Returns (cpu_percent, gpu_util_percent, gpu_mem_percent_used).
    """
    cpu = None
    if psutil is not None:
        try:
            cpu = float(psutil.cpu_percent(interval=None))
        except Exception:
            cpu = None

    gpu = None
    gpu_mem = None
    if device.type == "cuda":
        gpu, gpu_mem = _nvidia_smi_util()

    return cpu, gpu, gpu_mem


# -------------------------
# Analytic spectrum (from your viz script)
# -------------------------
def analytic_lambdas(k_list):
    k = np.asarray(k_list, dtype=int)
    # careful: k=0 handled outside
    return 1j * (1.0 - np.exp(1j * k)) / k


# -------------------------
# Robust, differentiable A solve
# -------------------------
def solve_A_ridge_safe(S0: torch.Tensor, S1: torch.Tensor, ridge: float, max_tries: int = 6) -> torch.Tensor:
    """
    Solve A = argmin ||S1 - S0 A^T||_F^2 + ridge||A||_F^2 in a robust way.

    Here S0,S1 are (N,n) (row-major samples), and we return A of shape (n,n) in the convention:
      s_next ≈ s @ A^T  (same as your earlier code)
    """
    # Build normal equations for solve:
    #   G = S0^T S0  (n,n)
    #   C = S1^T S0  (n,n)  so that A = C * (G + ridge I)^{-1}
    n = S0.shape[1]
    G = S0.T @ S0
    C = S1.T @ S0
    I = torch.eye(n, device=S0.device, dtype=S0.dtype)

    r = float(ridge)
    last_err = None
    for _ in range(max_tries):
        try:
            # Solve (G + r I) X = C^T, then A = X^T
            A = torch.linalg.solve(G + r * I, C.T).T
            return A
        except Exception as e:
            last_err = e
            r = max(r * 10.0, 1e-12)

    # Fallback: pinv (always succeeds), still differentiable but less stable
    A = C @ torch.linalg.pinv(G + r * I)
    # If even this fails (rare), raise the original
    if A is None:
        raise last_err
    return A


# -------------------------
# Loss
# -------------------------
def compute_multistep_losses(
    model: DLDKOModel,
    theta_seq: torch.Tensor,  # (B, H+1, K)
    lambda_lin: float,
    lambda_pred: float,
    gamma: float,
    ridge: float,
) -> tuple[torch.Tensor, dict]:
    B, Hp1, K = theta_seq.shape
    H = Hp1 - 1

    # Encode all steps (teacher-forcing)
    flat = theta_seq.reshape(B * (H + 1), K)
    s_flat = model.encode(flat)  # (B*(H+1), n)
    n = s_flat.shape[-1]
    s = s_flat.reshape(B, H + 1, n)

    # Solve A on the batch using all adjacent pairs in the snippet
    S0 = s[:, :-1, :].reshape(B * H, n)
    S1 = s[:, 1:, :].reshape(B * H, n)

    A = solve_A_ridge_safe(S0, S1, ridge=ridge)  # (n,n)

    s_t = s[:, 0, :]
    s_true = s[:, 1:, :]  # (B,H,n)

    # Rollout under A
    s_pred_all = model.rollout_latent(s_t, A, H=H)  # (B,H,n)

    lin_acc = 0.0
    pred_acc = 0.0
    wsum = 0.0

    for k in range(1, H + 1):
        s_pred = s_pred_all[:, k - 1, :]
        w = (gamma ** (k - 1)) if gamma is not None else 1.0
        wsum += w

        lin_k = F.mse_loss(s_pred, s_true[:, k - 1, :])
        lin_acc = lin_acc + w * lin_k

        pred_k = model.decode_nll(theta_seq[:, k, :], s_pred, reduction="mean")
        pred_acc = pred_acc + w * pred_k

    lin = lin_acc / wsum
    pred = pred_acc / wsum
    loss = lambda_lin * lin + lambda_pred * pred
    return loss, dict(lin=float(lin.item()), pred=float(pred.item()))


@torch.no_grad()
def eval_epoch(
    model: DLDKOModel,
    loader: DataLoader,
    device: torch.device,
    horizon: int,
    start_time: int,
    lambda_lin: float,
    lambda_pred: float,
    gamma: float,
    ridge: float,
):
    model.eval()
    sums = dict(loss=0.0, lin=0.0, pred=0.0)
    n_batches = 0

    for theta_traj in loader:
        theta_traj = theta_traj.to(device, non_blocking=True)  # (B,L,K)
        B, L, K = theta_traj.shape
        H = min(horizon, L - 1)
        t0 = min(start_time, L - (H + 1))
        theta_seq = theta_traj[:, t0:t0 + H + 1, :]

        loss, mets = compute_multistep_losses(
            model, theta_seq,
            lambda_lin=lambda_lin, lambda_pred=lambda_pred, gamma=gamma, ridge=ridge
        )
        sums["loss"] += float(loss.item())
        sums["lin"] += float(mets["lin"])
        sums["pred"] += float(mets["pred"])
        n_batches += 1

    return {k: v / max(1, n_batches) for k, v in sums.items()}


# -------------------------
# A_global + spectrum plotting
# -------------------------
@torch.no_grad()
def estimate_A_global(
    model: DLDKOModel,
    loader: DataLoader,
    device: torch.device,
    start_time: int,
    ridge: float,
    max_batches: int,
) -> np.ndarray:
    """
    Estimate a single A_global by aggregating many (s_t, s_{t+1}) pairs from the loader.
    Uses t=start_time only (one-step), like your viz script uses a single A.
    """
    model.eval()
    S0_list = []
    S1_list = []
    nb = 0
    for theta_traj in loader:
        theta_traj = theta_traj.to(device, non_blocking=True)  # (B,L,K)
        B, L, K = theta_traj.shape
        t0 = min(start_time, L - 2)
        theta0 = theta_traj[:, t0, :]
        theta1 = theta_traj[:, t0 + 1, :]
        s0 = model.encode(theta0)  # (B,n)
        s1 = model.encode(theta1)  # (B,n)
        S0_list.append(s0)
        S1_list.append(s1)
        nb += 1
        if nb >= max_batches:
            break

    S0 = torch.cat(S0_list, dim=0)  # (N,n)
    S1 = torch.cat(S1_list, dim=0)  # (N,n)
    A = solve_A_ridge_safe(S0, S1, ridge=ridge)  # (n,n)
    return A.detach().cpu().numpy()


def save_spectrum_plot(A: np.ndarray, out_png: str, n: int, kmax: int):
    eigs = np.linalg.eigvals(A)

    ks = [0] + list(range(1, kmax + 1)) + list(range(-1, -kmax - 1, -1))
    lam = np.zeros(len(ks), dtype=np.complex128)
    lam[0] = 1.0 + 0.0j
    lam[1:] = analytic_lambdas(ks[1:])

    fig = plt.figure(figsize=(6.2, 5.6), dpi=140)
    ax = plt.gca()
    tt = np.linspace(0, 2*np.pi, 512)
    ax.plot(np.cos(tt), np.sin(tt), alpha=0.5, linewidth=1.0)
    ax.scatter(eigs.real, eigs.imag, s=26, label="eig(A_global) — LS")
    ax.scatter(lam.real, lam.imag, s=60, marker="x", label=f"analytic λ_k (0, ±1..±{kmax})")
    ax.axhline(0, lw=0.8); ax.axvline(0, lw=0.8)
    ax.set_aspect("equal", "box"); ax.grid(True, ls="--", alpha=0.3)
    ax.set_xlabel("Re"); ax.set_ylabel("Im")
    ax.set_title(f"DL-DKO Sec.6.1 — spectrum of A_global (n={n})")
    ax.legend(loc="best")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png)
    plt.close(fig)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_npz", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_val", type=int, default=1000)
    ap.add_argument("--n_test", type=int, default=2000)

    ap.add_argument("--latent_dim", type=int, default=12)

    # Encoder
    ap.add_argument("--enc_width", type=int, default=64)
    ap.add_argument("--enc_depth", type=int, default=3)
    ap.add_argument("--enc_activation", type=str, default="sine", choices=["relu", "tanh", "sine"])
    ap.add_argument("--enc_omega0", type=float, default=10.0)

    # Decoder
    ap.add_argument("--dec_components", type=int, default=8)
    ap.add_argument("--dec_hidden", type=int, default=64)
    ap.add_argument("--dec_depth", type=int, default=2)
    ap.add_argument("--dec_activation", type=str, default="relu", choices=["relu", "tanh", "sine"])

    # Optim
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_measures", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=0.0)
    ap.add_argument("--amp", action="store_true")

    # Loss weights
    ap.add_argument("--lambda_lin", type=float, default=1.0)
    ap.add_argument("--lambda_pred", type=float, default=5.0)
    ap.add_argument("--gamma", type=float, default=1.0)

    # Ridge for LS(A)
    ap.add_argument("--ridge", type=float, default=1e-6)

    # Horizon curriculum
    ap.add_argument("--horizon_start", type=int, default=1)
    ap.add_argument("--horizon_max", type=int, default=19)
    ap.add_argument("--horizon_warmup_epochs", type=int, default=50)

    # Evaluation
    ap.add_argument("--eval_horizon", type=int, default=19)
    ap.add_argument("--eval_start_time", type=int, default=0)
    ap.add_argument("--select_metric", type=str, default="val_pred+lin",
                    choices=["val_loss", "val_pred", "val_lin", "val_pred+lin"])

    # Dataloader
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--pin_memory", action="store_true")

    # Logging/saving
    ap.add_argument("--log_every", type=int, default=50, help="log every N epochs (epoch-based logging)")
    ap.add_argument("--save_every", type=int, default=500, help="save every N optimization steps")
    ap.add_argument("--kmax", type=int, default=12, help="analytic spectrum compare up to kmax")
    ap.add_argument("--A_global_batches", type=int, default=4, help="batches to estimate A_global on val split")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    data = np.load(args.data_npz, allow_pickle=True)
    if "X_traj" in data:
        X_traj = data["X_traj"]
    else:
        X_init = data["X_init"]; X_next = data["X_next"]
        X_traj = np.stack([X_init, X_next], axis=0)

    L, K, m = X_traj.shape
    horizon_max = min(args.horizon_max, L - 1)
    eval_horizon = min(args.eval_horizon, L - 1)
    if horizon_max < 1:
        raise ValueError(f"trajectory length L={L} too short (need L>=2)")

    train_idx, val_idx, test_idx = split_indices(m, args.split_seed, args.n_train, args.n_val, args.n_test)

    train_ds = TrajMeasuresDataset(X_traj, train_idx)
    val_ds = TrajMeasuresDataset(X_traj, val_idx)
    test_ds = TrajMeasuresDataset(X_traj, test_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    print_device_banner(device, amp_enabled)
    print(f"[data] X_traj shape: L={L}, K={K}, m={m} | splits: {args.n_train}/{args.n_val}/{args.n_test}")
    print(f"[horizon] train horizon_max={horizon_max}, eval_horizon={eval_horizon}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_measures,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(args.pin_memory and device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_measures,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(args.pin_memory and device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_measures,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(args.pin_memory and device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    cfg = DLDKOConfig(
        n=args.latent_dim,
        enc_width=args.enc_width,
        enc_depth=args.enc_depth,
        enc_activation=args.enc_activation,
        enc_omega0=args.enc_omega0,
        dec_L=args.dec_components,
        dec_hidden=args.dec_hidden,
        dec_depth=args.dec_depth,
        dec_activation=args.dec_activation,
    )
    model = DLDKOModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    run_cfg = dict(
        data_npz=args.data_npz,
        X_shape=dict(L=int(L), K=int(K), m=int(m)),
        splits=dict(train=args.n_train, val=args.n_val, test=args.n_test),
        split_seed=args.split_seed,
        model_cfg=asdict(cfg),
        loss=dict(lambda_lin=args.lambda_lin, lambda_pred=args.lambda_pred, gamma=args.gamma, ridge=args.ridge),
        horizon=dict(start=args.horizon_start, max=int(horizon_max), warmup=args.horizon_warmup_epochs,
                     eval_horizon=int(eval_horizon), eval_start_time=int(args.eval_start_time)),
        select_metric=args.select_metric,
        optim=dict(lr=args.lr, batch_measures=args.batch_measures, num_workers=args.num_workers,
                   pin_memory=bool(args.pin_memory), amp=amp_enabled),
        device=str(device),
    )
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)

    log_path = os.path.join(args.out_dir, "train_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("step,epoch,H,train_loss,train_lin,train_pred,val_loss,val_lin,val_pred,select_score,cpu,gpu,gpu_mem\n")

    best_score = float("inf")
    best_path = os.path.join(args.out_dir, "ckpt_best.pt")

    # Initial eval
    v0 = eval_epoch(model, val_loader, device, eval_horizon, args.eval_start_time,
                    args.lambda_lin, args.lambda_pred, args.gamma, args.ridge)
    s0 = select_score(v0, args.select_metric)
    print(f"[init] val: loss={v0['loss']:.6f} lin={v0['lin']:.6f} pred={v0['pred']:.6f} | select={s0:.6f}")

    rng = np.random.default_rng(args.split_seed + 12345)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        H = horizon_schedule(epoch, args.epochs, args.horizon_start, horizon_max, args.horizon_warmup_epochs)

        sums = dict(loss=0.0, lin=0.0, pred=0.0)
        n_batches = 0

        for theta_traj in train_loader:
            global_step += 1
            theta_traj = theta_traj.to(device, non_blocking=True)  # (B,L,K)
            B, Lb, Kb = theta_traj.shape

            max_start = L - (H + 1)
            t0 = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
            theta_seq = theta_traj[:, t0:t0 + H + 1, :]

            opt.zero_grad(set_to_none=True)

            try:
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    loss, mets = compute_multistep_losses(
                        model, theta_seq,
                        lambda_lin=args.lambda_lin,
                        lambda_pred=args.lambda_pred,
                        gamma=args.gamma,
                        ridge=args.ridge,
                    )
                scaler.scale(loss).backward()
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                scaler.step(opt)
                scaler.update()
            except torch._C._LinAlgError as e:
                # If LS solve fails catastrophically, skip this batch and continue training.
                print(f"[warn] step={global_step} | LinAlgError in solve: {e} | skipping batch")
                continue
            except Exception as e:
                # Do not crash training due to occasional numerical issues; skip batch.
                print(f"[warn] step={global_step} | exception: {type(e).__name__}: {e} | skipping batch")
                continue

            sums["loss"] += float(loss.item())
            sums["lin"] += float(mets["lin"])
            sums["pred"] += float(mets["pred"])
            n_batches += 1

            # Periodic raw checkpoint
            if (global_step % args.save_every) == 0:
                ckpt_path = os.path.join(args.out_dir, f"ckpt_step{global_step:07d}.pt")
                torch.save(dict(step=global_step, epoch=epoch, model_state=model.state_dict(), cfg=asdict(cfg)), ckpt_path)


        # ---- end of epoch: evaluate / select best (epoch-based logging) ----
        train_m = {k: v / max(1, n_batches) for k, v in sums.items()}

        # Always compute validation to track best checkpoint reliably
        val_m = eval_epoch(
            model, val_loader, device,
            horizon=eval_horizon,
            start_time=args.eval_start_time,
            lambda_lin=args.lambda_lin,
            lambda_pred=args.lambda_pred,
            gamma=args.gamma,
            ridge=args.ridge,
        )
        score = select_score(val_m, args.select_metric)

        # Best checkpoint (evaluate every epoch; save best)
        if score < best_score:
            best_score = score
            ckpt = dict(
                step=global_step,
                epoch=epoch,
                model_state=model.state_dict(),
                cfg=asdict(cfg),
                val_metrics=val_m,
                best_score=best_score,
                select_metric=args.select_metric,
            )
            # Save A_global best-effort (computed only on log epochs below)
            torch.save(ckpt, best_path)

        # Utilization snapshot
        cpu_u, gpu_u, gpu_mem_u = get_utilization(device)

        # CSV logging (epoch-based; one line per epoch)
        with open(log_path, "a") as f:
            f.write(
                f"{global_step},{epoch},{H},{train_m['loss']},{train_m['lin']},{train_m['pred']},"
                f"{val_m['loss']},{val_m['lin']},{val_m['pred']},{score},"
                f"{'' if cpu_u is None else cpu_u},{'' if gpu_u is None else gpu_u},{'' if gpu_mem_u is None else gpu_mem_u}\n"
            )

        # Only print + spectrum plot every N epochs
        if (epoch % args.log_every) == 0:
            util_str = []
            if cpu_u is not None:
                util_str.append(f"CPU={cpu_u:.0f}%")
            if gpu_u is not None:
                util_str.append(f"GPU={gpu_u:.0f}%")
            if gpu_mem_u is not None:
                util_str.append(f"GPUmem={gpu_mem_u:.0f}%")
            util_str = (" | " + " ".join(util_str)) if util_str else ""

            print(
                f"[ep {epoch:04d}] H={H:02d} | "
                f"train loss={train_m['loss']:.6f} lin={train_m['lin']:.6f} pred={train_m['pred']:.6f} | "
                f"val loss={val_m['loss']:.6f} lin={val_m['lin']:.6f} pred={val_m['pred']:.6f} | "
                f"select={score:.6f}{util_str}"
            )

            # Compute A_global on val split and save spectrum plot (epoch-indexed filename)
            try:
                A_global = estimate_A_global(
                    model, val_loader, device,
                    start_time=int(args.eval_start_time),
                    ridge=args.ridge,
                    max_batches=int(args.A_global_batches),
                )
                spec_dir = os.path.join(args.out_dir, "spectrum_trace")
                out_png = os.path.join(spec_dir, f"spectrum_ep{epoch:04d}.png")
                save_spectrum_plot(A_global, out_png, n=args.latent_dim, kmax=args.kmax)

                # Also update best checkpoint with A_global if this epoch is best
                if score <= best_score + 1e-12:
                    try:
                        ckpt = torch.load(best_path, map_location=device)
                        ckpt["A_global"] = A_global
                        torch.save(ckpt, best_path)
                    except Exception:
                        pass
            except Exception as e:
                print(f"[warn] ep={epoch:04d} | failed to compute/plot A_global spectrum: {e}")
        # end epoch loop over loader

    print(f"[done] best score={best_score:.6f} | ckpt: {best_path}")

    # Final eval (best)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    val_m = eval_epoch(model, val_loader, device, eval_horizon, args.eval_start_time,
                       args.lambda_lin, args.lambda_pred, args.gamma, args.ridge)
    test_m = eval_epoch(model, test_loader, device, eval_horizon, args.eval_start_time,
                        args.lambda_lin, args.lambda_pred, args.gamma, args.ridge)

    print(f"[best] eval_horizon={eval_horizon} | val pred={val_m['pred']:.6f} lin={val_m['lin']:.6f} | "
          f"test pred={test_m['pred']:.6f} lin={test_m['lin']:.6f}")

    with open(os.path.join(args.out_dir, "final_metrics.json"), "w") as f:
        json.dump(dict(val=val_m, test=test_m, best_score=best_score, select_metric=args.select_metric), f, indent=2)


if __name__ == "__main__":
    main()