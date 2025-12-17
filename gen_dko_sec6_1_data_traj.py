#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate DKO Section 6.1 data — trajectories version

State space: circle parameterized by angle θ in [0, 2π).

Original (pair) construction:
- Partition the circle into M equal arcs; for each arc j:
  π_j = Uniform on arc j.
- One-step Markov pushforward kernel:
  θ_{t+1} = (θ_t + ν + ω_t) mod 2π,
  with ν = 1/2 and ω_t ~ Uniform[-1/2, 1/2] (radians), i.i.d. across particles and time.

This script can generate:
1) m_traj trajectories, each a length-L sequence of empirical measures with K particles.
   Output array X_traj with shape (L, K, m_traj).
2) For backward compatibility, we also save X_init = X_traj[0] and X_next = X_traj[1].

Usage (recommended):
  python3 gen_dko_sec6_1_data_traj.py --m_traj 7000 --K 1000 --traj_len 20 --seed 0 --out_dir outputs_sec6p1

This will create:
  outputs_sec6p1/sec6p1_traj_m7000_L20_K1000_seed0.npz
"""

import os
import argparse
import numpy as np


def sample_arc(j: int, M: int, K: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform samples on the j-th arc of M equal arcs over [0, 2π)."""
    a = 2.0 * np.pi * (j / M)
    b = 2.0 * np.pi * ((j + 1) / M)
    return rng.uniform(a, b, size=K)


def generate_trajectories_sec6p1(
    m_traj: int,
    traj_len: int,
    K: int,
    seed: int = 0,
    nu: float = 0.5,
    omega_low: float = -0.5,
    omega_high: float = 0.5,
    M_arcs: int | None = None,
    dtype=np.float32,
):
    """
    Returns:
      X_traj: (L, K, m_traj) angles in radians
      arc_idx: (m_traj,) arc index used to initialize each trajectory
    """
    rng = np.random.default_rng(seed)
    L = int(traj_len)
    M = int(M_arcs) if M_arcs is not None else int(m_traj)
    X = np.empty((L, K, m_traj), dtype=dtype)
    arc_idx = np.empty((m_traj,), dtype=np.int32)

    for i in range(m_traj):
        j = i % M
        arc_idx[i] = j
        theta = sample_arc(j, M, K, rng).astype(dtype, copy=False)
        X[0, :, i] = theta

        for t in range(1, L):
            omega = rng.uniform(omega_low, omega_high, size=K).astype(dtype, copy=False)
            theta = (theta + dtype(nu) + omega) % dtype(2.0 * np.pi)
            X[t, :, i] = theta

    return X, arc_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m_traj", type=int, default=7000, help="number of trajectories")
    ap.add_argument("--traj_len", type=int, default=20, help="trajectory length L (>=2)")
    ap.add_argument("--K", type=int, default=1000, help="samples per empirical measure")
    ap.add_argument("--seed", type=int, default=0, help="random seed")

    ap.add_argument("--nu", type=float, default=0.5, help="deterministic drift ν (radians)")
    ap.add_argument("--omega_low", type=float, default=-0.5, help="omega uniform low")
    ap.add_argument("--omega_high", type=float, default=0.5, help="omega uniform high")

    ap.add_argument("--M_arcs", type=int, default=None, help="number of arcs for initialization; default=m_traj")
    ap.add_argument("--out_dir", type=str, default="outputs_sec6p1")
    ap.add_argument("--prefix", type=str, default=None)
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    args = ap.parse_args()

    if args.traj_len < 2:
        raise ValueError("--traj_len must be >= 2")

    os.makedirs(args.out_dir, exist_ok=True)
    prefix = args.prefix or f"sec6p1_traj_m{args.m_traj}_L{args.traj_len}_K{args.K}_seed{args.seed}"
    npz_path = os.path.join(args.out_dir, f"{prefix}.npz")

    dtype = np.float32 if args.dtype == "float32" else np.float64

    X_traj, arc_idx = generate_trajectories_sec6p1(
        m_traj=args.m_traj,
        traj_len=args.traj_len,
        K=args.K,
        seed=args.seed,
        nu=args.nu,
        omega_low=args.omega_low,
        omega_high=args.omega_high,
        M_arcs=args.M_arcs,
        dtype=dtype,
    )

    # Compatibility keys (pair)
    X_init = X_traj[0, :, :]      # (K, m_traj)
    X_next = X_traj[1, :, :]      # (K, m_traj)

    np.savez_compressed(
        npz_path,
        X_traj=X_traj,
        X_init=X_init,
        X_next=X_next,
        arc_idx=arc_idx,
        meta=dict(
            m_traj=int(args.m_traj),
            traj_len=int(args.traj_len),
            K=int(args.K),
            seed=int(args.seed),
            nu=float(args.nu),
            omega_low=float(args.omega_low),
            omega_high=float(args.omega_high),
            M_arcs=int(args.M_arcs) if args.M_arcs is not None else int(args.m_traj),
            note="X_traj has shape (L,K,m_traj). X_init=X_traj[0], X_next=X_traj[1].",
        ),
    )

    print(f"[OK] Generated trajectories: m_traj={args.m_traj}, L={args.traj_len}, K={args.K}, seed={args.seed}")
    print(f"     Saved: {npz_path}")
    print(f"     Keys: X_traj (L,K,m), X_init (K,m), X_next (K,m), arc_idx (m), meta")


if __name__ == "__main__":
    main()
