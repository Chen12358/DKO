# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# DKO Sec.6.1 — Train Deep Distributional Koopman (DL-DKO) with trajectory multi-step loss

# Expected data:
# - New trajectory format (recommended):
#     X_traj: (L, K, m)  angles in radians
#   (This is produced by gen_dko_sec6_1_data_traj.py.)
# - Backward compatibility:
#     X_init, X_next: (K, m)  (treated as L=2)

# Splits (by trajectory index after shuffling): train=4000, val=1000, test=2000.

# Training objective (two-term):
# - Multi-step latent consistency:
#     L_lin = mean_{k=1..H} w_k * MSE( A^k s_t, s_{t+k} )
# - Multi-step predictive distribution likelihood:
#     L_pred = mean_{k=1..H} w_k * NLL( θ_{t+k} | A^k s_t )

# Key change vs the older script:
# - A is NOT a trainable parameter. At each step we solve A by differentiable
#   ridge least squares on the current batch latent pairs.

# Curriculum on horizon:
# - During training, horizon H increases gradually up to horizon_max.
# """

# from __future__ import annotations

# import os
# import json
# import argparse
# from dataclasses import asdict

# import numpy as np

# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader

# from dko_dl_sec6p1_model import DLDKOModel, DLDKOConfig, solve_A_ridge

# try:
#     import psutil  # type: ignore
# except Exception:
#     psutil = None

# try:
#     from pynvml import (  # type: ignore
#         nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex,
#         nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo
#     )
# except Exception:
#     nvmlInit = nvmlShutdown = nvmlDeviceGetHandleByIndex = None
#     nvmlDeviceGetUtilizationRates = nvmlDeviceGetMemoryInfo = None


# class TrajMeasuresDataset(Dataset):
#     """Trajectory dataset: returns theta_traj of shape (L, K)."""
#     def __init__(self, X_traj: np.ndarray, indices: np.ndarray):
#         assert X_traj.ndim == 3  # (L,K,m)
#         L, K, m = X_traj.shape
#         self.L = L
#         self.K = K
#         self.X = X_traj[:, :, indices].transpose(2, 0, 1).astype(np.float32)  # (N,L,K)

#     def __len__(self):
#         return self.X.shape[0]

#     def __getitem__(self, idx):
#         return self.X[idx]  # (L,K)


# def split_indices(m: int, seed: int, n_train: int, n_val: int, n_test: int):
#     rng = np.random.default_rng(seed)
#     perm = rng.permutation(m)
#     train_idx = perm[:n_train]
#     val_idx = perm[n_train:n_train + n_val]
#     test_idx = perm[n_train + n_val:n_train + n_val + n_test]
#     return train_idx, val_idx, test_idx


# def horizon_schedule(epoch: int, epochs: int, h_start: int, h_max: int, warmup_epochs: int) -> int:
#     if h_max <= h_start:
#         return int(h_max)
#     if epoch <= warmup_epochs:
#         return int(h_start)
#     prog = (epoch - warmup_epochs) / max(1, (epochs - warmup_epochs))
#     H = h_start + int(round(prog * (h_max - h_start)))
#     return int(max(h_start, min(h_max, H)))


# def select_score(val_metrics: dict, select_metric: str) -> float:
#     if select_metric == "val_loss":
#         return float(val_metrics["loss"])
#     if select_metric == "val_pred":
#         return float(val_metrics["pred"])
#     if select_metric == "val_lin":
#         return float(val_metrics["lin"])
#     if select_metric == "val_pred+lin":
#         return float(val_metrics["pred"] + val_metrics["lin"])
#     raise ValueError(f"Unknown select_metric={select_metric}")


# def print_device_banner(device: torch.device, amp: bool):
#     if device.type == "cuda":
#         name = torch.cuda.get_device_name(0)
#         cap = torch.cuda.get_device_capability(0)
#         mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
#         print(f"[device] Using GPU: {name} | capability={cap} | total_mem={mem_gb:.1f} GB")
#         print(f"[amp] AMP enabled: {bool(amp)} (torch.amp.autocast/GradScaler)")
#     else:
#         print("[device] Using CPU (no CUDA detected).")
#         print("[amp] AMP enabled: False (CPU)")


# def get_utilization(device: torch.device):
#     """Best-effort CPU/GPU utilization snapshot for logging."""
#     cpu = None
#     if psutil is not None:
#         try:
#             cpu = float(psutil.cpu_percent(interval=None))
#         except Exception:
#             cpu = None

#     gpu = None
#     gpu_mem = None
#     if device.type == "cuda" and nvmlInit is not None:
#         try:
#             nvmlInit()
#             h = nvmlDeviceGetHandleByIndex(0)
#             util = nvmlDeviceGetUtilizationRates(h)
#             mem = nvmlDeviceGetMemoryInfo(h)
#             gpu = float(util.gpu)
#             gpu_mem = float(100.0 * mem.used / max(1, mem.total))
#             nvmlShutdown()
#         except Exception:
#             try:
#                 nvmlShutdown()
#             except Exception:
#                 pass
#             gpu = None
#             gpu_mem = None

#     return cpu, gpu, gpu_mem


# def compute_multistep_losses(
#     model: DLDKOModel,
#     theta_seq: torch.Tensor,  # (B, H+1, K)
#     lambda_lin: float,
#     lambda_pred: float,
#     gamma: float,
#     ridge: float,
# ) -> tuple[torch.Tensor, dict]:
#     B, Hp1, K = theta_seq.shape
#     H = Hp1 - 1

#     flat = theta_seq.reshape(B * (H + 1), K)
#     s_flat = model.encode(flat)  # (B*(H+1), n)
#     n = s_flat.shape[-1]
#     s = s_flat.reshape(B, H + 1, n)

#     # Build a batch LS problem using *all* adjacent pairs in the snippet.
#     #   S0 = [s_t, s_{t+1}, ..., s_{t+H-1}]
#     #   S1 = [s_{t+1}, ..., s_{t+H}]
#     S0 = s[:, :-1, :].reshape(B * H, n)
#     S1 = s[:, 1:, :].reshape(B * H, n)
#     A = solve_A_ridge(S0, S1, ridge=ridge)  # (n,n), differentiable

#     s_t = s[:, 0, :]
#     s_true = s[:, 1:, :]  # (B,H,n)

#     lin_acc = 0.0
#     pred_acc = 0.0
#     wsum = 0.0

#     # Multi-step rollout under the batch-wise solved A.
#     s_pred_all = model.rollout_latent(s_t, A, H=H)  # (B,H,n)

#     for k in range(1, H + 1):
#         s_pred = s_pred_all[:, k - 1, :]

#         w = (gamma ** (k - 1)) if gamma is not None else 1.0
#         wsum += w

#         lin_k = F.mse_loss(s_pred, s_true[:, k - 1, :])
#         lin_acc = lin_acc + w * lin_k

#         pred_k = model.decode_nll(theta_seq[:, k, :], s_pred, reduction="mean")
#         pred_acc = pred_acc + w * pred_k

#     lin = lin_acc / wsum
#     pred = pred_acc / wsum

#     loss = lambda_lin * lin + lambda_pred * pred

#     return loss, dict(lin=float(lin.item()), pred=float(pred.item()))


# @torch.no_grad()
# def eval_epoch(
#     model: DLDKOModel,
#     loader: DataLoader,
#     device: torch.device,
#     horizon: int,
#     start_time: int,
#     lambda_lin: float,
#     lambda_pred: float,
#     gamma: float,
#     ridge: float,
# ):
#     model.eval()
#     sums = dict(loss=0.0, lin=0.0, pred=0.0)
#     n_batches = 0

#     for theta_traj in loader:
#         theta_traj = theta_traj.to(device, non_blocking=True)  # (B,L,K)
#         B, L, K = theta_traj.shape

#         H = min(horizon, L - 1)
#         t0 = min(start_time, L - (H + 1))

#         theta_seq = theta_traj[:, t0:t0 + H + 1, :]
#         loss, mets = compute_multistep_losses(
#             model, theta_seq,
#             lambda_lin=lambda_lin, lambda_pred=lambda_pred, gamma=gamma, ridge=ridge
#         )

#         sums["loss"] += float(loss.item())
#         sums["lin"] += float(mets["lin"])
#         sums["pred"] += float(mets["pred"])
#         n_batches += 1

#     return {k: v / max(1, n_batches) for k, v in sums.items()}


# @torch.no_grad()
# def estimate_A_global(
#     model: DLDKOModel,
#     loader: DataLoader,
#     device: torch.device,
#     horizon_pairs: int,
#     start_time: int,
#     ridge: float,
#     max_pairs: int = 20000,
# ) -> np.ndarray:
#     """Estimate a single A on a dataset split by ridge-LS on encoded pairs.

#     We aggregate pairs (s_t, s_{t+1}) across many trajectories to reduce variance.
#     This A is used for spectrum visualization and checkpoint metadata (not for training).
#     """
#     model.eval()
#     S0s = []
#     S1s = []
#     n_pairs = 0
#     for theta_traj in loader:
#         theta_traj = theta_traj.to(device, non_blocking=True)  # (B,L,K)
#         B, L, K = theta_traj.shape
#         H = min(int(horizon_pairs), L - 1)
#         t0 = min(int(start_time), L - (H + 1))
#         theta_seq = theta_traj[:, t0:t0 + H + 1, :]  # (B,H+1,K)

#         flat = theta_seq.reshape(B * (H + 1), K)
#         s_flat = model.encode(flat)
#         n = s_flat.shape[-1]
#         s = s_flat.reshape(B, H + 1, n)
#         s0 = s[:, :-1, :].reshape(B * H, n)
#         s1 = s[:, 1:, :].reshape(B * H, n)
#         S0s.append(s0)
#         S1s.append(s1)
#         n_pairs += s0.shape[0]
#         if n_pairs >= max_pairs:
#             break

#     if n_pairs == 0:
#         raise RuntimeError("Could not accumulate any pairs to estimate A.")
#     S0 = torch.cat(S0s, dim=0)[:max_pairs]
#     S1 = torch.cat(S1s, dim=0)[:max_pairs]
#     A = solve_A_ridge(S0, S1, ridge=ridge)
#     return A.detach().cpu().numpy()


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--data_npz", type=str, required=True)
#     ap.add_argument("--out_dir", type=str, required=True)

#     ap.add_argument("--n_train", type=int, default=4000)
#     ap.add_argument("--n_val", type=int, default=1000)
#     ap.add_argument("--n_test", type=int, default=2000)
#     ap.add_argument("--split_seed", type=int, default=0)

#     # model
#     ap.add_argument("--latent_dim", type=int, default=12)
#     ap.add_argument("--enc_width", type=int, default=64)
#     ap.add_argument("--enc_depth", type=int, default=3)
#     ap.add_argument("--enc_activation", type=str, default="sine", choices=["sine", "gelu", "tanh"])
#     ap.add_argument("--enc_omega0", type=float, default=10.0)

#     ap.add_argument("--ridge", type=float, default=1e-6,
#                     help="Ridge term for batch-wise LS solve of A: (S0^T S0 + ridge I)^{-1}.")

#     ap.add_argument("--dec_components", type=int, default=8)
#     ap.add_argument("--dec_hidden", type=int, default=64)
#     ap.add_argument("--dec_depth", type=int, default=2)
#     ap.add_argument("--dec_activation", type=str, default="gelu", choices=["gelu", "tanh", "relu"])

#     # optimization
#     ap.add_argument("--epochs", type=int, default=4000)
#     ap.add_argument("--batch_measures", type=int, default=512)
#     ap.add_argument("--lr", type=float, default=5e-4)
#     ap.add_argument("--weight_decay", type=float, default=0.0)
#     ap.add_argument("--grad_clip", type=float, default=1.0)
#     ap.add_argument("--amp", action="store_true")

#     # data loader perf
#     ap.add_argument("--num_workers", type=int, default=4)
#     ap.add_argument("--pin_memory", action="store_true")

#     # prediction-first weights
#     ap.add_argument("--lambda_lin", type=float, default=1.0)
#     ap.add_argument("--lambda_pred", type=float, default=5.0)
#     ap.add_argument("--gamma", type=float, default=1.0)

#     # horizon curriculum
#     ap.add_argument("--horizon_start", type=int, default=1)
#     ap.add_argument("--horizon_max", type=int, default=19)
#     ap.add_argument("--horizon_warmup_epochs", type=int, default=50)

#     # evaluation
#     ap.add_argument("--eval_horizon", type=int, default=19)
#     ap.add_argument("--eval_start_time", type=int, default=0)

#     # checkpoint selection
#     ap.add_argument("--select_metric", type=str, default="val_pred+lin",
#                     choices=["val_loss", "val_pred", "val_lin", "val_pred+lin"])

#     # logging/saving
#     ap.add_argument("--log_every", type=int, default=50)
#     ap.add_argument("--save_every", type=int, default=500)

#     args = ap.parse_args()
#     os.makedirs(args.out_dir, exist_ok=True)

#     data = np.load(args.data_npz, allow_pickle=True)
#     if "X_traj" in data:
#         X_traj = data["X_traj"]
#     else:
#         X_init = data["X_init"]
#         X_next = data["X_next"]
#         X_traj = np.stack([X_init, X_next], axis=0)

#     if X_traj.dtype != np.float32:
#         X_traj = X_traj.astype(np.float32, copy=False)

#     L, K, m = X_traj.shape

#     if args.n_train + args.n_val + args.n_test > m:
#         raise ValueError(f"split sizes exceed m={m}")

#     horizon_max = min(args.horizon_max, L - 1)
#     eval_horizon = min(args.eval_horizon, L - 1)
#     if horizon_max < 1:
#         raise ValueError(f"trajectory length L={L} too short (need L>=2)")

#     train_idx, val_idx, test_idx = split_indices(m, args.split_seed, args.n_train, args.n_val, args.n_test)

#     train_ds = TrajMeasuresDataset(X_traj, train_idx)
#     val_ds = TrajMeasuresDataset(X_traj, val_idx)
#     test_ds = TrajMeasuresDataset(X_traj, test_idx)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     amp_enabled = bool(args.amp and device.type == "cuda")
#     print_device_banner(device, amp_enabled)
#     print(f"[data] X_traj shape: L={L}, K={K}, m={m} | splits: {args.n_train}/{args.n_val}/{args.n_test}")
#     print(f"[horizon] train horizon_max={horizon_max}, eval_horizon={eval_horizon}")

#     train_loader = DataLoader(
#         train_ds,
#         batch_size=args.batch_measures,
#         shuffle=True,
#         drop_last=True,
#         num_workers=args.num_workers,
#         pin_memory=(args.pin_memory and device.type == "cuda"),
#         persistent_workers=(args.num_workers > 0),
#     )
#     val_loader = DataLoader(
#         val_ds,
#         batch_size=args.batch_measures,
#         shuffle=False,
#         drop_last=False,
#         num_workers=args.num_workers,
#         pin_memory=(args.pin_memory and device.type == "cuda"),
#         persistent_workers=(args.num_workers > 0),
#     )
#     test_loader = DataLoader(
#         test_ds,
#         batch_size=args.batch_measures,
#         shuffle=False,
#         drop_last=False,
#         num_workers=args.num_workers,
#         pin_memory=(args.pin_memory and device.type == "cuda"),
#         persistent_workers=(args.num_workers > 0),
#     )

#     cfg = DLDKOConfig(
#         n=args.latent_dim,
#         enc_width=args.enc_width,
#         enc_depth=args.enc_depth,
#         enc_activation=args.enc_activation,
#         enc_omega0=args.enc_omega0,
#         dec_L=args.dec_components,
#         dec_hidden=args.dec_hidden,
#         dec_depth=args.dec_depth,
#         dec_activation=args.dec_activation,
#     )
#     model = DLDKOModel(cfg).to(device)
#     opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#     scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

#     run_cfg = dict(
#         data_npz=args.data_npz,
#         X_shape=dict(L=int(L), K=int(K), m=int(m)),
#         splits=dict(train=args.n_train, val=args.n_val, test=args.n_test),
#         split_seed=args.split_seed,
#         model_cfg=asdict(cfg),
#         weights=dict(lambda_lin=args.lambda_lin, lambda_pred=args.lambda_pred, gamma=args.gamma, ridge=args.ridge),
#         horizon=dict(start=args.horizon_start, max=int(horizon_max), warmup=args.horizon_warmup_epochs,
#                      eval_horizon=int(eval_horizon), eval_start_time=int(args.eval_start_time)),
#         select_metric=args.select_metric,
#         optim=dict(lr=args.lr, batch_measures=args.batch_measures, num_workers=args.num_workers,
#                    pin_memory=bool(args.pin_memory), amp=amp_enabled),
#         device=str(device),
#     )
#     with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
#         json.dump(run_cfg, f, indent=2)

#     log_path = os.path.join(args.out_dir, "train_log.csv")
#     if not os.path.exists(log_path):
#         with open(log_path, "w") as f:
#             f.write("epoch,H,train_loss,train_lin,train_pred,val_loss,val_lin,val_pred,select_score,cpu_pct,gpu_pct,gpu_mem_pct\n")

#     best_score = float("inf")
#     best_path = os.path.join(args.out_dir, "ckpt_best.pt")

#     v0 = eval_epoch(model, val_loader, device, eval_horizon, args.eval_start_time,
#                     args.lambda_lin, args.lambda_pred, args.gamma, ridge=args.ridge)
#     s0 = select_score(v0, args.select_metric)
#     print(f"[init] val: loss={v0['loss']:.6f} lin={v0['lin']:.6f} pred={v0['pred']:.6f} | select={s0:.6f}")

#     rng = np.random.default_rng(args.split_seed + 12345)

#     for epoch in range(1, args.epochs + 1):
#         model.train()

#         H = horizon_schedule(epoch, args.epochs, args.horizon_start, horizon_max, args.horizon_warmup_epochs)
#         sums = dict(loss=0.0, lin=0.0, pred=0.0)
#         n_batches = 0

#         for theta_traj in train_loader:
#             theta_traj = theta_traj.to(device, non_blocking=True)  # (B,L,K)
#             B, Lb, Kb = theta_traj.shape

#             max_start = L - (H + 1)
#             t0 = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
#             theta_seq = theta_traj[:, t0:t0 + H + 1, :]

#             opt.zero_grad(set_to_none=True)

#             with torch.amp.autocast("cuda", enabled=amp_enabled):
#                 loss, mets = compute_multistep_losses(
#                     model, theta_seq,
#                     lambda_lin=args.lambda_lin,
#                     lambda_pred=args.lambda_pred,
#                     gamma=args.gamma,
#                     ridge=args.ridge,
#                 )

#             scaler.scale(loss).backward()
#             if args.grad_clip and args.grad_clip > 0:
#                 scaler.unscale_(opt)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
#             scaler.step(opt)
#             scaler.update()

#             sums["loss"] += float(loss.item())
#             sums["lin"] += float(mets["lin"])
#             sums["pred"] += float(mets["pred"])
#             n_batches += 1

#         train_m = {k: v / max(1, n_batches) for k, v in sums.items()}

#         if epoch % args.log_every == 0 or epoch == 1:
#             val_m = eval_epoch(model, val_loader, device, eval_horizon, args.eval_start_time,
#                                args.lambda_lin, args.lambda_pred, args.gamma, ridge=args.ridge)
#             score = select_score(val_m, args.select_metric)

#             cpu_pct, gpu_pct, gpu_mem_pct = get_utilization(device)

#             util_str = ""
#             if cpu_pct is not None:
#                 util_str += f" | CPU={cpu_pct:.1f}%"
#             if gpu_pct is not None:
#                 util_str += f" | GPU={gpu_pct:.1f}%"
#             if gpu_mem_pct is not None:
#                 util_str += f" | GPUmem={gpu_mem_pct:.1f}%"

#             print(
#                 f"[ep {epoch:04d}] H={H:02d} | "
#                 f"train loss={train_m['loss']:.6f} lin={train_m['lin']:.6f} pred={train_m['pred']:.6f} | "
#                 f"val loss={val_m['loss']:.6f} lin={val_m['lin']:.6f} pred={val_m['pred']:.6f} | "
#                 f"select={score:.6f}{util_str}"
#             )

#             with open(log_path, "a") as f:
#                 f.write(
#                     f"{epoch},{H},{train_m['loss']},{train_m['lin']},{train_m['pred']},"
#                     f"{val_m['loss']},{val_m['lin']},{val_m['pred']},{score},"
#                     f"{'' if cpu_pct is None else cpu_pct},"
#                     f"{'' if gpu_pct is None else gpu_pct},"
#                     f"{'' if gpu_mem_pct is None else gpu_mem_pct}\n"
#                 )

#             if score < best_score:
#                 best_score = score
#                 A_global = estimate_A_global(
#                     model, val_loader, device,
#                     horizon_pairs=min(3, eval_horizon),
#                     start_time=args.eval_start_time,
#                     ridge=args.ridge,
#                     max_pairs=20000,
#                 )
#                 torch.save(
#                     dict(
#                         epoch=epoch,
#                         model_state=model.state_dict(),
#                         cfg=asdict(cfg),
#                         val_metrics=val_m,
#                         best_score=best_score,
#                         select_metric=args.select_metric,
#                         A_global=A_global,
#                         ridge=float(args.ridge),
#                         H=int(H),
#                         eval_horizon=int(eval_horizon),
#                     ),
#                     best_path,
#                 )

#         if epoch % args.save_every == 0:
#             A_global = estimate_A_global(
#                 model, val_loader, device,
#                 horizon_pairs=min(3, eval_horizon),
#                 start_time=args.eval_start_time,
#                 ridge=args.ridge,
#                 max_pairs=20000,
#             )
#             torch.save(
#                 dict(
#                     epoch=epoch,
#                     model_state=model.state_dict(),
#                     cfg=asdict(cfg),
#                     A_global=A_global,
#                     ridge=float(args.ridge),
#                     H=int(H),
#                 ),
#                 os.path.join(args.out_dir, f"ckpt_ep{epoch:04d}.pt"),
#             )

#     print(f"[done] best score={best_score:.6f} | ckpt: {best_path}")

#     ckpt = torch.load(best_path, map_location=device)
#     model.load_state_dict(ckpt["model_state"])

#     val_m = eval_epoch(model, val_loader, device, eval_horizon, args.eval_start_time,
#                        args.lambda_lin, args.lambda_pred, args.gamma, ridge=args.ridge)
#     test_m = eval_epoch(model, test_loader, device, eval_horizon, args.eval_start_time,
#                         args.lambda_lin, args.lambda_pred, args.gamma, ridge=args.ridge)

#     print(f"[best] eval_horizon={eval_horizon} | val pred={val_m['pred']:.6f} lin={val_m['lin']:.6f} | "
#           f"test pred={test_m['pred']:.6f} lin={test_m['lin']:.6f}")

#     with open(os.path.join(args.out_dir, "final_metrics.json"), "w") as f:
#         json.dump(dict(val=val_m, test=test_m, best_score=best_score, select_metric=args.select_metric), f, indent=2)


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DKO Sec.6.1 — Two-stage training with differentiable ridge-LS A

Stage 1 (epochs_stage1):
  - Train encoder + decoder
  - At each batch, solve A* by differentiable ridge least squares using all adjacent pairs in the snippet
  - Loss = lambda_lin * multi-step latent MSE + lambda_pred * multi-step predictive NLL (rollout latent)

Stage 2 (epochs_stage2):
  - Freeze encoder
  - Compute a GLOBAL A_global after Stage 1 (ridge-LS on many training pairs)
  - Train decoder only, with fixed A_global
  - Loss = lambda_pred * multi-step predictive NLL (rollout latent)
    (No latent MSE term needed because A is fixed and encoder frozen.)

Logging:
  - CPU utilization via psutil if available
  - GPU utilization via nvidia-smi (always attempted on CUDA)

NLL subsampling:
  - nll_subsample (train): subsample K points per measure when computing NLL
  - nll_subsample_eval (eval): default uses full K unless set smaller
"""

from __future__ import annotations

import os
import json
import argparse
import subprocess
from dataclasses import asdict
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dko_dl_sec6p1_model import DLDKOModel, DLDKOConfig, solve_A_ridge


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
        L, K, m = X_traj.shape
        self.L = L
        self.K = K
        self.X = X_traj[:, :, indices].transpose(2, 0, 1).astype(np.float32)  # (N,L,K)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]  # (L,K)


def split_indices(m: int, seed: int, n_train: int, n_val: int, n_test: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(m)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:n_train + n_val + n_test]
    return train_idx, val_idx, test_idx


# -------------------------
# Schedules / utilities
# -------------------------
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


def query_gpu_util_nvidia_smi() -> Tuple[Optional[float], Optional[float], Optional[int], Optional[int]]:
    """
    Returns (gpu_util%, mem_util%, mem_used_MiB, mem_total_MiB) for GPU 0 (best-effort).
    Uses nvidia-smi; returns (None, None, None, None) on failure.
    """
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8").strip().splitlines()
        if len(out) == 0:
            return None, None, None, None
        # If multiple GPUs, take the first visible one.
        parts = [p.strip() for p in out[0].split(",")]
        gpu_util = float(parts[0])
        mem_util = float(parts[1])
        mem_used = int(float(parts[2]))
        mem_total = int(float(parts[3]))
        return gpu_util, mem_util, mem_used, mem_total
    except Exception:
        return None, None, None, None


def get_utilization_snapshot(device: torch.device):
    cpu = None
    if psutil is not None:
        try:
            cpu = float(psutil.cpu_percent(interval=None))
        except Exception:
            cpu = None

    gpu = gpu_mem = None
    mem_used = mem_total = None
    if device.type == "cuda":
        gpu, gpu_mem, mem_used, mem_total = query_gpu_util_nvidia_smi()
    return cpu, gpu, gpu_mem, mem_used, mem_total


def subsample_theta(theta: torch.Tensor, k_sub: int, rng: torch.Generator) -> torch.Tensor:
    """
    theta: (..., K)
    returns (..., k_sub) with uniform subsampling without replacement.
    """
    K = theta.shape[-1]
    if (k_sub is None) or (k_sub <= 0) or (k_sub >= K):
        return theta
    idx = torch.randperm(K, generator=rng, device=theta.device)[:k_sub]
    return theta.index_select(dim=-1, index=idx)


# -------------------------
# Loss computations
# -------------------------
@torch.no_grad()
def encode_snippet(model: DLDKOModel, theta_seq: torch.Tensor) -> torch.Tensor:
    """
    theta_seq: (B, H+1, K)
    returns s: (B, H+1, n)
    """
    B, Hp1, K = theta_seq.shape
    flat = theta_seq.reshape(B * Hp1, K)
    s_flat = model.encode(flat)  # (B*Hp1, n)
    n = s_flat.shape[-1]
    return s_flat.reshape(B, Hp1, n)


def compute_stage1_losses(
    model: DLDKOModel,
    theta_seq: torch.Tensor,          # (B, H+1, K)
    lambda_lin: float,
    lambda_pred: float,
    gamma: float,
    ridge: float,
    nll_subsample: int,
    rng: torch.Generator,
    amp_enabled: bool,
) -> Tuple[torch.Tensor, dict]:
    """
    Stage 1:
      - Solve batch A* by ridge-LS on all adjacent pairs in the snippet (differentiable)
      - Multi-step latent MSE + multi-step predictive NLL using rollout latent
    """
    B, Hp1, K = theta_seq.shape
    H = Hp1 - 1

    # Encode (with grad)
    flat = theta_seq.reshape(B * (H + 1), K)
    s_flat = model.encode(flat)  # (B*(H+1), n)
    n = s_flat.shape[-1]
    s = s_flat.reshape(B, H + 1, n)

    # Solve A* using all adjacent pairs in the snippet
    S0 = s[:, :-1, :].reshape(B * H, n)
    S1 = s[:, 1:, :].reshape(B * H, n)
    A = solve_A_ridge(S0, S1, ridge=ridge)  # (n,n), differentiable

    s_t = s[:, 0, :]
    s_true = s[:, 1:, :]  # (B,H,n)

    # Rollout
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

        theta_k = theta_seq[:, k, :]  # (B,K)
        theta_k = subsample_theta(theta_k, nll_subsample, rng=rng)
        pred_k = model.decode_nll(theta_k, s_pred, reduction="mean")
        pred_acc = pred_acc + w * pred_k

    lin = lin_acc / wsum
    pred = pred_acc / wsum
    loss = lambda_lin * lin + lambda_pred * pred
    return loss, dict(lin=float(lin.item()), pred=float(pred.item()))


def compute_stage2_losses(
    model: DLDKOModel,
    theta_seq: torch.Tensor,          # (B, H+1, K)
    A_global: torch.Tensor,           # (n,n)
    lambda_pred: float,
    gamma: float,
    nll_subsample: int,
    rng: torch.Generator,
) -> Tuple[torch.Tensor, dict]:
    """
    Stage 2:
      - Encoder frozen (no grad)
      - A_global fixed
      - Train decoder only: multi-step predictive NLL using rollout latent
    """
    B, Hp1, K = theta_seq.shape
    H = Hp1 - 1

    with torch.no_grad():
        s = encode_snippet(model, theta_seq)  # (B,H+1,n)
        s_t = s[:, 0, :]

    s_pred_all = model.rollout_latent(s_t, A_global, H=H)  # (B,H,n)

    pred_acc = 0.0
    wsum = 0.0
    for k in range(1, H + 1):
        s_pred = s_pred_all[:, k - 1, :]
        w = (gamma ** (k - 1)) if gamma is not None else 1.0
        wsum += w

        theta_k = theta_seq[:, k, :]
        theta_k = subsample_theta(theta_k, nll_subsample, rng=rng)
        pred_k = model.decode_nll(theta_k, s_pred, reduction="mean")
        pred_acc = pred_acc + w * pred_k

    pred = pred_acc / wsum
    loss = lambda_pred * pred
    return loss, dict(pred=float(pred.item()))


@torch.no_grad()
def eval_epoch_stage1(
    model: DLDKOModel,
    loader: DataLoader,
    device: torch.device,
    eval_horizon: int,
    eval_start_time: int,
    lambda_lin: float,
    lambda_pred: float,
    gamma: float,
    ridge: float,
    nll_subsample_eval: int,
) -> dict:
    model.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(1234)

    sums = dict(loss=0.0, lin=0.0, pred=0.0)
    n_batches = 0

    for theta_traj in loader:
        theta_traj = theta_traj.to(device, non_blocking=True)  # (B,L,K)
        B, L, K = theta_traj.shape

        t0 = int(eval_start_time)
        H = int(min(eval_horizon, L - 1))
        theta_seq = theta_traj[:, t0:t0 + H + 1, :]

        # Encode (no grad) but A is solved differentiably in stage1; for eval it's fine no grad.
        flat = theta_seq.reshape(B * (H + 1), K)
        s_flat = model.encode(flat)
        n = s_flat.shape[-1]
        s = s_flat.reshape(B, H + 1, n)

        S0 = s[:, :-1, :].reshape(B * H, n)
        S1 = s[:, 1:, :].reshape(B * H, n)
        A = solve_A_ridge(S0, S1, ridge=ridge)

        s_t = s[:, 0, :]
        s_true = s[:, 1:, :]
        s_pred_all = model.rollout_latent(s_t, A, H=H)

        lin_acc = 0.0
        pred_acc = 0.0
        wsum = 0.0
        for k in range(1, H + 1):
            s_pred = s_pred_all[:, k - 1, :]
            w = (gamma ** (k - 1)) if gamma is not None else 1.0
            wsum += w

            lin_acc = lin_acc + w * F.mse_loss(s_pred, s_true[:, k - 1, :])

            theta_k = subsample_theta(theta_seq[:, k, :], nll_subsample_eval, rng=rng)
            pred_acc = pred_acc + w * model.decode_nll(theta_k, s_pred, reduction="mean")

        lin = lin_acc / wsum
        pred = pred_acc / wsum
        loss = lambda_lin * lin + lambda_pred * pred

        sums["loss"] += float(loss.item())
        sums["lin"] += float(lin.item())
        sums["pred"] += float(pred.item())
        n_batches += 1

    return {k: v / max(1, n_batches) for k, v in sums.items()}


@torch.no_grad()
def eval_epoch_stage2(
    model: DLDKOModel,
    loader: DataLoader,
    device: torch.device,
    eval_horizon: int,
    eval_start_time: int,
    A_global: torch.Tensor,
    lambda_pred: float,
    gamma: float,
    nll_subsample_eval: int,
) -> dict:
    model.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(5678)

    sums = dict(loss=0.0, pred=0.0)
    n_batches = 0

    for theta_traj in loader:
        theta_traj = theta_traj.to(device, non_blocking=True)
        B, L, K = theta_traj.shape

        t0 = int(eval_start_time)
        H = int(min(eval_horizon, L - 1))
        theta_seq = theta_traj[:, t0:t0 + H + 1, :]

        s = encode_snippet(model, theta_seq)
        s_t = s[:, 0, :]
        s_pred_all = model.rollout_latent(s_t, A_global, H=H)

        pred_acc = 0.0
        wsum = 0.0
        for k in range(1, H + 1):
            s_pred = s_pred_all[:, k - 1, :]
            w = (gamma ** (k - 1)) if gamma is not None else 1.0
            wsum += w
            theta_k = subsample_theta(theta_seq[:, k, :], nll_subsample_eval, rng=rng)
            pred_acc = pred_acc + w * model.decode_nll(theta_k, s_pred, reduction="mean")

        pred = pred_acc / wsum
        loss = lambda_pred * pred
        sums["loss"] += float(loss.item())
        sums["pred"] += float(pred.item())
        n_batches += 1

    return {k: v / max(1, n_batches) for k, v in sums.items()}


@torch.no_grad()
def compute_A_global(
    model: DLDKOModel,
    loader: DataLoader,
    device: torch.device,
    ridge: float,
    max_pairs: int,
    horizon_for_pairs: int,
    start_time: int,
) -> torch.Tensor:
    """
    Compute a global A on a (potentially large) set of (s_t, s_{t+1}) pairs collected from loader.
    We collect adjacent pairs from a fixed snippet length (horizon_for_pairs+1) starting at start_time.
    """
    model.eval()
    S0_list = []
    S1_list = []
    total = 0

    for theta_traj in loader:
        theta_traj = theta_traj.to(device, non_blocking=True)  # (B,L,K)
        B, L, K = theta_traj.shape

        H = int(min(horizon_for_pairs, L - 1))
        t0 = int(min(max(0, start_time), L - (H + 1)))
        theta_seq = theta_traj[:, t0:t0 + H + 1, :]  # (B,H+1,K)

        s = encode_snippet(model, theta_seq)  # (B,H+1,n)
        n = s.shape[-1]

        # collect all adjacent pairs in this snippet
        S0 = s[:, :-1, :].reshape(B * H, n)
        S1 = s[:, 1:, :].reshape(B * H, n)

        S0_list.append(S0)
        S1_list.append(S1)
        total += S0.shape[0]
        if total >= max_pairs:
            break

    S0_all = torch.cat(S0_list, dim=0)[:max_pairs, :]
    S1_all = torch.cat(S1_list, dim=0)[:max_pairs, :]
    A_global = solve_A_ridge(S0_all, S1_all, ridge=ridge)  # (n,n)
    return A_global


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_npz", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_val", type=int, default=1000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--split_seed", type=int, default=0)

    # model
    ap.add_argument("--latent_dim", type=int, default=12)
    ap.add_argument("--enc_width", type=int, default=64)
    ap.add_argument("--enc_depth", type=int, default=3)
    ap.add_argument("--enc_activation", type=str, default="sine", choices=["sine", "gelu", "tanh"])
    ap.add_argument("--enc_omega0", type=float, default=10.0)

    ap.add_argument("--dec_components", type=int, default=8)
    ap.add_argument("--dec_hidden", type=int, default=64)
    ap.add_argument("--dec_depth", type=int, default=2)
    ap.add_argument("--dec_activation", type=str, default="gelu", choices=["gelu", "tanh", "relu"])

    # optimization
    ap.add_argument("--epochs_stage1", type=int, default=300)
    ap.add_argument("--epochs_stage2", type=int, default=200)
    ap.add_argument("--batch_measures", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr_stage2", type=float, default=None)  # default: same as lr
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true")

    # ridge for LS
    ap.add_argument("--ridge", type=float, default=1e-6)

    # NLL subsampling
    ap.add_argument("--nll_subsample", type=int, default=1000)
    ap.add_argument("--nll_subsample_eval", type=int, default=1000)

    # data loader perf
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--pin_memory", action="store_true")

    # weights
    ap.add_argument("--lambda_lin", type=float, default=1.0)
    ap.add_argument("--lambda_pred", type=float, default=5.0)
    ap.add_argument("--gamma", type=float, default=1.0)

    # horizon curriculum (stage1)
    ap.add_argument("--horizon_start", type=int, default=1)
    ap.add_argument("--horizon_max", type=int, default=19)
    ap.add_argument("--horizon_warmup_epochs", type=int, default=50)

    # evaluation
    ap.add_argument("--eval_horizon", type=int, default=19)
    ap.add_argument("--eval_start_time", type=int, default=0)

    # A_global estimation
    ap.add_argument("--A_global_pairs", type=int, default=200000)
    ap.add_argument("--A_global_horizon", type=int, default=19)
    ap.add_argument("--A_global_start_time", type=int, default=0)

    # checkpoint selection (stage1 selection)
    ap.add_argument("--select_metric", type=str, default="val_pred+lin",
                    choices=["val_loss", "val_pred", "val_lin", "val_pred+lin"])

    # logging/saving
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=500)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------------- data ----------------
    data = np.load(args.data_npz, allow_pickle=True)
    if "X_traj" in data:
        X_traj = data["X_traj"]
    else:
        # fallback compatibility
        X_init = data["X_init"]
        X_next = data["X_next"]
        X_traj = np.stack([X_init, X_next], axis=0)

    if X_traj.dtype != np.float32:
        X_traj = X_traj.astype(np.float32, copy=False)

    L, K, m = X_traj.shape
    if args.n_train + args.n_val + args.n_test > m:
        raise ValueError(f"split sizes exceed m={m}")

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
    print(f"[nll] subsample train={args.nll_subsample} eval={args.nll_subsample_eval}")
    print(f"[ridge] ridge={args.ridge:g}")

    pin = (args.pin_memory and device.type == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_measures,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_measures,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_measures,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=(args.num_workers > 0),
    )

    # ---------------- model ----------------
    cfg = DLDKOConfig(
        n=args.latent_dim,
        enc_width=args.enc_width,
        enc_depth=args.enc_depth,
        enc_activation=args.enc_activation,
        enc_omega0=args.enc_omega0,
        # A_init="identity",  # ignored by LS; kept for config completeness
        dec_L=args.dec_components,
        dec_hidden=args.dec_hidden,
        dec_depth=args.dec_depth,
        dec_activation=args.dec_activation,
    )
    model = DLDKOModel(cfg).to(device)

    # Stage-1 optimizer: encoder + decoder
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    run_cfg = dict(
        data_npz=args.data_npz,
        X_shape=dict(L=int(L), K=int(K), m=int(m)),
        splits=dict(train=args.n_train, val=args.n_val, test=args.n_test),
        split_seed=args.split_seed,
        model_cfg=asdict(cfg),
        weights=dict(lambda_lin=args.lambda_lin, lambda_pred=args.lambda_pred, gamma=args.gamma),
        ridge=float(args.ridge),
        nll=dict(train=int(args.nll_subsample), eval=int(args.nll_subsample_eval)),
        horizon=dict(start=args.horizon_start, max=int(horizon_max), warmup=args.horizon_warmup_epochs,
                     eval_horizon=int(eval_horizon), eval_start_time=int(args.eval_start_time)),
        A_global=dict(pairs=int(args.A_global_pairs), horizon=int(args.A_global_horizon), start_time=int(args.A_global_start_time)),
        optim=dict(lr=args.lr, lr_stage2=(args.lr_stage2 if args.lr_stage2 is not None else args.lr),
                   batch_measures=args.batch_measures, num_workers=args.num_workers,
                   pin_memory=bool(args.pin_memory), amp=amp_enabled),
        device=str(device),
    )
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)

    log_path = os.path.join(args.out_dir, "train_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write(
                "stage,epoch,H,train_loss,train_lin,train_pred,val_loss,val_lin,val_pred,select_score,"
                "cpu_util,gpu_util,gpu_mem_util,gpu_mem_used,gpu_mem_total\n"
            )

    best_score = float("inf")
    best_path = os.path.join(args.out_dir, "ckpt_best.pt")
    best_stage1_path = os.path.join(args.out_dir, "ckpt_best_stage1.pt")

    rng_np = np.random.default_rng(args.split_seed + 12345)
    rng_torch = torch.Generator(device=device)
    rng_torch.manual_seed(args.split_seed + 999)

    # -------------------------
    # Stage 1
    # -------------------------
    print(f"[stage1] epochs={args.epochs_stage1}")
    init_val = eval_epoch_stage1(
        model, val_loader, device,
        eval_horizon=eval_horizon,
        eval_start_time=args.eval_start_time,
        lambda_lin=args.lambda_lin,
        lambda_pred=args.lambda_pred,
        gamma=args.gamma,
        ridge=args.ridge,
        nll_subsample_eval=args.nll_subsample_eval,
    )
    init_score = select_score(init_val, args.select_metric)
    print(f"[init] val: loss={init_val['loss']:.6f} lin={init_val['lin']:.6f} pred={init_val['pred']:.6f} | select={init_score:.6f}")

    for epoch in range(1, args.epochs_stage1 + 1):
        model.train()
        H = horizon_schedule(epoch, args.epochs_stage1, args.horizon_start, horizon_max, args.horizon_warmup_epochs)

        sums = dict(loss=0.0, lin=0.0, pred=0.0)
        n_batches = 0

        for theta_traj in train_loader:
            theta_traj = theta_traj.to(device, non_blocking=True)  # (B,L,K)
            B, Lb, Kb = theta_traj.shape

            max_start = L - (H + 1)
            t0 = int(rng_np.integers(0, max_start + 1)) if max_start > 0 else 0
            theta_seq = theta_traj[:, t0:t0 + H + 1, :]

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss, mets = compute_stage1_losses(
                    model, theta_seq,
                    lambda_lin=args.lambda_lin,
                    lambda_pred=args.lambda_pred,
                    gamma=args.gamma,
                    ridge=args.ridge,
                    nll_subsample=args.nll_subsample,
                    rng=rng_torch,
                    amp_enabled=amp_enabled,
                )

            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(opt)
            scaler.update()

            sums["loss"] += float(loss.item())
            sums["lin"] += float(mets["lin"])
            sums["pred"] += float(mets["pred"])
            n_batches += 1

        train_m = {k: v / max(1, n_batches) for k, v in sums.items()}

        do_log = (epoch % args.log_every == 0) or (epoch == 1)
        do_save = (epoch % args.save_every == 0)

        if do_log:
            val_m = eval_epoch_stage1(
                model, val_loader, device,
                eval_horizon=eval_horizon,
                eval_start_time=args.eval_start_time,
                lambda_lin=args.lambda_lin,
                lambda_pred=args.lambda_pred,
                gamma=args.gamma,
                ridge=args.ridge,
                nll_subsample_eval=args.nll_subsample_eval,
            )
            score = select_score(val_m, args.select_metric)

            cpu_u, gpu_u, gpu_mu, mem_used, mem_total = get_utilization_snapshot(device)
            util_str = ""
            if cpu_u is not None:
                util_str += f" | CPU={cpu_u:.0f}%"
            if gpu_u is not None:
                util_str += f" | GPU={gpu_u:.0f}%"
            if mem_used is not None and mem_total is not None:
                util_str += f" | VRAM={mem_used}/{mem_total}MiB"

            print(
                f"[ep {epoch:04d}] H={H:02d} | "
                f"train loss={train_m['loss']:.6f} lin={train_m['lin']:.6f} pred={train_m['pred']:.6f} | "
                f"val loss={val_m['loss']:.6f} lin={val_m['lin']:.6f} pred={val_m['pred']:.6f} | "
                f"select={score:.6f}{util_str}"
            )

            with open(log_path, "a") as f:
                f.write(
                    f"stage1,{epoch},{H},"
                    f"{train_m['loss']},{train_m['lin']},{train_m['pred']},"
                    f"{val_m['loss']},{val_m['lin']},{val_m['pred']},{score},"
                    f"{cpu_u},{gpu_u},{gpu_mu},{mem_used},{mem_total}\n"
                )

            # Save best stage1
            if score < best_score:
                best_score = score
                torch.save(
                    dict(
                        stage="stage1",
                        epoch=epoch,
                        model_state=model.state_dict(),
                        cfg=asdict(cfg),
                        val_metrics=val_m,
                        best_score=best_score,
                        select_metric=args.select_metric,
                        ridge=float(args.ridge),
                        H=int(H),
                        eval_horizon=int(eval_horizon),
                    ),
                    best_stage1_path,
                )

        if do_save:
            torch.save(
                dict(
                    stage="stage1",
                    epoch=epoch,
                    model_state=model.state_dict(),
                    cfg=asdict(cfg),
                    ridge=float(args.ridge),
                    H=int(H),
                ),
                os.path.join(args.out_dir, f"ckpt_stage1_ep{epoch:04d}.pt"),
            )

    print(f"[stage1 done] best score={best_score:.6f} | ckpt: {best_stage1_path}")

    # Load best stage1 weights
    ckpt1 = torch.load(best_stage1_path, map_location=device)
    model.load_state_dict(ckpt1["model_state"])

    # -------------------------
    # Compute A_global (freeze target)
    # -------------------------
    print("[A_global] computing global A on training split ...")
    A_global = compute_A_global(
        model=model,
        loader=train_loader,
        device=device,
        ridge=float(args.ridge),
        max_pairs=int(args.A_global_pairs),
        horizon_for_pairs=int(min(args.A_global_horizon, L - 1)),
        start_time=int(args.A_global_start_time),
    ).to(device)
    A_global_np = A_global.detach().cpu().numpy()
    print("[A_global] done")

    # -------------------------
    # Stage 2 (decoder-only)
    # -------------------------
    print(f"[stage2] epochs={args.epochs_stage2} | freeze encoder, fixed A_global")

    # Freeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = False

    # Decoder-only optimizer
    lr2 = args.lr_stage2 if args.lr_stage2 is not None else args.lr
    dec_params = list(model.decoder.parameters())
    opt2 = torch.optim.AdamW(dec_params, lr=lr2, weight_decay=args.weight_decay)
    scaler2 = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_stage2 = float("inf")
    best_epoch2 = 0

    # Stage2 eval baseline
    val2 = eval_epoch_stage2(
        model, val_loader, device,
        eval_horizon=eval_horizon,
        eval_start_time=args.eval_start_time,
        A_global=A_global,
        lambda_pred=args.lambda_pred,
        gamma=args.gamma,
        nll_subsample_eval=args.nll_subsample_eval,
    )
    print(f"[stage2 init] val pred={val2['pred']:.6f} | loss={val2['loss']:.6f}")

    for epoch2 in range(1, args.epochs_stage2 + 1):
        model.train()
        # In stage2, you can keep horizon fixed to eval_horizon (or horizon_max). We'll use eval_horizon.
        H2 = int(eval_horizon)

        sums = dict(loss=0.0, pred=0.0)
        n_batches = 0

        for theta_traj in train_loader:
            theta_traj = theta_traj.to(device, non_blocking=True)  # (B,L,K)
            B, Lb, Kb = theta_traj.shape

            max_start = L - (H2 + 1)
            t0 = int(rng_np.integers(0, max_start + 1)) if max_start > 0 else 0
            theta_seq = theta_traj[:, t0:t0 + H2 + 1, :]

            opt2.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss, mets = compute_stage2_losses(
                    model, theta_seq,
                    A_global=A_global,
                    lambda_pred=args.lambda_pred,
                    gamma=args.gamma,
                    nll_subsample=args.nll_subsample,
                    rng=rng_torch,
                )

            scaler2.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler2.unscale_(opt2)
                torch.nn.utils.clip_grad_norm_(dec_params, max_norm=args.grad_clip)
            scaler2.step(opt2)
            scaler2.update()

            sums["loss"] += float(loss.item())
            sums["pred"] += float(mets["pred"])
            n_batches += 1

        train2 = {k: v / max(1, n_batches) for k, v in sums.items()}

        do_log = (epoch2 % args.log_every == 0) or (epoch2 == 1)
        do_save = (epoch2 % args.save_every == 0)

        if do_log:
            val_m2 = eval_epoch_stage2(
                model, val_loader, device,
                eval_horizon=eval_horizon,
                eval_start_time=args.eval_start_time,
                A_global=A_global,
                lambda_pred=args.lambda_pred,
                gamma=args.gamma,
                nll_subsample_eval=args.nll_subsample_eval,
            )
            score2 = float(val_m2["pred"])  # stage2 selection: minimize pred NLL

            cpu_u, gpu_u, gpu_mu, mem_used, mem_total = get_utilization_snapshot(device)
            util_str = ""
            if cpu_u is not None:
                util_str += f" | CPU={cpu_u:.0f}%"
            if gpu_u is not None:
                util_str += f" | GPU={gpu_u:.0f}%"
            if mem_used is not None and mem_total is not None:
                util_str += f" | VRAM={mem_used}/{mem_total}MiB"

            print(
                f"[ep2 {epoch2:04d}] H={H2:02d} | "
                f"train loss={train2['loss']:.6f} pred={train2['pred']:.6f} | "
                f"val loss={val_m2['loss']:.6f} pred={val_m2['pred']:.6f} | "
                f"select(pred)={score2:.6f}{util_str}"
            )

            with open(log_path, "a") as f:
                f.write(
                    f"stage2,{epoch2},{H2},"
                    f"{train2['loss']},, {train2['pred']},"
                    f"{val_m2['loss']},,{val_m2['pred']},{score2},"
                    f"{cpu_u},{gpu_u},{gpu_mu},{mem_used},{mem_total}\n"
                )

            if score2 < best_stage2:
                best_stage2 = score2
                best_epoch2 = epoch2
                torch.save(
                    dict(
                        stage="stage2",
                        epoch_stage2=epoch2,
                        model_state=model.state_dict(),
                        cfg=asdict(cfg),
                        ridge=float(args.ridge),
                        A_global=A_global_np,
                        val_metrics=val_m2,
                        best_pred=best_stage2,
                        eval_horizon=int(eval_horizon),
                    ),
                    best_path,
                )

        if do_save:
            torch.save(
                dict(
                    stage="stage2",
                    epoch_stage2=epoch2,
                    model_state=model.state_dict(),
                    cfg=asdict(cfg),
                    ridge=float(args.ridge),
                    A_global=A_global_np,
                ),
                os.path.join(args.out_dir, f"ckpt_stage2_ep{epoch2:04d}.pt"),
            )

    print(f"[done] best stage2 pred={best_stage2:.6f} at ep2={best_epoch2} | ckpt: {best_path}")

    # Final report on val/test using best checkpoint
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    A_global = torch.tensor(ckpt["A_global"], device=device, dtype=torch.float32)

    val_final = eval_epoch_stage2(
        model, val_loader, device,
        eval_horizon=eval_horizon,
        eval_start_time=args.eval_start_time,
        A_global=A_global,
        lambda_pred=args.lambda_pred,
        gamma=args.gamma,
        nll_subsample_eval=args.nll_subsample_eval,
    )
    test_final = eval_epoch_stage2(
        model, test_loader, device,
        eval_horizon=eval_horizon,
        eval_start_time=args.eval_start_time,
        A_global=A_global,
        lambda_pred=args.lambda_pred,
        gamma=args.gamma,
        nll_subsample_eval=args.nll_subsample_eval,
    )

    print(f"[best] val pred={val_final['pred']:.6f} | test pred={test_final['pred']:.6f}")

    with open(os.path.join(args.out_dir, "final_metrics.json"), "w") as f:
        json.dump(
            dict(
                stage1_best_score=float(best_score),
                stage2_best_pred=float(best_stage2),
                val=val_final,
                test=test_final,
                select_metric_stage1=args.select_metric,
                ridge=float(args.ridge),
                nll_subsample=int(args.nll_subsample),
                nll_subsample_eval=int(args.nll_subsample_eval),
            ),
            f,
            indent=2
        )


if __name__ == "__main__":
    main()
