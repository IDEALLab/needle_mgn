# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Precompute a per-step rigid-body bias correction from training rollouts.

For each of N training-split samples, runs an autoregressive rollout for
K steps with the trained model and accumulates the per-step rigid
component of the prediction error:

    e_k         = pred_Δu(k) − gt_Δu(k)       (normalised, needle nodes)
    Δt_k        = mean(e_k)                   (translation bias)
    Δω_k        = Σ(r × (e_k − Δt_k)) / Σ|r|²  (rotation bias; r in raw mm)

Both are accumulated and averaged across the N samples to produce
``(K, 3)`` per-step correction tensors.  Also stores their mean over
steps (so callers can opt for a single-value correction).

Output: pickled dict written to ``<stats_dir>/rigid_correction.pt`` (or
the user-specified path):
    {
        "delta_t_norm":           (K, 3) float32  per-step Δt
        "delta_omega_norm":       (K, 3) float32  per-step Δω
        "delta_t_norm_mean":      (3,)   float32  mean over steps
        "delta_omega_norm_mean":  (3,)   float32  mean over steps
        "n_samples":              int     samples actually used
        "n_steps":                int     K
        "exp_dir":                str
    }

At inference time, ``infer.py`` (and ``compare_models.RolloutEngine``)
subtract ``Δt_k + Δω_k × r_i`` from the predicted Δu at needle nodes
when ``apply_rigid_correction=true``.

Usage
-----
    uv run python examples/cfd/needle_tissue_cropped/compute_rigid_correction.py \\
        --exp_dir /path/to/experiment \\
        --data_dir /path/to/RUN-2 \\
        --n_samples 20 \\
        --n_rollout 16
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_models import (  # noqa: E402
    _denorm_per_key,
    _denorm_pred_u,
    _reload_dataset_with_K,
    load_experiment,
)
from dataset import NeedleTissueDataset  # noqa: E402
from freq_response import _dataset_kwargs_from_cfg  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


def _reload_train_split(exp_dir: str, data_dir: str, K: int) -> NeedleTissueDataset:
    cfg = OmegaConf.load(os.path.join(exp_dir, "outputs", ".hydra", "config.yaml"))
    stats_dir = os.path.join(exp_dir, "stats")
    kw = _dataset_kwargs_from_cfg(cfg, data_dir, stats_dir)
    kw["multistep_K"] = int(K)
    return NeedleTissueDataset(split="train", **kw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_path", default=None,
                        help="Where to write the .pt (default: <exp>/stats/rigid_correction.pt).")
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Number of training samples to average over.")
    parser.add_argument("--n_rollout", type=int, default=16,
                        help="Rollout horizon K.")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu)

    out_path = args.out_path or os.path.join(args.exp_dir, "stats", "rigid_correction.pt")

    print(f"Loading experiment {args.exp_dir} on {device} ...")
    _, _, engine = load_experiment(args.exp_dir, args.data_dir, device)
    # Replace test split with training split, future_deltas of length K.
    train_dataset = _reload_train_split(args.exp_dir, args.data_dir, args.n_rollout)
    engine.dataset = train_dataset

    n_use = min(args.n_samples, len(train_dataset))
    print(f"Using {n_use}/{len(train_dataset)} training samples, K={args.n_rollout}")

    sum_dt = torch.zeros(args.n_rollout, 3, dtype=torch.float64, device=device)
    sum_dw = torch.zeros(args.n_rollout, 3, dtype=torch.float64, device=device)
    count = torch.zeros(args.n_rollout, dtype=torch.int64, device=device)

    # Need u column slice within pred (engine.tgt_offsets["u"]).
    u_lo, u_hi = engine.tgt_offsets["u"]
    u_std = engine.u_t_std  # (1, 3) on device, normalisation stat for u target
    u_mean = engine.u_t_mean

    for sample_idx in range(n_use):
        try:
            graph = train_dataset[sample_idx]
        except Exception as e:
            print(f"  sample {sample_idx} load failed: {e}")
            continue
        graph = graph.to(device)
        needle_local = torch.nonzero(graph.is_needle, as_tuple=False).squeeze(-1)
        if needle_local.numel() < 8:
            continue
        future = graph.future_deltas  # (n_sub, K, output_dim)

        preds, _ = engine.rollout_with_states(graph, args.n_rollout, sample_idx=sample_idx)
        for k in range(args.n_rollout):
            pred_norm_u = preds[k][:, u_lo:u_hi]                              # (n_sub, 3)
            gt_norm_u = future[:, k, u_lo:u_hi]                                # (n_sub, 3)
            err_norm = (pred_norm_u - gt_norm_u)[needle_local].double()        # (n_needle, 3)
            pos = graph.pos[needle_local].double()                             # (n_needle, 3) mm
            centroid = pos.mean(dim=0)
            r = pos - centroid
            dt = err_norm.mean(dim=0)                                          # (3,)
            e_prime = err_norm - dt.view(1, 3)
            ang_num = torch.cross(r, e_prime, dim=-1).sum(dim=0)               # (3,)
            ang_den = (r * r).sum().clamp(min=1e-12)
            dw = ang_num / ang_den                                             # (3,)
            sum_dt[k] += dt
            sum_dw[k] += dw
            count[k] += 1

        if (sample_idx + 1) % 5 == 0:
            print(f"  {sample_idx + 1}/{n_use} samples processed")

    if int(count.sum().item()) == 0:
        raise RuntimeError("No rollouts were processed — check sample availability.")

    cnt = count.float().clamp(min=1.0).view(-1, 1)
    delta_t = (sum_dt / cnt).float().cpu()
    delta_w = (sum_dw / cnt).float().cpu()

    n_total = int(count.sum().item())
    delta_t_mean = (sum_dt.sum(0) / max(n_total, 1)).float().cpu()
    delta_w_mean = (sum_dw.sum(0) / max(n_total, 1)).float().cpu()

    payload = {
        "delta_t_norm": delta_t,                  # (K, 3)
        "delta_omega_norm": delta_w,              # (K, 3)
        "delta_t_norm_mean": delta_t_mean,        # (3,)
        "delta_omega_norm_mean": delta_w_mean,    # (3,)
        "n_samples": int(n_use),
        "n_steps": int(args.n_rollout),
        "exp_dir": str(args.exp_dir),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(payload, out_path)
    print(f"\nWrote rigid-body correction → {out_path}")
    print(f"  per-step Δt L2 (norm units): min={delta_t.norm(dim=-1).min().item():.4g}  "
          f"max={delta_t.norm(dim=-1).max().item():.4g}  "
          f"mean={delta_t.norm(dim=-1).mean().item():.4g}")
    print(f"  per-step Δω L2 (norm/mm):    min={delta_w.norm(dim=-1).min().item():.4g}  "
          f"max={delta_w.norm(dim=-1).max().item():.4g}  "
          f"mean={delta_w.norm(dim=-1).mean().item():.4g}")


if __name__ == "__main__":
    main()
