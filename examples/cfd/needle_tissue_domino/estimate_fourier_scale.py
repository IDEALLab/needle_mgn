# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Estimate the Fourier feature scale (sigma) for DCEL-DoMINO via the median heuristic.

The median heuristic (Garreau et al., 2017) sets sigma = 1 / median(||xi - xj||)
over random pairs drawn from the training distribution.  This places the RBF
kernel bandwidth at the natural scale of variation so that typical pairs are
neither always-similar nor always-different under the kernel.

For the DoMINO state encoder the "inputs" are the normalised per-node state
vectors (u, v, a, evf, s, cpress, and static material properties).
We draw random (node, node) pairs from random
training frames and compute their pairwise L2 distances.

We also report:
  - Per-dimension std of the state features (should be ~1 after normalisation)
  - Analytical lower bound: 1/sqrt(D) where D = feature dimension
    (the median distance of i.i.d. N(0,1) D-dim vectors is sqrt(D * chi2_median))
  - Recommended sigma value

Usage (from examples/cfd/needle_tissue_domino/):
    uv run python estimate_fourier_scale.py
    uv run python estimate_fourier_scale.py --n_pairs 50000 --n_frames 30
"""

import argparse
import os

import numpy as np
import torch

from physicsnemo.datapipes.gnn.utils import load_json


def _load_cropped_module():
    """Load the needle_tissue_cropped dataset module."""
    import importlib.util as ilu
    _path = os.path.join(os.path.dirname(__file__), "..", "needle_tissue_cropped", "dataset.py")
    spec = ilu.spec_from_file_location("_cropped", os.path.abspath(_path))
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(description="Estimate Fourier feature scale for state encoder")
    parser.add_argument("--data_dir", default="/home/nathanielhoffman/Desktop/cel/physicsnemo/RUN-2")
    parser.add_argument("--stats_dir", default="./stats")
    parser.add_argument("--n_frames", type=int, default=20,
                        help="Number of random training frames to sample from")
    parser.add_argument("--n_pairs", type=int, default=20000,
                        help="Number of random node pairs to use for distance estimation")
    args = parser.parse_args()

    stats_dir = os.path.abspath(args.stats_dir)
    data_dir = os.path.abspath(args.data_dir)

    # ---- Load normalisation stats -------------------------------------------
    stats_file = os.path.join(stats_dir, "domino_node_stats.json")
    if not os.path.exists(stats_file):
        raise FileNotFoundError(
            f"{stats_file} not found — run train.py first to generate stats."
        )
    node_stats = load_json(stats_file)

    # Import constants from dataset (same directory) so they stay in sync.
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("_domino_dataset", os.path.join(os.path.dirname(__file__), "dataset.py"))
    _ds = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_ds)
    STATE_KEYS = _ds.STATE_KEYS
    STATE_DIMS = _ds.STATE_DIMS
    STATIC_PROP_KEYS = _ds.STATIC_PROP_KEYS
    NODE_STATE_DIM = _ds.NODE_STATE_DIM

    # ---- Load preprocessed cache(s) -----------------------------------------
    _cropped = _load_cropped_module()
    is_multi = _cropped._is_multi_run(data_dir)

    # Gather (frame_tensors, node_props, run_id) for training runs only.
    run_cache_list = []
    if is_multi:
        run_groups = _cropped._group_vtu_by_run(data_dir, timestep_stride=1)
        run_ids = list(run_groups.keys())
        n_runs = len(run_ids)
        n_train_runs = max(1, int(n_runs * 0.8))
        train_run_ids = run_ids[:n_train_runs]
        for run_id in train_run_ids:
            cache_path = os.path.join(data_dir, f"preprocessed_cache_RUN-{run_id}.pt")
            if not os.path.exists(cache_path):
                print(f"  Warning: {cache_path} not found — skipping run {run_id}")
                continue
            raw = torch.load(cache_path, weights_only=False)
            run_cache_list.append((raw["frame_tensors"], raw.get("node_props", {})))
    else:
        cache_path = os.path.join(data_dir, "preprocessed_cache.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"{cache_path} not found — run train.py first.")
        raw = torch.load(cache_path, weights_only=False)
        run_cache_list.append((raw["frame_tensors"], raw.get("node_props", {})))

    if not run_cache_list:
        raise RuntimeError("No training caches found — run train.py first.")

    rng = np.random.default_rng(42)

    # Build pool of (run_idx, frame_idx) pairs from training runs.
    all_pairs = []
    for r_idx, (ft, _) in enumerate(run_cache_list):
        n_frames_run = ft["u"].shape[0]
        all_pairs.extend((r_idx, t) for t in range(n_frames_run - 1))

    selected = rng.choice(len(all_pairs),
                          size=min(args.n_frames, len(all_pairs)),
                          replace=False)
    selected_pairs = [all_pairs[i] for i in selected]

    frame_tensors_0, node_props = run_cache_list[0]
    n_nodes = frame_tensors_0["u"].shape[1]

    # ---- Collect normalised state vectors -----------------------------------
    print(f"Collecting state vectors from {len(selected_pairs)} frames "
          f"({n_nodes} nodes each)…")

    all_states = []
    for r_idx, t in selected_pairs:
        frame_tensors, np_props = run_cache_list[r_idx]
        parts = []
        for key, dim in zip(STATE_KEYS, STATE_DIMS):
            feat = frame_tensors[key][t].float()           # (N, dim)
            mean = torch.tensor(node_stats[f"{key}_mean"], dtype=torch.float32)
            std  = torch.tensor(node_stats[f"{key}_std"],  dtype=torch.float32)
            parts.append((feat - mean) / std.clamp(min=1e-8))
        for key in STATIC_PROP_KEYS:
            feat = np_props[key].float() if key in np_props else torch.zeros(n_nodes, 1)
            mean = torch.tensor(node_stats[f"{key}_mean"], dtype=torch.float32)
            std  = torch.tensor(node_stats[f"{key}_std"],  dtype=torch.float32)
            parts.append((feat - mean) / std.clamp(min=1e-8))
        all_states.append(torch.cat(parts, dim=-1))        # (N, NODE_STATE_DIM)

    states = torch.cat(all_states, dim=0).numpy()          # (N*F, NODE_STATE_DIM)
    print(f"  Total state vectors: {states.shape[0]:,}  |  dimension: {states.shape[1]}")

    # ---- Per-dimension statistics (sanity check) ----------------------------
    per_dim_std = states.std(axis=0)
    print(f"\nPer-dimension std after normalisation (should be ~1):")
    print(f"  min={per_dim_std.min():.3f}  mean={per_dim_std.mean():.3f}  "
          f"max={per_dim_std.max():.3f}")

    # ---- Pairwise distance estimation (random pairs) ------------------------
    print(f"\nSampling {args.n_pairs:,} random node pairs…")
    N = states.shape[0]
    idx_a = rng.integers(0, N, size=args.n_pairs)
    idx_b = rng.integers(0, N, size=args.n_pairs)
    # Avoid self-pairs
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]

    diff = states[idx_a] - states[idx_b]                   # (P, NODE_STATE_DIM)
    dists = np.linalg.norm(diff, axis=-1)                  # (P,)

    median_dist = np.median(dists)
    mean_dist   = dists.mean()
    p25, p75    = np.percentile(dists, [25, 75])

    print(f"\nPairwise L2 distance statistics (normalised state space):")
    print(f"  median = {median_dist:.4f}")
    print(f"  mean   = {mean_dist:.4f}")
    print(f"  25th–75th pct: [{p25:.4f}, {p75:.4f}]")

    # ---- Analytical reference -----------------------------------------------
    D = NODE_STATE_DIM
    # For i.i.d. N(0,1) inputs, E[||x-y||^2] = 2D, so E[||x-y||] ≈ sqrt(2D).
    analytical_ref = np.sqrt(2.0 * D)
    print(f"\n  Analytical reference for N(0,1) inputs (sqrt(2D)): {analytical_ref:.4f}")

    # ---- Recommended sigma --------------------------------------------------
    sigma_median = 1.0 / median_dist
    sigma_p75    = 1.0 / p75   # more conservative: emphasises lower frequencies

    print("\nRecommended fourier_scale_state values:")
    print(f"  sigma = 1/median  = {sigma_median:.4f}  (matches median pairwise distance)")
    print(f"  sigma = 1/p75     = {sigma_p75:.4f}  (more conservative, lower frequencies)")
    print("  sigma = 1.0              (current default — likely too high for normalised inputs)")
    print()
    print("Suggested config.yaml entry:")
    print(f"  fourier_scale_state: {sigma_median:.2f}")

    # ---- Per-group analysis --------------------------------------------------
    # Feature groups: u(0:3), v(3:6), a(6:9), evf(9:10), s(10:16), cpress(16:17)
    groups = [
        ("u",      slice(0,  3)),
        ("v",      slice(3,  6)),
        ("a",      slice(6,  9)),
        ("evf",    slice(9,  10)),
        ("s",      slice(10, 16)),
        ("cpress", slice(16, 17)),
    ]

    print("\nPer-group pairwise distance analysis:")
    print(f"  {'group':>5}  {'dims':>5}  {'median':>8}  {'mean':>8}  {'sigma_med':>10}  {'nonzero%':>9}")
    for name, sl in groups:
        g_diff = diff[:, sl]
        g_dists = np.linalg.norm(g_diff, axis=-1)
        g_median = np.median(g_dists)
        g_mean   = g_dists.mean()
        g_sigma  = 1.0 / g_median if g_median > 1e-8 else float("inf")
        # fraction of pairs where both nodes have nonzero values in this group
        g_a = states[idx_a, sl]
        g_b = states[idx_b, sl]
        nz_frac = np.mean(
            (np.abs(g_a).max(axis=-1) > 0.01) | (np.abs(g_b).max(axis=-1) > 0.01)
        )
        n_dims = sl.stop - sl.start
        print(f"  {name:>5}  {n_dims:>5}  {g_median:>8.4f}  {g_mean:>8.4f}  "
              f"{g_sigma:>10.4f}  {100*nz_frac:>8.1f}%")

    print()
    print("Note: groups with low nonzero% (e.g. cpress before contact) have artificially")
    print("small median distances.  Use a larger sigma for those groups so the features")
    print("remain sensitive to small nonzero values.")


if __name__ == "__main__":
    main()
