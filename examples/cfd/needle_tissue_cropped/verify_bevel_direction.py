# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Verify which world-frame direction the needle bevel points along.

The bias-spectrum analysis in compare_models.py uses the 2nd principal
axis of the needle point cloud as ``transverse1`` and labels it as
"bevel-aligned (≈ +Y)" under the assumption that the bevel is parallel
to world Y.  This script confirms that assumption by:

  1. Loading the full (un-cropped) needle from frame 0 of the chosen run.
  2. Computing the SVD principal axes with the same sign-anchoring rule
     compare_models.py uses.
  3. Binning the needle along its axial direction and tracking the
     cross-section centroid in two frames:
       - needle-aligned (PC2 / PC3)
       - world (X / Y)
     The "bevel-aligned" direction is the one in which the centroid
     drifts monotonically near the tip — the sharp edge offsets the
     local centre of mass away from the axis as the cross-section
     wedges down to a point.
  4. Reporting tip-vs-base centroid offset in world frame.
  5. Saving SVG plots: ortho views with PC2 / world-Y annotated, and
     the per-bin centroid profiles.

Usage
-----
    uv run python verify_bevel_direction.py \\
        --exp_dir /path/to/cropped_fiber_iso_invw \\
        --data_dir ../../../RUN-2 \\
        --out_dir ./bevel_verify_out

Any experiment dir works — the script only needs the dataset config to
know how to load RUN-2; checkpoints / stats aren't used.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import NeedleTissueDataset  # noqa: E402
from freq_response import _dataset_kwargs_from_cfg  # noqa: E402


def _stable_basis(pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (axis, pc2, pc3, centroid, singular_values) with the same sign
    convention used by ``compare_models.py._stable_needle_basis``.
    """
    centroid = pos.mean(axis=0)
    centred = pos - centroid
    _U, S, Vt = np.linalg.svd(centred, full_matrices=False)
    axis = Vt[0].copy()
    pc2 = Vt[1].copy()
    # Sign rule: largest-abs component positive.
    if axis[int(np.argmax(np.abs(axis)))] < 0:
        axis = -axis
    if pc2[int(np.argmax(np.abs(pc2)))] < 0:
        pc2 = -pc2
    pc3 = np.cross(axis, pc2)
    return axis, pc2, pc3, centroid, S


def _bin_centroids(centred: np.ndarray, axial: np.ndarray, n_bins: int,
                   directions: dict) -> Tuple[np.ndarray, dict, np.ndarray]:
    """Per-axial-bin centroid projected onto each provided direction.

    Returns (bin_centers, {name: per-bin centroid scalar}, counts).
    """
    edges = np.linspace(axial.min(), axial.max(), n_bins + 1)
    idx = np.clip(np.digitize(axial, edges) - 1, 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = {name: np.zeros(n_bins) for name in directions}
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = idx == b
        c = int(m.sum())
        if c == 0:
            continue
        pts = centred[m]
        counts[b] = c
        for name, d in directions.items():
            out[name][b] = (pts @ d).mean()
    return centers, out, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True, help="Any experiment dir (uses its dataset config).")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", default="./bevel_verify_out")
    parser.add_argument("--run_idx", type=int, default=0,
                        help="Index into dataset._run_data (split=test).")
    parser.add_argument("--frame_idx", type=int, default=0,
                        help="Frame within the run (0 = initial / mostly-undeformed).")
    parser.add_argument("--n_bins", type=int, default=40,
                        help="Number of axial bins for the centroid profile.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg = OmegaConf.load(os.path.join(args.exp_dir, "outputs", ".hydra", "config.yaml"))
    stats_dir = os.path.join(args.exp_dir, "stats")
    ds_kwargs = _dataset_kwargs_from_cfg(cfg, args.data_dir, stats_dir)
    print(f"Loading test split from {args.data_dir} ...")
    dataset = NeedleTissueDataset(split="test", **ds_kwargs)

    run_idx = max(0, min(args.run_idx, len(dataset._run_data) - 1))
    run = dataset._run_data[run_idx]
    n_frames = run["frame_tensors"]["coord"].shape[0]
    frame_idx = max(0, min(args.frame_idx, n_frames - 1))
    print(f"Using run {run_idx} ({n_frames} frames), frame {frame_idx}.")

    coord_full = run["frame_tensors"]["coord"][frame_idx].numpy()  # (n_nodes, 3)
    needle_global = dataset._needle_idx_t.numpy()
    pos = coord_full[needle_global]  # full needle, raw mm
    n_needle = pos.shape[0]
    print(f"Needle nodes: {n_needle}")

    axis, pc2, pc3, centroid, sv = _stable_basis(pos)
    centred = pos - centroid
    axial = centred @ axis

    world_x = np.array([1.0, 0.0, 0.0])
    world_y = np.array([0.0, 1.0, 0.0])
    world_z = np.array([0.0, 0.0, 1.0])

    centers, c_by_dir, counts = _bin_centroids(
        centred, axial, args.n_bins,
        {"PC2": pc2, "PC3": pc3, "world_X": world_x, "world_Y": world_y},
    )

    # Tip vs base offset in world frame (drop the axial component so we only
    # see transverse drift — but axis isn't exactly aligned with any world axis,
    # so subtract its world-frame contribution).
    tip_bin = int(np.argmax(counts > 0)) if counts[-1] == 0 else len(counts) - 1
    base_bin = 0
    # Use the per-axis world centroids directly.
    tip_world = np.array([c_by_dir["world_X"][-1], c_by_dir["world_Y"][-1], 0.0])
    base_world = np.array([c_by_dir["world_X"][0], c_by_dir["world_Y"][0], 0.0])
    tip_to_base_xy = tip_world - base_world  # (3,) — XY only

    print()
    print("=" * 60)
    print("Principal-axis decomposition (sign-anchored as in compare_models.py)")
    print("=" * 60)
    np.set_printoptions(formatter={"float_kind": lambda x: f"{x:+8.4f}"})
    print(f"axis (PC1):  {axis}   (singular value {sv[0]:.2f})")
    print(f"PC2:         {pc2}    (singular value {sv[1]:.2f})")
    print(f"PC3:         {pc3}    (singular value {sv[2]:.2f})")
    print(f"centroid:    {centroid}  (mm)")
    print()
    print("Alignment with world axes (dot products):")
    for v, name in [(axis, "axis"), (pc2, "PC2"), (pc3, "PC3")]:
        print(f"  {name} · X = {float(v @ world_x):+.4f}    "
              f"{name} · Y = {float(v @ world_y):+.4f}    "
              f"{name} · Z = {float(v @ world_z):+.4f}")
    print()
    print(f"|sv[1] / sv[2]| (transverse anisotropy): {sv[1] / max(sv[2], 1e-12):.3f}")
    print(f"  > 1.05  ⇒  bevel direction is well-defined (sv[1] dominates).")
    print(f"  ≈ 1.00  ⇒  near-degenerate; PC2/PC3 split is unstable.")
    print()
    print(f"Tip − base centroid offset in world XY (mm): {tip_to_base_xy[:2]}")
    norm_xy = np.linalg.norm(tip_to_base_xy[:2])
    if norm_xy > 1e-6:
        direction = tip_to_base_xy[:2] / norm_xy
        print(f"Unit direction (XY plane):                    "
              f"({direction[0]:+.4f}, {direction[1]:+.4f})")
        if abs(direction[1]) > abs(direction[0]):
            print(f"  →  Bevel offset is predominantly along world {'Y' if direction[1] > 0 else 'Y (negative)'}.")
        else:
            print(f"  →  Bevel offset is predominantly along world {'X' if direction[0] > 0 else 'X (negative)'}.")
    print()

    # ---- Plots ----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots.")
        return

    # Three ortho views of the needle.
    L = float(np.linalg.norm(pos.max(0) - pos.min(0)))
    fig = plt.figure(figsize=(15, 5))
    for i, (a, b, alab, blab) in enumerate([(0, 1, "X", "Y"), (0, 2, "X", "Z"), (1, 2, "Y", "Z")]):
        ax = fig.add_subplot(1, 3, i + 1)
        ax.scatter(pos[:, a], pos[:, b], s=1.5, c="tab:gray", alpha=0.4, label="needle nodes")
        tip_pt = pos[int(np.argmax(axial))]
        base_pt = pos[int(np.argmin(axial))]
        ax.plot([tip_pt[a]], [tip_pt[b]], "rx", markersize=12, mew=2, label="tip (max axial)")
        ax.plot([base_pt[a]], [base_pt[b]], "g+", markersize=12, mew=2, label="base (min axial)")

        # PC1 axis line through centroid.
        p0 = centroid - axis * L * 0.55
        p1 = centroid + axis * L * 0.55
        ax.plot([p0[a], p1[a]], [p0[b], p1[b]], "r--", alpha=0.6, linewidth=1, label="PC1 axis")

        # PC2 arrow (purple) and world Y arrow (orange).
        arrow_len = L * 0.15
        for d, color, name in [(pc2, "purple", "PC2"), (world_y, "orange", "world Y")]:
            head = centroid + d * arrow_len
            ax.annotate("", xy=(head[a], head[b]),
                        xytext=(centroid[a], centroid[b]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2))
            ax.text(head[a], head[b], f" {name}", color=color, fontsize=9)

        ax.set_xlabel(f"{alab} (mm)")
        ax.set_ylabel(f"{blab} (mm)")
        ax.set_title(f"{alab}{blab} view")
        ax.set_aspect("equal")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.25)
    fig.suptitle(f"Needle geometry (run {run_idx}, frame {frame_idx})")
    fig.tight_layout()
    out_svg = os.path.join(args.out_dir, "needle_ortho_views.svg")
    fig.savefig(out_svg)
    plt.close(fig)
    print(f"wrote → {out_svg}")

    # Centroid profile along axial coord — needle-aligned + world frames.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(centers, c_by_dir["PC2"], "o-", color="tab:purple", label="centroid · PC2")
    axes[0].plot(centers, c_by_dir["PC3"], "s-", color="tab:olive", label="centroid · PC3")
    axes[0].axhline(0, color="k", linestyle="--", alpha=0.4)
    axes[0].set_xlabel("axial coord (mm, from needle centroid)")
    axes[0].set_ylabel("bin centroid offset (mm)")
    axes[0].set_title("Cross-section centroid — needle-aligned frame")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(centers, c_by_dir["world_X"], "o-", color="tab:blue", label="centroid · world X")
    axes[1].plot(centers, c_by_dir["world_Y"], "s-", color="tab:orange", label="centroid · world Y")
    axes[1].axhline(0, color="k", linestyle="--", alpha=0.4)
    axes[1].set_xlabel("axial coord (mm, from needle centroid)")
    axes[1].set_ylabel("bin centroid offset (mm)")
    axes[1].set_title("Cross-section centroid — world frame")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.suptitle("Bevel detection: which transverse direction does the cross-section centroid drift along?")
    fig.tight_layout()
    out_svg = os.path.join(args.out_dir, "bevel_centroid_profile.svg")
    fig.savefig(out_svg)
    plt.close(fig)
    print(f"wrote → {out_svg}")


if __name__ == "__main__":
    main()
