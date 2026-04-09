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

"""Compare needle deflection between model prediction and ground truth.

Deflection is defined as the lateral displacement of each needle node from its
original (frame-0) position in the plane perpendicular to the needle's principal
axis (the axis of insertion).  Because the needle starts approximately along z,
deflection ≈ sqrt((x - x0)^2 + (y - y0)^2).

The script produces three plots:

  1. deflection_profile.png — deflection vs axial position for every rollout
     step, predicted (solid) vs GT (dashed).  Curves are coloured by time step.

  2. deflection_tip.png — tip deflection (max deflection along needle) vs
     rollout step, comparing prediction and GT over time.

  3. deflection_overlay.png — predicted vs GT deflection profile overlaid at
     each rollout step (up to 16 subplots).

Usage (from examples/cfd/needle_tissue_cropped/):
    python compare_deflection.py [--infer_dir ./outputs/inference_output]
                                    [--data_dir /path/to/RUN-2]
                                    [--out_dir ./outputs/deflection_plots]
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch

from dataset import _get_needle_tissue_node_sets, _sorted_vtu_files


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_needle_indices(data_dir: str) -> np.ndarray:
    """Return original-mesh node indices that belong to the needle (et=0).

    Reads the ``preprocessed_cache.pt`` built by ``train.py`` and extracts
    needle node indices via the HEX edge topology (no beam reduction).
    """
    cache_path = os.path.join(data_dir, "preprocessed_cache.pt")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"{cache_path} not found — run train.py first to build the cache."
        )
    cache = torch.load(cache_path, weights_only=False)
    needle_idx, _ = _get_needle_tissue_node_sets(
        cache["edge_index"], cache["edge_type_onehot"]
    )
    return needle_idx


def _load_predicted_vtus(infer_dir: str):
    """Return sorted list of predicted_XXXX.vtu paths."""
    pattern = os.path.join(infer_dir, "predicted_*.vtu")
    files = sorted(
        glob.glob(pattern),
        key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)),
    )
    if not files:
        raise FileNotFoundError(f"No predicted_*.vtu files found in {infer_dir}")
    return files


# ---------------------------------------------------------------------------
# Deflection computation
# ---------------------------------------------------------------------------

def _lateral_deflection(pos: np.ndarray, ref_pos: np.ndarray) -> np.ndarray:
    """Lateral deflection of each node from its reference position.

    pos     : (N, 3) current positions
    ref_pos : (N, 3) reference (frame-0) positions
    returns : (N,) lateral displacement magnitude in the XY plane
    """
    delta = pos - ref_pos
    return np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare needle deflection: prediction vs GT")
    parser.add_argument("--infer_dir", default="./outputs/inference_output",
                        help="Directory containing predicted_XXXX.vtu files")
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing raw output_XXXX.vtu files")
    parser.add_argument("--out_dir", default="./outputs/deflection_plots",
                        help="Directory to write plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Identify needle nodes in original mesh ---
    needle_idx = _get_needle_indices(args.data_dir)
    print(f"Found {len(needle_idx)} needle nodes.")

    # --- Reference positions (frame 0) ---
    vtu_files_all = _sorted_vtu_files(args.data_dir)
    ref_mesh = pv.read(vtu_files_all[0])
    ref_pos_needle = ref_mesh.points[needle_idx]   # (N_needle, 3)

    # Axial coordinate: project onto principal axis of needle at frame 0.
    # With PCA, first component = axis of variation = insertion axis.
    centred = ref_pos_needle - ref_pos_needle.mean(axis=0)
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    principal_axis = Vt[0]   # (3,) unit vector along needle axis
    axial_coords = centred @ principal_axis   # (N_needle,) axial position

    # Sort by axial position for clean profile plots
    sort_order = np.argsort(axial_coords)
    axial_sorted = axial_coords[sort_order]

    # --- Load all predicted VTUs ---
    pred_files = _load_predicted_vtus(args.infer_dir)
    n_steps = len(pred_files)
    print(f"Found {n_steps} predicted VTU files.")

    pred_deflections = []   # one array (N_needle,) per step
    gt_deflections   = []   # one array per step (or None)
    has_gt_list      = []

    for step, fpath in enumerate(pred_files):
        mesh = pv.read(fpath)

        # Predicted positions: mesh.points contains model-predicted geometry
        pred_pos_needle = mesh.points[needle_idx]
        pred_defl = _lateral_deflection(pred_pos_needle, ref_pos_needle)
        pred_deflections.append(pred_defl[sort_order])

        # GT positions stored as point_data["Points_gt"] by infer.py
        if "Points_gt" in mesh.point_data:
            gt_pos_needle = mesh.point_data["Points_gt"][needle_idx]
            gt_defl = _lateral_deflection(gt_pos_needle, ref_pos_needle)
            gt_deflections.append(gt_defl[sort_order])
            has_gt_list.append(True)
        else:
            gt_deflections.append(None)
            has_gt_list.append(False)

    # ---------------------------------------------------------------------------
    # Plot 1: Deflection profile (deflection vs axial position) per time step
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    cmap = cm.viridis
    colours = [cmap(i / max(n_steps - 1, 1)) for i in range(n_steps)]

    ax_pred, ax_gt = axes
    ax_pred.set_title("Predicted needle deflection profile")
    ax_gt.set_title("Ground-truth needle deflection profile")

    for step in range(n_steps):
        colour = colours[step]
        label = f"step {step + 1}" if step % max(1, n_steps // 6) == 0 else None
        ax_pred.plot(axial_sorted, pred_deflections[step],
                     color=colour, lw=1.0, label=label)
        if has_gt_list[step]:
            ax_gt.plot(axial_sorted, gt_deflections[step],
                       color=colour, lw=1.0, label=label)

    for ax in axes:
        ax.set_xlabel("Axial position along needle (mm)")
        ax.set_ylabel("Lateral deflection (mm)")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, lw=0.4, alpha=0.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=n_steps))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="Rollout step")
    profile_path = os.path.join(args.out_dir, "deflection_profile.png")
    fig.savefig(profile_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {profile_path}")

    # ---------------------------------------------------------------------------
    # Plot 2: Tip deflection vs rollout step
    # ---------------------------------------------------------------------------
    pred_tip = [d.max() for d in pred_deflections]
    gt_tip   = [d.max() if d is not None else np.nan for d in gt_deflections]
    steps = np.arange(1, n_steps + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, pred_tip, "b-o", ms=4, label="Predicted tip deflection")
    if any(has_gt_list):
        ax.plot(steps, gt_tip, "r--s", ms=4, label="GT tip deflection")
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Max lateral deflection (mm)")
    ax.set_title("Needle tip deflection vs rollout step")
    ax.legend()
    ax.grid(True, lw=0.4, alpha=0.5)
    tip_path = os.path.join(args.out_dir, "deflection_tip.png")
    fig.savefig(tip_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {tip_path}")

    # ---------------------------------------------------------------------------
    # Plot 3: Overlay predicted vs GT deflection profile at each step
    #         (one subplot per step, up to 16 steps)
    # ---------------------------------------------------------------------------
    n_plot = min(n_steps, 16)
    ncols = 4
    nrows = (n_plot + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for i in range(n_plot):
        ax = axes_flat[i]
        ax.plot(axial_sorted, pred_deflections[i], "b-", lw=1.2, label="pred")
        if has_gt_list[i]:
            ax.plot(axial_sorted, gt_deflections[i], "r--", lw=1.2, label="GT")
        ax.set_title(f"Step {i + 1}", fontsize=9)
        ax.set_xlabel("Axial pos (mm)", fontsize=7)
        ax.set_ylabel("Deflection (mm)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, lw=0.3, alpha=0.5)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(n_plot, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Predicted vs GT needle deflection profile per step", fontsize=11)
    fig.tight_layout()
    overlay_path = os.path.join(args.out_dir, "deflection_overlay.png")
    fig.savefig(overlay_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {overlay_path}")


if __name__ == "__main__":
    main()
