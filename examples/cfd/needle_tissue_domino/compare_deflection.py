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

"""Compare needle deflection between DCEL-DoMINO prediction and ground truth.

Identical analysis to needle_tissue_cropped/compare_deflection.py.
Needle node indices are read from the DoMINO cache (which loads them from
the preprocessed_cache.pt edge topology — no beam reduction).

Usage (from examples/cfd/needle_tissue_domino/):
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

from dataset import (
    _get_needle_tissue_node_sets,
    _process_all_frames,
    _sorted_vtu_files,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_needle_indices(data_dir: str, grid_res=(64, 32, 32)) -> np.ndarray:
    """Load needle node indices from the DoMINO cache (or preprocessed cache)."""
    # Try each grid resolution variant of the DoMINO cache
    for res in [grid_res, (64, 32, 32), (32, 16, 16)]:
        path = os.path.join(
            data_dir, f"domino_cache_{res[0]}x{res[1]}x{res[2]}.pt"
        )
        if os.path.exists(path):
            dc = torch.load(path, weights_only=False)
            return dc["needle_idx"].numpy()

    # Fall back to a topology-only cache built from a single VTU frame.
    # Needle indices come from the fixed HEX topology, which is identical
    # across all frames and runs — there is no need to process the full dataset.
    topo_path = os.path.join(data_dir, "preprocessed_topology.pt")
    if os.path.exists(topo_path):
        raw = torch.load(topo_path, weights_only=False)
    else:
        first_file = _sorted_vtu_files(data_dir)[0]
        print(f"Building topology cache from {os.path.basename(first_file)} ...")
        raw = _process_all_frames([first_file])
        tmp = topo_path + f".{os.getpid()}.tmp"
        torch.save(raw, tmp)
        os.replace(tmp, topo_path)
        print(f"  → saved to {topo_path}")
    needle_idx, _ = _get_needle_tissue_node_sets(
        raw["edge_index"], raw["edge_type_onehot"]
    )
    return needle_idx


def _load_predicted_vtus(infer_dir: str):
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
    delta = pos - ref_pos
    return np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare needle deflection: DCEL-DoMINO vs GT")
    parser.add_argument("--infer_dir", default="./inference_output")
    parser.add_argument("--data_dir", default=os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../RUN-2")
    ))
    parser.add_argument("--out_dir", default="./outputs/deflection_plots")
    parser.add_argument(
        "--grid_res", default="64,32,32",
        help="Grid resolution used during training (to locate the DoMINO cache)"
    )
    args = parser.parse_args()

    grid_res = tuple(int(x) for x in args.grid_res.split(","))
    os.makedirs(args.out_dir, exist_ok=True)

    needle_idx = _get_needle_indices(args.data_dir, grid_res)
    print(f"Found {len(needle_idx)} needle nodes.")

    vtu_files_all = _sorted_vtu_files(args.data_dir)
    ref_mesh = pv.read(vtu_files_all[0])
    ref_pos_needle = ref_mesh.points[needle_idx]

    centred = ref_pos_needle - ref_pos_needle.mean(axis=0)
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    principal_axis = Vt[0]
    axial_coords = centred @ principal_axis
    sort_order = np.argsort(axial_coords)
    axial_sorted = axial_coords[sort_order]

    pred_files = _load_predicted_vtus(args.infer_dir)
    n_steps = len(pred_files)
    print(f"Found {n_steps} predicted VTU files.")

    pred_deflections, gt_deflections, has_gt_list = [], [], []

    for fpath in pred_files:
        mesh = pv.read(fpath)
        pred_pos = mesh.points[needle_idx]
        pred_deflections.append(_lateral_deflection(pred_pos, ref_pos_needle)[sort_order])

        if "Points_gt" in mesh.point_data:
            gt_pos = mesh.point_data["Points_gt"][needle_idx]
            gt_deflections.append(_lateral_deflection(gt_pos, ref_pos_needle)[sort_order])
            has_gt_list.append(True)
        else:
            gt_deflections.append(None)
            has_gt_list.append(False)

    cmap = cm.viridis
    colours = [cmap(i / max(n_steps - 1, 1)) for i in range(n_steps)]

    # ---- Plot 1: side-by-side deflection profiles --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    ax_pred, ax_gt = axes
    ax_pred.set_title("DCEL-DoMINO predicted deflection profile")
    ax_gt.set_title("Ground-truth deflection profile")

    for i in range(n_steps):
        label = f"step {i + 1}" if i % max(1, n_steps // 6) == 0 else None
        ax_pred.plot(axial_sorted, pred_deflections[i], color=colours[i], lw=1.0, label=label)
        if has_gt_list[i]:
            ax_gt.plot(axial_sorted, gt_deflections[i], color=colours[i], lw=1.0, label=label)

    for ax in axes:
        ax.set_xlabel("Axial position along needle (mm)")
        ax.set_ylabel("Lateral deflection (mm)")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, lw=0.4, alpha=0.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=n_steps))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="Rollout step")
    path = os.path.join(args.out_dir, "deflection_profile.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ---- Plot 2: tip deflection vs step ------------------------------------
    pred_tip = [d.max() for d in pred_deflections]
    gt_tip = [d.max() if d is not None else np.nan for d in gt_deflections]
    steps = np.arange(1, n_steps + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, pred_tip, "b-o", ms=4, label="DCEL-DoMINO tip deflection")
    if any(has_gt_list):
        ax.plot(steps, gt_tip, "r--s", ms=4, label="GT tip deflection")
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Max lateral deflection (mm)")
    ax.set_title("Needle tip deflection vs rollout step")
    ax.legend()
    ax.grid(True, lw=0.4, alpha=0.5)
    path = os.path.join(args.out_dir, "deflection_tip.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ---- Plot 3: per-step overlay (up to 16) --------------------------------
    n_plot = min(n_steps, 16)
    ncols, nrows = 4, (n_plot + 3) // 4
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

    fig.suptitle("DCEL-DoMINO vs GT needle deflection per step", fontsize=11)
    fig.tight_layout()
    path = os.path.join(args.out_dir, "deflection_overlay.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
