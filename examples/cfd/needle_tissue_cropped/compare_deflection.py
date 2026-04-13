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

The script produces three plots and a CSV:

  1. deflection_profile.png — deflection vs axial position for every rollout
     step, predicted (solid) vs GT (dashed).  Curves are coloured by time step.

  2. deflection_tip.png — tip deflection (magnitude + x/y components) vs
     rollout step, comparing prediction and GT over time.

  3. deflection_overlay.png — predicted vs GT deflection profile overlaid at
     each rollout step (up to 16 subplots).

  4. tip_deflection.csv — per-step tip deflection (magnitude, x, y) for both
     prediction and GT, plus signed error (pred - GT).

Usage (from examples/cfd/needle_tissue_cropped/):
uv run compare_deflection.py --infer_dir ./inference_output --data_dir ../../../RUN-2 --out_dir ./outputs/deflection_plots
"""

import argparse
import csv
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402
import torch  # noqa: E402

from dataset import (
    _get_needle_tissue_node_sets,
    _sorted_vtu_files,
    _process_all_frames,
    _atomic_torch_save,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_needle_indices(data_dir: str) -> np.ndarray:
    """Return original-mesh node indices that belong to the needle (et=0).

    Reads the ``preprocessed_cache.pt`` built by ``train.py`` and extracts
    needle node indices via the HEX edge topology (no beam reduction).
    Rebuilds the cache if it is missing or the frame count has changed.
    """
    vtu_files = _sorted_vtu_files(data_dir)
    cache_path = os.path.join(data_dir, "preprocessed_cache.pt")

    need_rebuild = not os.path.exists(cache_path)
    if not need_rebuild:
        existing = torch.load(cache_path, weights_only=False)
        cached_n = len(existing.get("frame_tensors", {}).get("coord", []))
        if cached_n != len(vtu_files):
            print(
                f"Cache outdated (cached={cached_n}, on-disk={len(vtu_files)}) — rebuilding ..."
            )
            need_rebuild = True

    if need_rebuild:
        print(f"Building cache at {cache_path} ...")
        cache = _process_all_frames(vtu_files)
        _atomic_torch_save(cache, cache_path)
        print(f"  → saved to {cache_path}")
    else:
        cache = existing

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
    """Lateral deflection magnitude of each needle node from its reference position.

    pos     : (N, 3) current positions
    ref_pos : (N, 3) reference (frame-0) positions
    returns : (N,) lateral displacement magnitude in the XY plane
    """
    delta = pos - ref_pos
    return np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2)


def _tip_deflection_components(
    pos_needle: np.ndarray,
    ref_pos_needle: np.ndarray,
    tip_local_idx: int,
) -> tuple:
    """Return (magnitude, x, y) deflection at the needle tip node.

    pos_needle     : (N_needle, 3) current needle node positions
    ref_pos_needle : (N_needle, 3) reference (frame-0) needle positions
    tip_local_idx  : index into the needle array for the axially extreme node
    returns        : (magnitude, delta_x, delta_y) as floats
    """
    delta = pos_needle[tip_local_idx] - ref_pos_needle[tip_local_idx]
    mag = float(np.sqrt(delta[0] ** 2 + delta[1] ** 2))
    return mag, float(delta[0]), float(delta[1])


def analyze_run(
    infer_dir: str,
    needle_idx: np.ndarray,
    ref_pos_needle: np.ndarray,
    sort_order: np.ndarray,
    out_dir: str,
    run_label: str = "",
) -> list:
    """Analyse one inference run and write plots + CSV to *out_dir*.

    Parameters
    ----------
    infer_dir      : directory with predicted_XXXX.vtu files
    needle_idx     : global node indices for needle nodes
    ref_pos_needle : (N_needle, 3) reference positions (frame 0)
    sort_order     : permutation that sorts needle nodes by axial position
    out_dir        : output directory for plots and CSV
    run_label      : short string used in plot titles and file prefixes

    Returns
    -------
    List of dicts, one per rollout step, with keys:
        step, pred_mag, pred_x, pred_y,
        gt_mag, gt_x, gt_y,
        error_mag, error_x, error_y, has_gt
    """
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"{run_label}_" if run_label else ""
    axial_sorted = (ref_pos_needle - ref_pos_needle.mean(axis=0)) @ _principal_axis(ref_pos_needle)
    axial_sorted = axial_sorted[sort_order]
    tip_local_idx = int(sort_order[-1])

    pred_files = _load_predicted_vtus(infer_dir)
    n_steps = len(pred_files)
    print(f"  {run_label or 'run'}: {n_steps} predicted steps")

    pred_deflections, gt_deflections, has_gt_list = [], [], []
    rows = []

    for step, fpath in enumerate(pred_files):
        mesh = pv.read(fpath)
        pred_pos_needle = mesh.points[needle_idx]
        pred_defl = _lateral_deflection(pred_pos_needle, ref_pos_needle)
        pred_deflections.append(pred_defl[sort_order])
        pred_mag, pred_x, pred_y = _tip_deflection_components(
            pred_pos_needle, ref_pos_needle, tip_local_idx
        )

        if "Points_gt" in mesh.point_data:
            gt_pos_needle = mesh.point_data["Points_gt"][needle_idx]
            gt_defl = _lateral_deflection(gt_pos_needle, ref_pos_needle)
            gt_deflections.append(gt_defl[sort_order])
            gt_mag, gt_x, gt_y = _tip_deflection_components(
                gt_pos_needle, ref_pos_needle, tip_local_idx
            )
            has_gt = True
        else:
            gt_deflections.append(None)
            gt_mag = gt_x = gt_y = float("nan")
            has_gt = False

        has_gt_list.append(has_gt)
        rows.append({
            "step": step + 1,
            "pred_mag": pred_mag,
            "pred_x": pred_x,
            "pred_y": pred_y,
            "gt_mag": gt_mag,
            "gt_x": gt_x,
            "gt_y": gt_y,
            "error_mag": pred_mag - gt_mag,
            "error_x": pred_x - gt_x,
            "error_y": pred_y - gt_y,
            "has_gt": has_gt,
        })

    # ---- CSV ----------------------------------------------------------------
    csv_path = os.path.join(out_dir, f"{prefix}tip_deflection.csv")
    _write_csv(csv_path, rows)
    print(f"  Saved: {csv_path}")

    title_prefix = f"[{run_label}] " if run_label else ""

    # ---- Plot 1: deflection profile -----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    cmap = cm.viridis
    colours = [cmap(i / max(n_steps - 1, 1)) for i in range(n_steps)]
    ax_pred, ax_gt = axes
    ax_pred.set_title(f"{title_prefix}Predicted needle deflection profile")
    ax_gt.set_title(f"{title_prefix}Ground-truth needle deflection profile")
    for step in range(n_steps):
        colour = colours[step]
        label = f"step {step + 1}" if step % max(1, n_steps // 6) == 0 else None
        ax_pred.plot(axial_sorted, pred_deflections[step], color=colour, lw=1.0, label=label)
        if has_gt_list[step]:
            ax_gt.plot(axial_sorted, gt_deflections[step], color=colour, lw=1.0, label=label)
    for ax in axes:
        ax.set_xlabel("Axial position along needle (mm)")
        ax.set_ylabel("Lateral deflection (mm)")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, lw=0.4, alpha=0.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=n_steps))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="Rollout step")
    profile_path = os.path.join(out_dir, f"{prefix}deflection_profile.png")
    fig.savefig(profile_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {profile_path}")

    # ---- Plot 2: tip deflection (magnitude + x/y) ---------------------------
    steps_arr = np.arange(1, n_steps + 1)
    pred_mags = [r["pred_mag"] for r in rows]
    pred_xs   = [r["pred_x"]  for r in rows]
    pred_ys   = [r["pred_y"]  for r in rows]
    gt_mags   = [r["gt_mag"]  for r in rows]
    gt_xs     = [r["gt_x"]    for r in rows]
    gt_ys     = [r["gt_y"]    for r in rows]
    err_mags  = [r["error_mag"] for r in rows]
    err_xs    = [r["error_x"]   for r in rows]
    err_ys    = [r["error_y"]   for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for ax, pred_vals, gt_vals, err_vals, ylabel in zip(
        axes,
        [pred_mags, pred_xs, pred_ys],
        [gt_mags,   gt_xs,   gt_ys],
        [err_mags,  err_xs,  err_ys],
        ["Tip deflection magnitude (mm)", "Tip deflection X (mm)", "Tip deflection Y (mm)"],
    ):
        ax.plot(steps_arr, pred_vals, "b-o", ms=3, lw=1.2, label="Predicted")
        if any(has_gt_list):
            ax.plot(steps_arr, gt_vals, "r--s", ms=3, lw=1.2, label="GT")
            ax.plot(steps_arr, err_vals, "k:", lw=0.9, label="Error (pred−GT)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, lw=0.4, alpha=0.5)
    axes[-1].set_xlabel("Rollout step")
    fig.suptitle(f"{title_prefix}Needle tip deflection vs rollout step", fontsize=11)
    fig.tight_layout()
    tip_path = os.path.join(out_dir, f"{prefix}deflection_tip.png")
    fig.savefig(tip_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {tip_path}")

    # ---- Plot 3: overlay (up to 16 steps) -----------------------------------
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
    fig.suptitle(f"{title_prefix}Predicted vs GT needle deflection profile per step", fontsize=11)
    fig.tight_layout()
    overlay_path = os.path.join(out_dir, f"{prefix}deflection_overlay.png")
    fig.savefig(overlay_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {overlay_path}")

    return rows


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _principal_axis(pos: np.ndarray) -> np.ndarray:
    """Return the unit vector along the principal axis of *pos* (N, 3)."""
    centred = pos - pos.mean(axis=0)
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    return Vt[0]


def _write_csv(path: str, rows: list) -> None:
    fieldnames = [
        "step",
        "pred_mag", "pred_x", "pred_y",
        "gt_mag",   "gt_x",   "gt_y",
        "error_mag", "error_x", "error_y",
        "has_gt",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


# ---------------------------------------------------------------------------
# Main (single-run CLI)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare needle deflection: prediction vs GT")
    parser.add_argument("--infer_dir", default="./outputs/inference_output",
                        help="Directory containing predicted_XXXX.vtu files")
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing raw VTU files")
    parser.add_argument("--out_dir", default="./outputs/deflection_plots",
                        help="Directory to write plots and CSV")
    parser.add_argument("--run_label", default="",
                        help="Optional label used in plot titles and file prefixes")
    args = parser.parse_args()

    needle_idx = _get_needle_indices(args.data_dir)
    print(f"Found {len(needle_idx)} needle nodes.")

    vtu_files_all = _sorted_vtu_files(args.data_dir)
    ref_mesh = pv.read(vtu_files_all[0])
    ref_pos_needle = ref_mesh.points[needle_idx]

    principal = _principal_axis(ref_pos_needle)
    centred = ref_pos_needle - ref_pos_needle.mean(axis=0)
    axial_coords = centred @ principal
    sort_order = np.argsort(axial_coords)

    analyze_run(
        infer_dir=args.infer_dir,
        needle_idx=needle_idx,
        ref_pos_needle=ref_pos_needle,
        sort_order=sort_order,
        out_dir=args.out_dir,
        run_label=args.run_label,
    )


if __name__ == "__main__":
    main()
