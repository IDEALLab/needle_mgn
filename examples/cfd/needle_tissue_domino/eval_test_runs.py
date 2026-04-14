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

"""Evaluate needle deflection across all test-set runs for DCEL-DoMINO.

Uses the same train/val/test split as train.py and infer.py to identify test
runs, then analyses each run that has corresponding predicted VTU files.
Runs are processed in parallel across processes.

Expected directory layout (produced by run_test_inference.py):

    <infer_base_dir>/
        RUN-140/
            predicted_0000.vtu
            predicted_0001.vtu
            ...
        RUN-141/
            ...

Outputs:
  <out_dir>/
      RUN-<id>/          — per-run plots + tip_deflection.csv
      summary.csv        — all runs combined (run_id, step, metrics...)
      summary_tip_mag.png — tip deflection magnitude across runs
      summary_tip_xy.png  — x/y tip deflection across runs
      summary_error.png   — tip deflection error across runs

Usage (from examples/cfd/needle_tissue_domino/):
    uv run eval_test_runs.py
    uv run eval_test_runs.py --infer_base_dir ./inference_output --out_dir ./outputs/eval_test_runs
"""

import argparse
import csv
import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from omegaconf import OmegaConf

from compare_deflection import _get_needle_indices
from dataset import _group_vtu_by_run, _sorted_vtu_files

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CFG = OmegaConf.load(os.path.join(_SCRIPT_DIR, "conf", "config.yaml"))

_RAW_DATA_DIR: str = OmegaConf.select(_CFG, "data_dir", default="../../../RUN-2")
_DEFAULT_DATA_DIR: str = os.path.normpath(os.path.join(_SCRIPT_DIR, _RAW_DATA_DIR))
_DEFAULT_TRAIN_FRACTION: float = float(OmegaConf.select(_CFG, "train_fraction", default=0.8))
_DEFAULT_VAL_FRACTION: float = float(OmegaConf.select(_CFG, "val_fraction", default=0.1))
_DEFAULT_TIMESTEP_STRIDE: int = int(OmegaConf.select(_CFG, "timestep_stride", default=1))
_DEFAULT_GRID_RES: tuple = tuple(
    int(x) for x in OmegaConf.select(_CFG, "grid_res", default=[32, 16, 16])
)


# ---------------------------------------------------------------------------
# Test-run discovery
# ---------------------------------------------------------------------------

def _get_test_run_ids(
    data_dir: str,
    train_fraction: float,
    val_fraction: float,
    timestep_stride: int,
) -> list:
    """Return the run IDs assigned to the test split (same logic as infer.py)."""
    run_files = _group_vtu_by_run(data_dir, timestep_stride)
    run_ids = list(run_files.keys())
    n_runs = len(run_ids)
    n_train = max(1, int(n_runs * train_fraction))
    n_val   = max(1, int(n_runs * val_fraction))
    test_ids = run_ids[n_train + n_val:]
    if not test_ids:
        test_ids = run_ids[-1:]
    return test_ids


# ---------------------------------------------------------------------------
# Per-run analysis helpers
# ---------------------------------------------------------------------------

def _principal_axis(pos: np.ndarray) -> np.ndarray:
    """Return the unit vector along the principal axis of *pos* (N, 3)."""
    centred = pos - pos.mean(axis=0)
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    return Vt[0]


def _lateral_deflection(pos: np.ndarray, ref_pos: np.ndarray) -> np.ndarray:
    """Lateral deflection magnitude in the XY plane from reference positions."""
    delta = pos - ref_pos
    return np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2)


def _tip_deflection_components(
    pos_needle: np.ndarray,
    ref_pos_needle: np.ndarray,
    tip_local_idx: int,
) -> tuple:
    """Return (magnitude, delta_x, delta_y) deflection at the axially extreme node."""
    delta = pos_needle[tip_local_idx] - ref_pos_needle[tip_local_idx]
    mag = float(np.sqrt(delta[0] ** 2 + delta[1] ** 2))
    return mag, float(delta[0]), float(delta[1])


def _load_predicted_vtus(infer_dir: str) -> list:
    """Return sorted list of predicted_XXXX.vtu paths."""
    files = sorted(
        glob.glob(os.path.join(infer_dir, "predicted_*.vtu")),
        key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)),
    )
    if not files:
        raise FileNotFoundError(f"No predicted_*.vtu files found in {infer_dir}")
    return files


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


def analyze_run(
    infer_dir: str,
    needle_idx: np.ndarray,
    ref_pos_needle: np.ndarray,
    sort_order: np.ndarray,
    out_dir: str,
    run_label: str = "",
) -> list:
    """Analyse one inference run and write plots + CSV to *out_dir*.

    Returns a list of per-step dicts with keys:
        step, pred_mag, pred_x, pred_y,
        gt_mag, gt_x, gt_y,
        error_mag, error_x, error_y, has_gt
    """
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"{run_label}_" if run_label else ""
    title_prefix = f"[{run_label}] " if run_label else ""

    axial_coords = (ref_pos_needle - ref_pos_needle.mean(axis=0)) @ _principal_axis(ref_pos_needle)
    axial_sorted = axial_coords[sort_order]
    tip_local_idx = int(sort_order[-1])

    pred_files = _load_predicted_vtus(infer_dir)
    n_steps = len(pred_files)
    print(f"  {run_label or 'run'}: {n_steps} predicted steps")

    pred_deflections, gt_deflections, has_gt_list, rows = [], [], [], []

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
            "pred_mag": pred_mag, "pred_x": pred_x, "pred_y": pred_y,
            "gt_mag":   gt_mag,   "gt_x":   gt_x,   "gt_y":   gt_y,
            "error_mag": pred_mag - gt_mag,
            "error_x":   pred_x   - gt_x,
            "error_y":   pred_y   - gt_y,
            "has_gt": has_gt,
        })

    # ---- CSV ----------------------------------------------------------------
    csv_path = os.path.join(out_dir, f"{prefix}tip_deflection.csv")
    _write_csv(csv_path, rows)
    print(f"  Saved: {csv_path}")

    cmap = cm.viridis
    colours = [cmap(i / max(n_steps - 1, 1)) for i in range(n_steps)]

    # ---- Plot 1: deflection profile -----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    ax_pred, ax_gt = axes
    ax_pred.set_title(f"{title_prefix}DCEL-DoMINO predicted deflection profile")
    ax_gt.set_title(f"{title_prefix}Ground-truth deflection profile")
    for i in range(n_steps):
        colour = colours[i]
        label = f"step {i + 1}" if i % max(1, n_steps // 6) == 0 else None
        ax_pred.plot(axial_sorted, pred_deflections[i], color=colour, lw=1.0, label=label)
        if has_gt_list[i]:
            ax_gt.plot(axial_sorted, gt_deflections[i], color=colour, lw=1.0, label=label)
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
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for ax, pred_vals, gt_vals, err_vals, ylabel in zip(
        axes,
        [[r["pred_mag"] for r in rows], [r["pred_x"] for r in rows], [r["pred_y"] for r in rows]],
        [[r["gt_mag"]   for r in rows], [r["gt_x"]   for r in rows], [r["gt_y"]   for r in rows]],
        [[r["error_mag"] for r in rows], [r["error_x"] for r in rows], [r["error_y"] for r in rows]],
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
    fig.suptitle(f"{title_prefix}DCEL-DoMINO vs GT needle deflection profile per step", fontsize=11)
    fig.tight_layout()
    overlay_path = os.path.join(out_dir, f"{prefix}deflection_overlay.png")
    fig.savefig(overlay_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {overlay_path}")

    return rows


# ---------------------------------------------------------------------------
# Summary plots
# ---------------------------------------------------------------------------

def _plot_summary(all_rows: dict, out_dir: str) -> None:
    """Generate cross-run summary plots from per-run row dicts."""
    run_ids = list(all_rows.keys())
    cmap = plt.cm.tab10
    colours = {rid: cmap(i % 10) for i, rid in enumerate(run_ids)}

    # ---- Tip magnitude -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for rid, rows in all_rows.items():
        steps = [r["step"] for r in rows]
        c = colours[rid]
        axes[0].plot(steps, [r["pred_mag"] for r in rows], color=c, lw=1.2, label=f"RUN-{rid}")
        axes[1].plot(steps, [r["gt_mag"]   for r in rows], color=c, lw=1.2, label=f"RUN-{rid}")
    for ax, title in zip(axes, ["Predicted tip deflection (mm)", "GT tip deflection (mm)"]):
        ax.set_xlabel("Rollout step")
        ax.set_ylabel("Magnitude (mm)")
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, lw=0.4, alpha=0.5)
    fig.suptitle("Tip deflection magnitude — all test runs", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "summary_tip_mag.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ---- Tip X/Y components --------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=False)
    for rid, rows in all_rows.items():
        steps = [r["step"] for r in rows]
        c = colours[rid]
        axes[0, 0].plot(steps, [r["pred_x"] for r in rows], color=c, lw=1.2, label=f"RUN-{rid}")
        axes[0, 1].plot(steps, [r["pred_y"] for r in rows], color=c, lw=1.2)
        axes[1, 0].plot(steps, [r["gt_x"]   for r in rows], color=c, lw=1.2)
        axes[1, 1].plot(steps, [r["gt_y"]   for r in rows], color=c, lw=1.2)
    titles = [
        ("Predicted tip X (mm)", "Predicted tip Y (mm)"),
        ("GT tip X (mm)",        "GT tip Y (mm)"),
    ]
    for row_axes, row_titles in zip(axes, titles):
        for ax, title in zip(row_axes, row_titles):
            ax.set_xlabel("Rollout step")
            ax.set_ylabel("Deflection (mm)")
            ax.set_title(title)
            ax.grid(True, lw=0.4, alpha=0.5)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Tip deflection X/Y components — all test runs", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "summary_tip_xy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ---- Error (pred − GT) ---------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=False)
    for rid, rows in all_rows.items():
        steps = [r["step"] for r in rows]
        c = colours[rid]
        axes[0].plot(steps, [r["error_mag"] for r in rows], color=c, lw=1.2, label=f"RUN-{rid}")
        axes[1].plot(steps, [r["error_x"]   for r in rows], color=c, lw=1.2)
        axes[2].plot(steps, [r["error_y"]   for r in rows], color=c, lw=1.2)
    for ax, ylabel in zip(axes, ["Error magnitude (mm)", "Error X (mm)", "Error Y (mm)"]):
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.set_xlabel("Rollout step")
        ax.set_ylabel(ylabel)
        ax.grid(True, lw=0.4, alpha=0.5)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Tip deflection error (pred − GT) — all test runs", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "summary_error.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Per-run worker (runs in a subprocess)
# ---------------------------------------------------------------------------

def _analyse_run_worker(
    run_id: str,
    infer_dir: str,
    run_out_dir: str,
    needle_idx: np.ndarray,
    ref_pos_needle: np.ndarray,
    sort_order: np.ndarray,
) -> tuple:
    """Analyse one inference run and return (run_id, rows, error_str)."""
    try:
        rows = analyze_run(
            infer_dir=infer_dir,
            needle_idx=needle_idx,
            ref_pos_needle=ref_pos_needle,
            sort_order=sort_order,
            out_dir=run_out_dir,
            run_label=run_id,
        )
        return run_id, rows, None
    except Exception as exc:  # noqa: BLE001
        return run_id, None, str(exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DCEL-DoMINO needle deflection across all test-set runs."
    )
    parser.add_argument("--data_dir", default=_DEFAULT_DATA_DIR,
                        help="Directory containing raw VTU files (defaults to config.yaml data_dir)")
    parser.add_argument("--infer_base_dir", default="./inference_output",
                        help="Parent dir containing RUN-<id>/ subdirs of predicted VTUs")
    parser.add_argument("--out_dir", default="./outputs/eval_test_runs",
                        help="Directory to write per-run outputs and summary")
    parser.add_argument("--train_fraction", type=float, default=_DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--val_fraction",   type=float, default=_DEFAULT_VAL_FRACTION)
    parser.add_argument("--timestep_stride", type=int, default=_DEFAULT_TIMESTEP_STRIDE)
    parser.add_argument("--grid_res", default=",".join(str(x) for x in _DEFAULT_GRID_RES),
                        help="Grid resolution used during training (to locate the DoMINO cache)")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(),
                        help="Number of parallel worker processes (default: cpu count)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    grid_res = tuple(int(x) for x in args.grid_res.split(","))

    # --- Identify test runs ---------------------------------------------------
    test_run_ids = _get_test_run_ids(
        args.data_dir, args.train_fraction, args.val_fraction, args.timestep_stride
    )
    print(f"Test run IDs ({len(test_run_ids)}): {test_run_ids}")

    # --- Shared needle geometry (same mesh topology across all runs) ----------
    needle_idx = _get_needle_indices(args.data_dir, grid_res)
    print(f"Found {len(needle_idx)} needle nodes.")
    vtu_files_all = _sorted_vtu_files(args.data_dir)
    ref_mesh = pv.read(vtu_files_all[0])
    ref_pos_needle = ref_mesh.points[needle_idx]
    principal = _principal_axis(ref_pos_needle)
    centred = ref_pos_needle - ref_pos_needle.mean(axis=0)
    sort_order = np.argsort(centred @ principal)

    # --- Build work list (skip runs whose infer dir is missing) ---------------
    skipped = []
    work: list = []
    for run_id in test_run_ids:
        infer_dir = os.path.join(args.infer_base_dir, f"RUN-{run_id}")
        if not os.path.isdir(infer_dir):
            print(f"[skip] RUN-{run_id}: {infer_dir} not found")
            skipped.append(run_id)
            continue
        run_out_dir = os.path.join(args.out_dir, f"RUN-{run_id}")
        work.append((run_id, infer_dir, run_out_dir, needle_idx, ref_pos_needle, sort_order))

    # --- Per-run analysis in parallel ----------------------------------------
    results: dict = {}
    n_workers = min(len(work), args.num_workers) if work else 1
    print(f"\nAnalysing {len(work)} runs with {n_workers} workers ...")

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_analyse_run_worker, *item): item[0]
            for item in work
        }
        for future in as_completed(futures):
            run_id, rows, err = future.result()
            if err is not None:
                print(f"[skip] RUN-{run_id}: {err}")
                skipped.append(run_id)
            else:
                print(f"  ✓ RUN-{run_id}")
                results[run_id] = rows

    # Restore insertion order for deterministic summary output.
    all_rows = {rid: results[rid] for rid in test_run_ids if rid in results}

    if not all_rows:
        print("\nNo runs analysed — check --infer_base_dir.")
        return

    if skipped:
        print(f"\nSkipped runs (no predicted VTUs): {skipped}")

    # --- Summary CSV ----------------------------------------------------------
    summary_path = os.path.join(args.out_dir, "summary.csv")
    fieldnames = [
        "run_id", "step",
        "pred_mag", "pred_x", "pred_y",
        "gt_mag",   "gt_x",   "gt_y",
        "error_mag", "error_x", "error_y",
        "has_gt",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run_id, rows in all_rows.items():
            for row in rows:
                writer.writerow({"run_id": run_id, **{k: row[k] for k in fieldnames[1:]}})
    print(f"\nSaved: {summary_path}")

    # --- Summary plots --------------------------------------------------------
    _plot_summary(all_rows, args.out_dir)

    print(f"\nDone. Analysed {len(all_rows)} runs.")


if __name__ == "__main__":
    main()
