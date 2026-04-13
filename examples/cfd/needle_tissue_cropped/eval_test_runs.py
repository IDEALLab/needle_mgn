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

"""Evaluate needle deflection across all test-set runs.

Uses the same train/val/test split as train.py and infer.py to identify test
runs, then calls the compare_deflection analysis for each run that has
corresponding predicted VTU files.

Expected directory layout (produced by running infer.py with different
infer_run_id and infer_output_dir values):

    <infer_base_dir>/
        RUN-140/
            predicted_0000.vtu
            predicted_0001.vtu
            ...
        RUN-141/
            predicted_0000.vtu
            ...

Outputs:
  <out_dir>/
      RUN-<id>/          — per-run plots + tip_deflection.csv
      summary.csv        — all runs combined (run_id, step, metrics...)
      summary_tip_mag.png — tip deflection magnitude across runs
      summary_tip_xy.png  — x/y tip deflection across runs
      summary_error.png   — tip deflection error across runs

Usage:
    uv run eval_test_runs.py \\
        --data_dir /path/to/RUN-2 \\
        --infer_base_dir ./outputs/inference_output \\
        --out_dir ./outputs/eval_test_runs
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from compare_deflection import (
    _get_needle_indices,
    _principal_axis,
    analyze_run,
    _write_csv,
)
from dataset import _group_vtu_by_run, _sorted_vtu_files


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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate needle deflection across all test-set runs."
    )
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing raw VTU files (multi-run)")
    parser.add_argument("--infer_base_dir", required=True,
                        help="Parent dir containing RUN-<id>/ subdirs of predicted VTUs")
    parser.add_argument("--out_dir", default="./outputs/eval_test_runs",
                        help="Directory to write per-run outputs and summary")
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--val_fraction",   type=float, default=0.1)
    parser.add_argument("--timestep_stride", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Identify test runs ---------------------------------------------------
    test_run_ids = _get_test_run_ids(
        args.data_dir, args.train_fraction, args.val_fraction, args.timestep_stride
    )
    print(f"Test run IDs ({len(test_run_ids)}): {test_run_ids}")

    # --- Shared needle geometry (same mesh topology across all runs) ----------
    needle_idx = _get_needle_indices(args.data_dir)
    print(f"Found {len(needle_idx)} needle nodes.")
    vtu_files_all = _sorted_vtu_files(args.data_dir)
    ref_mesh = pv.read(vtu_files_all[0])
    ref_pos_needle = ref_mesh.points[needle_idx]
    principal = _principal_axis(ref_pos_needle)
    centred = ref_pos_needle - ref_pos_needle.mean(axis=0)
    axial_coords = centred @ principal
    sort_order = np.argsort(axial_coords)

    # --- Per-run analysis -----------------------------------------------------
    all_rows: dict = {}
    skipped = []

    for run_id in test_run_ids:
        infer_dir = os.path.join(args.infer_base_dir, f"RUN-{run_id}")
        if not os.path.isdir(infer_dir):
            print(f"[skip] RUN-{run_id}: {infer_dir} not found")
            skipped.append(run_id)
            continue

        run_out_dir = os.path.join(args.out_dir, f"RUN-{run_id}")
        print(f"\nAnalysing RUN-{run_id} ...")
        try:
            rows = analyze_run(
                infer_dir=infer_dir,
                needle_idx=needle_idx,
                ref_pos_needle=ref_pos_needle,
                sort_order=sort_order,
                out_dir=run_out_dir,
                run_label=run_id,
            )
            all_rows[run_id] = rows
        except FileNotFoundError as exc:
            print(f"[skip] RUN-{run_id}: {exc}")
            skipped.append(run_id)

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
