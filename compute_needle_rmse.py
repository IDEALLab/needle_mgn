#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot the evolution of needle-position RMSE over the rollout.

For each predicted frame in
``experiments/<variant>/inference_output/RUN-*/predicted_*.vtu``:

  1. Read predicted needle positions ``mesh.points[needle_idx]`` and
     ground-truth needle positions ``mesh.point_data["Points_gt"][needle_idx]``
     (the GT array is written by ``infer.py`` whenever the matching dataset
     frame is available).
  2. Compute the per-frame node-position RMSE:

         RMSE_t = sqrt( mean_i ||pred_i(t) - gt_i(t)||^2 )

     over the 3-D position vector of every needle node (so a 1 mm uniform
     offset reads as 1 mm).
  3. Aggregate across all rollout steps and across all runs of an
     experiment.

Outputs (per experiment, written to ``experiments/<variant>/eval/``):
  - ``needle_rmse.csv``         — one row per (run_id, step) with frame RMSE
  - ``needle_rmse_summary.csv`` — one row per run with mean/max/last RMSE
  - ``needle_rmse.png``         — per-run thin lines + mean over runs

A top-level summary across all experiments is written to
``experiments/needle_rmse_overall.csv`` and the comparison curves go to
``experiments/needle_rmse_overall.png``.

Usage
-----
    uv run python compute_needle_rmse.py /path/to/RUN-2
    uv run python compute_needle_rmse.py /path/to/RUN-2 \
        --experiments cropped_base_mgn_wu cropped_fiber_iso_invw
"""

import argparse
import csv
import os
import re
import sys
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXAMPLE_DIR = _SCRIPT_DIR / "examples" / "cfd" / "needle_tissue_cropped"
sys.path.insert(0, str(_EXAMPLE_DIR))

from compare_deflection import _get_needle_indices  # noqa: E402


def _list_predicted_vtus(run_dir: Path) -> list:
    return sorted(
        glob(str(run_dir / "predicted_*.vtu")),
        key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)),
    )


def _frame_rmse(pred_pos: np.ndarray, gt_pos: np.ndarray) -> float:
    """Node-position RMSE over the needle: sqrt(mean(||pred - gt||^2))."""
    delta = pred_pos - gt_pos
    if not np.all(np.isfinite(delta)):
        return float("nan")
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def _process_experiment(
    exp_dir: Path,
    needle_idx: np.ndarray,
) -> dict:
    """Compute per-frame and per-run needle-position RMSE.

    Returns ``None`` if the experiment has no inference output.  Per-frame
    rows missing GT (``Points_gt`` not on the mesh) are skipped — the
    rollout step is recorded with ``rmse=NaN`` so the time axis still
    aligns across runs/experiments.
    """
    infer_root = exp_dir / "inference_output"
    if not infer_root.is_dir():
        return None

    run_dirs = sorted(p for p in infer_root.iterdir() if p.is_dir())
    if not run_dirs:
        return None

    per_frame_rows = []
    per_run_rows = []
    all_rmse = []

    for run_dir in run_dirs:
        run_id = run_dir.name
        files = _list_predicted_vtus(run_dir)
        if not files:
            continue

        run_rmse_list = []
        for step, fpath in enumerate(files):
            try:
                mesh = pv.read(fpath)
            except Exception as e:
                print(f"    [warn] {fpath}: {e}")
                continue
            pred_pos = mesh.points[needle_idx]
            if "Points_gt" in mesh.point_data:
                gt_pos = np.asarray(mesh.point_data["Points_gt"])[needle_idx]
                rmse = _frame_rmse(pred_pos, gt_pos)
            else:
                rmse = float("nan")
            run_rmse_list.append(rmse)
            per_frame_rows.append({
                "run_id": run_id,
                "step": step + 1,  # 1-indexed for plotting / consistency
                "rmse": rmse,
            })

        run_arr = np.asarray(run_rmse_list, dtype=float)
        finite = run_arr[np.isfinite(run_arr)]
        # rmse_last = RMSE at the run's final rollout step (NaN if missing).
        last = float(run_arr[-1]) if run_arr.size else float("nan")
        per_run_rows.append({
            "run_id": run_id,
            "n_steps": len(run_arr),
            "n_finite": int(finite.size),
            "rmse_mean": float(finite.mean()) if finite.size else float("nan"),
            "rmse_max": float(finite.max()) if finite.size else float("nan"),
            "rmse_last": last,
        })
        all_rmse.extend(finite.tolist())

    if not per_run_rows:
        return None

    arr = np.asarray(all_rmse, dtype=float)
    return {
        "per_frame": per_frame_rows,
        "per_run": per_run_rows,
        "total_mean": float(arr.mean()) if arr.size else float("nan"),
        "total_max": float(arr.max()) if arr.size else float("nan"),
        "n_runs": len(per_run_rows),
        "n_frames": int(arr.size),
    }


def _per_step_matrix(per_frame_rows: list) -> tuple:
    """Stack per-frame rows into a (n_runs, max_steps) array, NaN-padded.

    Steps are 1-indexed in ``per_frame_rows`` (column 0 corresponds to
    step 1).
    """
    by_run = {}
    for r in per_frame_rows:
        by_run.setdefault(r["run_id"], []).append((r["step"], r["rmse"]))
    run_ids = sorted(by_run.keys())
    max_step = max(s for rows in by_run.values() for s, _ in rows)
    mat = np.full((len(run_ids), max_step), np.nan, dtype=float)
    for i, rid in enumerate(run_ids):
        for s, v in by_run[rid]:
            mat[i, s - 1] = v
    return run_ids, mat


def _plot_experiment(
    out_path: Path, name: str, run_ids: list, mat: np.ndarray
) -> None:
    """Per-experiment plot: thin line per run + bold mean across runs."""
    steps = np.arange(1, mat.shape[1] + 1)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(mat, axis=0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, rid in enumerate(run_ids):
        ax.plot(
            steps, mat[i],
            color="C0", alpha=0.3, linewidth=0.8,
            label=rid if i == 0 else None,
        )
    ax.plot(steps, mean, color="C3", linewidth=2.0, label="mean across runs")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("needle-position RMSE (mm)")
    ax.set_title(f"{name}: needle RMSE vs rollout step")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_overall(out_path: Path, exp_curves: dict) -> None:
    """One mean curve per experiment, log y-scale to compare growth rates."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, mean in exp_curves.items():
        steps = np.arange(1, len(mean) + 1)
        axes[0].plot(steps, mean, label=name, linewidth=1.2)
        axes[1].plot(steps, mean, label=name, linewidth=1.2)
    for ax in axes:
        ax.set_xlabel("rollout step")
        ax.set_ylabel("needle-position RMSE (mm), mean over runs")
        ax.grid(True, alpha=0.3, which="both")
    axes[0].set_title("Linear scale")
    axes[1].set_title("Log scale")
    axes[1].set_yscale("log")
    axes[1].legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _write_csv(path: Path, rows: list, fieldnames: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main():
    parser = argparse.ArgumentParser(
        description="Plot evolution of needle-position RMSE over rollout for "
                    "every experiment under experiments/*/inference_output/."
    )
    parser.add_argument(
        "data_dir",
        help="Directory of raw VTU simulation files (used to identify "
             "the needle node indices via the topology cache).",
    )
    parser.add_argument(
        "--experiments_dir", default=None,
        help="Root containing experiment subdirs (default: <project>/experiments)",
    )
    parser.add_argument(
        "--experiments", nargs="+", default=None,
        help="Explicit list of experiment names (default: every subdir of "
             "experiments_dir that has inference_output/).",
    )
    args = parser.parse_args()

    data_dir = os.path.realpath(args.data_dir)
    if not os.path.isdir(data_dir):
        print(f"ERROR: data_dir not found: {data_dir}")
        sys.exit(1)

    project_dir = _SCRIPT_DIR
    experiments_dir = (
        Path(args.experiments_dir) if args.experiments_dir
        else project_dir / "experiments"
    )
    if not experiments_dir.is_dir():
        print(f"ERROR: experiments_dir not found: {experiments_dir}")
        sys.exit(1)

    needle_idx = _get_needle_indices(data_dir)
    print(f"Needle nodes: {len(needle_idx)}")

    if args.experiments is not None:
        exp_dirs = []
        for name in args.experiments:
            p = Path(name)
            if not p.is_absolute():
                p = experiments_dir / name
            exp_dirs.append(p)
    else:
        exp_dirs = sorted(
            p for p in experiments_dir.iterdir()
            if p.is_dir() and (p / "inference_output").is_dir()
        )

    if not exp_dirs:
        print(f"No experiments with inference_output/ found in {experiments_dir}")
        sys.exit(1)

    print(f"Processing {len(exp_dirs)} experiment(s)\n")

    overall_rows = []
    overall_curves = {}
    for exp_dir in exp_dirs:
        name = exp_dir.name
        print(f"=== {name} ===")
        if not (exp_dir / "inference_output").is_dir():
            print("  [SKIP] no inference_output/")
            continue

        result = _process_experiment(exp_dir, needle_idx)
        if result is None:
            print("  [SKIP] no rollouts found")
            continue

        eval_dir = exp_dir / "eval"
        _write_csv(
            eval_dir / "needle_rmse.csv",
            result["per_frame"],
            ["run_id", "step", "rmse"],
        )
        _write_csv(
            eval_dir / "needle_rmse_summary.csv",
            result["per_run"],
            ["run_id", "n_steps", "n_finite",
             "rmse_mean", "rmse_max", "rmse_last"],
        )
        run_ids, mat = _per_step_matrix(result["per_frame"])
        _plot_experiment(eval_dir / "needle_rmse.png", name, run_ids, mat)
        with np.errstate(invalid="ignore"):
            mean_curve = np.nanmean(mat, axis=0)
        overall_curves[name] = mean_curve

        last_mean = (
            float(mean_curve[-1]) if np.isfinite(mean_curve[-1])
            else float("nan")
        )
        print(
            f"  runs={result['n_runs']}  frames={result['n_frames']}  "
            f"mean={result['total_mean']:.4e}  max={result['total_max']:.4e}  "
            f"final-step mean={last_mean:.4e}"
        )
        overall_rows.append({
            "experiment": name,
            "n_runs": result["n_runs"],
            "n_frames": result["n_frames"],
            "rmse_mean": result["total_mean"],
            "rmse_max": result["total_max"],
            "rmse_final_mean": last_mean,
        })

    if overall_rows:
        overall_rows.sort(key=lambda r: r["rmse_mean"])
        out_path = experiments_dir / "needle_rmse_overall.csv"
        _write_csv(
            out_path, overall_rows,
            ["experiment", "n_runs", "n_frames",
             "rmse_mean", "rmse_max", "rmse_final_mean"],
        )
        plot_path = experiments_dir / "needle_rmse_overall.png"
        _plot_overall(plot_path, overall_curves)
        print(f"\nOverall plot: {plot_path}")
        print(f"Overall summary (sorted by mean RMSE): {out_path}")
        print(f"  {'experiment':40s}  {'mean':>10s}  {'max':>10s}  {'final':>10s}")
        for r in overall_rows:
            print(
                f"  {r['experiment']:40s}  {r['rmse_mean']:10.4e}  "
                f"{r['rmse_max']:10.4e}  {r['rmse_final_mean']:10.4e}"
            )


if __name__ == "__main__":
    main()
