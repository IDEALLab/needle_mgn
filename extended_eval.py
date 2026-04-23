#!/usr/bin/env python3
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

"""Extended evaluation metrics across trained experiments.

Reads the eval/summary.csv produced by eval_test_runs.py for each experiment
and computes three additional metrics:

  1. Tip deflection percent error
       |pred_mag - gt_mag| / gt_mag * 100
       (only for steps where gt_mag > --mag_threshold)

  2. X-Y plane angle error
       Angle between the (pred_x, pred_y) and (gt_x, gt_y) vectors, in degrees.
       Absolute (degrees) and relative (% of 180°).
       (only for steps where both pred and gt vectors exceed --mag_threshold)

  3. P-value vs no-deflection baseline
       The "no-deflection" prediction is zero displacement at every step.
       Its error at each step is gt_mag (distance from zero to GT tip).
       A one-sided Wilcoxon signed-rank test compares model error_mag to
       no-deflection error (gt_mag), testing whether the model is significantly
       better.  A paired t-test is also reported.
       Effect size is Cohen's d on the per-step improvement.

Outputs (written to --out_dir):
  experiment_metrics.csv  — per-step metrics for all experiments combined
  summary_stats.csv       — per-experiment mean/median/std of each metric
  pvalue_table.csv        — p-values and effect sizes vs no-deflection baseline
  plot_pct_error.svg      — percent error over rollout steps
  plot_angle_error.svg    — angle error over rollout steps
  plot_summary_bars.svg   — bar chart of mean metrics per experiment
  plot_pvalue_bars.svg    — significance vs no-deflection baseline

Usage:
    uv run python extended_eval.py --experiments_dir experiments/
    uv run python extended_eval.py \\
        --experiments experiments/cropped_base experiments/domino_base \\
        --out_dir results/extended_eval
"""

import argparse
import csv
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_summary(exp_dir: str) -> pd.DataFrame | None:
    """Load eval/summary.csv from an experiment directory.  Returns None if missing."""
    path = os.path.join(exp_dir, "eval", "summary.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    df["has_gt"] = df["has_gt"].astype(str).str.lower() == "true"
    return df


def _angle_between_2d(
    ax: np.ndarray, ay: np.ndarray,
    bx: np.ndarray, by: np.ndarray,
) -> np.ndarray:
    """Angle (degrees) between 2-D vectors (ax, ay) and (bx, by), element-wise.

    Returns NaN where either vector is near-zero.
    """
    dot   = ax * bx + ay * by
    mag_a = np.sqrt(ax**2 + ay**2)
    mag_b = np.sqrt(bx**2 + by**2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cos_theta = dot / (mag_a * mag_b)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_theta))
    zero = (mag_a < 1e-9) | (mag_b < 1e-9)
    angle_deg[zero] = np.nan
    return angle_deg


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired samples: d = mean(a - b) / std(a - b)."""
    diff = a - b
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("nan")


# ---------------------------------------------------------------------------
# Per-experiment metric computation
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame, mag_threshold: float) -> pd.DataFrame:
    """Add derived metric columns to a summary dataframe.

    Parameters
    ----------
    df : DataFrame with columns from eval/summary.csv
    mag_threshold : minimum gt_mag (mm) to include a step in percentage/angle metrics

    Returns
    -------
    DataFrame with additional columns:
        euclid_error        — sqrt(error_x² + error_y²): true 2-D Euclidean tip error
        no_deflection_error — GT magnitude (equals Euclidean error for a zero prediction)
        pct_error           — euclid_error / gt_mag * 100
        angle_error_deg     — angle between pred and GT x-y vectors (degrees)
        angle_error_pct     — angle_error_deg / 180 * 100

    Note: ``error_mag`` in the CSV is the *signed* difference (pred_mag − gt_mag),
    not the Euclidean distance.  ``euclid_error`` is the proper absolute error.
    """
    df = df.copy()

    # True 2-D Euclidean tip error: distance from predicted tip to GT tip in x-y plane
    df["euclid_error"] = np.sqrt(df["error_x"] ** 2 + df["error_y"] ** 2)

    # No-deflection baseline error: predicting zero gives Euclidean error equal to GT magnitude
    df["no_deflection_error"] = df["gt_mag"]

    # Percent error (only where GT deflection is large enough to be meaningful)
    valid_mag = df["gt_mag"] > mag_threshold
    df["pct_error"] = np.where(
        valid_mag,
        (df["euclid_error"] / df["gt_mag"]) * 100.0,
        np.nan,
    )

    # X-Y plane angle error
    valid_angle = valid_mag & (np.sqrt(df["pred_x"]**2 + df["pred_y"]**2) > mag_threshold)
    df["angle_error_deg"] = np.nan
    if valid_angle.any():
        df.loc[valid_angle, "angle_error_deg"] = _angle_between_2d(
            df.loc[valid_angle, "pred_x"].to_numpy(),
            df.loc[valid_angle, "pred_y"].to_numpy(),
            df.loc[valid_angle, "gt_x"].to_numpy(),
            df.loc[valid_angle, "gt_y"].to_numpy(),
        )
    df["angle_error_pct"] = df["angle_error_deg"] / 180.0 * 100.0

    return df


# ---------------------------------------------------------------------------
# Statistical tests vs no-deflection baseline
# ---------------------------------------------------------------------------

def significance_vs_baseline(df: pd.DataFrame, exp_name: str) -> dict:
    """Test whether a model is significantly better than no-deflection prediction.

    Uses only GT-available steps.  Tests H0: model error >= no-deflection error
    (one-tailed alternative='less').

    Parameters
    ----------
    df : metrics DataFrame for one experiment (output of compute_metrics)
    exp_name : label for the result dict

    Returns
    -------
    dict with test statistics
    """
    sub = df[df["has_gt"]].dropna(subset=["euclid_error", "no_deflection_error"])
    if len(sub) < 8:
        return {
            "experiment": exp_name, "n_steps": len(sub),
            "mean_model_error_mm": np.nan, "mean_baseline_error_mm": np.nan,
            "mean_improvement_mm": np.nan, "pct_improvement": np.nan,
            "cohens_d": np.nan,
            "wilcoxon_stat": np.nan, "wilcoxon_p": np.nan,
            "ttest_stat": np.nan, "ttest_p": np.nan,
            "note": "insufficient data",
        }

    # Use Euclidean error (always non-negative) vs baseline (gt_mag, also non-negative)
    model_err    = sub["euclid_error"].to_numpy()
    baseline_err = sub["no_deflection_error"].to_numpy()
    improvement  = baseline_err - model_err  # positive = model is better

    mean_model    = float(model_err.mean())
    mean_baseline = float(baseline_err.mean())
    mean_improv   = float(improvement.mean())
    pct_improv    = float(mean_improv / mean_baseline * 100) if mean_baseline > 0 else np.nan
    d             = _cohens_d(baseline_err, model_err)  # larger = bigger improvement

    # Wilcoxon signed-rank test (non-parametric, paired, one-sided)
    try:
        w_stat, w_p = stats.wilcoxon(model_err, baseline_err, alternative="less")
    except ValueError as e:
        w_stat, w_p = np.nan, np.nan

    # Paired t-test (parametric, one-sided)
    t_stat, t_p = stats.ttest_rel(model_err, baseline_err, alternative="less")

    return {
        "experiment":            exp_name,
        "n_steps":               len(sub),
        "mean_model_error_mm":   round(mean_model,    4),
        "mean_baseline_error_mm":round(mean_baseline, 4),
        "mean_improvement_mm":   round(mean_improv,   4),
        "pct_improvement":       round(pct_improv,    2),
        "cohens_d":              round(d,              3),
        "wilcoxon_stat":         round(float(w_stat), 4) if not np.isnan(w_stat) else np.nan,
        "wilcoxon_p":            float(w_p),
        "ttest_stat":            round(float(t_stat), 4),
        "ttest_p":               float(t_p),
        "note": "",
    }


# ---------------------------------------------------------------------------
# Summary statistics per experiment
# ---------------------------------------------------------------------------

def per_experiment_stats(df: pd.DataFrame, exp_name: str) -> dict:
    """Compute aggregate statistics for one experiment."""
    sub = df[df["has_gt"]]

    def _stats(col: str) -> tuple:
        s = sub[col].dropna()
        if len(s) == 0:
            return np.nan, np.nan, np.nan
        return float(s.mean()), float(s.median()), float(s.std(ddof=1))

    err_mean, err_med, err_std           = _stats("euclid_error")
    pct_mean, pct_med, pct_std           = _stats("pct_error")
    ang_mean, ang_med, ang_std           = _stats("angle_error_deg")
    angp_mean, angp_med, angp_std        = _stats("angle_error_pct")
    base_mean, _,       _                = _stats("no_deflection_error")

    return {
        "experiment":             exp_name,
        "n_steps":                len(sub),
        "error_mag_mean_mm":      round(err_mean, 4),
        "error_mag_median_mm":    round(err_med,  4),
        "error_mag_std_mm":       round(err_std,  4),
        "pct_error_mean":         round(pct_mean, 2),
        "pct_error_median":       round(pct_med,  2),
        "pct_error_std":          round(pct_std,  2),
        "angle_error_deg_mean":   round(ang_mean, 2),
        "angle_error_deg_median": round(ang_med,  2),
        "angle_error_deg_std":    round(ang_std,  2),
        "angle_error_pct_mean":   round(angp_mean, 2),
        "angle_error_pct_median": round(angp_med,  2),
        "angle_error_pct_std":    round(angp_std,  2),
        "baseline_error_mean_mm": round(base_mean, 4),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

_COLORS = plt.cm.tab20.colors


def _mean_by_step(df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, mean_values) averaged across runs for one metric column."""
    g = df[df["has_gt"]].groupby("step")[col]
    mean = g.mean()
    return mean.index.to_numpy(), mean.to_numpy()


def _steps_to_depth(steps: np.ndarray, total_mm: float) -> np.ndarray:
    """Convert step indices to insertion depth (mm).

    Each experiment covers the same physical insertion (total_mm) regardless
    of temporal stride.  We normalise by the maximum step index so that the
    last step always maps to total_mm, making stride-10 and stride-1 rollouts
    directly comparable on the same x-axis.
    """
    if len(steps) == 0 or steps.max() == 0:
        return steps.astype(float)
    return steps / steps.max() * total_mm


def plot_pct_error(
    all_dfs: dict[str, pd.DataFrame],
    out_path: str,
    total_insertion_mm: float = 40.0,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, df) in enumerate(all_dfs.items()):
        steps, vals = _mean_by_step(df, "pct_error")
        depth = _steps_to_depth(steps, total_insertion_mm)
        ax.plot(depth, vals, color=_COLORS[i % 20], lw=1.5, label=name)
    ax.set_xlabel("Insertion depth (mm)")
    ax.set_ylabel("Tip deflection percent error (%)")
    ax.set_title("Tip deflection percent error — mean across test runs")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, lw=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_angle_error(
    all_dfs: dict[str, pd.DataFrame],
    out_path: str,
    total_insertion_mm: float = 40.0,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i, (name, df) in enumerate(all_dfs.items()):
        c = _COLORS[i % 20]
        steps, vals_deg = _mean_by_step(df, "angle_error_deg")
        _, vals_pct     = _mean_by_step(df, "angle_error_pct")
        depth = _steps_to_depth(steps, total_insertion_mm)
        axes[0].plot(depth, vals_deg, color=c, lw=1.5, label=name)
        axes[1].plot(depth, vals_pct, color=c, lw=1.5, label=name)
    axes[0].set_title("Angle error (degrees)")
    axes[1].set_title("Angle error (% of 180°)")
    for ax in axes:
        ax.set_xlabel("Insertion depth (mm)")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, lw=0.4, alpha=0.5)
    axes[0].set_ylabel("Angle error (°)")
    axes[1].set_ylabel("Angle error (%)")
    fig.suptitle("X-Y plane angle error — mean across test runs", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_summary_bars(summary_rows: list[dict], out_path: str) -> None:
    names    = [r["experiment"]           for r in summary_rows]
    err_mean = [r["error_mag_mean_mm"]    for r in summary_rows]
    err_std  = [r["error_mag_std_mm"]     for r in summary_rows]
    pct_mean = [r["pct_error_mean"]       for r in summary_rows]
    pct_std  = [r["pct_error_std"]        for r in summary_rows]
    ang_mean = [r["angle_error_deg_mean"] for r in summary_rows]
    ang_std  = [r["angle_error_deg_std"]  for r in summary_rows]
    base     = [r["baseline_error_mean_mm"] for r in summary_rows]

    x    = np.arange(len(names))
    w    = 0.6
    cols = [_COLORS[i % 20] for i in range(len(names))]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: absolute error (mm) with no-deflection baseline
    axes[0].bar(x, err_mean, w, yerr=err_std, capsize=4,
                color=cols, alpha=0.85, label="Model")
    axes[0].hlines(np.nanmean(base), x[0] - 0.5, x[-1] + 0.5,
                   colors="k", linestyles="--", lw=1.2, label="No-deflection baseline")
    axes[0].set_title("Mean tip error (mm)")
    axes[0].set_ylabel("Error (mm)")
    axes[0].legend(fontsize=8)

    # Panel 2: percent error
    axes[1].bar(x, pct_mean, w, yerr=pct_std, capsize=4, color=cols, alpha=0.85)
    axes[1].set_title("Mean tip percent error (%)")
    axes[1].set_ylabel("Percent error (%)")

    # Panel 3: angle error
    axes[2].bar(x, ang_mean, w, yerr=ang_std, capsize=4, color=cols, alpha=0.85)
    axes[2].set_title("Mean angle error (°)")
    axes[2].set_ylabel("Angle error (°)")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax.grid(axis="y", lw=0.4, alpha=0.5)

    fig.suptitle("Experiment summary — mean ± std across all test steps", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_pvalue_bars(pvalue_rows: list[dict], out_path: str) -> None:
    """Bar chart of -log10(p) for each experiment, with significance thresholds."""
    names   = [r["experiment"] for r in pvalue_rows]
    w_p     = [r["wilcoxon_p"] for r in pvalue_rows]
    t_p     = [r["ttest_p"]    for r in pvalue_rows]
    improv  = [r["pct_improvement"] for r in pvalue_rows]

    x  = np.arange(len(names))
    bw = 0.35

    def _neg_log10(vals):
        out = []
        for v in vals:
            try:
                out.append(-np.log10(max(float(v), 1e-300)))
            except (TypeError, ValueError):
                out.append(0.0)
        return np.array(out)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: -log10(p) for both tests
    nl_w = _neg_log10(w_p)
    nl_t = _neg_log10(t_p)
    axes[0].bar(x - bw/2, nl_w, bw, label="Wilcoxon (non-param.)", alpha=0.85)
    axes[0].bar(x + bw/2, nl_t, bw, label="Paired t-test",          alpha=0.85)
    for thresh, label, ls in [
        (1.301, "p=0.05", "--"), (2.0, "p=0.01", ":"), (3.0, "p=0.001", "-.")
    ]:
        axes[0].axhline(thresh, color="k", lw=0.9, ls=ls, label=label)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    axes[0].set_ylabel("-log₁₀(p)  [higher = more significant]")
    axes[0].set_title("Significance vs no-deflection baseline\n(one-sided, H₁: model error < baseline error)")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", lw=0.4, alpha=0.5)

    # Panel 2: percent improvement over baseline
    valid_improv = [v if not (isinstance(v, float) and np.isnan(v)) else 0 for v in improv]
    bars = axes[1].bar(x, valid_improv, 0.6,
                       color=[_COLORS[i % 20] for i in range(len(names))], alpha=0.85)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    axes[1].set_ylabel("Mean error reduction vs baseline (%)")
    axes[1].set_title("Error improvement over no-deflection baseline")
    axes[1].grid(axis="y", lw=0.4, alpha=0.5)

    # Annotate with actual p-values
    for i, (bar, pv) in enumerate(zip(bars, w_p)):
        try:
            label = f"p={float(pv):.3f}" if float(pv) >= 0.001 else f"p<0.001"
        except (TypeError, ValueError):
            label = "n/a"
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.3,
                     label, ha="center", va="bottom", fontsize=7)

    fig.suptitle("Statistical significance vs no-deflection baseline", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extended evaluation metrics across trained experiments."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--experiments_dir",
        help="Directory containing experiment subdirectories (auto-discovers all with eval/summary.csv)",
    )
    group.add_argument(
        "--experiments", nargs="+",
        help="Explicit list of experiment directories",
    )
    parser.add_argument(
        "--out_dir", default="results/extended_eval",
        help="Output directory for CSVs and plots (default: results/extended_eval)",
    )
    parser.add_argument(
        "--mag_threshold", type=float, default=0.05,
        help="Minimum GT tip deflection magnitude (mm) to include a step in "
             "percent-error and angle-error calculations (default: 0.05)",
    )
    parser.add_argument(
        "--total_insertion_mm", type=float, default=40.0,
        help="Total needle insertion depth (mm) used to convert rollout steps to "
             "physical depth on the x-axis of time-series plots (default: 40.0)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Discover experiment directories -------------------------------------
    if args.experiments_dir:
        base = args.experiments_dir
        candidates = sorted(
            os.path.join(base, d) for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))
        )
    else:
        candidates = args.experiments

    # Load and filter to experiments that have a summary CSV
    all_dfs: dict[str, pd.DataFrame] = {}
    for exp_dir in candidates:
        name = os.path.basename(exp_dir.rstrip("/"))
        df = _load_summary(exp_dir)
        if df is None:
            print(f"[skip] {name}: no eval/summary.csv found")
            continue
        df = compute_metrics(df, args.mag_threshold)
        all_dfs[name] = df
        print(f"Loaded {name}: {len(df)} rows, {df['has_gt'].sum()} with GT")

    if not all_dfs:
        print("No experiments with eval/summary.csv found. Run eval_test_runs.py first.")
        return

    print(f"\nComputing metrics for {len(all_dfs)} experiments ...\n")

    # --- Per-experiment summary stats ----------------------------------------
    summary_rows = [per_experiment_stats(df, name) for name, df in all_dfs.items()]

    summary_path = os.path.join(args.out_dir, "summary_stats.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved: {summary_path}")

    # --- P-values vs no-deflection baseline ----------------------------------
    pvalue_rows = [significance_vs_baseline(df, name) for name, df in all_dfs.items()]

    pvalue_path = os.path.join(args.out_dir, "pvalue_table.csv")
    with open(pvalue_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pvalue_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pvalue_rows)
    print(f"Saved: {pvalue_path}")

    # Print p-value table to stdout
    print("\n--- Significance vs no-deflection baseline ---")
    print(f"{'Experiment':<25} {'n':>5}  {'Model err':>10}  {'Baseline':>10}  "
          f"{'Improv%':>8}  {'Cohen d':>8}  {'Wilcoxon p':>11}  {'t-test p':>9}")
    print("-" * 95)
    for r in pvalue_rows:
        def _fmt(v, fmt=".4f"):
            try:
                return format(float(v), fmt)
            except (TypeError, ValueError):
                return "n/a"
        print(f"{r['experiment']:<25} {r['n_steps']:>5}  "
              f"{_fmt(r['mean_model_error_mm']):>10}  "
              f"{_fmt(r['mean_baseline_error_mm']):>10}  "
              f"{_fmt(r['pct_improvement'], '.1f'):>8}%  "
              f"{_fmt(r['cohens_d'], '.3f'):>8}  "
              f"{_fmt(r['wilcoxon_p'], '.2e'):>11}  "
              f"{_fmt(r['ttest_p'], '.2e'):>9}")

    # --- Combined per-step CSV -----------------------------------------------
    combined_parts = []
    for name, df in all_dfs.items():
        d = df.copy()
        d.insert(0, "experiment", name)
        combined_parts.append(d)
    combined = pd.concat(combined_parts, ignore_index=True)
    metrics_path = os.path.join(args.out_dir, "experiment_metrics.csv")
    combined.to_csv(metrics_path, index=False)
    print(f"\nSaved: {metrics_path}")

    # --- Plots ---------------------------------------------------------------
    plot_pct_error(all_dfs,   os.path.join(args.out_dir, "plot_pct_error.svg"),
                   total_insertion_mm=args.total_insertion_mm)
    plot_angle_error(all_dfs, os.path.join(args.out_dir, "plot_angle_error.svg"),
                     total_insertion_mm=args.total_insertion_mm)
    plot_summary_bars(summary_rows, os.path.join(args.out_dir, "plot_summary_bars.svg"))
    plot_pvalue_bars(pvalue_rows,   os.path.join(args.out_dir, "plot_pvalue_bars.svg"))

    print(f"\nAll outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()
