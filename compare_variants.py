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

"""Compare variant experiments against their base model.

For each (run_id, step) pair that appears in both the base and variant
experiments the Euclidean tip error is computed:

    euclid_error = sqrt(error_x² + error_y²)

Paired tests are then run to answer:
  "Is the variant significantly better/worse than the base?"

Tests performed (all two-sided so that improvements and regressions are
both visible; the sign of the mean difference tells you the direction):

  * Wilcoxon signed-rank test  — non-parametric, paired
  * Paired t-test              — parametric, paired
  * Cohen's d on per-step difference (variant − base)

Outputs written to --out_dir:
  variant_comparison.csv     — one row per (base, variant) pair
  plot_variant_pvalues.png   — -log10(p) bar chart for all comparisons
  plot_variant_errors.png    — bar chart of mean error ± std per experiment

Usage:
    uv run python compare_variants.py --experiments_dir experiments/
    uv run python compare_variants.py \\
        --comparisons cropped_base:cropped_noise cropped_base:cropped_fourier \\
        --out_dir results/variant_comparison
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
# Default comparison pairs (base → list of variants)
# ---------------------------------------------------------------------------

_DEFAULT_PAIRS = [
    ("cropped_base", "cropped_noise"),
    ("cropped_base", "cropped_fourier"),
    ("cropped_base", "cropped_cpress"),
    ("domino_base",  "domino_noise"),
    ("domino_base",  "domino_fourier"),
    ("domino_base",  "domino_cpress"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_summary(exp_dir: str) -> pd.DataFrame | None:
    """Load eval/summary.csv, returning None if missing."""
    path = os.path.join(exp_dir, "eval", "summary.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    df["has_gt"] = df["has_gt"].astype(str).str.lower() == "true"
    df["euclid_error"] = np.sqrt(df["error_x"] ** 2 + df["error_y"] ** 2)
    return df


def _cohens_d(diff: np.ndarray) -> float:
    """Cohen's d for a paired-difference array d = variant − base.

    Positive d → variant error is larger (worse).
    Negative d → variant error is smaller (better).
    """
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("nan")


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------

def compare_pair(
    base_df: pd.DataFrame,
    var_df: pd.DataFrame,
    base_name: str,
    var_name: str,
) -> dict:
    """Run paired statistical tests between base and variant experiments.

    Observations are matched on (run_id, step).  Only GT-available steps
    present in *both* DataFrames are used.

    Returns
    -------
    dict with test statistics and a human-readable note.
    """
    base_gt = base_df[base_df["has_gt"]][["run_id", "step", "euclid_error"]].rename(
        columns={"euclid_error": "base_err"}
    )
    var_gt = var_df[var_df["has_gt"]][["run_id", "step", "euclid_error"]].rename(
        columns={"euclid_error": "var_err"}
    )

    merged = base_gt.merge(var_gt, on=["run_id", "step"], how="inner")
    n = len(merged)

    empty = {
        "base": base_name,
        "variant": var_name,
        "n_paired_steps": n,
        "mean_base_error_mm": np.nan,
        "mean_variant_error_mm": np.nan,
        "mean_diff_mm": np.nan,
        "pct_change": np.nan,
        "cohens_d": np.nan,
        "wilcoxon_stat": np.nan,
        "wilcoxon_p": np.nan,
        "ttest_stat": np.nan,
        "ttest_p": np.nan,
        "note": "",
    }

    if n < 8:
        empty["note"] = "insufficient paired data"
        return empty

    base_err = merged["base_err"].to_numpy()
    var_err  = merged["var_err"].to_numpy()
    diff = var_err - base_err        # positive → variant is worse

    mean_base = float(base_err.mean())
    mean_var  = float(var_err.mean())
    mean_diff = float(diff.mean())
    pct_change = float(mean_diff / mean_base * 100) if mean_base > 0 else np.nan
    d = _cohens_d(diff)

    # Two-sided: we want to detect both improvements and regressions
    try:
        w_stat, w_p = stats.wilcoxon(var_err, base_err, alternative="two-sided")
    except ValueError:
        w_stat, w_p = np.nan, np.nan

    t_stat, t_p = stats.ttest_rel(var_err, base_err, alternative="two-sided")

    return {
        "base":                  base_name,
        "variant":               var_name,
        "n_paired_steps":        n,
        "mean_base_error_mm":    round(mean_base,  4),
        "mean_variant_error_mm": round(mean_var,   4),
        "mean_diff_mm":          round(mean_diff,  4),
        "pct_change":            round(pct_change, 2),
        "cohens_d":              round(d,           3),
        "wilcoxon_stat":         round(float(w_stat), 4) if not np.isnan(w_stat) else np.nan,
        "wilcoxon_p":            float(w_p),
        "ttest_stat":            round(float(t_stat), 4),
        "ttest_p":               float(t_p),
        "note": "",
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

_COLORS = plt.cm.tab10.colors


def plot_pvalue_bars(rows: list[dict], out_path: str) -> None:
    """Bar chart of -log10(p) for each variant vs its base model."""
    labels  = [f"{r['variant']}\nvs {r['base']}" for r in rows]
    w_p     = [r["wilcoxon_p"] for r in rows]
    t_p     = [r["ttest_p"]    for r in rows]
    pct_chg = [r["pct_change"] for r in rows]

    x  = np.arange(len(labels))
    bw = 0.35

    def _neg_log10(vals):
        out = []
        for v in vals:
            try:
                out.append(-np.log10(max(float(v), 1e-300)))
            except (TypeError, ValueError):
                out.append(0.0)
        return np.array(out)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: -log10(p)
    nl_w = _neg_log10(w_p)
    nl_t = _neg_log10(t_p)
    axes[0].bar(x - bw / 2, nl_w, bw, label="Wilcoxon (non-param.)", alpha=0.85)
    axes[0].bar(x + bw / 2, nl_t, bw, label="Paired t-test",          alpha=0.85)
    for thresh, lbl, ls in [
        (1.301, "p=0.05", "--"), (2.0, "p=0.01", ":"), (3.0, "p=0.001", "-.")
    ]:
        axes[0].axhline(thresh, color="k", lw=0.9, ls=ls, label=lbl)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("-log₁₀(p)  [higher = more significant]")
    axes[0].set_title("Variant vs base significance\n(two-sided paired test on Euclidean tip error)")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", lw=0.4, alpha=0.5)

    # Panel 2: percent change in mean error (negative = improvement)
    colors = [("tab:green" if v < 0 else "tab:red") for v in pct_chg]
    bars = axes[1].bar(x, pct_chg, 0.6, color=colors, alpha=0.85)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("Mean error change vs base (%)\n[negative = variant is better]")
    axes[1].set_title("Error change: variant relative to base")
    axes[1].grid(axis="y", lw=0.4, alpha=0.5)

    # Annotate bars with p-value
    for bar, pv in zip(bars, w_p):
        try:
            lbl = f"p={float(pv):.3f}" if float(pv) >= 0.001 else "p<0.001"
        except (TypeError, ValueError):
            lbl = "n/a"
        y_pos = bar.get_height()
        va = "bottom" if y_pos >= 0 else "top"
        offset = 0.3 if y_pos >= 0 else -0.3
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            y_pos + offset,
            lbl, ha="center", va=va, fontsize=7,
        )

    fig.suptitle("Variant vs base model comparison", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_error_bars(
    all_exps: dict[str, pd.DataFrame],
    pairs: list[tuple[str, str]],
    out_path: str,
) -> None:
    """Grouped bar chart: mean ± std Euclidean error per experiment, grouped by architecture."""
    # Collect all unique experiment names in pair order
    seen = []
    for base, var in pairs:
        for name in (base, var):
            if name not in seen:
                seen.append(name)

    names     = [n for n in seen if n in all_exps]
    means     = []
    stds      = []
    for name in names:
        errs = all_exps[name][all_exps[name]["has_gt"]]["euclid_error"].dropna()
        means.append(float(errs.mean()))
        stds.append(float(errs.std(ddof=1)))

    x    = np.arange(len(names))
    cols = [_COLORS[i % 10] for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, means, 0.6, yerr=stds, capsize=4, color=cols, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Mean Euclidean tip error ± std (mm)")
    ax.set_title("Mean tip error per experiment (all GT-available steps)")
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Paired statistical comparison of variant experiments vs their base model."
    )
    parser.add_argument(
        "--experiments_dir", default=None,
        help="Directory containing experiment subdirectories. "
             "Uses default pairs (cropped/domino base vs noise/fourier/cpress).",
    )
    parser.add_argument(
        "--comparisons", nargs="+", default=None,
        metavar="BASE:VARIANT",
        help="Explicit list of comparisons as 'base_name:variant_name' pairs. "
             "Each name is looked up under --experiments_dir.",
    )
    parser.add_argument(
        "--out_dir", default="results/variant_comparison",
        help="Output directory for CSVs and plots (default: results/variant_comparison)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Resolve experiment directories --------------------------------------
    if args.experiments_dir is None:
        # Default: look relative to this script
        args.experiments_dir = os.path.join(os.path.dirname(__file__), "experiments")

    exp_base = args.experiments_dir

    if args.comparisons:
        pairs = []
        for token in args.comparisons:
            parts = token.split(":")
            if len(parts) != 2:
                parser.error(f"Invalid comparison '{token}': expected 'base:variant'")
            pairs.append((parts[0], parts[1]))
    else:
        pairs = _DEFAULT_PAIRS

    # --- Load all referenced experiments ------------------------------------
    exp_names = sorted({name for pair in pairs for name in pair})
    all_exps: dict[str, pd.DataFrame] = {}
    for name in exp_names:
        exp_dir = os.path.join(exp_base, name)
        df = _load_summary(exp_dir)
        if df is None:
            print(f"[skip] {name}: no eval/summary.csv found at {exp_dir}")
        else:
            all_exps[name] = df
            print(f"Loaded {name}: {len(df)} rows, {df['has_gt'].sum()} with GT")

    if not all_exps:
        print("No experiments loaded. Check --experiments_dir.")
        return

    # --- Run paired comparisons ---------------------------------------------
    print(f"\nRunning {len(pairs)} paired comparisons ...\n")

    rows = []
    for base_name, var_name in pairs:
        if base_name not in all_exps:
            print(f"[skip] {base_name} vs {var_name}: base not loaded")
            continue
        if var_name not in all_exps:
            print(f"[skip] {base_name} vs {var_name}: variant not loaded")
            continue
        r = compare_pair(all_exps[base_name], all_exps[var_name], base_name, var_name)
        rows.append(r)

    if not rows:
        print("No comparisons could be run.")
        return

    # --- Print results -------------------------------------------------------
    def _fmt(v, fmt=".4f"):
        try:
            return format(float(v), fmt)
        except (TypeError, ValueError):
            return "n/a"

    print(f"{'Base':<20} {'Variant':<22} {'n':>5}  {'Base err':>9}  {'Var err':>9}  "
          f"{'Change%':>9}  {'Cohen d':>8}  {'Wilcox p':>10}  {'t-test p':>9}")
    print("-" * 103)
    for r in rows:
        sign = "+" if r["pct_change"] >= 0 else ""
        print(
            f"{r['base']:<20} {r['variant']:<22} {r['n_paired_steps']:>5}  "
            f"{_fmt(r['mean_base_error_mm']):>9}  "
            f"{_fmt(r['mean_variant_error_mm']):>9}  "
            f"{sign}{_fmt(r['pct_change'], '.1f'):>8}%  "
            f"{_fmt(r['cohens_d'], '.3f'):>8}  "
            f"{_fmt(r['wilcoxon_p'], '.2e'):>10}  "
            f"{_fmt(r['ttest_p'], '.2e'):>9}"
            + (f"  [{r['note']}]" if r["note"] else "")
        )

    # --- CSV -----------------------------------------------------------------
    csv_path = os.path.join(args.out_dir, "variant_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {csv_path}")

    # --- Plots ---------------------------------------------------------------
    plot_pvalue_bars(rows, os.path.join(args.out_dir, "plot_variant_pvalues.png"))
    plot_error_bars(all_exps, pairs, os.path.join(args.out_dir, "plot_variant_errors.png"))

    print(f"\nAll outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()
