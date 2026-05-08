#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute accumulated polyfit-residual noise for inference rollouts.

For each predicted frame in ``experiments/<variant>/inference_output/RUN-*``:

  1. Take the per-node displacement of the needle:
         ``disp = pred_pos[needle] - ref_pos[needle]``  (shape (N, 3))
  2. Fit a polynomial of ``axial_coord`` to each component of ``disp``
     (the same subspace used by :func:`_axial_polyfit_blend` in infer.py:
     rigid translation / tilt / parabolic / cubic bending).
  3. The residual ``disp - polyfit(disp)`` is the high-frequency, per-node
     noise that the post-processing blend would suppress.  Its per-frame
     RMS over (nodes, 3) is the "noise" metric.
  4. The "accumulated" metric is the sum of per-frame RMS across all
     rollout steps for a given run.

Outputs (per experiment, written to ``experiments/<variant>/eval/``):
  - ``polyfit_noise.csv``    — one row per (run_id, step) with frame RMS
  - ``polyfit_noise_summary.csv`` — one row per run with sum + mean RMS

A top-level summary across all experiments is written to
``polyfit_noise_overall.csv`` in the experiments directory.

Usage
-----
    uv run python compute_polyfit_noise.py /path/to/RUN-2
    uv run python compute_polyfit_noise.py /path/to/RUN-2 \
        --experiments cropped_base_mgn_wu cropped_fiber_iso_invw
    uv run python compute_polyfit_noise.py /path/to/RUN-2 --degree 3
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

from compare_deflection import _get_needle_indices, _principal_axis  # noqa: E402
from dataset import _sorted_vtu_files  # noqa: E402


def _polyfit_residual_rms(
    disp: np.ndarray,
    axial_coords: np.ndarray,
    degree: int,
) -> float:
    """RMS of ``disp - polyfit(disp)`` over (nodes, 3 components).

    ``disp`` is (N, 3) per-node displacement, ``axial_coords`` is (N,) the
    axial parametrisation along the needle's principal axis.  The polynomial
    is fit per-component in least-squares, mirroring ``_axial_polyfit_blend``.
    """
    if disp.shape[0] < degree + 1:
        return float("nan")
    s = axial_coords - axial_coords.mean()
    s_max = np.abs(s).max()
    if s_max < 1e-8:
        return float("nan")
    s = s / s_max
    V = np.stack([s ** k for k in range(degree + 1)], axis=1)  # (N, deg+1)
    coeffs, *_ = np.linalg.lstsq(V, disp, rcond=None)           # (deg+1, 3)
    fit = V @ coeffs
    resid = disp - fit
    return float(np.sqrt(np.mean(resid ** 2)))


def _orthonormal_poly_basis(axial_coords: np.ndarray, max_degree: int) -> np.ndarray:
    """Gram-Schmidt orthonormal polynomial basis on the discrete node set.

    Returns ``Q`` of shape ``(N, max_degree + 1)`` with ``Q.T @ Q = I``.
    Projecting a per-node signal ``f`` onto column ``k`` yields a coefficient
    whose square is that mode's energy contribution to ``||f||^2``.
    """
    s = axial_coords - axial_coords.mean()
    s_max = np.abs(s).max()
    if s_max < 1e-8:
        s = np.zeros_like(s)
    else:
        s = s / s_max
    V = np.stack([s ** k for k in range(max_degree + 1)], axis=1)  # (N, K+1)
    Q, _ = np.linalg.qr(V)                                          # (N, K+1)
    return Q


def _one_step_spectrum(
    disp: np.ndarray,
    Q: np.ndarray,
) -> np.ndarray:
    """Per-mode RMS of ``disp`` in the orthonormal polynomial basis ``Q``.

    ``disp`` is (N, 3); returns array of shape (K+1,) giving the RMS over the
    3 components of the projection coefficient onto each basis mode.
    Mode 0 = rigid translation, mode 1 = tilt, modes 2-3 = parabolic / cubic
    bending, modes ≥ degree+1 = "high frequency" residual that polyfit
    blending suppresses.
    """
    coefs = Q.T @ disp                              # (K+1, 3)
    return np.sqrt(np.mean(coefs ** 2, axis=1))     # (K+1,)


def _resample_axial(
    disp: np.ndarray,
    axial_coords: np.ndarray,
    n_uniform: int,
) -> np.ndarray:
    """Resample per-node ``disp`` (N, 3) onto a uniform axial grid of length
    ``n_uniform``.  Nodes sharing an axial coord (the needle is a 3-D solid,
    not a 1-D curve) are averaged into a single sample first; the resulting
    1-D signal is interpolated onto the uniform grid.
    """
    order = np.argsort(axial_coords)
    s_sorted = axial_coords[order]
    d_sorted = disp[order]
    uniq, inv = np.unique(s_sorted, return_inverse=True)
    if uniq.size < 2:
        return np.zeros((n_uniform, disp.shape[1]), dtype=float)
    means = np.zeros((uniq.size, disp.shape[1]), dtype=float)
    counts = np.zeros(uniq.size, dtype=int)
    for k in range(disp.shape[1]):
        np.add.at(means[:, k], inv, d_sorted[:, k])
    np.add.at(counts, inv, 1)
    means /= counts[:, None]
    grid = np.linspace(uniq[0], uniq[-1], n_uniform)
    out = np.empty((n_uniform, disp.shape[1]), dtype=float)
    for k in range(disp.shape[1]):
        out[:, k] = np.interp(grid, uniq, means[:, k])
    return out


def _fourier_spectrum(
    disp: np.ndarray,
    axial_coords: np.ndarray,
    n_uniform: int,
) -> tuple:
    """Magnitude spectrum of ``disp`` along the needle's axial coordinate.

    Returns ``(freqs, mag)`` where ``freqs`` is in cycles per unit axial
    length (same units as ``axial_coords``) and ``mag`` of shape
    ``(n_uniform // 2 + 1,)`` is the per-frequency RMS over the 3 spatial
    components.  DC component is the spatial mean (rigid translation).
    """
    sig = _resample_axial(disp, axial_coords, n_uniform)
    span = float(axial_coords.max() - axial_coords.min())
    if span <= 0:
        freqs = np.zeros(n_uniform // 2 + 1)
        mag = np.zeros_like(freqs)
        return freqs, mag
    dx = span / (n_uniform - 1)
    F = np.fft.rfft(sig, axis=0) / n_uniform              # (F, 3)
    mag = np.sqrt(np.mean(np.abs(F) ** 2, axis=1))         # (F,)
    freqs = np.fft.rfftfreq(n_uniform, d=dx)               # cycles per axial-length unit
    return freqs, mag


def _list_predicted_vtus(run_dir: Path) -> list:
    files = sorted(
        glob(str(run_dir / "predicted_*.vtu")),
        key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)),
    )
    return files


def _process_experiment(
    exp_dir: Path,
    needle_idx: np.ndarray,
    ref_pos_needle: np.ndarray,
    axial_coords: np.ndarray,
    degree: int,
    spectrum_basis: np.ndarray | None = None,
    fft_n_uniform: int | None = None,
    fft_steps: list | None = None,
) -> dict:
    """Walk inference_output/RUN-*/predicted_*.vtu and compute per-frame noise.

    Returns a dict with ``per_frame`` rows, ``per_run`` summary rows, and
    aggregate ``total`` / ``mean`` across all runs.  Returns None if the
    experiment has no inference output.
    """
    infer_root = exp_dir / "inference_output"
    if not infer_root.is_dir():
        return None

    run_dirs = sorted(p for p in infer_root.iterdir() if p.is_dir())
    if not run_dirs:
        return None

    per_frame_rows = []
    per_run_rows = []
    all_rms = []
    one_step_spectra = []  # one (K+1,) array per run, from first rollout frame
    # FFT collection: keyed by step label ("step1", "step2", ..., "final");
    # each entry is a list of (F,) magnitude arrays, one per run.
    fft_by_label: dict = {}
    fft_freqs = None
    fft_steps_set = set(fft_steps) if fft_steps else set()

    for run_dir in run_dirs:
        run_id = run_dir.name
        files = _list_predicted_vtus(run_dir)
        if not files:
            continue

        run_rms_list = []
        n_files = len(files)
        for step, fpath in enumerate(files):
            # 1-indexed step label exposed to the user (step 1 = first
            # predicted frame).  ``last_step_1based`` is the index of the
            # final rollout frame in this run.
            step_1based = step + 1
            is_final = (step == n_files - 1)
            try:
                mesh = pv.read(fpath)
            except Exception as e:
                print(f"    [warn] {fpath}: {e}")
                continue
            pred_pos_needle = mesh.points[needle_idx]
            disp = pred_pos_needle - ref_pos_needle
            if not np.all(np.isfinite(disp)):
                rms = float("nan")
            else:
                rms = _polyfit_residual_rms(disp, axial_coords, degree)
            run_rms_list.append(rms)
            per_frame_rows.append({
                "run_id": run_id,
                "step": step,
                "noise_rms": rms,
            })

            # One-step spectrum: spatial-mode RMS of disp at the first
            # rollout frame, before error compounds across steps.
            if (
                step == 0
                and spectrum_basis is not None
                and np.all(np.isfinite(disp))
            ):
                one_step_spectra.append(_one_step_spectrum(disp, spectrum_basis))
            if (
                fft_n_uniform is not None
                and np.all(np.isfinite(disp))
                and (step_1based in fft_steps_set or is_final)
            ):
                f, m = _fourier_spectrum(disp, axial_coords, fft_n_uniform)
                fft_freqs = f
                if step_1based in fft_steps_set:
                    fft_by_label.setdefault(f"step{step_1based}", []).append(m)
                if is_final:
                    fft_by_label.setdefault("final", []).append(m)

        run_arr = np.asarray(run_rms_list, dtype=float)
        finite = run_arr[np.isfinite(run_arr)]
        per_run_rows.append({
            "run_id": run_id,
            "n_steps": len(run_arr),
            "n_finite": int(finite.size),
            "noise_sum": float(finite.sum()) if finite.size else float("nan"),
            "noise_mean": float(finite.mean()) if finite.size else float("nan"),
            "noise_max": float(finite.max()) if finite.size else float("nan"),
        })
        all_rms.extend(finite.tolist())

    if not per_run_rows:
        return None

    arr = np.asarray(all_rms, dtype=float)
    spectrum_mean = (
        np.mean(np.stack(one_step_spectra, axis=0), axis=0)
        if one_step_spectra
        else None
    )
    fft_means = {
        label: np.mean(np.stack(arrs, axis=0), axis=0)
        for label, arrs in fft_by_label.items()
        if arrs
    }
    fft_n_runs = {label: len(arrs) for label, arrs in fft_by_label.items()}
    return {
        "per_frame": per_frame_rows,
        "per_run": per_run_rows,
        "total_sum": float(arr.sum()) if arr.size else float("nan"),
        "total_mean": float(arr.mean()) if arr.size else float("nan"),
        "total_max": float(arr.max()) if arr.size else float("nan"),
        "n_runs": len(per_run_rows),
        "n_frames": int(arr.size),
        "one_step_spectrum": spectrum_mean,
        "n_one_step": len(one_step_spectra),
        "fft_by_step": fft_means,
        "fft_freqs": fft_freqs,
        "fft_n_runs": fft_n_runs,
    }


def _per_step_matrix(per_frame_rows: list) -> tuple:
    """Stack per-frame rows into a (n_runs, max_steps) array, NaN-padded.

    Returns (run_ids, matrix) where matrix[i, t] is the RMS for run i at step t.
    """
    by_run = {}
    for r in per_frame_rows:
        by_run.setdefault(r["run_id"], []).append((r["step"], r["noise_rms"]))
    run_ids = sorted(by_run.keys())
    max_step = max(s for rows in by_run.values() for s, _ in rows) + 1
    mat = np.full((len(run_ids), max_step), np.nan, dtype=float)
    for i, rid in enumerate(run_ids):
        for s, v in by_run[rid]:
            mat[i, s] = v
    return run_ids, mat


def _plot_experiment(out_path: Path, name: str, run_ids: list, mat: np.ndarray) -> None:
    """Per-experiment plot: thin line per run + bold mean, instantaneous + cumulative."""
    steps = np.arange(mat.shape[1])
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(mat, axis=0)
    cum = np.nancumsum(np.where(np.isnan(mat), 0.0, mat), axis=1)
    cum_mean = np.nanmean(cum, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, rid in enumerate(run_ids):
        axes[0].plot(steps, mat[i], color="C0", alpha=0.3, linewidth=0.8, label=rid if i == 0 else None)
        axes[1].plot(steps, cum[i], color="C0", alpha=0.3, linewidth=0.8)
    axes[0].plot(steps, mean, color="C3", linewidth=2.0, label="mean across runs")
    axes[1].plot(steps, cum_mean, color="C3", linewidth=2.0, label="mean across runs")
    axes[0].set_xlabel("rollout step")
    axes[0].set_ylabel("polyfit-residual RMS")
    axes[0].set_title(f"{name}: per-step noise")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    axes[1].set_xlabel("rollout step")
    axes[1].set_ylabel("cumulative residual RMS")
    axes[1].set_title(f"{name}: accumulated noise")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_spectrum(
    out_path: Path,
    name: str,
    spectrum: np.ndarray,
    polyfit_degree: int,
    n_runs: int,
) -> None:
    """Per-mode RMS of the one-step error.  Highlights modes > polyfit_degree
    (which the polyfit blend filters out) versus low-order modes (rigid /
    bending) that the blend preserves.
    """
    modes = np.arange(spectrum.shape[0])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["C0" if k <= polyfit_degree else "C3" for k in modes]
    ax.bar(modes, spectrum, color=colors, alpha=0.85)
    ax.axvline(polyfit_degree + 0.5, color="k", linestyle="--", linewidth=0.8,
               label=f"polyfit cutoff (deg={polyfit_degree})")
    ax.set_xlabel("polynomial mode k")
    ax.set_ylabel("RMS of mode coefficient")
    ax.set_title(f"{name}: one-step error spectrum (mean over {n_runs} run(s))")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_fft(
    out_path: Path,
    name: str,
    freqs: np.ndarray,
    mag: np.ndarray,
    n_runs: int,
) -> None:
    """Per-frequency RMS magnitude of the one-step error along the needle axis."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(freqs, mag, marker="o", markersize=2.5, linewidth=1.0)
    ax.set_xlabel("spatial frequency (cycles per axial-length unit)")
    ax.set_ylabel("FFT magnitude RMS")
    ax.set_title(f"{name}: error FFT (mean over {n_runs} run(s))")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_overall_fft(
    out_path: Path,
    exp_fft: dict,
    freqs: np.ndarray,
    title_suffix: str = "",
) -> None:
    """One FFT magnitude curve per experiment, log y-scale."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, mag in exp_fft.items():
        ax.plot(freqs, mag, marker="o", markersize=2.5, linewidth=1.0, label=name)
    ax.set_xlabel("spatial frequency (cycles per axial-length unit)")
    ax.set_ylabel("FFT magnitude RMS")
    ax.set_title(f"Error FFT (mean over runs){title_suffix}")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_overall_spectrum(
    out_path: Path,
    exp_spectra: dict,
    polyfit_degree: int,
) -> None:
    """One spectrum line per experiment, log y-scale."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, spec in exp_spectra.items():
        modes = np.arange(spec.shape[0])
        ax.plot(modes, spec, marker="o", markersize=3, linewidth=1.0, label=name)
    ax.axvline(polyfit_degree + 0.5, color="k", linestyle="--", linewidth=0.8,
               label=f"polyfit cutoff (deg={polyfit_degree})")
    ax.set_xlabel("polynomial mode k")
    ax.set_ylabel("RMS of mode coefficient")
    ax.set_title("One-step error spectrum (mean over runs)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_overall(out_path: Path, exp_curves: dict) -> None:
    """One mean curve per experiment on shared axes (instantaneous + cumulative)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, (mean, cum_mean) in exp_curves.items():
        steps = np.arange(len(mean))
        axes[0].plot(steps, mean, label=name, linewidth=1.2)
        axes[1].plot(steps, cum_mean, label=name, linewidth=1.2)
    axes[0].set_xlabel("rollout step")
    axes[0].set_ylabel("polyfit-residual RMS (mean over runs)")
    axes[0].set_title("Per-step noise vs rollout step")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("rollout step")
    axes[1].set_ylabel("cumulative residual RMS")
    axes[1].set_title("Accumulated noise vs rollout step")
    axes[1].grid(True, alpha=0.3)
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
        description="Compute accumulated polyfit-residual noise across "
                    "inference rollouts in experiments/*/inference_output/."
    )
    parser.add_argument(
        "data_dir",
        help="Directory of raw VTU simulation files (used to identify "
             "needle nodes and the reference frame-0 positions).",
    )
    parser.add_argument(
        "--experiments_dir", default=None,
        help="Root containing experiment subdirs (default: <project>/experiments)",
    )
    parser.add_argument(
        "--experiments", nargs="+", default=None,
        help="Explicit list of experiment names (default: every subdir of "
             "experiments_dir that has inference_output/)",
    )
    parser.add_argument(
        "--degree", type=int, default=3,
        help="Polynomial degree for the axial fit (default: 3, matching "
             "axial_polyfit_degree default in infer.py).",
    )
    parser.add_argument(
        "--spectrum_max_degree", type=int, default=20,
        help="Highest polynomial mode in the one-step error spectrum "
             "(default: 20).  Modes 0..K are an orthonormal basis on the "
             "needle's axial coordinate.",
    )
    parser.add_argument(
        "--fft_n_uniform", type=int, default=128,
        help="Number of uniform-grid samples for the axial FFT (default: "
             "128).  Disp is averaged within shared axial coords, then "
             "interpolated onto a uniform grid of this length before rfft.",
    )
    parser.add_argument(
        "--fft_steps", default="1,2,5,10",
        help="Comma-separated 1-indexed rollout steps at which to compute "
             "the FFT (default: '1,2,5,10').  The final step of each run is "
             "always included as 'final'.",
    )
    args = parser.parse_args()

    data_dir = os.path.realpath(args.data_dir)
    if not os.path.isdir(data_dir):
        print(f"ERROR: data_dir not found: {data_dir}")
        sys.exit(1)

    project_dir = _SCRIPT_DIR
    experiments_dir = (
        Path(args.experiments_dir) if args.experiments_dir else project_dir / "experiments"
    )
    if not experiments_dir.is_dir():
        print(f"ERROR: experiments_dir not found: {experiments_dir}")
        sys.exit(1)

    # --- Reference geometry (frame 0 of the raw dataset) --------------------
    needle_idx = _get_needle_indices(data_dir)
    vtu_files = _sorted_vtu_files(data_dir)
    if not vtu_files:
        print(f"ERROR: no VTU files found in {data_dir}")
        sys.exit(1)
    ref_mesh = pv.read(vtu_files[0])
    ref_pos_needle = ref_mesh.points[needle_idx]
    principal = _principal_axis(ref_pos_needle)
    axial_coords = (ref_pos_needle - ref_pos_needle.mean(axis=0)) @ principal
    print(
        f"Needle nodes: {len(needle_idx)}  "
        f"axial range: [{axial_coords.min():.4f}, {axial_coords.max():.4f}]"
    )
    print(f"Polynomial degree: {args.degree}")
    print(f"Spectrum max mode: {args.spectrum_max_degree}")
    print(f"FFT uniform samples: {args.fft_n_uniform}")
    fft_steps_list = [int(s) for s in args.fft_steps.split(",") if s.strip()]
    print(f"FFT steps (1-indexed): {fft_steps_list} + final")

    spectrum_basis = _orthonormal_poly_basis(axial_coords, args.spectrum_max_degree)

    # --- Discover experiment dirs -------------------------------------------
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
    overall_spectra = {}  # name -> (K+1,) per-mode RMS (mean over runs)
    # overall_fft[step_label][exp_name] = (F,) FFT magnitude RMS (mean over runs)
    overall_fft: dict = {}
    overall_fft_freqs = None
    for exp_dir in exp_dirs:
        name = exp_dir.name
        print(f"=== {name} ===")
        if not (exp_dir / "inference_output").is_dir():
            print(f"  [SKIP] no inference_output/")
            continue

        result = _process_experiment(
            exp_dir, needle_idx, ref_pos_needle, axial_coords, args.degree,
            spectrum_basis=spectrum_basis,
            fft_n_uniform=args.fft_n_uniform,
            fft_steps=fft_steps_list,
        )
        if result is None:
            print(f"  [SKIP] no rollouts found")
            continue

        eval_dir = exp_dir / "eval"
        _write_csv(
            eval_dir / "polyfit_noise.csv",
            result["per_frame"],
            ["run_id", "step", "noise_rms"],
        )
        _write_csv(
            eval_dir / "polyfit_noise_summary.csv",
            result["per_run"],
            ["run_id", "n_steps", "n_finite",
             "noise_sum", "noise_mean", "noise_max"],
        )
        run_ids, mat = _per_step_matrix(result["per_frame"])
        _plot_experiment(eval_dir / "polyfit_noise.png", name, run_ids, mat)

        spec = result.get("one_step_spectrum")
        if spec is not None:
            spec_rows = [
                {"mode": k, "rms": float(spec[k])} for k in range(spec.shape[0])
            ]
            _write_csv(
                eval_dir / "one_step_spectrum.csv",
                spec_rows,
                ["mode", "rms"],
            )
            _plot_spectrum(
                eval_dir / "one_step_spectrum.png",
                name,
                spec,
                args.degree,
                result["n_one_step"],
            )
            overall_spectra[name] = spec

        fft_by_step = result.get("fft_by_step") or {}
        fft_freqs = result.get("fft_freqs")
        fft_n_runs = result.get("fft_n_runs") or {}
        if fft_by_step and fft_freqs is not None:
            overall_fft_freqs = fft_freqs
            for step_label, mag in fft_by_step.items():
                fft_rows = [
                    {"freq": float(fft_freqs[k]), "mag": float(mag[k])}
                    for k in range(mag.shape[0])
                ]
                _write_csv(
                    eval_dir / f"fft_{step_label}.csv",
                    fft_rows,
                    ["freq", "mag"],
                )
                _plot_fft(
                    eval_dir / f"fft_{step_label}.png",
                    f"{name} ({step_label})",
                    fft_freqs,
                    mag,
                    fft_n_runs.get(step_label, 0),
                )
                overall_fft.setdefault(step_label, {})[name] = mag
        with np.errstate(invalid="ignore"):
            mean_curve = np.nanmean(mat, axis=0)
        cum_curve = np.nanmean(
            np.nancumsum(np.where(np.isnan(mat), 0.0, mat), axis=1), axis=0
        )
        overall_curves[name] = (mean_curve, cum_curve)

        print(
            f"  runs={result['n_runs']}  frames={result['n_frames']}  "
            f"sum={result['total_sum']:.4e}  mean={result['total_mean']:.4e}  "
            f"max={result['total_max']:.4e}"
        )
        overall_rows.append({
            "experiment": name,
            "n_runs": result["n_runs"],
            "n_frames": result["n_frames"],
            "noise_sum": result["total_sum"],
            "noise_mean": result["total_mean"],
            "noise_max": result["total_max"],
        })

    if overall_rows:
        overall_rows.sort(key=lambda r: r["noise_mean"])
        out_path = experiments_dir / "polyfit_noise_overall.csv"
        _write_csv(
            out_path, overall_rows,
            ["experiment", "n_runs", "n_frames",
             "noise_sum", "noise_mean", "noise_max"],
        )
        plot_path = experiments_dir / "polyfit_noise_overall.png"
        _plot_overall(plot_path, overall_curves)
        print(f"\nOverall plot: {plot_path}")

        if overall_spectra:
            spec_path = experiments_dir / "one_step_spectrum_overall.png"
            _plot_overall_spectrum(spec_path, overall_spectra, args.degree)
            print(f"Overall spectrum plot: {spec_path}")

        if overall_fft and overall_fft_freqs is not None:
            # One overlay plot per requested step (and 'final').  Order:
            # numeric step labels in ascending step number, then 'final'.
            def _step_sort_key(lbl: str):
                if lbl == "final":
                    return (1, 0)
                return (0, int(lbl.replace("step", "")))

            for step_label in sorted(overall_fft.keys(), key=_step_sort_key):
                fft_path = experiments_dir / f"fft_{step_label}_overall.png"
                _plot_overall_fft(
                    fft_path,
                    overall_fft[step_label],
                    overall_fft_freqs,
                    title_suffix=f" — {step_label}",
                )
                print(f"Overall FFT plot ({step_label}): {fft_path}")
        print(f"Overall summary (sorted by mean RMS): {out_path}")
        print(f"  {'experiment':40s}  {'mean':>10s}  {'sum':>10s}  {'max':>10s}")
        for r in overall_rows:
            print(
                f"  {r['experiment']:40s}  {r['noise_mean']:10.4e}  "
                f"{r['noise_sum']:10.4e}  {r['noise_max']:10.4e}"
            )


if __name__ == "__main__":
    main()
