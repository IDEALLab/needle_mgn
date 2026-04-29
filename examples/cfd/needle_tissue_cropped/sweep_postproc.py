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

"""Sweep needle-displacement post-processing knobs and compare tip deflection.

Knobs that can be swept (each accepts a list of values; cartesian product):

  --polyfit_alphas   axial_polyfit_alpha   default [0.0, 0.5, 0.8, 1.0]
  --polyfit_degrees  axial_polyfit_degree  default [1, 3, 5]
  --procrustes_alphas  procrustes_alpha            default [0.0]
  --needle_attns       consensus_attenuation       default [0.0]
  --tissue_attns       tissue_consensus_attenuation default [0.0]

Only knobs with multiple values appear in the combo label, keeping output
directory names manageable.

Each combo runs ``infer.py`` once per ``--run_ids`` test run.  Combos are
distributed across the available GPUs (default 2) using a thread pool +
``CUDA_VISIBLE_DEVICES``-pinned subprocesses.

Outputs (under ``--output_dir`` or ``<exp_dir>/sweep``):

  <combo_label>/RUN-<id>/predicted_*.vtu  — per-combo inference output
  sweep_summary.csv                       — one row per (combo, run, step)
  sweep_tip_error.png                     — mean |tip error| vs step
  sweep_tip_mag.png                       — predicted tip magnitude vs step

Usage (from examples/cfd/needle_tissue_cropped/):

    uv run python sweep_postproc.py \\
        --exp_dir ../../../experiments/cropped_base \\
        --data_dir ../../../RUN-2 \\
        --num_gpus 2

Default sweep is the polyfit grid (4 alphas × 3 degrees = 12 combos).  Pass
multiple values to other ``--*`` flags to sweep additional dimensions.
"""

import argparse
import csv
import glob
import os
import queue
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from compare_deflection import _get_needle_indices, _principal_axis  # noqa: E402
from dataset import _group_vtu_by_run, _sorted_vtu_files  # noqa: E402

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CFG = OmegaConf.load(os.path.join(_SCRIPT_DIR, "conf", "config.yaml"))


# ---------------------------------------------------------------------------
# Knob spec — maps CLI arg name → Hydra config key
# ---------------------------------------------------------------------------

# Order here also controls the order in combo labels.
_KNOBS = [
    ("polyfit_alpha",   "axial_polyfit_alpha"),
    ("polyfit_degree",  "axial_polyfit_degree"),
    ("procrustes_alpha", "procrustes_alpha"),
    ("needle_attn",     "consensus_attenuation"),
    ("tissue_attn",     "tissue_consensus_attenuation"),
]


# ---------------------------------------------------------------------------
# Test-run discovery
# ---------------------------------------------------------------------------

def _get_test_run_ids(
    data_dir: str,
    train_fraction: float,
    val_fraction: float,
    timestep_stride: int,
) -> list:
    run_files = _group_vtu_by_run(data_dir, timestep_stride)
    run_ids = list(run_files.keys())
    n_runs = len(run_ids)
    n_train = max(1, int(n_runs * train_fraction))
    n_val = max(1, int(n_runs * val_fraction))
    test_ids = run_ids[n_train + n_val:]
    if not test_ids:
        test_ids = run_ids[-1:]
    return test_ids


# ---------------------------------------------------------------------------
# Tip-deflection metrics from a directory of predicted VTUs
# ---------------------------------------------------------------------------

def _tip_metrics(
    infer_dir: str,
    needle_idx: np.ndarray,
    ref_pos_needle: np.ndarray,
    tip_local_idx: int,
) -> list:
    pattern = os.path.join(infer_dir, "predicted_*.vtu")
    files = sorted(
        glob.glob(pattern),
        key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)),
    )
    if not files:
        raise FileNotFoundError(f"No predicted_*.vtu in {infer_dir}")

    rows = []
    for step, fpath in enumerate(files):
        mesh = pv.read(fpath)
        pred = mesh.points[needle_idx]
        d_pred = pred[tip_local_idx] - ref_pos_needle[tip_local_idx]
        pred_mag = float(np.sqrt(d_pred[0] ** 2 + d_pred[1] ** 2))

        if "Points_gt" in mesh.point_data:
            gt = mesh.point_data["Points_gt"][needle_idx]
            d_gt = gt[tip_local_idx] - ref_pos_needle[tip_local_idx]
            gt_mag = float(np.sqrt(d_gt[0] ** 2 + d_gt[1] ** 2))
            has_gt = True
        else:
            gt_mag = float("nan")
            has_gt = False

        rows.append({
            "step": step + 1,
            "pred_mag": pred_mag,
            "gt_mag": gt_mag,
            "error_mag": pred_mag - gt_mag,
            "has_gt": has_gt,
        })
    return rows


# ---------------------------------------------------------------------------
# Per-combo worker — runs infer.py on one GPU, then computes tip metrics
# ---------------------------------------------------------------------------

def _run_combo(
    combo: dict,
    gpu_q: queue.Queue,
    needle_idx: np.ndarray,
    ref_pos_needle: np.ndarray,
    tip_local_idx: int,
) -> dict:
    gpu_id = gpu_q.get()
    label = combo["label"]
    run_id = combo["run_id"]
    out_dir = combo["out_dir"]
    log_path = os.path.join(out_dir, "infer.log")
    os.makedirs(out_dir, exist_ok=True)

    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cmd = [
            sys.executable, "infer.py",
            f"infer_run_id={run_id}",
            f"infer_output_dir={out_dir}",
            f"data_dir={combo['data_dir']}",
            f"ckpt_path={combo['ckpt_path']}",
            f"stats_dir={combo['stats_dir']}",
            f"n_rollout={combo['n_rollout']}",
            "cuda_devices=null",
        ] + [f"{k}={v}" for k, v in combo["knob_overrides"].items()] \
          + list(combo["extra_overrides"])

        t0 = time.time()
        with open(log_path, "w") as logf:
            logf.write("CMD: " + " ".join(cmd) + "\n\n")
            logf.flush()
            res = subprocess.run(
                cmd, cwd=_SCRIPT_DIR, env=env,
                stdout=logf, stderr=subprocess.STDOUT,
            )
        dt = time.time() - t0
        if res.returncode != 0:
            return {**_combo_meta(combo, gpu_id),
                    "rows": None,
                    "error": f"infer.py exit={res.returncode} (see {log_path})",
                    "elapsed_s": dt}

        rows = _tip_metrics(out_dir, needle_idx, ref_pos_needle, tip_local_idx)
        return {**_combo_meta(combo, gpu_id),
                "rows": rows, "error": None, "elapsed_s": dt}
    except Exception as exc:  # noqa: BLE001
        return {**_combo_meta(combo, gpu_id),
                "rows": None, "error": str(exc), "elapsed_s": 0.0}
    finally:
        gpu_q.put(gpu_id)


def _combo_meta(combo: dict, gpu_id: int) -> dict:
    return {
        "combo": combo["label"],
        "run_id": combo["run_id"],
        "gpu": gpu_id,
        **{k: combo["knob_overrides"][cfg_key] for k, cfg_key in _KNOBS},
    }


# ---------------------------------------------------------------------------
# Plotting / summary
# ---------------------------------------------------------------------------

def _aggregate(results: list) -> dict:
    by_combo = {}
    for r in results:
        if r["rows"] is None:
            continue
        key = r["combo"]
        info = by_combo.setdefault(key, {
            "by_step": {},
            **{k: r[k] for k, _ in _KNOBS},
        })
        for row in r["rows"]:
            if not row["has_gt"]:
                continue
            d = info["by_step"].setdefault(row["step"], {"err": [], "pred": [], "gt": []})
            d["err"].append(abs(row["error_mag"]))
            d["pred"].append(row["pred_mag"])
            d["gt"].append(row["gt_mag"])

    summary = {}
    for label, info in by_combo.items():
        steps = sorted(info["by_step"].keys())
        summary[label] = {
            **{k: info[k] for k, _ in _KNOBS},
            "step": np.array(steps),
            "abs_err": np.array([np.mean(info["by_step"][s]["err"]) for s in steps]),
            "pred_mag": np.array([np.mean(info["by_step"][s]["pred"]) for s in steps]),
            "gt_mag": np.array([np.mean(info["by_step"][s]["gt"]) for s in steps]),
        }
    return summary


def _plot_sweep(summary: dict, out_dir: str) -> None:
    if not summary:
        return

    cmap = plt.cm.viridis
    labels = sorted(summary.keys())
    colours = {lab: cmap(i / max(len(labels) - 1, 1)) for i, lab in enumerate(labels)}

    fig, ax = plt.subplots(figsize=(11, 6))
    for lab in labels:
        v = summary[lab]
        ax.plot(v["step"], v["abs_err"], color=colours[lab], lw=1.4, label=lab)
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Mean |tip error| (mm)")
    ax.set_title("Tip-error magnitude — post-processing sweep")
    ax.grid(True, lw=0.4, alpha=0.5)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    p = os.path.join(out_dir, "sweep_tip_error.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")

    fig, ax = plt.subplots(figsize=(11, 6))
    for lab in labels:
        v = summary[lab]
        ax.plot(v["step"], v["pred_mag"], color=colours[lab], lw=1.2, label=lab)
    first = summary[labels[0]]
    ax.plot(first["step"], first["gt_mag"], "k-", lw=2.0, label="GT")
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Mean predicted tip magnitude (mm)")
    ax.set_title("Predicted tip magnitude — post-processing sweep")
    ax.grid(True, lw=0.4, alpha=0.5)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    p = os.path.join(out_dir, "sweep_tip_mag.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")


def _write_summary_csv(results: list, out_dir: str) -> None:
    path = os.path.join(out_dir, "sweep_summary.csv")
    knob_keys = [k for k, _ in _KNOBS]
    fields = ["combo"] + knob_keys + [
        "run_id", "step", "pred_mag", "gt_mag", "error_mag",
        "has_gt", "gpu", "elapsed_s",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            base = {
                "combo": r["combo"],
                **{k: r[k] for k in knob_keys},
                "run_id": r["run_id"],
                "gpu": r["gpu"],
                "elapsed_s": f"{r['elapsed_s']:.1f}",
            }
            if r["rows"] is None:
                w.writerow({**base, "step": "", "pred_mag": "", "gt_mag": "",
                            "error_mag": f"ERROR: {r['error']}", "has_gt": ""})
                continue
            for row in r["rows"]:
                w.writerow({**base, **{k: row[k] for k in
                                       ("step", "pred_mag", "gt_mag",
                                        "error_mag", "has_gt")}})
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Combo construction
# ---------------------------------------------------------------------------

def _build_combos(args, run_ids: list) -> list:
    """Cartesian product over all knob lists × run IDs.  Label only includes
    knobs that vary."""
    knob_values = {
        "polyfit_alpha":    args.polyfit_alphas,
        "polyfit_degree":   args.polyfit_degrees,
        "procrustes_alpha": args.procrustes_alphas,
        "needle_attn":      args.needle_attns,
        "tissue_attn":      args.tissue_attns,
    }
    varying = [k for k, v in knob_values.items() if len(v) > 1]
    fixed = [k for k, v in knob_values.items() if len(v) == 1]

    combos = []
    for combo_vals in product(*(knob_values[k] for k, _ in _KNOBS)):
        knob_dict = dict(zip([k for k, _ in _KNOBS], combo_vals))
        # Cast degree to int for clean override formatting
        knob_dict["polyfit_degree"] = int(knob_dict["polyfit_degree"])
        # Build label from varying knobs only
        if varying:
            label = "_".join(f"{k}{knob_dict[k]:g}" if isinstance(knob_dict[k], float)
                             else f"{k}{knob_dict[k]}"
                             for k in varying)
        else:
            label = "baseline"

        # Override dict keyed by Hydra config key
        knob_overrides = {cfg_key: knob_dict[cli_key] for cli_key, cfg_key in _KNOBS}

        for run_id in run_ids:
            combos.append({
                "label": label,
                "run_id": run_id,
                "knob_overrides": knob_overrides,
                **{k: knob_dict[k] for k in knob_dict},
            })
    if varying:
        print(f"Varying knobs: {varying}")
    if fixed:
        print(f"Fixed knobs: " + ", ".join(
            f"{k}={knob_values[k][0]}" for k in fixed
        ))
    return combos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sweep needle-displacement post-processing knobs."
    )
    parser.add_argument("--exp_dir", required=True,
                        help="Path to experiment directory containing checkpoints/, stats/")
    parser.add_argument("--data_dir", required=True,
                        help="Directory with raw VTU files (e.g. RUN-2)")
    parser.add_argument("--output_dir", default=None,
                        help="Output dir for sweep results (default: <exp_dir>/sweep)")
    parser.add_argument("--polyfit_alphas", type=float, nargs="+",
                        default=[0.0, 0.5, 0.8, 1.0],
                        help="axial_polyfit_alpha values to sweep")
    parser.add_argument("--polyfit_degrees", type=int, nargs="+",
                        default=[1, 3, 5],
                        help="axial_polyfit_degree values to sweep")
    parser.add_argument("--procrustes_alphas", type=float, nargs="+",
                        default=[0.0],
                        help="procrustes_alpha values (single value disables sweep)")
    parser.add_argument("--needle_attns", type=float, nargs="+",
                        default=[0.0],
                        help="consensus_attenuation values (single value disables sweep)")
    parser.add_argument("--tissue_attns", type=float, nargs="+",
                        default=[0.0],
                        help="tissue_consensus_attenuation values (single value disables sweep)")
    parser.add_argument("--run_ids", nargs="+", default=None,
                        help="Test run IDs to evaluate on (default: first test run)")
    parser.add_argument("--all_test_runs", action="store_true",
                        help="Evaluate on every test run instead of just the first")
    parser.add_argument("--n_rollout", type=int, default=16,
                        help="Rollout steps per run (default: 16)")
    parser.add_argument("--num_gpus", type=int, default=2,
                        help="Number of GPUs to use in parallel (default: 2)")
    parser.add_argument("--extra", default="",
                        help="Extra Hydra overrides forwarded verbatim to infer.py")
    parser.add_argument("--train_fraction", type=float,
                        default=float(OmegaConf.select(_CFG, "train_fraction", default=0.8)))
    parser.add_argument("--val_fraction", type=float,
                        default=float(OmegaConf.select(_CFG, "val_fraction", default=0.1)))
    parser.add_argument("--timestep_stride", type=int,
                        default=int(OmegaConf.select(_CFG, "timestep_stride", default=10)))
    args = parser.parse_args()

    exp_dir = os.path.abspath(args.exp_dir)
    data_dir = os.path.abspath(args.data_dir)
    ckpt_path = os.path.join(exp_dir, "checkpoints")
    stats_dir = os.path.join(exp_dir, "stats")
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(exp_dir, "sweep")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(ckpt_path):
        sys.exit(f"ERROR: checkpoints/ not found at {ckpt_path}")
    if not os.path.isdir(stats_dir):
        sys.exit(f"ERROR: stats/ not found at {stats_dir}")

    # --- Forward model/data settings from the experiment's saved config -----
    cfg_path = os.path.join(ckpt_path, "config.yaml")
    base_cfg = OmegaConf.load(cfg_path) if os.path.isfile(cfg_path) else None
    extra_overrides = []
    if base_cfg is not None:
        for key in (
            "model_type", "use_cpress", "per_region_norm", "use_bsms",
            "num_bsms_levels", "num_layers_bistride", "bistride_unet_levels",
            "input_dim_edges", "hidden_dim_node_encoder", "hidden_dim_edge_encoder",
            "hidden_dim_node_decoder", "hidden_dim_processor", "processor_size",
            "aggregation", "use_fourier_features", "n_fourier_features",
            "fourier_scale", "num_harmonics", "n_vec_outputs",
            "irreps_hidden", "l_max", "n_radial_basis", "r_max",
            "needle_crop_mm", "tissue_crop_mm", "world_edge_radius",
            "timestep_stride", "train_fraction", "val_fraction",
        ):
            if base_cfg.get(key) is not None:
                v = base_cfg.get(key)
                v_str = ("true" if v else "false") if isinstance(v, bool) else str(v)
                # subprocess.run with a list passes each element as one argv
                # entry, so spaces don't need shell-quoting here.
                extra_overrides.append(f"{key}={v_str}")
    if args.extra:
        extra_overrides += shlex.split(args.extra)

    # --- Discover test runs -------------------------------------------------
    if args.run_ids is not None:
        run_ids = [str(r) for r in args.run_ids]
    else:
        all_test = _get_test_run_ids(
            data_dir, args.train_fraction, args.val_fraction, args.timestep_stride
        )
        run_ids = all_test if args.all_test_runs else all_test[:1]
    print(f"Test runs to evaluate ({len(run_ids)}): {run_ids}")

    # --- Reference needle geometry (shared across all combos) ---------------
    needle_idx = _get_needle_indices(data_dir)
    ref_mesh = pv.read(_sorted_vtu_files(data_dir)[0])
    ref_pos_needle = ref_mesh.points[needle_idx]
    principal = _principal_axis(ref_pos_needle)
    centred = ref_pos_needle - ref_pos_needle.mean(axis=0)
    axial = centred @ principal
    sort_order = np.argsort(axial)
    tip_local_idx = int(sort_order[-1])
    print(f"Needle: {len(needle_idx)} nodes, tip = local idx {tip_local_idx}")

    # --- Build combos -------------------------------------------------------
    combos = _build_combos(args, run_ids)
    for c in combos:
        c["out_dir"] = os.path.join(output_dir, c["label"], f"RUN-{c['run_id']}")
        c["data_dir"] = data_dir
        c["ckpt_path"] = ckpt_path
        c["stats_dir"] = stats_dir
        c["n_rollout"] = args.n_rollout
        c["extra_overrides"] = extra_overrides

    n_unique_combos = len({c["label"] for c in combos})
    print(f"\n{n_unique_combos} unique combos × {len(run_ids)} runs "
          f"= {len(combos)} jobs.  GPUs: {args.num_gpus}.  Output: {output_dir}\n")

    # --- Dispatch across GPUs -----------------------------------------------
    gpu_q: queue.Queue = queue.Queue()
    for g in range(args.num_gpus):
        gpu_q.put(g)

    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.num_gpus) as pool:
        futures = {
            pool.submit(_run_combo, c, gpu_q, needle_idx, ref_pos_needle, tip_local_idx): c
            for c in combos
        }
        for fut in as_completed(futures):
            r = fut.result()
            tag = "OK" if r["error"] is None else "FAIL"
            print(f"  [{tag}] {r['combo']}  run={r['run_id']}  "
                  f"gpu={r['gpu']}  ({r['elapsed_s']:.1f}s)"
                  + (f"  — {r['error']}" if r["error"] else ""))
            results.append(r)
    print(f"\nAll combos done in {time.time() - t0:.1f}s")

    _write_summary_csv(results, output_dir)
    summary = _aggregate(results)
    _plot_sweep(summary, output_dir)

    failed = [r["combo"] + "/RUN-" + str(r["run_id"]) for r in results if r["error"] is not None]
    if failed:
        print(f"\nFailed combos ({len(failed)}): {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
