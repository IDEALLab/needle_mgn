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

"""Run inference + evaluation for all trained experiments.

Auto-discovers experiment directories under experiments/ by finding any that
contain a checkpoints/config.yaml.  All model parameters (architecture, crop
settings, timestep_stride, etc.) are read directly from that saved config so
there is no need to duplicate overrides in this script.

Only the paths (data_dir, ckpt_path, stats_dir) and the rollout limit are
substituted at eval time.  n_rollout is computed from timestep_stride as
VTU_LIMIT // timestep_stride (default VTU_LIMIT = 160).

Usage:
    uv run python run_eval_experiments.py /path/to/RUN-2
    uv run python run_eval_experiments.py /path/to/RUN-2 --experiments cropped_base domino_base
    uv run python run_eval_experiments.py /path/to/RUN-2 --vtu_limit 200  # no step cap
    uv run python run_eval_experiments.py /path/to/RUN-2 --force          # re-run even if eval exists
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Keys from the checkpoint config that should NOT be forwarded to infer.py.
# These are either controlled by this script or are training-only settings.
# ---------------------------------------------------------------------------
_SKIP_KEYS = {
    # Paths — overridden with current machine's local paths
    "data_dir", "ckpt_path", "stats_dir", "infer_output_dir", "infer_run_id",
    # Rollout control — set by this script from --vtu_limit / timestep_stride
    "infer_start_frame", "n_rollout",
    # Training-only hyperparameters
    "epochs", "batch_size", "num_workers", "lr", "amp", "jit",
    "noise_std", "val_every", "save_every",
    # WandB — not needed at inference time
    "wandb_mode", "wandb_project", "wandb_entity", "wandb_log_artifact",
    # Derived dimensions — infer.py re-derives these from the model and stats
    "input_dim_nodes", "output_dim",
    # GPU selection — let the calling environment decide
    "cuda_devices",
}

# Keys that only appear in DoMINO configs (used for model-family detection).
_DOMINO_ONLY_KEYS = {
    "use_fourier_features_state", "n_fourier_features_state",
    "fourier_scale_state", "grid_res", "num_sample_nodes", "global_features",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_model_family(cfg, ckpt_dir: Path) -> str:
    """Return 'cropped' or 'domino' from config keys or checkpoint filenames."""
    cfg_keys = set(OmegaConf.to_container(cfg, resolve=False).keys())
    if cfg_keys & _DOMINO_ONLY_KEYS:
        return "domino"
    if ckpt_dir.is_dir() and any(f.name.startswith("DoMINO.") for f in ckpt_dir.iterdir()):
        return "domino"
    return "cropped"


def _cfg_to_overrides(cfg) -> list:
    """Convert checkpoint config to a list of Hydra override strings.

    Skips keys in _SKIP_KEYS and any key whose value is a nested mapping
    (DoMINO model sub-config is handled via its own config path).
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=False)
    overrides = []
    for key, val in cfg_dict.items():
        if key in _SKIP_KEYS:
            continue
        if isinstance(val, dict):
            # Nested configs (e.g. DoMINO model sub-dict) are not simple
            # key=value overrides; skip them — they live in the checkpoint config
            # which infer.py loads separately for DoMINO.
            continue
        if val is None:
            overrides.append(f"{key}=null")
        elif isinstance(val, bool):
            overrides.append(f"{key}={'true' if val else 'false'}")
        elif isinstance(val, list):
            items = ",".join(str(v) for v in val)
            overrides.append(f"'{key}=[{items}]'")
        else:
            s = str(val)
            if " " in s:
                overrides.append(f"{key}='{s}'")
            else:
                overrides.append(f"{key}={s}")
    return overrides


# ---------------------------------------------------------------------------
# Per-experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    exp_dir: Path,
    data_dir: str,
    project_dir: Path,
    vtu_limit: int,
    force: bool,
) -> bool:
    """Run inference + eval for one experiment.  Returns True on success."""
    name = exp_dir.name
    ckpt_dir = exp_dir / "checkpoints"
    cfg_path = ckpt_dir / "config.yaml"

    print(f"\n{'=' * 66}")
    print(f"  {name}")
    print(f"{'=' * 66}")

    if not cfg_path.exists():
        print(f"  [SKIP] No checkpoints/config.yaml found")
        return True

    if not any(ckpt_dir.iterdir()):
        print(f"  [SKIP] Checkpoints directory is empty")
        return True

    infer_out = exp_dir / "inference_output"
    eval_out = exp_dir / "eval"

    if not force and (eval_out / "summary.csv").exists():
        print(f"  [SKIP] Eval already complete (eval/summary.csv exists)")
        return True

    # Load saved training config
    cfg = OmegaConf.load(cfg_path)
    model_family = _detect_model_family(cfg, ckpt_dir)
    script_dir = project_dir / "examples" / "cfd" / f"needle_tissue_{model_family}"

    # Compute n_rollout from timestep_stride so every experiment stops at
    # the same physical VTU timestep regardless of stride.
    timestep_stride = int(OmegaConf.select(cfg, "timestep_stride", default=10))
    n_rollout = vtu_limit // timestep_stride
    print(f"  model_family={model_family}, timestep_stride={timestep_stride}, n_rollout={n_rollout}")

    # Build override list: model/data params from saved config, paths from CLI.
    # Path and rollout overrides come LAST so they take priority over any
    # stale values in the checkpoint config.
    cfg_overrides = _cfg_to_overrides(cfg)
    path_overrides = [
        f"data_dir={data_dir}",
        f"ckpt_path={ckpt_dir}",
        f"stats_dir={exp_dir / 'stats'}",
    ]
    rollout_overrides = [f"n_rollout={n_rollout}"]
    extra = " ".join(cfg_overrides + path_overrides + rollout_overrides)

    infer_out.mkdir(parents=True, exist_ok=True)
    eval_out.mkdir(parents=True, exist_ok=True)

    # --- Inference -----------------------------------------------------------
    print(f"  [inference]")
    infer_cmd = [
        sys.executable,
        str(script_dir / "run_test_inference.py"),
        "--data_dir", data_dir,
        "--infer_output_base", str(infer_out),
        "--skip_existing",
        "--extra", extra,
    ]
    result = subprocess.run(infer_cmd, cwd=script_dir)
    if result.returncode != 0:
        print(f"  [ERROR] Inference failed")
        return False

    # --- Eval ----------------------------------------------------------------
    print(f"  [eval]")
    eval_cmd = [
        sys.executable,
        str(script_dir / "eval_test_runs.py"),
        "--data_dir", data_dir,
        "--infer_base_dir", str(infer_out),
        "--out_dir", str(eval_out),
    ]
    result = subprocess.run(eval_cmd, cwd=script_dir)
    if result.returncode != 0:
        print(f"  [ERROR] Eval failed")
        return False

    print(f"  Done: {eval_out}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run inference + evaluation for all trained experiments, "
                    "reading model settings from each experiment's checkpoint config."
    )
    parser.add_argument(
        "data_dir",
        help="Directory containing the raw VTU simulation files (e.g. /path/to/RUN-2)",
    )
    parser.add_argument(
        "--experiments_dir",
        default=None,
        help="Root directory containing experiment subdirectories "
             "(default: <project>/experiments/)",
    )
    parser.add_argument(
        "--experiments", nargs="+", default=None,
        help="Explicit list of experiment names or full paths to evaluate "
             "(default: all subdirectories of --experiments_dir that have "
             "checkpoints/config.yaml)",
    )
    parser.add_argument(
        "--vtu_limit", type=int, default=160,
        help="Stop inference at this VTU timestep number.  n_rollout is computed "
             "as vtu_limit // timestep_stride for each experiment, so all "
             "experiments stop at the same physical point regardless of stride.  "
             "Default: 160  (set to a large number to run the full insertion).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run eval even if eval/summary.csv already exists",
    )
    args = parser.parse_args()

    data_dir = os.path.realpath(args.data_dir)
    if not os.path.isdir(data_dir):
        print(f"ERROR: data_dir not found: {data_dir}")
        sys.exit(1)

    project_dir = Path(__file__).resolve().parent
    experiments_dir = Path(args.experiments_dir) if args.experiments_dir else project_dir / "experiments"

    # --- Discover experiment directories -------------------------------------
    if args.experiments is not None:
        exp_dirs = []
        for name_or_path in args.experiments:
            p = Path(name_or_path)
            if not p.is_absolute():
                p = experiments_dir / name_or_path
            exp_dirs.append(p)
    else:
        if not experiments_dir.is_dir():
            print(f"ERROR: experiments_dir not found: {experiments_dir}")
            sys.exit(1)
        exp_dirs = sorted(
            p for p in experiments_dir.iterdir()
            if p.is_dir() and (p / "checkpoints" / "config.yaml").exists()
        )
        if not exp_dirs:
            print(f"No experiments with checkpoints/config.yaml found in {experiments_dir}")
            sys.exit(1)

    print(f"Project:      {project_dir}")
    print(f"Data dir:     {data_dir}")
    print(f"Experiments:  {experiments_dir}")
    print(f"VTU limit:    {args.vtu_limit} (n_rollout = vtu_limit / timestep_stride)")
    print(f"Found {len(exp_dirs)} experiment(s) to process\n")

    failed = []
    for exp_dir in exp_dirs:
        ok = run_experiment(
            exp_dir=exp_dir,
            data_dir=data_dir,
            project_dir=project_dir,
            vtu_limit=args.vtu_limit,
            force=args.force,
        )
        if not ok:
            failed.append(exp_dir.name)

    print(f"\n{'=' * 66}")
    print(f"All experiments complete.  Results are in experiments/<name>/eval/")
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
