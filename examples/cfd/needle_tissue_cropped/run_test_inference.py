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

"""Run infer.py sequentially on every test-set run.

Discovers test run IDs using the same train/val/test split logic as train.py
and infer.py, then invokes infer.py once per run with the appropriate Hydra
overrides.  Predicted VTUs for each run are written to:

    <infer_output_base>/RUN-<id>/

Usage (from examples/cfd/needle_tissue_cropped/):
    uv run run_test_inference.py
    uv run run_test_inference.py --infer_output_base ./outputs/inference_output
    uv run run_test_inference.py --run_ids 140 141 142   # specific runs only
    uv run run_test_inference.py --extra "n_rollout=50 needle_crop_mm=15.0"
"""

import argparse
import os
import shlex
import subprocess
import sys

from omegaconf import OmegaConf

from dataset import _group_vtu_by_run

# Load conf/config.yaml once at import time so CLI defaults stay in sync.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CFG = OmegaConf.load(os.path.join(_SCRIPT_DIR, "conf", "config.yaml"))

# Resolve data_dir relative to the script directory (mirrors Hydra's
# to_absolute_path behaviour when the script is run from its own directory).
_RAW_DATA_DIR: str = OmegaConf.select(_CFG, "data_dir", default="../../../RUN-2")
_DEFAULT_DATA_DIR: str = os.path.normpath(os.path.join(_SCRIPT_DIR, _RAW_DATA_DIR))
_DEFAULT_TRAIN_FRACTION: float = float(OmegaConf.select(_CFG, "train_fraction", default=0.8))
_DEFAULT_VAL_FRACTION: float = float(OmegaConf.select(_CFG, "val_fraction", default=0.1))
_DEFAULT_TIMESTEP_STRIDE: int = int(OmegaConf.select(_CFG, "timestep_stride", default=1))


def _get_test_run_ids(
    data_dir: str,
    train_fraction: float,
    val_fraction: float,
    timestep_stride: int,
) -> list:
    """Return test run IDs using the same split logic as dataset.py / train.py."""
    import random as _random
    run_files = _group_vtu_by_run(data_dir, timestep_stride)
    run_ids = list(run_files.keys())
    # Must mirror dataset.py's deterministic shuffle (seed 42) so the test
    # set lines up with what the model never saw at training time.
    _random.Random(42).shuffle(run_ids)
    n_runs = len(run_ids)
    n_train = max(1, int(n_runs * train_fraction))
    n_val   = max(1, int(n_runs * val_fraction))
    test_ids = run_ids[n_train + n_val:]
    if not test_ids:
        test_ids = run_ids[-1:]
    return test_ids


def main():
    parser = argparse.ArgumentParser(
        description="Run infer.py on every test-set run."
    )
    parser.add_argument(
        "--data_dir",
        default=_DEFAULT_DATA_DIR,
        help="Directory containing raw VTU files (defaults to config.yaml data_dir)",
    )
    parser.add_argument(
        "--infer_output_base",
        default="./inference_output",
        help="Parent directory for per-run predicted VTUs; each run goes in RUN-<id>/",
    )
    parser.add_argument(
        "--train_fraction", type=float, default=_DEFAULT_TRAIN_FRACTION,
    )
    parser.add_argument(
        "--val_fraction", type=float, default=_DEFAULT_VAL_FRACTION,
    )
    parser.add_argument(
        "--timestep_stride", type=int, default=_DEFAULT_TIMESTEP_STRIDE,
    )
    parser.add_argument(
        "--run_ids", nargs="+", default=None,
        help="Explicit list of run IDs to infer on (overrides auto test-set discovery)",
    )
    parser.add_argument(
        "--extra", default="",
        help="Extra Hydra overrides passed verbatim to infer.py, e.g. 'n_rollout=50'",
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip runs whose output directory already contains predicted_*.vtu files",
    )
    args = parser.parse_args()

    # --- Discover test runs ---------------------------------------------------
    if args.run_ids is not None:
        test_run_ids = [str(r) for r in args.run_ids]
        print(f"Using explicit run IDs: {test_run_ids}")
    else:
        test_run_ids = _get_test_run_ids(
            args.data_dir, args.train_fraction, args.val_fraction, args.timestep_stride
        )
        print(f"Auto-discovered {len(test_run_ids)} test runs: {test_run_ids}")

    infer_output_base = os.path.abspath(args.infer_output_base)
    extra_overrides = shlex.split(args.extra) if args.extra else []

    # infer.py must be invoked from the same directory as this script so that
    # Hydra finds conf/config.yaml and relative paths (checkpoints, stats) resolve correctly.
    script_dir = os.path.dirname(os.path.abspath(__file__))

    failed = []
    for i, run_id in enumerate(test_run_ids):
        run_out_dir = os.path.join(infer_output_base, f"RUN-{run_id}")

        if args.skip_existing:
            existing = [
                f for f in os.listdir(run_out_dir)
                if f.startswith("predicted_") and f.endswith(".vtu")
            ] if os.path.isdir(run_out_dir) else []
            if existing:
                print(f"\n[{i+1}/{len(test_run_ids)}] RUN-{run_id}: skipping ({len(existing)} files already exist)")
                continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(test_run_ids)}] Inferring on RUN-{run_id} → {run_out_dir}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, "infer.py",
            f"infer_run_id={run_id}",
            f"infer_output_dir={run_out_dir}",
            # Suppress Hydra's per-run output dir chatter by keeping the default
            # hydra.run.dir but isolating the actual VTU output via infer_output_dir.
        ] + extra_overrides

        result = subprocess.run(cmd, cwd=script_dir)
        if result.returncode != 0:
            print(f"\n[ERROR] RUN-{run_id} failed with exit code {result.returncode}")
            failed.append(run_id)

    print(f"\n{'='*60}")
    print(f"Done. {len(test_run_ids) - len(failed)}/{len(test_run_ids)} runs succeeded.")
    if failed:
        print(f"Failed runs: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
