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

"""Compute maximum needle edge length change across training frames.

Reads the preprocessed VTU caches for training runs and finds the largest
change in length of any needle-to-needle edge between consecutive frames.
The result is saved to ``stats_dir/needle_edge_stats.json`` and is loaded by
``infer.py`` to cap predicted displacements during rollout (see
``needle_edge_cap`` in conf/config.yaml).

Must be run after the training caches have been built (i.e. after at least one
epoch of training, or after running build_stride1_cache.py for stride-1 runs).

Usage (from examples/cfd/needle_tissue_cropped/):
    uv run python compute_needle_edge_stats.py \\
        --data_dir /path/to/RUN-2 \\
        --stats_dir experiments/cropped_base/stats

    # Match training stride and split fractions if non-default:
    uv run python compute_needle_edge_stats.py \\
        --data_dir /path/to/RUN-2 \\
        --stats_dir experiments/cropped_stride1/stats \\
        --timestep_stride 1 \\
        --train_fraction 0.8
"""

import argparse
import json
import os

import numpy as np
import torch

from dataset import (
    _get_needle_tissue_node_sets,
    _group_vtu_by_run,
    _is_multi_run,
    _process_all_frames,
    _sorted_vtu_files,
    _atomic_torch_save,
)

_OUTPUT_FILENAME = "needle_edge_stats.json"


def _max_delta_edge_length(
    coords: torch.Tensor,
    needle_ei: torch.Tensor,
) -> float:
    """Return the max |Δlength| of any needle edge across consecutive frame pairs.

    Parameters
    ----------
    coords : Tensor, shape (n_frames, n_nodes, 3)
        Node coordinates for all frames of one run.
    needle_ei : Tensor, shape (2, E_needle)
        Needle-to-needle edge index in global node space.

    Returns
    -------
    float
        Maximum observed |length_{t+1} - length_t| in mm across all edges
        and all consecutive frame pairs in the run.
    """
    if needle_ei.shape[1] == 0 or coords.shape[0] < 2:
        return 0.0

    src, dst = needle_ei[0], needle_ei[1]
    # (n_frames, E_needle, 3)
    vecs = coords[:, src, :] - coords[:, dst, :]
    # (n_frames, E_needle)
    lengths = torch.linalg.norm(vecs, dim=-1)
    # (n_frames - 1, E_needle)
    delta = (lengths[1:] - lengths[:-1]).abs()
    return float(delta.max())


def main():
    parser = argparse.ArgumentParser(
        description="Compute max needle edge length change across training frames."
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Directory containing VTU simulation files (same as training data_dir)",
    )
    parser.add_argument(
        "--stats_dir", required=True,
        help="Directory where needle_edge_stats.json will be written "
             "(same as the experiment's stats_dir, alongside node_stats.json)",
    )
    parser.add_argument(
        "--timestep_stride", type=int, default=10,
        help="Timestep stride used during training (default: 10)",
    )
    parser.add_argument(
        "--train_fraction", type=float, default=0.8,
        help="Fraction of runs used for training (default: 0.8)",
    )
    parser.add_argument(
        "--val_fraction", type=float, default=0.1,
        help="Fraction of runs used for validation (default: 0.1)",
    )
    args = parser.parse_args()

    os.makedirs(args.stats_dir, exist_ok=True)
    out_path = os.path.join(args.stats_dir, _OUTPUT_FILENAME)

    # --- Discover training runs / files --------------------------------------
    if _is_multi_run(args.data_dir):
        run_files = _group_vtu_by_run(args.data_dir, args.timestep_stride)
        run_ids = list(run_files.keys())
        n_runs = len(run_ids)
        n_train = max(1, int(n_runs * args.train_fraction))
        train_run_ids = run_ids[:n_train]
        print(
            f"Multi-run dataset: {n_runs} runs total, "
            f"{len(train_run_ids)} training runs (stride={args.timestep_stride})"
        )
    else:
        # Single-run legacy mode: use the training frame slice only.
        # We load all frames and restrict below.
        train_run_ids = None
        vtu_files = _sorted_vtu_files(args.data_dir)
        n_frames = len(vtu_files)
        n_pairs = n_frames - 1
        n_train = int(n_pairs * args.train_fraction)
        print(f"Single-run dataset: {n_frames} frames, {n_train} training pairs")

    # --- Iterate over training data and compute stats ------------------------
    global_max_delta = 0.0
    needle_ei = None  # resolved from first cache

    if train_run_ids is not None:
        for run_id in train_run_ids:
            vtu_files = run_files[run_id]
            cache_path = os.path.join(args.data_dir, f"preprocessed_cache_RUN-{run_id}.pt")

            if os.path.exists(cache_path):
                cache = torch.load(cache_path, weights_only=False)
                cached_n = len(cache.get("frame_tensors", {}).get("coord", []))
                if cached_n != len(vtu_files):
                    print(f"  RUN-{run_id}: cache outdated — rebuilding ...")
                    cache = _process_all_frames(vtu_files)
                    _atomic_torch_save(cache, cache_path)
            else:
                print(f"  RUN-{run_id}: no cache found — building ...")
                cache = _process_all_frames(vtu_files)
                _atomic_torch_save(cache, cache_path)

            if needle_ei is None:
                # Needle-to-needle edges: edge_type column 0
                ei = cache["edge_index"]
                et = cache["edge_type_onehot"]
                is_needle = et[:, 0].bool()
                needle_ei = ei[:, is_needle]
                n_needle_edges = int(needle_ei.shape[1])
                needle_idx, _ = _get_needle_tissue_node_sets(ei, et)
                print(f"  Needle edges: {n_needle_edges}, needle nodes: {len(needle_idx)}")

            coords = cache["frame_tensors"]["coord"]  # (n_frames, n_nodes, 3)
            run_max = _max_delta_edge_length(coords, needle_ei)
            print(f"  RUN-{run_id}: max Δedge = {run_max:.6f} mm  ({len(vtu_files)} frames)")
            global_max_delta = max(global_max_delta, run_max)

    else:
        # Single-run mode
        cache_path = os.path.join(args.data_dir, "preprocessed_cache.pt")
        if os.path.exists(cache_path):
            cache = torch.load(cache_path, weights_only=False)
        else:
            print("Building cache ...")
            cache = _process_all_frames(vtu_files)
            _atomic_torch_save(cache, cache_path)

        ei = cache["edge_index"]
        et = cache["edge_type_onehot"]
        is_needle = et[:, 0].bool()
        needle_ei = ei[:, is_needle]
        print(f"  Needle edges: {needle_ei.shape[1]}")

        # Restrict to training frames only
        coords = cache["frame_tensors"]["coord"][:n_train + 1]
        global_max_delta = _max_delta_edge_length(coords, needle_ei)

    # --- Save ----------------------------------------------------------------
    stats = {
        "max_needle_edge_delta_mm": global_max_delta,
        "n_training_runs": len(train_run_ids) if train_run_ids is not None else 1,
        "timestep_stride": args.timestep_stride,
        "n_needle_edges": int(needle_ei.shape[1]) if needle_ei is not None else 0,
    }
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nMax needle edge Δlength across all training frames: {global_max_delta:.6f} mm")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
