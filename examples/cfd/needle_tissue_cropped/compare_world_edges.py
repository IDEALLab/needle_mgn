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

"""Compare GT world edges vs. KD-tree approximation across frames of one run.

Prints a per-frame summary showing:
  - GT edge count and which needle nodes have edges (by axial position)
  - KD-tree approximation edge count and which needle nodes it adds/misses
  - Frame at which GT edges disappear (full embedding)

Usage:
    cd examples/cfd/needle_tissue_cropped
    uv run python compare_world_edges.py --data_dir /path/to/RUN-2 --run_id 10

The script uses the same preprocessing as dataset.py so results are directly
comparable to what the model sees during training.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

# Make sure we can import from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import (
    _sorted_vtu_files,
    _group_vtu_by_run,
    _extract_world_edges,
    _get_needle_tissue_node_sets,
    _process_all_frames,
)


def _build_kdtree_world_edges(
    needle_pos: np.ndarray,
    tissue_pos: np.ndarray,
    needle_node_indices: np.ndarray,
    tissue_node_indices: np.ndarray,
    contact_radius: float,
) -> set:
    """Return set of (needle_global, tissue_global) pairs from KD-tree search."""
    tissue_kdtree = cKDTree(tissue_pos)
    pairs = tissue_kdtree.query_ball_point(needle_pos, contact_radius)
    result = set()
    for needle_j, neighbors in enumerate(pairs):
        needle_global = int(needle_node_indices[needle_j])
        for t_local in neighbors:
            result.add((needle_global, int(tissue_node_indices[t_local])))
    return result


def _needle_axial_positions(coords: np.ndarray, needle_idx: np.ndarray) -> np.ndarray:
    """Return axial coordinate (along needle principal axis) for each needle node."""
    pts = coords[needle_idx]
    centred = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axis = vt[0]
    return centred @ axis


def main():
    parser = argparse.ArgumentParser(description="Compare GT vs. KD-tree world edges")
    parser.add_argument("--data_dir", required=True, help="Directory containing VTU files")
    parser.add_argument("--run_id", default=None, help="Run ID to analyse (default: first run)")
    parser.add_argument("--out_dir", default="./outputs/world_edge_compare",
                        help="Directory to write comparison plots")
    parser.add_argument("--world_edge_radius", type=float, default=1.2,
                        help="Fixed search radius in mm — must match EDGE_RADIUS in "
                             "odb_to_mgn_input.py (default 1.2)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Find VTU files for the selected run ---
    if os.path.exists(os.path.join(args.data_dir, "preprocessed_cache.pt")):
        # Single-run layout
        vtu_files = _sorted_vtu_files(args.data_dir)
        run_label = "single"
    else:
        run_files = _group_vtu_by_run(args.data_dir)
        run_ids = list(run_files.keys())
        run_id = args.run_id if args.run_id is not None else run_ids[0]
        if run_id not in run_files:
            print(f"ERROR: run_id={run_id!r} not found. Available: {run_ids}")
            return
        vtu_files = run_files[run_id]
        run_label = run_id

    print(f"Analysing run {run_label}: {len(vtu_files)} frames")

    # --- Build/load cache ---
    cache_path = os.path.join(args.data_dir, f"preprocessed_cache_RUN-{run_label}.pt")
    if not os.path.exists(cache_path):
        cache_path = os.path.join(args.data_dir, "preprocessed_cache.pt")
    if os.path.exists(cache_path):
        import torch
        cache = torch.load(cache_path, weights_only=False)
        print(f"Loaded cache from {cache_path}")
    else:
        print("No cache found — building (may take a minute) ...")
        cache = _process_all_frames(vtu_files)
        print("Done building cache")

    hex_edge_index = cache["edge_index"]
    hex_edge_type_onehot = cache["edge_type_onehot"]
    frame_tensors = cache["frame_tensors"]
    world_edges = cache["world_edges"]
    coords_all = frame_tensors["coord"].numpy()  # (T, N, 3)

    needle_idx, tissue_idx = _get_needle_tissue_node_sets(hex_edge_index, hex_edge_type_onehot)
    print(f"Needle nodes: {len(needle_idx)}, tissue nodes: {len(tissue_idx)}")

    contact_radius = args.world_edge_radius
    print(f"world_edge_radius = {contact_radius:.4f} mm (matching odb_to_mgn_input.py EDGE_RADIUS)")

    # --- Per-frame analysis ---
    n_frames = len(world_edges)
    gt_needle_nodes_per_frame = []      # set of needle global indices with GT edges
    kd_needle_nodes_per_frame = []      # set from KD-tree
    gt_edge_counts = []
    kd_edge_counts = []
    last_gt_frame = -1

    for t in range(n_frames):
        coords = coords_all[t]
        needle_pos = coords[needle_idx]
        tissue_pos = coords[tissue_idx]

        # GT edges
        gt_ei, _ = world_edges[t]
        if gt_ei.shape[1] > 0:
            gt_srcs = gt_ei[0].numpy()
            gt_dsts = gt_ei[1].numpy()
            # Identify which are needle→tissue direction
            needle_set = set(needle_idx.tolist())
            gt_needle_nodes = set(int(s) for s in gt_srcs if int(s) in needle_set)
            gt_needle_nodes |= set(int(d) for d in gt_dsts if int(d) in needle_set)
            gt_edge_counts.append(gt_ei.shape[1])
            last_gt_frame = t
        else:
            gt_needle_nodes = set()
            gt_edge_counts.append(0)
        gt_needle_nodes_per_frame.append(gt_needle_nodes)

        # KD-tree approximation
        kd_pairs = _build_kdtree_world_edges(
            needle_pos, tissue_pos, needle_idx, tissue_idx, contact_radius
        )
        kd_needle_nodes = set(p[0] for p in kd_pairs)
        kd_edge_counts.append(len(kd_pairs) * 2)  # bidirectional
        kd_needle_nodes_per_frame.append(kd_needle_nodes)

    print(f"\nLast GT frame with edges: {last_gt_frame} / {n_frames - 1}")
    print(f"  (GT edges disappear at frame {last_gt_frame + 1})")

    # --- Detailed comparison for a few key frames ---
    axial = _needle_axial_positions(coords_all[0], needle_idx)
    needle_sort = np.argsort(axial)  # base → tip order

    sample_frames = sorted(set([
        0,
        n_frames // 4,
        n_frames // 2,
        last_gt_frame,
        min(last_gt_frame + 1, n_frames - 1),
        n_frames - 1,
    ]))

    print("\n--- Per-frame comparison ---")
    print(f"{'Frame':>6}  {'GT edges':>9}  {'KD edges':>9}  "
          f"{'GT needle nodes':>16}  {'KD-only nodes':>14}  {'GT-only nodes':>14}")

    for t in sample_frames:
        gt_nn = gt_needle_nodes_per_frame[t]
        kd_nn = kd_needle_nodes_per_frame[t]
        kd_only = kd_nn - gt_nn
        gt_only = gt_nn - kd_nn
        print(f"{t:>6}  {gt_edge_counts[t]:>9}  {kd_edge_counts[t]:>9}  "
              f"{len(gt_nn):>16}  {len(kd_only):>14}  {len(gt_only):>14}")

    # --- Axial distribution: which needle nodes have edges ---
    print("\n--- Axial distribution of needle nodes with GT edges (frame 0 → last GT frame) ---")
    print("  (negative axial = base, positive = tip)")

    # Map global needle indices to axial positions
    global_to_axial = {}
    for local_j, global_idx in enumerate(needle_idx):
        global_to_axial[int(global_idx)] = float(axial[local_j])

    if last_gt_frame >= 0:
        gt_nn_last = gt_needle_nodes_per_frame[last_gt_frame]
        if gt_nn_last:
            axials = sorted(global_to_axial[n] for n in gt_nn_last if n in global_to_axial)
            print(f"  Frame {last_gt_frame}: axial range [{min(axials):.1f}, {max(axials):.1f}] mm")

    gt_nn_0 = gt_needle_nodes_per_frame[0]
    if gt_nn_0:
        axials = sorted(global_to_axial[n] for n in gt_nn_0 if n in global_to_axial)
        print(f"  Frame 0: axial range [{min(axials):.1f}, {max(axials):.1f}] mm")

    # KD-tree extra nodes (not in GT) at frame 0
    kd_nn_0 = kd_needle_nodes_per_frame[0]
    kd_only_0 = kd_nn_0 - gt_nn_0
    if kd_only_0:
        axials_extra = sorted(global_to_axial[n] for n in kd_only_0 if n in global_to_axial)
        print(f"\n  Frame 0 KD-only nodes (spurious): "
              f"axial range [{min(axials_extra):.1f}, {max(axials_extra):.1f}] mm")
        print(f"  (These are the base/exterior edges the KD-tree adds incorrectly)")

    # --- Plot: edge count over time ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    frames = list(range(n_frames))
    axes[0].bar(frames, gt_edge_counts, color="steelblue", label="GT (training)")
    axes[0].set_ylabel("Edge count (bidirectional)")
    axes[0].set_title("World edge count per frame — GT vs. KD-tree")
    axes[0].legend()
    axes[0].axvline(last_gt_frame + 0.5, color="red", lw=1.5, ls="--",
                    label="GT contact deactivated")

    axes[1].bar(frames, kd_edge_counts, color="darkorange", alpha=0.7, label="KD-tree approx")
    axes[1].bar(frames, gt_edge_counts, color="steelblue", alpha=0.7, label="GT (training)")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Edge count")
    axes[1].legend()
    axes[1].axvline(last_gt_frame + 0.5, color="red", lw=1.5, ls="--")

    fig.tight_layout()
    path = os.path.join(args.out_dir, "world_edge_counts.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {path}")

    # --- Plot: axial distribution of needle nodes with edges ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sample_gt = [t for t in sample_frames if gt_edge_counts[t] > 0]
    for t in sample_gt:
        nn = gt_needle_nodes_per_frame[t]
        if nn:
            ax_vals = sorted(global_to_axial[n] for n in nn if n in global_to_axial)
            axes[0].scatter([t] * len(ax_vals), ax_vals, s=4, alpha=0.7, label=f"frame {t}")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Axial position (mm, negative=base, positive=tip)")
    axes[0].set_title("GT: needle nodes with world edges (axial distribution)")
    axes[0].legend(fontsize=7)

    for t in sample_frames:
        nn_kd = kd_needle_nodes_per_frame[t]
        nn_gt = gt_needle_nodes_per_frame[t]
        kd_only = nn_kd - nn_gt
        if kd_only:
            ax_vals = sorted(global_to_axial[n] for n in kd_only if n in global_to_axial)
            axes[1].scatter([t] * len(ax_vals), ax_vals, s=4, alpha=0.7,
                            color="red", label=f"frame {t}" if t == sample_frames[0] else None)
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Axial position (mm)")
    axes[1].set_title("KD-tree ONLY nodes (spurious, not in GT)")
    if kd_only_0:
        axes[1].legend(["KD-only (spurious)"], fontsize=8)

    fig.tight_layout()
    path = os.path.join(args.out_dir, "needle_node_axial.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    print(f"\nDone. Outputs written to {args.out_dir}")


if __name__ == "__main__":
    main()
