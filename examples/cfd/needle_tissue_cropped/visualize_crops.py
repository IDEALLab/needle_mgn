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

"""Export VTU files illustrating the spatial crop strategies used during training.

For each requested frame and crop strategy the script outputs two files:

  full_<strategy>_f<frame>.vtu
      Full mesh, with a ``crop_mask`` point field:
        0 = excluded tissue node
        1 = kept tissue node
        2 = kept needle node

  crop_<strategy>_f<frame>.vtu
      Only the kept nodes / cells, so the cropped subgraph can be inspected
      in isolation.

Usage (from examples/cfd/needle_tissue_cropped/):
    uv run python visualize_crops.py --data_dir /path/to/RUN-2

Select a specific run and frames:
    uv run python visualize_crops.py --data_dir /path/to/RUN-2 \\
        --run_id 42 --frames 0 5 10 15 19 \\
        --out_dir ./crop_viz

Override crop radii:
    uv run python visualize_crops.py --data_dir /path/to/RUN-2 \\
        --needle_crop_mm 10 --tissue_crop_mm 25
"""

import argparse
import os
import sys

import numpy as np
import pyvista as pv
import torch
from scipy.spatial import cKDTree

from dataset import (
    _get_needle_tissue_node_sets,
    _group_vtu_by_run,
    _is_multi_run,
    _process_all_frames,
    _sorted_vtu_files,
    _atomic_torch_save,
)


# ---------------------------------------------------------------------------
# Crop helpers (mirrors dataset.py without the class overhead)
# ---------------------------------------------------------------------------

def _crop_proximity(
    coord: np.ndarray,
    needle_idx: np.ndarray,
    tissue_idx: np.ndarray,
    needle_crop_mm: float,
    tissue_crop_mm: float,
) -> np.ndarray:
    """Return global node indices kept by the proximity strategy."""
    needle_pos = coord[needle_idx]
    tissue_pos = coord[tissue_idx]

    tissue_tree = cKDTree(tissue_pos)
    dist_n, _ = tissue_tree.query(needle_pos, k=1)
    keep_needle = dist_n <= needle_crop_mm

    kept_needle_pos = needle_pos[keep_needle]
    if len(kept_needle_pos) > 0:
        needle_tree = cKDTree(kept_needle_pos)
        dist_t, _ = needle_tree.query(tissue_pos, k=1)
        keep_tissue = dist_t <= tissue_crop_mm
    else:
        keep_tissue = np.zeros(len(tissue_pos), dtype=bool)

    kept = np.concatenate([needle_idx[keep_needle], tissue_idx[keep_tissue]])
    return np.sort(kept).astype(np.int64)


def _crop_slice(
    coord: np.ndarray,
    needle_idx: np.ndarray,
    tissue_idx: np.ndarray,
    slice_half_thickness_mm: float,
    axial_fraction: float = 0.5,
) -> np.ndarray:
    """Return global node indices kept by the slice strategy.

    Parameters
    ----------
    axial_fraction : float in [0, 1]
        Position of the slab centre along the needle axis (0 = tip, 1 = base).
        Defaults to 0.5 (mid-needle).
    """
    needle_pos = coord[needle_idx]
    tissue_pos = coord[tissue_idx]

    centred = needle_pos - needle_pos.mean(axis=0)
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    axis = Vt[0]

    needle_axial = centred @ axis
    tissue_axial = (tissue_pos - needle_pos.mean(axis=0)) @ axis

    center = needle_axial.min() + axial_fraction * (needle_axial.max() - needle_axial.min())
    half = slice_half_thickness_mm

    keep_needle = np.abs(needle_axial - center) <= half
    keep_tissue = np.abs(tissue_axial - center) <= half

    kept = np.concatenate([needle_idx[keep_needle], tissue_idx[keep_tissue]])
    if len(kept) == 0:
        # Fallback: return everything
        return np.concatenate([needle_idx, tissue_idx])
    return np.sort(kept).astype(np.int64)


def _crop_full_needle(
    coord: np.ndarray,
    needle_idx: np.ndarray,
    tissue_idx: np.ndarray,
    full_needle_tissue_mm: float,
) -> np.ndarray:
    """Return global node indices kept by the full-needle strategy."""
    needle_pos = coord[needle_idx]
    tissue_pos = coord[tissue_idx]

    needle_tree = cKDTree(needle_pos)
    dist_t, _ = needle_tree.query(tissue_pos, k=1)
    keep_tissue = dist_t <= full_needle_tissue_mm

    kept = np.concatenate([needle_idx, tissue_idx[keep_tissue]])
    return np.sort(kept).astype(np.int64)


# ---------------------------------------------------------------------------
# Mask and VTU helpers
# ---------------------------------------------------------------------------

def _build_crop_mask(
    n_nodes: int,
    kept: np.ndarray,
    needle_idx: np.ndarray,
) -> np.ndarray:
    """Build integer point field: 0=excluded, 1=kept tissue, 2=kept needle."""
    mask = np.zeros(n_nodes, dtype=np.int32)
    needle_set = set(needle_idx.tolist())
    for g in kept:
        mask[g] = 2 if g in needle_set else 1
    return mask


def _write_full_mesh(
    mesh: pv.UnstructuredGrid,
    crop_mask: np.ndarray,
    out_path: str,
) -> None:
    """Write the full mesh with a crop_mask point field."""
    out = mesh.copy(deep=True)
    out.point_data["crop_mask"] = crop_mask
    out.save(out_path)
    print(f"  Saved: {out_path}  (kept {int((crop_mask > 0).sum())}/{len(crop_mask)} nodes)")


def _write_cropped_mesh(
    mesh: pv.UnstructuredGrid,
    kept: np.ndarray,
    out_path: str,
) -> None:
    """Write the kept subgraph as a VTU, preserving HEX cell structure.

    Uses ``extract_points`` with ``adjacent_cells=True`` so cells whose nodes
    are all within the kept set are included.  This preserves the hex mesh
    topology rather than reducing to a bare point cloud.
    """
    sub = mesh.extract_points(kept, adjacent_cells=True)
    sub.save(out_path)
    print(f"  Saved: {out_path}  ({sub.n_points} nodes, {sub.n_cells} cells)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export VTU files showing training crop strategies."
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Directory containing VTU simulation files",
    )
    parser.add_argument(
        "--run_id", default=None,
        help="Run ID to visualise (multi-run datasets). Defaults to first test run.",
    )
    parser.add_argument(
        "--frames", nargs="+", type=int, default=None,
        help="Frame indices to visualise (0-based within the selected run). "
             "Defaults to 5 evenly-spaced frames across the run.",
    )
    parser.add_argument(
        "--timestep_stride", type=int, default=10,
        help="Timestep stride used to group files (must match training, default: 10)",
    )
    parser.add_argument(
        "--strategies", nargs="+",
        default=["proximity", "slice", "full_needle"],
        choices=["proximity", "slice", "full_needle"],
        help="Which crop strategies to export (default: all three)",
    )
    parser.add_argument(
        "--out_dir", default="./crop_viz",
        help="Output directory (default: ./crop_viz)",
    )
    # Crop parameters
    parser.add_argument("--needle_crop_mm",          type=float, default=10.0)
    parser.add_argument("--tissue_crop_mm",           type=float, default=25.0)
    parser.add_argument("--slice_half_thickness_mm",  type=float, default=15.0)
    parser.add_argument("--full_needle_tissue_mm",    type=float, default=7.0)
    # Slice axial positions to export (multiple slices along the needle)
    parser.add_argument(
        "--slice_fractions", nargs="+", type=float, default=[0.25, 0.5, 0.75],
        help="Axial fractions (0=tip, 1=base) for the slice strategy exports "
             "(default: 0.25 0.5 0.75)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Select VTU files ----------------------------------------------------
    if _is_multi_run(args.data_dir):
        run_files = _group_vtu_by_run(args.data_dir, args.timestep_stride)
        run_ids = list(run_files.keys())
        if args.run_id is not None:
            if args.run_id not in run_files:
                print(f"ERROR: run_id={args.run_id!r} not found. Available: {run_ids}")
                sys.exit(1)
            run_id = args.run_id
        else:
            run_id = run_ids[0]
            print(f"No --run_id specified; using first run: {run_id}")
        vtu_files = run_files[run_id]
        print(f"Run {run_id}: {len(vtu_files)} frames (stride={args.timestep_stride})")
    else:
        vtu_files = _sorted_vtu_files(args.data_dir)
        run_id = "0"
        print(f"Single-run dataset: {len(vtu_files)} frames")

    # --- Select frames -------------------------------------------------------
    n_frames = len(vtu_files)
    if args.frames is not None:
        frame_indices = args.frames
        for fi in frame_indices:
            if fi < 0 or fi >= n_frames:
                print(f"ERROR: frame index {fi} out of range [0, {n_frames - 1}]")
                sys.exit(1)
    else:
        n_export = min(5, n_frames)
        frame_indices = list(np.round(np.linspace(0, n_frames - 1, n_export)).astype(int))
        print(f"Auto-selected {n_export} frames: {frame_indices}")

    # --- Build/load cache for topology ---------------------------------------
    cache_path = os.path.join(args.data_dir, f"preprocessed_cache_RUN-{run_id}.pt")
    if not os.path.exists(cache_path):
        cache_path = os.path.join(args.data_dir, "preprocessed_cache.pt")
    if os.path.exists(cache_path):
        print(f"Loading cache: {cache_path}")
        cache = torch.load(cache_path, weights_only=False)
        # Validate frame count
        cached_n = cache.get("frame_tensors", {}).get("coord")
        cached_n = len(cached_n) if cached_n is not None else 0
        if cached_n != n_frames:
            print(f"Cache frame count mismatch ({cached_n} vs {n_frames}) — rebuilding ...")
            cache = _process_all_frames(vtu_files)
            _atomic_torch_save(cache, cache_path)
    else:
        print("No cache found — building (this may take a while) ...")
        cache = _process_all_frames(vtu_files)
        _atomic_torch_save(cache, cache_path)
        print(f"Cache saved to {cache_path}")

    needle_idx, tissue_idx = _get_needle_tissue_node_sets(
        cache["edge_index"], cache["edge_type_onehot"]
    )
    print(f"Mesh: {len(needle_idx)} needle nodes, {len(tissue_idx)} tissue nodes")

    # --- Export each frame × strategy ----------------------------------------
    for fi in frame_indices:
        vtu_path = vtu_files[fi]
        mesh = pv.read(vtu_path)
        coord = cache["frame_tensors"]["coord"][fi].numpy()
        n_nodes = coord.shape[0]

        print(f"\nFrame {fi}  ({os.path.basename(vtu_path)})")

        for strategy in args.strategies:
            if strategy == "proximity":
                kept = _crop_proximity(
                    coord, needle_idx, tissue_idx,
                    args.needle_crop_mm, args.tissue_crop_mm,
                )
                tag = f"proximity_f{fi:03d}"
                mask = _build_crop_mask(n_nodes, kept, needle_idx)
                _write_full_mesh(mesh, mask, os.path.join(args.out_dir, f"full_{tag}.vtu"))
                _write_cropped_mesh(mesh, kept, os.path.join(args.out_dir, f"crop_{tag}.vtu"))

            elif strategy == "slice":
                for frac in args.slice_fractions:
                    kept = _crop_slice(
                        coord, needle_idx, tissue_idx,
                        args.slice_half_thickness_mm, axial_fraction=frac,
                    )
                    frac_str = f"{frac:.2f}".replace(".", "p")
                    tag = f"slice{frac_str}_f{fi:03d}"
                    mask = _build_crop_mask(n_nodes, kept, needle_idx)
                    _write_full_mesh(mesh, mask, os.path.join(args.out_dir, f"full_{tag}.vtu"))
                    _write_cropped_mesh(mesh, kept, os.path.join(args.out_dir, f"crop_{tag}.vtu"))

            elif strategy == "full_needle":
                kept = _crop_full_needle(
                    coord, needle_idx, tissue_idx,
                    args.full_needle_tissue_mm,
                )
                tag = f"full_needle_f{fi:03d}"
                mask = _build_crop_mask(n_nodes, kept, needle_idx)
                _write_full_mesh(mesh, mask, os.path.join(args.out_dir, f"full_{tag}.vtu"))
                _write_cropped_mesh(mesh, kept, os.path.join(args.out_dir, f"crop_{tag}.vtu"))

    print(f"\nDone. Open the VTU files in ParaView.")
    print(f"  For the full_*.vtu files, color by 'crop_mask':")
    print(f"    0 = excluded tissue, 1 = kept tissue, 2 = kept needle")


if __name__ == "__main__":
    main()
