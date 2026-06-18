# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Precompute per-node bevel-face and surface-face normals for the needle.

The needle mesh topology is fixed across the dataset, so these geometric
features depend only on the rest-state (frame-0) coordinates and can be
computed once and reused for every sample.

This script generates ``<data_dir>/needle_geometry_features.pt`` with:
    {
      "needle_node_indices":     LongTensor (n_needle,),
      "bevel_node_normal":       FloatTensor (n_nodes, 3) — zero off the bevel,
      "surface_node_normal":     FloatTensor (n_nodes, 3) — zero off the
                                                            needle surface,
      "is_bevel_node":           BoolTensor (n_nodes,),
      "is_needle_surface_node":  BoolTensor (n_nodes,),
      "needle_axis":             FloatTensor (3,),
      "needle_centroid":         FloatTensor (3,),
      "arclen_to_clamp":         FloatTensor (n_nodes,) — normalised [0,1]
                                                         arc-length from the
                                                         clamp end, zero off
                                                         the needle,
      "needle_length_mm":        float,
    }

All vectors are unit length on the nodes where they are non-zero.

Bevel-face detection
--------------------
Surface faces of needle HEX cells are extracted (a face is a "surface" face
when it belongs to only one cell).  Each face's outward normal is computed
by averaging two diagonal cross products and then sign-fixing against the
direction from the owning cell's centroid to the face centroid.  A face is
classified as a *bevel* face when its outward normal has a meaningful
axial component (``normal · needle_axis`` in ``[bevel_axial_min,
bevel_axial_max]``, default 0.1 – 0.95) — the lateral cylinder normals
are ≈ perpendicular to the axis and so are excluded.

Per-node bevel normal = mean of incident bevel-face normals, re-normalised
to unit length.  Nodes not touched by any bevel face get zero.

Per-node surface normal = mean of incident surface-face normals, also unit
normalised.

Usage
-----
    uv run python compute_needle_geometry.py \\
        --data_dir ../../../RUN-2 \\
        [--bevel_axial_min 0.1 --bevel_axial_max 0.95]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pyvista as pv
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import _MULTI_RUN_PATTERN, _is_multi_run, _group_vtu_by_run, _sorted_vtu_files  # noqa: E402


# ---------------------------------------------------------------------------
# HEX cell face layout — 8 vertices, 6 quad faces.
# Order chosen so each face's right-hand normal points outward when vertices
# are in canonical HEX ordering.  Cross-check vs cell centroid below.
# ---------------------------------------------------------------------------
_HEX_FACE_LOCAL_NODES = np.array(
    [
        (0, 3, 2, 1),  # bottom (-z)
        (4, 5, 6, 7),  # top    (+z)
        (0, 1, 5, 4),  # -y
        (1, 2, 6, 5),  # +x
        (2, 3, 7, 6),  # +y
        (3, 0, 4, 7),  # -x
    ],
    dtype=np.int64,
)


def _load_first_vtu(data_dir: str) -> pv.UnstructuredGrid:
    if _is_multi_run(data_dir):
        runs = _group_vtu_by_run(data_dir, timestep_stride=1)
        first_run = sorted(runs.keys(), key=int)[0]
        first_file = runs[first_run][0]
    else:
        first_file = _sorted_vtu_files(data_dir)[0]
    print(f"Loading mesh from {first_file}")
    mesh = pv.read(first_file)
    return mesh


def _extract_needle_surface(mesh: pv.UnstructuredGrid) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (needle_node_indices, needle_hex_cells, needle_hex_centroids).

    needle_hex_cells: (n_needle_cells, 8) global node indices.
    needle_hex_centroids: (n_needle_cells, 3).
    """
    cell_types = mesh.celltypes
    element_type = mesh.cell_data["element_type"]
    conn = mesh.cell_connectivity
    points = np.asarray(mesh.points, dtype=np.float64)
    n_hex = int((cell_types == 12).sum())
    hex_nodes = conn[: n_hex * 8].reshape(n_hex, 8)
    hex_elem = element_type[:n_hex]
    needle_mask = hex_elem == 0
    needle_hex = hex_nodes[needle_mask]
    print(f"  {n_hex} HEX cells total, {int(needle_mask.sum())} needle HEX cells")
    needle_node_idx = np.unique(needle_hex.ravel())
    centroids = points[needle_hex].mean(axis=1)  # (n_needle_cells, 3)
    return needle_node_idx, needle_hex, centroids


def _compute_surface_faces(needle_hex: np.ndarray, points: np.ndarray):
    """Identify surface quad faces of needle HEX cells.

    A face is a "surface" face if it appears in exactly one cell.  Each
    surface face's outward normal is computed and sign-fixed by comparing
    against the direction from the owning cell's centroid to the face
    centroid.

    Returns
    -------
    face_node_indices : np.ndarray  (n_surf_faces, 4)
        Global node indices of each surface face.
    face_normals : np.ndarray  (n_surf_faces, 3)
        Outward-pointing unit normals.
    face_centroids : np.ndarray  (n_surf_faces, 3)
    """
    n_cells = needle_hex.shape[0]
    cell_centroids = points[needle_hex].mean(axis=1)  # (n_cells, 3)

    face_to_cells: Dict[Tuple[int, int, int, int], List[Tuple[int, np.ndarray]]] = {}
    for ci in range(n_cells):
        verts = needle_hex[ci]  # (8,)
        for face_local in _HEX_FACE_LOCAL_NODES:
            ord_nodes = verts[face_local]  # ordered (4,)
            key = tuple(sorted(int(v) for v in ord_nodes))
            face_to_cells.setdefault(key, []).append((ci, ord_nodes))

    surf_face_nodes = []
    surf_face_normals = []
    surf_face_centroids = []
    for key, owners in face_to_cells.items():
        if len(owners) != 1:
            continue
        ci, ord_nodes = owners[0]
        p = points[ord_nodes]  # (4, 3)
        # Quad normal: average of the two diagonal triangle normals.
        n1 = np.cross(p[1] - p[0], p[3] - p[0])
        n2 = np.cross(p[3] - p[2], p[1] - p[2])
        n = n1 + n2
        norm = float(np.linalg.norm(n))
        if norm < 1e-12:
            continue
        n /= norm
        # Outward sign: should point from cell centroid towards face centroid.
        face_c = p.mean(axis=0)
        if np.dot(n, face_c - cell_centroids[ci]) < 0:
            n = -n
        surf_face_nodes.append(ord_nodes)
        surf_face_normals.append(n)
        surf_face_centroids.append(face_c)
    print(f"  {len(surf_face_nodes)} surface faces")
    return (
        np.asarray(surf_face_nodes, dtype=np.int64),
        np.asarray(surf_face_normals, dtype=np.float64),
        np.asarray(surf_face_centroids, dtype=np.float64),
    )


def _compute_needle_axis(points: np.ndarray, needle_node_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    needle_pts = points[needle_node_idx]
    centroid = needle_pts.mean(axis=0)
    centred = needle_pts - centroid
    _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
    axis = vt[0]
    # Sign rule (match compare_models.py): largest-abs component positive.
    if axis[int(np.argmax(np.abs(axis)))] < 0:
        axis = -axis
    return axis, centroid


def _classify_bevel_faces(face_normals: np.ndarray, axis: np.ndarray,
                          axial_min: float, axial_max: float) -> np.ndarray:
    """A face is a bevel face if abs(normal · axis) lies in [axial_min, axial_max].

    Lateral cylinder faces have normal ⊥ axis (≈ 0); a perfectly flat tip cap
    would be ≈ 1.  The bevel is in between.  We take abs because the bevel
    normal can point in either axial direction depending on the cut.
    """
    dots = np.abs(face_normals @ axis)
    return (dots >= axial_min) & (dots <= axial_max)


def _compute_arclen_to_clamp(
    points: np.ndarray,
    needle_node_idx: np.ndarray,
    axis: np.ndarray,
    centroid: np.ndarray,
    bevel_touched: np.ndarray,
    n_nodes: int,
) -> Tuple[np.ndarray, float]:
    """Per-node arc-length coordinate measured from the needle clamp end.

    For each needle node we project its rest-state position onto the needle
    axis: ``proj_i = (p_i - centroid) · axis``.  The needle is approximately
    straight in the rest state, so this axial projection *is* the arc length
    along the centreline.

    The *clamp* (proximal, driven) end is taken to be the axial extreme
    **opposite** the bevel tip.  The bevel nodes (``bevel_touched``) localise
    the tip end; the clamp is then the far extreme.  The returned coordinate
    is normalised to ``[0, 1]`` with ``0`` at the clamp and ``≈1`` at the
    tip, and is zero on non-needle (tissue) nodes.

    Returns
    -------
    arclen : np.ndarray  (n_nodes,)
        Normalised arc-length-to-clamp coordinate, zero off the needle.
    length_mm : float
        Physical axial extent of the needle (proj_max - proj_min), in mm.
    """
    proj = (points[needle_node_idx] - centroid) @ axis  # (n_needle,)
    p_min, p_max = float(proj.min()), float(proj.max())
    length = max(p_max - p_min, 1e-8)

    # Which axial extreme is the bevel/tip?  Use the mean projection of the
    # bevel nodes; the clamp is the opposite extreme.
    bevel_in_needle = np.isin(needle_node_idx, np.nonzero(bevel_touched)[0])
    if bevel_in_needle.any():
        bevel_proj = float(proj[bevel_in_needle].mean())
        clamp_proj = p_min if abs(bevel_proj - p_max) <= abs(bevel_proj - p_min) else p_max
    else:
        # No bevel detected — fall back to the min-projection end as clamp.
        clamp_proj = p_min

    s_needle = np.abs(proj - clamp_proj) / length  # [0, 1], 0 at clamp
    arclen = np.zeros(n_nodes, dtype=np.float64)
    arclen[needle_node_idx] = s_needle
    return arclen, length


def _aggregate_per_node(face_node_indices: np.ndarray, face_normals: np.ndarray,
                        n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sum incident face normals per global node, then unit-normalise.

    Returns (per_node_normal, has_any).
    """
    accum = np.zeros((n_nodes, 3), dtype=np.float64)
    touched = np.zeros(n_nodes, dtype=bool)
    for nodes, n in zip(face_node_indices, face_normals):
        for v in nodes:
            accum[int(v)] += n
            touched[int(v)] = True
    norms = np.linalg.norm(accum, axis=-1, keepdims=True)
    safe = np.where(norms > 1e-12, accum / np.maximum(norms, 1e-12), 0.0)
    return safe, touched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_file", default=None,
                        help="Where to write the .pt (default: <data_dir>/needle_geometry_features.pt).")
    parser.add_argument("--bevel_axial_min", type=float, default=0.1)
    parser.add_argument("--bevel_axial_max", type=float, default=0.95)
    args = parser.parse_args()

    out_path = args.out_file or os.path.join(args.data_dir, "needle_geometry_features.pt")

    mesh = _load_first_vtu(args.data_dir)
    points = np.asarray(mesh.points, dtype=np.float64)
    n_nodes = points.shape[0]
    print(f"Mesh has {n_nodes} nodes total")

    needle_node_idx, needle_hex, _ = _extract_needle_surface(mesh)
    axis, centroid = _compute_needle_axis(points, needle_node_idx)
    print(f"Needle axis (sign-anchored): {axis}")
    print(f"Needle centroid: {centroid}")

    face_nodes, face_normals, face_centroids = _compute_surface_faces(needle_hex, points)

    # Per-face axial component for bevel classification (post sign-anchoring).
    bevel_mask = _classify_bevel_faces(face_normals, axis,
                                        args.bevel_axial_min, args.bevel_axial_max)
    n_bevel = int(bevel_mask.sum())
    print(f"  {n_bevel} bevel faces (|normal·axis| in "
          f"[{args.bevel_axial_min}, {args.bevel_axial_max}])")
    if n_bevel == 0:
        raise RuntimeError(
            "No bevel faces detected — try relaxing --bevel_axial_min/_max bounds."
        )

    surface_node_normal, surface_touched = _aggregate_per_node(
        face_nodes, face_normals, n_nodes
    )
    bevel_node_normal, bevel_touched = _aggregate_per_node(
        face_nodes[bevel_mask], face_normals[bevel_mask], n_nodes
    )

    n_surface_nodes = int(surface_touched.sum())
    n_bevel_nodes = int(bevel_touched.sum())
    print(f"  Per-node aggregation: {n_surface_nodes} surface nodes, "
          f"{n_bevel_nodes} bevel nodes")

    arclen_to_clamp, needle_length_mm = _compute_arclen_to_clamp(
        points, needle_node_idx, axis, centroid, bevel_touched, n_nodes
    )
    print(f"  Arc-length-to-clamp: needle length {needle_length_mm:.3f} mm, "
          f"coordinate in [0, 1] (0 = clamp, 1 = tip)")

    payload = {
        "needle_node_indices": torch.from_numpy(needle_node_idx).long(),
        "bevel_node_normal": torch.from_numpy(bevel_node_normal).float(),
        "surface_node_normal": torch.from_numpy(surface_node_normal).float(),
        "is_bevel_node": torch.from_numpy(bevel_touched),
        "is_needle_surface_node": torch.from_numpy(surface_touched),
        "needle_axis": torch.from_numpy(axis).float(),
        "needle_centroid": torch.from_numpy(centroid).float(),
        "arclen_to_clamp": torch.from_numpy(arclen_to_clamp).float(),
        "needle_length_mm": float(needle_length_mm),
        "params": {
            "bevel_axial_min": args.bevel_axial_min,
            "bevel_axial_max": args.bevel_axial_max,
        },
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(payload, out_path)
    print(f"\nWrote needle geometry features to {out_path}")


if __name__ == "__main__":
    main()
