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

"""Needle-tissue dataset with dynamic spatial cropping (full mesh, no beam reduction).

Each training sample crops the full needle-tissue graph to the active insertion zone:

  - Needle nodes (Lagrangian, element_type=0) more than ``needle_crop_mm`` from
    any tissue node are excluded (the shank still outside the tissue).
  - Tissue nodes (Eulerian, element_type=1) more than ``tissue_crop_mm`` from
    any kept needle node are excluded (far-field tissue).

No beam reduction is applied; the full ~7936-node needle mesh is used.
BSMS is not supported because the subgraph topology changes every training step.
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import coalesce as _pyg_coalesce
from torch_geometric.utils import subgraph, to_undirected

from physicsnemo.datapipes.gnn.utils import load_json

try:
    import sparse_dot_mkl  # noqa: F401 — presence check only

    _BSMS_AVAILABLE = True
except ImportError:
    _BSMS_AVAILABLE = False


def _atomic_torch_save(obj, path: str) -> None:
    """Write via torch.save to a temp file then atomically rename."""
    tmp = path + f".{os.getpid()}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_save_json(obj, path: str) -> None:
    """JSON equivalent of _atomic_torch_save."""
    import json

    tmp = path + f".{os.getpid()}.tmp"
    tensors = {k: v.numpy().tolist() for k, v in obj.items()}
    with open(tmp, "w") as f:
        json.dump(tensors, f)
    os.replace(tmp, path)


_HEX_LOCAL_EDGES: np.ndarray = np.array(
    [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ],
    dtype=np.int64,
)

_CACHE_FILENAME = "preprocessed_cache.pt"
_MULTI_RUN_PATTERN = re.compile(r"-RUN-(\d+)_(\d+)\.vtu$")


def _sorted_vtu_files(data_dir: str) -> List[str]:
    """Return VTU files in *data_dir* sorted numerically by the last number in their name."""
    entries = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".vtu")
    ]
    numbers = []
    for path in entries:
        # Match the last sequence of digits (handles output_0001.vtu etc.)
        m = re.findall(r"\d+", os.path.basename(path))
        numbers.append(int(m[-1]) if m else 0)
    order = np.argsort(numbers)
    return [entries[i] for i in order]


def _is_multi_run(data_dir: str) -> bool:
    """Return True if *data_dir* contains multi-run VTU files (``*-RUN-N_T.vtu``)."""
    for f in os.listdir(data_dir):
        if _MULTI_RUN_PATTERN.search(f):
            return True
    return False


def _group_vtu_by_run(
    data_dir: str, timestep_stride: int = 1
) -> Dict[str, List[str]]:
    """Group VTU files by run ID and apply a timestep stride within each run.

    Expects filenames matching ``*-RUN-{run_id}_{timestep:04d}.vtu``.
    Runs with fewer than 2 selected frames are excluded.

    Parameters
    ----------
    data_dir : str
        Directory containing the VTU files.
    timestep_stride : int
        Keep every *timestep_stride*-th frame within each run (1 = all frames).

    Returns
    -------
    dict mapping run_id (str) to sorted list of selected file paths.
    """
    runs: Dict[str, List[Tuple[int, str]]] = {}
    for f in os.listdir(data_dir):
        m = _MULTI_RUN_PATTERN.search(f)
        if m:
            run_id, ts = m.group(1), int(m.group(2))
            runs.setdefault(run_id, []).append((ts, os.path.join(data_dir, f)))
    result: Dict[str, List[str]] = {}
    for run_id in sorted(runs.keys(), key=int):
        frames = sorted(runs[run_id])  # sort by timestep
        paths = [p for _, p in frames[::timestep_stride]]
        if len(paths) >= 2:
            result[run_id] = paths
    return result


def _build_hex_edges(
    mesh: pv.UnstructuredGrid,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build bidirected HEX-cell edges (fixed mesh topology)."""
    cell_types = mesh.celltypes
    element_type = mesh.cell_data["element_type"]
    conn = mesh.cell_connectivity
    n_hex = int((cell_types == 12).sum())

    hex_nodes = conn[: n_hex * 8].reshape(n_hex, 8)
    src = hex_nodes[:, _HEX_LOCAL_EDGES[:, 0]].ravel().astype(np.int64)
    dst = hex_nodes[:, _HEX_LOCAL_EDGES[:, 1]].ravel().astype(np.int64)
    types = np.repeat(element_type[:n_hex], 12).astype(np.int64)

    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_types = torch.tensor(types, dtype=torch.long)
    edge_index_bi, edge_types_bi = to_undirected(edge_index, edge_attr=edge_types, reduce="min")

    edge_type_onehot = torch.zeros(edge_types_bi.shape[0], 3, dtype=torch.float32)
    edge_type_onehot.scatter_(1, edge_types_bi.unsqueeze(1), 1.0)
    return edge_index_bi, edge_type_onehot


def _extract_world_edges(
    mesh: pv.UnstructuredGrid,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract bidirected world/contact edges (LINE cells, element_type=2) for one frame."""
    cell_types = mesh.celltypes
    element_type = mesh.cell_data["element_type"]
    conn = mesh.cell_connectivity
    n_hex = int((cell_types == 12).sum())
    n_line = int((cell_types == 3).sum())

    if n_line == 0:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0, 3), dtype=torch.float32),
        )

    line_nodes = conn[n_hex * 8 : n_hex * 8 + n_line * 2].reshape(n_line, 2)
    src = line_nodes[:, 0].astype(np.int64)
    dst = line_nodes[:, 1].astype(np.int64)
    types = element_type[n_hex : n_hex + n_line].astype(np.int64)

    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_types = torch.tensor(types, dtype=torch.long)
    edge_index_bi, edge_types_bi = to_undirected(edge_index, edge_attr=edge_types, reduce="min")

    edge_type_onehot = torch.zeros(edge_types_bi.shape[0], 3, dtype=torch.float32)
    edge_type_onehot.scatter_(1, edge_types_bi.unsqueeze(1), 1.0)
    return edge_index_bi, edge_type_onehot


def _cell_features_to_nodes(
    mesh: pv.UnstructuredGrid, n_nodes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate HEX cell features (EVF_VOID, S) to nodes by averaging."""
    cell_types = mesh.celltypes
    conn = mesh.cell_connectivity
    n_hex = int((cell_types == 12).sum())

    hex_nodes = conn[: n_hex * 8].reshape(n_hex, 8)
    evf_cell = mesh.cell_data["EVF_VOID"][:n_hex]
    s_cell = mesh.cell_data["S"][:n_hex]

    flat_nodes = hex_nodes.ravel()
    evf_rep = np.repeat(evf_cell, 8)
    s_rep = np.repeat(s_cell, 8, axis=0)

    evf_node = np.zeros(n_nodes, dtype=np.float64)
    s_node = np.zeros((n_nodes, 6), dtype=np.float64)
    count = np.zeros(n_nodes, dtype=np.float64)

    np.add.at(evf_node, flat_nodes, evf_rep)
    np.add.at(s_node, flat_nodes, s_rep)
    np.add.at(count, flat_nodes, 1.0)

    count = np.maximum(count, 1.0)
    evf_node /= count
    s_node /= count[:, None]

    return evf_node[:, None].astype(np.float32), s_node.astype(np.float32)


# Static material property keys and their feature dimensions.
_STATIC_PROP_KEYS = ["mat_E", "mat_c10", "mat_density", "mat_fiber", "mat_k1", "mat_k2", "mat_kappa", "mat_nu"]
_STATIC_PROP_DIMS = [1, 1, 1, 3, 1, 1, 1, 1]


def _load_node_props(mesh: pv.UnstructuredGrid) -> Dict[str, torch.Tensor]:
    """Load static material node properties from the reference VTU frame.

    Each property is read from ``mesh.point_data`` if present; otherwise it
    defaults to zeros.  The result is a dict of ``(n_nodes, dim)`` float32
    tensors that remain constant across all simulation frames.

    Parameters
    ----------
    mesh : pv.UnstructuredGrid
        The reference frame mesh (typically frame 0).

    Returns
    -------
    dict mapping property name to ``(n_nodes, dim)`` float32 tensor.
    """
    n = mesh.n_points
    props: Dict[str, torch.Tensor] = {}
    for key, dim in zip(_STATIC_PROP_KEYS, _STATIC_PROP_DIMS):
        if key in mesh.point_data:
            val = mesh.point_data[key].astype(np.float32)
            if val.ndim == 1:
                val = val[:, None]
        else:
            val = np.zeros((n, dim), dtype=np.float32)
        props[key] = torch.from_numpy(val)
    return props


def _process_single_frame(path: str) -> Dict:
    """Load one VTU frame and return its per-frame features.

    Called from ThreadPoolExecutor workers — must not mutate shared state.
    """
    mesh = pv.read(path)
    n_nodes = mesh.n_points
    evf_node, s_node = _cell_features_to_nodes(mesh, n_nodes)
    if "CPRESS" in mesh.point_data:
        cp = mesh.point_data["CPRESS"].astype(np.float32)
        if cp.ndim == 1:
            cp = cp[:, None]
    else:
        cp = np.zeros((n_nodes, 1), dtype=np.float32)
    return {
        "coord": mesh.points.astype(np.float32),
        "u": mesh.point_data["U"].astype(np.float32),
        "v": mesh.point_data["V"].astype(np.float32),
        "a": mesh.point_data["A"].astype(np.float32),
        "evf": evf_node,
        "s": s_node,
        "cpress": cp,
        "world_edges": _extract_world_edges(mesh),
    }


def _process_all_frames(vtu_files: List[str], num_workers: int = 8) -> Dict:
    """Load all VTU frames in parallel, build fixed HEX topology and per-frame world edges."""
    import time
    from concurrent.futures import as_completed

    n = len(vtu_files)
    print(f"Processing {n} VTU frames with {num_workers} workers (result will be cached)...")
    t0 = time.time()

    # Read frame 0 on the main process to build fixed topology and static node props.
    mesh_ref = pv.read(vtu_files[0])
    node_props = _load_node_props(mesh_ref)
    print("Building fixed HEX topology...")
    edge_index, edge_type_onehot = _build_hex_edges(mesh_ref)
    print(f"  topology done ({time.time() - t0:.1f}s)")

    # Process all frames in parallel, printing progress as each completes.
    results = [None] * n
    done = 0
    report_every = max(1, n // 20)  # print at ~5% intervals

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process_single_frame, p): i for i, p in enumerate(vtu_files)}
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            done += 1
            if done % report_every == 0 or done == n:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (n - done) / rate if rate > 0 else 0
                print(f"  frames {done}/{n}  ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    keys = ("coord", "u", "v", "a", "evf", "s", "cpress")
    frame_tensors = {
        k: torch.from_numpy(np.stack([r[k] for r in results], axis=0)) for k in keys
    }
    world_edges = [r["world_edges"] for r in results]
    print(f"  stacking done — total {time.time() - t0:.1f}s")

    return {
        "edge_index": edge_index,
        "edge_type_onehot": edge_type_onehot,
        "world_edges": world_edges,
        "frame_tensors": frame_tensors,
        "node_props": node_props,
    }


def _get_needle_tissue_node_sets(
    edge_index: torch.Tensor, edge_type_onehot: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    """Return sorted numpy arrays of needle (et=0) and tissue (et=1) node indices."""
    et0 = edge_type_onehot[:, 0].bool()
    et1 = edge_type_onehot[:, 1].bool()
    needle_nodes = edge_index[:, et0].reshape(-1).unique().sort().values.numpy()
    tissue_nodes = edge_index[:, et1].reshape(-1).unique().sort().values.numpy()
    return needle_nodes, tissue_nodes


# ---------------------------------------------------------------------------
# Mesh reduction helpers
# ---------------------------------------------------------------------------

def _beam_assignment(
    needle_coords_frame0: np.ndarray, beam_spacing_mm: float
) -> Tuple[np.ndarray, int]:
    """Assign needle nodes to 1-D beam nodes using PCA on frame-0 positions.

    Cluster membership is computed in the Lagrangian reference configuration
    so it remains fixed as the needle bends through later frames.

    Parameters
    ----------
    needle_coords_frame0 : np.ndarray, shape (n_needle, 3)
        Needle node coordinates in the reference (frame 0) configuration.
    beam_spacing_mm : float
        Target spacing between beam nodes (mm).

    Returns
    -------
    beam_assignment : np.ndarray, shape (n_needle,) int64
        Index in ``[0, N_beam)`` for each needle node.
    N_beam : int
        Number of beam nodes.
    """
    centered = needle_coords_frame0 - needle_coords_frame0.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ Vt[0]
    proj_min, proj_max = float(proj.min()), float(proj.max())
    proj_range = proj_max - proj_min
    N_beam = max(2, int(np.ceil(proj_range / beam_spacing_mm)) + 1)
    proj_norm = (proj - proj_min) / max(proj_range, 1e-8)
    beam_asgn = np.round(proj_norm * (N_beam - 1)).astype(np.int64)
    beam_asgn = np.clip(beam_asgn, 0, N_beam - 1)
    return beam_asgn, N_beam


def _apply_beam_reduction(raw_cache: Dict, beam_spacing_mm: float) -> Dict:
    """Replace needle nodes with a coarse 1-D beam representation.

    Needle nodes (edge type 0) are clustered into ``N_beam`` beam nodes using
    PCA on frame-0 positions.  Per-frame beam features are cluster means.
    Needle HEX edges are replaced by a bidirectional chain.  World edges are
    remapped so needle endpoints become their beam node.  Tissue nodes are
    unchanged.  Per-frame world edges are stored with original node indices so
    that ``_build_graph`` can apply the ``_beam_old_to_new`` remapping at
    sample time.

    Parameters
    ----------
    raw_cache : Dict
        Output of ``_process_all_frames``.
    beam_spacing_mm : float
        Target spacing between successive beam nodes (mm).

    Returns
    -------
    Dict with the same structure as *raw_cache* plus beam metadata keys.
    """
    edge_index_orig = raw_cache["edge_index"]
    edge_type_onehot_orig = raw_cache["edge_type_onehot"]
    frame_tensors_orig = raw_cache["frame_tensors"]

    n_frames = frame_tensors_orig["coord"].shape[0]
    n_nodes_orig = frame_tensors_orig["coord"].shape[1]

    needle_node_indices, tissue_node_indices = _get_needle_tissue_node_sets(
        edge_index_orig, edge_type_onehot_orig
    )
    n_needle = len(needle_node_indices)
    n_tissue = len(tissue_node_indices)

    coords_f0 = frame_tensors_orig["coord"][0].numpy()
    beam_asgn, N_beam = _beam_assignment(coords_f0[needle_node_indices], beam_spacing_mm)
    n_nodes_new = n_tissue + N_beam

    print(
        f"Beam reduction: {n_needle} needle → {N_beam} beam nodes "
        f"({beam_spacing_mm:.2g} mm spacing) | total: {n_nodes_orig} → {n_nodes_new}"
    )

    # old global index → new index (tissue first, then beam nodes)
    old_to_new = np.full(n_nodes_orig, -1, dtype=np.int64)
    for new_i, old_i in enumerate(tissue_node_indices):
        old_to_new[old_i] = new_i
    for j, old_i in enumerate(needle_node_indices):
        old_to_new[old_i] = n_tissue + int(beam_asgn[j])

    # Remap static HEX edges: drop needle-needle (et=0), keep tissue+world, remap endpoints
    ei = edge_index_orig.numpy()
    et = edge_type_onehot_orig.numpy()
    keep = et[:, 0] == 0  # False where type=0 (needle HEX)
    src_k = old_to_new[ei[0, keep]]
    dst_k = old_to_new[ei[1, keep]]
    et_k = et[keep]

    # Bidirectional chain for beam nodes (edge type 0)
    idx = np.arange(N_beam - 1, dtype=np.int64)
    cs = np.concatenate([n_tissue + idx, n_tissue + idx + 1])
    cd = np.concatenate([n_tissue + idx + 1, n_tissue + idx])
    ce = np.zeros((len(cs), 3), dtype=np.float32)
    ce[:, 0] = 1.0

    all_src = np.concatenate([src_k, cs])
    all_dst = np.concatenate([dst_k, cd])
    all_et = np.concatenate([et_k, ce])

    edge_index_new = torch.from_numpy(np.stack([all_src, all_dst])).long()
    edge_type_onehot_new = torch.from_numpy(all_et)
    edge_index_new, edge_type_onehot_new = _pyg_coalesce(
        edge_index_new, edge_type_onehot_new, n_nodes_new, reduce="min"
    )

    # Build new frame tensors: scatter-mean needle features into beam nodes
    tissue_idx_t = torch.tensor(tissue_node_indices, dtype=torch.long)
    needle_idx_t = torch.tensor(needle_node_indices, dtype=torch.long)
    ba = torch.tensor(beam_asgn, dtype=torch.long)

    new_frame_tensors: Dict[str, torch.Tensor] = {}
    for key, orig in frame_tensors_orig.items():
        d = orig.shape[-1]
        t_feat = orig[:, tissue_idx_t, :]
        n_feat = orig[:, needle_idx_t, :]
        ba_exp = ba.view(1, -1, 1).expand(n_frames, n_needle, d)
        b_feat = torch.zeros(n_frames, N_beam, d, dtype=orig.dtype)
        b_feat.scatter_add_(1, ba_exp, n_feat)
        count = torch.zeros(N_beam).scatter_add_(0, ba, torch.ones(n_needle))
        b_feat /= count.view(1, N_beam, 1).clamp(min=1.0)
        new_frame_tensors[key] = torch.cat([t_feat, b_feat], dim=1)

    # Remap static material properties
    node_props_orig = raw_cache.get("node_props", {})
    new_node_props: Dict[str, torch.Tensor] = {}
    for key, prop in node_props_orig.items():
        d = prop.shape[-1]
        t_prop = prop[tissue_idx_t]
        n_prop = prop[needle_idx_t].float()
        ba_exp_1d = ba.view(-1, 1).expand(n_needle, d)
        b_prop = torch.zeros(N_beam, d, dtype=n_prop.dtype)
        b_prop.scatter_add_(0, ba_exp_1d, n_prop)
        count_1d = torch.zeros(N_beam).scatter_add_(0, ba, torch.ones(n_needle))
        b_prop /= count_1d.view(N_beam, 1).clamp(min=1.0)
        new_node_props[key] = torch.cat([t_prop, b_prop], dim=0)

    # Remap per-frame world edges to beam-reduced node space.
    # Multiple original needle nodes may map to the same beam node, so we
    # coalesce duplicates (keep first occurrence).
    new_world_edges = []
    for (w_ei, w_et) in raw_cache["world_edges"]:
        if w_ei.shape[1] == 0:
            new_world_edges.append((w_ei, w_et))
            continue
        src_r = torch.from_numpy(old_to_new[w_ei[0].numpy()])
        dst_r = torch.from_numpy(old_to_new[w_ei[1].numpy()])
        valid = (src_r >= 0) & (dst_r >= 0)
        we_new = torch.stack([src_r[valid], dst_r[valid]], dim=0)
        wet_new = w_et[valid]
        # Deduplicate edges that collapsed to the same beam node
        if we_new.shape[1] > 0:
            we_new, wet_new = _pyg_coalesce(we_new, wet_new, n_nodes_new, reduce="min")
        new_world_edges.append((we_new, wet_new))

    return {
        "edge_index": edge_index_new,
        "edge_type_onehot": edge_type_onehot_new,
        "frame_tensors": new_frame_tensors,
        "node_props": new_node_props,
        "world_edges": new_world_edges,
        "tissue_node_indices": tissue_idx_t,
        "needle_node_indices": needle_idx_t,
        "beam_assignment": ba,
        "n_nodes_orig": n_nodes_orig,
        "n_tissue": n_tissue,
    }


def _downsample_tissue(cache: Dict, spacing_mm: float) -> Dict:
    """Subsample tissue nodes so that no two kept nodes are closer than *spacing_mm*.

    Uses a voxel-grid approach: each tissue node is assigned to a 3-D voxel of
    side *spacing_mm*.  For each occupied voxel the node closest to the voxel
    centre is kept.  All beam/needle nodes are always kept.  Per-frame world
    edges that connect the needle to a removed tissue node are dropped.

    Parameters
    ----------
    cache : Dict
        Output of ``_apply_beam_reduction`` (or ``_process_all_frames`` if no
        beam reduction was applied).  Must contain the beam metadata keys so
        that needle vs tissue nodes can be identified.
    spacing_mm : float
        Voxel side length (mm).  Tissue nodes are subsampled to one per voxel.

    Returns
    -------
    Dict with the same structure as *cache*, restricted to kept nodes.
    """
    edge_index = cache["edge_index"]
    edge_type_onehot = cache["edge_type_onehot"]
    frame_tensors = cache["frame_tensors"]

    n_nodes = frame_tensors["coord"].shape[1]

    # Identify needle vs tissue in the (possibly beam-reduced) node set.
    needle_np, tissue_np = _get_needle_tissue_node_sets(edge_index, edge_type_onehot)

    # Use frame-0 positions for voxel assignment (tissue nodes are Eulerian,
    # so their positions are the same across all frames and runs).
    coord_f0 = frame_tensors["coord"][0].numpy()  # (n_nodes, 3)
    tissue_coords = coord_f0[tissue_np]            # (n_tissue, 3)

    # Assign each tissue node to a voxel key, keep closest to voxel centre.
    voxel_keys = np.floor(tissue_coords / spacing_mm).astype(np.int64)
    voxel_best: Dict[tuple, Tuple[float, int]] = {}  # key → (dist, local_idx)
    for local_i, (key_row, coord) in enumerate(zip(voxel_keys, tissue_coords)):
        key = tuple(key_row.tolist())
        centre = (key_row + 0.5) * spacing_mm
        dist = float(np.linalg.norm(coord - centre))
        if key not in voxel_best or dist < voxel_best[key][0]:
            voxel_best[key] = (dist, local_i)
    kept_local = sorted(v[1] for v in voxel_best.values())
    kept_tissue = tissue_np[kept_local]  # global indices of kept tissue nodes

    n_tissue_orig = len(tissue_np)
    n_tissue_kept = len(kept_tissue)
    n_needle = len(needle_np)
    n_nodes_new = n_tissue_kept + n_needle
    print(
        f"Tissue downsampling ({spacing_mm:.2g} mm): "
        f"{n_tissue_orig} → {n_tissue_kept} tissue nodes | "
        f"total: {n_nodes} → {n_nodes_new}"
    )

    # Build old → new index map (kept tissue first, then needle nodes unchanged)
    old_to_new = np.full(n_nodes, -1, dtype=np.int64)
    for new_i, old_i in enumerate(kept_tissue):
        old_to_new[old_i] = new_i
    for new_i, old_i in enumerate(needle_np):
        old_to_new[old_i] = n_tissue_kept + new_i

    # Filter and remap edges
    ei = edge_index.numpy()
    et = edge_type_onehot.numpy()
    src_new = old_to_new[ei[0]]
    dst_new = old_to_new[ei[1]]
    keep_mask = (src_new >= 0) & (dst_new >= 0)
    edge_index_new = torch.from_numpy(np.stack([src_new[keep_mask], dst_new[keep_mask]])).long()
    edge_type_onehot_new = torch.from_numpy(et[keep_mask])

    # Remap frame tensors (tissue subset + all needle/beam nodes)
    kept_all = np.concatenate([kept_tissue, needle_np])
    kept_all_t = torch.from_numpy(kept_all.astype(np.int64))
    new_frame_tensors = {k: v[:, kept_all_t, :] for k, v in frame_tensors.items()}

    # Remap node properties
    new_node_props = {k: v[kept_all_t] for k, v in cache.get("node_props", {}).items()}

    # Filter per-frame world edges (drop edges to removed tissue nodes)
    new_world_edges = []
    for (w_ei, w_et) in cache["world_edges"]:
        if w_ei.shape[1] == 0:
            new_world_edges.append((w_ei, w_et))
            continue
        src_r = torch.from_numpy(old_to_new[w_ei[0].numpy()])
        dst_r = torch.from_numpy(old_to_new[w_ei[1].numpy()])
        valid = (src_r >= 0) & (dst_r >= 0)
        new_world_edges.append((
            torch.stack([src_r[valid], dst_r[valid]], dim=0),
            w_et[valid],
        ))

    result = {
        "edge_index": edge_index_new,
        "edge_type_onehot": edge_type_onehot_new,
        "frame_tensors": new_frame_tensors,
        "node_props": new_node_props,
        # World edges are already remapped here (tissue downsampling removes nodes
        # permanently; we can't defer remapping to sample time).
        "world_edges": new_world_edges,
    }
    # Carry beam metadata through if present, updating indices to new node space
    if "tissue_node_indices" in cache:
        # After downsampling the needle indices shift to n_tissue_kept..n_nodes_new-1
        result["tissue_node_indices"] = torch.from_numpy(np.arange(n_tissue_kept, dtype=np.int64))
        result["needle_node_indices"] = cache["needle_node_indices"]
        result["beam_assignment"] = cache["beam_assignment"]
        result["n_nodes_orig"] = cache["n_nodes_orig"]
        result["n_tissue"] = n_tissue_kept
    return result


def _precompute_bsms_full(
    edge_index: torch.Tensor,
    coord_ref: torch.Tensor,
    n_nodes: int,
    num_levels: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Compute bi-stride multi-scale graph structure for the full (unpartitioned) mesh.

    Parameters
    ----------
    edge_index : torch.Tensor, shape (2, E)
        Static HEX edge index (full mesh).
    coord_ref : torch.Tensor, shape (N, 3)
        Node positions used for seed selection (frame-0 coordinates).
    n_nodes : int
        Total node count.
    num_levels : int
        Number of coarsening levels.

    Returns
    -------
    ms_edges : list of tensors, one per level (shape (2, E_l) each)
    ms_ids : list of tensors, one per level (shape (N_l,) each)
    """
    if not _BSMS_AVAILABLE:
        raise ImportError(
            "BiStride multi-scale graph requires sparse_dot_mkl. "
            "Install with: uv add sparse-dot-mkl"
        )
    import importlib

    BistrideMultiLayerGraph = importlib.import_module(
        "physicsnemo.datapipes.gnn.bsms"
    ).BistrideMultiLayerGraph

    part_data = Data(edge_index=edge_index, num_nodes=n_nodes)
    part_data.pos = coord_ref

    mlg = BistrideMultiLayerGraph(part_data, num_levels)
    _, ms_edges_raw, ms_ids_raw = mlg.get_multi_layer_graphs()

    ms_edges = [torch.tensor(e, dtype=torch.long) for e in ms_edges_raw]
    ms_ids = [torch.tensor(ids, dtype=torch.long) for ids in ms_ids_raw]
    level_strs = [f"L{i}={e.shape[1]}e" for i, e in enumerate(ms_edges)]
    for i, ids in enumerate(ms_ids):
        level_strs[i] += f"/{ids.shape[0]}n"
    print(f"BSMS: {n_nodes} nodes → " + ", ".join(level_strs))
    return ms_edges, ms_ids


class NeedleTissueDataset(Dataset):
    """
    Temporal needle-tissue dataset with dynamic spatial cropping.

    Uses the full needle mesh (no beam reduction).  Each sample corresponds to
    one temporal pair; the subgraph is dynamically cropped to the active needle-
    tissue interaction zone.  Three crop strategies are available and can be
    mixed during training via ``crop_strategy_weights``:

    ``"proximity"`` (default)
        Standard strategy: exclude needle nodes farther than ``needle_crop_mm``
        from any tissue node, then exclude tissue nodes farther than
        ``tissue_crop_mm`` from any kept needle node.

    ``"slice"``
        Take a slab of thickness ``2 * slice_half_thickness_mm`` perpendicular
        to the needle's principal axis, centred at a uniformly random axial
        position along the needle.  Keeps all needle and tissue nodes whose
        axial projection falls within the slab.  Provides augmentation by
        exposing different cross-sections of the needle-tissue interaction.

    ``"full_needle"``
        Keep every needle node and restrict tissue to nodes within
        ``full_needle_tissue_mm`` of any needle node.  Ensures the model always
        sees the complete needle geometry while limiting tissue extent.

    On ``"validation"`` and ``"test"`` splits only ``"proximity"`` is used,
    regardless of ``crop_strategy_weights``.

    Parameters
    ----------
    data_dir : str
        Directory containing ``output_XXXX.vtu`` files.
    split : str
        ``"train"``, ``"validation"``, or ``"test"``.
    needle_crop_mm : float
        Proximity strategy: max distance (mm) of needle nodes from any tissue
        node.
    tissue_crop_mm : float
        Proximity strategy: max distance (mm) of tissue nodes from any kept
        needle node.
    slice_half_thickness_mm : float
        Slice strategy: half-thickness (mm) of the axial slab.
    full_needle_tissue_mm : float
        Full-needle strategy: max distance (mm) of tissue nodes from any needle
        node.
    crop_strategy_weights : tuple of float
        Unnormalised sampling weights for ``("proximity", "slice",
        "full_needle")``.  Only applied during training.
    train_fraction, val_fraction : float
        Temporal split fractions.
    stats_path : str
        Directory for normalisation JSON files.
    cache_dir : str, optional
        Directory for the preprocessed cache file.
    """

    STATIC_PROP_KEYS = _STATIC_PROP_KEYS
    STATIC_PROP_DIMS = _STATIC_PROP_DIMS

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        needle_crop_mm: float = 10.0,
        tissue_crop_mm: float = 30.0,
        slice_half_thickness_mm: float = 20.0,
        full_needle_tissue_mm: float = 10.0,
        crop_strategy_weights: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        train_fraction: float = 0.8,
        val_fraction: float = 0.1,
        stats_path: str = ".",
        cache_dir: Optional[str] = None,
        timestep_stride: int = 1,
        use_cpress: bool = True,
        per_region_norm: bool = False,
        max_frames_per_run: Optional[int] = None,
        beam_spacing_mm: float = 0.0,
        tissue_downsample_mm: float = 0.0,
        use_bsms: bool = False,
        num_bsms_levels: int = 2,
    ):
        if split not in ("train", "validation", "test"):
            raise ValueError(f"split must be 'train', 'validation', or 'test', got '{split}'")

        self.use_cpress = use_cpress
        self.per_region_norm = per_region_norm
        if use_cpress:
            self.INPUT_KEYS = ["coord", "u", "v", "a", "evf", "s", "cpress"]
            self.INPUT_DIMS = [3, 3, 3, 3, 1, 6, 1]
            self.TARGET_KEYS = ["u", "v", "a", "evf", "s", "cpress"]
            self.TARGET_DIMS = [3, 3, 3, 1, 6, 1]
        else:
            self.INPUT_KEYS = ["coord", "u", "v", "a", "evf", "s"]
            self.INPUT_DIMS = [3, 3, 3, 3, 1, 6]
            self.TARGET_KEYS = ["u", "v", "a", "evf", "s"]
            self.TARGET_DIMS = [3, 3, 3, 1, 6]

        self.split = split
        self.stats_path = stats_path
        self.needle_crop_mm = needle_crop_mm
        self.tissue_crop_mm = tissue_crop_mm
        self.slice_half_thickness_mm = slice_half_thickness_mm
        self.full_needle_tissue_mm = full_needle_tissue_mm
        weights = np.array(crop_strategy_weights, dtype=np.float64)
        self._crop_probs = weights / weights.sum()
        os.makedirs(stats_path, exist_ok=True)

        # Per-run frame data; each entry holds frame tensors, world edges, and
        # static node props for one simulation run.
        self._run_data: List[Dict] = []
        # Flat list of (run_local_idx, t_local) pairs — one entry per training sample.
        self._samples: List[Tuple[int, int]] = []

        # Beam reduction: remap original needle indices → reduced node indices.
        # None = no beam reduction.
        self._beam_old_to_new: Optional[np.ndarray] = None

        # Bi-stride BSMS: (ms_edges, ms_ids) precomputed for the full mesh.
        self._bsms_data: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None
        self._use_bsms = use_bsms

        if _is_multi_run(data_dir):
            # ---- Multi-run mode: split by run, subsample timesteps ----------
            run_files = _group_vtu_by_run(data_dir, timestep_stride)
            run_ids = list(run_files.keys())
            n_runs = len(run_ids)

            n_train_runs = max(1, int(n_runs * train_fraction))
            n_val_runs = max(1, int(n_runs * val_fraction))

            if split == "train":
                split_run_ids = run_ids[:n_train_runs]
            elif split == "validation":
                split_run_ids = run_ids[n_train_runs : n_train_runs + n_val_runs]
            else:
                split_run_ids = run_ids[n_train_runs + n_val_runs :]

            if not split_run_ids:
                raise ValueError(
                    f"No runs assigned to '{split}' split ({n_runs} total)."
                )

            print(
                f"Multi-run '{split}' split: {len(split_run_ids)}/{n_runs} runs, "
                f"stride={timestep_stride}"
            )

            self.edge_index: Optional[torch.Tensor] = None
            self.edge_type_onehot: Optional[torch.Tensor] = None
            self.n_nodes: int = 0

            for r_idx, run_id in enumerate(split_run_ids):
                vtu_files = run_files[run_id]
                n_frames_run = len(vtu_files)
                cache_path = os.path.join(
                    cache_dir or data_dir,
                    f"preprocessed_cache_RUN-{run_id}.pt",
                )
                if os.path.exists(cache_path):
                    cache = torch.load(cache_path, weights_only=False)
                    cached_frames = cache.get("frame_tensors", {}).get("coord")
                    cached_n = len(cached_frames) if cached_frames is not None else 0
                    if (
                        "world_edges" not in cache
                        or "node_props" not in cache
                        or "cpress" not in cache.get("frame_tensors", {})
                        or cached_n != n_frames_run
                    ):
                        print(
                            f"Cache outdated for RUN-{run_id} "
                            f"(cached={cached_n}, on-disk={n_frames_run}) — regenerating ..."
                        )
                        cache = _process_all_frames(vtu_files)
                        _atomic_torch_save(cache, cache_path)
                else:
                    print(
                        f"Building cache for RUN-{run_id} ({n_frames_run} frames)..."
                    )
                    cache = _process_all_frames(vtu_files)
                    _atomic_torch_save(cache, cache_path)
                    print(f"  → saved to {cache_path}")

                # ---- Optional mesh reduction (beam + tissue) ----------------
                # Build a reduction cache tag so each (beam_mm, tissue_mm) combo
                # gets its own file and won't conflict with other experiments.
                _beam_tag = f"_b{beam_spacing_mm:.2g}mm" if beam_spacing_mm > 0.0 else ""
                _tissue_tag = f"_t{tissue_downsample_mm:.2g}mm" if tissue_downsample_mm > 0.0 else ""
                if _beam_tag or _tissue_tag:
                    red_name = (
                        f"reduced_cache_RUN-{run_id}{_beam_tag}{_tissue_tag}.pt"
                    )
                    red_path = os.path.join(cache_dir or data_dir, red_name)
                    if os.path.exists(red_path):
                        graph_cache = torch.load(red_path, weights_only=False)
                    else:
                        graph_cache = cache
                        if beam_spacing_mm > 0.0:
                            graph_cache = _apply_beam_reduction(graph_cache, beam_spacing_mm)
                        if tissue_downsample_mm > 0.0:
                            graph_cache = _downsample_tissue(graph_cache, tissue_downsample_mm)
                        _atomic_torch_save(graph_cache, red_path)
                        print(f"  → reduction cache saved to {red_path}")
                else:
                    graph_cache = cache

                if self.edge_index is None:
                    self.edge_index = graph_cache["edge_index"]
                    self.edge_type_onehot = graph_cache["edge_type_onehot"]
                    self.n_nodes = int(self.edge_index.max().item()) + 1

                # ---- Optionally subsample frames to cap RAM ------------------
                n_frames_graph = graph_cache["frame_tensors"]["coord"].shape[0]
                if max_frames_per_run is not None and n_frames_graph > max_frames_per_run:
                    keep_idx = np.round(
                        np.linspace(0, n_frames_graph - 1, max_frames_per_run)
                    ).astype(int)
                    keep_idx = sorted(set(keep_idx.tolist()))
                    run_frames: Dict = {
                        key: graph_cache["frame_tensors"][key][keep_idx]
                        for key in self.INPUT_KEYS
                    }
                    run_world_edges = [graph_cache["world_edges"][i] for i in keep_idx]
                    n_kept = len(keep_idx)
                    print(
                        f"  RUN-{run_id}: subsampled {n_frames_graph} → {n_kept} frames "
                        f"(max_frames_per_run={max_frames_per_run})"
                    )
                else:
                    run_frames = {
                        key: graph_cache["frame_tensors"][key]
                        for key in self.INPUT_KEYS
                    }
                    run_world_edges = graph_cache["world_edges"]
                    n_kept = n_frames_graph

                self._run_data.append(
                    {
                        "frame_tensors": run_frames,
                        "world_edges": run_world_edges,
                        "node_props": graph_cache["node_props"],
                    }
                )
                for t in range(n_kept - 1):
                    self._samples.append((r_idx, t))

        else:
            # ---- Single-run (legacy) mode -----------------------------------
            vtu_files = _sorted_vtu_files(data_dir)
            n_frames = len(vtu_files)
            n_pairs = n_frames - 1

            n_train = int(n_pairs * train_fraction)
            n_val = int(n_pairs * val_fraction)
            split_ranges = {
                "train": (0, n_train),
                "validation": (n_train, n_train + n_val),
                "test": (n_train + n_val, n_pairs),
            }
            start, end = split_ranges[split]
            n_pairs_split = end - start

            if n_pairs_split == 0:
                raise ValueError(
                    f"No pairs in '{split}' split for {n_frames} frames."
                )

            raw_cache_path = os.path.join(cache_dir or data_dir, _CACHE_FILENAME)
            if os.path.exists(raw_cache_path):
                cache = torch.load(raw_cache_path, weights_only=False)
                if (
                    "world_edges" not in cache
                    or "node_props" not in cache
                    or "cpress" not in cache.get("frame_tensors", {})
                ):
                    print("Cache outdated — regenerating ...")
                    cache = _process_all_frames(vtu_files)
                    _atomic_torch_save(cache, raw_cache_path)
            else:
                cache = _process_all_frames(vtu_files)
                _atomic_torch_save(cache, raw_cache_path)
                print(f"Cache saved to {raw_cache_path}")

            self.edge_index = cache["edge_index"]
            self.edge_type_onehot = cache["edge_type_onehot"]
            self.n_nodes = int(self.edge_index.max().item()) + 1

            frames_needed = list(range(start, end + 1))
            self._run_data.append(
                {
                    "frame_tensors": {
                        key: cache["frame_tensors"][key][frames_needed]
                        for key in self.INPUT_KEYS
                    },
                    "world_edges": [cache["world_edges"][i] for i in frames_needed],
                    "node_props": cache["node_props"],
                }
            )
            for t in range(n_pairs_split):
                self._samples.append((0, t))

        # ---- BSMS precomputation (bistride mode, fixed topology) ------------
        if use_bsms:
            _beam_tag = f"_b{beam_spacing_mm:.2g}mm" if beam_spacing_mm > 0.0 else ""
            _tissue_tag = f"_t{tissue_downsample_mm:.2g}mm" if tissue_downsample_mm > 0.0 else ""
            bsms_name = f"bsms_cache_l{num_bsms_levels}{_beam_tag}{_tissue_tag}.pt"
            bsms_path = os.path.join(cache_dir or data_dir, bsms_name)
            if os.path.exists(bsms_path):
                print(f"Loading BSMS cache from {bsms_path} ...")
                bsms_saved = torch.load(bsms_path, weights_only=False)
                self._bsms_data = (bsms_saved["ms_edges"], bsms_saved["ms_ids"])
            else:
                print(f"Precomputing BSMS ({num_bsms_levels} levels, {self.n_nodes} nodes)...")
                # Use frame-0 coordinates from first run for seed selection.
                coord_ref = self._run_data[0]["frame_tensors"]["coord"][0]
                ms_edges, ms_ids = _precompute_bsms_full(
                    self.edge_index, coord_ref, self.n_nodes, num_bsms_levels
                )
                self._bsms_data = (ms_edges, ms_ids)
                _atomic_torch_save({"ms_edges": ms_edges, "ms_ids": ms_ids}, bsms_path)
                print(f"BSMS cache saved to {bsms_path}")

        # ---- Needle / tissue node index sets (topology-invariant) ----------
        needle_idx, tissue_idx = _get_needle_tissue_node_sets(
            self.edge_index, self.edge_type_onehot
        )
        self.needle_node_indices: np.ndarray = needle_idx
        self.tissue_node_indices: np.ndarray = tissue_idx
        # Cached as torch long tensors for fast per-region indexing in _build_graph
        self._needle_idx_t = torch.from_numpy(needle_idx.astype(np.int64))
        self._tissue_idx_t = torch.from_numpy(tissue_idx.astype(np.int64))
        print(
            f"Node sets: {len(needle_idx)} needle, {len(tissue_idx)} tissue "
            f"(total {self.n_nodes})"
        )

        self.length = len(self._samples)

        strategy_names = ("proximity", "slice", "full_needle")
        active = [f"{s}({p:.0%})" for s, p in zip(strategy_names, self._crop_probs) if p > 0]
        print(
            f"'{split}' split ready: {self.length} samples "
            f"| {len(self._run_data)} run(s) "
            f"| crops: {', '.join(active)}"
        )

        # ---- Normalisation statistics ---------------------------------------
        if split == "train":
            self._node_stats, self._target_stats = self._compute_stats()
            _atomic_save_json(self._node_stats, os.path.join(stats_path, "node_stats.json"))
            _atomic_save_json(self._target_stats, os.path.join(stats_path, "target_stats.json"))
        else:
            self._node_stats = load_json(os.path.join(stats_path, "node_stats.json"))
            self._target_stats = load_json(os.path.join(stats_path, "target_stats.json"))

    # ------------------------------------------------------------------
    # Dimension properties (depend on use_cpress)
    # ------------------------------------------------------------------

    @property
    def input_dim_nodes(self) -> int:
        """Total node feature dimension: dynamic inputs + static material props."""
        return sum(self.INPUT_DIMS) + sum(self.STATIC_PROP_DIMS)

    @property
    def output_dim(self) -> int:
        """Total target output dimension."""
        return sum(self.TARGET_DIMS)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        graph = self._build_graph(idx)
        if not self._use_bsms:
            return graph
        ms_edges, ms_ids = self._bsms_data
        return {"graph": graph, "ms_edges": ms_edges, "ms_ids": ms_ids}

    def _crop_nodes(self, coord_t: torch.Tensor) -> torch.Tensor:
        """Dispatch to a crop strategy, sampling randomly during training."""
        if self.split != "train":
            return self._crop_proximity(coord_t)
        strategy = np.random.choice(
            ["proximity", "slice", "full_needle"], p=self._crop_probs
        )
        if strategy == "slice":
            return self._crop_slice(coord_t)
        if strategy == "full_needle":
            return self._crop_full_needle(coord_t)
        return self._crop_proximity(coord_t)

    def _crop_proximity(self, coord_t: torch.Tensor) -> torch.Tensor:
        """Standard proximity crop: insertion-zone needle nodes + nearby tissue."""
        needle_pos = coord_t[self.needle_node_indices].numpy()
        tissue_pos = coord_t[self.tissue_node_indices].numpy()

        if len(needle_pos) == 0 or len(tissue_pos) == 0:
            return torch.arange(self.n_nodes)

        tissue_tree = cKDTree(tissue_pos)
        dist_needle, _ = tissue_tree.query(needle_pos, k=1)
        keep_needle_mask = dist_needle <= self.needle_crop_mm

        kept_needle_pos = needle_pos[keep_needle_mask]
        if len(kept_needle_pos) > 0:
            kept_needle_tree = cKDTree(kept_needle_pos)
            dist_tissue, _ = kept_needle_tree.query(tissue_pos, k=1)
            keep_tissue_mask = dist_tissue <= self.tissue_crop_mm
        else:
            keep_tissue_mask = np.zeros(len(tissue_pos), dtype=bool)

        kept = np.concatenate([
            self.needle_node_indices[keep_needle_mask],
            self.tissue_node_indices[keep_tissue_mask],
        ])
        return torch.from_numpy(np.sort(kept).astype(np.int64))

    def _crop_slice(self, coord_t: torch.Tensor) -> torch.Tensor:
        """Perpendicular-slab crop: random axial centre ± slice_half_thickness_mm."""
        needle_pos = coord_t[self.needle_node_indices].numpy()
        tissue_pos = coord_t[self.tissue_node_indices].numpy()

        if len(needle_pos) == 0 or len(tissue_pos) == 0:
            return torch.arange(self.n_nodes)

        # Principal axis of the needle via PCA
        centred = needle_pos - needle_pos.mean(axis=0)
        _, _, Vt = np.linalg.svd(centred, full_matrices=False)
        axis = Vt[0]  # (3,) unit vector along insertion axis

        needle_axial = centred @ axis
        tissue_axial = (tissue_pos - needle_pos.mean(axis=0)) @ axis

        center = np.random.uniform(needle_axial.min(), needle_axial.max())
        half = self.slice_half_thickness_mm

        keep_needle_mask = np.abs(needle_axial - center) <= half
        keep_tissue_mask = np.abs(tissue_axial - center) <= half

        kept = np.concatenate([
            self.needle_node_indices[keep_needle_mask],
            self.tissue_node_indices[keep_tissue_mask],
        ])
        if len(kept) == 0:
            return self._crop_proximity(coord_t)
        return torch.from_numpy(np.sort(kept).astype(np.int64))

    def _crop_full_needle(self, coord_t: torch.Tensor) -> torch.Tensor:
        """Full-needle crop: all needle nodes + tissue within full_needle_tissue_mm."""
        needle_pos = coord_t[self.needle_node_indices].numpy()
        tissue_pos = coord_t[self.tissue_node_indices].numpy()

        if len(needle_pos) == 0 or len(tissue_pos) == 0:
            return torch.arange(self.n_nodes)

        needle_tree = cKDTree(needle_pos)
        dist_tissue, _ = needle_tree.query(tissue_pos, k=1)
        keep_tissue_mask = dist_tissue <= self.full_needle_tissue_mm

        kept = np.concatenate([
            self.needle_node_indices,
            self.tissue_node_indices[keep_tissue_mask],
        ])
        return torch.from_numpy(np.sort(kept).astype(np.int64))

    def _build_graph(self, sample_idx: int) -> Data:
        r_idx, t_local = self._samples[sample_idx]
        run = self._run_data[r_idx]
        ft = run["frame_tensors"]
        node_props = run["node_props"]
        t1_local = t_local + 1

        coord = ft["coord"][t_local]

        # Dynamic crop from current frame's needle positions
        part_nodes = self._crop_nodes(coord)

        # Node features (normalised): dynamic frame features + static material props
        x_parts = []
        for key in self.INPUT_KEYS:
            feat = ft[key][t_local]  # (n_nodes, dim)
            if self.per_region_norm:
                feat_norm = feat.clone()
                feat_norm[self._needle_idx_t] = (
                    feat[self._needle_idx_t] - self._node_stats[f"{key}_needle_mean"]
                ) / self._node_stats[f"{key}_needle_std"]
                feat_norm[self._tissue_idx_t] = (
                    feat[self._tissue_idx_t] - self._node_stats[f"{key}_tissue_mean"]
                ) / self._node_stats[f"{key}_tissue_std"]
                x_parts.append(feat_norm)
            else:
                x_parts.append((feat - self._node_stats[f"{key}_mean"]) / self._node_stats[f"{key}_std"])
        for key in self.STATIC_PROP_KEYS:
            feat = node_props[key]
            x_parts.append((feat - self._node_stats[f"{key}_mean"]) / self._node_stats[f"{key}_std"])
        x = torch.cat(x_parts, dim=-1)

        # Target: normalised increments Δf = f_{t+1} - f_t
        y_parts = []
        for key in self.TARGET_KEYS:
            delta = ft[key][t1_local] - ft[key][t_local]  # (n_nodes, dim)
            if self.per_region_norm:
                delta_norm = delta.clone()
                delta_norm[self._needle_idx_t] = (
                    delta[self._needle_idx_t] - self._target_stats[f"{key}_needle_mean"]
                ) / self._target_stats[f"{key}_needle_std"]
                delta_norm[self._tissue_idx_t] = (
                    delta[self._tissue_idx_t] - self._target_stats[f"{key}_tissue_mean"]
                ) / self._target_stats[f"{key}_tissue_std"]
                y_parts.append(delta_norm)
            else:
                y_parts.append((delta - self._target_stats[f"{key}_mean"]) / self._target_stats[f"{key}_std"])
        y = torch.cat(y_parts, dim=-1)

        # HEX subgraph restricted to the crop
        sub_ei_hex, sub_et_hex = subgraph(
            part_nodes, self.edge_index, self.edge_type_onehot,
            relabel_nodes=True, num_nodes=self.n_nodes,
        )

        # Per-frame world edges for this run
        world_ei, world_et = run["world_edges"][t_local]

        # Filter world edges to those entirely inside the crop
        all_ei, all_et = sub_ei_hex, sub_et_hex
        if world_ei.shape[1] > 0:
            in_part = torch.zeros(self.n_nodes, dtype=torch.bool)
            in_part[part_nodes] = True
            keep = in_part[world_ei[0]] & in_part[world_ei[1]]
            if keep.any():
                local_map = torch.full((self.n_nodes,), -1, dtype=torch.long)
                local_map[part_nodes] = torch.arange(len(part_nodes))
                world_ei_local = local_map[world_ei[:, keep]]
                all_ei = torch.cat([sub_ei_hex, world_ei_local], dim=1)
                all_et = torch.cat([sub_et_hex, world_et[keep]], dim=0)

        x_sub = x[part_nodes]
        y_sub = y[part_nodes]
        coord_sub = coord[part_nodes]

        src, dst = all_ei
        rel_pos = coord_sub[src] - coord_sub[dst]
        edge_len = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)
        edge_attr = torch.cat([rel_pos, edge_len, all_et], dim=-1)

        return Data(
            x=x_sub,
            y=y_sub,
            edge_index=all_ei,
            edge_attr=edge_attr,
            pos=coord_sub,
        )

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def _compute_stats(self) -> Tuple[Dict, Dict]:
        """Compute per-feature mean/std across all training samples and runs.

        When ``per_region_norm=True`` the stats JSON also contains
        ``{key}_needle_mean``, ``{key}_needle_std``, ``{key}_tissue_mean``,
        ``{key}_tissue_std`` entries computed from needle-only and tissue-only
        node subsets respectively.  This corrects the scale mismatch for features
        like contact pressure (non-zero only on needle contact nodes) and stress
        (very different magnitudes between the stiff needle and soft tissue).
        """
        key_data = {key: [] for key in self.INPUT_KEYS}
        prop_data = {key: [] for key in self.STATIC_PROP_KEYS}
        tgt_data = {key: [] for key in self.TARGET_KEYS}

        for r_idx, run in enumerate(self._run_data):
            ft = run["frame_tensors"]
            node_props = run["node_props"]
            pair_locals = [t for r, t in self._samples if r == r_idx]
            t1_locals = [t + 1 for t in pair_locals]

            for key in self.INPUT_KEYS:
                key_data[key].append(ft[key][pair_locals])
            for key in self.STATIC_PROP_KEYS:
                prop_data[key].append(node_props[key])
            for key in self.TARGET_KEYS:
                tgt_data[key].append(ft[key][t1_locals] - ft[key][pair_locals])

        node_stats: Dict[str, torch.Tensor] = {}

        for key, dim in zip(self.INPUT_KEYS, self.INPUT_DIMS):
            # all_data: (total_pairs, n_nodes, dim)
            all_data = torch.cat(key_data[key], dim=0).float()
            flat = all_data.reshape(-1, dim)
            node_stats[f"{key}_mean"] = flat.mean(0)
            node_stats[f"{key}_std"] = flat.std(0).clamp(min=1e-8)

            if self.per_region_norm:
                flat_n = all_data[:, self._needle_idx_t, :].reshape(-1, dim)
                node_stats[f"{key}_needle_mean"] = flat_n.mean(0)
                node_stats[f"{key}_needle_std"] = flat_n.std(0).clamp(min=1e-8)
                flat_t = all_data[:, self._tissue_idx_t, :].reshape(-1, dim)
                node_stats[f"{key}_tissue_mean"] = flat_t.mean(0)
                node_stats[f"{key}_tissue_std"] = flat_t.std(0).clamp(min=1e-8)

        for key, dim in zip(self.STATIC_PROP_KEYS, self.STATIC_PROP_DIMS):
            flat = torch.cat(prop_data[key], dim=0).float()
            node_stats[f"{key}_mean"] = flat.mean(0)
            node_stats[f"{key}_std"] = flat.std(0).clamp(min=1e-8)

        target_stats: Dict[str, torch.Tensor] = {}

        for key, dim in zip(self.TARGET_KEYS, self.TARGET_DIMS):
            all_data = torch.cat(tgt_data[key], dim=0).float()
            flat = all_data.reshape(-1, dim)
            target_stats[f"{key}_mean"] = flat.mean(0)
            target_stats[f"{key}_std"] = flat.std(0).clamp(min=1e-8)

            if self.per_region_norm:
                flat_n = all_data[:, self._needle_idx_t, :].reshape(-1, dim)
                target_stats[f"{key}_needle_mean"] = flat_n.mean(0)
                target_stats[f"{key}_needle_std"] = flat_n.std(0).clamp(min=1e-8)
                flat_t = all_data[:, self._tissue_idx_t, :].reshape(-1, dim)
                target_stats[f"{key}_tissue_mean"] = flat_t.mean(0)
                target_stats[f"{key}_tissue_std"] = flat_t.std(0).clamp(min=1e-8)

        return node_stats, target_stats

    @staticmethod
    def denormalize(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Reverse normalisation."""
        return tensor * std + mean
