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

"""Needle-tissue interaction dataset for temporal MeshGraphNet prediction."""

import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_undirected

from physicsnemo.core.version_check import check_version_spec
from physicsnemo.datapipes.gnn.utils import load_json
from torch_geometric.utils import coalesce as _pyg_coalesce

_BSMS_AVAILABLE = check_version_spec("sparse_dot_mkl", hard_fail=False)


def _atomic_torch_save(obj, path: str) -> None:
    """Write *obj* via torch.save to a temp file then atomically rename it.

    Prevents partial-write corruption when multiple MPI ranks race to create
    the same cache file on the first run.  On POSIX, os.replace is atomic;
    both ranks write identical data, so the last writer wins harmlessly.
    """
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

# HEX cell local edge connectivity (12 edges per hexahedron)
_HEX_LOCAL_EDGES: np.ndarray = np.array(
    [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical pillars
    ],
    dtype=np.int64,
)  # shape (12, 2)

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
    """Group VTU files by run ID and apply a timestep stride within each run."""
    runs: Dict[str, List[Tuple[int, str]]] = {}
    for f in os.listdir(data_dir):
        m = _MULTI_RUN_PATTERN.search(f)
        if m:
            run_id, ts = m.group(1), int(m.group(2))
            runs.setdefault(run_id, []).append((ts, os.path.join(data_dir, f)))
    result: Dict[str, List[str]] = {}
    for run_id in sorted(runs.keys(), key=int):
        frames = sorted(runs[run_id])
        paths = [p for _, p in frames[::timestep_stride]]
        if len(paths) >= 2:
            result[run_id] = paths
    return result


def _build_hex_edges(
    mesh: pv.UnstructuredGrid,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build bidirected HEX-cell edges only (fixed mesh topology).

    These edges are the same for every frame and are computed once.

    Returns
    -------
    edge_index : torch.Tensor, shape (2, n_hex_edges)
    edge_type_onehot : torch.Tensor, shape (n_hex_edges, 3)
    """
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
    """
    Extract bidirected world/contact edges (LINE cells, element_type 2) for one frame.

    World edges change each frame as the needle moves through the tissue, so
    this is called for every VTU file during preprocessing.

    Returns
    -------
    edge_index : torch.Tensor, shape (2, n_world_edges)   may be (2, 0) if no LINE cells
    edge_type_onehot : torch.Tensor, shape (n_world_edges, 3)
    """
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


def _process_all_frames(vtu_files: List[str]) -> Dict:
    """Load all VTU frames, build fixed HEX topology and per-frame world edges.

    Returns a cache dict with:
      ``edge_index`` / ``edge_type_onehot`` -- fixed HEX mesh edges (built from frame 0)
      ``world_edges`` -- list of (edge_index, edge_type_onehot) tuples, one per frame
      ``frame_tensors`` -- stacked node-feature tensors for all frames
      ``node_props`` -- static material property tensors (shape: n_nodes × dim)
    """
    print(f"Processing {len(vtu_files)} VTU frames (result will be cached)...")

    frame_tensors: Dict[str, List[np.ndarray]] = {
        k: [] for k in ("coord", "u", "v", "a", "evf", "s", "cpress")
    }
    world_edges: List[Tuple[torch.Tensor, torch.Tensor]] = []
    mesh_ref = None
    node_props: Optional[Dict[str, torch.Tensor]] = None

    for i, path in enumerate(vtu_files):
        mesh = pv.read(path)
        if mesh_ref is None:
            mesh_ref = mesh
            node_props = _load_node_props(mesh)
        n_nodes = mesh.n_points
        evf_node, s_node = _cell_features_to_nodes(mesh, n_nodes)
        frame_tensors["coord"].append(mesh.points.astype(np.float32))
        frame_tensors["u"].append(mesh.point_data["U"].astype(np.float32))
        frame_tensors["v"].append(mesh.point_data["V"].astype(np.float32))
        frame_tensors["a"].append(mesh.point_data["A"].astype(np.float32))
        frame_tensors["evf"].append(evf_node)
        frame_tensors["s"].append(s_node)
        if "CPRESS" in mesh.point_data:
            cp = mesh.point_data["CPRESS"].astype(np.float32)
            if cp.ndim == 1:
                cp = cp[:, None]
        else:
            cp = np.zeros((n_nodes, 1), dtype=np.float32)
        frame_tensors["cpress"].append(cp)
        world_edges.append(_extract_world_edges(mesh))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(vtu_files)} frames loaded")

    print("Building fixed HEX topology...")
    edge_index, edge_type_onehot = _build_hex_edges(mesh_ref)

    stacked = {
        k: torch.from_numpy(np.stack(v, axis=0)) for k, v in frame_tensors.items()
    }

    return {
        "edge_index": edge_index,
        "edge_type_onehot": edge_type_onehot,
        "world_edges": world_edges,   # List[(2,E), (E,3)] — one entry per frame
        "frame_tensors": stacked,
        "node_props": node_props,
    }


def _get_needle_tissue_node_sets(
    edge_index: torch.Tensor, edge_type_onehot: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return sorted node-index tensors for needle (et=0) and tissue (et=1) nodes."""
    et0 = edge_type_onehot[:, 0].bool()
    et1 = edge_type_onehot[:, 1].bool()
    needle_nodes = edge_index[:, et0].reshape(-1).unique().sort().values
    tissue_nodes = edge_index[:, et1].reshape(-1).unique().sort().values
    return needle_nodes, tissue_nodes


def _beam_assignment(
    needle_coords_frame0: np.ndarray, beam_spacing_mm: float
) -> Tuple[np.ndarray, int]:
    """
    Assign needle nodes to 1-D beam nodes using PCA on frame-0 positions.

    Cluster membership is computed in the Lagrangian reference configuration
    so it remains fixed as the needle bends through later frames.

    Parameters
    ----------
    needle_coords_frame0 : np.ndarray, shape (n_needle, 3)
        Needle node coordinates in the reference (frame 0) configuration.
    beam_spacing_mm : float
        Target spacing between beam nodes in the same world units as the
        coordinate data (assumed to be mm).

    Returns
    -------
    beam_assignment : np.ndarray, shape (n_needle,) int64
        Index in ``[0, N_beam)`` for each needle node.
    N_beam : int
        Number of beam nodes.
    """
    centered = needle_coords_frame0 - needle_coords_frame0.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ Vt[0]  # project onto first principal component

    proj_min, proj_max = float(proj.min()), float(proj.max())
    proj_range = proj_max - proj_min
    N_beam = max(2, int(np.ceil(proj_range / beam_spacing_mm)) + 1)

    proj_norm = (proj - proj_min) / max(proj_range, 1e-8)  # normalise to [0, 1]
    beam_asgn = np.round(proj_norm * (N_beam - 1)).astype(np.int64)
    beam_asgn = np.clip(beam_asgn, 0, N_beam - 1)
    return beam_asgn, N_beam


def _apply_beam_reduction(raw_cache: Dict, beam_spacing_mm: float) -> Dict:
    """
    Replace needle nodes with a coarse 1-D beam representation.

    Needle nodes (edge type 0) are clustered into ``N_beam`` beam nodes using
    PCA on frame-0 positions.  Cluster membership is fixed in material
    coordinates across all frames.  Per-frame beam node features are the
    cluster mean of the original needle nodes in that frame.

    Needle-needle HEX edges are discarded and replaced by a bidirectional
    chain connecting adjacent beam nodes (edge type 0).  World edges (type 2)
    are remapped so each needle endpoint is replaced by its beam node.
    Tissue nodes (edge type 1) and their edges are unchanged.

    The returned cache has the same structure as the input, so all downstream
    code (spatial partitioning, BSMS, normalisation) works without changes.

    Parameters
    ----------
    raw_cache : Dict
        Output of ``_process_all_frames``.
    beam_spacing_mm : float
        Target spacing in world units (mm) between successive beam nodes.

    Returns
    -------
    Dict
        Cache dict with the same keys as *raw_cache* but with needle nodes
        replaced by ``N_beam`` beam nodes.
    """
    edge_index_orig = raw_cache["edge_index"]
    edge_type_onehot_orig = raw_cache["edge_type_onehot"]
    frame_tensors_orig = raw_cache["frame_tensors"]

    n_frames, n_nodes_orig = frame_tensors_orig["coord"].shape[:2]

    needle_nodes_t, tissue_nodes_t = _get_needle_tissue_node_sets(
        edge_index_orig, edge_type_onehot_orig
    )
    needle_node_indices = needle_nodes_t.numpy()
    tissue_node_indices = tissue_nodes_t.numpy()
    n_needle = len(needle_node_indices)
    n_tissue = len(tissue_node_indices)

    coords_f0 = frame_tensors_orig["coord"][0].numpy()  # (n_nodes, 3)
    beam_asgn, N_beam = _beam_assignment(
        coords_f0[needle_node_indices], beam_spacing_mm
    )
    n_nodes_new = n_tissue + N_beam

    print(
        f"Beam reduction: {n_needle} needle nodes → {N_beam} beam nodes "
        f"({beam_spacing_mm:.2g} mm/node) | "
        f"total nodes: {n_nodes_orig} → {n_nodes_new}"
    )

    # Build old-index → new-index remapping
    old_to_new = np.full(n_nodes_orig, -1, dtype=np.int64)
    for new_i, old_i in enumerate(tissue_node_indices):
        old_to_new[old_i] = new_i
    for j, old_i in enumerate(needle_node_indices):
        old_to_new[old_i] = n_tissue + int(beam_asgn[j])

    # Remap tissue-tissue (et=1) and world (et=2) edges; drop needle HEX (et=0)
    ei = edge_index_orig.numpy()   # (2, E)
    et = edge_type_onehot_orig.numpy()  # (E, 3)
    keep = et[:, 0] == 0  # True where NOT type-0
    src_k = old_to_new[ei[0, keep]]
    dst_k = old_to_new[ei[1, keep]]
    et_k = et[keep]

    # Bidirectional chain edges for beam nodes (type 0 / needle)
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

    # Remove duplicate edges (world edges may fold multiple needle→beam mappings)
    edge_index_new, edge_type_onehot_new = _pyg_coalesce(
        edge_index_new, edge_type_onehot_new, n_nodes_new, reduce="min"
    )

    # Build new frame tensors
    tissue_idx_t = torch.tensor(tissue_node_indices, dtype=torch.long)
    needle_idx_t = torch.tensor(needle_node_indices, dtype=torch.long)
    ba = torch.tensor(beam_asgn, dtype=torch.long)  # (n_needle,)

    new_frame_tensors: Dict[str, torch.Tensor] = {}
    for key in ("coord", "u", "v", "a", "evf", "s", "cpress"):
        orig = frame_tensors_orig[key]  # (n_frames, n_nodes_orig, d)
        d = orig.shape[-1]

        t_feat = orig[:, tissue_idx_t, :]  # (n_frames, n_tissue, d)
        n_feat = orig[:, needle_idx_t, :]  # (n_frames, n_needle, d)

        # Scatter-mean needle features into beam nodes
        ba_exp = ba.view(1, -1, 1).expand(n_frames, n_needle, d)
        b_feat = torch.zeros(n_frames, N_beam, d, dtype=orig.dtype)
        b_feat.scatter_add_(1, ba_exp, n_feat)

        count = torch.zeros(N_beam).scatter_add_(0, ba, torch.ones(n_needle))
        b_feat /= count.view(1, N_beam, 1).clamp(min=1.0)

        new_frame_tensors[key] = torch.cat([t_feat, b_feat], dim=1)

    # Remap static material node properties (no time dimension)
    node_props_orig = raw_cache.get("node_props", {})
    new_node_props: Dict[str, torch.Tensor] = {}
    for key, prop in node_props_orig.items():
        # prop: (n_nodes_orig, d)
        d = prop.shape[-1]
        t_prop = prop[tissue_idx_t]                              # (n_tissue, d)
        n_prop = prop[needle_idx_t].float()                      # (n_needle, d)
        ba_exp_1d = ba.view(-1, 1).expand(n_needle, d)
        b_prop = torch.zeros(N_beam, d, dtype=n_prop.dtype)
        b_prop.scatter_add_(0, ba_exp_1d, n_prop)
        count_1d = torch.zeros(N_beam).scatter_add_(0, ba, torch.ones(n_needle))
        b_prop /= count_1d.view(N_beam, 1).clamp(min=1.0)
        new_node_props[key] = torch.cat([t_prop, b_prop], dim=0)

    return {
        "edge_index": edge_index_new,
        "edge_type_onehot": edge_type_onehot_new,
        "frame_tensors": new_frame_tensors,
        "node_props": new_node_props,
        # Mapping metadata needed by inference scripts to reconstruct the
        # original full mesh from beam-reduced predictions.
        "tissue_node_indices": tissue_idx_t,   # (n_tissue,) orig indices
        "needle_node_indices": needle_idx_t,   # (n_needle,) orig indices
        "beam_assignment": ba,                 # (n_needle,) → beam node index
        "n_nodes_orig": n_nodes_orig,
        "n_tissue": n_tissue,
    }


def _spatial_partitions(
    coord: torch.Tensor, num_parts: int
) -> List[torch.Tensor]:
    """
    Partition nodes into *num_parts* groups by sorting along the Z axis
    (the needle-insertion axis in RUN-2 geometry) and dividing into equal slices.

    Each returned tensor contains sorted global node indices for one partition.
    Sorting by a spatial axis keeps neighbours in the same partition, maximising
    the number of intra-partition edges retained after subgraph extraction.

    Parameters
    ----------
    coord : torch.Tensor, shape (n_nodes, 3)
        Node coordinates from the first frame.
    num_parts : int
        Number of spatial partitions.

    Returns
    -------
    List[torch.Tensor]
        List of ``num_parts`` 1-D tensors of global node indices.
    """
    z = coord[:, 2]
    sort_order = z.argsort()
    chunks = sort_order.chunk(num_parts)
    # Sort within each chunk so subgraph() can use binary search
    return [c.sort().values for c in chunks]


def _precompute_bsms(
    partitions: List[torch.Tensor],
    edge_index: torch.Tensor,
    coord_ref: torch.Tensor,
    n_nodes: int,
    num_levels: int,
) -> List[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
    """
    Compute bi-stride multi-scale graph structure for each spatial partition.

    The topology is fixed across frames, so this is computed once using the
    first frame's node positions for seed selection.

    Returns a list of (ms_edges, ms_ids) tuples — one per partition.
    ms_edges[i] is a list of edge-index tensors at each coarser scale.
    ms_ids[i] is a list of pooled node-index tensors at each coarser scale.
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

    results = []
    for i, part_nodes in enumerate(partitions):
        sub_edge_index, _ = subgraph(
            part_nodes, edge_index, relabel_nodes=True, num_nodes=n_nodes
        )
        sub_pos = coord_ref[part_nodes]
        part_data = Data(
            edge_index=sub_edge_index,
            num_nodes=part_nodes.shape[0],
        )
        part_data.pos = sub_pos

        mlg = BistrideMultiLayerGraph(part_data, num_levels)
        _, ms_edges_raw, ms_ids_raw = mlg.get_multi_layer_graphs()

        ms_edges = [torch.tensor(e, dtype=torch.long) for e in ms_edges_raw]
        ms_ids = [torch.tensor(ids, dtype=torch.long) for ids in ms_ids_raw]
        results.append((ms_edges, ms_ids))
        print(
            f"  BSMS partition {i + 1}/{len(partitions)}: "
            f"{part_nodes.shape[0]} nodes, {sub_edge_index.shape[1]} edges → "
            + ", ".join(f"L{j}={e.shape[1]}e" for j, e in enumerate(ms_edges))
        )

    return results


class NeedleTissueDataset(Dataset):
    """
    Temporal needle-tissue interaction dataset with optional spatial partitioning.

    Each sample is a (temporal pair, spatial partition) combination.
    Setting *num_parts=1* disables partitioning and returns full graphs.

    Node input features (19 total per node):
      - COORD (3): 3-D position (from mesh.points)
      - U (3): displacement
      - V (3): velocity
      - A (3): acceleration
      - EVF_VOID (1): void-fraction aggregated from incident HEX cells
      - S (6): Cauchy stress tensor aggregated from incident HEX cells

    Edge features (7 total per edge):
      - relative position COORD[src] - COORD[dst] (3)
      - edge length (1)
      - edge-type one-hot [Eulerian, Lagrangian, World] (3)

    Target (9 total per node) — increments relative to frame t:
      - ΔU = U_{t+1} - U_t (3)
      - ΔV = V_{t+1} - V_t (3)
      - ΔA = A_{t+1} - A_t (3)

    Parameters
    ----------
    data_dir : str
        Directory containing VTU files named ``output_XXXX.vtu``.
    split : str
        One of ``"train"``, ``"validation"``, or ``"test"``.
    num_parts : int
        Number of spatial partitions per temporal pair.
        Use 1 for full-graph mode (e.g. during validation with no_grad).
    train_fraction : float
        Fraction of temporal pairs reserved for training (default 0.8).
    val_fraction : float
        Fraction reserved for validation (default 0.1).
    use_bsms : bool
        If True, precompute bi-stride multi-scale graph structures for each
        spatial partition (requires ``sparse_dot_mkl``). The structures are
        cached alongside the preprocessed frame data. When enabled,
        ``__getitem__`` returns a dict ``{"graph": Data, "ms_edges": [...],
        "ms_ids": [...]}``. When False it returns a plain ``Data`` object.
    num_bsms_levels : int
        Number of coarsening levels for bi-stride (default 2).
    stats_path : str
        Directory where normalisation JSON files are read/written.
    cache_dir : str, optional
        Directory for the preprocessed cache file. Defaults to *data_dir*.
    """

    INPUT_KEYS = ["coord", "u", "v", "a", "evf", "s", "cpress"]
    INPUT_DIMS = [3, 3, 3, 3, 1, 6, 1]
    STATIC_PROP_KEYS = _STATIC_PROP_KEYS
    STATIC_PROP_DIMS = _STATIC_PROP_DIMS
    TARGET_KEYS = ["u", "v", "a"]

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        num_parts: int = 1,
        use_bsms: bool = False,
        num_bsms_levels: int = 2,
        train_fraction: float = 0.8,
        val_fraction: float = 0.1,
        stats_path: str = ".",
        cache_dir: Optional[str] = None,
        beam_spacing_mm: float = 0.0,
        timestep_stride: int = 1,
    ):
        if split not in ("train", "validation", "test"):
            raise ValueError(f"split must be 'train', 'validation', or 'test', got '{split}'")

        self.split = split
        self.stats_path = stats_path
        os.makedirs(stats_path, exist_ok=True)

        # Per-run data (frame tensors, world edges, node props)
        self._run_data: List[Dict] = []
        # (run_local_idx, t_local) — one entry per training sample
        self._samples: List[Tuple[int, int]] = []

        # The old-to-new beam remapping array (None = no beam reduction)
        self._beam_old_to_new: Optional[np.ndarray] = None

        # Topology shared across all runs (same mesh)
        self.edge_index: Optional[torch.Tensor] = None
        self.edge_type_onehot: Optional[torch.Tensor] = None
        self.n_nodes: int = 0

        # Determine beam assignment once (from first run); reused across all runs.
        _beam_asgn_cache: Optional[Tuple] = None  # (beam_asgn, N_beam, n_nodes_orig)

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

            for r_idx, run_id in enumerate(split_run_ids):
                vtu_files = run_files[run_id]
                n_frames_run = len(vtu_files)
                raw_cache_path = os.path.join(
                    cache_dir or data_dir,
                    f"preprocessed_cache_RUN-{run_id}.pt",
                )
                if os.path.exists(raw_cache_path):
                    raw_cache = torch.load(raw_cache_path, weights_only=False)
                    if (
                        "world_edges" not in raw_cache
                        or "node_props" not in raw_cache
                        or "cpress" not in raw_cache.get("frame_tensors", {})
                    ):
                        print(f"Cache outdated for RUN-{run_id} — regenerating ...")
                        raw_cache = _process_all_frames(vtu_files)
                        _atomic_torch_save(raw_cache, raw_cache_path)
                else:
                    print(
                        f"Building cache for RUN-{run_id} ({n_frames_run} frames)..."
                    )
                    raw_cache = _process_all_frames(vtu_files)
                    _atomic_torch_save(raw_cache, raw_cache_path)
                    print(f"  → saved to {raw_cache_path}")

                if beam_spacing_mm > 0.0:
                    bname = f"beam_cache_RUN-{run_id}_b{beam_spacing_mm:.2g}mm.pt"
                    beam_path = os.path.join(cache_dir or data_dir, bname)
                    if os.path.exists(beam_path):
                        graph_cache = torch.load(beam_path, weights_only=False)
                        if (
                            "tissue_node_indices" not in graph_cache
                            or "node_props" not in graph_cache
                            or "cpress" not in graph_cache.get("frame_tensors", {})
                        ):
                            print(
                                f"Beam cache outdated for RUN-{run_id} — rebuilding ..."
                            )
                            graph_cache = _apply_beam_reduction(
                                raw_cache, beam_spacing_mm
                            )
                            _atomic_torch_save(graph_cache, beam_path)
                    else:
                        graph_cache = _apply_beam_reduction(
                            raw_cache, beam_spacing_mm
                        )
                        _atomic_torch_save(graph_cache, beam_path)
                else:
                    graph_cache = raw_cache

                if self.edge_index is None:
                    self.edge_index = graph_cache["edge_index"]
                    self.edge_type_onehot = graph_cache["edge_type_onehot"]
                    self.n_nodes = int(self.edge_index.max().item()) + 1

                    if beam_spacing_mm > 0.0:
                        n_nodes_orig = int(graph_cache["n_nodes_orig"])
                        n_tissue_bc = int(graph_cache["n_tissue"])
                        tissue_idx_np = graph_cache["tissue_node_indices"].numpy()
                        needle_idx_np = graph_cache["needle_node_indices"].numpy()
                        beam_asgn_np = graph_cache["beam_assignment"].numpy()
                        old_to_new = np.full(n_nodes_orig, -1, dtype=np.int64)
                        for new_i, old_i in enumerate(tissue_idx_np):
                            old_to_new[old_i] = new_i
                        for j, old_i in enumerate(needle_idx_np):
                            old_to_new[old_i] = n_tissue_bc + int(beam_asgn_np[j])
                        self._beam_old_to_new = old_to_new

                self._run_data.append(
                    {
                        "frame_tensors": {
                            key: graph_cache["frame_tensors"][key]
                            for key in self.INPUT_KEYS
                        },
                        "world_edges": raw_cache["world_edges"],
                        "node_props": graph_cache["node_props"],
                    }
                )
                for t in range(n_frames_run - 1):
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
                raw_cache = torch.load(raw_cache_path, weights_only=False)
                if (
                    "world_edges" not in raw_cache
                    or "node_props" not in raw_cache
                    or "cpress" not in raw_cache.get("frame_tensors", {})
                ):
                    print("Cache outdated — regenerating ...")
                    raw_cache = _process_all_frames(vtu_files)
                    _atomic_torch_save(raw_cache, raw_cache_path)
                    for f in os.listdir(cache_dir or data_dir):
                        if f.startswith("beam_cache_") or f.startswith("bsms_cache_"):
                            os.remove(os.path.join(cache_dir or data_dir, f))
            else:
                raw_cache = _process_all_frames(vtu_files)
                _atomic_torch_save(raw_cache, raw_cache_path)
                print(f"Cache saved to {raw_cache_path}")

            if beam_spacing_mm > 0.0:
                bname = f"beam_cache_b{beam_spacing_mm:.2g}mm.pt"
                beam_path = os.path.join(cache_dir or data_dir, bname)
                if os.path.exists(beam_path):
                    beam_cache = torch.load(beam_path, weights_only=False)
                    if (
                        "tissue_node_indices" not in beam_cache
                        or "node_props" not in beam_cache
                        or "cpress" not in beam_cache.get("frame_tensors", {})
                    ):
                        print("Beam cache outdated — rebuilding ...")
                        beam_cache = _apply_beam_reduction(raw_cache, beam_spacing_mm)
                        _atomic_torch_save(beam_cache, beam_path)
                else:
                    print(
                        f"Building beam representation ({beam_spacing_mm:.2g} mm/node)..."
                    )
                    beam_cache = _apply_beam_reduction(raw_cache, beam_spacing_mm)
                    _atomic_torch_save(beam_cache, beam_path)
                    print(f"Beam cache saved to {beam_path}")

                n_nodes_orig = int(beam_cache["n_nodes_orig"])
                n_tissue_bc = int(beam_cache["n_tissue"])
                tissue_idx_np = beam_cache["tissue_node_indices"].numpy()
                needle_idx_np = beam_cache["needle_node_indices"].numpy()
                beam_asgn_np = beam_cache["beam_assignment"].numpy()
                old_to_new = np.full(n_nodes_orig, -1, dtype=np.int64)
                for new_i, old_i in enumerate(tissue_idx_np):
                    old_to_new[old_i] = new_i
                for j, old_i in enumerate(needle_idx_np):
                    old_to_new[old_i] = n_tissue_bc + int(beam_asgn_np[j])
                self._beam_old_to_new = old_to_new
                graph_cache = beam_cache
            else:
                graph_cache = raw_cache

            self.edge_index = graph_cache["edge_index"]
            self.edge_type_onehot = graph_cache["edge_type_onehot"]
            self.n_nodes = int(self.edge_index.max().item()) + 1

            frames_needed = list(range(start, end + 1))
            self._run_data.append(
                {
                    "frame_tensors": {
                        key: graph_cache["frame_tensors"][key][frames_needed]
                        for key in self.INPUT_KEYS
                    },
                    "world_edges": [raw_cache["world_edges"][i] for i in frames_needed],
                    "node_props": graph_cache["node_props"],
                }
            )
            for t in range(n_pairs_split):
                self._samples.append((0, t))

        # ---- Spatial partitioning (topology-based, use first run) ----------
        self._num_parts = max(1, num_parts)
        if self._num_parts > 1:
            coord_ref = self._run_data[0]["frame_tensors"]["coord"][0]
            self._partitions = _spatial_partitions(coord_ref, self._num_parts)
        else:
            self._partitions = [torch.arange(self.n_nodes)]

        self.length = len(self._samples) * self._num_parts

        # ---- Optional BSMS multi-scale structure ----------------------------
        self._use_bsms = use_bsms
        self._bsms_data: Optional[List[Tuple]] = None
        if use_bsms:
            beam_tag = f"_b{beam_spacing_mm:.2g}mm" if beam_spacing_mm > 0.0 else ""
            bsms_cache_path = os.path.join(
                cache_dir or data_dir,
                f"bsms_cache_p{self._num_parts}_l{num_bsms_levels}{beam_tag}.pt",
            )
            if os.path.exists(bsms_cache_path):
                print(f"Loading BSMS cache from {bsms_cache_path} ...")
                self._bsms_data = torch.load(bsms_cache_path, weights_only=False)
            else:
                print(
                    f"Precomputing bi-stride structures for {self._num_parts} partitions..."
                )
                coord_ref = self._run_data[0]["frame_tensors"]["coord"][0]
                self._bsms_data = _precompute_bsms(
                    self._partitions,
                    self.edge_index,
                    coord_ref,
                    self.n_nodes,
                    num_bsms_levels,
                )
                _atomic_torch_save(self._bsms_data, bsms_cache_path)
                print(f"BSMS cache saved to {bsms_cache_path}")

        print(
            f"'{split}' split ready: {len(self._samples)} pairs × {self._num_parts} parts "
            f"= {self.length} samples | {len(self._run_data)} run(s) "
            f"| {self.edge_index.shape[1]} total edges."
        )

        # ---- Normalisation statistics ----
        if split == "train":
            self._node_stats, self._target_stats = self._compute_stats()
            _atomic_save_json(self._node_stats, os.path.join(stats_path, "node_stats.json"))
            _atomic_save_json(self._target_stats, os.path.join(stats_path, "target_stats.json"))
        else:
            self._node_stats = load_json(os.path.join(stats_path, "node_stats.json"))
            self._target_stats = load_json(os.path.join(stats_path, "target_stats.json"))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        sample_idx = idx // self._num_parts
        part_idx = idx % self._num_parts
        graph = self._build_graph(sample_idx, part_idx)
        if not self._use_bsms:
            return graph
        ms_edges, ms_ids = self._bsms_data[part_idx]
        return {"graph": graph, "ms_edges": ms_edges, "ms_ids": ms_ids}

    def _build_graph(self, sample_idx: int, part_idx: int) -> Data:
        r_idx, t_local = self._samples[sample_idx]
        run = self._run_data[r_idx]
        ft = run["frame_tensors"]
        node_props = run["node_props"]
        t1_local = t_local + 1
        part_nodes = self._partitions[part_idx]

        coord = ft["coord"][t_local]

        x_parts = []
        for key in self.INPUT_KEYS:
            feat = ft[key][t_local]
            mean = self._node_stats[f"{key}_mean"]
            std = self._node_stats[f"{key}_std"]
            x_parts.append((feat - mean) / std)
        for key in self.STATIC_PROP_KEYS:
            feat = node_props[key]
            mean = self._node_stats[f"{key}_mean"]
            std = self._node_stats[f"{key}_std"]
            x_parts.append((feat - mean) / std)
        x = torch.cat(x_parts, dim=-1)

        # Target: normalised increment  Δf = f_{t+1} - f_t
        y_parts = []
        for key in self.TARGET_KEYS:
            delta = ft[key][t1_local] - ft[key][t_local]
            mean = self._target_stats[f"{key}_mean"]
            std = self._target_stats[f"{key}_std"]
            y_parts.append((delta - mean) / std)
        y = torch.cat(y_parts, dim=-1)

        # Fixed HEX subgraph for this spatial partition
        sub_ei_hex, sub_et_hex = subgraph(
            part_nodes, self.edge_index, self.edge_type_onehot,
            relabel_nodes=True, num_nodes=self.n_nodes,
        )

        # Per-frame world edges for this run (original node indices)
        world_ei, world_et = run["world_edges"][t_local]

        # Optionally remap original needle/tissue indices → beam-reduced indices
        if self._beam_old_to_new is not None and world_ei.shape[1] > 0:
            src_r = torch.from_numpy(self._beam_old_to_new[world_ei[0].numpy()])
            dst_r = torch.from_numpy(self._beam_old_to_new[world_ei[1].numpy()])
            valid = (src_r >= 0) & (dst_r >= 0)
            world_ei = torch.stack([src_r[valid], dst_r[valid]], dim=0)
            world_et = world_et[valid]

        # Filter world edges to those with both endpoints inside this partition
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
    # Normalisation helpers
    # ------------------------------------------------------------------

    def _compute_stats(
        self,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Compute per-feature mean/std across all training samples and runs."""
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
            flat = torch.cat(key_data[key], dim=0).reshape(-1, dim).float()
            node_stats[f"{key}_mean"] = flat.mean(0)
            node_stats[f"{key}_std"] = flat.std(0).clamp(min=1e-8)

        for key, dim in zip(self.STATIC_PROP_KEYS, self.STATIC_PROP_DIMS):
            flat = torch.cat(prop_data[key], dim=0).float()
            node_stats[f"{key}_mean"] = flat.mean(0)
            node_stats[f"{key}_std"] = flat.std(0).clamp(min=1e-8)

        target_stats: Dict[str, torch.Tensor] = {}
        for key in self.TARGET_KEYS:
            flat = torch.cat(tgt_data[key], dim=0).reshape(-1, 3).float()
            target_stats[f"{key}_mean"] = flat.mean(0)
            target_stats[f"{key}_std"] = flat.std(0).clamp(min=1e-8)

        return node_stats, target_stats

    @staticmethod
    def denormalize(
        tensor: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """Reverse normalisation."""
        return tensor * std + mean
