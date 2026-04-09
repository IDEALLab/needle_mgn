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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_undirected

from physicsnemo.datapipes.gnn.utils import load_json


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


def _sorted_vtu_files(data_dir: str) -> List[str]:
    """Return VTU files in *data_dir* sorted numerically."""
    entries = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".vtu")
    ]
    numbers = []
    for path in entries:
        m = re.search(r"(\d+)", os.path.basename(path))
        numbers.append(int(m.group(1)) if m else 0)
    order = np.argsort(numbers)
    return [entries[i] for i in order]


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


def _process_all_frames(vtu_files: List[str]) -> Dict:
    """Load all VTU frames, build fixed HEX topology and per-frame world edges."""
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
        "world_edges": world_edges,
        "frame_tensors": stacked,
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

    INPUT_KEYS = ["coord", "u", "v", "a", "evf", "s", "cpress"]
    INPUT_DIMS = [3, 3, 3, 3, 1, 6, 1]
    STATIC_PROP_KEYS = _STATIC_PROP_KEYS
    STATIC_PROP_DIMS = _STATIC_PROP_DIMS
    TARGET_KEYS = ["u", "v", "a"]

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
    ):
        if split not in ("train", "validation", "test"):
            raise ValueError(f"split must be 'train', 'validation', or 'test', got '{split}'")

        self.split = split
        self.stats_path = stats_path
        self.needle_crop_mm = needle_crop_mm
        self.tissue_crop_mm = tissue_crop_mm
        self.slice_half_thickness_mm = slice_half_thickness_mm
        self.full_needle_tissue_mm = full_needle_tissue_mm
        weights = np.array(crop_strategy_weights, dtype=np.float64)
        self._crop_probs = weights / weights.sum()
        os.makedirs(stats_path, exist_ok=True)

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
        self._pair_indices = list(range(start, end))
        n_pairs_split = len(self._pair_indices)

        if n_pairs_split == 0:
            raise ValueError(f"No pairs in '{split}' split for {n_frames} frames.")

        # ---- Raw preprocessed cache ----------------------------------------
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

        self.edge_index: torch.Tensor = cache["edge_index"]
        self.edge_type_onehot: torch.Tensor = cache["edge_type_onehot"]
        self.n_nodes: int = int(self.edge_index.max().item()) + 1

        # ---- Needle / tissue node index sets (from HEX topology) -----------
        # These are fixed across all frames (topology is frame-invariant).
        needle_idx, tissue_idx = _get_needle_tissue_node_sets(
            self.edge_index, self.edge_type_onehot
        )
        self.needle_node_indices: np.ndarray = needle_idx  # (n_needle,)
        self.tissue_node_indices: np.ndarray = tissue_idx  # (n_tissue,)
        print(
            f"Node sets: {len(needle_idx)} needle, {len(tissue_idx)} tissue "
            f"(total {self.n_nodes})"
        )

        # Static material node properties (constant across frames)
        self._node_props: Dict[str, torch.Tensor] = cache["node_props"]

        # Keep only frames required for this split
        frames_needed = sorted(set(range(start, end + 1)))
        self._frame_tensors: Dict[str, torch.Tensor] = {
            key: cache["frame_tensors"][key][frames_needed] for key in self.INPUT_KEYS
        }
        self._world_edges: List[Tuple[torch.Tensor, torch.Tensor]] = [
            cache["world_edges"][i] for i in frames_needed
        ]
        self._pair_local = [p - start for p in self._pair_indices]
        self.length = n_pairs_split

        strategy_names = ("proximity", "slice", "full_needle")
        active = [f"{s}({p:.0%})" for s, p in zip(strategy_names, self._crop_probs) if p > 0]
        print(
            f"'{split}' split ready: {n_pairs_split} samples "
            f"| crops: {', '.join(active)} "
            f"| {self.edge_index.shape[1]} HEX edges total."
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
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Data:
        return self._build_graph(idx)

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

    def _build_graph(self, pair_idx: int) -> Data:
        t_local = self._pair_local[pair_idx]
        t1_local = t_local + 1

        coord = self._frame_tensors["coord"][t_local]

        # Dynamic crop from current frame's needle positions
        part_nodes = self._crop_nodes(coord)

        # Node features (normalised): dynamic frame features + static material props
        x_parts = []
        for key in self.INPUT_KEYS:
            feat = self._frame_tensors[key][t_local]
            mean = self._node_stats[f"{key}_mean"]
            std = self._node_stats[f"{key}_std"]
            x_parts.append((feat - mean) / std)
        for key in self.STATIC_PROP_KEYS:
            feat = self._node_props[key]
            mean = self._node_stats[f"{key}_mean"]
            std = self._node_stats[f"{key}_std"]
            x_parts.append((feat - mean) / std)
        x = torch.cat(x_parts, dim=-1)

        # Target: normalised increments Δf = f_{t+1} - f_t
        y_parts = []
        for key in self.TARGET_KEYS:
            delta = self._frame_tensors[key][t1_local] - self._frame_tensors[key][t_local]
            mean = self._target_stats[f"{key}_mean"]
            std = self._target_stats[f"{key}_std"]
            y_parts.append((delta - mean) / std)
        y = torch.cat(y_parts, dim=-1)

        # HEX subgraph restricted to the crop
        sub_ei_hex, sub_et_hex = subgraph(
            part_nodes, self.edge_index, self.edge_type_onehot,
            relabel_nodes=True, num_nodes=self.n_nodes,
        )

        # Per-frame world edges — already in original node index space (no remapping)
        world_ei, world_et = self._world_edges[t_local]

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
        """Compute per-feature mean/std over all training frames (all nodes)."""
        node_stats: Dict[str, torch.Tensor] = {}
        target_stats: Dict[str, torch.Tensor] = {}

        for key, dim in zip(self.INPUT_KEYS, self.INPUT_DIMS):
            data = self._frame_tensors[key][self._pair_local]
            flat = data.reshape(-1, dim)
            node_stats[f"{key}_mean"] = flat.mean(0)
            node_stats[f"{key}_std"] = flat.std(0).clamp(min=1e-8)

        # Static material properties: stats over nodes only (no time dimension)
        for key, dim in zip(self.STATIC_PROP_KEYS, self.STATIC_PROP_DIMS):
            flat = self._node_props[key].float()
            node_stats[f"{key}_mean"] = flat.mean(0)
            node_stats[f"{key}_std"] = flat.std(0).clamp(min=1e-8)

        t1_locals = [p + 1 for p in self._pair_local]
        for key in self.TARGET_KEYS:
            delta = (
                self._frame_tensors[key][t1_locals]
                - self._frame_tensors[key][self._pair_local]
            )
            flat = delta.reshape(-1, 3)
            target_stats[f"{key}_mean"] = flat.mean(0)
            target_stats[f"{key}_std"] = flat.std(0).clamp(min=1e-8)

        return node_stats, target_stats

    @staticmethod
    def denormalize(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Reverse normalisation."""
        return tensor * std + mean
