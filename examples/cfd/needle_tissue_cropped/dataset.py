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
        timestep_stride: int = 1,
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

        # Per-run frame data; each entry holds frame tensors, world edges, and
        # static node props for one simulation run.
        self._run_data: List[Dict] = []
        # Flat list of (run_local_idx, t_local) pairs — one entry per training sample.
        self._samples: List[Tuple[int, int]] = []

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
                    if (
                        "world_edges" not in cache
                        or "node_props" not in cache
                        or "cpress" not in cache.get("frame_tensors", {})
                    ):
                        print(f"Cache outdated for RUN-{run_id} — regenerating ...")
                        cache = _process_all_frames(vtu_files)
                        _atomic_torch_save(cache, cache_path)
                else:
                    print(
                        f"Building cache for RUN-{run_id} ({n_frames_run} frames)..."
                    )
                    cache = _process_all_frames(vtu_files)
                    _atomic_torch_save(cache, cache_path)
                    print(f"  → saved to {cache_path}")

                if self.edge_index is None:
                    self.edge_index = cache["edge_index"]
                    self.edge_type_onehot = cache["edge_type_onehot"]
                    self.n_nodes = int(self.edge_index.max().item()) + 1

                self._run_data.append(
                    {
                        "frame_tensors": {
                            key: cache["frame_tensors"][key]
                            for key in self.INPUT_KEYS
                        },
                        "world_edges": cache["world_edges"],
                        "node_props": cache["node_props"],
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

        # ---- Needle / tissue node index sets (topology-invariant) ----------
        needle_idx, tissue_idx = _get_needle_tissue_node_sets(
            self.edge_index, self.edge_type_onehot
        )
        self.needle_node_indices: np.ndarray = needle_idx
        self.tissue_node_indices: np.ndarray = tissue_idx
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

        # Target: normalised increments Δf = f_{t+1} - f_t
        y_parts = []
        for key in self.TARGET_KEYS:
            delta = ft[key][t1_local] - ft[key][t_local]
            mean = self._target_stats[f"{key}_mean"]
            std = self._target_stats[f"{key}_std"]
            y_parts.append((delta - mean) / std)
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
    def denormalize(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Reverse normalisation."""
        return tensor * std + mean
