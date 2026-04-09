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

"""Dataset for DCEL-DoMINO on the needle-tissue problem.

Mapping to DoMINO concepts
--------------------------
geometry_coordinates   : Needle node positions (Lagrangian body driving dynamics)
volume_mesh_centers    : All nodes (needle + tissue) as query points
sdf_nodes              : Min distance from each node to the nearest needle node
pos_volume_closest     : Position of that nearest needle node
pos_volume_center_of_mass : Needle centroid broadcast to all nodes
state_vol              : Normalised per-node state (u, v, a, evf, s, cf, cpress,
                         mat_E, mat_c10, mat_density, mat_fiber, mat_k1, mat_k2,
                         mat_kappa, mat_nu)
grid / sdf_grid        : Regular bounding-box grid with needle-SDF values (cached)
surf_grid / sdf_surf_grid : Same grid reused (DoMINO requires these even in volume-only mode)

The geometry convolution sees the needle point cloud (xyz only, input_features=3).
Per-node physical state is injected separately into the volume position encoder
via the node_state_dim_vol extension added to DoMINO.

CFORCE defaults to zeros until VTU files include it.
CPRESS defaults to zeros until VTU files include it.
Static material properties (mat_E, mat_c10, etc.) default to zeros if absent.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

# Reuse preprocessing helpers from the cropped dataset via importlib to avoid
# a name collision (this file is also called dataset.py).
import importlib.util as _ilu

_cropped_path = os.path.join(
    os.path.dirname(__file__), "..", "needle_tissue_cropped", "dataset.py"
)
_spec = _ilu.spec_from_file_location("needle_tissue_cropped_dataset", _cropped_path)
_cropped = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cropped)

_get_needle_tissue_node_sets = _cropped._get_needle_tissue_node_sets
_process_all_frames = _cropped._process_all_frames
_sorted_vtu_files = _cropped._sorted_vtu_files
from physicsnemo.datapipes.gnn.utils import load_json


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Dynamic state features per node: u(3) v(3) a(3) evf(1) s(6) cf(3) cpress(1) = 20
STATE_KEYS = ["u", "v", "a", "evf", "s", "cf", "cpress"]
STATE_DIMS = [3, 3, 3, 1, 6, 3, 1]
# Static material properties: mat_E(1) mat_c10(1) mat_density(1) mat_fiber(3)
#                              mat_k1(1) mat_k2(1) mat_kappa(1) mat_nu(1) = 10
STATIC_PROP_KEYS = ["mat_E", "mat_c10", "mat_density", "mat_fiber", "mat_k1", "mat_k2", "mat_kappa", "mat_nu"]
STATIC_PROP_DIMS = [1, 1, 1, 3, 1, 1, 1, 1]
NODE_STATE_DIM = sum(STATE_DIMS) + sum(STATIC_PROP_DIMS)  # 20 + 10 = 30

_DOMINO_CACHE_FILE = "domino_cache.pt"
_RAW_CACHE_FILE = "preprocessed_cache.pt"


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _make_grid(
    min_xyz: np.ndarray,
    max_xyz: np.ndarray,
    grid_res: Tuple[int, int, int],
    margin: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a regular axis-aligned grid over the mesh bounding box.

    Parameters
    ----------
    min_xyz, max_xyz : (3,) arrays — bounding box corners in mesh units (mm)
    grid_res : (Nx, Ny, Nz) grid resolution
    margin : float — extra margin added on each side (mm)

    Returns
    -------
    grid_xyz : (Nx, Ny, Nz, 3) float32 — grid point coordinates
    grid_xyz_flat : (Nx*Ny*Nz, 3) float32 — flattened for KD-tree queries
    """
    lo = min_xyz - margin
    hi = max_xyz + margin
    xs = np.linspace(lo[0], hi[0], grid_res[0], dtype=np.float32)
    ys = np.linspace(lo[1], hi[1], grid_res[1], dtype=np.float32)
    zs = np.linspace(lo[2], hi[2], grid_res[2], dtype=np.float32)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_xyz = np.stack([gx, gy, gz], axis=-1)  # (Nx, Ny, Nz, 3)
    return grid_xyz, grid_xyz.reshape(-1, 3)


def _compute_sdf_grid(needle_pos: np.ndarray, grid_xyz_flat: np.ndarray) -> np.ndarray:
    """Unsigned distance from each grid point to the nearest needle node.

    Parameters
    ----------
    needle_pos : (N_needle, 3)
    grid_xyz_flat : (Ng, 3)

    Returns
    -------
    (Ng,) float32 distances
    """
    tree = cKDTree(needle_pos)
    dist, _ = tree.query(grid_xyz_flat, k=1, workers=-1)
    return dist.astype(np.float32)


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------

def _build_domino_cache(
    vtu_files: List[str],
    raw_cache: dict,
    grid_res: Tuple[int, int, int],
    cache_path: str,
) -> dict:
    """Compute and store the DCEL-DoMINO cache.

    This is the expensive one-time step.  It builds:
      * A fixed bounding-box grid (computed from frame 0 positions + margin)
      * Per-frame SDF on that grid (distance to nearest needle node)
      * Per-frame SDF at all mesh nodes
      * Per-frame closest needle-node position for each mesh node
      * Per-frame CFORCE arrays (zeros if not present in VTU)

    The grid is computed from the union of all-frame node positions so it
    covers the full insertion trajectory.
    """
    edge_index = raw_cache["edge_index"]
    edge_type_onehot = raw_cache["edge_type_onehot"]
    frame_tensors = raw_cache["frame_tensors"]
    n_frames = frame_tensors["coord"].shape[0]

    needle_idx, tissue_idx = _get_needle_tissue_node_sets(edge_index, edge_type_onehot)
    n_nodes = int(edge_index.max().item()) + 1

    print(f"Building DCEL-DoMINO cache: {n_frames} frames, grid {grid_res} ...")

    # Bounding box over all frames
    all_coords = frame_tensors["coord"].numpy()  # (F, N, 3)
    global_min = all_coords.min(axis=(0, 1))
    global_max = all_coords.max(axis=(0, 1))
    grid_xyz, grid_xyz_flat = _make_grid(global_min, global_max, grid_res)

    # Per-frame arrays
    frame_sdf_grid = np.empty((n_frames, *grid_res), dtype=np.float32)
    frame_sdf_nodes = np.empty((n_frames, n_nodes), dtype=np.float32)
    frame_closest = np.empty((n_frames, n_nodes, 3), dtype=np.float32)
    frame_cf = np.zeros((n_frames, n_nodes, 3), dtype=np.float32)

    for i in range(n_frames):
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n_frames} frames")

        needle_pos = frame_tensors["coord"][i][needle_idx].numpy()

        # SDF on grid
        sdf_flat = _compute_sdf_grid(needle_pos, grid_xyz_flat)
        frame_sdf_grid[i] = sdf_flat.reshape(grid_res)

        # SDF and closest node at all mesh nodes
        all_pos = frame_tensors["coord"][i].numpy()
        tree = cKDTree(needle_pos)
        dist, nn_idx = tree.query(all_pos, k=1, workers=-1)
        frame_sdf_nodes[i] = dist.astype(np.float32)
        frame_closest[i] = needle_pos[nn_idx].astype(np.float32)

    # CFORCE: try to load from VTU files
    print("  Loading CFORCE from VTU files (zeros if absent) ...")
    for i, path in enumerate(vtu_files):
        mesh = pv.read(path)
        if "CFORCE" in mesh.point_data:
            cf = mesh.point_data["CFORCE"].astype(np.float32)
            if cf.ndim == 1:
                cf = cf[:, None]  # scalar → (N, 1), pad to 3
                cf = np.concatenate([cf, np.zeros((cf.shape[0], 2), dtype=np.float32)], axis=1)
            frame_cf[i] = cf[:, :3]
        # else: remains zeros

    cache = {
        "needle_idx": torch.from_numpy(needle_idx),
        "tissue_idx": torch.from_numpy(tissue_idx),
        "grid_xyz": torch.from_numpy(grid_xyz),            # (Nx, Ny, Nz, 3)
        "frame_sdf_grid": torch.from_numpy(frame_sdf_grid),  # (F, Nx, Ny, Nz)
        "frame_sdf_nodes": torch.from_numpy(frame_sdf_nodes),  # (F, N)
        "frame_closest": torch.from_numpy(frame_closest),   # (F, N, 3)
        "frame_cf": torch.from_numpy(frame_cf),             # (F, N, 3)
        "grid_res": grid_res,
        "grid_min": torch.from_numpy(global_min.astype(np.float32)),
        "grid_max": torch.from_numpy(global_max.astype(np.float32)),
    }
    torch.save(cache, cache_path)
    print(f"  Cache saved → {cache_path}")
    return cache


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NeedleTissueDominoDataset(Dataset):
    """Temporal needle-tissue dataset formatted for DCEL-DoMINO.

    Uses the full mesh (no cropping).  Each sample is one temporal pair
    (frame t → frame t+1).  Returns a dict ready to pass directly to
    ``DoMINO.forward``, plus a ``"y"`` key with the normalised target increments.

    Parameters
    ----------
    data_dir : str
        Directory containing ``output_XXXX.vtu`` files and caches.
    split : str
        ``"train"``, ``"validation"``, or ``"test"``.
    grid_res : tuple of int
        Resolution of the regular bounding-box grid (Nx, Ny, Nz).
    train_fraction, val_fraction : float
    stats_path : str
        Directory for normalisation JSON files.
    num_sample_nodes : int, optional
        If set, randomly sample this many query nodes per batch (memory saving).
        ``None`` uses all nodes.
    """

    TARGET_KEYS = ["u", "v", "a"]

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        grid_res: Tuple[int, int, int] = (64, 32, 32),
        train_fraction: float = 0.8,
        val_fraction: float = 0.1,
        stats_path: str = ".",
        num_sample_nodes: Optional[int] = None,
    ):
        if split not in ("train", "validation", "test"):
            raise ValueError(f"Invalid split: '{split}'")

        self.split = split
        self.num_sample_nodes = num_sample_nodes
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
        if not self._pair_indices:
            raise ValueError(f"No pairs in '{split}' split for {n_frames} frames.")

        # ---- Raw preprocessed cache (reused from needle_tissue_cropped) -----
        raw_cache_path = os.path.join(data_dir, _RAW_CACHE_FILE)
        if os.path.exists(raw_cache_path):
            raw_cache = torch.load(raw_cache_path, weights_only=False)
            if (
                "world_edges" not in raw_cache
                or "node_props" not in raw_cache
                or "cpress" not in raw_cache.get("frame_tensors", {})
            ):
                print("Cache outdated — regenerating ...")
                raw_cache = _process_all_frames(vtu_files)
                torch.save(raw_cache, raw_cache_path)
        else:
            raw_cache = _process_all_frames(vtu_files)
            torch.save(raw_cache, raw_cache_path)

        self._frame_tensors: Dict[str, torch.Tensor] = raw_cache["frame_tensors"]
        self._node_props: Dict[str, torch.Tensor] = raw_cache["node_props"]

        # ---- DCEL-DoMINO cache (SDF grid, closest points, CFORCE) -----------
        domino_cache_path = os.path.join(
            data_dir, f"domino_cache_{grid_res[0]}x{grid_res[1]}x{grid_res[2]}.pt"
        )
        if os.path.exists(domino_cache_path):
            dc = torch.load(domino_cache_path, weights_only=False)
            if dc.get("grid_res") != grid_res:
                print("Grid resolution mismatch — rebuilding DoMINO cache ...")
                dc = _build_domino_cache(vtu_files, raw_cache, grid_res, domino_cache_path)
        else:
            dc = _build_domino_cache(vtu_files, raw_cache, grid_res, domino_cache_path)

        self.needle_idx: np.ndarray = dc["needle_idx"].numpy()
        self.tissue_idx: np.ndarray = dc["tissue_idx"].numpy()
        self.n_nodes: int = self._frame_tensors["coord"].shape[1]
        self._grid_xyz: torch.Tensor = dc["grid_xyz"]           # (Nx, Ny, Nz, 3)
        self._frame_sdf_grid: torch.Tensor = dc["frame_sdf_grid"]
        self._frame_sdf_nodes: torch.Tensor = dc["frame_sdf_nodes"]
        self._frame_closest: torch.Tensor = dc["frame_closest"]
        self._frame_cf: torch.Tensor = dc["frame_cf"]
        self._grid_min: torch.Tensor = dc["grid_min"]
        self._grid_max: torch.Tensor = dc["grid_max"]

        # Restrict to frames needed for this split
        frames_needed = sorted(set(range(start, end + 1)))
        self._pair_local = [p - start for p in self._pair_indices]
        self._frame_offset = start
        self.length = len(self._pair_indices)

        print(
            f"'{split}' split: {self.length} samples | "
            f"{len(self.needle_idx)} needle + {len(self.tissue_idx)} tissue nodes | "
            f"grid {grid_res}"
        )

        # ---- Normalisation statistics ---------------------------------------
        if split == "train":
            self._node_stats, self._target_stats = self._compute_stats(frames_needed)
            self._save_stats(stats_path)
        else:
            self._node_stats = load_json(os.path.join(stats_path, "domino_node_stats.json"))
            self._target_stats = load_json(os.path.join(stats_path, "domino_target_stats.json"))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self._pair_local[idx]          # local index into kept frame tensors
        t_abs = self._pair_indices[idx]    # absolute frame index for SDF/closest

        return self._build_data_dict(t, t_abs)

    def _build_data_dict(self, t_local: int, t_abs: int) -> Dict[str, torch.Tensor]:
        """Assemble the full data dict for one timestep pair."""
        ft = self._frame_tensors
        t1_local = t_local + 1

        coord = ft["coord"][t_local]                          # (N, 3)
        needle_pos = coord[self.needle_idx]                   # (N_needle, 3)
        needle_centroid = needle_pos.mean(dim=0, keepdim=True)  # (1, 3)

        # --- Normalise coordinates to [-1, 1] using global bounding box ------
        vol_min = self._grid_min                              # (3,)
        vol_max = self._grid_max                              # (3,)
        geo_coords = 2.0 * (needle_pos - vol_min) / (vol_max - vol_min) - 1.0
        vol_centers = 2.0 * (coord - vol_min) / (vol_max - vol_min) - 1.0
        centroid_norm = 2.0 * (needle_centroid - vol_min) / (vol_max - vol_min) - 1.0

        # --- SDF and closest ------------------------------------------------
        sdf_nodes = self._frame_sdf_nodes[t_abs].unsqueeze(-1)   # (N, 1)
        pos_closest = self._frame_closest[t_abs]                  # (N, 3)
        pos_com = centroid_norm.expand(self.n_nodes, -1)          # (N, 3)
        sdf_grid = self._frame_sdf_grid[t_abs]                    # (Nx, Ny, Nz)

        # Normalise SDF to [0, 1] using the 99th-percentile of tissue SDF
        # (so needle-adjacent tissue has sdf≈0, far tissue has sdf≈1)
        sdf_nodes = sdf_nodes / (sdf_nodes.max() + 1e-8)
        sdf_grid = sdf_grid / (sdf_grid.max() + 1e-8)

        # --- State features at all nodes (normalised) -----------------------
        state_parts = []
        cf = self._frame_cf[t_abs]                               # (N, 3)
        for key, dim in zip(STATE_KEYS, STATE_DIMS):
            if key == "cf":
                feat = cf
            else:
                feat = ft[key][t_local]
            mean = self._node_stats[f"{key}_mean"]
            std = self._node_stats[f"{key}_std"]
            state_parts.append((feat - mean) / std)
        for key in STATIC_PROP_KEYS:
            feat = self._node_props[key]
            mean = self._node_stats[f"{key}_mean"]
            std = self._node_stats[f"{key}_std"]
            state_parts.append((feat - mean) / std)
        state_vol = torch.cat(state_parts, dim=-1)               # (N, NODE_STATE_DIM)

        # --- Target: normalised increments ----------------------------------
        y_parts = []
        for key in self.TARGET_KEYS:
            delta = ft[key][t1_local] - ft[key][t_local]
            mean = self._target_stats[f"{key}_mean"]
            std = self._target_stats[f"{key}_std"]
            y_parts.append((delta - mean) / std)
        y = torch.cat(y_parts, dim=-1)                           # (N, 9)

        # --- Optional random node subsampling --------------------------------
        if self.num_sample_nodes is not None and self.num_sample_nodes < self.n_nodes:
            sample_idx = torch.randperm(self.n_nodes)[: self.num_sample_nodes]
            sdf_nodes = sdf_nodes[sample_idx]
            pos_closest = pos_closest[sample_idx]
            pos_com = pos_com[sample_idx]
            vol_centers = vol_centers[sample_idx]
            state_vol = state_vol[sample_idx]
            y = y[sample_idx]

        # --- Assemble DoMINO data_dict (batch dim = 1) ----------------------
        def b(t: torch.Tensor) -> torch.Tensor:
            """Add batch dimension."""
            return t.unsqueeze(0)

        grid_3d = self._grid_xyz                                  # (Nx, Ny, Nz, 3)
        # Normalise grid to [-1, 1]
        grid_3d_norm = 2.0 * (grid_3d - vol_min) / (vol_max - vol_min) - 1.0

        data_dict = {
            # Geometry (needle positions, Lagrangian body)
            "geometry_coordinates": b(geo_coords),               # (1, N_needle, 3)
            # Volume query points
            "volume_mesh_centers": b(vol_centers),               # (1, N_nodes, 3)
            # SDF and positional context at query nodes
            "sdf_nodes": b(sdf_nodes),                           # (1, N_nodes, 1)
            "pos_volume_closest": b(pos_closest),                # (1, N_nodes, 3)
            "pos_volume_center_of_mass": b(pos_com),             # (1, N_nodes, 3)
            # Per-node state features (DCEL extension)
            "state_vol": b(state_vol),                           # (1, N_nodes, 19)
            # Bounding box grid with SDF
            "grid": b(grid_3d_norm),                             # (1, Nx, Ny, Nz, 3)
            "sdf_grid": b(sdf_grid),                             # (1, Nx, Ny, Nz)
            # DoMINO requires surf_grid even in volume-only mode — reuse volume grid
            "surf_grid": b(grid_3d_norm),
            "sdf_surf_grid": b(sdf_grid),
            # Global conditioning: normalised timestep index
            "global_params_values": torch.tensor(
                [[[float(t_abs)]]], dtype=torch.float32
            ),
            "global_params_reference": torch.tensor(
                [[[float(self._frame_sdf_grid.shape[0] - 1)]]], dtype=torch.float32
            ),
            # Bounding box for DoMINO coordinate normalisation
            "volume_min_max": torch.stack([vol_min, vol_max], dim=0).unsqueeze(0),
            # Target
            "y": b(y),                                           # (1, N_nodes, 9)
        }
        return data_dict

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def _compute_stats(
        self, frames: List[int]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Compute mean/std over all training frames for state and target features."""
        node_stats: Dict[str, torch.Tensor] = {}
        target_stats: Dict[str, torch.Tensor] = {}

        ft = self._frame_tensors
        for key, dim in zip(STATE_KEYS, STATE_DIMS):
            if key == "cf":
                data = self._frame_cf[frames]          # (F, N, 3)
            else:
                data = ft[key][frames]
            flat = data.reshape(-1, dim).float()
            node_stats[f"{key}_mean"] = flat.mean(0)
            node_stats[f"{key}_std"] = flat.std(0).clamp(min=1e-8)

        # Static material properties: stats over nodes only (no time dimension)
        for key, dim in zip(STATIC_PROP_KEYS, STATIC_PROP_DIMS):
            flat = self._node_props[key].float()
            node_stats[f"{key}_mean"] = flat.mean(0)
            node_stats[f"{key}_std"] = flat.std(0).clamp(min=1e-8)

        pair_locals = [p - self._frame_offset for p in self._pair_indices]
        t1_locals = [p + 1 for p in pair_locals]
        for key in self.TARGET_KEYS:
            delta = ft[key][t1_locals].float() - ft[key][pair_locals].float()
            flat = delta.reshape(-1, 3)
            target_stats[f"{key}_mean"] = flat.mean(0)
            target_stats[f"{key}_std"] = flat.std(0).clamp(min=1e-8)

        return node_stats, target_stats

    def _save_stats(self, stats_path: str) -> None:
        import json

        def _to_json(d):
            return {k: v.numpy().tolist() for k, v in d.items()}

        with open(os.path.join(stats_path, "domino_node_stats.json"), "w") as f:
            json.dump(_to_json(self._node_stats), f)
        with open(os.path.join(stats_path, "domino_target_stats.json"), "w") as f:
            json.dump(_to_json(self._target_stats), f)
