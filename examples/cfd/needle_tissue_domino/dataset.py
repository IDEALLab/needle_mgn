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
state_vol              : Normalised per-node state (u, v, a, evf, s, cpress,
                         mat_E, mat_c10, mat_density, mat_fiber, mat_k1, mat_k2,
                         mat_kappa, mat_nu)
grid / sdf_grid        : Regular bounding-box grid with needle-SDF values (cached)
surf_grid / sdf_surf_grid : Same grid reused (DoMINO requires these even in volume-only mode)

The geometry convolution sees the needle point cloud (xyz only, input_features=3).
Per-node physical state is injected separately into the volume position encoder
via the node_state_dim_vol extension added to DoMINO.

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
_is_multi_run = _cropped._is_multi_run
_group_vtu_by_run = _cropped._group_vtu_by_run
from physicsnemo.datapipes.gnn.utils import load_json


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Dynamic state features per node: u(3) v(3) a(3) evf(1) s(6) cpress(1) = 17
STATE_KEYS = ["u", "v", "a", "evf", "s", "cpress"]
STATE_DIMS = [3, 3, 3, 1, 6, 1]
# Static material properties: mat_E(1) mat_c10(1) mat_density(1) mat_fiber(3)
#                              mat_k1(1) mat_k2(1) mat_kappa(1) mat_nu(1) = 10
STATIC_PROP_KEYS = ["mat_E", "mat_c10", "mat_density", "mat_fiber", "mat_k1", "mat_k2", "mat_kappa", "mat_nu"]
STATIC_PROP_DIMS = [1, 1, 1, 3, 1, 1, 1, 1]
NODE_STATE_DIM = sum(STATE_DIMS) + sum(STATIC_PROP_DIMS)  # 17 + 10 = 27

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

    cache = {
        "needle_idx": torch.from_numpy(needle_idx),
        "tissue_idx": torch.from_numpy(tissue_idx),
        "grid_xyz": torch.from_numpy(grid_xyz),            # (Nx, Ny, Nz, 3)
        "frame_sdf_grid": torch.from_numpy(frame_sdf_grid),  # (F, Nx, Ny, Nz)
        "frame_sdf_nodes": torch.from_numpy(frame_sdf_nodes),  # (F, N)
        "frame_closest": torch.from_numpy(frame_closest),   # (F, N, 3)
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

    TARGET_KEYS = ["u", "v", "a", "evf", "s", "cpress"]
    TARGET_DIMS = [3, 3, 3, 1, 6, 1]

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        grid_res: Tuple[int, int, int] = (64, 32, 32),
        train_fraction: float = 0.8,
        val_fraction: float = 0.1,
        stats_path: str = ".",
        num_sample_nodes: Optional[int] = None,
        timestep_stride: int = 1,
    ):
        if split not in ("train", "validation", "test"):
            raise ValueError(f"Invalid split: '{split}'")

        self.split = split
        self.num_sample_nodes = num_sample_nodes
        os.makedirs(stats_path, exist_ok=True)

        # Per-run data; each entry has frame tensors, cf, sdf/closest, grid info
        self._run_data: List[Dict] = []
        # (run_local_idx, t_local) pairs
        self._samples: List[Tuple[int, int]] = []

        # Shared topology info (set from first run)
        self.needle_idx: Optional[np.ndarray] = None
        self.tissue_idx: Optional[np.ndarray] = None
        self.n_nodes: int = 0

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
                    data_dir, f"preprocessed_cache_RUN-{run_id}.pt"
                )
                if os.path.exists(raw_cache_path):
                    raw_cache = torch.load(raw_cache_path, weights_only=False)
                    cached_n = len(raw_cache.get("frame_tensors", {}).get("coord", []))
                    if (
                        "world_edges" not in raw_cache
                        or "node_props" not in raw_cache
                        or "cpress" not in raw_cache.get("frame_tensors", {})
                        or cached_n != n_frames_run
                    ):
                        print(
                            f"Cache outdated for RUN-{run_id} "
                            f"(cached={cached_n}, on-disk={n_frames_run}) — regenerating ..."
                        )
                        raw_cache = _process_all_frames(vtu_files)
                        torch.save(raw_cache, raw_cache_path)
                else:
                    print(
                        f"Building cache for RUN-{run_id} ({n_frames_run} frames)..."
                    )
                    raw_cache = _process_all_frames(vtu_files)
                    torch.save(raw_cache, raw_cache_path)

                domino_cache_path = os.path.join(
                    data_dir,
                    f"domino_cache_RUN-{run_id}_{grid_res[0]}x{grid_res[1]}x{grid_res[2]}.pt",
                )
                if os.path.exists(domino_cache_path):
                    dc = torch.load(domino_cache_path, weights_only=False)
                    dc_n = dc.get("frame_sdf_grid", torch.empty(0)).shape[0]
                    if dc.get("grid_res") != grid_res or dc_n != n_frames_run:
                        print(
                            f"DoMINO cache outdated for RUN-{run_id} "
                            f"(cached={dc_n}, on-disk={n_frames_run}) — rebuilding ..."
                        )
                        dc = _build_domino_cache(
                            vtu_files, raw_cache, grid_res, domino_cache_path
                        )
                else:
                    dc = _build_domino_cache(
                        vtu_files, raw_cache, grid_res, domino_cache_path
                    )

                if self.needle_idx is None:
                    self.needle_idx = dc["needle_idx"].numpy()
                    self.tissue_idx = dc["tissue_idx"].numpy()
                    self.n_nodes = raw_cache["frame_tensors"]["coord"].shape[1]

                self._run_data.append(
                    {
                        "frame_tensors": raw_cache["frame_tensors"],
                        "node_props": raw_cache["node_props"],
                        "frame_sdf_grid": dc["frame_sdf_grid"],
                        "frame_sdf_nodes": dc["frame_sdf_nodes"],
                        "frame_closest": dc["frame_closest"],
                        "grid_xyz": dc["grid_xyz"],
                        "grid_min": dc["grid_min"],
                        "grid_max": dc["grid_max"],
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
            if start == end:
                raise ValueError(
                    f"No pairs in '{split}' split for {n_frames} frames."
                )

            raw_cache_path = os.path.join(data_dir, _RAW_CACHE_FILE)
            if os.path.exists(raw_cache_path):
                raw_cache = torch.load(raw_cache_path, weights_only=False)
                cached_n = len(raw_cache.get("frame_tensors", {}).get("coord", []))
                if (
                    "world_edges" not in raw_cache
                    or "node_props" not in raw_cache
                    or "cpress" not in raw_cache.get("frame_tensors", {})
                    or cached_n != n_frames
                ):
                    print(
                        f"Cache outdated (cached={cached_n}, on-disk={n_frames}) — regenerating ..."
                    )
                    raw_cache = _process_all_frames(vtu_files)
                    torch.save(raw_cache, raw_cache_path)
            else:
                raw_cache = _process_all_frames(vtu_files)
                torch.save(raw_cache, raw_cache_path)

            domino_cache_path = os.path.join(
                data_dir,
                f"domino_cache_{grid_res[0]}x{grid_res[1]}x{grid_res[2]}.pt",
            )
            if os.path.exists(domino_cache_path):
                dc = torch.load(domino_cache_path, weights_only=False)
                dc_n = dc.get("frame_sdf_grid", torch.empty(0)).shape[0]
                if dc.get("grid_res") != grid_res or dc_n != n_frames:
                    print(
                        f"DoMINO cache outdated (cached={dc_n}, on-disk={n_frames}) — rebuilding ..."
                    )
                    dc = _build_domino_cache(
                        vtu_files, raw_cache, grid_res, domino_cache_path
                    )
            else:
                dc = _build_domino_cache(
                    vtu_files, raw_cache, grid_res, domino_cache_path
                )

            self.needle_idx = dc["needle_idx"].numpy()
            self.tissue_idx = dc["tissue_idx"].numpy()
            self.n_nodes = raw_cache["frame_tensors"]["coord"].shape[1]

            self._run_data.append(
                {
                    "frame_tensors": raw_cache["frame_tensors"],
                    "node_props": raw_cache["node_props"],
                    "frame_sdf_grid": dc["frame_sdf_grid"],
                    "frame_sdf_nodes": dc["frame_sdf_nodes"],
                    "frame_closest": dc["frame_closest"],
                    "frame_cf": dc["frame_cf"],
                    "grid_xyz": dc["grid_xyz"],
                    "grid_min": dc["grid_min"],
                    "grid_max": dc["grid_max"],
                }
            )
            for t in range(start, end):
                self._samples.append((0, t - start))

        self.length = len(self._samples)

        print(
            f"'{split}' split: {self.length} samples | "
            f"{len(self.needle_idx)} needle + {len(self.tissue_idx)} tissue nodes | "
            f"grid {grid_res} | {len(self._run_data)} run(s)"
        )

        # ---- Normalisation statistics ---------------------------------------
        if split == "train":
            self._node_stats, self._target_stats = self._compute_stats()
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
        r_idx, t_local = self._samples[idx]
        return self._build_data_dict(r_idx, t_local)

    def _build_data_dict(self, r_idx: int, t_local: int) -> Dict[str, torch.Tensor]:
        """Assemble the full data dict for one run/timestep pair."""
        run = self._run_data[r_idx]
        ft = run["frame_tensors"]
        node_props = run["node_props"]
        t1_local = t_local + 1

        coord = ft["coord"][t_local]                              # (N, 3)
        needle_pos = coord[self.needle_idx]                       # (N_needle, 3)
        needle_centroid = needle_pos.mean(dim=0, keepdim=True)    # (1, 3)

        vol_min = run["grid_min"]
        vol_max = run["grid_max"]
        geo_coords = 2.0 * (needle_pos - vol_min) / (vol_max - vol_min) - 1.0
        vol_centers = 2.0 * (coord - vol_min) / (vol_max - vol_min) - 1.0
        centroid_norm = 2.0 * (needle_centroid - vol_min) / (vol_max - vol_min) - 1.0

        sdf_nodes = run["frame_sdf_nodes"][t_local].unsqueeze(-1)  # (N, 1)
        pos_closest = run["frame_closest"][t_local]                # (N, 3)
        pos_com = centroid_norm.expand(self.n_nodes, -1)           # (N, 3)
        sdf_grid = run["frame_sdf_grid"][t_local]                  # (Nx, Ny, Nz)

        sdf_nodes = sdf_nodes / (sdf_nodes.max() + 1e-8)
        sdf_grid = sdf_grid / (sdf_grid.max() + 1e-8)

        # --- State features at all nodes (normalised) -----------------------
        state_parts = []
        for key in STATE_KEYS:
            feat = ft[key][t_local]
            mean = self._node_stats[f"{key}_mean"]
            std = self._node_stats[f"{key}_std"]
            state_parts.append((feat - mean) / std)
        for key in STATIC_PROP_KEYS:
            feat = node_props[key]
            mean = self._node_stats[f"{key}_mean"]
            std = self._node_stats[f"{key}_std"]
            state_parts.append((feat - mean) / std)
        state_vol = torch.cat(state_parts, dim=-1)                 # (N, NODE_STATE_DIM)

        # --- Target: normalised increments ----------------------------------
        y_parts = []
        for key in self.TARGET_KEYS:
            delta = ft[key][t1_local] - ft[key][t_local]
            mean = self._target_stats[f"{key}_mean"]
            std = self._target_stats[f"{key}_std"]
            y_parts.append((delta - mean) / std)
        y = torch.cat(y_parts, dim=-1)                            # (N, 9)

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
            return t.unsqueeze(0)

        grid_3d = run["grid_xyz"]
        grid_3d_norm = 2.0 * (grid_3d - vol_min) / (vol_max - vol_min) - 1.0
        n_frames_run = run["frame_sdf_grid"].shape[0]

        data_dict = {
            "geometry_coordinates": b(geo_coords),
            "volume_mesh_centers": b(vol_centers),
            "sdf_nodes": b(sdf_nodes),
            "pos_volume_closest": b(pos_closest),
            "pos_volume_center_of_mass": b(pos_com),
            "state_vol": b(state_vol),
            "grid": b(grid_3d_norm),
            "sdf_grid": b(sdf_grid),
            "surf_grid": b(grid_3d_norm),
            "sdf_surf_grid": b(sdf_grid),
            "global_params_values": torch.tensor(
                [[[float(t_local)]]], dtype=torch.float32
            ),
            "global_params_reference": torch.tensor(
                [[[float(n_frames_run - 1)]]], dtype=torch.float32
            ),
            "volume_min_max": torch.stack([vol_min, vol_max], dim=0).unsqueeze(0),
            "y": b(y),
        }
        return data_dict

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def _compute_stats(
        self,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Compute mean/std across all training runs and frames."""
        key_data = {key: [] for key in STATE_KEYS}
        prop_data = {key: [] for key in STATIC_PROP_KEYS}
        tgt_data = {key: [] for key in self.TARGET_KEYS}

        for r_idx, run in enumerate(self._run_data):
            ft = run["frame_tensors"]
            node_props = run["node_props"]
            pair_locals = [t for r, t in self._samples if r == r_idx]
            t1_locals = [t + 1 for t in pair_locals]

            for key in STATE_KEYS:
                key_data[key].append(ft[key][pair_locals])
            for key in STATIC_PROP_KEYS:
                prop_data[key].append(node_props[key])
            for key in self.TARGET_KEYS:
                tgt_data[key].append(ft[key][t1_locals] - ft[key][pair_locals])

        node_stats: Dict[str, torch.Tensor] = {}
        for key, dim in zip(STATE_KEYS, STATE_DIMS):
            flat = torch.cat(key_data[key], dim=0).reshape(-1, dim).float()
            node_stats[f"{key}_mean"] = flat.mean(0)
            node_stats[f"{key}_std"] = flat.std(0).clamp(min=1e-8)

        for key, dim in zip(STATIC_PROP_KEYS, STATIC_PROP_DIMS):
            flat = torch.cat(prop_data[key], dim=0).float()
            node_stats[f"{key}_mean"] = flat.mean(0)
            node_stats[f"{key}_std"] = flat.std(0).clamp(min=1e-8)

        target_stats: Dict[str, torch.Tensor] = {}
        for key, dim in zip(self.TARGET_KEYS, self.TARGET_DIMS):
            flat = torch.cat(tgt_data[key], dim=0).reshape(-1, dim).float()
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
