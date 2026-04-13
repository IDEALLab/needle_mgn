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

"""Autoregressive rollout inference for DCEL-DoMINO on the needle-tissue dataset.

At each rollout step:
  1. Compute SDF on the bounding-box grid from current needle positions.
  2. Build the DoMINO data_dict (geometry, SDF, state).
  3. Run the model → predicted increments Δu, Δv, Δa for all nodes.
  4. Accumulate increments into running state; advance needle positions.
  5. Write the predicted VTU asynchronously.

Output VTUs contain U_pred, V_pred, A_pred and (when GT is available)
U_gt, V_gt, A_gt, Points_gt — identical layout to the MGN infer.py so
compare_deflection.py works without changes.

Usage (from examples/cfd/needle_tissue_domino/):
    uv run python infer.py
    uv run python infer.py infer_start_frame=159 n_rollout=40
"""

import os
from concurrent.futures import ProcessPoolExecutor

import hydra
import numpy as np
import pyvista as pv
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from scipy.spatial import cKDTree

from dataset import (
    NODE_STATE_DIM,
    STATE_DIMS,
    STATE_KEYS,
    STATIC_PROP_KEYS,
    STATIC_PROP_DIMS,
    _build_domino_cache,
    _compute_sdf_grid,
    _process_all_frames,
    _sorted_vtu_files,
    _is_multi_run,
    _group_vtu_by_run,
)
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.models.domino.model import DoMINO
from physicsnemo.utils import load_checkpoint
from physicsnemo.datapipes.gnn.utils import load_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_step_data_dict(
    state: dict,
    node_props: dict,
    needle_idx: np.ndarray,
    n_nodes: int,
    grid_xyz: torch.Tensor,
    grid_xyz_flat: np.ndarray,
    grid_min: torch.Tensor,
    grid_max: torch.Tensor,
    node_stats: dict,
    t_abs: int,
    n_total_frames: int,
) -> dict:
    """Construct the DoMINO data_dict from the current running state."""
    coord = state["coord"]                            # (N, 3)
    needle_pos = coord[needle_idx]                    # (N_needle, 3)

    # SDF on grid from current needle positions
    needle_pos_np = needle_pos.numpy()
    sdf_flat = _compute_sdf_grid(needle_pos_np, grid_xyz_flat)
    grid_shape = tuple(grid_xyz.shape[:3])
    sdf_grid = torch.from_numpy(sdf_flat.reshape(grid_shape))  # (Nx, Ny, Nz)

    # SDF and closest needle node at all mesh nodes
    tree = cKDTree(needle_pos_np)
    dist_nodes, nn_idx = tree.query(coord.numpy(), k=1, workers=-1)
    sdf_nodes = torch.from_numpy(dist_nodes.astype(np.float32)).unsqueeze(-1)  # (N, 1)
    pos_closest = torch.from_numpy(needle_pos_np[nn_idx].astype(np.float32))  # (N, 3)

    # Needle centroid → center-of-mass positional encoding
    centroid = needle_pos.mean(dim=0, keepdim=True).expand(n_nodes, -1)        # (N, 3)

    # Normalise positions to [-1, 1]
    rng = (grid_max - grid_min).clamp(min=1e-8)
    geo_norm = 2.0 * (needle_pos - grid_min) / rng - 1.0
    vol_norm = 2.0 * (coord - grid_min) / rng - 1.0
    cen_norm = 2.0 * (centroid - grid_min) / rng - 1.0
    grid_norm = 2.0 * (grid_xyz - grid_min) / rng - 1.0

    # Normalise SDF
    sdf_scale = sdf_nodes.max().clamp(min=1e-8)
    sdf_nodes_n = sdf_nodes / sdf_scale
    sdf_grid_n = sdf_grid / (sdf_grid.max().clamp(min=1e-8))

    # Per-node state features (dynamic + static material props)
    state_parts = []
    for key, dim in zip(STATE_KEYS, STATE_DIMS):
        feat = state.get(key, torch.zeros(n_nodes, dim))
        mean = node_stats[f"{key}_mean"]
        std = node_stats[f"{key}_std"]
        state_parts.append((feat - mean) / std)
    for key in STATIC_PROP_KEYS:
        feat = node_props.get(key, torch.zeros(n_nodes, 1))
        mean = node_stats[f"{key}_mean"]
        std = node_stats[f"{key}_std"]
        state_parts.append((feat - mean) / std)
    state_vol = torch.cat(state_parts, dim=-1)        # (N, NODE_STATE_DIM)

    def b(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(0)

    return {
        "geometry_coordinates": b(geo_norm),
        "volume_mesh_centers": b(vol_norm),
        "sdf_nodes": b(sdf_nodes_n),
        "pos_volume_closest": b(pos_closest),
        "pos_volume_center_of_mass": b(cen_norm),
        "state_vol": b(state_vol),
        "grid": b(grid_norm),
        "sdf_grid": b(sdf_grid_n),
        "surf_grid": b(grid_norm),
        "sdf_surf_grid": b(sdf_grid_n),
        "global_params_values": torch.tensor([[[float(t_abs)]]], dtype=torch.float32),
        "global_params_reference": torch.tensor(
            [[[float(n_total_frames - 1)]]], dtype=torch.float32
        ),
        "volume_min_max": torch.stack([grid_min, grid_max], dim=0).unsqueeze(0),
    }


def _denorm_target(pred: torch.Tensor, target_stats: dict) -> dict:
    """Un-normalise model output (N, 9) → {u, v, a}."""
    out = {}
    offset = 0
    for key in ("u", "v", "a"):
        mean = target_stats[f"{key}_mean"]
        std = target_stats[f"{key}_std"]
        out[key] = pred[:, offset : offset + 3] * std + mean
        offset += 3
    return out


def _build_needle_local_edge_index(
    edge_index: torch.Tensor,
    needle_idx: np.ndarray,
    n_nodes: int,
) -> torch.Tensor:
    """Extract needle-only edges and re-index to local needle indices.

    Returns a (2, E_needle) tensor of local indices into needle_idx, or an
    empty (2, 0) tensor if there are no needle-needle edges.
    """
    needle_idx_t = torch.from_numpy(needle_idx.astype(np.int64))
    is_needle = torch.zeros(n_nodes, dtype=torch.bool)
    is_needle[needle_idx_t] = True

    src, dst = edge_index
    mask = is_needle[src] & is_needle[dst]
    if mask.sum() == 0:
        return torch.zeros(2, 0, dtype=torch.long)

    global_to_local = torch.full((n_nodes,), -1, dtype=torch.long)
    global_to_local[needle_idx_t] = torch.arange(len(needle_idx), dtype=torch.long)
    return torch.stack([global_to_local[src[mask]], global_to_local[dst[mask]]])


def _consensus_filter(
    disp: torch.Tensor,
    needle_local_ei: torch.Tensor,
    attenuation: float,
) -> torch.Tensor:
    """Attenuate displacement components not correlated with neighbouring nodes.

    For each node i:
        consensus_i  = mean displacement of neighbours
        residual_i   = disp_i - consensus_i
        filtered_i   = consensus_i + (1 - attenuation) * residual_i

    Nodes with no neighbours are left unchanged.

    Parameters
    ----------
    disp : (N_needle, 3)
    needle_local_ei : (2, E)  edges in local needle indices (may be directed)
    attenuation : fraction of non-consensus displacement to remove (e.g. 0.95)
    """
    if attenuation == 0.0 or needle_local_ei.shape[1] == 0:
        return disp

    N = disp.shape[0]
    src, dst = needle_local_ei                              # both (E,)

    # Accumulate neighbour displacements into each destination node
    nbr_sum = torch.zeros_like(disp)
    nbr_sum.scatter_add_(0, dst.unsqueeze(-1).expand(-1, 3), disp[src])
    nbr_cnt = torch.zeros(N, 1, dtype=disp.dtype)
    nbr_cnt.scatter_add_(0, dst.unsqueeze(-1), torch.ones(src.shape[0], 1, dtype=disp.dtype))

    has_nbr = (nbr_cnt.squeeze(-1) > 0)
    consensus = nbr_sum / nbr_cnt.clamp(min=1.0)
    residual  = disp - consensus
    filtered  = consensus + (1.0 - attenuation) * residual

    # Fall back to original for isolated nodes
    return torch.where(has_nbr.unsqueeze(-1), filtered, disp)


def _save_vtu_worker(
    out_path: str,
    points: np.ndarray,
    cells_flat: np.ndarray,
    celltypes: np.ndarray,
    cell_data: dict,
    point_data: dict,
) -> None:
    import pyvista as pv
    mesh = pv.UnstructuredGrid(cells_flat, celltypes, points)
    for k, v in cell_data.items():
        mesh.cell_data[k] = v
    for k, v in point_data.items():
        mesh.point_data[k] = v
    mesh.save(out_path)


def _write_pvd(out_dir: str, entries: list) -> str:
    pvd_path = os.path.join(out_dir, "predicted.pvd")
    with open(pvd_path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1">\n')
        f.write("  <Collection>\n")
        for t, fname in entries:
            f.write(f'    <DataSet timestep="{t}" file="{fname}"/>\n')
        f.write("  </Collection>\n")
        f.write("</VTKFile>\n")
    return pvd_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.get("cuda_devices") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.cuda_devices)
    DistributedManager.initialize()
    dist = DistributedManager()

    data_dir = to_absolute_path(cfg.data_dir)
    stats_dir = to_absolute_path(cfg.stats_dir)

    timestep_stride = int(OmegaConf.select(cfg, "timestep_stride", default=1))
    if _is_multi_run(data_dir):
        run_files = _group_vtu_by_run(data_dir, timestep_stride)
        run_ids = list(run_files.keys())
        n_runs = len(run_ids)
        n_train_runs = max(1, int(n_runs * cfg.train_fraction))
        n_val_runs = max(1, int(n_runs * cfg.val_fraction))
        test_run_ids = run_ids[n_train_runs + n_val_runs :] or run_ids[-1:]
        _infer_run_id = OmegaConf.select(cfg, "infer_run_id", default=None)
        infer_run_id = str(_infer_run_id) if _infer_run_id is not None else test_run_ids[0]
        if infer_run_id not in run_files:
            raise ValueError(
                f"infer_run_id={infer_run_id!r} not found. Available: {list(run_files.keys())}"
            )
        vtu_files = run_files[infer_run_id]
        raw_cache_filename = f"preprocessed_cache_RUN-{infer_run_id}.pt"
        grid_res = tuple(cfg.grid_res)
        domino_cache_filename = (
            f"domino_cache_RUN-{infer_run_id}_{grid_res[0]}x{grid_res[1]}x{grid_res[2]}.pt"
        )
        default_start = 0
        print(f"Inferring on RUN-{infer_run_id}: {len(vtu_files)} frames")
    else:
        vtu_files = _sorted_vtu_files(data_dir)
        raw_cache_filename = "preprocessed_cache.pt"
        grid_res = tuple(cfg.grid_res)
        domino_cache_filename = f"domino_cache_{grid_res[0]}x{grid_res[1]}x{grid_res[2]}.pt"
        n_pairs = len(vtu_files) - 1
        n_train = int(n_pairs * cfg.train_fraction)
        n_val = int(n_pairs * cfg.val_fraction)
        default_start = n_train + n_val

    n_frames = len(vtu_files)

    infer_start = int(OmegaConf.select(cfg, "infer_start_frame", default=default_start) or default_start)
    n_rollout = int(OmegaConf.select(cfg, "n_rollout", default=20))
    out_dir = to_absolute_path(
        OmegaConf.select(cfg, "infer_output_dir", default="./inference_output")
    )
    os.makedirs(out_dir, exist_ok=True)

    print(
        f"Rollout: start_frame={infer_start}, n_rollout={n_rollout}, "
        f"GT available for {n_frames - 1 - infer_start} steps"
    )

    # ---- Model --------------------------------------------------------------
    model = DoMINO(
        input_features=3,
        output_features_vol=cfg.output_dim,
        output_features_surf=None,
        global_features=cfg.global_features,
        model_parameters=OmegaConf.to_container(cfg.model, resolve=True),
        node_state_dim_vol=NODE_STATE_DIM,
        use_fourier_features_state=cfg.get("use_fourier_features_state", False),
        n_fourier_features_state=cfg.get("n_fourier_features_state", 64),
        fourier_scale_state=cfg.get("fourier_scale_state", 1.0),
    ).to(dist.device)
    load_checkpoint(to_absolute_path(cfg.ckpt_path), models=model, device=dist.device)
    model.eval()

    # ---- Stats and caches ---------------------------------------------------
    node_stats = load_json(os.path.join(stats_dir, "domino_node_stats.json"))
    target_stats = load_json(os.path.join(stats_dir, "domino_target_stats.json"))

    raw_cache_path = os.path.join(data_dir, raw_cache_filename)
    _need_raw = not os.path.exists(raw_cache_path)
    if not _need_raw:
        raw_cache = torch.load(raw_cache_path, weights_only=False)
        _cached_n = len(raw_cache.get("frame_tensors", {}).get("coord", []))
        if _cached_n != len(vtu_files):
            print(f"Raw cache outdated (cached={_cached_n}, on-disk={len(vtu_files)}) — rebuilding ...")
            _need_raw = True
    if _need_raw:
        print(f"Building raw cache at {raw_cache_path} ...")
        raw_cache = _process_all_frames(vtu_files)
        torch.save(raw_cache, raw_cache_path)
        print(f"  → saved to {raw_cache_path}")
    frame_tensors = raw_cache["frame_tensors"]
    node_props = raw_cache.get("node_props", {})
    edge_index = raw_cache["edge_index"]
    edge_type_onehot = raw_cache["edge_type_onehot"]
    n_nodes = int(edge_index.max().item()) + 1

    domino_cache_path = os.path.join(data_dir, domino_cache_filename)
    _need_dc = not os.path.exists(domino_cache_path)
    if not _need_dc:
        dc = torch.load(domino_cache_path, weights_only=False)
        _dc_n = dc.get("frame_sdf_grid", torch.empty(0)).shape[0]
        if dc.get("grid_res") != list(grid_res) or _dc_n != len(vtu_files):
            print(f"DoMINO cache outdated (cached={_dc_n}, on-disk={len(vtu_files)}) — rebuilding ...")
            _need_dc = True
    if _need_dc:
        print(f"Building DoMINO cache at {domino_cache_path} ...")
        dc = _build_domino_cache(vtu_files, raw_cache, grid_res, domino_cache_path)

    needle_idx: np.ndarray = dc["needle_idx"].numpy()
    grid_xyz: torch.Tensor = dc["grid_xyz"]               # (Nx, Ny, Nz, 3)
    grid_xyz_flat: np.ndarray = grid_xyz.reshape(-1, 3).numpy()
    grid_min: torch.Tensor = dc["grid_min"]
    grid_max: torch.Tensor = dc["grid_max"]

    print(
        f"Node sets: {len(needle_idx)} needle, "
        f"{n_nodes - len(needle_idx)} tissue (total {n_nodes})"
    )

    # ---- Initial state -------------------------------------------------------
    # cpress uses the start-frame value and stays fixed during rollout.
    state: dict = {k: frame_tensors[k][infer_start].clone().float() for k in STATE_KEYS}
    state["coord"] = frame_tensors["coord"][infer_start].clone().float()

    # ---- Consensus filter precomputation ------------------------------------
    consensus_attenuation = float(OmegaConf.select(cfg, "consensus_attenuation", default=0.0))
    needle_local_ei = _build_needle_local_edge_index(edge_index, needle_idx, n_nodes)
    if consensus_attenuation > 0.0:
        print(
            f"Consensus filter: attenuation={consensus_attenuation:.2f}, "
            f"needle-needle edges={needle_local_ei.shape[1]}"
        )

    # ---- Output mesh topology (HEX cells only — no world edges for DoMINO) -
    ref_mesh = pv.read(vtu_files[infer_start])
    pred_points = ref_mesh.points.copy()
    hex_cell_idx = np.where(ref_mesh.celltypes == 12)[0]
    hex_submesh = ref_mesh.extract_cells(hex_cell_idx)
    topo_cells_flat = hex_submesh.cells.copy()
    topo_celltypes = hex_submesh.celltypes.copy()
    topo_cell_data = {k: hex_submesh.cell_data[k].copy() for k in hex_submesh.cell_data.keys()}

    # ---- Rollout ------------------------------------------------------------
    pvd_entries = []
    write_futures = []

    with ProcessPoolExecutor(max_workers=2) as write_executor, torch.no_grad():
        for step in range(n_rollout):
            t_abs = infer_start + step

            data_dict = _build_step_data_dict(
                state, node_props, needle_idx, n_nodes,
                grid_xyz, grid_xyz_flat,
                grid_min, grid_max,
                node_stats, t_abs, n_frames,
            )
            data_dict = {
                k: v.to(dist.device) if isinstance(v, torch.Tensor) else v
                for k, v in data_dict.items()
            }

            pred_vol, _ = model(data_dict)               # (1, N, 9)
            pred_vol = pred_vol.squeeze(0).cpu()         # (N, 9)
            next_uvw = _denorm_target(pred_vol, target_stats)

            # Consensus filter: attenuate needle displacement not shared by neighbours
            if consensus_attenuation > 0.0:
                next_uvw["u"][needle_idx] = _consensus_filter(
                    next_uvw["u"][needle_idx], needle_local_ei, consensus_attenuation
                )

            # Accumulate increments
            state["u"] = state["u"] + next_uvw["u"]
            state["v"] = state["v"] + next_uvw["v"]
            state["a"] = state["a"] + next_uvw["a"]

            # Advance Lagrangian needle positions
            state["coord"][needle_idx] += next_uvw["u"][needle_idx]
            pred_points[needle_idx] = state["coord"][needle_idx].numpy()

            # GT data for comparison
            gt_frame_idx = infer_start + step + 1
            has_gt = gt_frame_idx < len(vtu_files)
            point_data = {
                "U_pred": state["u"].numpy().copy(),
                "V_pred": state["v"].numpy().copy(),
                "A_pred": state["a"].numpy().copy(),
            }
            if has_gt:
                gt_mesh = pv.read(vtu_files[gt_frame_idx])
                point_data["U_gt"] = gt_mesh.point_data["U"].copy()
                point_data["V_gt"] = gt_mesh.point_data["V"].copy()
                point_data["A_gt"] = gt_mesh.point_data["A"].copy()
                point_data["Points_gt"] = gt_mesh.points.astype(np.float32)

            fname = f"predicted_{step:04d}.vtu"
            future = write_executor.submit(
                _save_vtu_worker,
                os.path.join(out_dir, fname),
                pred_points.copy(),
                topo_cells_flat,
                topo_celltypes,
                topo_cell_data,
                point_data,
            )
            write_futures.append(future)
            pvd_entries.append((gt_frame_idx, fname))

            print(
                f"  step {step + 1:3d}/{n_rollout}  →  {fname}"
                + ("  (+ GT)" if has_gt else "  (extrapolating beyond GT)")
            )

    for f in write_futures:
        f.result()

    pvd_path = _write_pvd(out_dir, pvd_entries)
    print(f"\nDone. Open {pvd_path} in Paraview to animate.")


if __name__ == "__main__":
    main()
