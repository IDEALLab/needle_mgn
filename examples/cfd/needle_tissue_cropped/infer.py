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

"""Autoregressive rollout inference for the cropped needle-tissue MeshGraphNet.

Uses the full needle mesh (no beam reduction).  Each rollout step applies the
same dynamic spatial crop used during training, keeping only needle/tissue nodes
near the active insertion zone.  State is updated only for nodes inside the crop;
nodes outside retain their previous-step values.

Output VTUs use predicted needle geometry (positions updated by ΔU) and contain:
  U_pred, V_pred, A_pred  -- accumulated model predictions
  U_gt,   V_gt,   A_gt   -- ground truth (when GT frame exists)

Usage (from examples/cfd/needle_tissue_cropped/):
    uv run python infer.py

Override config values with Hydra syntax:
    uv run python infer.py infer_start_frame=159 n_rollout=40
    uv run python infer.py needle_crop_mm=15.0 tissue_crop_mm=40.0
"""

import os
import re
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np
import hydra
import pyvista as pv
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from scipy.spatial import cKDTree
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from dataset import (
    _sorted_vtu_files,
    _get_needle_tissue_node_sets,
    _is_multi_run,
    _group_vtu_by_run,
)
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.models.meshgraphnet import MeshGraphNet
from physicsnemo.utils import load_checkpoint
from physicsnemo.datapipes.gnn.utils import load_json


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

INPUT_KEYS = ["coord", "u", "v", "a", "evf", "s", "cpress"]
STATIC_PROP_KEYS = ["mat_E", "mat_c10", "mat_density", "mat_fiber", "mat_k1", "mat_k2", "mat_kappa", "mat_nu"]
TARGET_KEYS = ["u", "v", "a"]

_WORLD_EDGE_TYPE = torch.tensor([[0.0, 0.0, 1.0]])


def _normalize(state: dict, node_props: dict, node_stats: dict) -> torch.Tensor:
    """Concatenate and normalise all input features (dynamic + static material props)."""
    parts = []
    for key in INPUT_KEYS:
        feat = state[key]
        mean = node_stats[f"{key}_mean"]
        std = node_stats[f"{key}_std"]
        parts.append((feat - mean) / std)
    for key in STATIC_PROP_KEYS:
        feat = node_props[key]
        mean = node_stats[f"{key}_mean"]
        std = node_stats[f"{key}_std"]
        parts.append((feat - mean) / std)
    return torch.cat(parts, dim=-1)


def _denorm_target(pred: torch.Tensor, target_stats: dict) -> dict:
    """Un-normalise model output (N, 9) → dict of {u, v, a} tensors."""
    out = {}
    offset = 0
    for key in TARGET_KEYS:
        mean = target_stats[f"{key}_mean"]
        std = target_stats[f"{key}_std"]
        out[key] = pred[:, offset : offset + 3] * std + mean
        offset += 3
    return out


def _build_needle_local_edge_index(
    edge_index: torch.Tensor,
    needle_node_indices: np.ndarray,
    n_nodes: int,
) -> torch.Tensor:
    """Extract needle-only edges and re-index to local needle indices.

    Returns a (2, E_needle) tensor of local indices, or (2, 0) if none found.
    """
    needle_idx_t = torch.from_numpy(needle_node_indices.astype(np.int64))
    is_needle = torch.zeros(n_nodes, dtype=torch.bool)
    is_needle[needle_idx_t] = True

    src, dst = edge_index
    mask = is_needle[src] & is_needle[dst]
    if mask.sum() == 0:
        return torch.zeros(2, 0, dtype=torch.long)

    global_to_local = torch.full((n_nodes,), -1, dtype=torch.long)
    global_to_local[needle_idx_t] = torch.arange(len(needle_node_indices), dtype=torch.long)
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
    """
    if attenuation == 0.0 or needle_local_ei.shape[1] == 0:
        return disp

    N = disp.shape[0]
    src, dst = needle_local_ei

    nbr_sum = torch.zeros_like(disp)
    nbr_sum.scatter_add_(0, dst.unsqueeze(-1).expand(-1, 3), disp[src])
    nbr_cnt = torch.zeros(N, 1, dtype=disp.dtype)
    nbr_cnt.scatter_add_(0, dst.unsqueeze(-1), torch.ones(src.shape[0], 1, dtype=disp.dtype))

    has_nbr = (nbr_cnt.squeeze(-1) > 0)
    consensus = nbr_sum / nbr_cnt.clamp(min=1.0)
    residual  = disp - consensus
    filtered  = consensus + (1.0 - attenuation) * residual

    return torch.where(has_nbr.unsqueeze(-1), filtered, disp)


def _crop_nodes(
    coord: torch.Tensor,
    needle_node_indices: np.ndarray,
    tissue_node_indices: np.ndarray,
    needle_crop_mm: float,
    tissue_crop_mm: float,
) -> torch.Tensor:
    """Compute the cropped node set from current needle positions."""
    needle_pos = coord[needle_node_indices].numpy()
    tissue_pos = coord[tissue_node_indices].numpy()

    if len(needle_pos) == 0 or len(tissue_pos) == 0:
        return torch.arange(coord.shape[0])

    tissue_tree = cKDTree(tissue_pos)
    dist_needle, _ = tissue_tree.query(needle_pos, k=1)
    keep_needle_mask = dist_needle <= needle_crop_mm

    kept_needle_pos = needle_pos[keep_needle_mask]
    if len(kept_needle_pos) > 0:
        kept_needle_tree = cKDTree(kept_needle_pos)
        dist_tissue, _ = kept_needle_tree.query(tissue_pos, k=1)
        keep_tissue_mask = dist_tissue <= tissue_crop_mm
    else:
        keep_tissue_mask = np.zeros(len(tissue_pos), dtype=bool)

    kept_needle_global = needle_node_indices[keep_needle_mask]
    kept_tissue_global = tissue_node_indices[keep_tissue_mask]
    kept = np.sort(np.concatenate([kept_needle_global, kept_tissue_global]))
    return torch.from_numpy(kept.astype(np.int64))


def _build_world_edges(
    needle_pos: np.ndarray,
    tissue_kdtree: cKDTree,
    contact_radius: float,
    needle_node_indices: np.ndarray,
    tissue_node_indices: np.ndarray,
) -> tuple:
    """Build bidirectional world edges by proximity search on full needle nodes."""
    pairs = tissue_kdtree.query_ball_point(needle_pos, contact_radius)
    src_list, dst_list = [], []
    for needle_j, tissue_neighbors in enumerate(pairs):
        needle_global = int(needle_node_indices[needle_j])
        for t_local_idx in tissue_neighbors:
            tissue_global = int(tissue_node_indices[t_local_idx])
            src_list.append(needle_global)
            dst_list.append(tissue_global)
    if not src_list:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0, 3), dtype=torch.float32),
        )
    src = torch.tensor(src_list, dtype=torch.long)
    dst = torch.tensor(dst_list, dtype=torch.long)
    world_ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    world_et = _WORLD_EDGE_TYPE.expand(world_ei.shape[1], -1)
    return world_ei, world_et


def _build_step_graph(
    state: dict,
    node_props: dict,
    part_nodes: torch.Tensor,
    hex_edge_index: torch.Tensor,
    hex_edge_type_onehot: torch.Tensor,
    world_ei: torch.Tensor,
    world_et: torch.Tensor,
    n_nodes: int,
    node_stats: dict,
) -> Data:
    """Build the normalised PyG Data object for one rollout step on the cropped subgraph."""
    sub_ei_hex, sub_et_hex = subgraph(
        part_nodes, hex_edge_index, hex_edge_type_onehot,
        relabel_nodes=True, num_nodes=n_nodes,
    )

    all_ei, all_et = sub_ei_hex, sub_et_hex
    if world_ei.shape[1] > 0:
        in_part = torch.zeros(n_nodes, dtype=torch.bool)
        in_part[part_nodes] = True
        keep = in_part[world_ei[0]] & in_part[world_ei[1]]
        if keep.any():
            local_map = torch.full((n_nodes,), -1, dtype=torch.long)
            local_map[part_nodes] = torch.arange(len(part_nodes))
            world_ei_local = local_map[world_ei[:, keep]]
            all_ei = torch.cat([sub_ei_hex, world_ei_local], dim=1)
            all_et = torch.cat([sub_et_hex, world_et[keep]], dim=0)

    coord_sub = state["coord"][part_nodes]
    x_sub = _normalize(state, node_props, node_stats)[part_nodes]

    src, dst = all_ei
    rel_pos = coord_sub[src] - coord_sub[dst]
    edge_len = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)
    edge_attr = torch.cat([rel_pos, edge_len, all_et], dim=-1)

    return Data(
        x=x_sub,
        edge_attr=edge_attr,
        edge_index=all_ei,
        pos=coord_sub,
        num_nodes=len(part_nodes),
    )


def _save_vtu_worker(
    out_path: str,
    points: np.ndarray,
    cells_flat: np.ndarray,
    celltypes: np.ndarray,
    cell_data: dict,
    point_data: dict,
) -> None:
    """Reconstruct an UnstructuredGrid from raw arrays and save. Runs in a worker process."""
    import pyvista as pv

    mesh = pv.UnstructuredGrid(cells_flat, celltypes, points)
    for k, v in cell_data.items():
        mesh.cell_data[k] = v
    for k, v in point_data.items():
        mesh.point_data[k] = v
    mesh.save(out_path)


def _write_pvd(out_dir: str, entries: list) -> str:
    """Write a Paraview .pvd time-series file."""
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

    # ---- Select VTU files for inference -------------------------------------
    # Multi-run: pick the run specified by infer_run_id (or first test run).
    # Single-run: legacy behaviour — use the global test split.
    timestep_stride = int(OmegaConf.select(cfg, "timestep_stride", default=1))
    if _is_multi_run(data_dir):
        run_files = _group_vtu_by_run(data_dir, timestep_stride)
        run_ids = list(run_files.keys())
        n_runs = len(run_ids)
        n_train_runs = max(1, int(n_runs * cfg.train_fraction))
        n_val_runs = max(1, int(n_runs * cfg.val_fraction))
        test_run_ids = run_ids[n_train_runs + n_val_runs :]
        if not test_run_ids:
            test_run_ids = run_ids[-1:]  # fallback: last run

        _infer_run_id = OmegaConf.select(cfg, "infer_run_id", default=None)
        infer_run_id = str(_infer_run_id) if _infer_run_id is not None else test_run_ids[0]
        if infer_run_id not in run_files:
            raise ValueError(
                f"infer_run_id={infer_run_id!r} not found in {data_dir}. "
                f"Available: {list(run_files.keys())}"
            )
        vtu_files = run_files[infer_run_id]
        cache_filename = f"preprocessed_cache_RUN-{infer_run_id}.pt"
        print(f"Inferring on RUN-{infer_run_id}: {len(vtu_files)} frames")
    else:
        vtu_files = _sorted_vtu_files(data_dir)
        cache_filename = "preprocessed_cache.pt"

    n_frames = len(vtu_files)
    n_pairs = n_frames - 1

    if not _is_multi_run(data_dir):
        n_train = int(n_pairs * cfg.train_fraction)
        n_val = int(n_pairs * cfg.val_fraction)
        default_start = n_train + n_val
    else:
        default_start = 0  # infer from the beginning of the selected run

    _raw_start = OmegaConf.select(cfg, "infer_start_frame", default=None)
    infer_start = default_start if (_raw_start is None) else int(_raw_start)
    n_rollout = int(OmegaConf.select(cfg, "n_rollout", default=20))
    n_steps_with_gt = n_frames - 1 - infer_start
    out_dir = to_absolute_path(
        OmegaConf.select(cfg, "infer_output_dir", default="./inference_output")
    )
    os.makedirs(out_dir, exist_ok=True)

    needle_crop_mm = float(cfg.needle_crop_mm)
    tissue_crop_mm = float(cfg.tissue_crop_mm)

    print(
        f"Rollout: start_frame={infer_start}, n_rollout={n_rollout}, "
        f"GT available for {n_steps_with_gt} steps\n"
        f"Crop: needle≤{needle_crop_mm}mm, tissue≤{tissue_crop_mm}mm"
    )

    # ---- Model -----------------------------------------------------------
    model = MeshGraphNet(
        input_dim_nodes=cfg.input_dim_nodes,
        input_dim_edges=cfg.input_dim_edges,
        output_dim=cfg.output_dim,
        processor_size=cfg.processor_size,
        hidden_dim_node_encoder=cfg.hidden_dim_node_encoder,
        hidden_dim_edge_encoder=cfg.hidden_dim_edge_encoder,
        hidden_dim_node_decoder=cfg.hidden_dim_node_decoder,
        hidden_dim_processor=cfg.hidden_dim_processor,
        aggregation=cfg.aggregation,
        use_fourier_features=cfg.get("use_fourier_features", False),
        n_fourier_features=cfg.get("n_fourier_features", 64),
        fourier_scale=cfg.get("fourier_scale", 1.0),
    )
    model = model.to(dist.device)
    load_checkpoint(to_absolute_path(cfg.ckpt_path), models=model, device=dist.device)
    model.eval()

    # ---- Stats and cache -------------------------------------------------
    node_stats = load_json(os.path.join(stats_dir, "node_stats.json"))
    target_stats = load_json(os.path.join(stats_dir, "target_stats.json"))

    raw_cache_path = os.path.join(data_dir, cache_filename)
    if not os.path.exists(raw_cache_path):
        raise FileNotFoundError(f"{raw_cache_path} not found — run train.py first.")
    cache = torch.load(raw_cache_path, weights_only=False)
    if "world_edges" not in cache:
        raise KeyError(
            f"Raw cache missing per-frame world edges. "
            f"Delete {raw_cache_path} and re-run train.py to rebuild it."
        )

    hex_edge_index = cache["edge_index"]
    hex_edge_type_onehot = cache["edge_type_onehot"]
    frame_tensors = cache["frame_tensors"]
    node_props = cache.get("node_props", {})
    n_nodes = int(hex_edge_index.max().item()) + 1

    # ---- Needle / tissue node index sets ---------------------------------
    needle_node_indices, tissue_node_indices = _get_needle_tissue_node_sets(
        hex_edge_index, hex_edge_type_onehot
    )
    print(
        f"Node sets: {len(needle_node_indices)} needle, "
        f"{len(tissue_node_indices)} tissue (total {n_nodes})"
    )

    # ---- World edge contact radius (estimated from start-frame edges) ----
    world_ei_start, _ = cache["world_edges"][infer_start]

    # Fixed KD-tree on tissue positions (Eulerian — positions don't change)
    tissue_pos_np = frame_tensors["coord"][infer_start][tissue_node_indices].numpy()
    tissue_kdtree = cKDTree(tissue_pos_np)

    if world_ei_start.shape[1] > 0:
        coords_np = frame_tensors["coord"][infer_start].numpy()
        dists = np.linalg.norm(
            coords_np[world_ei_start[0].numpy()] - coords_np[world_ei_start[1].numpy()],
            axis=1,
        )
        contact_radius = float(np.percentile(dists, 95)) * 1.2
    else:
        # No contact at start frame — fall back to a small search radius
        contact_radius = 2.0  # mm
    print(f"  World edge contact_radius = {contact_radius:.4f} mesh units")

    # ---- Consensus filter precomputation ------------------------------------
    consensus_attenuation = float(OmegaConf.select(cfg, "consensus_attenuation", default=0.0))
    needle_local_ei = _build_needle_local_edge_index(
        hex_edge_index, needle_node_indices, n_nodes
    )
    if consensus_attenuation > 0.0:
        print(
            f"Consensus filter: attenuation={consensus_attenuation:.2f}, "
            f"needle-needle edges={needle_local_ei.shape[1]}"
        )

    # ---- Initial state ---------------------------------------------------
    # cpress is included in INPUT_KEYS; it stays fixed at the start-frame value
    # during rollout since the model does not predict it.
    state = {k: frame_tensors[k][infer_start].clone().float() for k in INPUT_KEYS}

    # ---- Output mesh topology -----------------------------------------------
    # HEX cells are fixed; LINE cells (world/contact edges) update each step.
    ref_mesh_start = pv.read(vtu_files[infer_start])
    pred_points = ref_mesh_start.points.copy()  # (n_nodes, 3), updated each step

    hex_cell_idx = np.where(ref_mesh_start.celltypes == 12)[0]
    hex_submesh = ref_mesh_start.extract_cells(hex_cell_idx)
    hex_cells_flat = hex_submesh.cells.copy()
    hex_celltypes = hex_submesh.celltypes.copy()
    hex_cell_data = {k: hex_submesh.cell_data[k].copy() for k in hex_submesh.cell_data.keys()}

    # ---- Rollout ---------------------------------------------------------
    pvd_entries = []
    write_futures = []

    with ProcessPoolExecutor(max_workers=2) as write_executor, torch.no_grad():
        for step in range(n_rollout):
            # --- Full mesh (no spatial crop) ---
            part_nodes = torch.arange(n_nodes, dtype=torch.long)

            # --- World edges: use GT cache when available, else dynamic KD-tree ---
            input_frame_idx = infer_start + step
            if input_frame_idx < len(cache["world_edges"]):
                world_ei, world_et = cache["world_edges"][input_frame_idx]
            else:
                needle_pos_np = state["coord"][needle_node_indices].numpy()
                world_ei, world_et = _build_world_edges(
                    needle_pos_np,
                    tissue_kdtree,
                    contact_radius,
                    needle_node_indices,
                    tissue_node_indices,
                )

            # --- Build and run model on cropped subgraph ---
            graph = _build_step_graph(
                state, node_props, part_nodes,
                hex_edge_index, hex_edge_type_onehot,
                world_ei, world_et,
                n_nodes, node_stats,
            )
            graph = graph.to(dist.device)
            pred_sub = model(graph.x, graph.edge_attr, graph).cpu()
            next_uvw_sub = _denorm_target(pred_sub, target_stats)

            # --- Consensus filter on needle displacement ----------------------
            # part_nodes is the full mesh here, so needle_local_ei (global) maps directly.
            # Re-index needle displacement via the local_ei which is already in needle-local space.
            if consensus_attenuation > 0.0:
                needle_in_part = torch.isin(
                    torch.from_numpy(needle_node_indices), part_nodes
                )
                local_map_filter = torch.full((n_nodes,), -1, dtype=torch.long)
                local_map_filter[part_nodes] = torch.arange(len(part_nodes))
                needle_local_in_part = local_map_filter[needle_node_indices[needle_in_part.numpy()]]
                u_needle = next_uvw_sub["u"][needle_local_in_part]
                u_needle_filtered = _consensus_filter(u_needle, needle_local_ei, consensus_attenuation)
                next_uvw_sub["u"][needle_local_in_part] = u_needle_filtered

            # --- Integrate increments — only for nodes inside the crop ---
            state["u"][part_nodes] = state["u"][part_nodes] + next_uvw_sub["u"]
            state["v"][part_nodes] = state["v"][part_nodes] + next_uvw_sub["v"]
            state["a"][part_nodes] = state["a"][part_nodes] + next_uvw_sub["a"]

            # Advance Lagrangian (needle) node positions for cropped needle nodes
            needle_in_crop = part_nodes[
                torch.isin(part_nodes, torch.from_numpy(needle_node_indices))
            ]
            if len(needle_in_crop) > 0:
                # delta_u for these nodes from pred_sub, indexed via local position in part_nodes
                local_map = torch.full((n_nodes,), -1, dtype=torch.long)
                local_map[part_nodes] = torch.arange(len(part_nodes))
                local_idx = local_map[needle_in_crop]
                state["coord"][needle_in_crop] += next_uvw_sub["u"][local_idx]

            # Update predicted output point positions for all needle nodes
            pred_points[needle_node_indices] = state["coord"][needle_node_indices].numpy()

            # --- Build per-step LINE cells from current world edges ----------
            # world_ei is bidirectional; deduplicate to unique pairs (src < dst).
            we_src = world_ei[0].numpy()
            we_dst = world_ei[1].numpy()
            keep_uni = we_src < we_dst
            src_u, dst_u = we_src[keep_uni], we_dst[keep_uni]
            n_lines = len(src_u)

            if n_lines > 0:
                line_flat = np.empty(n_lines * 3, dtype=np.int64)
                line_flat[0::3] = 2
                line_flat[1::3] = src_u
                line_flat[2::3] = dst_u
                step_cells_flat = np.concatenate([hex_cells_flat, line_flat])
                step_celltypes = np.concatenate(
                    [hex_celltypes, np.full(n_lines, 3, dtype=hex_celltypes.dtype)]
                )
                step_cell_data = {}
                for k, v in hex_cell_data.items():
                    if k == "element_type":
                        line_vals = np.full(n_lines, 2, dtype=v.dtype)
                    else:
                        line_vals = np.zeros((n_lines,) + v.shape[1:], dtype=v.dtype)
                    step_cell_data[k] = np.concatenate([v, line_vals])
            else:
                step_cells_flat = hex_cells_flat
                step_celltypes = hex_celltypes
                step_cell_data = hex_cell_data

            # --- Assemble point_data and dispatch async write ---
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
            out_path = os.path.join(out_dir, fname)
            future = write_executor.submit(
                _save_vtu_worker,
                out_path,
                pred_points.copy(),
                step_cells_flat,
                step_celltypes,
                step_cell_data,
                point_data,
            )
            write_futures.append((step, fname, future))
            pvd_entries.append((infer_start + step + 1, fname))

            print(
                f"  step {step + 1:3d}/{n_rollout}  →  {fname}"
                f"  ({len(part_nodes)} nodes, full mesh)"
                + ("  (+ GT)" if has_gt else "  (extrapolating beyond GT)")
            )

    for _step, _fname, future in write_futures:
        future.result()

    pvd_path = _write_pvd(out_dir, pvd_entries)
    print(f"\nDone. Open {pvd_path} in Paraview to animate.")
    print("Tip: in Paraview use Filters → Warp By Vector on U_pred to visualise displacement.")


if __name__ == "__main__":
    main()
