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

import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np
import hydra
import pyvista as pv
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from scipy.spatial import cKDTree


def _abspath(p: str) -> str:
    """Resolve a Hydra config path to an absolute OS path.

    On Windows, Git Bash passes paths as POSIX-style absolute paths
    (``/c/Users/...``).  Python's ``os.path.isabs`` returns ``False`` for
    these, so ``to_absolute_path`` treats them as relative and incorrectly
    prepends the Hydra working directory.  This helper converts the leading
    ``/X/`` drive prefix to ``X:/`` before resolution.
    """
    if os.name == "nt":
        p = re.sub(r"^/([A-Za-z])/", lambda m: m.group(1).upper() + ":/", p)
    return to_absolute_path(p)
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from dataset import (
    _sorted_vtu_files,
    _get_needle_tissue_node_sets,
    _is_multi_run,
    _group_vtu_by_run,
    _process_all_frames,
    _atomic_torch_save,
)
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.models.meshgraphnet import MeshGraphNet, MeshGraphKAN, FiberEquivariantMGN, FiberEquivariantKAN, TFNMeshGraphNet
from physicsnemo.utils import load_checkpoint
from physicsnemo.datapipes.gnn.utils import load_json


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

_ALL_INPUT_KEYS    = ["coord", "u", "v", "a", "evf", "s", "cpress"]
_ALL_INPUT_DIMS    = [3, 3, 3, 3, 1, 6, 1]
_ALL_TARGET_KEYS   = ["u", "v", "a", "evf", "s", "cpress"]
_ALL_TARGET_DIMS   = [3, 3, 3, 1, 6, 1]
STATIC_PROP_KEYS   = ["mat_E", "mat_c10", "mat_density", "mat_fiber", "mat_k1", "mat_k2", "mat_kappa", "mat_nu"]
_STATIC_PROP_DIMS  = [1, 1, 1, 3, 1, 1, 1, 1]

_WORLD_EDGE_TYPE = torch.tensor([[0.0, 0.0, 1.0]])


def _normalize(
    state: dict,
    node_props: dict,
    node_stats: dict,
    input_keys: list,
    needle_idx_t: Optional[torch.Tensor] = None,
    tissue_idx_t: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Concatenate and normalise all input features (dynamic + static material props).

    When ``needle_idx_t`` and ``tissue_idx_t`` are provided, per-region
    normalization is applied: needle nodes use ``{key}_needle_*`` stats and
    tissue nodes use ``{key}_tissue_*`` stats.
    """
    parts = []
    for key in input_keys:
        feat = state[key]
        if needle_idx_t is not None:
            feat_norm = feat.clone()
            feat_norm[needle_idx_t] = (
                feat[needle_idx_t] - node_stats[f"{key}_needle_mean"]
            ) / node_stats[f"{key}_needle_std"]
            feat_norm[tissue_idx_t] = (
                feat[tissue_idx_t] - node_stats[f"{key}_tissue_mean"]
            ) / node_stats[f"{key}_tissue_std"]
            parts.append(feat_norm)
        else:
            parts.append((feat - node_stats[f"{key}_mean"]) / node_stats[f"{key}_std"])
    for key in STATIC_PROP_KEYS:
        feat = node_props[key]
        parts.append((feat - node_stats[f"{key}_mean"]) / node_stats[f"{key}_std"])
    return torch.cat(parts, dim=-1)


def _denorm_target(
    pred: torch.Tensor,
    target_stats: dict,
    target_keys: list,
    target_dims: list,
    needle_mask: Optional[torch.Tensor] = None,
    tissue_mask: Optional[torch.Tensor] = None,
) -> dict:
    """Un-normalise model output → dict of predicted-state tensors.

    When ``needle_mask`` and ``tissue_mask`` (bool tensors over the crop) are
    provided, per-region denormalization is applied using ``{key}_needle_*``
    and ``{key}_tissue_*`` stats respectively.
    """
    out = {}
    offset = 0
    for key, dim in zip(target_keys, target_dims):
        chunk = pred[:, offset : offset + dim]
        if needle_mask is not None:
            result = torch.empty_like(chunk)
            result[needle_mask] = (
                chunk[needle_mask] * target_stats[f"{key}_needle_std"]
                + target_stats[f"{key}_needle_mean"]
            )
            result[tissue_mask] = (
                chunk[tissue_mask] * target_stats[f"{key}_tissue_std"]
                + target_stats[f"{key}_tissue_mean"]
            )
            out[key] = result
        else:
            out[key] = chunk * target_stats[f"{key}_std"] + target_stats[f"{key}_mean"]
        offset += dim
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


def _apply_needle_edge_cap(
    delta_u_needle: torch.Tensor,
    coord_needle: torch.Tensor,
    needle_local_ei: torch.Tensor,
    max_delta_mm: float,
) -> torch.Tensor:
    """Scale needle displacement so no edge changes length by more than max_delta_mm.

    Parameters
    ----------
    delta_u_needle : Tensor, shape (n_needle, 3)
        Predicted displacement for needle nodes in needle-local index space.
    coord_needle : Tensor, shape (n_needle, 3)
        Current needle node positions in needle-local index space.
    needle_local_ei : Tensor, shape (2, E_needle)
        Needle-to-needle edge index in needle-local space.
    max_delta_mm : float
        Maximum allowed change in edge length (mm).  From needle_edge_stats.json.

    Returns
    -------
    Tensor, shape (n_needle, 3)
        Displacement scaled so all edge length changes are within the cap.
        Returned unchanged if no edge exceeds the cap.
    """
    if needle_local_ei.shape[1] == 0:
        return delta_u_needle

    src, dst = needle_local_ei
    len_current = torch.linalg.norm(
        coord_needle[src] - coord_needle[dst], dim=-1
    )
    new_coord = coord_needle + delta_u_needle
    len_predicted = torch.linalg.norm(
        new_coord[src] - new_coord[dst], dim=-1
    )
    max_delta = (len_predicted - len_current).abs().max().item()

    if max_delta <= max_delta_mm or max_delta == 0.0:
        return delta_u_needle

    scale = max_delta_mm / max_delta
    return delta_u_needle * scale


def _split_tfn_features(
    x: torch.Tensor, use_cpress: bool
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Split flat node features into (x_scalar, x_vec) for TFNMeshGraphNet.

    Mirrors ``NeedleTissueDataset._split_tfn_features``.  Coordinates are
    excluded for translational equivariance.
    """
    if use_cpress:
        x_vec = torch.cat([x[:, 3:12], x[:, 23:26]], dim=-1)
        x_scalar = torch.cat([x[:, 12:23], x[:, 26:30]], dim=-1)
    else:
        x_vec = torch.cat([x[:, 3:12], x[:, 22:25]], dim=-1)
        x_scalar = torch.cat([x[:, 12:22], x[:, 25:29]], dim=-1)
    return x_scalar, x_vec


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
    input_keys: list,
    needle_idx_t: Optional[torch.Tensor] = None,
    tissue_idx_t: Optional[torch.Tensor] = None,
    fiber_dir_full: Optional[torch.Tensor] = None,
    use_cpress: bool = True,
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
    x_sub = _normalize(
        state, node_props, node_stats, input_keys, needle_idx_t, tissue_idx_t
    )[part_nodes]

    src, dst = all_ei
    rel_pos = coord_sub[src] - coord_sub[dst]
    edge_len = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)
    edge_attr = torch.cat([rel_pos, edge_len, all_et], dim=-1)

    # Unit fiber direction per node (passed through to FiberEquivariantMGN).
    fiber_dir_sub = None
    if fiber_dir_full is not None:
        fiber_dir_sub = fiber_dir_full[part_nodes]

    # Split into scalar/vector parts for TFNMeshGraphNet.
    x_scalar_sub, x_vec_sub = _split_tfn_features(x_sub, use_cpress)

    return Data(
        x=x_sub,
        edge_attr=edge_attr,
        edge_index=all_ei,
        pos=coord_sub,
        num_nodes=len(part_nodes),
        fiber_dir=fiber_dir_sub,
        x_scalar=x_scalar_sub,
        x_vec=x_vec_sub,
    )


def _save_vtu_worker(
    out_path: str,
    points: np.ndarray,
    cells_flat: np.ndarray,
    celltypes: np.ndarray,
    cell_data: dict,
    point_data: dict,
) -> None:
    """Reconstruct an UnstructuredGrid from raw arrays and save. Runs in a worker process.

    Uses vtkXMLUnstructuredGridWriter directly with SetWriteArrayMetaData(False)
    to suppress L2_NORM_RANGE information-key entries that newer VTK writes but
    older ParaView versions cannot parse.
    """
    import pyvista as pv
    import vtk

    mesh = pv.UnstructuredGrid(cells_flat, celltypes, points)
    for k, v in cell_data.items():
        mesh.cell_data[k] = v
    for k, v in point_data.items():
        mesh.point_data[k] = v

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(out_path)
    writer.SetInputData(mesh)
    if hasattr(writer, "SetWriteArrayMetaData"):
        writer.SetWriteArrayMetaData(False)
    writer.Write()


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

    data_dir = _abspath(cfg.data_dir)
    stats_dir = _abspath(cfg.stats_dir)

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
    out_dir = _abspath(
        OmegaConf.select(cfg, "infer_output_dir", default="./inference_output")
    )
    os.makedirs(out_dir, exist_ok=True)

    needle_crop_mm = float(cfg.needle_crop_mm)
    tissue_crop_mm = float(cfg.tissue_crop_mm)

    # ---- Feature lists (controlled by use_cpress) ------------------------
    use_cpress = bool(cfg.get("use_cpress", True))
    if use_cpress:
        input_keys  = _ALL_INPUT_KEYS
        input_dims  = _ALL_INPUT_DIMS
        target_keys = _ALL_TARGET_KEYS
        target_dims = _ALL_TARGET_DIMS
    else:
        input_keys  = [k for k in _ALL_INPUT_KEYS  if k != "cpress"]
        input_dims  = [d for k, d in zip(_ALL_INPUT_KEYS, _ALL_INPUT_DIMS)  if k != "cpress"]
        target_keys = [k for k in _ALL_TARGET_KEYS if k != "cpress"]
        target_dims = [d for k, d in zip(_ALL_TARGET_KEYS, _ALL_TARGET_DIMS) if k != "cpress"]

    input_dim_nodes = sum(input_dims) + sum(_STATIC_PROP_DIMS)
    output_dim      = sum(target_dims)

    print(
        f"Rollout: start_frame={infer_start}, n_rollout={n_rollout}, "
        f"GT available for {n_steps_with_gt} steps\n"
        f"Crop: needle≤{needle_crop_mm}mm, tissue≤{tissue_crop_mm}mm\n"
        f"use_cpress={use_cpress}: input_dim_nodes={input_dim_nodes}, output_dim={output_dim}"
    )

    # ---- Model -----------------------------------------------------------
    model_type = str(OmegaConf.select(cfg, "model_type", default="mgn")).lower()
    _shared_kwargs = dict(
        input_dim_nodes=input_dim_nodes,
        input_dim_edges=cfg.input_dim_edges,
        output_dim=output_dim,
        processor_size=cfg.processor_size,
        hidden_dim_node_encoder=cfg.hidden_dim_node_encoder,
        hidden_dim_edge_encoder=cfg.hidden_dim_edge_encoder,
        hidden_dim_node_decoder=cfg.hidden_dim_node_decoder,
        hidden_dim_processor=cfg.hidden_dim_processor,
        aggregation=cfg.aggregation,
    )
    if model_type == "kan":
        model = MeshGraphKAN(
            **_shared_kwargs,
            num_harmonics=int(OmegaConf.select(cfg, "num_harmonics", default=5)),
        )
    elif model_type == "fiber":
        model = FiberEquivariantMGN(
            **_shared_kwargs,
            n_vec_outputs=int(OmegaConf.select(cfg, "n_vec_outputs", default=3)),
        )
    elif model_type == "fiber_kan":
        model = FiberEquivariantKAN(
            **_shared_kwargs,
            n_vec_outputs=int(OmegaConf.select(cfg, "n_vec_outputs", default=3)),
            num_harmonics=int(OmegaConf.select(cfg, "num_harmonics", default=5)),
        )
    elif model_type == "tfn":
        n_tfn_scalar = 15 if use_cpress else 14
        model = TFNMeshGraphNet(
            n_node_scalar=n_tfn_scalar,
            n_node_vec=4,
            output_dim=output_dim,
            irreps_hidden=str(OmegaConf.select(cfg, "irreps_hidden", default="16x0e + 8x1o + 4x2e")),
            l_max=int(OmegaConf.select(cfg, "l_max", default=2)),
            n_radial_basis=int(OmegaConf.select(cfg, "n_radial_basis", default=8)),
            r_max=float(OmegaConf.select(cfg, "r_max", default=60.0)),
            processor_size=cfg.processor_size,
            n_vec_outputs=int(OmegaConf.select(cfg, "n_vec_outputs", default=3)),
            checkpoint_layers=False,  # not needed at inference; model.eval() disables it anyway
        )
    else:
        model = MeshGraphNet(
            **_shared_kwargs,
            use_fourier_features=cfg.get("use_fourier_features", False),
            n_fourier_features=cfg.get("n_fourier_features", 64),
            fourier_scale=cfg.get("fourier_scale", 1.0),
        )
    model = model.to(dist.device)
    load_checkpoint(_abspath(cfg.ckpt_path), models=model, device=dist.device)
    model.eval()

    # ---- Stats and cache -------------------------------------------------
    node_stats = load_json(os.path.join(stats_dir, "node_stats.json"))
    target_stats = load_json(os.path.join(stats_dir, "target_stats.json"))

    # Per-region normalization: detected automatically from the stats file.
    # When enabled, needle and tissue nodes use separate mean/std for each feature.
    per_region_norm = f"u_needle_mean" in node_stats

    raw_cache_path = os.path.join(data_dir, cache_filename)
    _need_rebuild = not os.path.exists(raw_cache_path)
    if not _need_rebuild:
        _existing = torch.load(raw_cache_path, weights_only=False)
        _cached_n = len(_existing.get("frame_tensors", {}).get("coord", []))
        if (
            "world_edges" not in _existing
            or _cached_n != len(vtu_files)
        ):
            print(
                f"Cache outdated (cached={_cached_n}, on-disk={len(vtu_files)}) — rebuilding ..."
            )
            _need_rebuild = True
        else:
            cache = _existing
    if _need_rebuild:
        print(f"Building cache at {raw_cache_path} ...")
        cache = _process_all_frames(vtu_files)
        _atomic_torch_save(cache, raw_cache_path)
        print(f"  → saved to {raw_cache_path}")

    hex_edge_index = cache["edge_index"]
    hex_edge_type_onehot = cache["edge_type_onehot"]
    frame_tensors = cache["frame_tensors"]
    node_props = cache.get("node_props", {})
    n_nodes = int(hex_edge_index.max().item()) + 1

    # Pre-compute unit fiber direction for every node (used by FiberEquivariantMGN).
    if "mat_fiber" in node_props:
        _fiber_raw = node_props["mat_fiber"].float()   # (N, 3)
        _fiber_norm = torch.linalg.norm(_fiber_raw, dim=-1, keepdim=True).clamp(min=1e-8)
        fiber_dir_full = _fiber_raw / _fiber_norm       # (N, 3) unit vectors
    else:
        fiber_dir_full = None

    # ---- Needle / tissue node index sets ---------------------------------
    needle_node_indices, tissue_node_indices = _get_needle_tissue_node_sets(
        hex_edge_index, hex_edge_type_onehot
    )
    print(
        f"Node sets: {len(needle_node_indices)} needle, "
        f"{len(tissue_node_indices)} tissue (total {n_nodes})"
    )
    if per_region_norm:
        print("Per-region normalization: enabled (detected from stats file)")
    needle_idx_t = torch.from_numpy(needle_node_indices.astype(np.int64)) if per_region_norm else None
    tissue_idx_t = torch.from_numpy(tissue_node_indices.astype(np.int64)) if per_region_norm else None

    # Fixed KD-tree on tissue positions (Eulerian — positions don't change)
    tissue_pos_np = frame_tensors["coord"][infer_start][tissue_node_indices].numpy()
    tissue_kdtree = cKDTree(tissue_pos_np)

    # ---- World edge contact radius (estimated from all GT frames with edges) --
    # Scanning all cached frames gives a more robust estimate than the start
    # frame alone: the start frame may have few or no contacts if the needle
    # has just entered tissue, causing the radius to be under-estimated for
    # later frames where the tip is deeper.
    all_edge_dists = []
    _coords_all = frame_tensors["coord"].numpy()  # (T, N, 3)
    for _t, (_ei, _) in enumerate(cache["world_edges"]):
        if _ei.shape[1] == 0:
            continue
        _c = _coords_all[_t]
        all_edge_dists.append(
            np.linalg.norm(_c[_ei[0].numpy()] - _c[_ei[1].numpy()], axis=1)
        )

    if all_edge_dists:
        contact_radius = float(np.percentile(np.concatenate(all_edge_dists), 95)) * 1.2
    else:
        contact_radius = 2.0  # mm — no GT contact edges found anywhere
    n_frames_with_edges = len(all_edge_dists)
    print(
        f"  World edge contact_radius = {contact_radius:.4f} mesh units "
        f"(from {n_frames_with_edges}/{len(cache['world_edges'])} GT frames with edges)"
    )

    # ---- Consensus filter + needle edge cap precomputation ------------------
    consensus_attenuation = float(OmegaConf.select(cfg, "consensus_attenuation", default=0.0))
    needle_local_ei = _build_needle_local_edge_index(
        hex_edge_index, needle_node_indices, n_nodes
    )
    if consensus_attenuation > 0.0:
        print(
            f"Consensus filter: attenuation={consensus_attenuation:.2f}, "
            f"needle-needle edges={needle_local_ei.shape[1]}"
        )

    needle_edge_cap_mm: Optional[float] = None
    if bool(OmegaConf.select(cfg, "needle_edge_cap", default=False)):
        stats_file = os.path.join(stats_dir, "needle_edge_stats.json")
        if not os.path.isfile(stats_file):
            raise FileNotFoundError(
                f"needle_edge_cap=true but {stats_file} not found. "
                f"Run compute_needle_edge_stats.py first."
            )
        with open(stats_file) as f:
            _edge_stats = json.load(f)
        needle_edge_cap_mm = float(_edge_stats["max_needle_edge_delta_mm"])
        print(
            f"Needle edge cap: {needle_edge_cap_mm:.6f} mm  "
            f"(from {stats_file})"
        )

    # ---- Initial state ---------------------------------------------------
    state = {k: frame_tensors[k][infer_start].clone().float() for k in input_keys}

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
    t0_rollout = time.time()

    with ProcessPoolExecutor(max_workers=2) as write_executor, torch.no_grad():
        for step in range(n_rollout):
            t_step = time.time()

            # --- Full mesh (no spatial crop) ---
            part_nodes = torch.arange(n_nodes, dtype=torch.long)

            # --- World edges: always build dynamically from predicted positions ---
            # GT-cached edges are NOT used here for two reasons:
            # (a) predicted needle positions diverge from GT during rollout, so
            #     GT edge topology becomes physically wrong;
            # (b) FEA contact solvers deactivate contact elements once the needle
            #     tip is fully embedded, so late GT frames have no world edges at
            #     the tip even though the predicted needle is still in contact.
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
                n_nodes, node_stats, input_keys,
                needle_idx_t=needle_idx_t,
                tissue_idx_t=tissue_idx_t,
                fiber_dir_full=fiber_dir_full,
                use_cpress=use_cpress,
            )
            graph = graph.to(dist.device)
            pred_sub = model(graph.x, graph.edge_attr, graph).cpu()

            # Per-region denorm: compute needle/tissue masks in crop-local index space
            if per_region_norm:
                part_nodes_np = part_nodes.numpy()
                needle_mask = torch.from_numpy(np.isin(part_nodes_np, needle_node_indices))
                tissue_mask = ~needle_mask
            else:
                needle_mask = tissue_mask = None
            next_uvw_sub = _denorm_target(
                pred_sub, target_stats, target_keys, target_dims, needle_mask, tissue_mask
            )

            # --- Post-processing on needle displacement -----------------------
            # Both the consensus filter and the edge-length cap operate in
            # needle-local index space (0..n_needle-1).  Since part_nodes is
            # the full mesh, next_uvw_sub["u"][needle_node_indices] maps
            # directly to needle-local displacement.
            needle_idx_t_np = needle_node_indices  # np.ndarray, global indices
            needle_idx_local = torch.from_numpy(needle_idx_t_np.astype("int64"))

            if consensus_attenuation > 0.0:
                u_needle = next_uvw_sub["u"][needle_idx_local]
                u_needle = _consensus_filter(u_needle, needle_local_ei, consensus_attenuation)
                next_uvw_sub["u"][needle_idx_local] = u_needle

            if needle_edge_cap_mm is not None:
                coord_needle = state["coord"][needle_idx_local]
                u_needle = next_uvw_sub["u"][needle_idx_local]
                u_needle = _apply_needle_edge_cap(
                    u_needle, coord_needle, needle_local_ei, needle_edge_cap_mm
                )
                next_uvw_sub["u"][needle_idx_local] = u_needle

            # --- Integrate increments — only for nodes inside the crop ---
            for _key in target_keys:
                state[_key][part_nodes] = state[_key][part_nodes] + next_uvw_sub[_key]

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

            step_ms = (time.time() - t_step) * 1000
            print(
                f"  step {step + 1:3d}/{n_rollout}  →  {fname}"
                f"  ({len(part_nodes)} nodes, full mesh)"
                f"  [{step_ms:.0f}ms]"
                + ("  (+ GT)" if has_gt else "  (extrapolating beyond GT)")
            )

    for _step, _fname, future in write_futures:
        future.result()

    pvd_path = _write_pvd(out_dir, pvd_entries)
    total_s = time.time() - t0_rollout
    print(f"\nDone in {total_s:.1f}s ({total_s / n_rollout * 1000:.0f}ms/step). Open {pvd_path} in Paraview to animate.")
    print("Tip: in Paraview use Filters → Warp By Vector on U_pred to visualise displacement.")


if __name__ == "__main__":
    main()
