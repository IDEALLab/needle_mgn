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
from physicsnemo.models.meshgraphnet.bsms_mgn import BiStrideMeshGraphNet
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


def _build_part_local_edge_index(
    edge_index: torch.Tensor,
    part_node_indices: np.ndarray,
    n_nodes: int,
) -> torch.Tensor:
    """Extract edges with both endpoints in *part_node_indices* and relabel locally.

    Returns a (2, E_part) tensor of local indices into ``part_node_indices``, or
    (2, 0) if no qualifying edges exist.
    """
    part_idx_t = torch.from_numpy(part_node_indices.astype(np.int64))
    in_part = torch.zeros(n_nodes, dtype=torch.bool)
    in_part[part_idx_t] = True

    src, dst = edge_index
    mask = in_part[src] & in_part[dst]
    if mask.sum() == 0:
        return torch.zeros(2, 0, dtype=torch.long)

    global_to_local = torch.full((n_nodes,), -1, dtype=torch.long)
    global_to_local[part_idx_t] = torch.arange(len(part_node_indices), dtype=torch.long)
    return torch.stack([global_to_local[src[mask]], global_to_local[dst[mask]]])


def _build_needle_local_edge_index(
    edge_index: torch.Tensor,
    needle_node_indices: np.ndarray,
    n_nodes: int,
) -> torch.Tensor:
    """Backwards-compatible wrapper — equivalent to :func:`_build_part_local_edge_index`."""
    return _build_part_local_edge_index(edge_index, needle_node_indices, n_nodes)


def _axial_polyfit_blend(
    disp: torch.Tensor,
    axial_coords: torch.Tensor,
    degree: int,
    alpha: float,
    step: int = -1,
) -> torch.Tensor:
    """Project per-node displacement onto a polynomial-of-axial-coordinate subspace.

    For each spatial component, fit ``disp[:, c]`` as a polynomial of degree
    ``degree`` in ``axial_coords`` (least-squares), then blend:

        ``alpha * fit + (1 - alpha) * disp``

    The polynomial subspace captures displacement modes that vary smoothly
    along the needle axis: rigid translation (deg≥0), tilt (deg≥1), parabolic
    bending (deg≥2), cubic / S-curve (deg≥3).  Per-node high-frequency noise
    that doesn't lie in this subspace is suppressed by ``1 - alpha``.

    This is more permissive than :func:`_procrustes_blend` — Procrustes only
    keeps the 6-DoF rigid subspace, while a degree-3 polyfit keeps a
    ``3 * (degree + 1)``-dimensional subspace that includes real bending.

    Parameters
    ----------
    disp : torch.Tensor
        Per-node displacement, shape ``(N, 3)``.
    axial_coords : torch.Tensor
        Axial parametrisation of each node, shape ``(N,)``.  Computed once
        from the reference needle geometry by projecting onto its principal
        axis.
    degree : int
        Polynomial degree.  3 is a good default for clamped-cantilever
        bending; 1 ≈ Procrustes (rigid only); 5 admits S-curves.
    alpha : float
        Blend factor in ``[0, 1]``.  ``alpha=0`` disables; ``alpha=1``
        replaces ``disp`` entirely with the polynomial fit.
    """
    if alpha <= 0.0 or disp.shape[0] < degree + 1 or degree < 0:
        return disp

    if not torch.isfinite(disp).all():
        # disp has NaN/inf — torch.linalg.lstsq's MKL backend reports this
        # as the cryptic "SGELSY parameter 6" error.  Surface it clearly
        # instead.  The usual cause is the trained model producing
        # numerically unstable predictions that compound over the rollout
        # (especially with TFN at l_max=1 + small irreps).
        n_nan = int((~torch.isfinite(disp)).any(dim=-1).sum().item())
        raise RuntimeError(
            f"_axial_polyfit_blend: disp has NaN/inf on {n_nan} of "
            f"{disp.shape[0]} nodes (step={step}).  This usually means "
            f"the model's rollout state has diverged.  Rerun with "
            f"axial_polyfit_alpha=0 to see where the divergence starts, "
            f"and check pred_sub / state['v'] each step."
        )

    s_c = axial_coords - axial_coords.mean()
    s_max = s_c.abs().max()
    if s_max < 1e-8:
        return disp
    s_c = s_c / s_max  # rescale to [-1, 1] for numerical conditioning

    V = torch.stack([s_c ** k for k in range(degree + 1)], dim=1)  # (N, deg+1)
    coeffs = torch.linalg.lstsq(V, disp).solution                   # (deg+1, 3)
    fitted = V @ coeffs                                             # (N, 3)
    return alpha * fitted + (1.0 - alpha) * disp


def _procrustes_blend(
    prev_pos: torch.Tensor,
    disp: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Blend a predicted displacement field toward its rigid-body Kabsch fit.

    Given previous positions ``prev_pos`` (N, 3) and the model's predicted
    per-node displacement ``disp`` (N, 3), find the best rigid transform
    (R, t) that maps ``prev_pos`` onto ``prev_pos + disp`` in least-squares,
    and return:

        ``alpha * (rigid_pos - prev_pos) + (1 - alpha) * disp``

    With ``alpha=0`` the displacement is unchanged; with ``alpha=1`` the
    displacement is replaced by its rigid projection so the part moves as a
    perfectly rigid body.  Intermediate values dampen non-rigid per-node
    deviations (which are typically network noise on a quasi-rigid part)
    while preserving coherent translation/rotation.
    """
    if alpha <= 0.0 or prev_pos.shape[0] < 3:
        return disp
    pred_pos = prev_pos + disp
    mu_p = prev_pos.mean(dim=0, keepdim=True)
    mu_q = pred_pos.mean(dim=0, keepdim=True)
    P = prev_pos - mu_p
    Q = pred_pos - mu_q
    H = P.t() @ Q  # (3, 3) cross-covariance
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.t() @ U.t()))
    D = torch.eye(3, dtype=H.dtype, device=H.device)
    D[2, 2] = d
    R = Vt.t() @ D @ U.t()
    t = mu_q.squeeze(0) - mu_p.squeeze(0) @ R.t()
    rigid_pos = prev_pos @ R.t() + t
    rigid_disp = rigid_pos - prev_pos
    return alpha * rigid_disp + (1.0 - alpha) * disp


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
    radius: float,
    needle_node_indices: np.ndarray,
    tissue_node_indices: np.ndarray,
) -> tuple:
    """Build bidirectional world edges matching the odb_to_mgn_input.py algorithm.

    Replicates the dataset creation step exactly: for each needle node, find
    all tissue (Eulerian) nodes within ``radius`` using the fixed tissue
    KD-tree, and add a bidirectional edge.

    The tissue KD-tree is built once from the reference (undeformed) Eulerian
    positions because the Eulerian mesh does not move.  Needle positions are
    the current predicted/deformed positions, updated every rollout step.

    Parameters
    ----------
    needle_pos : np.ndarray, shape (N_needle, 3)
        Current predicted positions of needle nodes.
    tissue_kdtree : cKDTree
        KD-tree of reference Eulerian tissue node positions (built once).
    radius : float
        Search radius in mesh units (mm).  Must match ``EDGE_RADIUS`` in
        ``odb_to_mgn_input.py`` (default 1.2 mm).
    needle_node_indices : np.ndarray
        Global node indices for needle nodes.
    tissue_node_indices : np.ndarray
        Global node indices for tissue nodes.
    """
    neighbor_lists = tissue_kdtree.query_ball_point(needle_pos, r=radius)
    src_list, dst_list = [], []
    for local_ndl, neighbors in enumerate(neighbor_lists):
        needle_global = int(needle_node_indices[local_ndl])
        for local_eul in neighbors:
            src_list.append(needle_global)
            dst_list.append(int(tissue_node_indices[local_eul]))

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
    x: torch.Tensor,
    use_cpress: bool,
    mgn_paper_features: bool = False,
    mgn_include_mat_fiber: bool = False,
    mgn_include_prev_v: bool = False,
    mgn_include_evf: bool = False,
    mgn_include_arclen_clamp: bool = False,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Split flat node features into (x_scalar, x_vec) for TFNMeshGraphNet.

    Mirrors ``NeedleTissueDataset._split_tfn_features``.  In MGN-paper mode
    the layout is [node_type(2), evf(1)?, arclen(1)?, mat_fiber(3)?,
    prev_v(3)?] — all scalars first, all 1o vectors after, so the split is
    positional.
    """
    if mgn_paper_features:
        n_scalar = 2 + (1 if mgn_include_evf else 0) + (1 if mgn_include_arclen_clamp else 0)
        return x[:, :n_scalar], x[:, n_scalar:]
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
    mgn_paper_features: bool = False,
    mgn_node_features: Optional[torch.Tensor] = None,
    mgn_ref_pos: Optional[torch.Tensor] = None,
    mgn_include_mat_fiber: bool = False,
    mgn_include_prev_v: bool = False,
    mgn_include_evf: bool = False,
    mgn_include_arclen_clamp: bool = False,
    bevel_node_normal_full: Optional[torch.Tensor] = None,
    surface_node_normal_full: Optional[torch.Tensor] = None,
    arclen_clamp_full: Optional[torch.Tensor] = None,
    global_needle_vecs: bool = False,
    needle_idx_global: Optional[torch.Tensor] = None,
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

    if mgn_paper_features:
        # MGN paper input scheme: 2-dim node-type one-hot, edge_attr augmented
        # with mesh-space rel_pos (and its norm).  Layout matches dataset.py.
        if mgn_node_features is None or mgn_ref_pos is None:
            raise RuntimeError(
                "mgn_paper_features=True requires mgn_node_features and "
                "mgn_ref_pos to be supplied."
            )
        x_sub = mgn_node_features[part_nodes]
        # Scalars first (evf), then 1o vectors (mat_fiber, prev_v) — match
        # dataset.py ordering so x_scalar/x_vec splits stay aligned.
        if mgn_include_evf:
            evf_t = state["evf"]
            evf_norm = (evf_t - node_stats["evf_mean"]) / node_stats["evf_std"]
            x_sub = torch.cat([x_sub, evf_norm[part_nodes]], dim=-1)
        if mgn_include_arclen_clamp:
            if arclen_clamp_full is None:
                raise RuntimeError(
                    "mgn_include_arclen_clamp=True but arclen_clamp_full is None."
                )
            x_sub = torch.cat([x_sub, arclen_clamp_full[part_nodes]], dim=-1)
        if mgn_include_mat_fiber:
            if fiber_dir_full is None:
                raise RuntimeError(
                    "mgn_include_mat_fiber=True but fiber_dir_full is None — "
                    "node_props must contain mat_fiber."
                )
            x_sub = torch.cat([x_sub, fiber_dir_full[part_nodes]], dim=-1)
        if mgn_include_prev_v:
            v_t = state["v"]
            v_mean = node_stats["v_mean"]
            v_std = node_stats["v_std"]
            v_norm = (v_t - v_mean) / v_std
            x_sub = torch.cat([x_sub, v_norm[part_nodes]], dim=-1)
        ref_pos_sub = mgn_ref_pos[part_nodes]
    else:
        x_sub = _normalize(
            state, node_props, node_stats, input_keys, needle_idx_t, tissue_idx_t
        )[part_nodes]

    src, dst = all_ei
    rel_pos = coord_sub[src] - coord_sub[dst]
    edge_len = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)
    if mgn_paper_features:
        # Per Pfaff et al. (2020): only mesh edges carry mesh-space rel-pos.
        # World/contact edges get zeros there since the connected nodes were
        # disjoint in the rest-state mesh.  Must match dataset.py exactly.
        mesh_rel = ref_pos_sub[src] - ref_pos_sub[dst]
        mesh_d = torch.linalg.norm(mesh_rel, dim=-1, keepdim=True)
        is_world = all_et[:, 2] > 0.5
        if is_world.any():
            mesh_rel[is_world] = 0.0
            mesh_d[is_world] = 0.0
        edge_attr = torch.cat(
            [rel_pos, edge_len, mesh_rel, mesh_d, all_et], dim=-1
        )
    else:
        edge_attr = torch.cat([rel_pos, edge_len, all_et], dim=-1)

    # Unit fiber direction per node (passed through to FiberEquivariantMGN).
    fiber_dir_sub = None
    if fiber_dir_full is not None:
        fiber_dir_sub = fiber_dir_full[part_nodes]

    # Split into scalar/vector parts for TFNMeshGraphNet.
    x_scalar_sub, x_vec_sub = _split_tfn_features(
        x_sub,
        use_cpress,
        mgn_paper_features=mgn_paper_features,
        mgn_include_mat_fiber=mgn_include_mat_fiber,
        mgn_include_prev_v=mgn_include_prev_v,
        mgn_include_evf=mgn_include_evf,
        mgn_include_arclen_clamp=mgn_include_arclen_clamp,
    )

    # Always attach normalised current velocity for the fiber-extra edge
    # invariants (also harmless when those flags are off).
    v_node_full = state["v"]
    v_norm_full = (v_node_full - node_stats["v_mean"]) / node_stats["v_std"]
    node_velocity_sub = v_norm_full[part_nodes]

    # Per-node 1o vector input for FiberEquivariantMGN(extra_node_vec=True).
    # bevel_node_normal_full: precomputed bevel-face normal (zero off bevel).
    # surface_node_normal_full: surface normal, masked per-step to nodes with
    # at least one world (contact) edge.
    extra_node_vec_sub = None
    if bevel_node_normal_full is not None:
        extra_node_vec_sub = bevel_node_normal_full[part_nodes]
    elif surface_node_normal_full is not None:
        extra_node_vec_sub = surface_node_normal_full[part_nodes].clone()
        has_contact = torch.zeros(part_nodes.shape[0], dtype=torch.bool)
        world_edge_mask = all_et[:, 2] > 0.5
        if world_edge_mask.any():
            wei = all_ei[:, world_edge_mask]
            has_contact[wei[0]] = True
            has_contact[wei[1]] = True
        extra_node_vec_sub[~has_contact] = 0.0

    # Global per-frame needle features (centroid, axis_dir, centroid_v,
    # ang_v) — computed from the CURRENT autoregressive state (not from
    # frame_tensors, which haven't been advanced).  Mirrors what the
    # dataset emits during training but recomputed each step here.
    global_needle_vecs_sub = None
    if global_needle_vecs:
        if needle_idx_global is None:
            raise RuntimeError(
                "global_needle_vecs=True requires needle_idx_global "
                "(global indices of needle nodes in the full mesh)."
            )
        pos_full = state["coord"]
        v_full = state["v"]
        needle_pos_full = pos_full[needle_idx_global].float()
        needle_v_full = v_full[needle_idx_global].float()

        centroid = needle_pos_full.mean(dim=0)
        centred = needle_pos_full - centroid
        _, _, _Vt_g = torch.linalg.svd(centred, full_matrices=False)
        axis_dir = _Vt_g[0]
        if float(axis_dir[int(torch.argmax(torch.abs(axis_dir)).item())]) < 0:
            axis_dir = -axis_dir

        centroid_v = needle_v_full.mean(dim=0)
        v_rel = needle_v_full - centroid_v
        ang_num = torch.cross(centred, v_rel, dim=-1).sum(dim=0)
        ang_den = (centred * centred).sum().clamp(min=1e-12)
        ang_v = ang_num / ang_den

        coord_std_t = node_stats["coord_std"].float() if isinstance(
            node_stats["coord_std"], torch.Tensor
        ) else torch.tensor(node_stats["coord_std"], dtype=torch.float32)
        v_std_t = node_stats["v_std"].float() if isinstance(
            node_stats["v_std"], torch.Tensor
        ) else torch.tensor(node_stats["v_std"], dtype=torch.float32)
        coord_std_t = coord_std_t.view(1, 3)
        v_std_t = v_std_t.view(1, 3)

        # Centroid as a per-local-node 1o vector: (centroid − pos_local) / coord_std.
        local_pos = coord_sub.float()
        centroid_rel = (centroid.view(1, 3) - local_pos) / coord_std_t
        axis_bcast = axis_dir.view(1, 3).expand(part_nodes.shape[0], 3).clone()
        centroid_v_bcast = (centroid_v.view(1, 3) / v_std_t).expand(part_nodes.shape[0], 3).clone()
        ang_v_bcast = ang_v.view(1, 3).expand(part_nodes.shape[0], 3).clone()

        # Mask tissue nodes to zero (same convention as the dataset).
        is_needle_full = torch.zeros(n_nodes, dtype=torch.bool)
        is_needle_full[needle_idx_global] = True
        is_needle_local_t = is_needle_full[part_nodes]
        not_needle = ~is_needle_local_t
        if not_needle.any():
            centroid_rel[not_needle] = 0.0
            axis_bcast[not_needle] = 0.0
            centroid_v_bcast[not_needle] = 0.0
            ang_v_bcast[not_needle] = 0.0

        global_needle_vecs_sub = torch.stack(
            [centroid_rel, axis_bcast, centroid_v_bcast, ang_v_bcast], dim=1
        )  # (n_sub, 4, 3)

    return Data(
        x=x_sub,
        edge_attr=edge_attr,
        edge_index=all_ei,
        pos=coord_sub,
        num_nodes=len(part_nodes),
        fiber_dir=fiber_dir_sub,
        x_scalar=x_scalar_sub,
        x_vec=x_vec_sub,
        node_velocity=node_velocity_sub,
        extra_node_vec=extra_node_vec_sub,
        global_needle_vecs=global_needle_vecs_sub,
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

def _load_ckpt_config_over_defaults(cfg: DictConfig) -> DictConfig:
    """Make ``<ckpt_path>/config.yaml`` the source of truth.

    Logic (in order of precedence, low → high):
      1. defaults baked into ``conf/config.yaml`` (Hydra loads this)
      2. values saved in ``<ckpt_path>/config.yaml`` (the cfg used at the
         time of training, written by train.py when wandb_log_artifact=true)
      3. any keys explicitly overridden on the CLI

    The intent: running ``infer.py ckpt_path=/path/to/exp/checkpoints
    data_dir=/path/to/RUN-2`` should reproduce the model's training-time
    feature scheme automatically, regardless of what ``conf/config.yaml``
    currently says — and CLI overrides like ``base_clamp_mm`` /
    ``apply_rigid_correction`` should still win.
    """
    from hydra.core.hydra_config import HydraConfig  # noqa: PLC0415 — runtime import

    try:
        cli_overrides = list(HydraConfig.get().overrides.task)
    except Exception:
        cli_overrides = []
    cli_keys = set()
    for o in cli_overrides:
        if "=" in o:
            k = o.split("=", 1)[0].lstrip("+~")
            cli_keys.add(k)

    ckpt_path_abs = _abspath(cfg.ckpt_path)
    ckpt_cfg_file = os.path.join(ckpt_path_abs, "config.yaml")
    if not os.path.exists(ckpt_cfg_file):
        print(f"  (no checkpoint config at {ckpt_cfg_file}; using conf/config.yaml + CLI overrides only)")
        return cfg

    ckpt_cfg = OmegaConf.load(ckpt_cfg_file)
    # Compose: ckpt values overlay defaults, then CLI overrides take precedence.
    merged = OmegaConf.merge(cfg, ckpt_cfg)
    # Re-apply CLI overrides on top of the checkpoint cfg.
    if cli_keys:
        for key in cli_keys:
            try:
                val = OmegaConf.select(cfg, key)
                OmegaConf.update(merged, key, val, merge=True)
            except Exception:
                pass  # silently skip keys that can't be re-applied
    overridden_str = ", ".join(sorted(cli_keys)) if cli_keys else "(none)"
    print(f"  loaded experiment config from {ckpt_cfg_file}")
    print(f"  CLI overrides preserved: {overridden_str}")
    return merged


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = _load_ckpt_config_over_defaults(cfg)
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
        # Must mirror dataset.py's deterministic shuffle so the test set
        # held out at inference matches the runs the model never saw
        # during training.  Seed 42 is hard-coded in dataset.py.
        import random as _random
        _random.Random(42).shuffle(run_ids)
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

    # Drop kinematic targets the model wasn't trained to predict.
    drop_targets = list(OmegaConf.select(cfg, "drop_targets", default=[]) or [])
    if drop_targets:
        keep = [(k, d) for k, d in zip(target_keys, target_dims) if k not in drop_targets]
        target_keys = [k for k, _ in keep]
        target_dims = [d for _, d in keep]
        print(f"  drop_targets: {drop_targets} → predicting {target_keys}")

    # MGN-paper feature scheme: node-type one-hot + edge_attr augmented with
    # mesh-space rel_pos.  Must match the dataset construction at training
    # time.  When enabled, input_dim_nodes is 2 regardless of use_cpress
    # (or 5 if mgn_include_mat_fiber adds the unit fiber direction).
    mgn_paper_features = bool(OmegaConf.select(cfg, "mgn_paper_features", default=False))
    mgn_include_mat_fiber = bool(OmegaConf.select(cfg, "mgn_include_mat_fiber", default=False))
    mgn_include_prev_v = bool(OmegaConf.select(cfg, "mgn_include_prev_v", default=False))
    mgn_include_evf = bool(OmegaConf.select(cfg, "mgn_include_evf", default=False))
    if mgn_paper_features:
        extra = ""
        input_dim_nodes = 2
        if mgn_include_evf:
            input_dim_nodes += 1
            extra += " + evf(1)"
        if bool(OmegaConf.select(cfg, "mgn_include_arclen_clamp", default=False)):
            input_dim_nodes += 1
            extra += " + arclen_clamp(1)"
        if mgn_include_mat_fiber:
            input_dim_nodes += 3
            extra += " + mat_fiber(3)"
        if mgn_include_prev_v:
            input_dim_nodes += 3
            extra += " + prev_v(3)"
        print(
            f"  mgn_paper_features=true: input_dim_nodes={input_dim_nodes} "
            f"(node_type(2){extra})"
        )
    else:
        input_dim_nodes = sum(input_dims) + sum(_STATIC_PROP_DIMS)
    output_dim = sum(target_dims)

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
    if model_type == "bistride":
        model = BiStrideMeshGraphNet(
            **_shared_kwargs,
            num_mesh_levels=int(OmegaConf.select(cfg, "num_bsms_levels", default=2)),
            bistride_pos_dim=3,
            num_layers_bistride=int(OmegaConf.select(cfg, "num_layers_bistride", default=2)),
            bistride_unet_levels=int(OmegaConf.select(cfg, "bistride_unet_levels", default=1)),
            num_processor_checkpoint_segments=int(OmegaConf.select(cfg, "num_processor_checkpoint_segments", default=0)),
        )
    elif model_type == "kan":
        model = MeshGraphKAN(
            **_shared_kwargs,
            num_harmonics=int(OmegaConf.select(cfg, "num_harmonics", default=5)),
        )
    elif model_type == "fiber":
        _extra_node_vec = bool(
            OmegaConf.select(cfg, "bevel_normal_feature", default=False)
            or OmegaConf.select(cfg, "surface_contact_normal_feature", default=False)
        )
        _n_global_needle_vecs = 4 if bool(
            OmegaConf.select(cfg, "global_needle_vecs", default=False)
        ) else 0
        model = FiberEquivariantMGN(
            **_shared_kwargs,
            n_vec_outputs=int(OmegaConf.select(cfg, "n_vec_outputs", default=3)),
            extra_edge_invariants=bool(OmegaConf.select(cfg, "fiber_extra_invariants", default=False)),
            extra_decoder_basis=bool(OmegaConf.select(cfg, "fiber_extra_decoder_basis", default=False)),
            contact_decoder_basis=bool(OmegaConf.select(cfg, "contact_decoder_basis", default=False)),
            extra_node_vec=_extra_node_vec,
            n_global_needle_vecs=_n_global_needle_vecs,
            displacement_bevel_ref=bool(OmegaConf.select(cfg, "displacement_bevel_ref", default=False)),
            bevel_axis=list(OmegaConf.select(cfg, "bevel_axis", default=[1.0, 0.0, 0.0])),
        )
    elif model_type == "fiber_kan":
        model = FiberEquivariantKAN(
            **_shared_kwargs,
            n_vec_outputs=int(OmegaConf.select(cfg, "n_vec_outputs", default=3)),
            num_harmonics=int(OmegaConf.select(cfg, "num_harmonics", default=5)),
            extra_edge_invariants=bool(OmegaConf.select(cfg, "fiber_extra_invariants", default=False)),
            extra_decoder_basis=bool(OmegaConf.select(cfg, "fiber_extra_decoder_basis", default=False)),
            contact_decoder_basis=bool(OmegaConf.select(cfg, "contact_decoder_basis", default=False)),
        )
    elif model_type == "tfn":
        if mgn_paper_features:
            n_tfn_scalar = 2 + int(mgn_include_evf)
            n_tfn_vec = int(mgn_include_mat_fiber) + int(mgn_include_prev_v)
        else:
            n_tfn_scalar = 15 if use_cpress else 14
            n_tfn_vec = 4
        model = TFNMeshGraphNet(
            n_node_scalar=n_tfn_scalar,
            n_node_vec=n_tfn_vec,
            output_dim=output_dim,
            irreps_hidden=str(OmegaConf.select(cfg, "irreps_hidden", default="16x0e + 8x1o + 4x2e")),
            l_max=int(OmegaConf.select(cfg, "l_max", default=2)),
            n_radial_basis=int(OmegaConf.select(cfg, "n_radial_basis", default=8)),
            r_max=float(OmegaConf.select(cfg, "r_max", default=60.0)),
            n_edge_extra_scalar=int(cfg.input_dim_edges) - 3,
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

    # Per-rollout-step physical dt — used to integrate Δv → Δu when "u" is
    # dropped from training targets.  Saved to target_stats.json by dataset.py
    # during training; falls back to a manual override or to NaN otherwise.
    rollout_dt: float
    _dt_cfg = OmegaConf.select(cfg, "rollout_dt", default=None)
    if _dt_cfg is not None:
        rollout_dt = float(_dt_cfg)
        print(f"  rollout_dt = {rollout_dt:.6g} s (from cfg override)")
    elif "rollout_dt" in target_stats:
        _dt_t = target_stats["rollout_dt"]
        rollout_dt = float(_dt_t.item() if hasattr(_dt_t, "item") else _dt_t[0] if hasattr(_dt_t, "__getitem__") else _dt_t)
        print(f"  rollout_dt = {rollout_dt:.6g} s (estimated from training data)")
    else:
        rollout_dt = float("nan")
        if "u" in drop_targets:
            print(
                "  WARNING: 'u' dropped but rollout_dt not in target_stats and not "
                "supplied via cfg.rollout_dt — Δu integration will produce NaN."
            )

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

    # ---- BSMS multi-scale edges (BiStride model only) ----------------------
    # Load from the cache file written by dataset.py during training, or
    # compute on the fly if it is missing (requires sparse_dot_mkl / MKL).
    bsms_ms_edges: list = []
    bsms_ms_ids: list = []
    if model_type == "bistride":
        num_bsms_levels = int(OmegaConf.select(cfg, "num_bsms_levels", default=2))
        beam_spacing_mm = float(OmegaConf.select(cfg, "beam_spacing_mm", default=0.0))
        tissue_downsample_mm = float(OmegaConf.select(cfg, "tissue_downsample_mm", default=0.0))
        _beam_tag = f"_b{beam_spacing_mm:.2g}mm" if beam_spacing_mm > 0.0 else ""
        _tissue_tag = f"_t{tissue_downsample_mm:.2g}mm" if tissue_downsample_mm > 0.0 else ""
        bsms_name = f"bsms_cache_l{num_bsms_levels}{_beam_tag}{_tissue_tag}.pt"
        bsms_path = os.path.join(data_dir, bsms_name)
        if os.path.exists(bsms_path):
            print(f"Loading BSMS cache from {bsms_path} ...")
            bsms_saved = torch.load(bsms_path, weights_only=False)
            bsms_ms_edges = [e.to(dist.device) for e in bsms_saved["ms_edges"]]
            bsms_ms_ids = [ids.to(dist.device) for ids in bsms_saved["ms_ids"]]
        else:
            print(f"BSMS cache not found at {bsms_path}, computing from frame {infer_start} ...")
            from dataset import _precompute_bsms_full
            coord_ref = frame_tensors["coord"][infer_start]
            ms_edges_cpu, ms_ids_cpu = _precompute_bsms_full(
                hex_edge_index, coord_ref, n_nodes, num_bsms_levels
            )
            bsms_ms_edges = [e.to(dist.device) for e in ms_edges_cpu]
            bsms_ms_ids = [ids.to(dist.device) for ids in ms_ids_cpu]

    # ---- Needle / tissue node index sets ---------------------------------
    needle_node_indices, tissue_node_indices = _get_needle_tissue_node_sets(
        hex_edge_index, hex_edge_type_onehot
    )

    # Override mat_fiber on needle nodes with the principal axis of the
    # frame-0 needle geometry — must match dataset.py's training-time override
    # when needle_fiber_axis=true so the equivariant decoder receives a
    # transverse basis and the input mat_fiber statistics line up with the
    # training stats.
    needle_fiber_axis = bool(OmegaConf.select(cfg, "needle_fiber_axis", default=False))
    if needle_fiber_axis:
        _needle_idx_t = torch.from_numpy(needle_node_indices.astype(np.int64))
        _ref_coord = frame_tensors["coord"][0].float()
        _needle_pts = _ref_coord[_needle_idx_t]
        _centered = _needle_pts - _needle_pts.mean(dim=0, keepdim=True)
        _, _, _Vt_axis = torch.linalg.svd(_centered, full_matrices=False)
        _axis = _Vt_axis[0]
        _axis = _axis / _axis.norm().clamp(min=1e-8)
        if "mat_fiber" not in node_props:
            node_props["mat_fiber"] = torch.zeros(n_nodes, 3, dtype=torch.float32)
        _new_mf = node_props["mat_fiber"].clone().float()
        _new_mf[_needle_idx_t] = _axis.expand(_needle_idx_t.numel(), 3)
        node_props["mat_fiber"] = _new_mf
        print(
            f"  needle_fiber_axis: overrode mat_fiber on {_needle_idx_t.numel()} "
            f"needle nodes with principal axis {_axis.tolist()}"
        )

    # Pre-compute unit fiber direction for every node (used by FiberEquivariantMGN).
    if "mat_fiber" in node_props:
        _fiber_raw = node_props["mat_fiber"].float()   # (N, 3)
        _fiber_norm = torch.linalg.norm(_fiber_raw, dim=-1, keepdim=True).clamp(min=1e-8)
        fiber_dir_full = _fiber_raw / _fiber_norm       # (N, 3) unit vectors
    else:
        fiber_dir_full = None

    # ---- Optional precomputed needle-geometry features --------------------
    # Variants cropped_fiber_iso_invw_bevel / _contact ship with a per-node
    # 1o vector input (bevel-face or surface-contact normal) loaded once
    # from <data_dir>/needle_geometry_features.pt and attached to each
    # step's graph as `extra_node_vec`.  FiberEquivariantMGN(extra_node_vec=
    # True) reads it.
    bevel_normal_feature = bool(OmegaConf.select(cfg, "bevel_normal_feature", default=False))
    surface_contact_normal_feature = bool(
        OmegaConf.select(cfg, "surface_contact_normal_feature", default=False)
    )
    mgn_include_arclen_clamp = bool(
        OmegaConf.select(cfg, "mgn_include_arclen_clamp", default=False)
    )
    bevel_node_normal_full = None
    surface_node_normal_full = None
    arclen_clamp_full = None
    if bevel_normal_feature or surface_contact_normal_feature or mgn_include_arclen_clamp:
        _geom_path = OmegaConf.select(cfg, "needle_geometry_path", default=None)
        if _geom_path is None:
            _geom_path = os.path.join(_abspath(cfg.data_dir), "needle_geometry_features.pt")
        else:
            _geom_path = _abspath(_geom_path)
        if not os.path.exists(_geom_path):
            raise FileNotFoundError(
                f"needle geometry features not found at {_geom_path}.  "
                f"Run compute_needle_geometry.py to generate it."
            )
        _geom = torch.load(_geom_path, weights_only=False)
        if bevel_normal_feature:
            bevel_node_normal_full = _geom["bevel_node_normal"].float()
            print(f"  bevel_normal_feature: loaded from {_geom_path}")
        if surface_contact_normal_feature:
            surface_node_normal_full = _geom["surface_node_normal"].float()
            print(f"  surface_contact_normal_feature: loaded from {_geom_path}")
        if mgn_include_arclen_clamp:
            if "arclen_to_clamp" not in _geom:
                raise KeyError(
                    f"'arclen_to_clamp' missing from {_geom_path}. Re-run "
                    f"compute_needle_geometry.py to regenerate it."
                )
            arclen_clamp_full = _geom["arclen_to_clamp"].float().view(-1, 1)
            print(f"  mgn_include_arclen_clamp: loaded from {_geom_path}")

    # ---- Rigid-body bias correction --------------------------------------
    # When apply_rigid_correction=true, load a cached (Δt_k, Δω_k) tensor
    # produced by compute_rigid_correction.py and subtract the per-step
    # rigid component from the model's normalised Δu prediction on needle
    # nodes before applying the state update.  Reduces systematic
    # translation / rotation drift averaged out of training rollouts.
    apply_rigid_correction = bool(
        OmegaConf.select(cfg, "apply_rigid_correction", default=False)
    )
    rigid_correction_per_step = bool(
        OmegaConf.select(cfg, "rigid_correction_per_step", default=True)
    )
    rigid_correction = None
    if apply_rigid_correction:
        _rc_path = OmegaConf.select(cfg, "rigid_correction_path", default=None)
        if _rc_path is None:
            _rc_path = os.path.join(stats_dir, "rigid_correction.pt")
        else:
            _rc_path = _abspath(_rc_path)
        if not os.path.exists(_rc_path):
            raise FileNotFoundError(
                f"apply_rigid_correction=true but no cache at {_rc_path}.  "
                f"Run compute_rigid_correction.py first."
            )
        rigid_correction = torch.load(_rc_path, weights_only=False)
        print(
            f"  apply_rigid_correction: loaded from {_rc_path} "
            f"(n_samples={rigid_correction['n_samples']}, "
            f"K={rigid_correction['n_steps']}, "
            f"per_step={rigid_correction_per_step})"
        )

    # ---- Global per-frame needle features --------------------------------
    # When global_needle_vecs=true, the model expects a per-step (n_sub, 4, 3)
    # tensor (centroid_rel, axis_dir, centroid_v, ang_v) computed from the
    # *current* autoregressive state.  Mirrors the dataset emission but
    # rebuilt each step here.  Just needs needle_idx_t — already available.
    global_needle_vecs_flag = bool(
        OmegaConf.select(cfg, "global_needle_vecs", default=False)
    )
    if global_needle_vecs_flag:
        print(f"  global_needle_vecs: enabled (4 channels per needle node)")

    # MGN-paper feature precomputation — match dataset.py: 2-dim node-type
    # one-hot per node, and per-run reference positions (frame-0 coord minus
    # frame-0 displacement) for mesh-space rel_pos in the edge encoder.
    if mgn_paper_features:
        mgn_node_features = torch.zeros(n_nodes, 2, dtype=torch.float32)
        mgn_node_features[needle_node_indices, 0] = 1.0
        mgn_node_features[tissue_node_indices, 1] = 1.0
        mgn_ref_pos = (
            frame_tensors["coord"][0] - frame_tensors["u"][0]
        ).float()
        print(
            f"  mgn_paper_features: edge_attr expanded to "
            f"{cfg.input_dim_edges}-dim "
            f"[world_rel(3), world_d(1), mesh_rel(3), mesh_d(1), edge_type(3)]"
        )
    else:
        mgn_node_features = None
        mgn_ref_pos = None

    print(
        f"Node sets: {len(needle_node_indices)} needle, "
        f"{len(tissue_node_indices)} tissue (total {n_nodes})"
    )
    if per_region_norm:
        print("Per-region normalization: enabled (detected from stats file)")
    needle_idx_t = torch.from_numpy(needle_node_indices.astype(np.int64)) if per_region_norm else None
    tissue_idx_t = torch.from_numpy(tissue_node_indices.astype(np.int64)) if per_region_norm else None

    # ---- World edge radius -----------------------------------------------
    # Must match EDGE_RADIUS in odb_to_mgn_input.py (default 1.2 mm) so that
    # inference generates exactly the same connectivity pattern as training.
    # The tissue (Eulerian) mesh is fixed, so we build the KD-tree once from
    # the reference frame and reuse it for all rollout steps.
    world_edge_radius = float(
        OmegaConf.select(cfg, "world_edge_radius", default=1.2)
    )
    tissue_pos_np = frame_tensors["coord"][infer_start][tissue_node_indices].numpy()
    tissue_kdtree = cKDTree(tissue_pos_np)
    print(f"  World edge radius = {world_edge_radius:.4f} mm (Eulerian tissue KD-tree built)")

    # Read the start VTU for output mesh topology (HEX cells are fixed across frames).
    ref_mesh_start = pv.read(vtu_files[infer_start])

    # ---- Post-processing precomputation ------------------------------------
    consensus_attenuation = float(OmegaConf.select(cfg, "consensus_attenuation", default=0.0))
    tissue_consensus_attenuation = float(
        OmegaConf.select(cfg, "tissue_consensus_attenuation", default=0.0)
    )
    procrustes_alpha = float(OmegaConf.select(cfg, "procrustes_alpha", default=0.0))
    axial_polyfit_alpha = float(OmegaConf.select(cfg, "axial_polyfit_alpha", default=0.0))
    axial_polyfit_degree = int(OmegaConf.select(cfg, "axial_polyfit_degree", default=3))

    needle_local_ei = _build_part_local_edge_index(
        hex_edge_index, needle_node_indices, n_nodes
    )
    tissue_local_ei = _build_part_local_edge_index(
        hex_edge_index, tissue_node_indices, n_nodes
    )
    needle_idx_local = torch.from_numpy(needle_node_indices.astype(np.int64))
    tissue_idx_local = torch.from_numpy(tissue_node_indices.astype(np.int64))

    if consensus_attenuation > 0.0:
        print(
            f"Needle consensus filter: attenuation={consensus_attenuation:.2f}, "
            f"needle-needle edges={needle_local_ei.shape[1]}"
        )
    if tissue_consensus_attenuation > 0.0:
        print(
            f"Tissue consensus filter: attenuation={tissue_consensus_attenuation:.2f}, "
            f"tissue-tissue edges={tissue_local_ei.shape[1]}"
        )
    if procrustes_alpha > 0.0:
        print(
            f"Procrustes rigid blend on needle: alpha={procrustes_alpha:.2f} "
            f"({len(needle_node_indices)} needle nodes)"
        )

    # Axial parametrisation of the needle from the reference (start) frame.
    # Each needle node's coordinate along the principal SVD axis is computed
    # once and reused every rollout step for the polyfit projection.
    _ref_needle_pos = frame_tensors["coord"][infer_start][needle_node_indices].float()
    _ref_centered = _ref_needle_pos - _ref_needle_pos.mean(dim=0, keepdim=True)
    _, _, _Vt_needle = torch.linalg.svd(_ref_centered, full_matrices=False)
    needle_axial_coords = _ref_centered @ _Vt_needle[0]  # (n_needle,)
    if axial_polyfit_alpha > 0.0:
        print(
            f"Axial polyfit on needle: alpha={axial_polyfit_alpha:.2f}, "
            f"degree={axial_polyfit_degree}, "
            f"axial range={needle_axial_coords.min():.1f}..{needle_axial_coords.max():.1f} mm"
        )

    # ---- Base-clamp inference scheme -------------------------------------
    # When base_clamp_mm > 0, the rollout overrides needle nodes within
    # base_clamp_mm of the *base* end of the needle with their ground-truth
    # state at every step.  This simulates having a known/tracked base (e.g.
    # a robotic insertion fixture) while letting the model predict the rest
    # of the needle and the tissue.  Base = minimum axial coord (tip is at
    # max axial per the sign convention used elsewhere in this codebase).
    #
    # The override is applied AFTER the state update, on the dynamic state
    # keys (u, v, coord, a if predicted), so the next step's input reflects
    # the clamped value.
    base_clamp_mm = float(OmegaConf.select(cfg, "base_clamp_mm", default=0.0))
    base_clamp_keys = ("u", "v", "a", "coord")
    base_node_global_idx: Optional[np.ndarray] = None
    if base_clamp_mm > 0.0:
        axial_min = float(needle_axial_coords.min().item())
        keep = (needle_axial_coords <= axial_min + base_clamp_mm).numpy()
        base_node_global_idx = needle_node_indices[keep]
        print(
            f"Base clamp: {len(base_node_global_idx)} needle nodes "
            f"(axial coord ≤ {axial_min + base_clamp_mm:.2f} mm, "
            f"base_clamp_mm={base_clamp_mm})"
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

            # --- World edges: KD-tree matching odb_to_mgn_input.py --------------
            # Build edges from predicted needle positions using the same fixed
            # radius (world_edge_radius) and fixed Eulerian tissue KD-tree that
            # odb_to_mgn_input.py used when creating the training VTU files.
            # This exactly replicates the training distribution for any rollout
            # step, adapting to the predicted needle geometry as it evolves.
            needle_pos_np = state["coord"][needle_node_indices].numpy()
            world_ei, world_et = _build_world_edges(
                needle_pos_np,
                tissue_kdtree,
                world_edge_radius,
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
                mgn_paper_features=mgn_paper_features,
                mgn_node_features=mgn_node_features,
                mgn_ref_pos=mgn_ref_pos,
                mgn_include_mat_fiber=mgn_include_mat_fiber,
                mgn_include_prev_v=mgn_include_prev_v,
                mgn_include_evf=mgn_include_evf,
                mgn_include_arclen_clamp=mgn_include_arclen_clamp,
                bevel_node_normal_full=bevel_node_normal_full,
                surface_node_normal_full=surface_node_normal_full,
                arclen_clamp_full=arclen_clamp_full,
                global_needle_vecs=global_needle_vecs_flag,
                needle_idx_global=(
                    torch.from_numpy(needle_node_indices.astype(np.int64))
                    if global_needle_vecs_flag else None
                ),
            )
            graph = graph.to(dist.device)
            if model_type == "bistride":
                pred_sub = model(graph.x, graph.edge_attr, graph, bsms_ms_edges, bsms_ms_ids).cpu()
            else:
                pred_sub = model(graph.x, graph.edge_attr, graph).cpu()

            # Diverged-rollout guard: surface the bad step early instead of
            # letting NaN propagate into post-processing where it appears as
            # cryptic MKL/lstsq errors.  Reports per-target counts so it's
            # obvious whether v, s, evf, ... are the source.
            if not torch.isfinite(pred_sub).all():
                offset = 0
                breakdown = []
                for key, dim in zip(target_keys, target_dims):
                    chunk = pred_sub[:, offset : offset + dim]
                    n_bad = int((~torch.isfinite(chunk)).any(dim=-1).sum().item())
                    breakdown.append(f"{key}: {n_bad}/{chunk.shape[0]} nodes")
                    offset += dim
                # Also surface the largest finite |Δ| before the divergence
                # so it's clear whether the model was already pushing scale.
                fin = pred_sub[torch.isfinite(pred_sub)]
                max_abs = float(fin.abs().max().item()) if fin.numel() > 0 else float("nan")
                raise RuntimeError(
                    f"Model produced NaN/inf at rollout step {step} (run "
                    f"{infer_run_id}).  Per-target NaN counts: "
                    f"{', '.join(breakdown)}.  Max |finite|={max_abs:.3g}.  "
                    f"This usually indicates rollout drift compounding into "
                    f"numerical overflow — common with TFN at small irreps "
                    f"and noise_std=0.  Try: (1) shorter n_rollout to confirm "
                    f"the model is fine for early steps, (2) retrain with "
                    f"noise_std=3e-3, (3) inspect state['v'] norms each step."
                )

            # ---- Rigid-body bias correction (normalised space) -------------
            # Subtract Δt_k + Δω_k × r_i from pred_sub[u_cols] on needle nodes
            # using cached per-step training-rollout averages.
            if rigid_correction is not None and "u" in target_keys:
                u_off = 0
                for _k, _d in zip(target_keys, target_dims):
                    if _k == "u":
                        break
                    u_off += _d
                if rigid_correction_per_step:
                    K_cache = int(rigid_correction["n_steps"])
                    k_idx = min(step, K_cache - 1)
                    dt = rigid_correction["delta_t_norm"][k_idx]      # (3,)
                    dw = rigid_correction["delta_omega_norm"][k_idx]  # (3,)
                else:
                    dt = rigid_correction["delta_t_norm_mean"]
                    dw = rigid_correction["delta_omega_norm_mean"]
                part_nodes_np = part_nodes.numpy()
                needle_in_part_mask = torch.from_numpy(np.isin(part_nodes_np, needle_node_indices))
                if needle_in_part_mask.any():
                    pos_needle_local = state["coord"][part_nodes][needle_in_part_mask].float()
                    centroid_n = pos_needle_local.mean(dim=0)
                    r = pos_needle_local - centroid_n                  # (n_needle_local, 3)
                    rigid_corr = dt.view(1, 3) + torch.cross(
                        dw.view(1, 3).expand_as(r), r, dim=-1
                    )                                                  # (n_needle_local, 3)
                    pred_sub[needle_in_part_mask, u_off : u_off + 3] -= rigid_corr

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

            # --- Derive Δu from Δv when the model wasn't trained to predict u.
            # Trapezoidal rule: Δu = v_t · dt + 0.5 · Δv_pred · dt
            #   = 0.5 · (v_t + v_{t+1}) · dt   (semi-implicit, energy-preserving)
            # This produces a "u" entry in next_uvw_sub that the standard
            # post-processing and state-update paths below can consume
            # unchanged.
            if "u" in drop_targets:
                if "v" not in next_uvw_sub:
                    raise RuntimeError(
                        "drop_targets includes 'u' but the model didn't predict 'v' — "
                        "cannot integrate to recover Δu."
                    )
                v_t_sub = state["v"][part_nodes]
                delta_v_pred = next_uvw_sub["v"]
                next_uvw_sub["u"] = (v_t_sub + 0.5 * delta_v_pred) * rollout_dt

            # --- Post-processing on predicted displacement --------------------
            # All filters operate in part-local index space (0..n_part-1).
            # Since part_nodes is the full mesh, indexing next_uvw_sub["u"]
            # by global node indices works directly.
            if consensus_attenuation > 0.0:
                u_needle = next_uvw_sub["u"][needle_idx_local]
                u_needle = _consensus_filter(u_needle, needle_local_ei, consensus_attenuation)
                next_uvw_sub["u"][needle_idx_local] = u_needle

            if axial_polyfit_alpha > 0.0:
                u_needle = next_uvw_sub["u"][needle_idx_local]
                u_needle = _axial_polyfit_blend(
                    u_needle, needle_axial_coords,
                    degree=axial_polyfit_degree,
                    alpha=axial_polyfit_alpha,
                    step=step,
                )
                next_uvw_sub["u"][needle_idx_local] = u_needle

            if procrustes_alpha > 0.0:
                prev_needle_pos = state["coord"][needle_idx_local]
                u_needle = next_uvw_sub["u"][needle_idx_local]
                u_needle = _procrustes_blend(prev_needle_pos, u_needle, procrustes_alpha)
                next_uvw_sub["u"][needle_idx_local] = u_needle

            if needle_edge_cap_mm is not None:
                coord_needle = state["coord"][needle_idx_local]
                u_needle = next_uvw_sub["u"][needle_idx_local]
                u_needle = _apply_needle_edge_cap(
                    u_needle, coord_needle, needle_local_ei, needle_edge_cap_mm
                )
                next_uvw_sub["u"][needle_idx_local] = u_needle

            if tissue_consensus_attenuation > 0.0:
                u_tissue = next_uvw_sub["u"][tissue_idx_local]
                u_tissue = _consensus_filter(
                    u_tissue, tissue_local_ei, tissue_consensus_attenuation
                )
                next_uvw_sub["u"][tissue_idx_local] = u_tissue

            # --- Integrate increments — only for nodes inside the crop ---
            # Apply every Δ in next_uvw_sub, including any keys derived from
            # other predictions (e.g. Δu from Δv).  Dropped-but-not-derived
            # keys (e.g. 'a' in u-only experiments) stay at their previous
            # value, which is the intended ablation semantics.
            for _key, _delta in next_uvw_sub.items():
                state[_key][part_nodes] = state[_key][part_nodes] + _delta

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

            # --- Base-clamp override (after state update, before pred_points)
            # Replace state at base needle nodes with ground truth at frame t+1.
            # If we've run past the GT trajectory, leave state alone (no GT).
            if base_node_global_idx is not None:
                t1 = infer_start + step + 1
                if t1 < n_frames:
                    base_t = torch.from_numpy(base_node_global_idx).long()
                    for _bk in base_clamp_keys:
                        if _bk not in frame_tensors:
                            continue
                        gt = frame_tensors[_bk][t1][base_t].float()
                        # Only override if the key is part of the tracked state.
                        if _bk in state:
                            state[_bk][base_t] = gt
                        elif _bk == "coord":
                            # 'coord' always exists in state (initialised above).
                            state["coord"][base_t] = gt
                elif step == 0:
                    print(
                        f"  (base_clamp_mm={base_clamp_mm} active but only "
                        f"{n_frames - infer_start - 1} GT future frames available; "
                        f"the clamp will deactivate when GT runs out.)"
                    )

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
