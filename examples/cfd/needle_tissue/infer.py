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

"""Autoregressive rollout inference for needle-tissue MeshGraphNet.

Loads a trained checkpoint, runs forward from a starting frame, and writes
one VTU file per predicted step plus a .pvd time-series descriptor that
Paraview can open directly as an animation.

Each output VTU contains:
  U_pred, V_pred, A_pred  -- model predictions
  U_gt,   V_gt,   A_gt   -- ground truth (when the GT frame exists)

Usage (from examples/cfd/needle_tissue/):
    LD_LIBRARY_PATH=.venv/lib:$LD_LIBRARY_PATH uv run python infer.py

Override any config value with Hydra syntax, e.g.:
    uv run python infer.py infer_start_frame=159 n_rollout=40
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

from dataset import _precompute_bsms, _sorted_vtu_files
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.models.meshgraphnet import MeshGraphNet
from physicsnemo.models.meshgraphnet.bsms_mgn import BiStrideMeshGraphNet
from physicsnemo.utils import load_checkpoint
from physicsnemo.datapipes.gnn.utils import load_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INPUT_KEYS = ["coord", "u", "v", "a", "evf", "s", "cpress"]
STATIC_PROP_KEYS = ["mat_E", "mat_c10", "mat_density", "mat_fiber", "mat_k1", "mat_k2", "mat_kappa", "mat_nu"]
TARGET_KEYS = ["u", "v", "a"]


def _build_edge_attr(
    coord: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type_onehot: torch.Tensor,
) -> torch.Tensor:
    """Compute edge attributes from current node positions."""
    src, dst = edge_index
    rel_pos = coord[src] - coord[dst]
    edge_len = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)
    return torch.cat([rel_pos, edge_len, edge_type_onehot], dim=-1)


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


def _beam_to_orig(
    pred_state: dict,
    n_tissue: int,
    tissue_idx: torch.Tensor,
    needle_idx: torch.Tensor,
    beam_asgn: torch.Tensor,
    n_nodes_orig: int,
) -> dict:
    """Broadcast beam-reduced predictions back to original full mesh."""
    result = {}
    for key in TARGET_KEYS:
        out = torch.zeros(n_nodes_orig, 3)
        # Tissue nodes: direct copy
        out[tissue_idx] = pred_state[key][:n_tissue]
        # Needle nodes: broadcast each beam node to its cluster
        beam_feats = pred_state[key][n_tissue:]  # (N_beam, 3)
        out[needle_idx] = beam_feats[beam_asgn]
        result[key] = out
    return result


# One-hot type for world (contact/LINE) edges: [needle=0, tissue=0, world=1]
_WORLD_EDGE_TYPE = torch.tensor([[0.0, 0.0, 1.0]])


def _build_world_edges(
    beam_pos: np.ndarray,
    tissue_kdtree: cKDTree,
    contact_radius: float,
    n_tissue: int,
) -> tuple:
    """Build bidirectional world edges by proximity search.

    For each beam node, finds all tissue nodes within *contact_radius* and
    creates a bidirectional edge.  This is called every rollout step so that
    contact reflects the current predicted needle geometry.

    Args:
        beam_pos: (n_beam, 3) current beam-node positions.
        tissue_kdtree: KD-tree built from fixed tissue node positions.
        contact_radius: search radius in mesh units.
        n_tissue: number of tissue nodes (beam node indices = n_tissue + j).

    Returns:
        (edge_index, edge_type_onehot) tensors, may have 0 edges.
    """
    pairs = tissue_kdtree.query_ball_point(beam_pos, contact_radius)
    src_list, dst_list = [], []
    for beam_j, tissue_neighbors in enumerate(pairs):
        for tissue_i in tissue_neighbors:
            src_list.append(n_tissue + beam_j)
            dst_list.append(tissue_i)
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


def _save_vtu_worker(
    out_path: str,
    points: np.ndarray,
    cells_flat: np.ndarray,
    celltypes: np.ndarray,
    cell_data: dict,
    point_data: dict,
) -> None:
    """Reconstruct an UnstructuredGrid from raw arrays and save to disk.

    Runs in a worker process so the main inference loop is not blocked by I/O.
    All arguments must be picklable (numpy arrays / plain dicts).
    """
    import pyvista as pv  # import inside worker process

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
    DistributedManager.initialize()
    dist = DistributedManager()

    data_dir = to_absolute_path(cfg.data_dir)
    stats_dir = to_absolute_path(cfg.stats_dir)

    # ---- Inference-specific config with defaults -------------------------
    # First test frame = n_train + n_val (0-indexed into the full sequence)
    vtu_files = _sorted_vtu_files(data_dir)
    n_frames = len(vtu_files)
    n_pairs = n_frames - 1
    n_train = int(n_pairs * cfg.train_fraction)
    n_val = int(n_pairs * cfg.val_fraction)
    default_start = n_train + n_val  # first test frame

    _raw_start = OmegaConf.select(cfg, "infer_start_frame", default=None)
    infer_start = default_start if (_raw_start is None) else int(_raw_start)
    n_rollout = int(OmegaConf.select(cfg, "n_rollout", default=20))
    n_steps_with_gt = n_frames - 1 - infer_start
    out_dir = to_absolute_path(OmegaConf.select(cfg, "infer_output_dir", default="./inference_output"))
    os.makedirs(out_dir, exist_ok=True)

    print(
        f"Rollout: start_frame={infer_start}, n_rollout={n_rollout}, "
        f"GT available for {n_steps_with_gt} steps, output={out_dir}"
    )

    # ---- Model -----------------------------------------------------------
    if cfg.use_bsms:
        model = BiStrideMeshGraphNet(
            input_dim_nodes=cfg.input_dim_nodes,
            input_dim_edges=cfg.input_dim_edges,
            output_dim=cfg.output_dim,
            processor_size=cfg.processor_size,
            hidden_dim_node_encoder=cfg.hidden_dim_node_encoder,
            hidden_dim_edge_encoder=cfg.hidden_dim_edge_encoder,
            hidden_dim_node_decoder=cfg.hidden_dim_node_decoder,
            hidden_dim_processor=cfg.hidden_dim_processor,
            aggregation=cfg.aggregation,
            num_mesh_levels=cfg.num_bsms_levels,
            bistride_pos_dim=3,
        )
    else:
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
        )

    model = model.to(dist.device)
    load_checkpoint(to_absolute_path(cfg.ckpt_path), models=model, device=dist.device)
    model.eval()

    # ---- Normalisation stats --------------------------------------------
    node_stats = load_json(os.path.join(stats_dir, "node_stats.json"))
    target_stats = load_json(os.path.join(stats_dir, "target_stats.json"))

    # ---- Raw cache (always needed — contains per-frame world edges) ------
    raw_cache_path = os.path.join(data_dir, "preprocessed_cache.pt")
    if not os.path.exists(raw_cache_path):
        raise FileNotFoundError(
            f"{raw_cache_path} not found — run train.py first."
        )
    raw_cache = torch.load(raw_cache_path, weights_only=False)
    if "world_edges" not in raw_cache:
        raise KeyError(
            "Raw cache missing per-frame world edges. Delete "
            f"{raw_cache_path} and re-run train.py to rebuild it."
        )

    # ---- Graph cache (beam-reduced or full) ------------------------------
    beam_old_to_new: Optional[np.ndarray] = None
    if cfg.beam_spacing_mm > 0.0:
        bname = f"beam_cache_b{cfg.beam_spacing_mm:.2g}mm.pt"
        beam_path = os.path.join(data_dir, bname)
        if not os.path.exists(beam_path):
            raise FileNotFoundError(
                f"{beam_path} not found — run train.py first to build the beam cache."
            )
        graph_cache = torch.load(beam_path, weights_only=False)
        if "tissue_node_indices" not in graph_cache:
            raise KeyError(
                "Beam cache is missing mapping metadata. Delete "
                f"{beam_path} and re-run train.py to rebuild it."
            )
        tissue_idx = graph_cache["tissue_node_indices"]
        needle_idx = graph_cache["needle_node_indices"]
        beam_asgn = graph_cache["beam_assignment"]
        n_nodes_orig = int(graph_cache["n_nodes_orig"])
        n_tissue = int(graph_cache["n_tissue"])

        # Build old-to-new mapping for world edge remapping
        old_to_new = np.full(n_nodes_orig, -1, dtype=np.int64)
        for new_i, old_i in enumerate(tissue_idx.numpy()):
            old_to_new[old_i] = new_i
        for j, old_i in enumerate(needle_idx.numpy()):
            old_to_new[old_i] = n_tissue + int(beam_asgn[j])
        beam_old_to_new = old_to_new
    else:
        graph_cache = raw_cache
        tissue_idx = needle_idx = beam_asgn = None
        n_tissue = n_nodes_orig = None

    frame_tensors = graph_cache["frame_tensors"]
    node_props = graph_cache.get("node_props", {})
    edge_index = graph_cache["edge_index"]       # HEX edges (fixed)
    edge_type_onehot = graph_cache["edge_type_onehot"]
    n_nodes = int(edge_index.max().item()) + 1

    # ---- World edge strategy ---------------------------------------------
    # With beam reduction: tissue positions are fixed (Eulerian), so we build
    # a KD-tree once and query it every step with the current predicted beam
    # positions.  Contact radius is estimated from the start-frame world edges.
    #
    # Without beam reduction: fall back to start-frame frozen edges (needle
    # indices are not tracked separately in the no-beam path).
    tissue_kdtree: Optional[cKDTree] = None
    contact_radius: float = 0.0
    dynamic_world_edges: bool = False

    # Map start-frame raw world edges into the reduced graph space
    world_ei_raw, world_et_raw = raw_cache["world_edges"][infer_start]
    if beam_old_to_new is not None and world_ei_raw.shape[1] > 0:
        src_r = torch.from_numpy(beam_old_to_new[world_ei_raw[0].numpy()])
        dst_r = torch.from_numpy(beam_old_to_new[world_ei_raw[1].numpy()])
        valid = (src_r >= 0) & (dst_r >= 0)
        world_ei_start = torch.stack([src_r[valid], dst_r[valid]], dim=0)
    else:
        world_ei_start = world_ei_raw

    if cfg.beam_spacing_mm > 0.0:
        # Tissue positions: indices 0..n_tissue-1, fixed throughout rollout
        tissue_pos_np = frame_tensors["coord"][infer_start][:n_tissue].numpy()
        tissue_kdtree = cKDTree(tissue_pos_np)

        # Estimate contact radius from start-frame world edge lengths
        if world_ei_start.shape[1] > 0:
            coords_np = frame_tensors["coord"][infer_start].numpy()
            dists = np.linalg.norm(
                coords_np[world_ei_start[0].numpy()] - coords_np[world_ei_start[1].numpy()],
                axis=1,
            )
            # 95th-percentile + 20% margin to avoid dropping valid contacts
            contact_radius = float(np.percentile(dists, 95)) * 1.2
        else:
            # No contact at start frame: use 2× beam spacing as fallback
            contact_radius = cfg.beam_spacing_mm * 2.0

        dynamic_world_edges = True
        print(f"  Dynamic world edges: contact_radius = {contact_radius:.4f} mesh units")

    # ---- Full-graph BSMS structures (num_parts=1, HEX topology only) ----
    ms_edges, ms_ids = [], []
    if cfg.use_bsms:
        beam_tag = f"_b{cfg.beam_spacing_mm:.2g}mm" if cfg.beam_spacing_mm > 0.0 else ""
        bsms_path = os.path.join(
            data_dir, f"bsms_cache_p1_l{cfg.num_bsms_levels}{beam_tag}.pt"
        )
        if not os.path.exists(bsms_path):
            print("Computing full-graph BSMS structure for inference...")
            coord_ref = frame_tensors["coord"][0]
            bsms_data = _precompute_bsms(
                [torch.arange(n_nodes)],
                edge_index,          # HEX edges for BSMS topology
                coord_ref,
                n_nodes,
                cfg.num_bsms_levels,
            )
            torch.save(bsms_data, bsms_path)
            print(f"  Saved to {bsms_path}")
        else:
            bsms_data = torch.load(bsms_path, weights_only=False)
        ms_edges_raw, ms_ids_raw = bsms_data[0]
        ms_edges = [e.to(dist.device) for e in ms_edges_raw]
        ms_ids = [ids.to(dist.device) for ids in ms_ids_raw]

    # ---- Initial state ---------------------------------------------------
    # cpress is included in INPUT_KEYS; it stays fixed at the start-frame value
    # during rollout since the model does not predict it.
    state = {k: frame_tensors[k][infer_start].clone().float() for k in INPUT_KEYS}

    # In the beam-reduced graph, indices 0..n_tissue-1 are tissue (Eulerian,
    # positions fixed) and n_tissue.. are beam/needle (Lagrangian, positions update).
    needle_node_mask_reduced = None
    if cfg.beam_spacing_mm > 0.0:
        needle_node_mask_reduced = torch.zeros(n_nodes, dtype=torch.bool)
        needle_node_mask_reduced[n_tissue:] = True

    # Predicted mesh point positions in the original (full) mesh space.
    # Initialised from the GT start frame; needle positions are updated each step.
    ref_mesh_start = pv.read(vtu_files[infer_start])
    pred_points = ref_mesh_start.points.copy()   # (n_nodes_orig, 3)
    needle_idx_np = needle_idx.numpy() if needle_idx is not None else None
    beam_asgn_np = beam_asgn.numpy() if beam_asgn is not None else None

    # Extract fixed mesh topology once (connectivity/cell types are frame-invariant).
    # Workers receive raw numpy arrays to avoid VTK thread-safety concerns.
    topo_cells_flat = ref_mesh_start.cells.copy()
    topo_celltypes = ref_mesh_start.celltypes.copy()
    topo_cell_data = {k: ref_mesh_start.cell_data[k].copy()
                      for k in ref_mesh_start.cell_data.keys()}

    # ---- Rollout ---------------------------------------------------------
    pvd_entries = []
    write_futures = []   # (step, fname, Future) — collected to propagate errors

    with ProcessPoolExecutor(max_workers=32) as write_executor, torch.no_grad():
        for step in range(n_rollout):
            # Build world edges from current predicted beam positions (or use
            # start-frame edges when dynamic world edges are not available).
            if dynamic_world_edges:
                beam_pos_np = state["coord"][n_tissue:].numpy()
                world_ei, world_et = _build_world_edges(
                    beam_pos_np, tissue_kdtree, contact_radius, n_tissue
                )
            else:
                world_ei, world_et = world_ei_start, world_et_raw

            if world_ei.shape[1] > 0:
                full_ei = torch.cat([edge_index, world_ei], dim=1)
                full_et = torch.cat([edge_type_onehot, world_et], dim=0)
            else:
                full_ei, full_et = edge_index, edge_type_onehot

            # Build normalised node features and edge attributes.
            x = _normalize(state, node_props, node_stats).to(dist.device)
            edge_attr = _build_edge_attr(
                state["coord"], full_ei, full_et
            ).to(dist.device)

            graph = Data(
                x=x,
                edge_attr=edge_attr,
                edge_index=full_ei.to(dist.device),
                pos=state["coord"].to(dist.device),
                num_nodes=n_nodes,
            )

            pred = model(graph.x, graph.edge_attr, graph,
                         ms_edges=ms_edges, ms_ids=ms_ids)
            next_uvw = _denorm_target(pred.cpu(), target_stats)

            # Model predicts increments Δu, Δv, Δa.
            # Integrate: u_{t+1} = u_t + Δu, etc.
            delta_u = next_uvw["u"]
            state["u"] = state["u"] + delta_u
            state["v"] = state["v"] + next_uvw["v"]
            state["a"] = state["a"] + next_uvw["a"]

            # Advance Lagrangian (needle) node positions by Δu.
            # Eulerian (tissue) positions stay fixed.
            if needle_node_mask_reduced is not None:
                state["coord"][needle_node_mask_reduced] += (
                    delta_u[needle_node_mask_reduced]
                )

            # Update predicted mesh points for needle nodes (Lagrangian).
            # Broadcast each beam node's predicted position to its cluster members.
            if cfg.beam_spacing_mm > 0.0:
                beam_coords = state["coord"][n_tissue:].numpy()  # (n_beam, 3)
                pred_points[needle_idx_np] = beam_coords[beam_asgn_np]
            else:
                pred_points = state["coord"].numpy().copy()

            # Map predictions back to the original (full) mesh
            if cfg.beam_spacing_mm > 0.0:
                orig_pred = _beam_to_orig(
                    state, n_tissue, tissue_idx, needle_idx, beam_asgn, n_nodes_orig
                )
            else:
                orig_pred = {k: state[k] for k in TARGET_KEYS}

            # Assemble point_data for this step (GT arrays read in main process
            # to avoid racing; the write itself is dispatched to a worker).
            gt_frame_idx = infer_start + step + 1
            has_gt = gt_frame_idx < len(vtu_files)

            point_data = {
                "U_pred": orig_pred["u"].numpy().copy(),
                "V_pred": orig_pred["v"].numpy().copy(),
                "A_pred": orig_pred["a"].numpy().copy(),
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
                topo_cells_flat,
                topo_celltypes,
                topo_cell_data,
                point_data,
            )
            write_futures.append((step, fname, future))
            pvd_entries.append((infer_start + step + 1, fname))
            print(f"  step {step + 1:3d}/{n_rollout}  →  {fname}"
                  + ("  (+ GT)" if has_gt else "  (extrapolating beyond GT)"))

    # Wait for all writes to finish and surface any errors.
    for _step, _fname, future in write_futures:
        future.result()

    pvd_path = _write_pvd(out_dir, pvd_entries)
    print(f"\nDone. Open {pvd_path} in Paraview to animate.")
    print("Tip: in Paraview use Filters → Warp By Vector on U_pred to visualise displacement.")


if __name__ == "__main__":
    main()
