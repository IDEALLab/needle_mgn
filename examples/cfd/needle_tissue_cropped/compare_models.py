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

"""Side-by-side comparison experiments for two cropped-needle MGN variants.

Modes (--mode):
  perturb_propagation
      Sweep transverse Gabor wavelets (multiple centres × wavelengths).  For
      each (centre, λ) and each model, run an unperturbed rollout and a
      perturbed rollout in lockstep for max(K_steps) steps and record the
      L2 energy of the deviation (perturbed state − unperturbed state) on
      needle nodes at steps 1, 5, 10.  Each model runs on its own GPU.

  base_traj
      Run model A's autoregressive rollout, saving the intermediate states.
      At each state evaluate *both* models with one forward pass each;
      report per-step L2 error against ground-truth Δ and L2 disagreement
      between the two predictions.

  error_coherence
      Each model is rolled out independently.  Per-step error e_k =
      pred_k − GT_k is recorded; we then report
        - time correlation:  cos⟨e_k, e_{k+Δ}⟩ averaged over k, for
                              Δ = 1..K-1, as a curve vs Δ;
        - spectral coherence γ²(ω) of u-error along the needle axis,
                              averaged over consecutive step pairs.

  all
      Run all three.  Outputs land in subfolders under --out_dir.

Both --exp_a and --exp_b are experiment directories that contain
`outputs/.hydra/config.yaml`, `checkpoints/`, and `stats/`.  Dataset flags
are read from each config.yaml so the two models can use different input
schemes (standard / mgn-paper) without manual configuration.

Parallelism: where the workload is naturally per-model (perturb_propagation,
error_coherence), one worker process per GPU runs in parallel.  For
base_traj both models execute sequentially on cuda:0 since the fiber
model has to wait for the base model's rollout states.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import NeedleTissueDataset  # noqa: E402
from freq_response import (  # noqa: E402
    _build_model,
    _dataset_kwargs_from_cfg,
    _gabor_wavelet,
    _needle_axis_and_transverse,
    _stat_to_tensor,
)

from physicsnemo.utils import load_checkpoint  # noqa: E402


# ===========================================================================
# Rollout engine — one per (model, dataset, cfg).  Mirrors the
# MGNTrainer._apply_rollout_step / _rebuild_edges logic but free-standing.
# ===========================================================================

class RolloutEngine:
    def __init__(self, model, dataset: NeedleTissueDataset, cfg, device: torch.device,
                 world_edge_radius: float = 1.2):
        self.model = model
        self.dataset = dataset
        self.cfg = cfg
        self.device = device
        self.world_edge_radius = float(world_edge_radius)
        self.mgn_paper = bool(cfg.get("mgn_paper_features", False))
        self.mgn_include_evf = bool(cfg.get("mgn_include_evf", False))
        self.mgn_include_mat_fiber = bool(cfg.get("mgn_include_mat_fiber", False))
        self.mgn_include_prev_v = bool(cfg.get("mgn_include_prev_v", False))
        self._build_stats()

    # ---- Stats / column mappings ------------------------------------------

    def _build_stats(self):
        ds = self.dataset
        node_stats = ds._node_stats
        target_stats = ds._target_stats
        device = self.device

        def _t(stat) -> torch.Tensor:
            if isinstance(stat, torch.Tensor):
                return stat.to(device).float().view(1, -1)
            return torch.tensor(stat, device=device).float().view(1, -1)

        # Target offsets in pred.
        self.tgt_offsets: Dict[str, Tuple[int, int]] = {}
        off = 0
        for k, d in zip(ds.TARGET_KEYS, ds.TARGET_DIMS):
            self.tgt_offsets[k] = (off, off + d)
            off += d
        self.output_dim = off

        self._updates: List[Tuple[Tuple[int, int], Tuple[int, int], torch.Tensor, torch.Tensor]] = []
        if not self.mgn_paper:
            in_offsets: Dict[str, Tuple[int, int]] = {}
            off = 0
            for k, d in zip(ds.INPUT_KEYS, ds.INPUT_DIMS):
                in_offsets[k] = (off, off + d)
                off += d
            for tk in ds.TARGET_KEYS:
                if tk not in in_offsets:
                    continue
                t_lo, t_hi = self.tgt_offsets[tk]
                i_lo, i_hi = in_offsets[tk]
                ratio = _t(target_stats[f"{tk}_std"]) / _t(node_stats[f"{tk}_std"])
                bias = _t(target_stats[f"{tk}_mean"]) / _t(node_stats[f"{tk}_std"])
                self._updates.append(((t_lo, t_hi), (i_lo, i_hi), ratio, bias))
            if "u" in self.tgt_offsets and "coord" in in_offsets:
                t_lo, t_hi = self.tgt_offsets["u"]
                c_lo, c_hi = in_offsets["coord"]
                ratio = _t(target_stats["u_std"]) / _t(node_stats["coord_std"])
                bias = _t(target_stats["u_mean"]) / _t(node_stats["coord_std"])
                self._updates.append(((t_lo, t_hi), (c_lo, c_hi), ratio, bias))
        else:
            off_x = 2  # after node_type
            if self.mgn_include_evf:
                if "evf" in self.tgt_offsets:
                    t_lo, t_hi = self.tgt_offsets["evf"]
                    ratio = _t(target_stats["evf_std"]) / _t(node_stats["evf_std"])
                    bias = _t(target_stats["evf_mean"]) / _t(node_stats["evf_std"])
                    self._updates.append(((t_lo, t_hi), (off_x, off_x + 1), ratio, bias))
                off_x += 1
            if self.mgn_include_mat_fiber:
                off_x += 3
            if self.mgn_include_prev_v:
                if "v" in self.tgt_offsets:
                    t_lo, t_hi = self.tgt_offsets["v"]
                    ratio = _t(target_stats["v_std"]) / _t(node_stats["v_std"])
                    bias = _t(target_stats["v_mean"]) / _t(node_stats["v_std"])
                    self._updates.append(((t_lo, t_hi), (off_x, off_x + 3), ratio, bias))
                off_x += 3

        if "u" not in self.tgt_offsets:
            raise ValueError("Rollout requires 'u' in TARGET_KEYS to advance geometry.")
        self.u_t_mean = _t(target_stats["u_mean"])
        self.u_t_std = _t(target_stats["u_std"])

        if "v" in self.tgt_offsets:
            t_lo, t_hi = self.tgt_offsets["v"]
            ratio = _t(target_stats["v_std"]) / _t(node_stats["v_std"])
            bias = _t(target_stats["v_mean"]) / _t(node_stats["v_std"])
            self._v_update = ((t_lo, t_hi), ratio, bias)
        else:
            self._v_update = None

    # ---- Edge rebuild (matches trainer) -----------------------------------

    def _rebuild_edges(self, pos: torch.Tensor, edge_index: torch.Tensor,
                       edge_attr: torch.Tensor, is_needle: torch.Tensor):
        type_oh = edge_attr[:, -3:]
        hex_mask = ~(type_oh[:, 2] > 0.5)
        extra_dim = edge_attr.shape[1] - 7
        if extra_dim < 0:
            raise RuntimeError(f"edge_attr has {edge_attr.shape[1]} cols; expected 7 or 11.")
        hex_ei = edge_index[:, hex_mask]
        hex_oh = type_oh[hex_mask]
        hex_extra = edge_attr[hex_mask, 4:4 + extra_dim] if extra_dim > 0 else None

        pos_np = pos.detach().cpu().numpy()
        needle_local = torch.nonzero(is_needle, as_tuple=False).squeeze(-1)
        tissue_local = torch.nonzero(~is_needle, as_tuple=False).squeeze(-1)
        new_ei, new_oh, new_extra = hex_ei, hex_oh, hex_extra
        if needle_local.numel() > 0 and tissue_local.numel() > 0:
            tissue_pts = pos_np[tissue_local.cpu().numpy()]
            needle_pts = pos_np[needle_local.cpu().numpy()]
            tree = cKDTree(tissue_pts)
            neigh = tree.query_ball_point(needle_pts, r=self.world_edge_radius)
            src_l, dst_l = [], []
            n_np = needle_local.cpu().numpy()
            t_np = tissue_local.cpu().numpy()
            for i, nbrs in enumerate(neigh):
                if not nbrs:
                    continue
                ng = int(n_np[i])
                for j in nbrs:
                    src_l.append(ng)
                    dst_l.append(int(t_np[j]))
            if src_l:
                src_t = torch.tensor(src_l, dtype=torch.long, device=pos.device)
                dst_t = torch.tensor(dst_l, dtype=torch.long, device=pos.device)
                world_ei = torch.stack(
                    [torch.cat([src_t, dst_t]), torch.cat([dst_t, src_t])], dim=0
                )
                world_oh = torch.zeros(world_ei.shape[1], 3, dtype=hex_oh.dtype, device=pos.device)
                world_oh[:, 2] = 1.0
                new_ei = torch.cat([hex_ei, world_ei], dim=1)
                new_oh = torch.cat([hex_oh, world_oh], dim=0)
                if extra_dim > 0:
                    world_extra = torch.zeros(
                        world_ei.shape[1], extra_dim,
                        dtype=hex_extra.dtype, device=pos.device,
                    )
                    new_extra = torch.cat([hex_extra, world_extra], dim=0)

        src, dst = new_ei
        rel_pos = pos[src] - pos[dst]
        edge_len = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)
        if extra_dim > 0:
            new_attr = torch.cat([rel_pos, edge_len, new_extra, new_oh], dim=-1)
        else:
            new_attr = torch.cat([rel_pos, edge_len, new_oh], dim=-1)
        return new_ei, new_attr

    # ---- One rollout step (state update from a fresh prediction) ----------

    def step(self, graph: Data, pred: torch.Tensor) -> Data:
        pred = pred.float()
        x = graph.x.float()
        new_x = x.clone()
        for (t_lo, t_hi), (i_lo, i_hi), ratio, bias in self._updates:
            new_x[:, i_lo:i_hi] = new_x[:, i_lo:i_hi] + ratio * pred[:, t_lo:t_hi] + bias

        u_lo, u_hi = self.tgt_offsets["u"]
        u_raw = pred[:, u_lo:u_hi] * self.u_t_std + self.u_t_mean
        new_pos = graph.pos + u_raw

        new_nv = graph.node_velocity
        if self._v_update is not None:
            (v_lo, v_hi), v_ratio, v_bias = self._v_update
            new_nv = graph.node_velocity + v_ratio * pred[:, v_lo:v_hi] + v_bias

        new_ei, new_attr = self._rebuild_edges(new_pos, graph.edge_index, graph.edge_attr, graph.is_needle)
        graph.x = new_x
        graph.pos = new_pos
        graph.edge_index = new_ei
        graph.edge_attr = new_attr
        graph.node_velocity = new_nv
        return graph

    # ---- Forward (no grad) ------------------------------------------------

    @torch.no_grad()
    def forward(self, graph: Data) -> torch.Tensor:
        return self.model(graph.x, graph.edge_attr, graph)

    # ---- Rollout: returns list of normalised preds per step ---------------

    @torch.no_grad()
    def rollout(self, graph: Data, n_steps: int) -> List[torch.Tensor]:
        graph = graph.clone()
        preds: List[torch.Tensor] = []
        for k in range(n_steps):
            pred = self.forward(graph)
            preds.append(pred.detach())
            if k < n_steps - 1:
                self.step(graph, pred)
        return preds

    # ---- Rollout with state snapshots -------------------------------------

    @torch.no_grad()
    def rollout_with_states(self, graph: Data, n_steps: int) -> Tuple[List[torch.Tensor], List[Data]]:
        """Return (preds, states) where states[k] is the graph the model saw
        at step k (before applying its k-th prediction).  states[0] is the
        original graph (cloned)."""
        g = graph.clone()
        preds: List[torch.Tensor] = []
        states: List[Data] = [g.clone()]
        for k in range(n_steps):
            pred = self.forward(g)
            preds.append(pred.detach())
            self.step(g, pred)
            if k < n_steps - 1:
                states.append(g.clone())
        return preds, states


# ===========================================================================
# Experiment loading.
# ===========================================================================

def load_experiment(exp_dir: str, data_dir: str, device: torch.device) -> Tuple[OmegaConf, NeedleTissueDataset, RolloutEngine]:
    cfg_path = os.path.join(exp_dir, "outputs", ".hydra", "config.yaml")
    cfg = OmegaConf.load(cfg_path)
    ckpt_path = os.path.join(exp_dir, "checkpoints")
    stats_dir = os.path.join(exp_dir, "stats")
    ds_kwargs = _dataset_kwargs_from_cfg(cfg, data_dir, stats_dir)
    # We need future_deltas for ground-truth comparison; the caller can
    # bump multistep_K later by re-instantiating if a longer horizon is
    # needed.  Default to multistep_K=1 here; experiment-specific code
    # overrides.
    dataset = NeedleTissueDataset(split="test", **ds_kwargs)
    model = _build_model(cfg, dataset, device)
    load_checkpoint(ckpt_path, models=model, device=device)
    model.eval()
    engine = RolloutEngine(model, dataset, cfg, device,
                           world_edge_radius=float(cfg.get("world_edge_radius", 1.2)))
    return cfg, dataset, engine


def _reload_dataset_with_K(exp_dir: str, data_dir: str, K: int) -> NeedleTissueDataset:
    cfg = OmegaConf.load(os.path.join(exp_dir, "outputs", ".hydra", "config.yaml"))
    stats_dir = os.path.join(exp_dir, "stats")
    kw = _dataset_kwargs_from_cfg(cfg, data_dir, stats_dir)
    kw["multistep_K"] = int(K)
    return NeedleTissueDataset(split="test", **kw)


# ===========================================================================
# Common helpers.
# ===========================================================================

def _denorm_pred_u(pred_norm: torch.Tensor, engine: RolloutEngine) -> torch.Tensor:
    u_lo, u_hi = engine.tgt_offsets["u"]
    return pred_norm[:, u_lo:u_hi] * engine.u_t_std + engine.u_t_mean


def _translate_pred(pred_src: torch.Tensor, engine_src: RolloutEngine,
                    engine_dst: RolloutEngine) -> torch.Tensor:
    """Re-normalise a prediction from engine_src's target scheme into
    engine_dst's target scheme.

    For keys in engine_dst.TARGET_KEYS that engine_src doesn't predict, the
    output's normalised value is set so that engine_dst.step adds raw Δ=0
    for that key (i.e. ``pred_dst[:, slice] = -mean_dst / std_dst`` makes
    ``pred_dst * std_dst + mean_dst = 0``).
    """
    device = pred_src.device
    out = torch.zeros(pred_src.shape[0], engine_dst.output_dim,
                      dtype=pred_src.dtype, device=device)
    ts_src = engine_src.dataset._target_stats
    ts_dst = engine_dst.dataset._target_stats
    for key, (d_lo, d_hi) in engine_dst.tgt_offsets.items():
        mean_d = _stat_to_tensor(ts_dst[f"{key}_mean"], device).view(1, -1)
        std_d = _stat_to_tensor(ts_dst[f"{key}_std"], device).view(1, -1)
        if key in engine_src.tgt_offsets:
            s_lo, s_hi = engine_src.tgt_offsets[key]
            mean_s = _stat_to_tensor(ts_src[f"{key}_mean"], device).view(1, -1)
            std_s = _stat_to_tensor(ts_src[f"{key}_std"], device).view(1, -1)
            raw = pred_src[:, s_lo:s_hi] * std_s + mean_s
            out[:, d_lo:d_hi] = (raw - mean_d) / std_d.clamp(min=1e-30)
        else:
            # No source prediction for this key → encode "zero raw Δ".
            out[:, d_lo:d_hi] = -mean_d / std_d.clamp(min=1e-30)
    return out


def _denorm_per_key(pred_norm: torch.Tensor, engine: RolloutEngine) -> Dict[str, torch.Tensor]:
    """Return raw Δ per target key.  ``pred - other_pred`` denormalises with
    only the std (means cancel), but for a single pred we need mean+std."""
    ts = engine.dataset._target_stats
    out: Dict[str, torch.Tensor] = {}
    for k, (lo, hi) in engine.tgt_offsets.items():
        mean = _stat_to_tensor(ts[f"{k}_mean"], pred_norm.device).view(1, -1)
        std = _stat_to_tensor(ts[f"{k}_std"], pred_norm.device).view(1, -1)
        out[k] = pred_norm[:, lo:hi] * std + mean
    return out


def _write_csv(rows: List[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, "w") as f:
        f.write(",".join(fields) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in fields) + "\n")


def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ===========================================================================
# Experiment 1: perturbation propagation.
# ===========================================================================

def _worker_perturb(rank: int, exp_dir: str, data_dir: str, sample_idx: int,
                    n_centres: int, wavelengths: List[float], amplitude: float,
                    sigma_n_lambda: float, step_horizons: List[int],
                    out_path: str, label: str):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    print(f"[{label}|cuda:{rank}] loading ...", flush=True)
    cfg, dataset, engine = load_experiment(exp_dir, data_dir, device)
    sample_idx = max(0, min(sample_idx, len(dataset) - 1))
    base_graph = dataset[sample_idx].to(device)

    needle_local = torch.nonzero(base_graph.is_needle, as_tuple=False).squeeze(-1)
    axis, transverse, axial_coord = _needle_axis_and_transverse(base_graph.pos[needle_local])
    s_min, s_max = float(axial_coord.min()), float(axial_coord.max())
    centres = np.linspace(s_min, s_max, n_centres).tolist()

    max_steps = max(step_horizons)
    rows = []
    for c_i, centre in enumerate(centres):
        for lam in wavelengths:
            scalar = _gabor_wavelet(
                axial_coord, centre_mm=float(centre), wavelength_mm=float(lam),
                sigma_n_lambda=sigma_n_lambda, amplitude=amplitude,
            )
            delta_u = torch.zeros((base_graph.x.shape[0], 3), dtype=torch.float32, device=device)
            delta_u[needle_local] = scalar.unsqueeze(-1) * transverse.view(1, 3)

            # Build perturbed initial state (in graph form).
            from freq_response import _apply_perturbation  # noqa: E402 (deferred for clarity)
            g_unp = base_graph.clone()
            g_pert = _apply_perturbation(base_graph, delta_u, dataset, cfg).to(device)

            # Walk both rollouts in lockstep, recording the deviation in raw
            # needle positions at the requested step horizons.
            input_energy = float((delta_u[needle_local] ** 2).sum().item())
            wanted = set(step_horizons)
            captured: Dict[int, float] = {}
            for k in range(1, max_steps + 1):
                with torch.no_grad():
                    p_unp = engine.forward(g_unp)
                    p_pert = engine.forward(g_pert)
                engine.step(g_unp, p_unp)
                engine.step(g_pert, p_pert)
                if k in wanted:
                    dev = g_pert.pos[needle_local] - g_unp.pos[needle_local]
                    captured[k] = float((dev ** 2).sum().item())
            for k in step_horizons:
                rows.append({
                    "label": label,
                    "centre_idx": c_i,
                    "centre_mm": float(centre),
                    "wavelength_mm": float(lam),
                    "step": int(k),
                    "input_energy": input_energy,
                    "deviation_energy": captured[k],
                    "energy_gain": captured[k] / max(input_energy, 1e-30),
                })
        print(f"[{label}|cuda:{rank}] centre {c_i+1}/{n_centres} done", flush=True)
    _write_csv(rows, out_path)
    print(f"[{label}|cuda:{rank}] wrote {len(rows)} rows → {out_path}", flush=True)


def exp_perturb_propagation(args, out_subdir: str):
    out_dir = os.path.join(args.out_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    label_a = args.label_a
    label_b = args.label_b
    csv_a = os.path.join(out_dir, f"{label_a}.csv")
    csv_b = os.path.join(out_dir, f"{label_b}.csv")

    wavelengths = np.geomspace(args.lam_min, args.lam_max, args.n_lams).tolist()
    step_horizons = [int(s) for s in args.step_horizons.split(",") if s]

    ctx = mp.get_context("spawn")
    procs = []
    for rank, (exp_dir, csv_path, label) in enumerate([
        (args.exp_a, csv_a, label_a),
        (args.exp_b, csv_b, label_b),
    ]):
        p = ctx.Process(
            target=_worker_perturb,
            args=(rank, exp_dir, args.data_dir, args.sample_idx,
                  args.n_centres, wavelengths, args.amplitude,
                  args.sigma_n_lambda, step_horizons, csv_path, label),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"perturb worker exited {p.exitcode}")

    _plot_perturb(out_dir, csv_a, csv_b, label_a, label_b, wavelengths, step_horizons)


def _plot_perturb(out_dir, csv_a, csv_b, label_a, label_b, wavelengths, step_horizons):
    plt = _matplotlib()

    def _read(path):
        with open(path) as f:
            header = f.readline().strip().split(",")
            data = [dict(zip(header, line.strip().split(","))) for line in f if line.strip()]
        for d in data:
            for k in ("centre_mm", "wavelength_mm", "input_energy",
                      "deviation_energy", "energy_gain"):
                d[k] = float(d[k])
            d["centre_idx"] = int(d["centre_idx"])
            d["step"] = int(d["step"])
        return data

    def _grid(data):
        # Aggregate gain over centres for each (step, λ).  Median.
        steps = step_horizons
        lams = wavelengths
        G = np.full((len(steps), len(lams)), np.nan)
        for si, s in enumerate(steps):
            for li, lam in enumerate(lams):
                vals = [r["energy_gain"] for r in data if r["step"] == s and abs(r["wavelength_mm"] - lam) < 1e-9]
                if vals:
                    G[si, li] = np.median(vals)
        return G

    da = _read(csv_a)
    db = _read(csv_b)
    Ga = _grid(da)
    Gb = _grid(db)
    # Compare on a common log colour scale.
    finite = np.concatenate([Ga[np.isfinite(Ga)], Gb[np.isfinite(Gb)]])
    if finite.size == 0:
        print("(no finite energies; skipping plot)", flush=True)
        return
    vmin, vmax = float(np.percentile(finite, 2)), float(np.percentile(finite, 98))
    vmin = max(vmin, 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    for ax, G, ttl in [(axes[0], Ga, label_a), (axes[1], Gb, label_b)]:
        im = ax.imshow(G, aspect="auto", origin="lower",
                       extent=[np.log10(wavelengths[0]), np.log10(wavelengths[-1]),
                               -0.5, len(step_horizons) - 0.5],
                       cmap="magma", norm=__import__("matplotlib").colors.LogNorm(vmin=vmin, vmax=vmax))
        ax.set_yticks(range(len(step_horizons)))
        ax.set_yticklabels([str(s) for s in step_horizons])
        ax.set_xlabel("log10 wavelength (mm)")
        ax.set_ylabel("rollout steps")
        ax.set_title(f"Energy gain — {ttl}")
        plt.colorbar(im, ax=ax)
    ratio = Gb / np.where(Ga > 0, Ga, np.nan)
    rmax = float(np.nanpercentile(np.abs(np.log10(ratio)), 98))
    norm = __import__("matplotlib").colors.LogNorm(vmin=10 ** -rmax, vmax=10 ** rmax)
    im2 = axes[2].imshow(ratio, aspect="auto", origin="lower",
                         extent=[np.log10(wavelengths[0]), np.log10(wavelengths[-1]),
                                 -0.5, len(step_horizons) - 0.5],
                         cmap="RdBu_r", norm=norm)
    axes[2].set_yticks(range(len(step_horizons)))
    axes[2].set_yticklabels([str(s) for s in step_horizons])
    axes[2].set_xlabel("log10 wavelength (mm)")
    axes[2].set_ylabel("rollout steps")
    axes[2].set_title(f"Gain ratio ({label_b} / {label_a})")
    plt.colorbar(im2, ax=axes[2])
    fig.suptitle("Perturbation-energy propagation (median over centres)")
    fig.tight_layout()
    out_svg = os.path.join(out_dir, "energy_heatmaps.svg")
    fig.savefig(out_svg)
    plt.close(fig)
    print(f"wrote → {out_svg}", flush=True)


# ===========================================================================
# Experiment 2: base trajectory; both models evaluated at each saved state.
# ===========================================================================

def exp_base_trajectory(args, out_subdir: str):
    """Runs engine_a on cuda:0 and engine_b on cuda:1 in lockstep so two
    24 GB cards (e.g. RTX 3090) can hold one ~20 GB model each.  Per step,
    the small (n_sub × output_dim) prediction tensors are moved between
    devices; the GNN forward / step computations stay device-local."""
    out_dir = os.path.join(args.out_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    device_a = torch.device("cuda:0")
    device_b = torch.device("cuda:1") if torch.cuda.device_count() >= 2 else device_a

    K = int(args.n_rollout)
    label_base = args.label_a
    label_other = args.label_b

    print(f"[base_traj] loading {label_base} on {device_a} ...", flush=True)
    _cfg_a, dataset_a, engine_a = load_experiment(args.exp_a, args.data_dir, device_a)
    dataset_a = _reload_dataset_with_K(args.exp_a, args.data_dir, K)
    engine_a.dataset = dataset_a

    print(f"[base_traj] loading {label_other} on {device_b} ...", flush=True)
    _cfg_b, dataset_b, engine_b = load_experiment(args.exp_b, args.data_dir, device_b)
    dataset_b = _reload_dataset_with_K(args.exp_b, args.data_dir, K)
    engine_b.dataset = dataset_b

    sample_idx = max(0, min(args.sample_idx, len(dataset_a) - 1))
    graph_a = dataset_a[sample_idx].to(device_a)
    graph_b = dataset_b[sample_idx].to(device_b)
    # Ground truth increments are scheme-independent in raw units; pick from
    # either dataset's future_deltas after denormalising with that dataset's
    # target stats.  We use dataset_a's future_deltas + engine_a for that.
    future_deltas_a = graph_a.future_deltas  # (n_sub, K, output_dim_a)

    shared_keys = [k for k in engine_a.tgt_offsets.keys() if k in engine_b.tgt_offsets]
    if not shared_keys:
        raise RuntimeError(
            "base_traj needs at least one shared TARGET_KEY between the two models; "
            f"got A={list(engine_a.tgt_offsets)} B={list(engine_b.tgt_offsets)}."
        )
    print(f"[base_traj] shared target keys: {shared_keys}", flush=True)

    # Walk through K rollout steps.  At each step:
    #   * engine_a (the base model) predicts on its current state, advancing
    #     itself.  Its predictions define the "shared physical trajectory".
    #   * engine_b is advanced by engine_a's prediction (translated through
    #     each engine's own normalisation), so engine_b's input graph at
    #     step k corresponds to the *same physical state* engine_a is at.
    #     We then evaluate engine_b.forward on that input.
    g_a = graph_a.clone()
    g_b = graph_b.clone()
    # Per-key node-keep mask from the dataset's loss_mask.  Under
    # mgn_kinematic_needle_only=true, tissue-u / tissue-v / tissue-a are
    # zeroed in the training loss, so the model's predictions on those nodes
    # are essentially untrained noise — computing ‖pred-GT‖ over all nodes
    # pins the metric at the noise floor.  We mirror the training mask here.
    loss_mask_a = getattr(graph_a, "loss_mask", None)
    key_keep: Dict[str, Optional[torch.Tensor]] = {}
    for key in shared_keys:
        if loss_mask_a is None:
            key_keep[key] = None
            continue
        t_lo, _ = engine_a.tgt_offsets[key]
        keep = loss_mask_a[:, t_lo] > 0.5
        key_keep[key] = keep if not keep.all() else None  # None = use all nodes
    rows = []
    for k in range(K):
        # Predictions on each engine's current input (which represent the
        # same physical state by construction).
        with torch.no_grad():
            pred_a = engine_a.forward(g_a)
            pred_b = engine_b.forward(g_b)

        # All scalar metrics are computed on device_a; pred_b is small
        # (n_sub × output_dim) so the cross-device transfer is cheap.
        gt_a_raw = _denorm_per_key(future_deltas_a[:, k, :], engine_a)
        pred_a_raw = _denorm_per_key(pred_a, engine_a)
        pred_b_raw = _denorm_per_key(pred_b, engine_b)

        row = {"step": k}
        for key in shared_keys:
            pa = pred_a_raw[key]
            pb_on_a = pred_b_raw[key].to(device_a)
            gt = gt_a_raw[key]
            keep = key_keep[key]
            if keep is not None:
                pa = pa[keep]
                pb_on_a = pb_on_a[keep]
                gt = gt[keep]
            err_a = float(torch.linalg.norm(pa - gt).item())
            err_b = float(torch.linalg.norm(pb_on_a - gt).item())
            disagree = float(torch.linalg.norm(pa - pb_on_a).item())
            gt_norm = float(torch.linalg.norm(gt).item())
            row[f"err_{key}_{label_base}"] = err_a
            row[f"err_{key}_{label_other}"] = err_b
            row[f"disagree_{key}"] = disagree
            row[f"gt_norm_{key}"] = gt_norm
            row[f"n_keep_{key}"] = int(keep.sum().item()) if keep is not None else int(pa.shape[0])
        rows.append(row)

        if k < K - 1:
            # Advance both engines along engine_a's prediction.  engine_b's
            # state lives on device_b; translate engine_a's pred to engine_b's
            # scheme *on device_b* by moving pred_a there before translating.
            engine_a.step(g_a, pred_a)
            pred_a_on_b = pred_a.to(device_b)
            pred_a_in_b = _translate_pred(pred_a_on_b, engine_a, engine_b)
            engine_b.step(g_b, pred_a_in_b)

    csv_path = os.path.join(out_dir, "per_step.csv")
    _write_csv(rows, csv_path)
    print(f"[base_traj] wrote {csv_path}", flush=True)
    _plot_base_traj(out_dir, csv_path, shared_keys, label_base, label_other)


def _plot_base_traj(out_dir, csv_path, target_keys, label_base, label_other):
    plt = _matplotlib()
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
        data = [dict(zip(header, line.strip().split(","))) for line in f if line.strip()]
    for d in data:
        for k, v in d.items():
            try:
                d[k] = float(v)
            except ValueError:
                pass
        d["step"] = int(d["step"])
    steps = [d["step"] for d in data]

    n_keys = len(target_keys)
    fig, axes = plt.subplots(1, n_keys, figsize=(5 * n_keys, 4.5), squeeze=False)
    for i, key in enumerate(target_keys):
        ax = axes[0, i]
        ea = [d[f"err_{key}_{label_base}"] for d in data]
        eb = [d[f"err_{key}_{label_other}"] for d in data]
        dis = [d[f"disagree_{key}"] for d in data]
        gt = [d[f"gt_norm_{key}"] for d in data]
        ax.plot(steps, ea, "o-", label=f"{label_base} error vs GT", color="tab:blue")
        ax.plot(steps, eb, "s-", label=f"{label_other} error vs GT", color="tab:red")
        ax.plot(steps, dis, "x--", label="A↔B disagreement", color="tab:green")
        ax.plot(steps, gt, ":", label="‖GT‖", color="k", alpha=0.5)
        ax.set_xlabel("rollout step (from base trajectory)")
        ax.set_ylabel(f"‖·‖ ({key}, raw units)")
        ax.set_title(f"target = {key}")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Both models evaluated on {label_base}'s rollout trajectory")
    fig.tight_layout()
    out_svg = os.path.join(out_dir, "error_vs_step.svg")
    fig.savefig(out_svg)
    plt.close(fig)
    print(f"wrote → {out_svg}", flush=True)


# ===========================================================================
# Experiment 3: error coherence across rollout steps.
# ===========================================================================

def _spectral_coherence(errs_axial: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Given (K, n_grid) per-step error sampled on a uniform axial grid,
    return (freqs, coherence) using K-1 consecutive step pairs as the
    averaging ensemble.

    γ²(ω) = |<X(ω)·Y*(ω)>|² / (<|X(ω)|²> · <|Y(ω)|²>),
    with X = e_k, Y = e_{k+1}.
    """
    n_grid = errs_axial.shape[1]
    E = np.fft.rfft(errs_axial, axis=1)  # (K, F)
    Ek = E[:-1]
    Ek1 = E[1:]
    Sxy = np.mean(Ek * np.conj(Ek1), axis=0)
    Sxx = np.mean(np.abs(Ek) ** 2, axis=0)
    Syy = np.mean(np.abs(Ek1) ** 2, axis=0)
    coh = np.abs(Sxy) ** 2 / np.maximum(Sxx * Syy, 1e-30)
    return coh, np.fft.rfftfreq(n_grid, d=1.0)  # caller scales freqs to mm⁻¹


def _worker_coherence(rank: int, exp_dir: str, data_dir: str,
                      n_steps: int, out_path: str, label: str):
    """Iterates over every sample in the test split and pools:
      * time-correlation cosine similarities  →  (K-1) lists, one per lag,
        each of length Σ_samples (K - Δ).
      * spectral cross-products on a *normalised* axial grid (each sample's
        needle length rescaled to [0, 1] before FFT) so freq bins line up
        across samples; coherence γ² is then computed from the pooled
        ⟨S_xy⟩, ⟨S_xx⟩, ⟨S_yy⟩ over all step pairs × samples.
    """
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    print(f"[{label}|cuda:{rank}] loading ...", flush=True)
    _, _, engine = load_experiment(exp_dir, data_dir, device)
    dataset = _reload_dataset_with_K(exp_dir, data_dir, n_steps)
    engine.dataset = dataset

    n_samples = len(dataset)
    n_grid = 512

    time_sims: Dict[int, List[float]] = {d: [] for d in range(1, n_steps)}
    Sxy_acc: Optional[np.ndarray] = None
    Sxx_acc: Optional[np.ndarray] = None
    Syy_acc: Optional[np.ndarray] = None
    n_pairs_total = 0
    n_samples_used = 0

    for sample_idx in range(n_samples):
        try:
            graph = dataset[sample_idx].to(device)
        except Exception as e:
            print(f"[{label}|cuda:{rank}] sample {sample_idx} load failed: {e}", flush=True)
            continue
        future = graph.future_deltas
        preds, _ = engine.rollout_with_states(graph, n_steps)

        needle_local = torch.nonzero(graph.is_needle, as_tuple=False).squeeze(-1)
        if needle_local.numel() < 8:
            del graph, preds
            continue
        _axis, transverse, axial = _needle_axis_and_transverse(graph.pos[needle_local])

        err_vecs: List[torch.Tensor] = []
        errs_proj: List[torch.Tensor] = []
        for k in range(n_steps):
            pred_raw = _denorm_pred_u(preds[k], engine)
            gt_raw = _denorm_per_key(future[:, k, :], engine)["u"]
            diff = (pred_raw - gt_raw)[needle_local]
            err_vecs.append(diff)
            errs_proj.append((diff * transverse.view(1, 3)).sum(-1))

        # Time correlation: pool cosine sims per lag.
        E = torch.stack(err_vecs, dim=0).reshape(n_steps, -1)
        norms = torch.linalg.norm(E, dim=1).clamp(min=1e-30)
        Enrm = E / norms.unsqueeze(-1)
        for delta in range(1, n_steps):
            sims = (Enrm[:-delta] * Enrm[delta:]).sum(-1).cpu().numpy()
            time_sims[delta].extend(sims.tolist())

        # Spectral cross-products on a per-sample normalised axial grid.
        s = axial.detach().cpu().numpy()
        order = np.argsort(s)
        s_sorted = s[order]
        s_min, s_max = float(s_sorted[0]), float(s_sorted[-1])
        L_sample = s_max - s_min
        if L_sample <= 0.0:
            del graph, preds
            continue
        s_norm = (s_sorted - s_min) / L_sample
        s_uni = np.linspace(0.0, 1.0, n_grid)
        grids = np.zeros((n_steps, n_grid), dtype=np.float64)
        for k, ep in enumerate(errs_proj):
            grids[k] = np.interp(s_uni, s_norm, ep.detach().cpu().numpy()[order])
        Efft = np.fft.rfft(grids, axis=1)
        Ek = Efft[:-1]
        Ek1 = Efft[1:]
        if Sxy_acc is None:
            F = Ek.shape[1]
            Sxy_acc = np.zeros(F, dtype=np.complex128)
            Sxx_acc = np.zeros(F, dtype=np.float64)
            Syy_acc = np.zeros(F, dtype=np.float64)
        Sxy_acc += np.sum(Ek * np.conj(Ek1), axis=0)
        Sxx_acc += np.sum(np.abs(Ek) ** 2, axis=0)
        Syy_acc += np.sum(np.abs(Ek1) ** 2, axis=0)
        n_pairs_total += int(Ek.shape[0])
        n_samples_used += 1

        del graph, preds, E, Enrm, err_vecs, errs_proj
        if (sample_idx + 1) % 5 == 0:
            print(f"[{label}|cuda:{rank}] {sample_idx + 1}/{n_samples} samples processed", flush=True)

    print(f"[{label}|cuda:{rank}] used {n_samples_used}/{n_samples} samples, "
          f"{n_pairs_total} spectral step-pairs", flush=True)

    time_rows = []
    for delta in range(1, n_steps):
        sims = np.array(time_sims[delta])
        time_rows.append({
            "label": label,
            "lag": int(delta),
            "cos_mean": float(np.mean(sims)) if sims.size else float("nan"),
            "cos_std": float(np.std(sims)) if sims.size else float("nan"),
            "n_pairs": int(sims.size),
        })

    if Sxy_acc is None:
        coh = np.zeros(n_grid // 2 + 1)
        freqs_unit = np.zeros(n_grid // 2 + 1)
    else:
        coh = np.abs(Sxy_acc) ** 2 / np.maximum(Sxx_acc * Syy_acc, 1e-30)
        # rfftfreq over [0,1] with n_grid samples → bin spacing 1/(n_grid-1)
        # which on the normalised axis is "cycles per full needle length".
        # Multiply by the bin index to recover that: rfftfreq with d=1/(n_grid-1)
        # already does the right thing.
        freqs_unit = np.fft.rfftfreq(n_grid, d=1.0 / (n_grid - 1))

    spec_rows = [
        {"label": label,
         "k_per_needle_length": float(freqs_unit[i]),
         "coherence": float(coh[i]),
         "n_pairs": int(n_pairs_total)}
        for i in range(len(coh))
    ]

    _write_csv(time_rows, out_path + ".time.csv")
    _write_csv(spec_rows, out_path + ".spec.csv")
    print(f"[{label}|cuda:{rank}] wrote → {out_path}.time.csv / .spec.csv", flush=True)


def exp_error_coherence(args, out_subdir: str):
    out_dir = os.path.join(args.out_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    label_a, label_b = args.label_a, args.label_b
    out_a = os.path.join(out_dir, label_a)
    out_b = os.path.join(out_dir, label_b)
    n_steps = int(args.n_rollout_coherence)

    ctx = mp.get_context("spawn")
    procs = []
    for rank, (exp_dir, base, label) in enumerate([
        (args.exp_a, out_a, label_a),
        (args.exp_b, out_b, label_b),
    ]):
        p = ctx.Process(
            target=_worker_coherence,
            args=(rank, exp_dir, args.data_dir,
                  n_steps, base, label),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"coherence worker exited {p.exitcode}")

    _plot_coherence(out_dir,
                    [(out_a + ".time.csv", label_a, "tab:blue"),
                     (out_b + ".time.csv", label_b, "tab:red")],
                    [(out_a + ".spec.csv", label_a, "tab:blue"),
                     (out_b + ".spec.csv", label_b, "tab:red")])


def _plot_coherence(out_dir, time_csvs, spec_csvs):
    plt = _matplotlib()

    def _read(path, cols):
        with open(path) as f:
            header = f.readline().strip().split(",")
            data = [dict(zip(header, line.strip().split(","))) for line in f if line.strip()]
        for d in data:
            for k in cols:
                d[k] = float(d[k])
        return data

    fig, ax = plt.subplots(figsize=(7, 5))
    for path, lab, color in time_csvs:
        data = _read(path, ["lag", "cos_mean", "cos_std"])
        lags = [d["lag"] for d in data]
        m = np.array([d["cos_mean"] for d in data])
        s = np.array([d["cos_std"] for d in data])
        ax.plot(lags, m, "o-", label=f"{lab} (mean)", color=color)
        ax.fill_between(lags, m - s, m + s, alpha=0.2, color=color,
                        label=f"{lab} (±1 std over step pairs)")
    ax.axhline(0.0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("rollout step lag Δ")
    ax.set_ylabel("cos⟨e_k, e_{k+Δ}⟩  averaged over k")
    ax.set_title("Per-node error: step-to-step cosine similarity")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    out_svg = os.path.join(out_dir, "time_corr.svg")
    fig.tight_layout()
    fig.savefig(out_svg)
    plt.close(fig)
    print(f"wrote → {out_svg}", flush=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    for path, lab, color in spec_csvs:
        data = _read(path, ["k_per_needle_length", "coherence"])
        ks = np.array([d["k_per_needle_length"] for d in data])
        cs = np.array([d["coherence"] for d in data])
        mask = ks > 0
        ax.plot(ks[mask], cs[mask], "-", label=lab, color=color)
    ax.set_xscale("log")
    ax.set_xlabel("wavenumber k (cycles per needle length)")
    ax.set_ylabel("spectral coherence γ²")
    ax.set_ylim(0, 1)
    ax.set_title("Spectral coherence of u-transverse error (pooled across test samples)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    out_svg = os.path.join(out_dir, "spectral_coherence.svg")
    fig.tight_layout()
    fig.savefig(out_svg)
    plt.close(fig)
    print(f"wrote → {out_svg}", flush=True)


# ===========================================================================
# Experiment 4: mat_fiber noise sensitivity.
# ===========================================================================
#
# At each rollout step we replace the unit fiber direction with
#   perturbed_k = unit_normalize(fiber_dir_initial + N(0, σ²·I₃))
# where N is freshly sampled per step but always added to the *original*
# fiber_dir (not the previous step's perturbed value), so the perturbation
# has zero mean over time and the trajectory has no drift contribution
# from fiber-noise accumulation.  The model's autoregressive state still
# evolves via its own predictions; only the static mat_fiber input is
# repeatedly re-randomised around the truth.
#
# Two metrics per step:
#   * deviation_u  = ‖pred_u_perturbed − pred_u_unperturbed‖  (needle nodes)
#   * error_u_gt   = ‖pred_u_perturbed − GT_u‖                (needle nodes)
# An amplitude=0 baseline run gives the no-noise references.
#
# Note: RolloutEngine.step applies *no* post-prediction smoothing
# (no consensus_attenuation / procrustes_alpha / axial_polyfit_alpha /
# tissue_consensus_attenuation / needle_edge_cap) — so the rollout used
# here is the raw model output exactly, satisfying the "smoothing off"
# requirement.

def _locate_mat_fiber_cols(engine: RolloutEngine) -> Optional[Tuple[int, int]]:
    """Return the (lo, hi) column slice in x where mat_fiber sits, or None if
    mat_fiber doesn't appear in x for this scheme.  In all schemes,
    `graph.fiber_dir` is still updated (it's a separate Data attribute).
    """
    if engine.mgn_paper:
        if not engine.mgn_include_mat_fiber:
            return None
        off = 2 + (1 if engine.mgn_include_evf else 0)
        return (off, off + 3)
    # Standard scheme: mat_fiber lives in the STATIC_PROP block, after the
    # dynamic INPUT block.  STATIC_PROP_KEYS = [mat_E, mat_c10, mat_density,
    # mat_fiber, ...] with dims [1, 1, 1, 3, 1, 1, 1, 1].
    ds = engine.dataset
    sp_off = sum(ds.INPUT_DIMS)
    for k, d in zip(ds.STATIC_PROP_KEYS, ds.STATIC_PROP_DIMS):
        if k == "mat_fiber":
            return (sp_off, sp_off + d)
        sp_off += d
    return None


def _apply_fiber_dir(graph: Data, new_fiber_dir: torch.Tensor,
                     engine: RolloutEngine, x_slice: Optional[Tuple[int, int]],
                     mf_mean: Optional[torch.Tensor], mf_std: Optional[torch.Tensor]):
    """In-place update of graph.fiber_dir + the appropriate x column slice.

    For mgn_paper schemes, the x slice holds the unit fiber direction
    directly.  For the standard scheme, the x slice holds the
    (raw_mat_fiber - mean) / std representation; we approximate the
    "raw" mat_fiber by the unit fiber direction (true for needle nodes
    under needle_fiber_axis=true, approximately true for tissue nodes).
    """
    graph.fiber_dir = new_fiber_dir
    if x_slice is None:
        return
    lo, hi = x_slice
    new_x = graph.x.clone()
    if engine.mgn_paper:
        new_x[:, lo:hi] = new_fiber_dir
    else:
        # Approximation: treat the perturbed unit direction as the new raw
        # mat_fiber, then renormalise with the dataset's stats.
        new_x[:, lo:hi] = (new_fiber_dir - mf_mean) / mf_std.clamp(min=1e-12)
    graph.x = new_x


def _worker_mat_fiber_noise(rank: int, exp_dir: str, data_dir: str, n_steps: int,
                            amplitudes: List[float], n_samples_max: Optional[int],
                            out_path: str, label: str, seed: int = 0):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    print(f"[{label}|cuda:{rank}] loading ...", flush=True)
    _, _, engine = load_experiment(exp_dir, data_dir, device)
    dataset = _reload_dataset_with_K(exp_dir, data_dir, n_steps)
    engine.dataset = dataset

    x_slice = _locate_mat_fiber_cols(engine)
    ns = engine.dataset._node_stats
    mf_mean = None
    mf_std = None
    if x_slice is not None and not engine.mgn_paper:
        mf_mean = _stat_to_tensor(ns["mat_fiber_mean"], device).view(1, -1)
        mf_std = _stat_to_tensor(ns["mat_fiber_std"], device).view(1, -1)

    n_samples = len(dataset)
    if n_samples_max is not None:
        n_samples = min(n_samples, int(n_samples_max))

    rows = []
    for sample_idx in range(n_samples):
        try:
            graph = dataset[sample_idx].to(device)
        except Exception as e:
            print(f"[{label}|cuda:{rank}] sample {sample_idx} load failed: {e}", flush=True)
            continue
        future = graph.future_deltas
        needle_local = torch.nonzero(graph.is_needle, as_tuple=False).squeeze(-1)
        if needle_local.numel() < 8:
            continue

        original_fiber_dir = graph.fiber_dir.clone()

        # Unperturbed rollout (baseline reference).
        with torch.no_grad():
            preds_unp, _ = engine.rollout_with_states(graph, n_steps)
        gt_u_per_step = [
            _denorm_per_key(future[:, k, :], engine)["u"] for k in range(n_steps)
        ]
        pred_u_unp_per_step = [_denorm_pred_u(preds_unp[k], engine) for k in range(n_steps)]

        gen = torch.Generator(device=device).manual_seed(seed + 1000 * sample_idx)

        for amp in amplitudes:
            g_p = graph.clone()
            # Reset fiber_dir / x on the cloned graph at each amplitude.
            _apply_fiber_dir(g_p, original_fiber_dir, engine, x_slice, mf_mean, mf_std)
            for k in range(n_steps):
                # Fresh noise this step, applied to the ORIGINAL fiber_dir
                # (not the previous perturbed value) — no drift accumulation.
                noise = torch.randn(
                    original_fiber_dir.shape, device=device,
                    dtype=original_fiber_dir.dtype, generator=gen,
                ) * float(amp)
                perturbed = original_fiber_dir + noise
                perturbed = perturbed / torch.linalg.norm(perturbed, dim=-1, keepdim=True).clamp(min=1e-8)
                _apply_fiber_dir(g_p, perturbed, engine, x_slice, mf_mean, mf_std)

                with torch.no_grad():
                    pred_p = engine.forward(g_p)

                pred_u_p = _denorm_pred_u(pred_p, engine)
                pred_u_unp = pred_u_unp_per_step[k]
                gt_u = gt_u_per_step[k]
                dev = (pred_u_p - pred_u_unp)[needle_local]
                err = (pred_u_p - gt_u)[needle_local]
                base = (pred_u_unp - gt_u)[needle_local]
                gt_n = gt_u[needle_local]

                rows.append({
                    "label": label,
                    "sample_idx": sample_idx,
                    "amplitude": float(amp),
                    "step": int(k),
                    "l2_dev_u": float(torch.linalg.norm(dev).item()),
                    "l2_err_u": float(torch.linalg.norm(err).item()),
                    "l2_baseline_err_u": float(torch.linalg.norm(base).item()),
                    "l2_gt_norm": float(torch.linalg.norm(gt_n).item()),
                    "n_needle": int(needle_local.numel()),
                })

                if k < n_steps - 1:
                    engine.step(g_p, pred_p)

        if (sample_idx + 1) % 5 == 0:
            print(f"[{label}|cuda:{rank}] {sample_idx + 1}/{n_samples} samples done", flush=True)

    _write_csv(rows, out_path)
    print(f"[{label}|cuda:{rank}] wrote {len(rows)} rows → {out_path}", flush=True)


def exp_mat_fiber_noise(args, out_subdir: str):
    out_dir = os.path.join(args.out_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    label_a, label_b = args.label_a, args.label_b
    csv_a = os.path.join(out_dir, f"{label_a}.csv")
    csv_b = os.path.join(out_dir, f"{label_b}.csv")
    amplitudes = [float(a) for a in args.fiber_noise_amplitudes.split(",") if a]
    n_steps = int(args.n_rollout_fiber_noise)
    n_samples_max = None if args.n_samples_fiber_noise <= 0 else int(args.n_samples_fiber_noise)

    ctx = mp.get_context("spawn")
    procs = []
    for rank, (exp_dir, csv_path, label) in enumerate([
        (args.exp_a, csv_a, label_a),
        (args.exp_b, csv_b, label_b),
    ]):
        p = ctx.Process(
            target=_worker_mat_fiber_noise,
            args=(rank, exp_dir, args.data_dir, n_steps, amplitudes,
                  n_samples_max, csv_path, label, args.seed),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"mat_fiber_noise worker exited {p.exitcode}")

    _plot_mat_fiber_noise(out_dir, csv_a, csv_b, label_a, label_b, amplitudes)


def _plot_mat_fiber_noise(out_dir, csv_a, csv_b, label_a, label_b, amplitudes):
    plt = _matplotlib()

    def _read(path):
        with open(path) as f:
            header = f.readline().strip().split(",")
            data = [dict(zip(header, line.strip().split(","))) for line in f if line.strip()]
        for d in data:
            for k in ("amplitude", "l2_dev_u", "l2_err_u", "l2_baseline_err_u", "l2_gt_norm"):
                d[k] = float(d[k])
            d["sample_idx"] = int(d["sample_idx"])
            d["step"] = int(d["step"])
        return data

    da = _read(csv_a)
    db = _read(csv_b)

    def _agg(data, metric: str):
        # → dict {amplitude: {step: (mean, std)}}, aggregated over samples.
        from collections import defaultdict
        buckets = defaultdict(lambda: defaultdict(list))
        for d in data:
            buckets[d["amplitude"]][d["step"]].append(d[metric])
        out = {}
        for amp, by_step in buckets.items():
            steps = sorted(by_step.keys())
            means = [float(np.mean(by_step[s])) for s in steps]
            stds = [float(np.std(by_step[s])) for s in steps]
            out[amp] = (np.array(steps), np.array(means), np.array(stds))
        return out

    # ---- Figure 1: deviation from unperturbed rollout, vs step. ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, data, lab in [(axes[0], da, label_a), (axes[1], db, label_b)]:
        agg = _agg(data, "l2_dev_u")
        amps_sorted = sorted(agg.keys())
        for i, amp in enumerate(amps_sorted):
            color = cmap(i / max(1, len(amps_sorted) - 1))
            steps, m, s = agg[amp]
            ax.plot(steps, m, "o-", color=color, label=f"σ = {amp:.3g} (mean)")
            ax.fill_between(steps, np.maximum(m - s, 1e-30), m + s, alpha=0.18, color=color,
                            label=f"σ = {amp:.3g} (±1 std over samples)")
        ax.set_yscale("log")
        ax.set_xlabel("rollout step")
        ax.set_ylabel("‖pred_u(σ) − pred_u(0)‖  (mm, needle nodes)")
        ax.set_title(lab)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("mat_fiber noise propagation — deviation from unperturbed rollout")
    fig.tight_layout()
    out_svg = os.path.join(out_dir, "deviation_vs_step.svg")
    fig.savefig(out_svg)
    plt.close(fig)
    print(f"wrote → {out_svg}", flush=True)

    # ---- Figure 2: error vs GT, with σ=0 baseline. -----------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, data, lab in [(axes[0], da, label_a), (axes[1], db, label_b)]:
        # Baseline σ=0 reference: l2_baseline_err_u is the same for all
        # rows of a given (sample, step) — average to get one curve.
        base_agg = _agg(data, "l2_baseline_err_u")
        amps_sorted = sorted(_agg(data, "l2_err_u").keys())
        for i, amp in enumerate(amps_sorted):
            color = cmap(i / max(1, len(amps_sorted) - 1))
            steps, m, s = _agg(data, "l2_err_u")[amp]
            ax.plot(steps, m, "o-", color=color, label=f"σ = {amp:.3g}")
            ax.fill_between(steps, np.maximum(m - s, 1e-30), m + s, alpha=0.15, color=color)
        # Pick one amplitude's baseline curve (they're all equivalent).
        if amps_sorted:
            steps_b, mb, sb = base_agg[amps_sorted[0]]
            ax.plot(steps_b, mb, "k--", label="σ = 0 (no perturbation)", linewidth=1.5)
        ax.set_yscale("log")
        ax.set_xlabel("rollout step")
        ax.set_ylabel("‖pred_u − GT_u‖  (mm, needle nodes)")
        ax.set_title(lab)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("mat_fiber noise → degradation in 1-step error vs ground truth")
    fig.tight_layout()
    out_svg = os.path.join(out_dir, "accuracy_vs_step.svg")
    fig.savefig(out_svg)
    plt.close(fig)
    print(f"wrote → {out_svg}", flush=True)


# ===========================================================================
# Driver.
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_a", required=True)
    parser.add_argument("--exp_b", required=True)
    parser.add_argument("--label_a", default=None)
    parser.add_argument("--label_b", default=None)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", default="./compare_models_out")
    parser.add_argument("--mode", choices=["perturb_propagation", "base_traj",
                                           "error_coherence", "mat_fiber_noise",
                                           "all"], default="all")
    parser.add_argument("--sample_idx", type=int, default=0)
    # Exp 1 knobs
    parser.add_argument("--n_centres", type=int, default=5)
    parser.add_argument("--lam_min", type=float, default=0.5)
    parser.add_argument("--lam_max", type=float, default=50.0)
    parser.add_argument("--n_lams", type=int, default=12)
    parser.add_argument("--amplitude", type=float, default=1e-3)
    parser.add_argument("--sigma_n_lambda", type=float, default=1.5)
    parser.add_argument("--step_horizons", default="1,5,10",
                        help="Comma-separated list of step counts at which to record deviation energy.")
    # Exp 2/3 knobs
    parser.add_argument("--n_rollout", type=int, default=10,
                        help="Number of rollout steps for base_traj.")
    parser.add_argument("--n_rollout_coherence", type=int, default=16,
                        help="Number of rollout steps for error_coherence "
                             "(both time-correlation lags and spectral-coherence averaging).")
    # Exp 4 knobs (mat_fiber noise sensitivity)
    parser.add_argument("--n_rollout_fiber_noise", type=int, default=16,
                        help="Number of rollout steps for mat_fiber_noise.")
    parser.add_argument("--fiber_noise_amplitudes", default="0.01,0.03,0.1,0.3,1.0",
                        help="Comma-separated noise amplitudes σ for the per-step "
                             "Gaussian on the unit fiber direction.  σ=1 ≈ noise of "
                             "the same magnitude as the original direction.")
    parser.add_argument("--n_samples_fiber_noise", type=int, default=0,
                        help="Cap on the number of test samples used by mat_fiber_noise. "
                             "0 = use all.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base RNG seed for per-step fiber-noise sampling.")
    args = parser.parse_args()

    args.label_a = args.label_a or os.path.basename(os.path.normpath(args.exp_a))
    args.label_b = args.label_b or os.path.basename(os.path.normpath(args.exp_b))
    os.makedirs(args.out_dir, exist_ok=True)

    n_gpus = torch.cuda.device_count()
    if n_gpus < 2 and args.mode in ("perturb_propagation", "error_coherence",
                                     "mat_fiber_noise", "all"):
        raise RuntimeError(
            f"Modes perturb_propagation / error_coherence / all need 2+ visible GPUs (found {n_gpus})."
        )

    summary = {
        "exp_a": args.exp_a, "exp_b": args.exp_b,
        "label_a": args.label_a, "label_b": args.label_b,
        "mode": args.mode, "sample_idx": args.sample_idx,
        "n_rollout": args.n_rollout,
        "n_rollout_coherence": args.n_rollout_coherence,
        "step_horizons": args.step_horizons,
        "n_lams": args.n_lams, "n_centres": args.n_centres,
    }
    print(json.dumps(summary, indent=2))

    if args.mode in ("perturb_propagation", "all"):
        exp_perturb_propagation(args, "perturb_propagation")
    if args.mode in ("base_traj", "all"):
        exp_base_trajectory(args, "base_traj")
    if args.mode in ("error_coherence", "all"):
        exp_error_coherence(args, "error_coherence")
    if args.mode in ("mat_fiber_noise", "all"):
        exp_mat_fiber_noise(args, "mat_fiber_noise")


if __name__ == "__main__":
    main()
