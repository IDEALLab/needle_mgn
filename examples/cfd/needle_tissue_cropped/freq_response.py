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

"""Bode-style frequency-response evaluation for cropped needle-tissue models.

For each of two trained models, this script:
  1. Loads a single test-split state.
  2. Picks 5 perturbation centres along the needle (tip → base, evenly spaced).
  3. For each centre, sweeps a log-spaced range of wavelengths (default 0.5–50 mm).
  4. At each (centre, wavelength), applies a transverse Gabor wavelet
     ``δu = A · exp(-(d/σ)²) · sin(2π d/λ) · ê_⊥`` to the needle u, runs the
     model on the unperturbed and perturbed states, and reports two gains:
        - L2 gain   ‖Δpred_u‖ / ‖δu‖   over needle nodes (raw mm).
        - FFT gain  |Û(k_λ) / δ̂(k_λ)| from a uniform-axial-grid resampling.
  5. Saves a CSV per model and a comparison PNG.

Two models run in parallel — one per visible CUDA device — via
``torch.multiprocessing.spawn``.  Each worker loads its model once and
re-uses it for every (centre, wavelength) tuple.

Usage
-----
    uv run python freq_response.py \\
        --exp_a /path/to/cropped_base_mgn_wu \\
        --exp_b /path/to/cropped_fiber_iso_invw \\
        --data_dir ../../../RUN-2 \\
        --out_dir ./freq_response_out

The ``outputs/.hydra/config.yaml`` from each experiment is loaded to derive
the model architecture and dataset flags; ``checkpoints/`` and ``stats/``
inside each ``exp_dir`` are used directly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf

# Local modules — added to path so we can import dataset.py from this dir
# regardless of where the script is invoked from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import NeedleTissueDataset  # noqa: E402

from physicsnemo.models.meshgraphnet import (  # noqa: E402
    FiberEquivariantKAN,
    FiberEquivariantMGN,
    MeshGraphKAN,
    MeshGraphNet,
    TFNMeshGraphNet,
)
from physicsnemo.utils import load_checkpoint  # noqa: E402


# ---------------------------------------------------------------------------
# Model loading (mirrors the construction logic in train.py / infer.py).
# ---------------------------------------------------------------------------

def _build_model(cfg, dataset, device):
    model_type = str(cfg.get("model_type", "mgn")).lower()
    shared = dict(
        input_dim_nodes=dataset.input_dim_nodes,
        input_dim_edges=int(cfg.get("input_dim_edges", 7)),
        output_dim=dataset.output_dim,
        processor_size=int(cfg.processor_size),
        hidden_dim_node_encoder=int(cfg.hidden_dim_node_encoder),
        hidden_dim_edge_encoder=int(cfg.hidden_dim_edge_encoder),
        hidden_dim_node_decoder=int(cfg.hidden_dim_node_decoder),
        hidden_dim_processor=int(cfg.hidden_dim_processor),
        aggregation=str(cfg.aggregation),
    )
    if model_type == "kan":
        model = MeshGraphKAN(**shared, num_harmonics=int(cfg.get("num_harmonics", 5)))
    elif model_type == "fiber":
        model = FiberEquivariantMGN(
            **shared,
            n_vec_outputs=int(cfg.get("n_vec_outputs", 3)),
            extra_edge_invariants=bool(cfg.get("fiber_extra_invariants", False)),
            extra_decoder_basis=bool(cfg.get("fiber_extra_decoder_basis", False)),
        )
    elif model_type == "fiber_kan":
        model = FiberEquivariantKAN(
            **shared,
            n_vec_outputs=int(cfg.get("n_vec_outputs", 3)),
            num_harmonics=int(cfg.get("num_harmonics", 5)),
            extra_edge_invariants=bool(cfg.get("fiber_extra_invariants", False)),
            extra_decoder_basis=bool(cfg.get("fiber_extra_decoder_basis", False)),
        )
    elif model_type == "tfn":
        model = TFNMeshGraphNet(
            n_node_scalar=dataset.n_tfn_scalar,
            n_node_vec=dataset.n_tfn_vec,
            output_dim=dataset.output_dim,
            irreps_hidden=str(cfg.get("irreps_hidden", "16x0e + 8x1o + 4x2e")),
            l_max=int(cfg.get("l_max", 2)),
            n_radial_basis=int(cfg.get("n_radial_basis", 8)),
            r_max=float(cfg.get("r_max", 60.0)),
            n_edge_extra_scalar=int(cfg.get("input_dim_edges", 7)) - 3,
            processor_size=int(cfg.processor_size),
            n_vec_outputs=int(cfg.get("n_vec_outputs", 3)),
            checkpoint_layers=False,
        )
    else:
        model = MeshGraphNet(
            **shared,
            use_fourier_features=bool(cfg.get("use_fourier_features", False)),
            n_fourier_features=int(cfg.get("n_fourier_features", 64)),
            fourier_scale=float(cfg.get("fourier_scale", 1.0)),
        )
    return model.to(device)


def _dataset_kwargs_from_cfg(cfg, data_dir, stats_dir):
    weights = list(cfg.get("crop_strategy_weights", [1.0, 0.0, 0.0]))
    return dict(
        data_dir=data_dir,
        needle_crop_mm=float(cfg.needle_crop_mm),
        tissue_crop_mm=float(cfg.tissue_crop_mm),
        slice_half_thickness_mm=float(cfg.slice_half_thickness_mm),
        full_needle_tissue_mm=float(cfg.full_needle_tissue_mm),
        crop_strategy_weights=tuple(weights),
        train_fraction=float(cfg.train_fraction),
        val_fraction=float(cfg.val_fraction),
        stats_path=stats_dir,
        cache_dir=data_dir,
        timestep_stride=int(cfg.get("timestep_stride", 1)),
        use_cpress=bool(cfg.get("use_cpress", True)),
        per_region_norm=bool(cfg.get("per_region_norm", False)),
        max_frames_per_run=cfg.get("max_frames_per_run", None),
        beam_spacing_mm=float(cfg.get("beam_spacing_mm", 0.0)),
        tissue_downsample_mm=float(cfg.get("tissue_downsample_mm", 0.0)),
        use_bsms=False,
        num_bsms_levels=int(cfg.get("num_bsms_levels", 2)),
        vector_iso_norm=bool(cfg.get("vector_iso_norm", False)),
        needle_fiber_axis=bool(cfg.get("needle_fiber_axis", False)),
        drop_targets=list(cfg.get("drop_targets", []) or []),
        mgn_paper_features=bool(cfg.get("mgn_paper_features", False)),
        mgn_include_mat_fiber=bool(cfg.get("mgn_include_mat_fiber", False)),
        mgn_include_prev_v=bool(cfg.get("mgn_include_prev_v", False)),
        mgn_include_evf=bool(cfg.get("mgn_include_evf", False)),
        mgn_kinematic_needle_only=bool(cfg.get("mgn_kinematic_needle_only", False)),
        multistep_K=1,
    )


# ---------------------------------------------------------------------------
# Perturbation: transverse Gabor wavelet on needle u.
# ---------------------------------------------------------------------------

def _needle_axis_and_transverse(pos_needle: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (axis, transverse, axial_coord_mm).

    ``axis`` is the unit principal SVD axis of the needle node positions.
    ``transverse`` is a unit vector orthogonal to ``axis`` chosen along the
    second principal direction (the larger of the two transverse modes).
    ``axial_coord_mm`` is the per-needle-node projection onto ``axis``,
    relative to the needle centroid.
    """
    pts = pos_needle.detach().cpu().numpy().astype(np.float64)
    centroid = pts.mean(axis=0, keepdims=True)
    centred = pts - centroid
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    axis = Vt[0]
    transverse = Vt[1]
    # Re-orthogonalise transverse against axis (defensive: SVD should already
    # give orthogonal singular vectors, but rounding can drift).
    transverse = transverse - axis * float(np.dot(axis, transverse))
    transverse = transverse / max(float(np.linalg.norm(transverse)), 1e-12)
    axial = centred @ axis  # (N_needle,)
    device = pos_needle.device
    return (
        torch.tensor(axis, dtype=torch.float32, device=device),
        torch.tensor(transverse, dtype=torch.float32, device=device),
        torch.tensor(axial, dtype=torch.float32, device=device),
    )


def _gabor_wavelet(axial_coord_mm: torch.Tensor, centre_mm: float, wavelength_mm: float,
                   sigma_n_lambda: float = 1.5, amplitude: float = 1e-3) -> torch.Tensor:
    """Sample a Gabor wavelet at each needle node's axial coordinate.

    Returns a 1D tensor of scalar amplitudes; multiply by the transverse unit
    direction to get a 3-vector perturbation per needle node.
    """
    sigma = sigma_n_lambda * wavelength_mm
    d = axial_coord_mm - centre_mm
    env = torch.exp(-(d / sigma) ** 2)
    car = torch.sin(2.0 * np.pi * d / wavelength_mm)
    return amplitude * env * car


# ---------------------------------------------------------------------------
# Apply a perturbation to a graph (cloned) under either input scheme.
# ---------------------------------------------------------------------------

def _stat_to_tensor(stat, device, dtype=torch.float32) -> torch.Tensor:
    """``load_json`` returns python lists; cfg stats are torch tensors during
    training but JSON-loaded lists at test time.  Coerce both to a 1-D tensor.
    """
    if isinstance(stat, torch.Tensor):
        return stat.to(device=device, dtype=dtype)
    return torch.tensor(stat, device=device, dtype=dtype)


def _apply_perturbation(
    graph,
    delta_u_full: torch.Tensor,
    dataset: NeedleTissueDataset,
    cfg,
):
    """Return a NEW graph with pos / x / edge_attr updated for the perturbation.

    ``delta_u_full`` is a (n_sub, 3) tensor — non-zero only on needle nodes.
    """
    g = graph.clone()
    device = g.x.device
    g.pos = g.pos + delta_u_full

    if not bool(cfg.get("mgn_paper_features", False)):
        # Standard input scheme: x has [coord, u, v, a, evf, s, (cpress)].
        # δcoord_norm = δu_raw / coord_std,   δu_norm = δu_raw / u_std.
        coord_std = _stat_to_tensor(dataset._node_stats["coord_std"], device).view(1, -1)
        u_std = _stat_to_tensor(dataset._node_stats["u_std"], device).view(1, -1)
        new_x = g.x.clone()
        new_x[:, 0:3] = new_x[:, 0:3] + delta_u_full / coord_std
        new_x[:, 3:6] = new_x[:, 3:6] + delta_u_full / u_std
        g.x = new_x
    # mgn_paper_features: x is the static node-type one-hot (+ evf/fiber/v,
    # all velocity / contact rather than displacement); x is unchanged by a
    # pure-displacement perturbation.  The model still sees δu indirectly
    # via the recomputed edge_attr below.

    # Rebuild edge_attr first 4 cols (rel_pos + edge_len) from the new pos.
    src, dst = g.edge_index
    new_attr = g.edge_attr.clone()
    rel_pos = g.pos[src] - g.pos[dst]
    new_attr[:, 0:3] = rel_pos
    new_attr[:, 3:4] = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)
    g.edge_attr = new_attr
    return g


# ---------------------------------------------------------------------------
# Bode metrics: L2 gain + FFT gain at the input centre wavenumber.
# ---------------------------------------------------------------------------

def _denorm_pred_u(pred_norm: torch.Tensor, dataset: NeedleTissueDataset) -> torch.Tensor:
    """Convert a normalised model output back to raw Δu (mm).

    For a *difference* of two predictions, the target_mean cancels; only the
    target_std multiplier remains.
    """
    # u is the first target key in TARGET_KEYS (always when not dropped).
    if "u" not in dataset.TARGET_KEYS:
        raise RuntimeError("Frequency-response analysis requires 'u' in TARGET_KEYS.")
    off = 0
    for k, d in zip(dataset.TARGET_KEYS, dataset.TARGET_DIMS):
        if k == "u":
            u_off, u_dim = off, d
            break
        off += d
    u_std = _stat_to_tensor(dataset._target_stats["u_std"], pred_norm.device).view(1, -1)
    return pred_norm[:, u_off : u_off + u_dim] * u_std


def _spectral_metrics(F_delta, F_diff, freqs, target_freq):
    """Energy-based descriptors of the output spectrum.

    centroid_k        — spectral centroid of |F_diff|² (cycles/mm).
    centroid_ratio    — centroid / target_freq.
    bandwidth_k       — std of |F_diff|² around its centroid.
    bandwidth_ratio   — bandwidth / target_freq.
    hf_fraction_2x    — fraction of output energy above 2 × k_in.
    hf_fraction_5x    — fraction of output energy above 5 × k_in.
    total_energy      — Σ |F_diff|².
    """
    eps = 1e-30
    E_diff = np.abs(F_diff) ** 2
    total_energy = float(np.sum(E_diff))
    centroid = float(np.sum(freqs * E_diff) / max(total_energy, eps))
    bandwidth = float(np.sqrt(np.sum((freqs - centroid) ** 2 * E_diff) / max(total_energy, eps)))
    hf_mask = freqs > 2.0 * target_freq
    vhf_mask = freqs > 5.0 * target_freq
    hf_fraction = float(np.sum(E_diff[hf_mask]) / max(total_energy, eps))
    vhf_fraction = float(np.sum(E_diff[vhf_mask]) / max(total_energy, eps))
    return {
        "centroid_k": centroid,
        "centroid_ratio": centroid / target_freq if target_freq > 0 else float("nan"),
        "bandwidth_k": bandwidth,
        "bandwidth_ratio": bandwidth / target_freq if target_freq > 0 else float("nan"),
        "hf_fraction_2x": hf_fraction,
        "hf_fraction_5x": vhf_fraction,
        "total_energy": total_energy,
    }


def _bode_metrics(
    delta_u_needle_raw: torch.Tensor,   # (n_needle, 3)
    diff_u_needle_raw: torch.Tensor,    # (n_needle, 3)
    transverse_dir: torch.Tensor,       # (3,)
    axial_coord: torch.Tensor,          # (n_needle,) — sortable axial mm
    wavelength: float,
    n_grid: int = 512,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Return (scalar metrics, freqs_per_mm, |F_delta|, |F_diff|).

    Scalar metrics:
        l2_gain                — ‖Δpred_u‖ / ‖δu‖ on needle nodes (raw mm).
        fft_gain               — |F_diff(k_in)| / |F_delta(k_in)|.
        fft_freq_per_mm        — actual frequency of the FFT bin used for fft_gain.
        out_peak_freq_per_mm   — frequency at which |F_diff| is maximised (DC excluded).
        out_peak_wavelength_mm — 1 / out_peak_freq_per_mm.
        out_peak_gain          — |F_diff(k_peak)| / |F_delta(k_in)|: how much
                                 stronger the output's dominant mode is than
                                 the input amplitude at the driven frequency.
    """
    eps = 1e-30
    l2_in = float(torch.linalg.norm(delta_u_needle_raw))
    l2_out = float(torch.linalg.norm(diff_u_needle_raw))
    l2_gain = l2_out / max(l2_in, eps)

    # Project both signals onto the perturbation's transverse direction so
    # the FFT operates on a scalar field over axial position.
    delta_proj = (delta_u_needle_raw * transverse_dir.view(1, 3)).sum(-1)
    diff_proj = (diff_u_needle_raw * transverse_dir.view(1, 3)).sum(-1)

    # Resample to a uniform axial grid (linear interp on sort-by-axial).
    s = axial_coord.detach().cpu().numpy()
    order = np.argsort(s)
    s_sorted = s[order]
    delta_sorted = delta_proj.detach().cpu().numpy()[order]
    diff_sorted = diff_proj.detach().cpu().numpy()[order]
    s_min, s_max = float(s_sorted[0]), float(s_sorted[-1])
    s_uniform = np.linspace(s_min, s_max, n_grid)
    delta_grid = np.interp(s_uniform, s_sorted, delta_sorted)
    diff_grid = np.interp(s_uniform, s_sorted, diff_sorted)

    L = s_max - s_min
    nan_metrics = {
        "l2_gain": l2_gain, "fft_gain": float("nan"), "fft_freq": float("nan"),
        "out_peak_freq": float("nan"), "out_peak_wavelength": float("nan"),
        "out_peak_gain": float("nan"),
        "centroid_k": float("nan"), "centroid_ratio": float("nan"),
        "bandwidth_k": float("nan"), "bandwidth_ratio": float("nan"),
        "hf_fraction_2x": float("nan"), "hf_fraction_5x": float("nan"),
        "total_energy": float("nan"),
    }
    if L <= 0.0 or n_grid < 4:
        empty = np.zeros(n_grid // 2 + 1)
        return nan_metrics, empty, empty.copy(), empty.copy()

    dx = L / (n_grid - 1)
    F_delta = np.fft.rfft(delta_grid)
    F_diff = np.fft.rfft(diff_grid)
    freqs = np.fft.rfftfreq(n_grid, d=dx)  # cycles per mm
    F_delta_abs = np.abs(F_delta)
    F_diff_abs = np.abs(F_diff)

    # ---- Gain at the driven frequency ----------------------------------
    target_freq = 1.0 / wavelength
    bin_idx = int(round(target_freq / freqs[1])) if len(freqs) > 1 else 1
    bin_idx = max(1, min(bin_idx, len(freqs) - 1))
    denom_in = F_delta_abs[bin_idx]
    fft_gain = float(F_diff_abs[bin_idx] / denom_in) if denom_in > eps else float("nan")

    # ---- Output peak frequency (DC bin excluded) -----------------------
    # The Gabor envelope deposits some energy at very low wavenumbers from
    # the envelope's spectral lobe; the meaningful "where does the model
    # put energy" question is about non-DC content, so we search bins ≥ 1.
    peak_bin = int(1 + np.argmax(F_diff_abs[1:]))
    out_peak_freq = float(freqs[peak_bin])
    out_peak_wavelength = float("inf") if out_peak_freq <= 0.0 else 1.0 / out_peak_freq
    out_peak_gain = float(F_diff_abs[peak_bin] / denom_in) if denom_in > eps else float("nan")

    spectral = _spectral_metrics(F_delta, F_diff, freqs, target_freq)
    metrics = {
        "l2_gain": l2_gain,
        "fft_gain": fft_gain,
        "fft_freq": float(freqs[bin_idx]),
        "out_peak_freq": out_peak_freq,
        "out_peak_wavelength": out_peak_wavelength,
        "out_peak_gain": out_peak_gain,
        **spectral,
    }
    return metrics, freqs, F_delta_abs, F_diff_abs


# ---------------------------------------------------------------------------
# Per-GPU worker: load model once, sweep all (centre, wavelength).
# ---------------------------------------------------------------------------

def _worker(
    rank: int,
    exp_dir: str,
    data_dir: str,
    sample_idx: int,
    n_centres: int,
    wavelengths: List[float],
    amplitude: float,
    sigma_n_lambda: float,
    out_path: str,
    label: str,
):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    cfg_path = os.path.join(exp_dir, "outputs", ".hydra", "config.yaml")
    cfg = OmegaConf.load(cfg_path)
    ckpt_path = os.path.join(exp_dir, "checkpoints")
    stats_dir = os.path.join(exp_dir, "stats")

    print(f"[{label}|cuda:{rank}] loading dataset ...", flush=True)
    ds_kwargs = _dataset_kwargs_from_cfg(cfg, data_dir, stats_dir)
    dataset = NeedleTissueDataset(split="test", **ds_kwargs)

    print(f"[{label}|cuda:{rank}] building model ({cfg.get('model_type','mgn')}) ...", flush=True)
    model = _build_model(cfg, dataset, device)
    load_checkpoint(ckpt_path, models=model, device=device)
    model.eval()

    sample_idx = max(0, min(sample_idx, len(dataset) - 1))
    print(f"[{label}|cuda:{rank}] using test sample {sample_idx} of {len(dataset)}", flush=True)
    graph = dataset[sample_idx].to(device)

    # Needle local indices and axial geometry.
    needle_local = torch.nonzero(graph.is_needle, as_tuple=False).squeeze(-1)
    if needle_local.numel() < 8:
        raise RuntimeError(
            f"Test sample has only {needle_local.numel()} needle nodes in the crop; "
            "freq-response analysis needs more.  Pick a different sample_idx."
        )
    pos_needle = graph.pos[needle_local]
    axis, transverse, axial_coord = _needle_axis_and_transverse(pos_needle)

    # Centres along the needle (tip → base).
    s_min = float(axial_coord.min().item())
    s_max = float(axial_coord.max().item())
    centres = np.linspace(s_min, s_max, n_centres).tolist()

    # Baseline prediction (run once per sample).
    with torch.no_grad():
        pred_baseline = model(graph.x, graph.edge_attr, graph)

    rows = []
    n_grid = 512
    spec_freqs: np.ndarray | None = None
    in_spectra = np.zeros((n_centres, len(wavelengths), n_grid // 2 + 1), dtype=np.float32)
    out_spectra = np.zeros_like(in_spectra)

    for c_i, centre in enumerate(centres):
        for w_i, lam in enumerate(wavelengths):
            scalar = _gabor_wavelet(
                axial_coord, centre_mm=float(centre), wavelength_mm=float(lam),
                sigma_n_lambda=sigma_n_lambda, amplitude=amplitude,
            )
            delta_u = torch.zeros((graph.x.shape[0], 3), dtype=torch.float32, device=device)
            delta_u[needle_local] = scalar.unsqueeze(-1) * transverse.view(1, 3)

            graph_p = _apply_perturbation(graph, delta_u, dataset, cfg)
            with torch.no_grad():
                pred_p = model(graph_p.x, graph_p.edge_attr, graph_p)
            diff_norm = pred_p - pred_baseline
            diff_u_raw = _denorm_pred_u(diff_norm, dataset)  # (n_sub, 3)

            metrics, freqs, F_in_abs, F_out_abs = _bode_metrics(
                delta_u_needle_raw=delta_u[needle_local],
                diff_u_needle_raw=diff_u_raw[needle_local],
                transverse_dir=transverse,
                axial_coord=axial_coord,
                wavelength=float(lam),
                n_grid=n_grid,
            )
            if spec_freqs is None:
                spec_freqs = freqs
            in_spectra[c_i, w_i] = F_in_abs.astype(np.float32)
            out_spectra[c_i, w_i] = F_out_abs.astype(np.float32)
            rows.append({
                "label": label,
                "centre_idx": c_i,
                "centre_mm": float(centre),
                "wavelength_mm": float(lam),
                "wavenumber_per_mm": 1.0 / float(lam),
                "l2_gain": metrics["l2_gain"],
                "fft_gain": metrics["fft_gain"],
                "fft_freq_per_mm": metrics["fft_freq"],
                "out_peak_freq_per_mm": metrics["out_peak_freq"],
                "out_peak_wavelength_mm": metrics["out_peak_wavelength"],
                "out_peak_gain": metrics["out_peak_gain"],
                "centroid_k_per_mm": metrics["centroid_k"],
                "centroid_ratio": metrics["centroid_ratio"],
                "bandwidth_k_per_mm": metrics["bandwidth_k"],
                "bandwidth_ratio": metrics["bandwidth_ratio"],
                "hf_fraction_2x": metrics["hf_fraction_2x"],
                "hf_fraction_5x": metrics["hf_fraction_5x"],
                "total_energy": metrics["total_energy"],
            })
        print(f"[{label}|cuda:{rank}] centre {c_i+1}/{n_centres} done", flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Compact CSV write — no pandas dependency.
    fields = list(rows[0].keys())
    with open(out_path, "w") as f:
        f.write(",".join(fields) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in fields) + "\n")
    print(f"[{label}|cuda:{rank}] wrote {len(rows)} rows → {out_path}", flush=True)

    # Persist the full input/output spectra so the plotter can show the
    # output spectrum at several input wavelengths without re-running the
    # whole sweep.  Indexed as [centre_idx, wavelength_idx, freq_bin].
    npz_path = out_path.replace(".csv", "_spectra.npz")
    np.savez_compressed(
        npz_path,
        wavelengths_mm=np.asarray(wavelengths, dtype=np.float32),
        centres_mm=np.asarray(centres, dtype=np.float32),
        freqs_per_mm=spec_freqs.astype(np.float32) if spec_freqs is not None else np.array([], dtype=np.float32),
        input_spectra=in_spectra,
        output_spectra=out_spectra,
    )
    print(f"[{label}|cuda:{rank}] wrote spectra → {npz_path}", flush=True)


# ---------------------------------------------------------------------------
# Driver: spawn 2 workers, wait, plot.
# ---------------------------------------------------------------------------

def _plot(out_dir: str, csv_a: str, csv_b: str, label_a: str, label_b: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.", flush=True)
        return

    def _read(path):
        with open(path) as f:
            header = f.readline().strip().split(",")
            data = [dict(zip(header, line.strip().split(","))) for line in f if line.strip()]
        float_cols = (
            "centre_mm", "wavelength_mm", "wavenumber_per_mm",
            "l2_gain", "fft_gain", "fft_freq_per_mm",
            "out_peak_freq_per_mm", "out_peak_wavelength_mm", "out_peak_gain",
            "centroid_k_per_mm", "centroid_ratio",
            "bandwidth_k_per_mm", "bandwidth_ratio",
            "hf_fraction_2x", "hf_fraction_5x", "total_energy",
        )
        for d in data:
            for k in float_cols:
                if k in d:
                    d[k] = float(d[k])
            d["centre_idx"] = int(d["centre_idx"])
        return data

    da = _read(csv_a)
    db = _read(csv_b)

    def _agg(data):
        # mean & std over centres, per wavelength.
        by_lam: Dict[float, List[Tuple[float, float, float]]] = {}
        for r in data:
            by_lam.setdefault(r["wavelength_mm"], []).append(
                (r["l2_gain"], r["fft_gain"], r["out_peak_wavelength_mm"])
            )
        lams = sorted(by_lam.keys())
        l2_mean = [np.mean([v[0] for v in by_lam[lam]]) for lam in lams]
        l2_std = [np.std([v[0] for v in by_lam[lam]]) for lam in lams]
        fft_mean = [np.nanmean([v[1] for v in by_lam[lam]]) for lam in lams]
        fft_std = [np.nanstd([v[1] for v in by_lam[lam]]) for lam in lams]
        # Replace +inf (output peak at DC, λ → ∞) with NaN so the log scatter
        # doesn't blow up.  Median is used here (more robust than mean to the
        # bimodality that shows up when some centres land on a node).
        def _finite_peaks(lam_key):
            return [np.nan if not np.isfinite(v[2]) else v[2] for v in by_lam[lam_key]]

        peak_med = [np.nanmedian(_finite_peaks(lam)) for lam in lams]
        peak_lo = [np.nanpercentile(_finite_peaks(lam), 25) for lam in lams]
        peak_hi = [np.nanpercentile(_finite_peaks(lam), 75) for lam in lams]
        return (
            np.array(lams), np.array(l2_mean), np.array(l2_std),
            np.array(fft_mean), np.array(fft_std),
            np.array(peak_med), np.array(peak_lo), np.array(peak_hi),
        )

    # ---- Panel 1: gain curves (L2 + FFT). ---------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    agg_cache = {}
    for data, lab, color in [(da, label_a, "tab:blue"), (db, label_b, "tab:red")]:
        lams, l2m, l2s, ftm, fts, pkm, plo, phi = _agg(data)
        agg_cache[lab] = (lams, l2m, l2s, ftm, fts, pkm, plo, phi, color)
        axes[0].plot(lams, l2m, "o-", label=f"{lab} (mean)", color=color)
        axes[0].fill_between(lams, l2m - l2s, l2m + l2s, alpha=0.2, color=color,
                             label=f"{lab} (±1 std over centres)")
        axes[1].plot(lams, ftm, "o-", label=f"{lab} (mean)", color=color)
        axes[1].fill_between(lams, np.maximum(ftm - fts, 1e-12), ftm + fts, alpha=0.2,
                             color=color, label=f"{lab} (±1 std over centres)")
        # Output peak wavelength vs input wavelength (with IQR band).
        axes[2].plot(lams, pkm, "o-", label=f"{lab} (median)", color=color)
        axes[2].fill_between(lams, plo, phi, alpha=0.2, color=color,
                             label=f"{lab} (IQR over centres)")
    # y=x reference line on the peak-wavelength panel.
    lam_min = min(agg_cache[label_a][0].min(), agg_cache[label_b][0].min())
    lam_max = max(agg_cache[label_a][0].max(), agg_cache[label_b][0].max())
    axes[2].plot([lam_min, lam_max], [lam_min, lam_max], "k--", alpha=0.5, label="y = x (output mirrors input)")
    for ax, ttl, ylab in [
        (axes[0], "L2 gain  ‖Δpred_u‖ / ‖δu‖   (needle nodes)", "gain (raw mm / raw mm)"),
        (axes[1], "FFT gain  |Û(k_λ) / δ̂(k_λ)|  at the driven frequency", "gain"),
        (axes[2], "Output peak wavelength vs input wavelength", "output peak λ (mm)"),
    ]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("input wavelength λ (mm)")
        ax.set_ylabel(ylab)
        ax.set_title(ttl)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle("Frequency response — transverse Gabor wavelet on needle u")
    fig.tight_layout()
    out_svg = os.path.join(out_dir, "freq_response.svg")
    fig.savefig(out_svg)
    plt.close(fig)
    print(f"wrote plot → {out_svg}", flush=True)

    # ---- Spectral-descriptor figure (centroid / bandwidth / HF energy). --
    def _agg_spec(data):
        by_lam: Dict[float, List[Tuple[float, float, float, float]]] = {}
        for r in data:
            by_lam.setdefault(r["wavelength_mm"], []).append((
                r["centroid_ratio"], r["bandwidth_ratio"],
                r["hf_fraction_2x"], r["hf_fraction_5x"],
            ))
        lams = sorted(by_lam.keys())
        cr_med = np.array([np.nanmedian([v[0] for v in by_lam[lam]]) for lam in lams])
        cr_lo = np.array([np.nanpercentile([v[0] for v in by_lam[lam]], 25) for lam in lams])
        cr_hi = np.array([np.nanpercentile([v[0] for v in by_lam[lam]], 75) for lam in lams])
        bw_med = np.array([np.nanmedian([v[1] for v in by_lam[lam]]) for lam in lams])
        bw_lo = np.array([np.nanpercentile([v[1] for v in by_lam[lam]], 25) for lam in lams])
        bw_hi = np.array([np.nanpercentile([v[1] for v in by_lam[lam]], 75) for lam in lams])
        hf2_med = np.array([np.nanmedian([v[2] for v in by_lam[lam]]) for lam in lams])
        hf2_std = np.array([np.nanstd([v[2] for v in by_lam[lam]]) for lam in lams])
        hf5_med = np.array([np.nanmedian([v[3] for v in by_lam[lam]]) for lam in lams])
        hf5_std = np.array([np.nanstd([v[3] for v in by_lam[lam]]) for lam in lams])
        return (
            np.array(lams),
            cr_med, cr_lo, cr_hi,
            bw_med, bw_lo, bw_hi,
            hf2_med, hf2_std,
            hf5_med, hf5_std,
        )

    fig_s, axes_s = plt.subplots(1, 3, figsize=(18, 5))
    for data, lab, color in [(da, label_a, "tab:blue"), (db, label_b, "tab:red")]:
        (lams, cr, crl, crh, bw, bwl, bwh,
         hf2, hf2s, hf5, hf5s) = _agg_spec(data)
        axes_s[0].plot(lams, cr, "o-", label=f"{lab} (median)", color=color)
        axes_s[0].fill_between(lams, crl, crh, alpha=0.2, color=color,
                               label=f"{lab} (IQR over centres)")
        axes_s[1].plot(lams, bw, "o-", label=f"{lab} (median)", color=color)
        axes_s[1].fill_between(lams, bwl, bwh, alpha=0.2, color=color,
                               label=f"{lab} (IQR over centres)")
        axes_s[2].plot(lams, hf2, "o-", label=f"{lab} (>2× k_in, median)", color=color)
        axes_s[2].fill_between(lams, np.clip(hf2 - hf2s, 0, 1), np.clip(hf2 + hf2s, 0, 1),
                               alpha=0.2, color=color,
                               label=f"{lab} (>2× k_in, ±1 std over centres)")
        axes_s[2].plot(lams, hf5, "s--", label=f"{lab} (>5× k_in, median)", color=color, alpha=0.7)
    # Reference: centroid_ratio = 1 means output centred at input frequency.
    axes_s[0].axhline(1.0, color="k", linestyle="--", alpha=0.5, label="centroid = k_in")
    for ax, ttl, ylab, ylog in [
        (axes_s[0], "Spectral centroid ratio  (centroid / k_in)", "ratio (–)", True),
        (axes_s[1], "Spectral bandwidth ratio  (bandwidth / k_in)", "ratio (–)", True),
        (axes_s[2], "Fraction of output energy above 2× / 5× k_in", "energy fraction (–)", False),
    ]:
        ax.set_xscale("log")
        if ylog:
            ax.set_yscale("log")
        ax.set_xlabel("input wavelength λ (mm)")
        ax.set_ylabel(ylab)
        ax.set_title(ttl)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig_s.suptitle("Spectral descriptors of the model's response")
    fig_s.tight_layout()
    out_svg_s = os.path.join(out_dir, "freq_response_spectral.svg")
    fig_s.savefig(out_svg_s)
    plt.close(fig_s)
    print(f"wrote spectral-descriptor plot → {out_svg_s}", flush=True)

    # ---- Panel 2: output spectra at a selection of input wavelengths. ----
    npz_a = csv_a.replace(".csv", "_spectra.npz")
    npz_b = csv_b.replace(".csv", "_spectra.npz")
    if not (os.path.exists(npz_a) and os.path.exists(npz_b)):
        print("(no spectra NPZ found; skipping spectrum panel)", flush=True)
        return
    spec_a = np.load(npz_a)
    spec_b = np.load(npz_b)
    lams_a = spec_a["wavelengths_mm"]
    freqs = spec_a["freqs_per_mm"]
    # Pick a handful of representative input wavelengths log-spaced through the sweep.
    n_show = 6
    show_idx = np.unique(np.round(np.linspace(0, len(lams_a) - 1, n_show)).astype(int))
    n_show = len(show_idx)
    n_cols = 3
    n_rows = int(np.ceil(n_show / n_cols))
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.4 * n_rows), squeeze=False)
    # Aggregate spectra across centres → mean magnitude per freq bin.
    out_a = spec_a["output_spectra"].mean(axis=0)  # (n_lams, n_freqs)
    out_b = spec_b["output_spectra"].mean(axis=0)
    in_a = spec_a["input_spectra"].mean(axis=0)
    # Mask DC bin so the log plot isn't dominated by the envelope's DC lobe.
    mask = freqs > 0
    for k, idx in enumerate(show_idx):
        ax = axes2[k // n_cols, k % n_cols]
        lam = float(lams_a[idx])
        k_in = 1.0 / lam
        ax.plot(freqs[mask], in_a[idx][mask], "k:", alpha=0.5, label="input |δ̂(k)|")
        ax.plot(freqs[mask], out_a[idx][mask], "-", color="tab:blue", label=f"out: {label_a}")
        ax.plot(freqs[mask], out_b[idx][mask], "-", color="tab:red", label=f"out: {label_b}")
        ax.axvline(k_in, color="green", linestyle="--", alpha=0.6, label=f"driven k = {k_in:.3g}/mm")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"input λ = {lam:.3g} mm")
        ax.set_xlabel("wavenumber k (1/mm)")
        ax.set_ylabel("|F|")
        ax.grid(True, which="both", alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8)
    for k in range(n_show, n_rows * n_cols):
        axes2[k // n_cols, k % n_cols].axis("off")
    fig2.suptitle("Output spectra at selected input wavelengths (mean over centres)")
    fig2.tight_layout()
    out_svg2 = os.path.join(out_dir, "freq_response_spectra.svg")
    fig2.savefig(out_svg2)
    plt.close(fig2)
    print(f"wrote spectrum plot → {out_svg2}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_a", required=True, help="Path to experiment A directory.")
    parser.add_argument("--exp_b", required=True, help="Path to experiment B directory.")
    parser.add_argument("--label_a", default=None, help="Plot label for A (default: dir basename).")
    parser.add_argument("--label_b", default=None, help="Plot label for B (default: dir basename).")
    parser.add_argument("--data_dir", required=True, help="Path to RUN-2 (or equivalent) dataset directory.")
    parser.add_argument("--out_dir", default="./freq_response_out")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Index into the test split's sample list.")
    parser.add_argument("--n_centres", type=int, default=5,
                        help="Number of axial perturbation centres along the needle.")
    parser.add_argument("--lam_min", type=float, default=0.5)
    parser.add_argument("--lam_max", type=float, default=50.0)
    parser.add_argument("--n_lams", type=int, default=24)
    parser.add_argument("--amplitude", type=float, default=1e-3,
                        help="Peak displacement amplitude (mm).  Should keep response in linear regime.")
    parser.add_argument("--sigma_n_lambda", type=float, default=1.5,
                        help="Gabor envelope std as a multiple of wavelength.")
    args = parser.parse_args()

    label_a = args.label_a or os.path.basename(os.path.normpath(args.exp_a))
    label_b = args.label_b or os.path.basename(os.path.normpath(args.exp_b))
    os.makedirs(args.out_dir, exist_ok=True)
    csv_a = os.path.join(args.out_dir, f"{label_a}.csv")
    csv_b = os.path.join(args.out_dir, f"{label_b}.csv")

    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        raise RuntimeError(
            f"This script parallelises one model per GPU and expects 2+ visible GPUs; "
            f"found {n_gpus}.  Set CUDA_VISIBLE_DEVICES to expose 2 devices."
        )

    wavelengths = np.geomspace(args.lam_min, args.lam_max, args.n_lams).tolist()
    print(json.dumps({
        "exp_a": args.exp_a, "exp_b": args.exp_b,
        "label_a": label_a, "label_b": label_b,
        "wavelengths_mm": wavelengths, "n_centres": args.n_centres,
        "amplitude_mm": args.amplitude, "sigma_n_lambda": args.sigma_n_lambda,
        "sample_idx": args.sample_idx,
    }, indent=2))

    # Spawn 2 workers — rank 0 → exp_a on cuda:0, rank 1 → exp_b on cuda:1.
    ctx = mp.get_context("spawn")
    procs = []
    for rank, (exp_dir, out_path, label) in enumerate([
        (args.exp_a, csv_a, label_a),
        (args.exp_b, csv_b, label_b),
    ]):
        p = ctx.Process(
            target=_worker,
            args=(rank, exp_dir, args.data_dir, args.sample_idx,
                  args.n_centres, wavelengths, args.amplitude, args.sigma_n_lambda,
                  out_path, label),
        )
        p.start()
        procs.append(p)
    failed = []
    for p in procs:
        p.join()
        if p.exitcode != 0:
            failed.append(p.exitcode)
    if failed:
        raise RuntimeError(f"Worker(s) failed: exit codes {failed}")

    _plot(args.out_dir, csv_a, csv_b, label_a, label_b)


if __name__ == "__main__":
    main()
