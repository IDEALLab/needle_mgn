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

"""Training script for needle-tissue MeshGraphNet with dynamic spatial cropping."""

import os
import re
import time

import hydra
import numpy as np
import torch
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from scipy.spatial import cKDTree
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel


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
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    import apex
except Exception:
    pass

from dataset import NeedleTissueDataset
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.models.meshgraphnet import MeshGraphNet, MeshGraphKAN, FiberEquivariantMGN, FiberEquivariantKAN, TFNMeshGraphNet
from physicsnemo.models.meshgraphnet.bsms_mgn import BiStrideMeshGraphNet
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils.logging.wandb import initialize_wandb


def _collate(batch):
    """Collate a list of PyG Data objects into a single Batch."""
    return Batch.from_data_list(batch)


def _bsms_collate(batch):
    """Collate for BSMS mode (batch_size=1).

    The dataset returns dicts ``{"graph": Data, "ms_edges": [...], "ms_ids": [...]}``.
    With batch_size=1 we unwrap the outer list so the training loop receives
    the dict directly without any tensor stacking.
    """
    assert len(batch) == 1, "BSMS training requires batch_size=1"
    return batch[0]


class MGNTrainer:
    def __init__(self, cfg: DictConfig, dist, rank_zero_logger):
        self.dist = dist
        self.amp = cfg.amp
        self.noise_std = float(cfg.get("noise_std", 0.0))
        self.use_bsms = bool(cfg.get("use_bsms", False))

        stats_dir = _abspath(cfg.stats_dir)
        data_dir = _abspath(cfg.data_dir)

        crop_strategy_weights = tuple(cfg.crop_strategy_weights)
        use_cpress = bool(cfg.get("use_cpress", True))
        per_region_norm = bool(cfg.get("per_region_norm", False))
        max_frames_per_run = cfg.get("max_frames_per_run", None)
        if max_frames_per_run is not None:
            max_frames_per_run = int(max_frames_per_run)
        beam_spacing_mm = float(cfg.get("beam_spacing_mm", 0.0))
        tissue_downsample_mm = float(cfg.get("tissue_downsample_mm", 0.0))
        num_bsms_levels = int(cfg.get("num_bsms_levels", 2))

        _shared_dataset_kwargs = dict(
            data_dir=data_dir,
            needle_crop_mm=cfg.needle_crop_mm,
            tissue_crop_mm=cfg.tissue_crop_mm,
            slice_half_thickness_mm=cfg.slice_half_thickness_mm,
            full_needle_tissue_mm=cfg.full_needle_tissue_mm,
            crop_strategy_weights=crop_strategy_weights,
            train_fraction=cfg.train_fraction,
            val_fraction=cfg.val_fraction,
            stats_path=stats_dir,
            cache_dir=data_dir,
            timestep_stride=cfg.get("timestep_stride", 1),
            use_cpress=use_cpress,
            per_region_norm=per_region_norm,
            max_frames_per_run=max_frames_per_run,
            beam_spacing_mm=beam_spacing_mm,
            tissue_downsample_mm=tissue_downsample_mm,
            use_bsms=self.use_bsms,
            num_bsms_levels=num_bsms_levels,
            vector_iso_norm=bool(cfg.get("vector_iso_norm", False)),
            needle_fiber_axis=bool(cfg.get("needle_fiber_axis", False)),
            drop_targets=list(cfg.get("drop_targets", []) or []),
            mgn_paper_features=bool(cfg.get("mgn_paper_features", False)),
            mgn_include_mat_fiber=bool(cfg.get("mgn_include_mat_fiber", False)),
            mgn_include_prev_v=bool(cfg.get("mgn_include_prev_v", False)),
            mgn_include_evf=bool(cfg.get("mgn_include_evf", False)),
            mgn_kinematic_needle_only=bool(cfg.get("mgn_kinematic_needle_only", False)),
            multistep_K=int(cfg.get("multistep_K", 1)),
        )
        train_dataset = NeedleTissueDataset(split="train", **_shared_dataset_kwargs)
        # Val keeps K=1 so the metric is the standard 1-step rel-err and
        # short val runs aren't pruned for lack of K future frames.
        _val_kwargs = dict(_shared_dataset_kwargs)
        _val_kwargs["multistep_K"] = 1
        val_dataset = NeedleTissueDataset(split="validation", **_val_kwargs)
        # Stash so validation() can label its rel-err output by the actual
        # TARGET_KEYS (which depend on use_cpress and drop_targets) rather
        # than the previously-hardcoded ["u","v","a"].
        self._target_keys = train_dataset.TARGET_KEYS
        self._target_dims = train_dataset.TARGET_DIMS

        train_sampler = DistributedSampler(
            train_dataset,
            shuffle=True,
            drop_last=True,
            num_replicas=dist.world_size,
            rank=dist.rank,
        )

        self.dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            sampler=train_sampler,
            collate_fn=_bsms_collate if self.use_bsms else _collate,
            pin_memory=True,
            num_workers=cfg.num_workers,
        )
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=_bsms_collate if self.use_bsms else _collate,
            pin_memory=True,
            num_workers=cfg.num_workers,
        )

        model_type = str(cfg.get("model_type", "mgn")).lower()
        _shared_kwargs = dict(
            input_dim_nodes=train_dataset.input_dim_nodes,
            input_dim_edges=cfg.input_dim_edges,
            output_dim=train_dataset.output_dim,
            processor_size=cfg.processor_size,
            hidden_dim_node_encoder=cfg.hidden_dim_node_encoder,
            hidden_dim_edge_encoder=cfg.hidden_dim_edge_encoder,
            hidden_dim_node_decoder=cfg.hidden_dim_node_decoder,
            hidden_dim_processor=cfg.hidden_dim_processor,
            aggregation=cfg.aggregation,
        )
        if model_type == "bistride":
            self.model = BiStrideMeshGraphNet(
                **_shared_kwargs,
                num_mesh_levels=int(cfg.get("num_bsms_levels", 2)),
                bistride_pos_dim=3,
                num_layers_bistride=int(cfg.get("num_layers_bistride", 2)),
                bistride_unet_levels=int(cfg.get("bistride_unet_levels", 1)),
                num_processor_checkpoint_segments=int(cfg.get("num_processor_checkpoint_segments", 0)),
            )
        elif model_type == "kan":
            self.model = MeshGraphKAN(
                **_shared_kwargs,
                num_harmonics=int(cfg.get("num_harmonics", 5)),
            )
        elif model_type == "fiber":
            self.model = FiberEquivariantMGN(
                **_shared_kwargs,
                n_vec_outputs=int(cfg.get("n_vec_outputs", 3)),
                extra_edge_invariants=bool(cfg.get("fiber_extra_invariants", False)),
                extra_decoder_basis=bool(cfg.get("fiber_extra_decoder_basis", False)),
            )
        elif model_type == "fiber_kan":
            self.model = FiberEquivariantKAN(
                **_shared_kwargs,
                n_vec_outputs=int(cfg.get("n_vec_outputs", 3)),
                num_harmonics=int(cfg.get("num_harmonics", 5)),
                extra_edge_invariants=bool(cfg.get("fiber_extra_invariants", False)),
                extra_decoder_basis=bool(cfg.get("fiber_extra_decoder_basis", False)),
            )
        elif model_type == "tfn":
            n_tfn_scalar = train_dataset.n_tfn_scalar
            # n_edge_extra_scalar = (everything in edge_attr after the first 3
            # columns of physical rel_pos).  Standard layout: 7 - 3 = 4.
            # MGN-paper layout: 11 - 3 = 8.
            self.model = TFNMeshGraphNet(
                n_node_scalar=n_tfn_scalar,
                n_node_vec=train_dataset.n_tfn_vec,
                output_dim=train_dataset.output_dim,
                irreps_hidden=str(cfg.get("irreps_hidden", "16x0e + 8x1o + 4x2e")),
                l_max=int(cfg.get("l_max", 2)),
                n_radial_basis=int(cfg.get("n_radial_basis", 8)),
                r_max=float(cfg.get("r_max", 60.0)),
                n_edge_extra_scalar=int(cfg.input_dim_edges) - 3,
                processor_size=cfg.processor_size,
                n_vec_outputs=int(cfg.get("n_vec_outputs", 3)),
                checkpoint_layers=bool(cfg.get("tfn_checkpoint_layers", True)),
            )
        else:
            self.model = MeshGraphNet(
                **_shared_kwargs,
                use_fourier_features=cfg.get("use_fourier_features", False),
                n_fourier_features=cfg.get("n_fourier_features", 64),
                fourier_scale=cfg.get("fourier_scale", 1.0),
            )

        if cfg.jit:
            self.model = torch.compile(self.model.to(dist.device))
        else:
            self.model = self.model.to(dist.device)

        if dist.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[dist.local_rank],
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )

        self.model.train()
        self.criterion = torch.nn.MSELoss()

        try:
            self.optimizer = apex.optimizers.FusedAdam(
                self.model.parameters(), lr=cfg.lr
            )
            rank_zero_logger.info("Using FusedAdam optimizer")
        except Exception:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.epochs * len(self.dataloader),
            eta_min=cfg.lr * 0.01,
        )
        self.scaler = GradScaler()

        # ---- Multi-step rollout curriculum --------------------------------
        self.multistep_K = int(cfg.get("multistep_K", 1))
        self.multistep_w_max = float(cfg.get("multistep_w_max", 0.7))
        self.multistep_warmup_epochs = int(cfg.get("multistep_warmup_epochs", 5))
        self.world_edge_radius = float(cfg.get("world_edge_radius", 1.2))
        self._multistep_total_epochs = int(cfg.epochs)
        self._current_w = 0.0
        if self.multistep_K > 1:
            if bool(cfg.get("per_region_norm", False)) or bool(cfg.get("mgn_paper_features", False)):
                raise ValueError(
                    "multistep_K>1 currently requires per_region_norm=false and "
                    "mgn_paper_features=false (rollout state-update path)."
                )
            if int(cfg.batch_size) != 1:
                raise ValueError("multistep_K>1 requires batch_size=1.")
            self._build_rollout_stats(train_dataset)

        if dist.world_size > 1:
            torch.distributed.barrier()
        self.epoch_init = load_checkpoint(
            _abspath(cfg.ckpt_path),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=dist.device,
        )

    def _build_rollout_stats(self, dataset):
        """Cache the (target_std/input_std) ratios needed to update the
        normalised x in-place after each rollout step.

        For each TARGET_KEY k that maps to an INPUT_KEY (u, v, a, evf, s,
        cpress), compute the slice of x that holds that input feature, the
        slice of pred that holds the corresponding output, and the per-channel
        scaling tensors used by the update rule

            Δx_norm = Δf_raw / input_std,    Δf_raw = pred_norm * tgt_std + tgt_mean.
        """
        device = self.dist.device
        node_stats = dataset._node_stats
        target_stats = dataset._target_stats

        # Column offsets in pred (concat of TARGET_KEYS).
        tgt_offsets = {}
        off = 0
        for k, d in zip(dataset.TARGET_KEYS, dataset.TARGET_DIMS):
            tgt_offsets[k] = (off, off + d)
            off += d

        # Column offsets in x (concat of INPUT_KEYS, then STATIC_PROP_KEYS).
        in_offsets = {}
        off = 0
        for k, d in zip(dataset.INPUT_KEYS, dataset.INPUT_DIMS):
            in_offsets[k] = (off, off + d)
            off += d

        # u-target updates BOTH u-input and coord-input; record both.
        self._rollout_updates = []  # list of (tgt_slice, input_slice_or_None, ratio, bias)
        for tkey in dataset.TARGET_KEYS:
            if tkey not in in_offsets:
                # Target without a matching input feature (shouldn't happen
                # in current setup, but skip just in case).
                continue
            t_lo, t_hi = tgt_offsets[tkey]
            i_lo, i_hi = in_offsets[tkey]
            t_mean = target_stats[f"{tkey}_mean"].to(device).float()
            t_std = target_stats[f"{tkey}_std"].to(device).float()
            i_std = node_stats[f"{tkey}_std"].to(device).float()
            ratio = (t_std / i_std).view(1, -1)
            bias = (t_mean / i_std).view(1, -1)
            self._rollout_updates.append(("input", tkey, (t_lo, t_hi), (i_lo, i_hi), ratio, bias))

        # u-target also updates coord-input (raw): coord_norm += Δu_raw / coord_std.
        if "u" in tgt_offsets and "coord" in in_offsets:
            t_lo, t_hi = tgt_offsets["u"]
            c_lo, c_hi = in_offsets["coord"]
            t_mean = target_stats["u_mean"].to(device).float()
            t_std = target_stats["u_std"].to(device).float()
            c_std = node_stats["coord_std"].to(device).float()
            ratio = (t_std / c_std).view(1, -1)
            bias = (t_mean / c_std).view(1, -1)
            self._rollout_updates.append(("coord", "u", (t_lo, t_hi), (c_lo, c_hi), ratio, bias))

        # u-target also updates pos (raw mm).  Δpos_raw = pred_norm * t_std + t_mean.
        self._u_tgt_slice = tgt_offsets.get("u")
        self._u_t_mean = target_stats["u_mean"].to(device).float().view(1, -1)
        self._u_t_std = target_stats["u_std"].to(device).float().view(1, -1)

        # v-target updates node_velocity: v_norm += Δv_raw / v_input_std.
        if "v" in tgt_offsets:
            t_lo, t_hi = tgt_offsets["v"]
            t_mean = target_stats["v_mean"].to(device).float().view(1, -1)
            t_std = target_stats["v_std"].to(device).float().view(1, -1)
            v_std = node_stats["v_std"].to(device).float().view(1, -1)
            self._v_update = ((t_lo, t_hi), t_std / v_std, t_mean / v_std)
        else:
            self._v_update = None

        # Hex edge type one-hots (col 2 = world).  Cached for rebuilding.
        self._world_edge_radius = self.world_edge_radius

    def _rebuild_edges(self, pos, edge_index, edge_attr, is_needle):
        """Rebuild contact (world) edges from current pos; keep hex edges fixed.

        Returns (new_edge_index, new_edge_attr) where edge_attr's first 4
        columns (rel_pos, edge_len) are computed against the live pos tensor
        so gradients flow through the geometry.  Edge connectivity itself is
        a discrete np.int64 tensor and carries no gradient — that's fine.
        """
        # Identify hex (mesh) vs world edges by the one-hot in the trailing
        # 3 columns of edge_attr.
        type_oh = edge_attr[:, -3:]
        is_world_edge = type_oh[:, 2] > 0.5
        hex_mask = ~is_world_edge

        hex_ei = edge_index[:, hex_mask]
        hex_oh = type_oh[hex_mask]

        # Rebuild world edges from current needle / tissue positions.
        pos_np = pos.detach().cpu().numpy()
        needle_local = torch.nonzero(is_needle, as_tuple=False).squeeze(-1)
        tissue_local = torch.nonzero(~is_needle, as_tuple=False).squeeze(-1)
        if needle_local.numel() == 0 or tissue_local.numel() == 0:
            new_ei = hex_ei
            new_oh = hex_oh
        else:
            tissue_pts = pos_np[tissue_local.cpu().numpy()]
            needle_pts = pos_np[needle_local.cpu().numpy()]
            tree = cKDTree(tissue_pts)
            neigh = tree.query_ball_point(needle_pts, r=self._world_edge_radius)
            src_l, dst_l = [], []
            n_loc_np = needle_local.cpu().numpy()
            t_loc_np = tissue_local.cpu().numpy()
            for i, nbrs in enumerate(neigh):
                if not nbrs:
                    continue
                ng = int(n_loc_np[i])
                for j in nbrs:
                    src_l.append(ng)
                    dst_l.append(int(t_loc_np[j]))
            if not src_l:
                new_ei = hex_ei
                new_oh = hex_oh
            else:
                src_t = torch.tensor(src_l, dtype=torch.long, device=pos.device)
                dst_t = torch.tensor(dst_l, dtype=torch.long, device=pos.device)
                world_ei = torch.stack(
                    [torch.cat([src_t, dst_t]), torch.cat([dst_t, src_t])], dim=0
                )
                world_oh = torch.zeros(world_ei.shape[1], 3, dtype=hex_oh.dtype, device=pos.device)
                world_oh[:, 2] = 1.0
                new_ei = torch.cat([hex_ei, world_ei], dim=1)
                new_oh = torch.cat([hex_oh, world_oh], dim=0)

        src, dst = new_ei
        rel_pos = pos[src] - pos[dst]
        edge_len = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)
        new_attr = torch.cat([rel_pos, edge_len, new_oh], dim=-1)
        # If edge_attr had extra columns (mgn_paper_features), they would sit
        # between edge_len and the type one-hot.  Multistep is gated to
        # mgn_paper_features=false so the standard 7-col layout always holds.
        return new_ei, new_attr

    def _apply_rollout_step(self, graph, pred):
        """Update graph state in place using the model's prediction.

        Updates: graph.x (input feature normalised state), graph.pos (raw),
        graph.node_velocity (normalised v), graph.edge_index, graph.edge_attr.
        Also recomputes graph.x_scalar / graph.x_vec splits to mirror the
        dataset (TFN path); for non-TFN models these aren't read but we keep
        them consistent.
        """
        # Run state update in float32 to avoid autocast-induced precision
        # drift across many rollout steps.
        pred = pred.float()
        x = graph.x.float()
        new_x = x.clone()

        # Per-target updates to x.
        for _kind, _key, t_slice, i_slice, ratio, bias in self._rollout_updates:
            t_lo, t_hi = t_slice
            i_lo, i_hi = i_slice
            pred_slice = pred[:, t_lo:t_hi]
            new_x[:, i_lo:i_hi] = new_x[:, i_lo:i_hi] + ratio * pred_slice + bias

        # Raw Δu for pos update and node_velocity (computed separately below).
        u_lo, u_hi = self._u_tgt_slice
        u_pred_norm = pred[:, u_lo:u_hi]
        u_raw = u_pred_norm * self._u_t_std + self._u_t_mean
        new_pos = graph.pos + u_raw

        # node_velocity update via v target.
        new_nv = graph.node_velocity
        if self._v_update is not None:
            (v_lo, v_hi), v_ratio, v_bias = self._v_update
            v_pred = pred[:, v_lo:v_hi]
            new_nv = graph.node_velocity + v_ratio * v_pred + v_bias

        # Rebuild edges (hex fixed, world recomputed from new_pos).
        new_ei, new_attr = self._rebuild_edges(
            new_pos, graph.edge_index, graph.edge_attr, graph.is_needle
        )
        graph.x = new_x
        graph.pos = new_pos
        graph.edge_index = new_ei
        graph.edge_attr = new_attr
        graph.node_velocity = new_nv
        return graph

    def _masked_mse(self, pred, y, mask):
        if mask is not None:
            diff_sq = (pred - y) ** 2
            return (diff_sq * mask).sum() / mask.sum().clamp(min=1.0)
        return self.criterion(pred, y)

    def _curriculum_w(self, epoch):
        if self.multistep_K <= 1:
            return 0.0
        if epoch < self.multistep_warmup_epochs:
            return 0.0
        denom = max(1, self._multistep_total_epochs - self.multistep_warmup_epochs)
        frac = (epoch - self.multistep_warmup_epochs) / denom
        return min(1.0, frac) * self.multistep_w_max

    def _unpack_batch(self, batch):
        """Return (graph, ms_edges, ms_ids) regardless of BSMS mode."""
        if self.use_bsms:
            graph = batch["graph"].to(self.dist.device)
            ms_edges = [e.to(self.dist.device) for e in batch["ms_edges"]]
            ms_ids = [ids.to(self.dist.device) for ids in batch["ms_ids"]]
        else:
            graph = batch.to(self.dist.device)
            ms_edges, ms_ids = [], []
        return graph, ms_edges, ms_ids

    def train(self, batch):
        self.optimizer.zero_grad()
        graph, ms_edges, ms_ids = self._unpack_batch(batch)
        loss = self.forward(graph, ms_edges, ms_ids)
        self.backward(loss)
        self.scheduler.step()
        return loss

    def forward(self, graph, ms_edges=(), ms_ids=()):
        with autocast(device_type=self.dist.device.type, enabled=self.amp):
            x, y = graph.x, graph.y
            if self.noise_std > 0.0:
                # Add Gaussian noise to normalised u/v/a inputs (indices 3:12).
                # Subtract same noise from target so the model learns to predict
                # the true increment from the noisy current state.
                noise = torch.randn(x.shape[0], 9, device=x.device, dtype=x.dtype) * self.noise_std
                x = x.clone()
                x[:, 3:12] = x[:, 3:12] + noise
                y = y.clone()
                y[:, :9] = y[:, :9] - noise
            if self.use_bsms:
                pred = self.model(x, graph.edge_attr, graph, ms_edges, ms_ids)
            else:
                pred = self.model(x, graph.edge_attr, graph)
            mask = getattr(graph, "loss_mask", None)
            l1 = self._masked_mse(pred, y, mask)

            w = self._current_w
            if self.multistep_K <= 1 or w <= 0.0:
                return l1

            # ---- K-step rollout branch ----
            # Update graph from the just-computed pred and unroll K-1 more
            # forward passes, comparing each to the cached future_deltas.
            future = graph.future_deltas  # (n_sub, K, output_dim)
            # L_K = mean MSE over all K rollout steps (k=0 is the 1-step
            # prediction we already made; subsequent steps unroll the model
            # autoregressively, rebuilding world edges each step).
            step_losses = [l1]
            self._apply_rollout_step(graph, pred)
            for k in range(1, self.multistep_K):
                pred_k = self.model(graph.x, graph.edge_attr, graph)
                target_k = future[:, k, :]
                step_losses.append(self._masked_mse(pred_k, target_k, mask))
                if k < self.multistep_K - 1:
                    self._apply_rollout_step(graph, pred_k)
            l_k = torch.stack(step_losses).mean()
            wandb.log({
                "train_l1": l1.detach().item(),
                "train_lk": l_k.detach().item(),
                "rollout_w": w,
            })
            return (1.0 - w) * l1 + w * l_k

    def backward(self, loss):
        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        wandb.log({"lr": self.get_lr()})

    def get_lr(self):
        for pg in self.optimizer.param_groups:
            return pg["lr"]

    @torch.no_grad()
    def validation(self):
        # Use the dataset's actual TARGET_KEYS so the labels track
        # use_cpress and drop_targets — otherwise (e.g. with
        # drop_targets=[u,a,evf] the output is [v(3), s(6)] but the
        # logged "val_u_rel_err" was secretly the v error and "val_v_rel_err"
        # / "val_a_rel_err" were halves of the stress error.
        keys = self._target_keys
        dims = self._target_dims
        errors = {k: 0.0 for k in keys}

        for batch in self.val_dataloader:
            graph, ms_edges, ms_ids = self._unpack_batch(batch)
            if self.use_bsms:
                pred = self.model(graph.x, graph.edge_attr, graph, ms_edges, ms_ids)
            else:
                pred = self.model(graph.x, graph.edge_attr, graph)
            mask = getattr(graph, "loss_mask", None)
            offset = 0
            for key, d in zip(keys, dims):
                p = pred[:, offset : offset + d]
                t = graph.y[:, offset : offset + d]
                if mask is not None:
                    # Per-key membership in the loss mask is uniform across
                    # the d columns — column 0 captures it.  Skip nodes
                    # zeroed out for this key (e.g. tissue nodes for u/v/a
                    # under mgn_kinematic_needle_only).
                    keep = mask[:, offset] > 0.5
                    if keep.any():
                        p = p[keep]
                        t = t[keep]
                    else:
                        offset += d
                        continue
                errors[key] += (
                    torch.linalg.norm(p - t) / torch.linalg.norm(t).clamp(min=1e-8)
                ).item()
                offset += d

        n = len(self.val_dataloader)
        log_dict = {f"val_{key}_rel_err": errors[key] / n for key in keys}
        wandb.log(log_dict)
        return log_dict


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.get("cuda_devices") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.cuda_devices)
    DistributedManager.initialize()
    dist = DistributedManager()

    initialize_wandb(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name="NeedleTissue-Cropped-MGN",
        group="NeedleTissue-DDP-Group",
        mode=cfg.wandb_mode,
    )

    logger = PythonLogger("main")
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)
    rank_zero_logger.file_logging()

    trainer = MGNTrainer(cfg, dist, rank_zero_logger)
    start = time.time()
    rank_zero_logger.info("Training started...")

    for epoch in range(trainer.epoch_init, cfg.epochs):
        trainer.dataloader.sampler.set_epoch(epoch)
        trainer._current_w = trainer._curriculum_w(epoch)
        loss_agg = 0.0
        for batch in trainer.dataloader:
            loss = trainer.train(batch)
            loss_agg += loss.detach().item()
        loss_agg /= len(trainer.dataloader)
        rank_zero_logger.info(
            f"epoch: {epoch}, loss: {loss_agg:10.3e}, "
            f"lr: {trainer.get_lr():.3e}, "
            f"time: {(time.time() - start):.1f}s"
        )
        wandb.log({"train_loss": loss_agg, "epoch": epoch})

        if dist.rank == 0 and (epoch + 1) % cfg.val_every == 0:
            val_errs = trainer.validation()
            rank_zero_logger.info(
                "  val errors: "
                + ", ".join(f"{k}={v:.4f}" for k, v in val_errs.items())
            )

        if dist.world_size > 1:
            torch.distributed.barrier()
        if dist.rank == 0 and (epoch + 1) % cfg.save_every == 0:
            ckpt_path = _abspath(cfg.ckpt_path)
            save_checkpoint(
                ckpt_path,
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
            )
            rank_zero_logger.info(f"Checkpoint saved at epoch {epoch}")
            if cfg.get("wandb_log_artifact", False):
                cfg_file = os.path.join(ckpt_path, "config.yaml")
                with open(cfg_file, "w") as f:
                    f.write(OmegaConf.to_yaml(cfg, resolve=True))
                artifact = wandb.Artifact(
                    name="needle-tissue-cropped-mgn",
                    type="model",
                    metadata={"epoch": epoch, "train_loss": loss_agg},
                )
                artifact.add_dir(ckpt_path)
                wandb.log_artifact(artifact)
                rank_zero_logger.info(f"Checkpoint logged as wandb artifact at epoch {epoch}")
        start = time.time()

    rank_zero_logger.info("Training complete.")


if __name__ == "__main__":
    main()
