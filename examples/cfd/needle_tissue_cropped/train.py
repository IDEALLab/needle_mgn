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
import time

import hydra
import torch
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
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

        stats_dir = to_absolute_path(cfg.stats_dir)
        data_dir = to_absolute_path(cfg.data_dir)

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
        )
        train_dataset = NeedleTissueDataset(split="train", **_shared_dataset_kwargs)
        val_dataset = NeedleTissueDataset(split="validation", **_shared_dataset_kwargs)

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
            )
        elif model_type == "fiber_kan":
            self.model = FiberEquivariantKAN(
                **_shared_kwargs,
                n_vec_outputs=int(cfg.get("n_vec_outputs", 3)),
                num_harmonics=int(cfg.get("num_harmonics", 5)),
            )
        elif model_type == "tfn":
            n_tfn_scalar = train_dataset.n_tfn_scalar
            self.model = TFNMeshGraphNet(
                n_node_scalar=n_tfn_scalar,
                n_node_vec=train_dataset.n_tfn_vec,
                output_dim=train_dataset.output_dim,
                irreps_hidden=str(cfg.get("irreps_hidden", "16x0e + 8x1o + 4x2e")),
                l_max=int(cfg.get("l_max", 2)),
                n_radial_basis=int(cfg.get("n_radial_basis", 8)),
                r_max=float(cfg.get("r_max", 60.0)),
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

        if dist.world_size > 1:
            torch.distributed.barrier()
        self.epoch_init = load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=dist.device,
        )

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
            return self.criterion(pred, y)

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
        keys = ["u", "v", "a"]
        dims = [3, 3, 3]
        errors = {k: 0.0 for k in keys}

        for batch in self.val_dataloader:
            graph, ms_edges, ms_ids = self._unpack_batch(batch)
            if self.use_bsms:
                pred = self.model(graph.x, graph.edge_attr, graph, ms_edges, ms_ids)
            else:
                pred = self.model(graph.x, graph.edge_attr, graph)
            offset = 0
            for key, d in zip(keys, dims):
                p = pred[:, offset : offset + d]
                t = graph.y[:, offset : offset + d]
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
            ckpt_path = to_absolute_path(cfg.ckpt_path)
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
