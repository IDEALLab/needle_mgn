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
from physicsnemo.models.meshgraphnet import MeshGraphNet
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils.logging.wandb import initialize_wandb


def _collate(batch):
    """Unwrap single-item batch into a PyG Batch object."""
    assert len(batch) == 1
    return Batch.from_data_list(batch)


class MGNTrainer:
    def __init__(self, cfg: DictConfig, dist, rank_zero_logger):
        self.dist = dist
        self.amp = cfg.amp

        stats_dir = to_absolute_path(cfg.stats_dir)
        data_dir = to_absolute_path(cfg.data_dir)

        crop_strategy_weights = tuple(cfg.crop_strategy_weights)
        train_dataset = NeedleTissueDataset(
            data_dir=data_dir,
            split="train",
            needle_crop_mm=cfg.needle_crop_mm,
            tissue_crop_mm=cfg.tissue_crop_mm,
            slice_half_thickness_mm=cfg.slice_half_thickness_mm,
            full_needle_tissue_mm=cfg.full_needle_tissue_mm,
            crop_strategy_weights=crop_strategy_weights,
            train_fraction=cfg.train_fraction,
            val_fraction=cfg.val_fraction,
            stats_path=stats_dir,
            cache_dir=data_dir,
        )
        val_dataset = NeedleTissueDataset(
            data_dir=data_dir,
            split="validation",
            needle_crop_mm=cfg.needle_crop_mm,
            tissue_crop_mm=cfg.tissue_crop_mm,
            slice_half_thickness_mm=cfg.slice_half_thickness_mm,
            full_needle_tissue_mm=cfg.full_needle_tissue_mm,
            crop_strategy_weights=crop_strategy_weights,
            train_fraction=cfg.train_fraction,
            val_fraction=cfg.val_fraction,
            stats_path=stats_dir,
            cache_dir=data_dir,
        )

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
            collate_fn=_collate,
            pin_memory=True,
            num_workers=cfg.num_workers,
        )
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=_collate,
            pin_memory=True,
            num_workers=cfg.num_workers,
        )

        self.model = MeshGraphNet(
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

    def train(self, batch):
        graph = batch.to(self.dist.device)
        self.optimizer.zero_grad()
        loss = self.forward(graph)
        self.backward(loss)
        self.scheduler.step()
        return loss

    def forward(self, graph):
        with autocast(device_type=self.dist.device.type, enabled=self.amp):
            pred = self.model(graph.x, graph.edge_attr, graph)
            return self.criterion(pred, graph.y)

    def backward(self, loss):
        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
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
            graph = batch.to(self.dist.device)
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
