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

"""Training script for DCEL-DoMINO on the needle-tissue dataset.

Uses volume-only mode.  The needle nodes form the Lagrangian geometry that
drives the geometry convolution.  All nodes (needle + tissue) are query points
for the volume output.  Per-node physical state (u, v, a, evf, s, cf) is
injected into the volume position encoder via the node_state_dim_vol extension.

Usage (from examples/cfd/needle_tissue_domino/):
    uv run python train.py
"""

import os

import hydra
import torch
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import NODE_STATE_DIM, NeedleTissueDominoDataset
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.models.domino.model import DoMINO
from physicsnemo.utils import load_checkpoint, save_checkpoint


def _collate_squeeze(batch):
    """Collate a single-item batch: move batch dim from dataset (1, ...) to loader dim."""
    # Each item already has a leading batch-1 dim from the dataset.
    # With DataLoader batch_size=1 we'd get (1, 1, ...).  Squeeze the outer one.
    assert len(batch) == 1, "DCEL-DoMINO requires batch_size=1"
    return batch[0]


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.get("cuda_devices") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.cuda_devices)
    DistributedManager.initialize()
    dist = DistributedManager()

    data_dir = to_absolute_path(cfg.data_dir)
    stats_dir = to_absolute_path(cfg.stats_dir)
    ckpt_dir = to_absolute_path(cfg.ckpt_path)
    os.makedirs(ckpt_dir, exist_ok=True)

    if dist.rank == 0:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            mode=cfg.wandb_mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    grid_res = tuple(cfg.grid_res)

    # ---- Datasets -----------------------------------------------------------
    train_dataset = NeedleTissueDominoDataset(
        data_dir=data_dir,
        split="train",
        grid_res=grid_res,
        train_fraction=cfg.train_fraction,
        val_fraction=cfg.val_fraction,
        stats_path=stats_dir,
        num_sample_nodes=cfg.get("num_sample_nodes", None),
    )
    val_dataset = NeedleTissueDominoDataset(
        data_dir=data_dir,
        split="validation",
        grid_res=grid_res,
        train_fraction=cfg.train_fraction,
        val_fraction=cfg.val_fraction,
        stats_path=stats_dir,
        num_sample_nodes=cfg.get("num_sample_nodes", None),
    )

    train_sampler = DistributedSampler(
        train_dataset, shuffle=True, drop_last=True,
        num_replicas=dist.world_size, rank=dist.rank,
    )
    val_sampler = DistributedSampler(
        val_dataset, shuffle=False, drop_last=False,
        num_replicas=dist.world_size, rank=dist.rank,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=1, sampler=train_sampler,
        num_workers=cfg.num_workers, collate_fn=_collate_squeeze,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, sampler=val_sampler,
        num_workers=cfg.num_workers, collate_fn=_collate_squeeze,
        pin_memory=False,
    )

    # ---- Model --------------------------------------------------------------
    model = DoMINO(
        input_features=3,                          # needle xyz only in geometry conv
        output_features_vol=cfg.output_dim,        # Δu Δv Δa = 9
        output_features_surf=None,                 # volume-only mode
        global_features=cfg.global_features,
        model_parameters=OmegaConf.to_container(cfg.model, resolve=True),
        node_state_dim_vol=NODE_STATE_DIM,         # 30: u v a evf s cf cpress + 8 mat props
        use_fourier_features_state=cfg.get("use_fourier_features_state", False),
        n_fourier_features_state=cfg.get("n_fourier_features_state", 64),
        fourier_scale_state=cfg.get("fourier_scale_state", 1.0),
    ).to(dist.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = GradScaler(enabled=cfg.amp)

    load_checkpoint(ckpt_dir, models=model, optimizer=optimizer, device=dist.device)

    # ---- Training loop ------------------------------------------------------
    for epoch in range(cfg.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(dist.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            y_true = batch.pop("y")                              # (1, N, 9)

            optimizer.zero_grad()
            with autocast("cuda", enabled=cfg.amp):
                pred_vol, _ = model(batch)                       # (1, N, 9)
                loss = torch.nn.functional.mse_loss(pred_vol, y_true)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        # ---- Validation -----------------------------------------------------
        if (epoch + 1) % cfg.val_every == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(dist.device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                    y_true = batch.pop("y")
                    with autocast("cuda", enabled=cfg.amp):
                        pred_vol, _ = model(batch)
                        val_loss += torch.nn.functional.mse_loss(pred_vol, y_true).item()
            val_loss /= max(len(val_loader), 1)

            if dist.rank == 0:
                print(
                    f"Epoch {epoch + 1:4d}/{cfg.epochs}  "
                    f"train={train_loss:.6f}  val={val_loss:.6f}"
                )
                wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch + 1})
        elif dist.rank == 0:
            print(f"Epoch {epoch + 1:4d}/{cfg.epochs}  train={train_loss:.6f}")
            wandb.log({"train_loss": train_loss, "epoch": epoch + 1})

        # ---- Checkpoint -----------------------------------------------------
        if (epoch + 1) % cfg.save_every == 0 and dist.rank == 0:
            save_checkpoint(
                ckpt_dir,
                models=model,
                optimizer=optimizer,
                epoch=epoch,
                metadata={"train_loss": train_loss},
            )
            if cfg.get("wandb_log_artifact", False):
                cfg_file = os.path.join(ckpt_dir, "config.yaml")
                with open(cfg_file, "w") as f:
                    f.write(OmegaConf.to_yaml(cfg, resolve=True))
                artifact = wandb.Artifact(
                    name="needle-tissue-domino",
                    type="model",
                    metadata={"epoch": epoch, "train_loss": train_loss},
                )
                artifact.add_dir(ckpt_dir)
                wandb.log_artifact(artifact)

    if dist.rank == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
