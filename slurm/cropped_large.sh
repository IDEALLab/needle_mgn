#!/bin/bash
#SBATCH --job-name=ndl_crp_lrg
#SBATCH --account=fuge-prj-eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:1
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_large/slurm-%j.out
#SBATCH --error=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_large/slurm-%j.err

# Experiment: Cropped MGN with increased model capacity.
#
# The base model (hidden_dim=256, processor_size=15) showed degradation when
# adding features (cpress, Fourier) that expand the input/output dimensionality.
# A plausible cause is that the existing capacity is too small to represent
# the joint needle-tissue dynamics with additional channels.
#
# This experiment increases:
#   hidden_dim_{node_encoder,edge_encoder,node_decoder,processor}: 256 → 384
#   processor_size: 15 → 20
#
# The base feature set (no cpress, no Fourier) is used so any improvement is
# attributable to capacity alone.  A separate run combining large model + cpress
# can be done after confirming this improves on the base.

module load apptainer

export CUDA_VISIBLE_DEVICES=0
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 10000 ))
export WANDB_API_KEY=$(cat ~/.wandb_api_key)
export UV_CACHE_DIR=/tmp/uv-cache-${SLURM_JOB_ID}
export WARP_CACHE_PATH=/tmp/warp-cache-${SLURM_JOB_ID}
export WANDB_DATA_DIR=/tmp/wandb-${SLURM_JOB_ID}
export LOCAL_CACHE=/tmp/physicsnemo-cache-${SLURM_JOB_ID}

SIF=/home/nhoffma1/scratch.fuge-prj/needle_mgn/needle_mgn.sif
DATA=/scratch/zt1/project/fuge-prj/user/nhoffma1/needle_mgn/RUN-2
EXP=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_large

mkdir -p ${EXP}/checkpoints ${EXP}/stats ${EXP}/outputs

apptainer exec --nv \
    --bind ${DATA}:/data/RUN-2 \
    --bind ${EXP}:/opt/needle_mgn/results \
    ${SIF} \
    uv run python /opt/needle_mgn/examples/cfd/needle_tissue_cropped/train.py \
        wandb_mode=offline \
        wandb_project=PhysicsNeMo-Cropped-Ablation \
        data_dir=/data/RUN-2 \
        ckpt_path=/opt/needle_mgn/results/checkpoints \
        stats_dir=/opt/needle_mgn/results/stats \
        'hydra.run.dir=/opt/needle_mgn/results/outputs' \
        epochs=100 \
        batch_size=1 \
        noise_std=0.0 \
        use_fourier_features=false \
        use_cpress=false \
        per_region_norm=false \
        timestep_stride=10 \
        hidden_dim_node_encoder=384 \
        hidden_dim_edge_encoder=384 \
        hidden_dim_node_decoder=384 \
        hidden_dim_processor=384 \
        processor_size=20 \
        save_every=10 \
        cuda_devices=null
