#!/bin/bash
#SBATCH --job-name=ndl_crp_fbr_invw_rotaug
#SBATCH --account=fuge-prj-eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:1
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_invw_rotaug/slurm-%j.out
#SBATCH --error=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_invw_rotaug/slurm-%j.err

# Variant of cropped_fiber_iso_invw trained with rotation data augmentation:
#
#   needle_axis_rot_aug=true
#       Each training sample is rotated by a random angle about the (frame-0)
#       needle axis.  All 1o vector quantities (coord, u, v, a, mat_fiber)
#       co-rotate; rotation-invariant scalars / edge-invariants and the
#       scalar-decoded stress / evf are left unchanged (the fiber model decodes
#       those with a rotation-invariant head, and a rotation about the axis
#       preserves every invariant feature, so leaving them fixed is the
#       architecturally-consistent choice).  Augmentation is applied on the
#       train split only; val/test are unrotated.
#
# Motivation
# ----------
# The needle axis IS the rotation axis, so the needle's fiber direction maps to
# itself while transverse deflections rotate to every azimuth.  This exposes
# the (only approximately equivariant) fiber model to deflections in all
# transverse directions, directly targeting the x-z flattening / near-constant
# tip-deflection degeneracy without changing the architecture.
#
# Identical to cropped_fiber_iso_invw except for the added augmentation flag.

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
EXP=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_invw_rotaug

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
        noise_std=0 \
        use_cpress=false \
        timestep_stride=10 \
        model_type=fiber \
        n_vec_outputs=3 \
        hidden_dim_node_encoder=256 \
        hidden_dim_edge_encoder=256 \
        hidden_dim_node_decoder=256 \
        hidden_dim_processor=256 \
        processor_size=15 \
        ++per_region_norm=false \
        ++vector_iso_norm=true \
        ++needle_fiber_axis=true \
        ++fiber_extra_invariants=true \
        ++fiber_extra_decoder_basis=true \
        ++needle_axis_rot_aug=true \
        save_every=10 \
        cuda_devices=null
