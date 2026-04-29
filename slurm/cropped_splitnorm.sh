#!/bin/bash
#SBATCH --job-name=ndl_crp_snrm
#SBATCH --account=fuge-prj-eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:1
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_splitnorm/slurm-%j.out
#SBATCH --error=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_splitnorm/slurm-%j.err

# Experiment: Cropped MGN with per-region normalization (per_region_norm=true).
#
# Global normalization conflates needle and tissue statistics, causing scale
# mismatch for features that differ strongly between the two regions:
#   - cpress: non-zero only on needle contact nodes; global std is dominated
#     by the zero values from tissue nodes, leaving contact pressure
#     effectively un-normalised at the needle surface.
#   - s (stress tensor): orders of magnitude larger in the stiff needle than
#     in the soft tissue.
#
# With per_region_norm=true, separate mean/std are computed for needle and
# tissue nodes for every input and target feature.  The stats JSON is augmented
# with {key}_needle_mean / _std / _tissue_mean / _std entries; infer.py
# detects these automatically and applies the matching denormalization.
#
# Also enables cpress to test whether the normalization fix resolves the
# earlier degradation observed when adding cpress to the base model.

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
EXP=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_splitnorm

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
        use_cpress=true \
        per_region_norm=true \
        timestep_stride=10 \
        save_every=10 \
        cuda_devices=null
