#!/bin/bash
#SBATCH --job-name=ndl_crp_str1
#SBATCH --account=fuge-prj-eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=256G
#SBATCH --gpus=a100:1
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_stride1/slurm-%j.out
#SBATCH --error=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_stride1/slurm-%j.err

# Experiment: Cropped MGN with timestep_stride=1 (single-frame increments).
#
# The standard ablations use timestep_stride=10, so each training sample
# predicts a 10-frame state increment.  With stride=1 the model learns the
# smallest available increment, which may be easier to learn and should
# reduce rollout drift.  Inference uses n_rollout=200 so the rollout covers
# the same physical duration as stride=10 / n_rollout=20.
#
# Memory: stride=1 loads 201 frames/run vs ~20 at stride=10.  To keep RAM
# comparable to the base experiment, max_frames_per_run=25 uniformly
# subsamples 25 frames from the 201 cached frames.  The full per-run caches
# (all 201 frames) are still written to disk on first run.  We also request
# 256G of RAM to handle cache-build peak before subsampling takes effect.
# --mem=256G is added to the SBATCH directives above.

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
EXP=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_stride1

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
        timestep_stride=1 \
        max_frames_per_run=25 \
        n_rollout=200 \
        save_every=10 \
        cuda_devices=null
