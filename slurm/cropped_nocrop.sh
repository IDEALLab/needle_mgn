#!/bin/bash
#SBATCH --job-name=ndl_crp_full
#SBATCH --account=fuge-prj-eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:1
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_nocrop/slurm-%j.out
#SBATCH --error=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_nocrop/slurm-%j.err

# Experiment: Cropped MGN trained on the full mesh (no spatial cropping).
#
# The standard ablations crop the graph to the active needle-tissue insertion
# zone each step (needle_crop_mm=10, tissue_crop_mm=25).  This reduces the
# graph size but the model never sees the full needle geometry in one pass.
# Setting both crop radii to 10000 mm means every node is always included,
# so the model trains on the complete mesh at every step.
#
# Trade-off: each training sample is much larger (~10× more nodes), which
# increases memory use and slows per-step training.  The model does however
# see global context (shank, tissue far-field) that the cropped version misses.
# Inference in infer.py already uses the full mesh, so this makes train/infer
# conditions consistent.

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
EXP=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_nocrop

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
        needle_crop_mm=10000 \
        tissue_crop_mm=10000 \
        'crop_strategy_weights=[1,0,0]' \
        save_every=10 \
        cuda_devices=null
