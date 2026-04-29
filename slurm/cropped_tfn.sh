#!/bin/bash
#SBATCH --job-name=ndl_crp_tfn
#SBATCH --account=fuge-prj-eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:1
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_tfn/slurm-%j.out
#SBATCH --error=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_tfn/slurm-%j.err

# Experiment: TFNMeshGraphNet — SE(3)-equivariant Tensor Field Network.
#
# Uses real spherical harmonics Y_ℓ(r̂_ij) up to l_max=2 and a Gaussian radial
# basis to compute per-edge equivariant tensor product messages.  Scalar node
# inputs (evf, s, material scalars) are encoded as 0e irreps; vector inputs
# (u, v, a, mat_fiber) are encoded as 1o irreps.  The decoder projects hidden
# irreps to n_vec_outputs=3 polar vectors (u/v/a) plus scalar outputs.
#
# Speed-optimised config targeting ~10x faster training vs the 16/8/4 ps=15
# checkpoint baseline (4300 s/epoch → ~430 s/epoch on A100):
#   - irreps_hidden "8x0e + 4x1o + 2x2e": weight_numel 864→216 (4x fewer weights/edge)
#   - l_max=1: drops l=2 SH paths, further reducing TP cost (~1.5x)
#   - processor_size=5: 3x fewer layers (no recompute needed → ckpt off)
#   - timestep_stride=15: 1.5x fewer training steps/epoch vs stride=10
# Combined measured speedup: ~11.5x.  Parameter count: ~96 K.

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
EXP=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_tfn

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
        model_type=tfn \
        n_vec_outputs=3 \
        'irreps_hidden=8x0e + 4x1o + 2x2e' \
        l_max=1 \
        n_radial_basis=8 \
        r_max=60.0 \
        processor_size=15 \
        tfn_checkpoint_layers=false \
        ++per_region_norm=false \
        save_every=10 \
        cuda_devices=null
