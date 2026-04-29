#!/bin/bash
#SBATCH --job-name=ndl_crp_tfn_iso
#SBATCH --account=fuge-prj-eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:1
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_tfn_iso/slurm-%j.out
#SBATCH --error=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_tfn_iso/slurm-%j.err

# Experiment: TFNMeshGraphNet with the two equivariance fixes applied.
#
# TFN is rigorously SE(3)-equivariant end-to-end (every layer is an o3.Linear
# or e3nn TensorProduct), so both fixes are *required* — not optional — for
# this model to function correctly.  Without them, anisotropic per-component
# target normalisation forces the model to produce normalised outputs whose
# components don't transform as a 1o vector under rotation, and the e3nn
# weights (which scale all 3 components of a 1o irrep together) can't
# compensate.  The same applies to the *input* x_vec channels (u, v, a,
# mat_fiber): with per-component normalisation, the input to the first
# o3.Linear is no longer a proper 1o vector either.
#
#   vector_iso_norm=true
#       Stores stats for u, v, a, mat_fiber as (mean=0, std=[s,s,s]) where s
#       is a single scalar across xyz components.  Both inputs (x_vec) and
#       targets remain valid 1o vectors after normalisation.
#
#   needle_fiber_axis=true
#       Overrides mat_fiber on needle nodes with the unit principal SVD axis
#       of frame-0 needle coordinates.  Without this, mat_fiber=0 for every
#       needle node — TFN sees a zero 1o input vector and has no mechanism to
#       distinguish needle node orientations from one another, leading to a
#       collapse to nearly-constant predictions across runs.
#
# Architecture and training settings otherwise match cropped_tfn.

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
EXP=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_tfn_iso

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
        ++vector_iso_norm=true \
        ++needle_fiber_axis=true \
        save_every=10 \
        cuda_devices=null
