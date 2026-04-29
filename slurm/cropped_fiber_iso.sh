#!/bin/bash
#SBATCH --job-name=ndl_crp_fiber_iso
#SBATCH --account=fuge-prj-eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:1
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso/slurm-%j.out
#SBATCH --error=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso/slurm-%j.err

# Experiment: FiberEquivariantMGN with the two equivariance fixes applied.
#
#   vector_iso_norm=true
#       Normalises u, v, a, mat_fiber by a single scalar std (mean=0) instead
#       of per-component (mean, std).  Per-component normalisation breaks the
#       1o-vector property of the targets — under rotation, the normalised
#       components don't transform as a vector — and the equivariant decoder
#       (whose weights can only uniformly scale a 1o irrep's xyz components)
#       cannot compensate.  This caused the un-fixed cropped_fiber to
#       underpredict needle xy-deflection by ~70% across all test runs.
#
#   needle_fiber_axis=true
#       Replaces mat_fiber on needle nodes (which odb_to_mgn_input.py exports
#       as zero) with the unit principal SVD axis of each run's frame-0 needle
#       coordinates.  Without this, the fiber decoder's {V, d, V×d} basis
#       collapses on needle nodes to just V (which lies along the needle axis
#       due to cylindrical symmetry of the message graph), so the model has no
#       equivariant basis vector pointing transverse to the needle and cannot
#       express deflection.  After the override, V × axis spans the bending
#       plane.
#
# All other architecture/training settings match cropped_fiber for a clean
# comparison.

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
EXP=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso

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
        save_every=10 \
        cuda_devices=null
