#!/bin/bash
#SBATCH --job-name=ndl_crp_fbr_invw_labmlp
#SBATCH --account=fuge-prj-eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:1
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_invw_labmlp/slurm-%j.out
#SBATCH --error=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_invw_labmlp/slurm-%j.err

# Variant of cropped_fiber_iso_invw with a NON-EQUIVARIANT, lab-frame
# kinematic decoder:
#
#   displacement_lab_mlp=true
#       The displacement (u/v/a) outputs are decoded by a plain MLP head that
#       reads the same invariant node embeddings as the constitutive head plus
#       the raw per-node fiber direction (3), and emits u/v/a directly in lab
#       coordinates.  The equivariant basis {V, d, V×d, W, W×d} is removed from
#       the kinematic head entirely.
#
# The scalar / constitutive head (stress S, contact pressure, EVF) is UNCHANGED
# and stays equivariant — its inputs keep the invariant summaries (||V||, V·d,
# ||W||, W·d) of the fiber-anchored aggregates.
#
# Why
# ---
# Anchoring the displacement to the equivariant basis {V, d, V×d, ...} limited
# how well the model could represent the lab-fixed (bevel-steering) deflection;
# appending the bevel axis as a basis vector (cropped_fiber_iso_invw_bevelref)
# helped only a little.  This variant drops the equivariant constraint on the
# kinematics entirely: a free lab-frame MLP that sees the fiber orientation as
# an ordinary input feature, while the constitutive outputs stay equivariant.
#
# The processor still computes V/W (their invariant summaries feed the
# constitutive head), so fiber_extra_decoder_basis is kept on as in
# cropped_fiber_iso_invw.  All other settings match cropped_fiber_iso_invw.

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
EXP=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_invw_labmlp

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
        ++displacement_lab_mlp=true \
        save_every=10 \
        cuda_devices=null
