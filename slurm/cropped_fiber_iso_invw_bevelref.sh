#!/bin/bash
#SBATCH --job-name=ndl_crp_fbr_invw_bevelref
#SBATCH --account=fuge-prj-eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:1
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_invw_bevelref/slurm-%j.out
#SBATCH --error=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_invw_bevelref/slurm-%j.err

# Variant of cropped_fiber_iso_invw with a LAB-FIXED reference in the
# displacement decoder:
#
#   displacement_bevel_ref=true
#   bevel_axis=[1,0,0]                 (lab-frame bevel steering direction)
#       The vector (displacement) decoder basis gains a fixed lab-frame vector
#       — the needle bevel axis — whose coefficient is still produced from the
#       rotation-invariant decoder features.  So the displacement output can
#       carry a lab-fixed component (the bevel-steering deflection) that the
#       purely equivariant basis {V, d, V×d, W, W×d, C} cannot represent.
#
# The scalar / constitutive head (stress S, contact pressure, EVF) is UNCHANGED
# and stays rotation-invariant — only the displacement head sees the lab-fixed
# reference.
#
# Why
# ---
# The needle deflection has a real lab-fixed component (bevel steering, ~the
# bevel axis direction) that an SE(3)-equivariant decoder fundamentally cannot
# produce (it has no preferred lab direction).  This was the un-fixable part of
# the bias in the analysis: rotation augmentation made it worse, and even a
# perfectly equivariant model can only emit deflection built from {V, d, ...}.
# Injecting the bevel axis as a fixed basis vector gives the displacement head
# exactly the lab-frame degree of freedom it was missing, while keeping the
# constitutive outputs equivariant.
#
# bevel_axis=[1,0,0] is the in-plane transverse steering direction for this
# dataset (the bevel symmetry plane is X-Z; the needle axis is +Z).  If the
# dataset's bevel orientation changes, update bevel_axis accordingly.
#
# All other settings match cropped_fiber_iso_invw.

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
EXP=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_invw_bevelref

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
        ++displacement_bevel_ref=true \
        'bevel_axis=[1.0, 0.0, 0.0]' \
        save_every=10 \
        cuda_devices=null
