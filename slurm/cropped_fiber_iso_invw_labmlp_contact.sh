#!/bin/bash
#SBATCH --job-name=ndl_crp_fbr_invw_labmlp_contact
#SBATCH --account=fuge-prj-eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:1
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_invw_labmlp_contact/slurm-%j.out
#SBATCH --error=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_invw_labmlp_contact/slurm-%j.err

# cropped_fiber_iso_invw_labmlp (non-equivariant lab-frame kinematic head) PLUS
# an equivariant feature derived from the needle-tissue contact-surface normal:
#
#   displacement_lab_mlp=true            (lab-frame MLP kinematic head)
#   surface_contact_normal_feature=true  (equivariant contact-surface-normal feature)
#       Ships the outward unit surface normal of the needle (precomputed by
#       compute_needle_geometry.py), masked per frame to needle-surface nodes
#       that have at least one world (contact) edge — i.e. the normal of the
#       live needle-tissue contact interface; zero elsewhere.  The model reads
#       it as a per-node 1o vector (graph.extra_node_vec) and augments the edge
#       encoder with three ROTATION-INVARIANT scalars
#         cos(normal, ê_ij),  cos(normal_i, normal_j),  |normal_i|
#       so the contact-interface geometry informs the model EQUIVARIANTLY (the
#       feature co-rotates with the scene; only invariants enter the encoder).
#
# Why
# ---
# The lab-frame kinematic head is free to fit lab-fixed deflection, but it had
# no physically-grounded contact signal.  The contact-surface normal is the
# direction the tissue reacts against the needle at the contact interface — a
# natural driver of transverse deflection.  Feeding it as an equivariant
# feature lets both the constitutive head and (via the shared invariant node
# embeddings) the lab kinematic head use the contact geometry, without breaking
# the equivariance of everything except the displacement head.
#
# Requires needle_geometry_features.pt (surface_node_normal); generated below
# if absent (same step as cropped_fiber_iso_invw_contact).
#
# All other settings match cropped_fiber_iso_invw_labmlp.

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
EXP=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_invw_labmlp_contact

mkdir -p ${EXP}/checkpoints ${EXP}/stats ${EXP}/outputs

# Precompute needle geometry (surface normals) if not already present.
if [ ! -f ${DATA}/needle_geometry_features.pt ]; then
    apptainer exec --nv \
        --bind ${DATA}:/data/RUN-2 \
        ${SIF} \
        uv run python /opt/needle_mgn/examples/cfd/needle_tissue_cropped/compute_needle_geometry.py \
            --data_dir /data/RUN-2
fi

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
        ++bevel_normal_feature=false \
        ++surface_contact_normal_feature=true \
        save_every=10 \
        cuda_devices=null
