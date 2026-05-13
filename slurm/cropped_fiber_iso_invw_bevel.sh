#!/bin/bash
#SBATCH --job-name=ndl_crp_fbr_invw_bevel
#SBATCH --account=fuge-prj-eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:1
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_invw_bevel/slurm-%j.out
#SBATCH --error=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_invw_bevel/slurm-%j.err

# Variant of cropped_fiber_iso_invw with an additional equivariant 1o
# node input: the outward unit normal of the *bevel face* of the needle,
# applied to needle nodes that touch a bevel-classified surface face;
# zero elsewhere.
#
# Plumbing:
#   1. compute_needle_geometry.py is run ONCE before training to
#      precompute the bevel-face and surface-face per-node normals from
#      the fixed needle HEX mesh and save them to
#      <data_dir>/needle_geometry_features.pt.
#   2. The dataset loads this file when bevel_normal_feature=true and
#      attaches `graph.extra_node_vec` per sample (cropped to the
#      sample's needle subset).
#   3. FiberEquivariantMGN(extra_node_vec=True) appends 3 extra edge
#      invariants — cos_theta_g, cos_phi_g, |g_src| — to the edge
#      encoder.  No change to the equivariant decoder basis.
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
EXP=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_invw_bevel

mkdir -p ${EXP}/checkpoints ${EXP}/stats ${EXP}/outputs

# Precompute the bevel-face / surface-face normals once if missing.
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
        ++bevel_normal_feature=true \
        ++surface_contact_normal_feature=false \
        save_every=10 \
        cuda_devices=null
