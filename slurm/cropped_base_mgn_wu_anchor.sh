#!/bin/bash
#SBATCH --job-name=ndl_crp_base_mgn_anchor
#SBATCH --account=fuge-prj-eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:1
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_base_mgn_wu_anchor/slurm-%j.out
#SBATCH --error=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_base_mgn_wu_anchor/slurm-%j.err

# Variant of cropped_base_mgn_wu (MGN-paper feature scheme, predicts u/v/s,
# drop_targets=[a]) with one additional absolute "anchor" node scalar:
#
#   mgn_include_arclen_clamp=true
#       Appends a single per-node scalar: the normalised [0,1] arc-length of
#       each needle node measured along the needle axis from the clamp end
#       (0 = clamp, ~1 = bevel tip), zero on tissue nodes.
#
# Motivation
# ----------
# MeshGraphNet sees only relative edge geometry, so interior needle nodes are
# locally indistinguishable — the network cannot tell how far along the shaft
# a node sits, which is exactly the information a clamped-beam deflection
# profile depends on.  The arc-length-to-clamp coordinate is a single absolute
# anchor that breaks this degeneracy without adding any per-frame state.
#
# The coordinate is rest-state geometry (frame-0), computed once by
# compute_needle_geometry.py and stored in needle_geometry_features.pt as
# "arclen_to_clamp".  The clamp end is taken as the axial extreme opposite the
# detected bevel tip.  The geometry file is regenerated below so the key is
# present (the regeneration is additive and idempotent — it leaves the bevel /
# surface-normal keys used by the other variants unchanged).
#
# All other settings match cropped_base_mgn_wu.

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
EXP=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_base_mgn_wu_anchor

mkdir -p ${EXP}/checkpoints ${EXP}/stats ${EXP}/outputs

# Regenerate needle geometry features so arclen_to_clamp is present.
apptainer exec --nv \
    --bind ${DATA}:/data/RUN-2 \
    ${SIF} \
    uv run python /opt/needle_mgn/examples/cfd/needle_tissue_cropped/compute_needle_geometry.py \
        --data_dir /data/RUN-2

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
        use_cpress=false \
        timestep_stride=10 \
        model_type=mgn \
        use_fourier_features=false \
        hidden_dim_node_encoder=256 \
        hidden_dim_edge_encoder=256 \
        hidden_dim_node_decoder=256 \
        hidden_dim_processor=256 \
        processor_size=15 \
        ++mgn_paper_features=true \
        ++mgn_include_mat_fiber=true \
        ++mgn_include_prev_v=true \
        ++mgn_include_evf=true \
        ++mgn_include_arclen_clamp=true \
        ++mgn_kinematic_needle_only=true \
        ++needle_fiber_axis=true \
        input_dim_edges=11 \
        'drop_targets=[a]' \
        save_every=10 \
        cuda_devices=null
