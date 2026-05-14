#!/bin/bash
#SBATCH --job-name=ndl_crp_fbr_invw_global
#SBATCH --account=fuge-prj-eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:1
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_invw_global/slurm-%j.out
#SBATCH --error=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_invw_global/slurm-%j.err

# Variant of cropped_fiber_iso_invw with four global needle features broadcast
# to every needle node (zero on tissue), giving the model whole-needle context
# even when the crop hides part of the needle:
#
#   ch 0: centroid_rel = (full-needle centroid − local_node_pos) / coord_std
#         (per-node 1o vector — translation-invariant)
#   ch 1: axis_dir     = unit principal-axis vector of the full needle
#                        (broadcast same to every needle node)
#   ch 2: centroid_v   = mean needle-node velocity / v_std   (broadcast)
#   ch 3: ang_v        = ω = Σ(r × v_rel) / Σ|r|²            (broadcast,
#                        rad/s, unnormalised — the MLP encoder absorbs scale)
#
# All four are computed pre-crop in the dataset using the *full* needle node
# set (self._needle_idx_t indexed into ft[key][t_local]), so the values are
# accurate regardless of the per-sample spatial crop.  Stacked into
# graph.global_needle_vecs of shape (n_sub, 4, 3).
#
# FiberEquivariantMGN reads it via n_global_needle_vecs=4 and adds 3*4 = 12
# scalar edge invariants (cos_theta, cos_phi, |g|) per channel to the edge
# encoder input.  No change to the equivariant decoder basis — full SE(3)
# equivariance preserved.
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
EXP=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_invw_global

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
        ++global_needle_vecs=true \
        save_every=10 \
        cuda_devices=null
