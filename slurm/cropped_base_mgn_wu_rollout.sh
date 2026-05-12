#!/bin/bash
#SBATCH --job-name=ndl_crp_base_mgn_wu_ro
#SBATCH --account=fuge-prj-eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_base_mgn_wu_rollout/slurm-%j.out
#SBATCH --error=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_base_mgn_wu_rollout/slurm-%j.err

# Multi-step rollout variant of cropped_base_mgn_wu — identical architecture
# (standard MGN with the Pfaff et al. 2020 input scheme: node-type one-hot +
# evf + unit fiber + prev_v on nodes; rel_pos + edge_len + mesh_rel + mesh_d
# + edge-type on edges), trained with the pushforward-trick K-step curriculum
# from cropped_fiber_iso_invw_rollout.
#
# Curriculum: L = (1 - w) * L_1step + w * mean_{k=0..K-1} L_k
#   - First multistep_warmup_epochs (5): w = 0  (pure 1-step training).
#   - Remaining epochs: w ramps linearly from 0 to multistep_w_max (0.7).
#
# Pushforward trick: per-step backward; state detached between steps so
# peak memory ≈ 1× base, time ≈ K× per training iter.
#
# Per rollout step:
#   - Predicted Δu is denormalised, raw needle position is updated.
#   - World (contact) edges are rebuilt from the new positions via cKDTree
#     at world_edge_radius (matches the dataset's construction).  Hex mesh
#     edges keep their rest-state mesh_rel / mesh_d block; newly-added
#     world edges get zeros there (per dataset convention).
#   - The prev_v and evf input slices in x are updated from the predicted
#     Δv / Δevf so the next step's forward sees a self-consistent state.
#     The node-type one-hot and unit fiber direction are static.
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
EXP=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_base_mgn_wu_rollout

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
        ++mgn_kinematic_needle_only=true \
        ++needle_fiber_axis=true \
        input_dim_edges=11 \
        'drop_targets=[a]' \
        ++multistep_K=5 \
        ++multistep_w_max=0.7 \
        ++multistep_warmup_epochs=5 \
        save_every=10 \
        cuda_devices=null
