#!/bin/bash
#SBATCH --job-name=ndl_crp_fbr_invw_rigid
#SBATCH --account=fuge-prj-eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:1
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_invw_rigid/slurm-%j.out
#SBATCH --error=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_invw_rigid/slurm-%j.err

# Variant of cropped_fiber_iso_invw with an additional rigid-body error loss
# term.  At each step, the per-needle-node Δu prediction error e_i =
# pred_u_i - gt_u_i is decomposed into:
#
#   Δt = mean(e)                                  (translational mode)
#   Δω = Σ(r × (e − Δt)) / Σ|r|²                  (best-fit small rotation)
#   rigid_i = Δt + Δω × r_i                       (rigid component of error)
#
# where r_i = pos_i − centroid(pos_needle) is the per-needle-node coord
# relative to the current needle centroid.  We then add
#
#   L_total = L_MSE + rigid_loss_weight · mean(rigid_i²)
#
# Penalises whole-body translation / rotation errors preferentially over
# bending — encourages the model to nail rigid-body motion first.
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
EXP=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_invw_rigid

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
        ++rigid_loss_weight=1.0 \
        save_every=10 \
        cuda_devices=null
