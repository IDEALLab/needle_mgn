#!/bin/bash
#SBATCH --job-name=ndl_crp_fbr_iso_nc
#SBATCH --account=fuge-prj-eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:1
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_nocrop/slurm-%j.out
#SBATCH --error=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_nocrop/slurm-%j.err

# Experiment: cropped_fiber_iso trained on the full mesh (no dynamic cropping).
#
# Hypothesis: per-node noise in the fiber decoder's V_i basis is partly caused
# by the dynamic proximity crop — each training sample exposes the same
# needle node to a different set of neighbors, so the learned per-edge α_ij
# weights converge to something that's only a "sample average" rather than a
# function of a fixed graph.  At inference (which always uses the full mesh)
# the model then sees a topology it never trained on and produces per-node
# V_i variations that compound into rollout drift.
#
# Setting needle_crop_mm = tissue_crop_mm = 10000 mm and the proximity-only
# crop strategy disables the dynamic crop — every training sample uses the
# full mesh, matching inference exactly.
#
# Trade-off: ~10x larger graphs per step, so wallclock per epoch grows
# accordingly.  Time budget bumped from 18h to 36h.
#
# All other settings match cropped_fiber_iso.

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
EXP=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_nocrop

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
        needle_crop_mm=10000 \
        tissue_crop_mm=10000 \
        'crop_strategy_weights=[1,0,0]' \
        ++per_region_norm=false \
        ++vector_iso_norm=true \
        ++needle_fiber_axis=true \
        save_every=10 \
        cuda_devices=null
