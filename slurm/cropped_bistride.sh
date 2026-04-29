#!/bin/bash
#SBATCH --job-name=ndl_crp_bst
#SBATCH --account=fuge-prj-eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:1
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_bistride/slurm-%j.out
#SBATCH --error=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_bistride/slurm-%j.err

# Experiment: BiStride MeshGraphNet on the full (uncropped) mesh.
#
# BiStrideMGN augments vanilla MGN with a U-Net multi-scale message passing
# that alternates coarsening and refinement passes, improving long-range
# interaction modelling.  The bi-stride coarsening is precomputed once for the
# fixed mesh topology (bsms_cache_l2.pt) and reused across all training steps.
#
# Dynamic cropping is disabled (crop radii = 10000 mm) so the topology is fixed
# every step and the precomputed BSMS hierarchy remains valid.
#
# First run will build both the raw VTU caches and the BSMS cache — expect a
# longer startup time.  Subsequent runs load from disk.
#
# Compare against: cropped_nocrop (same mesh, vanilla MGN).

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

SIF=/home/nhoffma1/scratch.fuge-prj/needle_mgn/needle_mgn2.sif
DATA=/scratch/zt1/project/fuge-prj/user/nhoffma1/needle_mgn/RUN-2
EXP=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_bistride

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
        per_region_norm=false \
        timestep_stride=10 \
        needle_crop_mm=10000 \
        tissue_crop_mm=10000 \
        'crop_strategy_weights=[1,0,0]' \
        model_type=bistride \
        use_bsms=true \
        num_bsms_levels=2 \
        num_layers_bistride=2 \
        bistride_unet_levels=1 \
        hidden_dim_node_encoder=256 \
        hidden_dim_edge_encoder=256 \
        hidden_dim_node_decoder=256 \
        hidden_dim_processor=256 \
        processor_size=15 \
        num_processor_checkpoint_segments=10 \
        save_every=10 \
        cuda_devices=null
