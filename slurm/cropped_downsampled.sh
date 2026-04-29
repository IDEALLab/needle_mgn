#!/bin/bash
#SBATCH --job-name=ndl_crp_dws
#SBATCH --account=fuge-prj-eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:1
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_downsampled/slurm-%j.out
#SBATCH --error=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_downsampled/slurm-%j.err

# Experiment: Standard MGN on a reduced mesh (needle beam + downsampled tissue).
#
# Needle: 7936 HEX nodes replaced by a 1-D beam chain at 2 mm spacing
#   (~100 nodes at PCA-projected intervals along the needle axis).
#   Beam node features are scatter-means of the original HEX cluster.
#   World (contact) edges are remapped from original needle nodes to their beam
#   cluster representative, preserving needle-tissue interaction signals.
#
# Tissue: 39025 nodes subsampled to one per 3 mm voxel (voxel-grid approach,
#   keeping the actual node closest to each voxel centre).  The tissue mesh is
#   finer near the needle, so contact-zone coverage is preserved at 3 mm.
#   World edges to removed tissue nodes are dropped.
#
# Reduction caches (reduced_cache_RUN-*.pt) are built on first run and reused.
#
# Compare against: cropped_nocrop (full mesh, same MGN).

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

SIF=/home/nhoffma1/scratch.fuge-prj/needle_mgn/needle_mgn3.sif
DATA=/scratch/zt1/project/fuge-prj/user/nhoffma1/needle_mgn/RUN-2
EXP=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_downsampled

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
        model_type=mgn \
        beam_spacing_mm=2.0 \
        tissue_downsample_mm=3.0 \
        hidden_dim_node_encoder=256 \
        hidden_dim_edge_encoder=256 \
        hidden_dim_node_decoder=256 \
        hidden_dim_processor=256 \
        processor_size=15 \
        save_every=10 \
        cuda_devices=null
