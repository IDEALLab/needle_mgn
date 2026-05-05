#!/bin/bash
#SBATCH --job-name=ndl_crp_fbr_mgn
#SBATCH --account=fuge-prj-eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:1
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_mgn/slurm-%j.out
#SBATCH --error=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_mgn/slurm-%j.err

# Experiment: cropped_fiber_iso with MGN-paper inputs (Pfaff 2020).
#
# Combines:
#   - Equivariance fixes: vector_iso_norm + needle_fiber_axis (so the fiber
#     decoder has a transverse basis V × axis and target normalisation
#     respects 1o-vector structure).
#   - MGN-paper feature scheme: 2-dim node-type one-hot in x, edge_attr
#     augmented with mesh-space rel_pos.  All u/v/a/evf/s/cpress and
#     material props removed from inputs; predict only velocity v and
#     stress s.
#
# The hypothesis: the equivariant fiber model benefits especially from
# this scheme because (a) it removes the per-component-normalised vector
# inputs that already broke equivariance through u, v, a in x, and (b)
# the mesh+world rel_pos in edges encodes deformation in a basis the
# equivariant edge encoder can use directly.

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
EXP=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_mgn

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
        n_vec_outputs=1 \
        hidden_dim_node_encoder=256 \
        hidden_dim_edge_encoder=256 \
        hidden_dim_node_decoder=256 \
        hidden_dim_processor=256 \
        processor_size=15 \
        ++per_region_norm=false \
        ++vector_iso_norm=true \
        ++needle_fiber_axis=true \
        ++mgn_paper_features=true \
        ++mgn_include_prev_v=true \
        ++mgn_include_evf=true \
        ++mgn_kinematic_needle_only=true \
        input_dim_edges=11 \
        'drop_targets=[u,a]' \
        save_every=10 \
        cuda_devices=null
