#!/bin/bash
#SBATCH --job-name=ndl_crp_tfn_mgn
#SBATCH --account=fuge-prj-eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:1
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_tfn_iso_mgn/slurm-%j.out
#SBATCH --error=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_tfn_iso_mgn/slurm-%j.err

# Experiment: cropped_tfn_iso with MGN-paper inputs (Pfaff 2020).
#
# TFN architecturally benefits the most from this scheme:
#   - Inputs are now strictly scalar (n_node_scalar=2, n_node_vec=0): all
#     directional information enters through the spherical-harmonic edge
#     features, which is exactly the regime e3nn was designed for.  The
#     previous u/v/a/mat_fiber 1o inputs were redundant with the edge SH
#     under TFN's tensor-product mixing.
#   - Edge features include both world and mesh rel_pos: l_max=1 SH on
#     world_rel gives orientation, while mesh_rel + the two distances
#     enter the radial MLP as scalar context — exactly the MGN paper's
#     formulation.
#   - n_edge_extra_scalar grows from 4 → 8 (everything after the first 3
#     world_rel columns) and is set automatically from input_dim_edges-3.

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
EXP=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_tfn_iso_mgn

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
        model_type=tfn \
        n_vec_outputs=1 \
        'irreps_hidden=8x0e + 4x1o + 2x2e' \
        l_max=1 \
        n_radial_basis=8 \
        r_max=60.0 \
        processor_size=15 \
        tfn_checkpoint_layers=false \
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
