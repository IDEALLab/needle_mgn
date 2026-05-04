#!/bin/bash
#SBATCH --job-name=ndl_crp_base_mgn
#SBATCH --account=fuge-prj-eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:1
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_base_mgn/slurm-%j.out
#SBATCH --error=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_base_mgn/slurm-%j.err

# Experiment: cropped_base trained with the MGN-paper feature scheme
# (Pfaff et al. 2020, "Hyperelastic plate" example) + mat_fiber as a node
# input.
#
# Inputs:
#   - Per node: [node_type one-hot(2), unit fiber direction(3)]   = 5 dims
#   - Per edge: [world_rel(3), world_d(1), mesh_rel(3), mesh_d(1),
#                edge_type_onehot(3)]   = 11 dims
#     World/contact edges have the mesh_rel/mesh_d block zeroed
#     (those nodes were disjoint in the rest mesh).
# Outputs:
#   - Lagrangian (world-space) velocity v (3 dims)
#   - Cauchy stress voigt s (6 dims)
#   - Total output_dim = 9
#
# All state inputs (u, v, a, evf, s, cpress) and other material props are
# dropped from x; the model gets node_type, the rest-state fiber direction,
# and the mesh+world edge geometry — closer to the MGN-paper input scheme
# than the full feature setup but with the per-node anisotropy that the
# tissue's transverse-isotropic constitutive model encodes.  needle_fiber_axis
# is on so needle nodes get the principal-axis vector instead of a zero
# fiber input.

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
EXP=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_base_mgn

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
        ++needle_fiber_axis=true \
        input_dim_edges=11 \
        'drop_targets=[u,a,evf]' \
        save_every=10 \
        cuda_devices=null
