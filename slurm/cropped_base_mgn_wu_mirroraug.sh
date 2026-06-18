#!/bin/bash
#SBATCH --job-name=ndl_crp_base_mgn_mirroraug
#SBATCH --account=fuge-prj-eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:1
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_base_mgn_wu_mirroraug/slurm-%j.out
#SBATCH --error=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_base_mgn_wu_mirroraug/slurm-%j.err

# Variant of cropped_base_mgn_wu (MGN-paper feature scheme, predicts u/v/evf/s,
# drop_targets=[a]) trained with the DE-BIASING mirror augmentation:
#
#   needle_axis_mirror_aug=true
#   mirror_plane_normal_deg=90          (reflect y -> -y, the X-Z plane)
#   aug_transform_stress=true           (co-transform the stress tensor)
#       With 50% probability each training sample is REFLECTED across the bevel
#       symmetry plane (containing the frame-0 needle axis).  All 1o vector
#       quantities (coord, u, v, a, mat_fiber) reflect; the Voigt stress 's'
#       co-transforms as a rank-2 tensor M S Mᵀ; invariant scalars (evf, node
#       type, material props) are unchanged.  Train split only.
#
# Why
# ---
# The dataset only simulated fiber orientations on one azimuthal half (every
# run's transverse fiber lies on the y=x diagonal; the y=-x half is empty), so
# the model never learns the missing half and under-predicts the fiber-
# dependent deflection.  A beveled needle has a REFLECTION symmetry (not a
# rotational one), so reflecting each sample across the bevel symmetry plane
# synthesises the missing-half fiber orientations as PHYSICALLY VALID data, for
# free.  Unlike the equivariant fiber model, the base MGN decodes every output
# with a plain MLP (no built-in invariance), so aug_transform_stress=true is
# set here so the stress target stays consistent with the reflected geometry.
#
# Mirror plane: mirror_plane_normal_deg=90 is the VERIFIED bevel symmetry plane
# for this dataset (reflect y -> -y across the X-Z plane; the needle maps onto
# itself there, mean self-overlap 0.054 mm vs 0.120 mm for x -> -x which flips
# the bevel), confirmed with mirror_vtu.py.  If the dataset's bevel orientation
# changes, re-confirm with mirror_vtu.py (the bevel must map to itself).
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
EXP=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_base_mgn_wu_mirroraug

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
        ++needle_axis_mirror_aug=true \
        ++mirror_plane_normal_deg=90.0 \
        ++aug_transform_stress=true \
        input_dim_edges=11 \
        'drop_targets=[a]' \
        save_every=10 \
        cuda_devices=null
