#!/bin/bash
#SBATCH --job-name=ndl_crp_fbr_invw
#SBATCH --account=fuge-prj-eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:1
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_invw/slurm-%j.out
#SBATCH --error=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_invw/slurm-%j.err

# Variant A+B: cropped_fiber_iso with extra edge invariants (Variant A)
# AND a widened equivariant decoder basis (Variant B).
#
# A. fiber_extra_invariants=true
#    Adds (cos_θ_dst, bond_corr, dv_along_edge, dv_norm) to the edge encoder,
#    so the per-edge α weights can respond to velocity asymmetries instead
#    of being conditioned solely on fiber alignment.
#
# B. fiber_extra_decoder_basis=true
#    A second equivariant aggregate W = Σⱼ βⱼ · (d_i × ê_ij) is computed
#    each processor layer.  Because every per-edge contribution is by
#    construction perpendicular to d_i, W ⊥ d_i, and the decoder basis
#    grows from {V, d, V × d} (collapses to 1-D when V ∥ d, common for
#    needle nodes) to {V, d, V × d, W, W × d} which spans 3-D reliably.
#    This is the structural fix for the x-z flattening: even when the
#    needle's neighbours are entirely axial, the model can still produce
#    transverse motion via W and W × d.
#
# Adds 1 beta_head per processor layer, +2 invariants in the node block,
# +2 in the decoder input, and the vec_coef_head output grows from
# n_vec_outputs * 3 to n_vec_outputs * 5.  Modest parameter increase; full
# checkpoint compatibility with cropped_fiber_iso is broken (this is a new
# architecture).
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
EXP=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_invw

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
        save_every=10 \
        cuda_devices=null
