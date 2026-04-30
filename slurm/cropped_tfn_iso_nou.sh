#!/bin/bash
#SBATCH --job-name=ndl_crp_tfn_nou
#SBATCH --account=fuge-prj-eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:1
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_tfn_iso_nou/slurm-%j.out
#SBATCH --error=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_tfn_iso_nou/slurm-%j.err

# Experiment: cropped_tfn_iso with u dropped from output targets;
# Δu is reconstructed at inference time from the predicted Δv via the
# trapezoidal integration rule:
#       Δu = (v_t + 0.5 · Δv_pred) · rollout_dt
# This is the GNS / MGN-paper integration approach applied to the TFN
# model.  TFN's strict end-to-end equivariance benefits especially:
#   - Only one equivariant 1o output (Δv) instead of three (Δu, Δv, Δa),
#     reducing the load on each o3.Linear and tensor product.
#   - The kinematic u_{t+1} = u_t + ⟨v⟩ · dt is enforced exactly,
#     consistent with the model's symmetry assumptions.
#   - Per-rollout-step Δu integration error grows as t² (slow) rather
#     than as t (linear) for direct prediction, on the assumption that
#     bias is small relative to the integrated signal.
#
# rollout_dt is auto-estimated by dataset.py and saved into
# target_stats.json.
#
# All other settings match cropped_tfn_iso.

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
EXP=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_tfn_iso_nou

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
        n_vec_outputs=3 \
        'irreps_hidden=8x0e + 4x1o + 2x2e' \
        l_max=1 \
        n_radial_basis=8 \
        r_max=60.0 \
        processor_size=15 \
        tfn_checkpoint_layers=false \
        ++per_region_norm=false \
        ++vector_iso_norm=true \
        ++needle_fiber_axis=true \
        'drop_targets=[u]' \
        save_every=10 \
        cuda_devices=null
