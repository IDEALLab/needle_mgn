#!/bin/bash
#SBATCH --job-name=ndl_crp_fbr_invw_labmlp_pregion
#SBATCH --account=fuge-prj-eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:1
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_invw_labmlp_pregion/slurm-%j.out
#SBATCH --error=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_invw_labmlp_pregion/slurm-%j.err

# cropped_fiber_iso_invw_labmlp (non-equivariant lab-frame kinematic head) but
# with PER-REGION normalization instead of isotropic vector normalization:
#
#   per_region_norm=true
#   vector_iso_norm=false        (the two are mutually exclusive)
#       Targets/inputs are normalized with SEPARATE needle and tissue stats,
#       per component.  This stops the needle deflection from being drowned out
#       by the much larger tissue motion in the loss: under the isotropic
#       (single-scalar) u-normalization, the iso-std (~0.46 mm) is dominated by
#       the tissue's axial flow, so the needle's transverse deflection — the
#       quantity of interest — gets only ~1% of the kinematic-loss budget.
#       Per-region (+ per-component) normalization puts needle and tissue, and
#       axial vs transverse, on a common scale, up-weighting the deflection.
#
# Why this is allowed here (and blocked for the standard equivariant invw):
#   Per-region stats are per-component (anisotropic), which breaks the
#   1o-vector property the EQUIVARIANT displacement decoder needs — hence
#   vector_iso_norm is mutually exclusive with per_region_norm.  This variant's
#   displacement head is a plain lab-frame MLP (displacement_lab_mlp=true), so
#   it does not need the 1o-vector property; anisotropic per-region target
#   normalization is fine for it.  The constitutive head (stress S, contact
#   pressure, EVF) decodes scalars, for which per-region/per-component
#   normalization is already the intended scheme (needle stiff vs tissue soft).
#
# NOTE: inference (infer.py) auto-detects and handles per-region normalization.
# The bias-analysis tools that go through compare_models (fiber_bias_regression,
# rigid_mode_error_process) currently denormalize with the GLOBAL stats and
# would mis-scale per-region predictions — use infer.py / per-region-aware
# tooling to evaluate this variant.
#
# All other settings match cropped_fiber_iso_invw_labmlp.

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
EXP=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_invw_labmlp_pregion

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
        ++per_region_norm=true \
        ++vector_iso_norm=false \
        ++needle_fiber_axis=true \
        ++fiber_extra_invariants=true \
        ++fiber_extra_decoder_basis=true \
        ++displacement_lab_mlp=true \
        save_every=10 \
        cuda_devices=null
