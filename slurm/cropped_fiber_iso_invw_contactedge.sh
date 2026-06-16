#!/bin/bash
#SBATCH --job-name=ndl_crp_fbr_invw_cedge
#SBATCH --account=fuge-prj-eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:1
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --output=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_invw_contactedge/slurm-%j.out
#SBATCH --error=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_invw_contactedge/slurm-%j.err

# Variant of cropped_fiber_iso_invw that adds a dedicated *contact-edge*
# equivariant decoder-basis vector (contact_decoder_basis=true).
#
# Motivation
# ----------
# The structural mesh edges on the needle are almost entirely axial, so the
# existing equivariant aggregates degenerate on needle nodes:
#   * V ∥ d  ⇒  V × d ≈ 0
#   * mat_fiber = 0 on needle nodes  ⇒  d = 0  ⇒  W = Σ β (d × ê) = 0 and the
#     decoder basis {V, d, V×d} collapses to 1-D {V}.
# The needle–tissue contact (world) edges already feed V/W, but they share the
# same alpha/beta heads as mesh edges and W still vanishes when d = 0.
#
# Feature
# -------
# contact_decoder_basis=true adds a third equivariant aggregate built *only*
# from the world (contact) edges, with its own per-layer gamma head:
#       C_i = Σ_{j ∈ contact} γ_ij · ê_ij
# Contact edges point radially from the needle surface into the surrounding
# tissue — transverse to the axis by construction — so C is a transverse
# direction that does NOT depend on d_i and therefore survives the d = 0
# collapse on needle nodes.  C is appended to the decoder basis, which here
# (with fiber_extra_decoder_basis=true) becomes {V, d, V×d, W, W×d, C}.
#
# Plumbing
# --------
#   * Model: FiberEquivariantMGN(contact_decoder_basis=True) adds one
#     gamma_head per processor layer, 2 invariants (||C||, C·d) in the node
#     block and decoder, and grows vec_coef_head output by n_vec_outputs.
#   * Dataset: attaches a per-edge boolean graph.world_edge_mask
#     (all_et[:, 2] > 0.5).  No extra precompute / geometry file needed.
#   * NOT checkpoint-compatible with cropped_fiber_iso_invw (new heads + wider
#     decoder).
#
# All other settings match cropped_fiber_iso_invw.

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
EXP=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments/cropped_fiber_iso_invw_contactedge

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
        ++contact_decoder_basis=true \
        save_every=10 \
        cuda_devices=null
