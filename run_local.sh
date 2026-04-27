#!/bin/bash
# Run a training experiment locally (no Apptainer, no SLURM).
#
# Usage:
#   bash run_local.sh <experiment_name> [data_dir]
#
# Arguments:
#   experiment_name  One of the experiment names below (e.g. cropped_base)
#   data_dir         Path to the RUN-2 directory (default: ./RUN-2)
#
# Results are written to experiments/<experiment_name>/{checkpoints,stats,outputs}.
#
# Examples:
#   bash run_local.sh cropped_base
#   bash run_local.sh cropped_bistride /data/RUN-2

set -euo pipefail

EXP_NAME=${1:-""}
DATA_DIR=${2:-"$(dirname "$0")/RUN-2"}

if [ -z "$EXP_NAME" ]; then
    echo "Usage: $0 <experiment_name> [data_dir]"
    echo ""
    echo "Available experiments:"
    echo "  cropped_base       cropped_noise      cropped_fourier    cropped_cpress"
    echo "  cropped_stride1    cropped_splitnorm  cropped_large      cropped_nocrop"
    echo "  cropped_kan        cropped_bistride   cropped_downsampled  cropped_fiber  cropped_fiber_kan"
    echo "  domino_base        domino_noise       domino_fourier     domino_cpress"
    exit 1
fi

DATA_DIR=$(realpath "$DATA_DIR")
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: data_dir not found: $DATA_DIR"
    exit 1
fi

PROJECT=$(realpath "$(dirname "$0")")
EXP_DIR="${PROJECT}/experiments/${EXP_NAME}"
mkdir -p "${EXP_DIR}/checkpoints" "${EXP_DIR}/stats" "${EXP_DIR}/outputs"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=$(( 20000 + $$ % 10000 ))
export UV_CACHE_DIR=/tmp/uv-cache-$$
export WARP_CACHE_PATH=/tmp/warp-cache-$$
export WANDB_DATA_DIR=/tmp/wandb-$$

# Intel MKL runtime required by sparse_dot_mkl (used for BiStride BSMS).
# uv add mkl installs libmkl_rt.so into the venv but not onto LD_LIBRARY_PATH.
_MKL_SO="${PROJECT}/.venv/lib/libmkl_rt.so.2"
if [ -f "${_MKL_SO}" ]; then
    export MKL_RT="${_MKL_SO}"
fi

# ---------------------------------------------------------------------------
# Common overrides for all experiments
# ---------------------------------------------------------------------------
COMMON=(
    data_dir="${DATA_DIR}"
    ckpt_path="${EXP_DIR}/checkpoints"
    stats_dir="${EXP_DIR}/stats"
    "hydra.run.dir=${EXP_DIR}/outputs"
    wandb_mode=offline
    epochs=100
    batch_size=1
    save_every=10
    cuda_devices=null
)

# ---------------------------------------------------------------------------
# Per-experiment overrides
# ---------------------------------------------------------------------------
case "$EXP_NAME" in

cropped_base)
    SCRIPT="${PROJECT}/examples/cfd/needle_tissue_cropped/train.py"
    OVERRIDES=(
        wandb_project=PhysicsNeMo-Cropped-Ablation
        noise_std=0.0
        use_fourier_features=false
        use_cpress=false
    )
    ;;

cropped_noise)
    SCRIPT="${PROJECT}/examples/cfd/needle_tissue_cropped/train.py"
    OVERRIDES=(
        wandb_project=PhysicsNeMo-Cropped-Ablation
        noise_std=3e-3
        use_fourier_features=false
        use_cpress=false
    )
    ;;

cropped_fourier)
    SCRIPT="${PROJECT}/examples/cfd/needle_tissue_cropped/train.py"
    OVERRIDES=(
        wandb_project=PhysicsNeMo-Cropped-Ablation
        noise_std=0.0
        use_fourier_features=true
        n_fourier_features=64
        fourier_scale=0.36
        use_cpress=false
    )
    ;;

cropped_cpress)
    SCRIPT="${PROJECT}/examples/cfd/needle_tissue_cropped/train.py"
    OVERRIDES=(
        wandb_project=PhysicsNeMo-Cropped-Ablation
        noise_std=0.0
        use_fourier_features=false
        use_cpress=true
    )
    ;;

cropped_stride1)
    SCRIPT="${PROJECT}/examples/cfd/needle_tissue_cropped/train.py"
    OVERRIDES=(
        wandb_project=PhysicsNeMo-Cropped-Ablation
        noise_std=0.0
        use_fourier_features=false
        use_cpress=false
        per_region_norm=false
        timestep_stride=1
        max_frames_per_run=25
        n_rollout=200
    )
    ;;

cropped_splitnorm)
    SCRIPT="${PROJECT}/examples/cfd/needle_tissue_cropped/train.py"
    OVERRIDES=(
        wandb_project=PhysicsNeMo-Cropped-Ablation
        noise_std=0.0
        use_fourier_features=false
        use_cpress=true
        per_region_norm=true
        timestep_stride=10
    )
    ;;

cropped_large)
    SCRIPT="${PROJECT}/examples/cfd/needle_tissue_cropped/train.py"
    OVERRIDES=(
        wandb_project=PhysicsNeMo-Cropped-Ablation
        noise_std=0.0
        use_fourier_features=false
        use_cpress=false
        per_region_norm=false
        timestep_stride=10
        hidden_dim_node_encoder=384
        hidden_dim_edge_encoder=384
        hidden_dim_node_decoder=384
        hidden_dim_processor=384
        processor_size=20
    )
    ;;

cropped_nocrop)
    SCRIPT="${PROJECT}/examples/cfd/needle_tissue_cropped/train.py"
    OVERRIDES=(
        wandb_project=PhysicsNeMo-Cropped-Ablation
        noise_std=0.0
        use_fourier_features=false
        use_cpress=false
        per_region_norm=false
        timestep_stride=10
        needle_crop_mm=10000
        tissue_crop_mm=10000
        'crop_strategy_weights=[1,0,0]'
    )
    ;;

cropped_kan)
    SCRIPT="${PROJECT}/examples/cfd/needle_tissue_cropped/train.py"
    OVERRIDES=(
        wandb_project=PhysicsNeMo-Cropped-Ablation
        noise_std=0.0
        use_cpress=false
        timestep_stride=10
        model_type=kan
        num_harmonics=5
        hidden_dim_node_encoder=256
        hidden_dim_edge_encoder=256
        hidden_dim_node_decoder=256
        hidden_dim_processor=256
        processor_size=15
        +per_region_norm=false
    )
    ;;

cropped_bistride)
    SCRIPT="${PROJECT}/examples/cfd/needle_tissue_cropped/train.py"
    OVERRIDES=(
        wandb_project=PhysicsNeMo-Cropped-Ablation
        noise_std=0.0
        use_cpress=false
        per_region_norm=false
        timestep_stride=10
        needle_crop_mm=10000
        tissue_crop_mm=10000
        'crop_strategy_weights=[1,0,0]'
        model_type=bistride
        use_bsms=true
        num_bsms_levels=2
        num_layers_bistride=2
        bistride_unet_levels=1
        hidden_dim_node_encoder=256
        hidden_dim_edge_encoder=256
        hidden_dim_node_decoder=256
        hidden_dim_processor=256
        processor_size=15
        num_processor_checkpoint_segments=10
    )
    ;;

cropped_downsampled)
    SCRIPT="${PROJECT}/examples/cfd/needle_tissue_cropped/train.py"
    OVERRIDES=(
        wandb_project=PhysicsNeMo-Cropped-Ablation
        noise_std=0.0
        use_cpress=false
        per_region_norm=false
        timestep_stride=10
        needle_crop_mm=10000
        tissue_crop_mm=10000
        'crop_strategy_weights=[1,0,0]'
        model_type=mgn
        beam_spacing_mm=2.0
        tissue_downsample_mm=3.0
        hidden_dim_node_encoder=256
        hidden_dim_edge_encoder=256
        hidden_dim_node_decoder=256
        hidden_dim_processor=256
        processor_size=15
    )
    ;;

cropped_fiber)
    SCRIPT="${PROJECT}/examples/cfd/needle_tissue_cropped/train.py"
    OVERRIDES=(
        wandb_project=PhysicsNeMo-Cropped-Ablation
        noise_std=0
        use_cpress=false
        timestep_stride=10
        model_type=fiber
        n_vec_outputs=3
        hidden_dim_node_encoder=256
        hidden_dim_edge_encoder=256
        hidden_dim_node_decoder=256
        hidden_dim_processor=256
        processor_size=15
        ++per_region_norm=false
    )
    ;;

cropped_fiber_kan)
    SCRIPT="${PROJECT}/examples/cfd/needle_tissue_cropped/train.py"
    OVERRIDES=(
        wandb_project=PhysicsNeMo-Cropped-Ablation
        noise_std=0
        use_cpress=false
        timestep_stride=10
        model_type=fiber_kan
        n_vec_outputs=3
        num_harmonics=5
        hidden_dim_node_encoder=256
        hidden_dim_edge_encoder=256
        hidden_dim_node_decoder=256
        hidden_dim_processor=256
        processor_size=15
        ++per_region_norm=false
    )
    ;;

domino_base)
    SCRIPT="${PROJECT}/examples/cfd/needle_tissue_domino/train.py"
    OVERRIDES=(
        wandb_project=PhysicsNeMo-DoMINO-Ablation
        noise_std=0.0
        use_fourier_features_state=false
        use_cpress=false
    )
    ;;

domino_noise)
    SCRIPT="${PROJECT}/examples/cfd/needle_tissue_domino/train.py"
    OVERRIDES=(
        wandb_project=PhysicsNeMo-DoMINO-Ablation
        noise_std=3e-3
        use_fourier_features_state=false
        use_cpress=false
    )
    ;;

domino_fourier)
    SCRIPT="${PROJECT}/examples/cfd/needle_tissue_domino/train.py"
    OVERRIDES=(
        wandb_project=PhysicsNeMo-DoMINO-Ablation
        noise_std=0.0
        use_fourier_features_state=true
        n_fourier_features_state=64
        fourier_scale_state=0.36
        use_cpress=false
    )
    ;;

domino_cpress)
    SCRIPT="${PROJECT}/examples/cfd/needle_tissue_domino/train.py"
    OVERRIDES=(
        wandb_project=PhysicsNeMo-DoMINO-Ablation
        noise_std=0.0
        use_fourier_features_state=false
        use_cpress=true
    )
    ;;

*)
    echo "ERROR: Unknown experiment '${EXP_NAME}'"
    echo "Run '$0' with no arguments to see available experiments."
    exit 1
    ;;
esac

echo "======================================================================"
echo "  Experiment : ${EXP_NAME}"
echo "  Script     : ${SCRIPT}"
echo "  Data       : ${DATA_DIR}"
echo "  Results    : ${EXP_DIR}"
echo "======================================================================"

cd "$(dirname "$SCRIPT")"
uv run python "$(basename "$SCRIPT")" "${COMMON[@]}" "${OVERRIDES[@]}"
