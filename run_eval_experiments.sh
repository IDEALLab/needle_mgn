#!/bin/bash
# Run inference + evaluation for all trained experiments.
#
# Usage:
#   bash run_eval_experiments.sh /path/to/RUN-2
#
# Each experiment gets:
#   experiments/<name>/inference_output/RUN-<id>/  — predicted VTUs per test run
#   experiments/<name>/eval/                       — summary CSVs and plots
#
# NOTE: cropped_bistride and cropped_downsampled may require infer.py to be
# updated to support BiStrideMeshGraphNet and beam/tissue mesh reduction
# respectively if those paths have not yet been wired up.

set -euo pipefail

DATA_DIR=${1:-""}
if [ -z "$DATA_DIR" ] || [ ! -d "$DATA_DIR" ]; then
    echo "Usage: $0 /path/to/RUN-2"
    exit 1
fi
DATA_DIR=$(realpath "$DATA_DIR")

# Confirm the directory actually contains VTU files (catches the common mistake
# of passing the experiments/ directory instead of the data directory).
# if ! ls "${DATA_DIR}"/*.vtu >/dev/null 2>&1; then
#     echo "ERROR: No .vtu files found in ${DATA_DIR}"
#     echo "Pass the RUN-2 simulation data directory, not the experiments directory."
#     exit 1
# fi

PROJECT=$(realpath "$(dirname "$0")")
EXPERIMENTS=${PROJECT}/experiments

echo "Project:     ${PROJECT}"
echo "Data dir:    ${DATA_DIR}"
echo "Experiments: ${EXPERIMENTS}"
echo ""

# ---------------------------------------------------------------------------
# run_experiment <model> <exp_name> <extra_hydra_overrides>
#
#   model  — "cropped" or "domino"
#   extra  — space-separated Hydra key=value pairs passed to infer.py
#             (data_dir, ckpt_path, stats_dir are added automatically)
# ---------------------------------------------------------------------------
run_experiment() {
    local model="$1"
    local exp="$2"
    local extra="$3"
    # n_rollout_override: how many steps to roll out.  Caller passes the 4th
    # argument to override the default.  With timestep_stride=10 (default),
    # 16 steps covers VTU timesteps 0–160; stride=1 experiments use 160.
    local n_rollout="${4:-16}"

    local script_dir="${PROJECT}/examples/cfd/needle_tissue_${model}"
    local exp_dir="${EXPERIMENTS}/${exp}"
    local infer_out="${exp_dir}/inference_output"
    local eval_out="${exp_dir}/eval"

    echo "======================================================================"
    echo "  ${exp}"
    echo "======================================================================"

    if [ ! -d "${exp_dir}/checkpoints" ] || [ -z "$(ls -A "${exp_dir}/checkpoints" 2>/dev/null)" ]; then
        echo "  [SKIP] No checkpoints found at ${exp_dir}/checkpoints"
        echo ""
        return
    fi

    if [ -f "${eval_out}/summary.csv" ]; then
        echo "  [SKIP] Eval already complete (${eval_out}/summary.csv exists)"
        echo ""
        return
    fi

    mkdir -p "${infer_out}" "${eval_out}"

    # --- Inference: run infer.py on every test run ---------------------------
    echo "  [inference] ${exp}  (n_rollout=${n_rollout})"
    uv run python "${script_dir}/run_test_inference.py" \
        --data_dir "${DATA_DIR}" \
        --infer_output_base "${infer_out}" \
        --skip_existing \
        --extra "data_dir=${DATA_DIR} ckpt_path=${exp_dir}/checkpoints stats_dir=${exp_dir}/stats n_rollout=${n_rollout} ${extra}"

    # --- Eval: compare predictions against GT --------------------------------
    echo "  [eval] ${exp}"
    uv run python "${script_dir}/eval_test_runs.py" \
        --data_dir "${DATA_DIR}" \
        --infer_base_dir "${infer_out}" \
        --out_dir "${eval_out}"

    echo "  Done: ${exp_dir}/eval"
    echo ""
}

# ---------------------------------------------------------------------------
# Cropped MGN experiments
# ---------------------------------------------------------------------------
run_experiment cropped cropped_base \
    "use_cpress=false use_fourier_features=false"

run_experiment cropped cropped_noise \
    "use_cpress=false use_fourier_features=false"

run_experiment cropped cropped_fourier \
    "use_cpress=false use_fourier_features=true n_fourier_features=64 fourier_scale=0.36"

run_experiment cropped cropped_cpress \
    "use_cpress=true use_fourier_features=false"

run_experiment cropped cropped_stride1 \
    "use_cpress=false use_fourier_features=false per_region_norm=false timestep_stride=1" \
    160

run_experiment cropped cropped_splitnorm \
    "use_cpress=true use_fourier_features=false per_region_norm=true"

run_experiment cropped cropped_large \
    "use_cpress=false use_fourier_features=false per_region_norm=false \
     hidden_dim_node_encoder=384 hidden_dim_edge_encoder=384 \
     hidden_dim_node_decoder=384 hidden_dim_processor=384 processor_size=20"

run_experiment cropped cropped_nocrop \
    "use_cpress=false use_fourier_features=false per_region_norm=false \
     needle_crop_mm=10000 tissue_crop_mm=10000 'crop_strategy_weights=[1,0,0]'"

run_experiment cropped cropped_kan \
    "use_cpress=false per_region_norm=false model_type=kan num_harmonics=5 \
     hidden_dim_node_encoder=256 hidden_dim_edge_encoder=256 \
     hidden_dim_node_decoder=256 hidden_dim_processor=256 processor_size=15"

run_experiment cropped cropped_bistride \
    "use_cpress=false per_region_norm=false \
     needle_crop_mm=10000 tissue_crop_mm=10000 'crop_strategy_weights=[1,0,0]' \
     model_type=bistride use_bsms=true num_bsms_levels=2 \
     num_layers_bistride=2 bistride_unet_levels=1 \
     hidden_dim_node_encoder=256 hidden_dim_edge_encoder=256 \
     hidden_dim_node_decoder=256 hidden_dim_processor=256 processor_size=15"

# NOTE: cropped_downsampled requires infer.py to apply beam reduction and
# tissue downsampling (beam_spacing_mm / tissue_downsample_mm) before
# inference will succeed.
run_experiment cropped cropped_downsampled \
    "use_cpress=false per_region_norm=false \
     needle_crop_mm=10000 tissue_crop_mm=10000 'crop_strategy_weights=[1,0,0]' \
     beam_spacing_mm=2.0 tissue_downsample_mm=3.0 \
     hidden_dim_node_encoder=256 hidden_dim_edge_encoder=256 \
     hidden_dim_node_decoder=256 hidden_dim_processor=256 processor_size=15"

run_experiment cropped cropped_fiber \
    "use_cpress=false per_region_norm=false \
     model_type=fiber n_vec_outputs=3 \
     hidden_dim_node_encoder=256 hidden_dim_edge_encoder=256 \
     hidden_dim_node_decoder=256 hidden_dim_processor=256 processor_size=15"

run_experiment cropped cropped_fiber_kan \
    "use_cpress=false per_region_norm=false \
     model_type=fiber_kan n_vec_outputs=3 num_harmonics=5 \
     hidden_dim_node_encoder=256 hidden_dim_edge_encoder=256 \
     hidden_dim_node_decoder=256 hidden_dim_processor=256 processor_size=15"

run_experiment cropped cropped_tfn \
    "use_cpress=false per_region_norm=false \
     model_type=tfn n_vec_outputs=3 \
     'irreps_hidden=8x0e + 4x1o + 2x2e' l_max=1 \
     n_radial_basis=8 r_max=60.0 processor_size=15"

# ---------------------------------------------------------------------------
# DoMINO experiments
# ---------------------------------------------------------------------------
run_experiment domino domino_base \
    "use_cpress=false use_fourier_features_state=false"

run_experiment domino domino_noise \
    "use_cpress=false use_fourier_features_state=false"

run_experiment domino domino_fourier \
    "use_cpress=false use_fourier_features_state=true n_fourier_features_state=64 fourier_scale_state=0.36"

run_experiment domino domino_cpress \
    "use_cpress=true use_fourier_features_state=false"

echo "======================================================================"
echo "All experiments complete."
echo "Results are in ${EXPERIMENTS}/<name>/eval/"
echo "======================================================================"
