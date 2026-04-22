#!/bin/bash
# Run inference + evaluation for all 8 trained experiments.
#
# Usage:
#   bash run_eval_experiments.sh /path/to/RUN-2
#
# Each experiment gets:
#   experiments/<name>/inference_output/RUN-<id>/  — predicted VTUs per test run
#   experiments/<name>/eval/                       — summary CSVs and plots

set -euo pipefail

DATA_DIR=${1:-""}
if [ -z "$DATA_DIR" ] || [ ! -d "$DATA_DIR" ]; then
    echo "Usage: $0 /path/to/RUN-2"
    exit 1
fi
DATA_DIR=$(realpath "$DATA_DIR")

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
    echo "  [inference] ${exp}"
    uv run python "${script_dir}/run_test_inference.py" \
        --data_dir "${DATA_DIR}" \
        --infer_output_base "${infer_out}" \
        --skip_existing \
        --extra "data_dir=${DATA_DIR} ckpt_path=${exp_dir}/checkpoints stats_dir=${exp_dir}/stats ${extra}"

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
