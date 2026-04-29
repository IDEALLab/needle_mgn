#!/bin/bash
# Run a training experiment locally (no Apptainer, no SLURM).
#
# Usage:
#   bash run_local.sh <experiment_name> [data_dir]
#
# Arguments:
#   experiment_name  Name matching slurm/<name>.sh (e.g. cropped_base)
#   data_dir         Path to the RUN-2 directory (default: ./RUN-2)
#
# Hydra overrides are extracted from slurm/<experiment_name>.sh — the slurm
# file is the single source of truth for each experiment's configuration.
# This script substitutes local paths (data_dir, ckpt_path, stats_dir,
# hydra.run.dir) and runs the training script directly via uv.
#
# Results are written to experiments/<experiment_name>/{checkpoints,stats,outputs}.
#
# Examples:
#   bash run_local.sh cropped_base
#   bash run_local.sh cropped_fiber_iso /data/RUN-2

set -euo pipefail

PROJECT=$(realpath "$(dirname "$0")")
SLURM_DIR="${PROJECT}/slurm"

EXP_NAME=${1:-""}
DATA_DIR=${2:-"${PROJECT}/RUN-2"}

if [ -z "$EXP_NAME" ]; then
    echo "Usage: $0 <experiment_name> [data_dir]"
    echo ""
    echo "Available experiments (from slurm/):"
    for f in "${SLURM_DIR}"/*.sh; do
        name=$(basename "$f" .sh)
        [ "$name" = "setup_experiment_dirs" ] && continue
        echo "  $name"
    done
    exit 1
fi

SLURM_FILE="${SLURM_DIR}/${EXP_NAME}.sh"
if [ ! -f "$SLURM_FILE" ]; then
    echo "ERROR: slurm file not found: $SLURM_FILE"
    echo "Run '$0' with no arguments to see available experiments."
    exit 1
fi

DATA_DIR=$(realpath "$DATA_DIR")
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: data_dir not found: $DATA_DIR"
    exit 1
fi

EXP_DIR="${PROJECT}/experiments/${EXP_NAME}"
mkdir -p "${EXP_DIR}/checkpoints" "${EXP_DIR}/stats" "${EXP_DIR}/outputs"

# ---------------------------------------------------------------------------
# Parse slurm file: extract train.py invocation + Hydra overrides.
# Returns: SCRIPT_REL on stdout line 1, then one Hydra token per subsequent
# line (with shell quotes already stripped).
# ---------------------------------------------------------------------------
PARSE_PY=$(cat <<'EOF'
import os, re, shlex, sys

text = open(sys.argv[1]).read()
# Join shell line continuations so the train.py invocation is on one line.
joined = re.sub(r"\\\n", " ", text)

for raw_line in joined.split("\n"):
    line = raw_line.strip()
    if "train.py" not in line:
        continue
    # Strip a leading apptainer/uv prelude — shlex tokenizes the whole line.
    toks = shlex.split(line, comments=True)
    # Find the train.py argument; everything after it is Hydra overrides.
    for i, t in enumerate(toks):
        if t.endswith("train.py"):
            # Path of the script relative to PROJECT.
            m = re.search(r"examples/cfd/needle_tissue_(cropped|domino)/train\.py", t)
            if not m:
                sys.stderr.write(f"could not match train.py path in token: {t!r}\n")
                sys.exit(1)
            print(m.group(0))
            for tok in toks[i + 1 :]:
                print(tok)
            sys.exit(0)
    break

sys.stderr.write("no train.py invocation found in slurm file\n")
sys.exit(1)
EOF
)

mapfile -t PARSED < <(python3 -c "$PARSE_PY" "$SLURM_FILE")
if [ ${#PARSED[@]} -lt 1 ]; then
    echo "ERROR: failed to parse $SLURM_FILE"
    exit 1
fi

SCRIPT_REL="${PARSED[0]}"
SCRIPT="${PROJECT}/${SCRIPT_REL}"
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: training script not found: $SCRIPT"
    exit 1
fi

# Filter out path overrides — we override these to point at local directories.
SLURM_OVERRIDES=()
for tok in "${PARSED[@]:1}"; do
    [ -z "$tok" ] && continue
    case "$tok" in
        data_dir=*|ckpt_path=*|stats_dir=*|hydra.run.dir=*) ;;
        *) SLURM_OVERRIDES+=("$tok") ;;
    esac
done

# ---------------------------------------------------------------------------
# Local environment + path overrides
# ---------------------------------------------------------------------------
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

# Common overrides go LAST so they take precedence over slurm-side paths.
COMMON=(
    "data_dir=${DATA_DIR}"
    "ckpt_path=${EXP_DIR}/checkpoints"
    "stats_dir=${EXP_DIR}/stats"
    "hydra.run.dir=${EXP_DIR}/outputs"
    cuda_devices=null
)