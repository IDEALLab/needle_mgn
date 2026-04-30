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
# Environment knobs:
#   DEBUG=1          Verbose tracing (set -x) plus extra environment dumps
#   DRY_RUN=1        Print the final uv run command but don't execute it
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
#   DEBUG=1 bash run_local.sh cropped_tfn_iso
#   DRY_RUN=1 bash run_local.sh cropped_tfn_iso

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Diagnostics: surface failures that would otherwise be silent.
# ---------------------------------------------------------------------------
on_err() {
    local exit_code=$?
    local line_no=${1:-?}
    echo "" >&2
    echo "[run_local.sh] FAILED at line ${line_no} (exit ${exit_code})" >&2
    echo "[run_local.sh] last command attempted: ${BASH_COMMAND}" >&2
    if [ "${DEBUG:-0}" != "1" ]; then
        echo "[run_local.sh] re-run with DEBUG=1 for full trace" >&2
    fi
    exit "$exit_code"
}
trap 'on_err $LINENO' ERR

if [ "${DEBUG:-0}" = "1" ]; then
    set -x
fi

# Bash 4+ required for `mapfile`.  macOS ships bash 3.2 by default.
if (( BASH_VERSINFO[0] < 4 )); then
    echo "ERROR: this script requires bash >= 4 (found ${BASH_VERSION})." >&2
    echo "  On macOS, install via: brew install bash, then run with: $(brew --prefix)/bin/bash $0 ..." >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------
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
    echo "ERROR: slurm file not found: $SLURM_FILE" >&2
    echo "Run '$0' with no arguments to see available experiments." >&2
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: data_dir not found: $DATA_DIR" >&2
    echo "  (resolved from arg 2 or default '${PROJECT}/RUN-2')" >&2
    exit 1
fi
DATA_DIR=$(realpath "$DATA_DIR")

EXP_DIR="${PROJECT}/experiments/${EXP_NAME}"
mkdir -p "${EXP_DIR}/checkpoints" "${EXP_DIR}/stats" "${EXP_DIR}/outputs"

# ---------------------------------------------------------------------------
# Pre-flight: required tools.  Fail loudly if missing instead of silently
# letting later steps print no output.
#
# Only `uv` is required on PATH — the slurm-file parser and the training
# script both run via `uv run python`, which uses uv's managed Python and
# doesn't depend on a system Python install (important on Windows).
# ---------------------------------------------------------------------------
UV_BIN=$(command -v uv || true)
if [ -z "$UV_BIN" ]; then
    echo "ERROR: uv not found on PATH (needed for both parsing and training)." >&2
    echo "  PATH=$PATH" >&2
    echo "  Install: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 127
fi

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

# Capture stderr separately so a parser failure surfaces — the previous
# version routed parser stderr to the terminal but never confirmed parser
# success, so an empty parse looked the same as a successful parse.
PARSE_STDERR=$(mktemp)
trap 'rm -f "$PARSE_STDERR"' EXIT

set +e
mapfile -t PARSED < <(
    cd "$PROJECT" && "$UV_BIN" run --project "$PROJECT" python -c "$PARSE_PY" "$SLURM_FILE" \
        2> "$PARSE_STDERR"
)
PARSE_RC=$?
set -e

if [ "$PARSE_RC" -ne 0 ] || [ "${#PARSED[@]}" -lt 1 ]; then
    echo "ERROR: failed to parse $SLURM_FILE (python rc=$PARSE_RC, tokens=${#PARSED[@]})" >&2
    if [ -s "$PARSE_STDERR" ]; then
        echo "  parser stderr:" >&2
        sed 's/^/    /' "$PARSE_STDERR" >&2
    else
        echo "  (parser produced no stderr — check the train.py line in $SLURM_FILE)" >&2
    fi
    exit 1
fi

SCRIPT_REL="${PARSED[0]}"
SCRIPT="${PROJECT}/${SCRIPT_REL}"
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: training script not found: $SCRIPT" >&2
    echo "  (parsed from slurm file: ${SCRIPT_REL})" >&2
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

if [ "${#SLURM_OVERRIDES[@]}" -eq 0 ]; then
    echo "ERROR: parsed slurm file but found 0 Hydra overrides — likely a parser issue." >&2
    echo "  raw parsed tokens (${#PARSED[@]}):" >&2
    printf '    %s\n' "${PARSED[@]}" >&2
    exit 1
fi

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

# ---------------------------------------------------------------------------
# Diagnostic banner — visible on every run so failures inside uv/python don't
# look like silent exits.
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "  Experiment   : ${EXP_NAME}"
echo "  Slurm file   : ${SLURM_FILE}"
echo "  Script       : ${SCRIPT}"
echo "  Project      : ${PROJECT}"
echo "  Data         : ${DATA_DIR}"
echo "  Results      : ${EXP_DIR}"
echo "  bash         : ${BASH_VERSION}"
echo "  uv           : ${UV_BIN} ($("$UV_BIN" --version 2>&1))"
echo "  uv python    : $("$UV_BIN" run --project "$PROJECT" python --version 2>&1)"
echo "  Overrides    : ${#SLURM_OVERRIDES[@]} from slurm file + ${#COMMON[@]} local"
echo "----------------------------------------------------------------------"
printf '  %s\n' "${SLURM_OVERRIDES[@]}" "${COMMON[@]}"
echo "======================================================================"

if [ "${DEBUG:-0}" = "1" ]; then
    echo "[debug] env (filtered):"
    env | grep -E '^(CUDA|RANK|LOCAL_RANK|WORLD_SIZE|MASTER|UV_|WARP_|WANDB_|MKL)' | sed 's/^/  /'
    echo "[debug] cwd: $(pwd)"
fi

cd "$(dirname "$SCRIPT")"
echo "[run_local.sh] cd $(pwd)"
echo "[run_local.sh] launching: uv run python $(basename "$SCRIPT") <overrides>"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[run_local.sh] DRY_RUN=1 — not executing.  Full command:"
    printf '%q ' "$UV_BIN" run python "$(basename "$SCRIPT")" "${SLURM_OVERRIDES[@]}" "${COMMON[@]}"
    echo
    exit 0
fi

# Disable -e here so we can capture the rc and print a clear final status —
# otherwise a non-zero exit from uv/python is reported only as a generic ERR.
set +e
"$UV_BIN" run python "$(basename "$SCRIPT")" "${SLURM_OVERRIDES[@]}" "${COMMON[@]}"
RC=$?
set -e

echo "======================================================================"
if [ "$RC" -eq 0 ]; then
    echo "[run_local.sh] training exited cleanly (rc=0)"
else
    echo "[run_local.sh] training FAILED (rc=$RC)" >&2
fi
echo "  Results in: ${EXP_DIR}"
echo "======================================================================"
exit "$RC"
