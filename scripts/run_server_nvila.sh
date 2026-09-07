#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VILA_DIR="${VILA_DIR:-${REPO_ROOT}/VILA}"

DEFAULT_MODEL_NAME="3DWay-15B"
FINETUNED_ROOT="${FINETUNED_ROOT:-${REPO_ROOT}/checkpoints/finetuned/nvila}"
MODEL_PATH="${MODEL_PATH:-${1:-${FINETUNED_ROOT}/${MODEL_NAME:-${DEFAULT_MODEL_NAME}}}}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
CONV_MODE="${CONV_MODE:-vicuna_v1}"
WORKERS="${WORKERS:-1}"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Model checkpoint does not exist: ${MODEL_PATH}" >&2
    echo "Pass its path as the first argument or set MODEL_PATH." >&2
    exit 1
fi

bash "${SCRIPT_DIR}/apply_vila_patch.sh" --require-applied

cd -- "${VILA_DIR}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
exec python -W ignore server.py \
    --host "${HOST}" \
    --port "${PORT}" \
    --model-path "${MODEL_PATH}" \
    --conv-mode "${CONV_MODE}" \
    --workers "${WORKERS}"
