#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VILA_DIR="${VILA_DIR:-${REPO_ROOT}/VILA}"

if [[ ! -f "${VILA_DIR}/environment_setup.sh" ]]; then
    echo "VILA submodule is missing or incomplete: ${VILA_DIR}" >&2
    echo "Run: git submodule update --init --recursive" >&2
    exit 1
fi

# Patch the pinned, clean source tree before `pip install -e` creates
# vila.egg-info and other environment-specific build artifacts.
VILA_DIR="${VILA_DIR}" bash "${SCRIPT_DIR}/apply_vila_patch.sh"

cd -- "${VILA_DIR}"
exec bash ./environment_setup.sh "$@"
