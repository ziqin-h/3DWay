#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VILA_DIR="${VILA_DIR:-${REPO_ROOT}/VILA}"
PATCH_FILE="${SCRIPT_DIR}/patches/vila-1806a64.patch"
EXPECTED_COMMIT="1806a646c24233726fe61bfff30bc5ef91444a77"
MODE="apply"

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [--check|--require-applied]" >&2
    exit 2
fi

case "${1:-}" in
    "") ;;
    --check) MODE="check" ;;
    --require-applied) MODE="require-applied" ;;
    *)
        echo "Usage: $0 [--check|--require-applied]" >&2
        exit 2
        ;;
esac

if [[ ! -e "${VILA_DIR}/.git" ]]; then
    echo "VILA submodule is not initialized: ${VILA_DIR}" >&2
    echo "Run: git submodule update --init --recursive" >&2
    exit 1
fi

actual_commit="$(git -C "${VILA_DIR}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
    echo "Unsupported VILA revision: ${actual_commit}" >&2
    echo "Expected: ${EXPECTED_COMMIT}" >&2
    echo "Run: git submodule update --init --recursive" >&2
    exit 1
fi

if git -C "${VILA_DIR}" apply --unidiff-zero --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
    echo "VILA patch is already applied (${EXPECTED_COMMIT:0:12})."
    exit 0
fi

if [[ "${MODE}" == "require-applied" ]]; then
    echo "Required VILA patch is not applied." >&2
    echo "Apply it before installing VILA: bash ${SCRIPT_DIR}/apply_vila_patch.sh" >&2
    exit 1
fi

if [[ -n "$(git -C "${VILA_DIR}" status --porcelain --untracked-files=all)" ]]; then
    echo "Refusing to patch a VILA worktree containing unrelated changes." >&2
    git -C "${VILA_DIR}" status --short >&2
    exit 1
fi

git -C "${VILA_DIR}" apply --unidiff-zero --check "${PATCH_FILE}"
if [[ "${MODE}" == "check" ]]; then
    echo "VILA patch applies cleanly (${EXPECTED_COMMIT:0:12})."
else
    git -C "${VILA_DIR}" apply --unidiff-zero "${PATCH_FILE}"
    echo "Applied VILA patch (${EXPECTED_COMMIT:0:12})."
fi
