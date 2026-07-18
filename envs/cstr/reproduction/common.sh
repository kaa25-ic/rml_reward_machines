#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"
RESULTS_DIR="${REPO_ROOT}/envs/cstr/results_and_evaluation"
PPO_DIR="${RESULTS_DIR}/ppo"
MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/rml_rm_matplotlib}"
PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/private/tmp/rml_pycache}"
export MPLCONFIGDIR
export PYTHONPYCACHEPREFIX

cd "${REPO_ROOT}"
mkdir -p "${MPLCONFIGDIR}" "${PYTHONPYCACHEPREFIX}"

run_cmd() {
  echo
  echo "$*"
  "$@"
}
