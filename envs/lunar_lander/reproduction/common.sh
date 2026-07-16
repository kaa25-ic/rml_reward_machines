#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"
RESULTS_DIR="${REPO_ROOT}/envs/lunar_lander/results_and_evaluation"
PPO_DIR="${RESULTS_DIR}/ppo"
TWO_STAGE_DIR="${PPO_DIR}/two_stage_training"
WARM_START_RUN="${PPO_DIR}/semantic_progress_success_aligned_seed0"
WARM_START_MODEL="${WARM_START_MODEL:-${WARM_START_RUN}/best_model.zip}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/rml_rm_matplotlib}"
PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/private/tmp/rml_pycache}"
export MPLCONFIGDIR
export PYTHONPYCACHEPREFIX

cd "${REPO_ROOT}"
mkdir -p "${MPLCONFIGDIR}" "${PYTHONPYCACHEPREFIX}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

run_cmd() {
  echo
  echo "$*"
  "$@"
}
