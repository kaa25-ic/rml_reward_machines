#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
RESULTS_DIR="${REPO_ROOT}/envs/letter_env/results_and_evaluation"

GRU_CHECKPOINT="${RESULTS_DIR}/encoder_pretraining/gru_dim16_seed0/best_student.pt"
GNN_CHECKPOINT="${RESULTS_DIR}/encoder_pretraining/gnn_basic_seed0/best_dynamics_encoder.pt"

cd "${REPO_ROOT}"

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
