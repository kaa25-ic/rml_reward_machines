#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
RESULTS_DIR="${REPO_ROOT}/envs/multitask_letter_env/results_and_evaluation"
MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/rml_rm_matplotlib}"
export MPLCONFIGDIR

GRU_DATASET="${RESULTS_DIR}/encoder_pretraining/gru_dataset_small_v1_seed0/dataset.jsonl"
GRU_CHECKPOINT="${RESULTS_DIR}/encoder_pretraining/gru_dim32_seed0/best_student.pt"
GNN_CORPUS="${RESULTS_DIR}/encoder_pretraining/gnn_corpus_small_v1_seed0/monitor_states.jsonl"
GNN_CHECKPOINT="${RESULTS_DIR}/encoder_pretraining/gnn_basic_seed0/best_dynamics_encoder.pt"

cd "${REPO_ROOT}"
mkdir -p "${MPLCONFIGDIR}"

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
