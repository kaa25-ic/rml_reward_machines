#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
RESULTS_DIR="${REPO_ROOT}/envs/randomized_letter_env/results_and_evaluation"

cd "${REPO_ROOT}"

run_cmd() {
  echo
  echo "$*"
  "$@"
}
