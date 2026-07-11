#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

run_cmd "${PYTHON_BIN}" -m envs.multitask_letter_env.analysis.generate_figures \
  --formats pdf png \
  --success-threshold 0.9 \
  --max-learning-steps 250000
