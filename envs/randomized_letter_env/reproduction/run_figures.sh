#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

run_cmd "${PYTHON_BIN}" -m envs.randomized_letter_env.analysis.generate_figures \
  --formats pdf png
