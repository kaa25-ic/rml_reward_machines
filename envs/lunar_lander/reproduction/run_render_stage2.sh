#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

run_cmd "${PYTHON_BIN}" envs/lunar_lander/experiments/render_policy.py \
  --runs-root "${TWO_STAGE_DIR}" \
  --stage stage2_stabilization \
  --model model_final \
  --record-gif \
  --seed 10000
