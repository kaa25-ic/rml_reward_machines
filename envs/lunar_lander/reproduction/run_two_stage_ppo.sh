#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

SEEDS="${SEEDS:-0 1 2 3 4}"
require_file "${WARM_START_MODEL}"

for SEED in ${SEEDS}; do
  run_cmd "${PYTHON_BIN}" envs/lunar_lander/experiments/train_ppo_two_stage.py \
    --seed "${SEED}" \
    --run-name "semantic_progress_two_stage_seed${SEED}" \
    --output-root "${TWO_STAGE_DIR}" \
    --stage1-initial-model "${WARM_START_MODEL}" \
    --stage1-timesteps 1000000 \
    --stage1-learning-rate 0.0003 \
    --stage2-timesteps 300000 \
    --stage2-learning-rate 0.0001 \
    --n-eval-episodes 50 \
    --eval-freq 50000 \
    --success-bonus 200 \
    --failure-penalty -100 \
    --landing-target-bonus 0 \
    --landing-angle-bonus 0 \
    --post-descent-landing-bonus 0 \
    --post-descent-protocol-miss-penalty 0
done
