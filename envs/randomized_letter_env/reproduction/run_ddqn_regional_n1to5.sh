#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

for seed in 0 1 2 3 4; do
  run_cmd "${PYTHON_BIN}" -m envs.randomized_letter_env.experiments.train_ddqn \
    --encoding semantic_progress \
    --placement-mode regional \
    --n-value 5 \
    --sample-n \
    --seed "${seed}" \
    --output-dir "${RESULTS_DIR}/ddqn/regional_randomness_n1to5/semantic_progress_n1to5_seed${seed}" \
    --total-timesteps 800000 \
    --learning-rate 3e-4 \
    --batch-size 128 \
    --gamma 0.9 \
    --buffer-size 300000 \
    --learning-starts 20000 \
    --target-update-interval 8000 \
    --exploration-fraction 0.20 \
    --exploration-final-eps 0.04 \
    --monitor-progress-bonus 10
done
