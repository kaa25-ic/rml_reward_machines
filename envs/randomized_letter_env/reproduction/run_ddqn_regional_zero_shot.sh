#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

MODEL_PATH="${RESULTS_DIR}/ddqn/regional_randomness_n1to5/semantic_progress_n1to5_seed0/model_final.zip"

for eval_n in 10 15 20; do
  run_cmd "${PYTHON_BIN}" -m envs.randomized_letter_env.experiments.evaluate_zero_shot \
    --encoding semantic_progress \
    --placement-mode regional \
    --train-seed 0 \
    --eval-n "${eval_n}" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${RESULTS_DIR}/generalization/semantic_progress_regional_zeroshot_n${eval_n}_seed0" \
    --n-eval-episodes 100 \
    --eval-seed-base 50000 \
    --max-episode-steps 200 \
    --monitor-progress-bonus 10
done
