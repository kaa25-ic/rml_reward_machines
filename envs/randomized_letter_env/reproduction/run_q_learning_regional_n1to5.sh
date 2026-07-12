#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

for seed in 0 1 2 3 4; do
  run_cmd "${PYTHON_BIN}" -m envs.randomized_letter_env.experiments.train_q_learning \
    --encoding semantic_progress \
    --placement-mode regional \
    --n-value 5 \
    --sample-n \
    --seed "${seed}" \
    --output-dir "${RESULTS_DIR}/q_learning/semantic_progress_n1to5_seed${seed}" \
    --episodes 50000 \
    --alpha 0.5 \
    --gamma 0.9 \
    --epsilon 0.4 \
    --epsilon-decay 0.99995 \
    --min-epsilon 0.05 \
    --eval-freq-episodes 1000 \
    --train-log-freq-episodes 1000 \
    --n-eval-episodes 25 \
    --monitor-progress-bonus 10
done
