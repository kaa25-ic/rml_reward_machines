#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

TRAIN_ROOT="${RESULTS_DIR}/experiments_with_variable_n/tabular_n_1_to_5"
OUTPUT_ROOT="${RESULTS_DIR}/generalization_experiments_with_zero_shot_on_larger_n/tabular"

for encoding in simple one_hot numerical; do
  q_table="${TRAIN_ROOT}/${encoding}_seed0/q_tables/${encoding}_iteration0_n5_seed0.pkl"
  require_file "${q_table}"

  for n in 10 15 20; do
    run_cmd "${PYTHON_BIN}" -m envs.letter_env.experiments.evaluate_zero_shot \
      --algorithm tabular \
      --encoding "${encoding}" \
      --train-seed 0 \
      --eval-n "${n}" \
      --model-path "${q_table}" \
      --output-dir "${OUTPUT_ROOT}/${encoding}_zeroshot_n${n}_seed0" \
      --n-eval-episodes 20
  done
done
