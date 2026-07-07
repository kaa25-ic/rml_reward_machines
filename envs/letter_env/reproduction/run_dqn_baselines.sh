#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

for seed in 0 1 2 3 4; do
  for encoding in numerical one_hot; do
    run_cmd "${PYTHON_BIN}" -m envs.letter_env.experiments.train_dqn \
      --algorithm dqn \
      --encoding "${encoding}" \
      --n-value 5 \
      --seed "${seed}" \
      --output-dir "${RESULTS_DIR}/experiments_with_variable_n/dqn/${encoding}_n_1to5_seed${seed}" \
      --total-timesteps 500000
  done
done
