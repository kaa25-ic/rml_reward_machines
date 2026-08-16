#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

OUTPUT_ROOT="${RESULTS_DIR}/experiments_with_variable_n/tabular_n_1_to_5"

for encoding in simple one_hot numerical; do
  run_cmd "${PYTHON_BIN}" -m envs.letter_env.experiments.train_tabular \
    --encoding "${encoding}" \
    --max-n 5 \
    --n-values 5 \
    --sample-n \
    --seed-base 0 \
    --output-dir "${OUTPUT_ROOT}/${encoding}_seed0"
done
