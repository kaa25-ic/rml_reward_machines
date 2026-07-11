#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

for seed in 0 1 2 3 4; do
  for encoding in numerical one_hot; do
    output_dir="${RESULTS_DIR}/tabular/${encoding}_n_1to5_seed${seed}"

    run_cmd "${PYTHON_BIN}" -m envs.multitask_letter_env.experiments.train_tabular \
      --encoding "${encoding}" \
      --task-suite small_v1 \
      --max-n 5 \
      --episodes 25000 \
      --seed "${seed}" \
      --output-dir "${output_dir}"
  done
done
