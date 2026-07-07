#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_file "${GRU_CHECKPOINT}"
require_file "${GNN_CHECKPOINT}"

for seed in 0 1 2 3 4; do
  for encoding in numerical one_hot semantic_progress learned_gru learned_graph; do
    output_dir="${RESULTS_DIR}/experiments_with_variable_n/ddqn/${encoding}_n_1to5_seed${seed}"
    extra_args=()
    if [[ "${encoding}" == "learned_gru" ]]; then
      extra_args+=(--learned-gru-checkpoint "${GRU_CHECKPOINT}")
    elif [[ "${encoding}" == "learned_graph" ]]; then
      extra_args+=(--learned-graph-checkpoint "${GNN_CHECKPOINT}")
    fi

    run_cmd "${PYTHON_BIN}" -m envs.letter_env.experiments.train_dqn \
      --algorithm ddqn \
      --encoding "${encoding}" \
      "${extra_args[@]}" \
      --n-value 5 \
      --seed "${seed}" \
      --output-dir "${output_dir}" \
      --total-timesteps 500000
  done
done
