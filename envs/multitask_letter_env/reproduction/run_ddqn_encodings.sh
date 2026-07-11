#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_file "${GRU_CHECKPOINT}"
require_file "${GNN_CHECKPOINT}"

for seed in 0 1 2 3 4; do
  for encoding in numerical one_hot learned_gru learned_graph; do
    output_dir="${RESULTS_DIR}/ddqn/${encoding}_n_1to5_seed${seed}"
    extra_args=()
    if [[ "${encoding}" == "learned_gru" ]]; then
      extra_args+=(--learned-gru-checkpoint "${GRU_CHECKPOINT}")
    elif [[ "${encoding}" == "learned_graph" ]]; then
      extra_args+=(--learned-graph-checkpoint "${GNN_CHECKPOINT}")
    fi

    run_cmd "${PYTHON_BIN}" -m envs.multitask_letter_env.experiments.train_ddqn \
      --encoding "${encoding}" \
      "${extra_args[@]}" \
      --task-suite small_v1 \
      --max-n 5 \
      --seed "${seed}" \
      --output-dir "${output_dir}" \
      --total-timesteps 500000
  done
done
