#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_file "${GRU_CHECKPOINT}"
require_file "${GNN_CHECKPOINT}"

run_zero_shot() {
  local algorithm="$1"
  local encoding="$2"
  local n="$3"
  local model_path="${RESULTS_DIR}/experiments_with_variable_n/${algorithm}/${encoding}_n_1to5_seed0/best_model.zip"
  local output_dir="${RESULTS_DIR}/generalization_experiments_with_zero_shot_on_larger_n/${algorithm}/${encoding}_zeroshot_n${n}_seed0"
  local extra_args=()

  require_file "${model_path}"

  if [[ "${encoding}" == "learned_gru" ]]; then
    extra_args+=(--learned-gru-checkpoint "${GRU_CHECKPOINT}")
  elif [[ "${encoding}" == "learned_graph" ]]; then
    extra_args+=(--learned-graph-checkpoint "${GNN_CHECKPOINT}")
  fi

  run_cmd "${PYTHON_BIN}" -m envs.letter_env.experiments.evaluate_zero_shot \
    --algorithm "${algorithm}" \
    --encoding "${encoding}" \
    --train-seed 0 \
    --eval-n "${n}" \
    --model-path "${model_path}" \
    "${extra_args[@]}" \
    --output-dir "${output_dir}" \
    --n-eval-episodes 20
}

for n in 10 15 20; do
  for algorithm in dqn ppo; do
    for encoding in numerical one_hot; do
      run_zero_shot "${algorithm}" "${encoding}" "${n}"
    done
  done

  for encoding in numerical one_hot semantic_progress learned_gru learned_graph; do
    run_zero_shot ddqn "${encoding}" "${n}"
  done
done
