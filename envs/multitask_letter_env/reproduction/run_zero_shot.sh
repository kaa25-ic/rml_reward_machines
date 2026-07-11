#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

for encoding in numerical one_hot learned_gru learned_graph; do
  artifact_path="${RESULTS_DIR}/ddqn/${encoding}_n_1to5_seed0/model_final.zip"
  require_file "${artifact_path}"

  for eval_n in 10 15 20; do
    run_cmd "${PYTHON_BIN}" -m envs.multitask_letter_env.experiments.evaluate_zero_shot \
      --algorithm ddqn \
      --encoding "${encoding}" \
      --train-seed 0 \
      --eval-n "${eval_n}" \
      --artifact-path "${artifact_path}" \
      --output-dir "${RESULTS_DIR}/generalization/ddqn/${encoding}_zeroshot_n${eval_n}_seed0"
  done
done

for encoding in numerical one_hot; do
  artifact_path="${RESULTS_DIR}/tabular/${encoding}_n_1to5_seed0/q_table.pkl"
  require_file "${artifact_path}"

  for eval_n in 10 15 20; do
    run_cmd "${PYTHON_BIN}" -m envs.multitask_letter_env.experiments.evaluate_zero_shot \
      --algorithm tabular \
      --encoding "${encoding}" \
      --train-seed 0 \
      --eval-n "${eval_n}" \
      --artifact-path "${artifact_path}" \
      --output-dir "${RESULTS_DIR}/generalization/tabular/${encoding}_zeroshot_n${eval_n}_seed0"
  done
done
