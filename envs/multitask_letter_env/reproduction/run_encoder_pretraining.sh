#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

run_cmd "${PYTHON_BIN}" -m envs.multitask_letter_env.experiments.generate_gru_monitor_dataset \
  --task-suite small_v1 \
  --max-n 5 \
  --seed 0 \
  --output-path "${GRU_DATASET}"

run_cmd "${PYTHON_BIN}" -m envs.multitask_letter_env.experiments.train_gru_encoder \
  --dataset-path "${GRU_DATASET}" \
  --output-dir "${RESULTS_DIR}/encoder_pretraining/gru_dim32_seed0" \
  --seed 0

run_cmd "${PYTHON_BIN}" -m envs.multitask_letter_env.experiments.generate_gnn_monitor_corpus \
  --task-suite small_v1 \
  --max-n 5 \
  --seed 0 \
  --output-path "${GNN_CORPUS}"

run_cmd "${PYTHON_BIN}" -m envs.multitask_letter_env.experiments.train_gnn_encoder \
  --dataset-path "${GNN_CORPUS}" \
  --output-dir "${RESULTS_DIR}/encoder_pretraining/gnn_basic_seed0" \
  --seed 0
