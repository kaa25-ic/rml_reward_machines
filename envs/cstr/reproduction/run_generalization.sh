#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

SEED="${SEED:-0}"
EVAL_SOAK_STEPS="${EVAL_SOAK_STEPS:-15}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RESULTS_DIR}/generalization/soak${EVAL_SOAK_STEPS}_seed${SEED}}"
GRAPH_ENCODER_CHECKPOINT="${GRAPH_ENCODER_CHECKPOINT:-${RESULTS_DIR}/encoder_pretraining/gnn_dynamics_phase_count_reference_seed0/best_dynamics_encoder.pt}"

run_cmd "${PYTHON_BIN}" -m envs.cstr.experiments.evaluate_generalization \
  --train-seed "${SEED}" \
  --eval-soak-steps "${EVAL_SOAK_STEPS}" \
  --model-root "${PPO_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --graph-encoder-checkpoint "${GRAPH_ENCODER_CHECKPOINT}"
