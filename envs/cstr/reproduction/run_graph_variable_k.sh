#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000}"
SEED="${SEED:-1}"
TRAIN_SOAK_STEPS=(${TRAIN_SOAK_STEPS:-5 8 10 12})
EVAL_SOAK_STEPS=(${EVAL_SOAK_STEPS:-5 8 10 12})
OUTPUT_DIR="${OUTPUT_DIR:-${PPO_DIR}/rml_graph_variable_k_seed${SEED}}"
GRAPH_ENCODER_CHECKPOINT="${GRAPH_ENCODER_CHECKPOINT:-${RESULTS_DIR}/encoder_pretraining/gnn_dynamics_phase_count_reference_seed0/best_dynamics_encoder.pt}"

run_cmd "${PYTHON_BIN}" -m envs.cstr.experiments.train_cstr_graph_variable_k \
  --train-soak-steps "${TRAIN_SOAK_STEPS[@]}" \
  --eval-soak-steps "${EVAL_SOAK_STEPS[@]}" \
  --graph-encoder-checkpoint "${GRAPH_ENCODER_CHECKPOINT}" \
  --total-timesteps "${TOTAL_TIMESTEPS}" \
  --seed "${SEED}" \
  --eval-freq 20000 \
  --n-eval-episodes 10 \
  --max-episode-steps 300 \
  --deadline-steps 100 \
  --n-steps 1024 \
  --batch-size 64 \
  --n-epochs 10 \
  --learning-rate 0.0003 \
  --ent-coef 0.001 \
  --log-std-init -1.5 \
  --training-failure-penalty -25.0 \
  --output-dir "${OUTPUT_DIR}"
