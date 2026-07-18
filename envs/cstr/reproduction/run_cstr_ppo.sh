#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000}"
SEED="${SEED:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PPO_DIR}}"
GRAPH_ENCODER_CHECKPOINT="${GRAPH_ENCODER_CHECKPOINT:-${RESULTS_DIR}/encoder_pretraining/gnn_dynamics_phase_count_reference_seed0/best_dynamics_encoder.pt}"

COMMON_ARGS=(
  --total-timesteps "${TOTAL_TIMESTEPS}"
  --seed "${SEED}"
  --max-episode-steps 300
  --regulation-violation-steps 10
  --soak-steps 10
  --monitor-state-limit 16
  --fixed-initial-state
  --eval-freq 20000
  --n-eval-episodes 10
  --ent-coef 0.001
  --log-std-init -1.5
  --stable-step-bonus 3.0
  --rml-heating-rate-penalty 0.0
  --heating-rate-penalty 0.0
  --deadline-steps 100
  --require-soak-concentration-band
  --production-temp-low 346.0
  --production-temp-high 354.0
  --concentration-tolerance 0.08
)

run_cmd "${PYTHON_BIN}" -m envs.cstr.experiments.train_cstr_ppo \
  --env-variant baseline \
  --reward-mode env \
  "${COMMON_ARGS[@]}" \
  --output-dir "${OUTPUT_ROOT}/baseline_seed${SEED}"

run_cmd "${PYTHON_BIN}" -m envs.cstr.experiments.train_cstr_ppo \
  --env-variant rml_hidden \
  --reward-mode env_rml \
  "${COMMON_ARGS[@]}" \
  --output-dir "${OUTPUT_ROOT}/rml_hidden_seed${SEED}"

run_cmd "${PYTHON_BIN}" -m envs.cstr.experiments.train_cstr_ppo \
  --env-variant semantic_progress \
  --reward-mode env_rml \
  "${COMMON_ARGS[@]}" \
  --output-dir "${OUTPUT_ROOT}/rml_semantic_progress_seed${SEED}"

run_cmd "${PYTHON_BIN}" -m envs.cstr.experiments.train_cstr_ppo \
  --env-variant manual_rm_semantic_progress \
  --reward-mode env_rml \
  "${COMMON_ARGS[@]}" \
  --output-dir "${OUTPUT_ROOT}/manual_rm_semantic_progress_seed${SEED}"

if [[ -n "${GRAPH_ENCODER_CHECKPOINT:-}" ]]; then
  run_cmd "${PYTHON_BIN}" -m envs.cstr.experiments.train_cstr_ppo \
    --env-variant rml_graph \
    --reward-mode env_rml \
    --graph-encoder-checkpoint "${GRAPH_ENCODER_CHECKPOINT}" \
    "${COMMON_ARGS[@]}" \
    --output-dir "${OUTPUT_ROOT}/rml_graph_encoder_seed${SEED}"
fi
