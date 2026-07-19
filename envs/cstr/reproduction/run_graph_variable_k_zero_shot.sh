#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

SEED="${SEED:-1}"
ZERO_SHOT_SOAK_STEPS=(${ZERO_SHOT_SOAK_STEPS:-15 18 20})
MODEL_PATH="${MODEL_PATH:-${PPO_DIR}/rml_graph_variable_k_seed${SEED}/best_model.zip}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RESULTS_DIR}/generalization/rml_graph_variable_k_seed${SEED}}"
GRAPH_ENCODER_CHECKPOINT="${GRAPH_ENCODER_CHECKPOINT:-${RESULTS_DIR}/encoder_pretraining/gnn_dynamics_phase_count_reference_seed0/best_dynamics_encoder.pt}"

for SOAK_STEPS in "${ZERO_SHOT_SOAK_STEPS[@]}"; do
  run_cmd "${PYTHON_BIN}" -m envs.cstr.experiments.evaluate_cstr_ppo \
    --model-path "${MODEL_PATH}" \
    --env-variant rml_graph \
    --reward-mode env_rml \
    --episodes 10 \
    --seed 10000 \
    --max-episode-steps 300 \
    --regulation-violation-steps 10 \
    --soak-steps "${SOAK_STEPS}" \
    --monitor-state-limit 32 \
    --graph-encoder-checkpoint "${GRAPH_ENCODER_CHECKPOINT}" \
    --safe-step-bonus 0.10 \
    --stable-step-bonus 3.0 \
    --regulation-entry-bonus 5.0 \
    --success-bonus 50.0 \
    --failure-penalty -50.0 \
    --rml-heating-rate-penalty 0.0 \
    --preheat-distance-weight 0.08 \
    --preheat-warming-weight 0.25 \
    --soak-entry-bonus 5.0 \
    --soak-progress-bonus 0.75 \
    --soak-reset-penalty -3.0 \
    --soak-lost-step-penalty 0.50 \
    --approach-distance-weight 1.0 \
    --approach-progress-bonus 5.0 \
    --approach-ca-progress-bonus 4.0 \
    --approach-temp-progress-bonus 4.0 \
    --approach-warming-weight 0.50 \
    --production-entry-bonus 10.0 \
    --regulate-recovery-penalty -10.0 \
    --deadline-steps 100 \
    --tracking-weight 0.5 \
    --heating-rate-penalty 0.0 \
    --critical-penalty 200.0 \
    --production-temp-low 346.0 \
    --production-temp-high 354.0 \
    --concentration-tolerance 0.08 \
    --ca-overshoot-low 0.44 \
    --require-soak-concentration-band \
    --soak-concentration-low 0.58 \
    --soak-concentration-high 0.74 \
    --fixed-initial-state \
    --temp-weight 0.015 \
    --action-weight 0.0002 \
    --warning-penalty 0.25 \
    --output-dir "${OUTPUT_ROOT}/k${SOAK_STEPS}"
done
