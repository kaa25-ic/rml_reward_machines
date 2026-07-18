#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

CORPUS_DIR="${RESULTS_DIR}/encoder_pretraining/gnn_corpus_seed0"
ENCODER_DIR="${RESULTS_DIR}/encoder_pretraining/gnn_dynamics_phase_count_epoch4_seed0"

run_cmd "${PYTHON_BIN}" -m envs.cstr.experiments.generate_cstr_monitor_corpus \
  --output-dir "${CORPUS_DIR}" \
  --seed 0 \
  --max-episode-steps 300 \
  --soak-steps 10 \
  --deadline-steps 300 \
  --concentration-tolerance 0.08 \
  --production-temp-low 346.0 \
  --production-temp-high 354.0

run_cmd "${PYTHON_BIN}" -m envs.cstr.experiments.train_gnn_encoder \
  --dataset-path "${CORPUS_DIR}/monitor_states.jsonl" \
  --output-dir "${ENCODER_DIR}" \
  --seed 0 \
  --epochs 4 \
  --batch-size 8 \
  --dropout 0.1 \
  --node-value-embedding-dim 16 \
  --node-value-dropout 0.1 \
  --output-layer-norm \
  --phase-loss-weight 2.0 \
  --phase-class-weighting \
  --balanced-phase-sampling \
  --use-graph-structural-features \
  --device cpu
