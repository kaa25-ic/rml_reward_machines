#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

FIGURES_DIR="${FIGURES_DIR:-${RESULTS_DIR}/figures}"
GENERALIZATION_DIR="${GENERALIZATION_DIR:-${RESULTS_DIR}/generalization}"
SEEDS="${SEEDS:-0 1 2 3 4}"
SEED0="${SEED0:-0}"
RML_HIDDEN_TRAJECTORY_SEED="${RML_HIDDEN_TRAJECTORY_SEED:-2}"
MANUAL_RM_TRAJECTORY_SEED="${MANUAL_RM_TRAJECTORY_SEED:-1}"
TRAJECTORY_SEED="${TRAJECTORY_SEED:-10000}"
SKIP_TRAJECTORIES="${SKIP_TRAJECTORIES:-0}"

read -r -a SEED_ARGS <<< "${SEEDS}"
EXTRA_ARGS=()
if [[ "${SKIP_TRAJECTORIES}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-trajectories)
fi

run_cmd "${PYTHON_BIN}" -m envs.cstr.analysis.generate_figures \
  --ppo-root "${PPO_DIR}" \
  --generalization-root "${GENERALIZATION_DIR}" \
  --output-dir "${FIGURES_DIR}" \
  --seeds "${SEED_ARGS[@]}" \
  --seed0 "${SEED0}" \
  --rml-hidden-trajectory-seed "${RML_HIDDEN_TRAJECTORY_SEED}" \
  --manual-rm-trajectory-seed "${MANUAL_RM_TRAJECTORY_SEED}" \
  --trajectory-seed "${TRAJECTORY_SEED}" \
  "${EXTRA_ARGS[@]}"
