#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_encoder_pretraining.sh"
"${SCRIPT_DIR}/run_ddqn_encodings.sh"
"${SCRIPT_DIR}/run_tabular_baselines.sh"
"${SCRIPT_DIR}/run_zero_shot.sh"
"${SCRIPT_DIR}/run_figures.sh"
