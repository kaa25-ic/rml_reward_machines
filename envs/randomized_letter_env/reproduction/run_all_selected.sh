#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_ddqn_full_random_n1.sh"
"${SCRIPT_DIR}/run_ddqn_regional_n1to5.sh"
"${SCRIPT_DIR}/run_q_learning_regional_n1to5.sh"
"${SCRIPT_DIR}/run_ddqn_regional_zero_shot.sh"
"${SCRIPT_DIR}/run_figures.sh"
