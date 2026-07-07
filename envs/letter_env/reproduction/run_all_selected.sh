#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/run_ddqn_encodings.sh"
bash "${SCRIPT_DIR}/run_dqn_baselines.sh"
bash "${SCRIPT_DIR}/run_ppo_baselines.sh"
bash "${SCRIPT_DIR}/run_zero_shot.sh"
bash "${SCRIPT_DIR}/run_figures.sh"
