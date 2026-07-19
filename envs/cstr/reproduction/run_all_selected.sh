#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/run_cstr_ppo.sh"
bash "${SCRIPT_DIR}/run_graph_variable_k.sh"
bash "${SCRIPT_DIR}/run_graph_variable_k_zero_shot.sh"
bash "${SCRIPT_DIR}/run_figures.sh"
