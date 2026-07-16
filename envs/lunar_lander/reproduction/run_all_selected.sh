#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_two_stage_ppo.sh"
"${SCRIPT_DIR}/run_render_stage2.sh"
"${SCRIPT_DIR}/run_figures.sh"
