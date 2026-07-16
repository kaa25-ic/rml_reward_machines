#!/usr/bin/env bash
set -euo pipefail

# Reproduce the retained warm-start source run.
#
# This run intentionally preserves the earlier terminal reward values recorded
# in results_and_evaluation/ppo/semantic_progress_success_aligned_seed0/config.json.
# The newer shaping fields are explicitly set to zero so the command remains
# stable even if LunarLander training defaults change again.

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python3}"
OUTPUT_DIR="${OUTPUT_DIR:-/private/tmp/rml_lunar_reproduction/semantic_progress_success_aligned_seed0}"

MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/mplconfig}" \
PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/private/tmp/rml_pycache}" \
"${PYTHON_BIN}" envs/lunar_lander/experiments/train_ppo.py \
  --encoding semantic_progress \
  --seed 0 \
  --total-timesteps 500000 \
  --learning-rate 0.0003 \
  --n-steps 2048 \
  --batch-size 64 \
  --n-epochs 10 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --clip-range 0.2 \
  --ent-coef 0.0 \
  --vf-coef 0.5 \
  --max-grad-norm 0.5 \
  --eval-freq 20000 \
  --n-eval-episodes 50 \
  --eval-seed-base 10000 \
  --max-episode-steps 1000 \
  --monitor-progress-bonus 20.0 \
  --hover-step-bonus 2.0 \
  --hover-complete-bonus 30.0 \
  --controlled-descent-bonus 20.0 \
  --success-bonus 100.0 \
  --failure-penalty -25.0 \
  --landing-target-bonus 0.0 \
  --landing-angle-bonus 0.0 \
  --post-descent-landing-bonus 0.0 \
  --post-descent-protocol-miss-penalty 0.0 \
  --output-dir "${OUTPUT_DIR}"
