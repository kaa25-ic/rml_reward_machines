# LunarLander Protocol

This environment evaluates an RML-monitored LunarLander landing protocol. The
agent observes the standard LunarLander state together with an encoded RML
monitor state. The monitor tracks whether the lander enters the descent corridor,
hovers for the required duration, performs controlled descent, and lands safely
inside the target zone.

The Python environment only derives low-level propositions from the simulator
state. Protocol progress, success, and failure are handled by the RML monitor in
`specs/lunar_lander_protocol.pl`.

## Current Setup

- Algorithm: PPO
- Encoding: `semantic_progress`
- Main reward setting: `+200` on strict RML success, `-100` on strict RML failure
- Additional shaping: semantic monitor-progress, hover completion, controlled descent,
  success, and failure terms
- Base simulator: Gymnasium LunarLander

The Python side does not implement a protocol tracker. It only derives simulator
propositions such as corridor, hover, controlled descent, target zone, contact,
and successful simulator landing. The RML monitor determines protocol progress,
success, and failure.

## Two-Stage PPO

`experiments/train_ppo_two_stage.py` is a separate orchestration script for the
discovery-then-stabilization PPO experiment. Stage 1 trains with the discovery
learning rate and saves its normal artifacts under `stage1_discovery`. Stage 2
then automatically loads `stage1_discovery/best_model.zip` and fine-tunes it
with the lower stabilization learning rate under `stage2_stabilization`.

By default, all two-stage runs are saved under:

```text
envs/lunar_lander/results_and_evaluation/ppo/two_stage_training/
```

The single-stage `train_ppo.py` path is unchanged.

The main clean experiment is the from-scratch two-stage PPO run:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig PYTHONPYCACHEPREFIX=/private/tmp/rml_pycache ./.venv/bin/python3 envs/lunar_lander/experiments/train_ppo_two_stage.py \
  --seed 0 \
  --run-name semantic_progress_two_stage_from_scratch_seed0 \
  --stage1-timesteps 1000000 \
  --stage1-learning-rate 0.0003 \
  --stage2-timesteps 300000 \
  --stage2-learning-rate 0.0001 \
  --n-eval-episodes 50 \
  --eval-freq 50000 \
  --success-bonus 200 \
  --failure-penalty -100 \
  --landing-target-bonus 0 \
  --landing-angle-bonus 0 \
  --post-descent-landing-bonus 0 \
  --post-descent-protocol-miss-penalty 0
```

Repeat with seeds `0` through `4`, changing only `--seed` and `--run-name`.
The warm-start two-stage runs are retained as an ablation; their stage 1 starts
from `results_and_evaluation/ppo/semantic_progress_success_aligned_seed0/best_model.zip`.

The cleaned PPO results folder intentionally keeps only:

```text
envs/lunar_lander/results_and_evaluation/ppo/semantic_progress_success_aligned_seed0/
envs/lunar_lander/results_and_evaluation/ppo/two_stage_training/
```

## Rendering Policies

`experiments/render_policy.py` renders trained PPO checkpoints through the same
RML monitor stack used during training. It can render one run directory or batch
render all runs under a two-stage result root. Each render writes an episode
summary, a step-by-step trajectory CSV, and a root-level render index. Use
`--record-gif` for dependency-light visual artifacts in the current project
environment; `--record-video` is also supported when MP4 video dependencies are
installed.

Render the final stage-2 policy for all two-stage seeds:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig PYTHONPYCACHEPREFIX=/private/tmp/rml_pycache ./.venv/bin/python3 envs/lunar_lander/experiments/render_policy.py \
  --runs-root envs/lunar_lander/results_and_evaluation/ppo/two_stage_training \
  --stage stage2_stabilization \
  --model model_final \
  --record-gif \
  --seed 10000
```

The default output location for that command is:

```text
envs/lunar_lander/results_and_evaluation/ppo/two_stage_training/rendering/stage2_stabilization_model_final/
```
