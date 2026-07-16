# LunarLander Protocol

This environment evaluates an RML-monitored LunarLander landing protocol. The
agent observes the standard LunarLander state together with an encoded RML
monitor state. The monitor tracks whether the lander enters the descent
corridor, hovers for the required duration, performs controlled descent, and
lands safely inside the target zone.

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

The selected experiment uses a two-stage PPO procedure. Stage 1 is a discovery
stage initialized from the retained single-stage seed-0 policy. Stage 2
stabilizes the best stage-1 checkpoint with a lower learning rate. This keeps
the final policy stable while preserving the strictly RML-based protocol
monitoring.

## Testing

From the repository root, run the shared core tests and LunarLander protocol
tests with:

```bash
./.venv/bin/python3 -m pytest tests/core tests/lunar_lander
```

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

The selected two-stage experiment uses the retained seed-0 single-stage policy
as the stage-1 warm start, then trains one two-stage run per seed. Example
seed-0 command:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig PYTHONPYCACHEPREFIX=/private/tmp/rml_pycache ./.venv/bin/python3 envs/lunar_lander/experiments/train_ppo_two_stage.py \
  --seed 0 \
  --run-name semantic_progress_two_stage_seed0 \
  --stage1-initial-model envs/lunar_lander/results_and_evaluation/ppo/semantic_progress_success_aligned_seed0/best_model.zip \
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

Each two-stage run writes:

- `stage1_discovery/`: stage-1 model checkpoints, monitor configs, logs,
  evaluation CSV, and summary JSON;
- `stage2_stabilization/`: stage-2 model checkpoints, monitor configs, logs,
  evaluation CSV, and summary JSON;
- `combined_eval_metrics.csv`: a stage-labelled evaluation curve with global
  training steps;
- `summary.json`: the combined two-stage configuration and artifact index.

The retained warm-start source was trained with the earlier terminal reward
values recorded in its saved `config.json` (`success_bonus=100`,
`failure_penalty=-25`). To reproduce that source run safely after the defaults
refactor, use the explicit command file:

```bash
bash envs/lunar_lander/reproduction/run_warm_start_source_seed0.sh
```

By default, the script writes to:

```text
/private/tmp/rml_lunar_reproduction/semantic_progress_success_aligned_seed0
```

Set `OUTPUT_DIR=...` if you want a different destination. The script does not
overwrite the retained committed run unless you deliberately point `OUTPUT_DIR`
at that result folder.

The retained PPO artifacts used by the selected experiment are:

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

The trajectory CSVs are used by the qualitative phase-colored trajectory figure
and by the golden monitor-state tests under `tests/lunar_lander/`.

## Figures

Report figures are generated from the saved two-stage evaluation CSVs and the
rendered stage-2 trajectories:

- landing vs RML protocol learning curves over successful seeds;
- the landing/protocol success gap;
- seed-0 landing vs protocol learning curve;
- a phase-colored successful landing trajectory.

Generate figures with:

```bash
./.venv/bin/python3 -m envs.lunar_lander.analysis.generate_figures --formats pdf png
```

The default output location is:

```text
envs/lunar_lander/results_and_evaluation/figures/
```

## Reproducing Results

Reproduction scripts are provided in `reproduction/`. They use the same output
layout as the tracked experiment artifacts:

```text
results_and_evaluation/
  ppo/
    semantic_progress_success_aligned_seed0/
    two_stage_training/
      semantic_progress_two_stage_seed0/
      ...
      rendering/
  figures/
```

Run the selected two-stage PPO experiments for seeds `0..4`:

```bash
bash envs/lunar_lander/reproduction/run_two_stage_ppo.sh
```

Render the final stage-2 policies:

```bash
bash envs/lunar_lander/reproduction/run_render_stage2.sh
```

Generate report figures:

```bash
bash envs/lunar_lander/reproduction/run_figures.sh
```

The selected LunarLander pipeline can be run with:

```bash
bash envs/lunar_lander/reproduction/run_all_selected.sh
```

The full pipeline reruns the five-seed two-stage PPO experiments, renders the
stage-2 final policies, and regenerates the figure artifacts. It does not delete
existing result folders before running. Use `SEEDS="0 1"` with
`run_two_stage_ppo.sh` for a smaller partial rerun.

All reproduction scripts accept `PYTHON_BIN=...` if a different Python
interpreter should be used.
