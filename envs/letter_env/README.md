# LetterEnv

LetterEnv is a sequential reinforcement learning environment for evaluating
how reward-machine monitor states can be represented for learning. Each episode
samples or fixes a target sequence length `n`. The agent must complete the
letter sequence while an RML monitor tracks progress through the task and
provides the monitor state used by the observation encodings.

This environment is used to compare tabular learning, DQN, Double DQN, and PPO
under monitor-state encodings that expose the same reward-machine state in
different forms.

## Environment Setup

The environment package contains:

- `env.py`: the Gymnasium-compatible LetterEnv implementation.
- `builder.py`: environment construction helpers used by training scripts.
- `encodings.py`: LetterEnv observation and monitor-state encoding helpers.
- `configs/letter_env.yaml`: the monitor runtime configuration template.
- `configs/monitor_state_catalogue.json`: the monitor-state catalogue used by
  one-hot and numerical encodings.
- `specs/letter_env_spec_numerical_runtime_compatible.pl`: the RML monitor
  specification used by the current experiments.
- `experiments/`: command-line entry points for training and evaluation.

RML monitoring requires SWI-Prolog. Install the project and verify SWI-Prolog
from the repository root:

```bash
source .venv/bin/activate
python -m pip install -e .
swipl --version
```

## Encodings

The implemented neural experiments currently use:

- `one_hot`: a one-hot vector derived from the RML monitor state.
- `numerical`: a compact numerical representation derived from the RML monitor
  state.

The tabular reproduction also includes `simple`, which is used for comparison
with the baseline tabular state abstraction for this environment.

Additional semantic, recurrent, and graph-based encodings can be added through
the shared encoding interface in `rml_rm` while keeping the LetterEnv training
entry points unchanged.

## Experiment Groups

Generated outputs should be written under:

```text
envs/letter_env/results_and_evaluation/
```

This directory is ignored by git. The current experiment layout is:

```text
results_and_evaluation/
  experiments_with_variable_n/
    dqn/
    ddqn/
    ppo/
    original_tabular_reproduction/
  generalization_experiments_with_zero_shot_on_larger_n/
    dqn/
    ddqn/
    ppo/
```

Each neural run writes its own folder containing the run configuration, final
model checkpoint, monitor logs, training monitor CSVs, evaluation metrics, and
summary JSON. Tabular reproduction runs write episode-level metrics and a
summary JSON.

## Training

DQN and Double DQN share the same entry point. Use `--algorithm dqn` for DQN
and `--algorithm ddqn` for Double DQN.

Example Double DQN run with numerical encoding and variable `n` from 1 to 5:

```bash
python -m envs.letter_env.experiments.train_dqn \
  --algorithm ddqn \
  --encoding numerical \
  --n-value 5 \
  --seed 0 \
  --output-dir envs/letter_env/results_and_evaluation/experiments_with_variable_n/ddqn/numerical_n_1to5_seed0 \
  --total-timesteps 500000 \
  --learning-rate 0.001 \
  --buffer-size 100000 \
  --learning-starts 5000 \
  --batch-size 64 \
  --gamma 0.9 \
  --exploration-fraction 0.4 \
  --eval-freq 20000 \
  --n-eval-episodes 20 \
  --monitor-progress-bonus 10 \
  --monitor-regression-penalty 0
```

Example PPO run:

```bash
python -m envs.letter_env.experiments.train_ppo \
  --encoding numerical \
  --n-value 5 \
  --seed 0 \
  --output-dir envs/letter_env/results_and_evaluation/experiments_with_variable_n/ppo/numerical_n_1to5_seed0 \
  --total-timesteps 500000 \
  --n-steps 16384 \
  --batch-size 64 \
  --ent-coef 0.05 \
  --step-penalty 0.05 \
  --eval-freq 20000 \
  --n-eval-episodes 20 \
  --monitor-progress-bonus 10 \
  --monitor-regression-penalty 0
```

Example tabular reproduction run:

```bash
python -m envs.letter_env.experiments.train_tabular \
  --encoding all \
  --max-n 10 \
  --n-values 1 2 3 4 5 6 7 8 9 10 \
  --iterations 1 \
  --output-dir envs/letter_env/results_and_evaluation/experiments_with_variable_n/original_tabular_reproduction/tabular_reproduction_from_previous_thesis
```

For multi-seed experiments, use one output folder per seed, for example
`one_hot_n_1to5_seed0`, `one_hot_n_1to5_seed1`, and so on. A reproduction
script will be added later to run the selected experiment batches from a single
command.

## Zero-Shot Evaluation

Saved neural policies can be evaluated on fixed larger values of `n` using:

```bash
python -m envs.letter_env.experiments.evaluate_zero_shot \
  --algorithm ddqn \
  --encoding numerical \
  --train-seed 0 \
  --eval-n 10 \
  --model-path envs/letter_env/results_and_evaluation/experiments_with_variable_n/ddqn/numerical_n_1to5_seed0/model_final.zip \
  --output-dir envs/letter_env/results_and_evaluation/generalization_experiments_with_zero_shot_on_larger_n/ddqn/numerical_zeroshot_n10_seed0 \
  --n-eval-episodes 20
```

The zero-shot output folder contains an evaluation CSV, monitor logs, copied
runtime monitor configuration, and summary JSON.

## Tests And Smoke Checks

Smoke checks should write to `/private/tmp` or another temporary location, not
to `results_and_evaluation/`. This keeps repository-local results reserved for
intentional experiment runs.

Basic command checks:

```bash
python -m envs.letter_env.experiments.train_dqn --help
python -m envs.letter_env.experiments.train_ppo --help
python -m envs.letter_env.experiments.train_tabular --help
python -m envs.letter_env.experiments.evaluate_zero_shot --help
```

Environment-specific tests will be added under a dedicated test directory once
the final experiment set is fixed.
