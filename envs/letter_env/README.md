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
  one-hot, numerical, and semantic progress encodings.
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

The neural experiments use five monitor-state encodings:

- `one_hot`: a one-hot vector derived directly from the RML monitor state.
- `numerical`: a compact numerical representation derived from the RML monitor
  state.
- `semantic_progress`: a task-phase one-hot vector derived from the RML monitor
  state.
- `learned_gru`: a frozen 16-dimensional recurrent encoder trained from a
  teacher dataset collected from a trained LetterEnv policy.
- `learned_graph`: a frozen 32-dimensional graph encoder trained from
  parameterized RML monitor-transition data.

The tabular reproduction also includes `simple`, which is used for comparison
with the baseline tabular state abstraction for this environment.

The learned encoders are stored under
`results_and_evaluation/encoder_pretraining/`. The canonical checkpoints used by
the experiments are `gru_dim16_seed0/best_student.pt` and
`gnn_basic_seed0/best_dynamics_encoder.pt`.

## Experiment Groups

Generated outputs should be written under:

```text
envs/letter_env/results_and_evaluation/
```

This directory is ignored by git. The current experiment layout is:

```text
results_and_evaluation/
  encoder_pretraining/
    gru_teacher_dataset_n1to5_seed0/
    gru_dim16_seed0/
    gnn_parameterized_corpus_n1to5_seed0/
    gnn_basic_seed0/
  experiments_with_variable_n/
    dqn/
    ddqn/
    ppo/
    original_tabular_reproduction/
  generalization_experiments_with_zero_shot_on_larger_n/
    dqn/
    ddqn/
    ppo/
  figures/
```

Each neural run writes its own folder containing the run configuration, final
model checkpoint, monitor logs, training monitor CSVs, evaluation metrics, and
summary JSON. Tabular reproduction runs write episode-level metrics and a
summary JSON.

## Learned Encoder Pretraining

The learned GRU encoder is trained in two stages: first collect a teacher
dataset, then distill a frozen monitor encoder from that dataset.

```bash
python -m envs.letter_env.experiments.collect_gru_teacher_dataset \
  --teacher-model-path envs/letter_env/results_and_evaluation/experiments_with_variable_n/ddqn/numerical_n_1to5_seed0/best_model.zip \
  --output-dir envs/letter_env/results_and_evaluation/encoder_pretraining/gru_teacher_dataset_n1to5_seed0

python -m envs.letter_env.experiments.train_gru_encoder \
  --dataset-path envs/letter_env/results_and_evaluation/encoder_pretraining/gru_teacher_dataset_n1to5_seed0/dataset.jsonl \
  --output-dir envs/letter_env/results_and_evaluation/encoder_pretraining/gru_dim16_seed0 \
  --seed 0
```

The learned graph encoder is also trained in two stages: first generate a
parameterized monitor-transition corpus, then train the basic graph dynamics
encoder.

```bash
python -m envs.letter_env.experiments.generate_gnn_monitor_corpus \
  --output-dir envs/letter_env/results_and_evaluation/encoder_pretraining/gnn_parameterized_corpus_n1to5_seed0 \
  --max-count 5

python -m envs.letter_env.experiments.train_gnn_encoder \
  --dataset-path envs/letter_env/results_and_evaluation/encoder_pretraining/gnn_parameterized_corpus_n1to5_seed0/monitor_states.jsonl \
  --output-dir envs/letter_env/results_and_evaluation/encoder_pretraining/gnn_basic_seed0 \
  --epochs 80 \
  --batch-size 128 \
  --seed 0
```

The generated datasets, full training logs, and noncanonical checkpoints are
not required in git. The tracked checkpoint files are enough to rerun the
learned-encoding RL experiments without repeating pretraining.

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

Example Double DQN run with a learned GRU monitor encoder:

```bash
python -m envs.letter_env.experiments.train_dqn \
  --algorithm ddqn \
  --encoding learned_gru \
  --learned-gru-checkpoint envs/letter_env/results_and_evaluation/encoder_pretraining/gru_dim16_seed0/best_student.pt \
  --n-value 5 \
  --seed 0 \
  --output-dir envs/letter_env/results_and_evaluation/experiments_with_variable_n/ddqn/learned_gru_n_1to5_seed0 \
  --total-timesteps 500000
```

Example Double DQN run with a learned graph monitor encoder:

```bash
python -m envs.letter_env.experiments.train_dqn \
  --algorithm ddqn \
  --encoding learned_graph \
  --learned-graph-checkpoint envs/letter_env/results_and_evaluation/encoder_pretraining/gnn_basic_seed0/best_dynamics_encoder.pt \
  --n-value 5 \
  --seed 0 \
  --output-dir envs/letter_env/results_and_evaluation/experiments_with_variable_n/ddqn/learned_graph_n_1to5_seed0 \
  --total-timesteps 500000
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

## Reproduction Scripts

Selected experiment batches can be regenerated from the repository root using
the scripts in `reproduction/`:

```bash
bash envs/letter_env/reproduction/run_ddqn_encodings.sh
bash envs/letter_env/reproduction/run_dqn_baselines.sh
bash envs/letter_env/reproduction/run_ppo_baselines.sh
bash envs/letter_env/reproduction/run_zero_shot.sh
bash envs/letter_env/reproduction/run_figures.sh
```

The full selected LetterEnv batch can be run with:

```bash
bash envs/letter_env/reproduction/run_all_selected.sh
```

The scripts write outputs to `results_and_evaluation/` using the same folder
layout documented above. They do not delete existing results before running.
For learned encodings, the canonical GRU and GNN checkpoints under
`encoder_pretraining/` are used directly.

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

For learned encodings, pass the same encoder checkpoint used during training:

```bash
python -m envs.letter_env.experiments.evaluate_zero_shot \
  --algorithm ddqn \
  --encoding learned_graph \
  --train-seed 0 \
  --eval-n 20 \
  --model-path envs/letter_env/results_and_evaluation/experiments_with_variable_n/ddqn/learned_graph_n_1to5_seed0/best_model.zip \
  --learned-graph-checkpoint envs/letter_env/results_and_evaluation/encoder_pretraining/gnn_basic_seed0/best_dynamics_encoder.pt \
  --output-dir envs/letter_env/results_and_evaluation/generalization_experiments_with_zero_shot_on_larger_n/ddqn/learned_graph_zeroshot_n20_seed0 \
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
python -m envs.letter_env.experiments.train_gru_encoder --help
python -m envs.letter_env.experiments.train_gnn_encoder --help
python -m envs.letter_env.experiments.evaluate_zero_shot --help
```

Environment-specific tests will be added under a dedicated test directory once
the final experiment set is fixed.
