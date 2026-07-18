# LetterEnv

LetterEnv is a sequential reinforcement learning environment for evaluating
how reward-machine monitor states can be represented for learning. Each episode
samples or fixes a target sequence length `n`. The agent must complete the
letter sequence while an RML monitor tracks progress through the task and
provides the monitor state used by the observation encodings.

The task is to visit letters in the order `A`, `B`, `C`, then `D` repeated
`n` times. In compact form, the target sequence is `A B C D^n`. Episodes use
variable `n` in the main training setting, with `n` sampled from 1 to 5 unless a
fixed value is specified.

This environment is used to compare tabular learning, DQN, Double DQN, and PPO
under monitor-state encodings that expose the same reward-machine state in
different forms.

## Environment Setup

The environment package contains:

- `env.py`: the single-task LetterEnv task definition for `A B C D^n`.
- `builder.py`: environment construction helpers used by training scripts.
- `encodings.py`: LetterEnv observation and monitor-state encoding helpers.
- `configs/letter_env.yaml`: the monitor runtime configuration template.
- `configs/monitor_state_catalogue.json`: the monitor-state catalogue used by
  one-hot, numerical, and semantic progress encodings.
- `specs/letter_env_monitor.pl`: the RML monitor specification.
- `experiments/`: command-line entry points for training and evaluation.

Shared grid mechanics are provided by `envs.letter_env_core`, including
movement, boundary handling, proposition placement, proposition replacement,
rendering, and raw observation construction. This keeps the completed
single-task environment stable while allowing multitask and randomized
LetterEnv variants to reuse the same grid implementation.

RML monitoring requires SWI-Prolog. Install the project and verify SWI-Prolog
from the repository root:

```bash
source .venv/bin/activate
python -m pip install -e .
swipl --version
```

## Encodings

The neural experiments use six monitor-state encodings:

- `one_hot`: a one-hot vector derived directly from the RML monitor state.
- `numerical`: a compact numerical representation derived from the RML monitor
  state.
- `semantic_progress`: a task-phase one-hot vector derived from the RML monitor
  state.
- `learned_gru`: a frozen 16-dimensional recurrent encoder trained from a
  teacher dataset collected from a trained LetterEnv policy.
- `learned_graph`: a frozen 32-dimensional graph encoder trained from
  parameterized RML monitor-transition data.
- `hidden_monitor_state`: a constant monitor-state vector. The RML monitor and
  its rewards remain active, but the policy does not observe monitor progress.

The tabular reproduction also includes `simple`, which is used for comparison
with the baseline tabular state abstraction for this environment.

Learned encoder checkpoints are stored under
`results_and_evaluation/encoder_pretraining/`. The experiments use
`gru_dim16_seed0/best_student.pt` for GRU encodings and
`gnn_basic_seed0/best_dynamics_encoder.pt` for graph encodings.

## Testing

From the repository root, run the shared core tests, shared grid tests, and
single-task LetterEnv tests with:

```bash
./.venv/bin/python3 -m pytest tests/core tests/letter_env_core tests/letter_env
```

## Experiment Groups

Generated outputs are written under:

```text
envs/letter_env/results_and_evaluation/
```

The experiment layout is:

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

Each neural run writes its own folder containing the run configuration, model
checkpoint, monitor logs, training monitor CSVs, evaluation metrics, and summary
JSON. Tabular reproduction runs write episode-level metrics and a summary JSON.

## Learned Encoder Pretraining

The learned GRU encoder is trained in two stages: first collect a teacher
dataset, then distill a frozen monitor encoder from that dataset. Below are
example commands to run the two stages:

```bash
python -m envs.letter_env.experiments.collect_gru_teacher_dataset \
  --teacher-model-path envs/letter_env/results_and_evaluation/experiments_with_variable_n/ddqn/numerical_n_1to5_seed0/best_model.zip \
  --output-dir envs/letter_env/results_and_evaluation/encoder_pretraining/gru_teacher_dataset_n1to5_seed0

python -m envs.letter_env.experiments.train_gru_encoder \
  --dataset-path envs/letter_env/results_and_evaluation/encoder_pretraining/gru_teacher_dataset_n1to5_seed0/dataset.jsonl \
  --output-dir envs/letter_env/results_and_evaluation/encoder_pretraining/gru_dim16_seed0 \
  --seed 0
```

The learned GNN graph encoder is also trained in two stages: first generate a
parameterized monitor-transition corpus, then train the basic graph dynamics
encoder. Below are example commands to run the two stages:

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

The generated datasets and training logs document encoder pretraining. The GRU
and GNN checkpoints listed above are provided and sufficient to rerun the learned-encoding
RL experiments without repeating pretraining.

## Figures

Figures are generated from the saved learning and zero-shot CSV summaries:

- learning curves comparing DQN, Double DQN, and PPO with the numerical
  monitor encoding.
- learning curves comparing monitor-state encodings.
- sample-efficiency bars using first success at `SR >= 0.9`.
- zero-shot success rate at held-out sequence lengths.
- zero-shot episode length at held-out sequence lengths.

Generate figures with:

```bash
python -m envs.letter_env.analysis.generate_figures \
  --formats pdf png \
  --success-threshold 0.9 \
  --max-learning-steps 250000
```

The reproduction wrapper runs the same command:

```bash
bash envs/letter_env/reproduction/run_figures.sh
```

Figures and source CSV summaries are written to:

```text
envs/letter_env/results_and_evaluation/figures/
```

## Reproduction Scripts

Experiment batches can be regenerated from the repository root using the
scripts in `reproduction/`:

```bash
bash envs/letter_env/reproduction/run_ddqn_encodings.sh
bash envs/letter_env/reproduction/run_dqn_baselines.sh
bash envs/letter_env/reproduction/run_ppo_baselines.sh
bash envs/letter_env/reproduction/run_zero_shot.sh
bash envs/letter_env/reproduction/run_figures.sh
```

The full LetterEnv reproduction batch can be run with:

```bash
bash envs/letter_env/reproduction/run_all_selected.sh
```

The scripts write outputs to `results_and_evaluation/` using the folder layout
documented above. They do not delete existing results before running. Learned
encoding runs use the GRU and GNN checkpoints under `encoder_pretraining/`.
