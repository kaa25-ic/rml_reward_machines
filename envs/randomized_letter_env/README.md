# Randomized LetterEnv

Randomized LetterEnv evaluates the RML task `A, B, C, D^n` when the letter
locations vary across episodes. The Python environment controls grid movement
and letter placement; task progress, success, and failure are defined by the
RML monitor.

The agent starts from a fixed location. `A`, `C`, and `D` are placed at reset,
and `B` appears at the same location as `A` after `A` has been observed. The
observation contains normalized agent coordinates, proposition features for the
current cell, and normalized target coordinates for `A`, `B`, `C`, and `D`.

## Task Variants

Two randomized variants are supported.

### Fully Randomized, Fixed `n = 1`

In the fully randomized variant, `A`, `C`, and `D` are sampled from the full
grid, excluding the agent start location. This is the harder setting because
incorrect letters can appear near the start or along short paths to the current
target.

Use:

```bash
--placement-mode full_random --fixed-n 1
```

Results are stored under:

```text
results_and_evaluation/ddqn/full_random_n1/
```

### Regional Randomization, `n = 1..5`

In the regional variant, the letters are still randomized, but each one is
sampled from a fixed non-overlapping region:

```text
A: upper-left
C: upper-right
D: lower-left
```

This keeps the locations variable while reducing unnecessary search difficulty.
Episodes sample `n` uniformly from `1` to `5`.

Use:

```bash
--placement-mode regional --n-value 5 --sample-n
```

Results are stored under:

```text
results_and_evaluation/ddqn/regional_randomness_n1to5/
results_and_evaluation/q_learning/
```

## Encoding

The current experiments use `semantic_progress`, a compact monitor-state
encoding for the phases of the shared `A, B, C, D^n` task. The encoding is
computed from the RML monitor-state string, not from a Python task tracker.

## Algorithms

The neural experiments use Double DQN. The tabular baseline uses dictionary-
backed Q-learning with the same semantic-progress monitor information.

## Reproducing Runs

Reproduction scripts are provided in `reproduction/`. They write outputs to the
same layout used by the tracked experiment artifacts:

```text
results_and_evaluation/
  ddqn/
    full_random_n1/
    regional_randomness_n1to5/
  q_learning/
  generalization/
  figures/
```

The main training scripts run the selected experiments for seeds `0..4`.

DDQN, fully randomized with fixed `n = 1`:

```bash
bash envs/randomized_letter_env/reproduction/run_ddqn_full_random_n1.sh
```

DDQN, regional randomization with `n = 1..5`:

```bash
bash envs/randomized_letter_env/reproduction/run_ddqn_regional_n1to5.sh
```

Q-learning, regional randomization with `n = 1..5`:

```bash
bash envs/randomized_letter_env/reproduction/run_q_learning_regional_n1to5.sh
```

Run seed-0 zero-shot evaluations for the regional DDQN policy at `n=10`,
`n=15`, and `n=20`:

```bash
bash envs/randomized_letter_env/reproduction/run_ddqn_regional_zero_shot.sh
```

Generate report figures from the saved CSV summaries:

```bash
bash envs/randomized_letter_env/reproduction/run_figures.sh
```

All selected randomized LetterEnv experiments:

```bash
bash envs/randomized_letter_env/reproduction/run_all_selected.sh
```

The full selected pipeline reruns the five-seed DDQN experiments, the five-seed
Q-learning baseline, seed-0 zero-shot evaluations, and figure generation. It
does not delete existing result folders before running.

The monitor reward configuration defaults to:

```text
configs/random_letter_en.yaml
```

The figure script can also be called directly:

```bash
python -m envs.randomized_letter_env.analysis.generate_figures --formats pdf png
```
