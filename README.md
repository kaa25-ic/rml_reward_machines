# RML Reward Machines

This repository contains reinforcement learning experiments that use Runtime
Monitoring Language (RML) specifications as reward machines. The main objective
is to study how explicit temporal-task monitors can improve learning and
generalization using deep reinforcement learning agents, and how different monitor-state encodings affect policy
performance.

Shared RML monitoring, encoding, runtime, and reinforcement-learning utilities
live in `rml_rm/`. Each environment keeps its own dynamics, RML specifications,
experiments, analysis, results, and reproduction scripts under `envs/`.

## Project Scope

The experiments cover:

- single-task LetterEnv counting tasks. 
- multitask LetterEnv counting and zero-shot generalization.
- randomized LetterEnv placement experiments.
- an RML-monitored LunarLander landing protocol.
- a continuous-control CSTR startup and regulation task.

Across these environments, the repository compares tabular baselines, DQN,
Double DQN, PPO, manual reward-machine baselines, semantic progress encodings,
hidden-monitor ablations, GRU monitor encoders, and graph monitor encoders.

## External Requirements

Install these system-level programs before running monitored experiments:

- Python `>=3.10,<3.12`; Python 3.11 is recommended.
- SWI-Prolog, available as `swipl` on the command line.
- A POSIX shell with `bash`.
- `swig`, only if your platform needs to build the Box2D Python package from
  source.

On macOS with Homebrew:

```bash
brew install python@3.11 swi-prolog swig
```

The Prolog monitor runtime used by the experiments is vendored under
`rml_rm/monitors/rml`. You do not need to copy monitor scripts into a separate
RML checkout. Python experiment scripts start and stop local monitor processes
through the shared runtime helpers.

## Python Setup

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Verify the Setup

Run the dependency and unit-test checks:

```bash
python -m pip check
./.venv/bin/python3 -m pytest tests/core
```

Run quick repository tests:

```bash
./.venv/bin/python3 -m pytest tests
```

Run the integration monitor-process test when `swipl` is installed:

```bash
./.venv/bin/python3 -m pytest tests/integration
```

Environment-specific README files provide narrower commands. Common examples:

```bash
./.venv/bin/python3 -m pytest tests/core tests/letter_env_core tests/letter_env
./.venv/bin/python3 -m pytest tests/core tests/letter_env_core tests/multitask_letter_env
./.venv/bin/python3 -m pytest tests/core tests/letter_env_core tests/randomized_letter_env
./.venv/bin/python3 -m pytest tests/core tests/lunar_lander
./.venv/bin/python3 -m pytest tests/core tests/cstr
```

These tests check monitor-state
normalization, monitor transactions, wrappers, environment contracts, generated
specifications, encodings, reward-shaping logic, and selected deterministic
runtime behavior.

## Repository Layout

```text
rml_reward_machines/
  rml_rm/
    agents/          # shared tabular, DQN, DDQN, PPO, feature, and policy code
    encodings/       # vector, semantic, GRU, graph, and frozen monitor encoders
    experiments/     # shared runtime, seeding, monitor lifecycle, and JSON helpers
    monitors/        # WebSocket client, monitor process manager, vendored RML scripts
    wrappers/        # Gymnasium wrappers for monitor observations and rewards
  envs/
    letter_env/              # single-task A B C D^n experiments
    letter_env_core/         # shared grid mechanics for LetterEnv variants
    multitask_letter_env/    # task-family counting experiments
    randomized_letter_env/   # randomized letter-placement experiments
    lunar_lander/            # RML-monitored LunarLander protocol
    cstr/                    # continuous stirred-tank reactor startup task
  tests/
    core/                    # shared monitor, wrapper, and normalization tests
    integration/             # tests that require the external monitor process
    letter_env*/
    multitask_letter_env/
    randomized_letter_env/
    lunar_lander/
    cstr/
  requirements.txt
  pyproject.toml
```

Each environment package follows the same general pattern:

```text
env.py              # native dynamics or task definition
builder.py          # monitored environment construction
encodings.py        # environment-specific monitor-state encodings
specs/              # RML Prolog specifications
configs/            # monitor YAML configuration templates
experiments/        # train/evaluate/pretrain entry points
analysis/           # figure and table generation
reproduction/       # command scripts for selected runs
results_and_evaluation/
```

## Running Experiments

Use the environment README files as the authoritative instructions for each
experiment group:

- `envs/letter_env/README.md`
- `envs/multitask_letter_env/README.md`
- `envs/randomized_letter_env/README.md`
- `envs/lunar_lander/README.md`
- `envs/cstr/README.md`

Selected reproduction scripts are available under each environment's
`reproduction/` directory. 

Most scripts expose seed and output-directory variables through environment
variables. Check the corresponding script or README before launching long
training jobs.

## Results and Checkpoints

Experiment outputs are written under each environment's
`results_and_evaluation/` directory. These folders may contain:

- run configurations.
- monitor runtime configs and logs.
- training and evaluation CSV files.
- model checkpoints.
- encoder pretraining corpora and checkpoints.
- compact summaries and generated figures.

Retained checkpoints, compact summaries, and generated figures are kept next to
the relevant environment and documented by that environment's README. 

## Notes on RML Monitoring

The Python side derives instantaneous propositions from the environment state
or info dictionary. RML monitors consume those propositions over a WebSocket
connection and return:

- a verdict, such as `currently_false`, `true`, or `false`;
- the raw RML monitor state;
- the monitor reward associated with the verdict/configuration.

Shared wrappers and runtime helpers manage payload construction, monitor-state
normalization, local port allocation, monitor startup/shutdown, and train/eval
monitor separation. Environment-specific code is responsible for domain
propositions, task-specific shaping, and analysis metrics.

## License & Attribution

This project is released under the MIT License (see [LICENSE](LICENSE)).
Copyright (c) 2026 Khalid Alahmadi.

It builds on prior work, acknowledged below.

- **RML runtime-monitoring toolkit** — the Prolog monitor files under
  `rml_rm/monitors/rml/` (`trace_expressions_semantics.pl`, `deep_subdict.pl`,
  `online_monitor_edit.sh`, `online_monitor_edit_fast.pl`) are vendored from the
  RML project. MIT License, © 2019–2022 Davide Ancona, Luca Franceschini,
  Angelo Ferrando, and Viviana Mascardi (RML@DIBRIS, University of Genoa). Their
  MIT notices are retained in the file headers.

- **rml_reward_machines** — Daniel Donnelly
  (https://github.com/danieldonnelly7/rml_reward_machines), MIT License. The
  LetterEnv and OfficeWorld environments and the reward-machine experiment
  design in this repository derive from this project.

- **RMLGym** — H. Unniyankal
  (https://github.com/hishamunniyankal/rml-gym). The monitor-wrapper interface
  in this repository follows the same general RMLGym integration pattern:
  environment propositions are sent to an external RML monitor, and the returned
  verdict and monitor state are incorporated into the Gymnasium environment.

The monitor-state encodings (semantic-progress, GRU sequential, and GNN
graph), the reinforcement-learning training and evaluation, and all environment
implementations beyond those noted above are original to this work.
