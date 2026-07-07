# RML Reward Machines

This repository contains reinforcement learning experiments with reward
machines specified in RML. The project compares monitor-state encodings and
training protocols across several environments, including LetterEnv,
OfficeWorld, LunarLander, CSTR, and safety-oriented point environments.

The repository is organized to separate reusable implementation code from
environment-specific experiments and reproduction scripts.

## Repository Layout

```text
rml_reward_machines/
  rml_rm/       # shared monitor, encoding, wrapper, agent, and utility code
  envs/         # environment-specific packages and experiment entry points
  scripts/      # cross-environment reproduction and analysis scripts
  tests/        # smoke tests and regression tests
  results/      # generated outputs, ignored by git
  legacy/       # selected baseline code required for reproduction
```

The final repository tracks source code, configuration files, monitor
specifications, tests, and documentation. Generated results, checkpoints, logs,
and large local runtime bundles are excluded from version control.

## Python Setup

Use Python 3.11. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

If `python3.11` is not available, install Python 3.11 first. On macOS with
Homebrew:

```bash
brew install python@3.11
```

## SWI-Prolog

RML monitor execution requires SWI-Prolog. The recommended setup is to install
SWI-Prolog so that `swipl` is available on the command line:

```bash
swipl --version
```

On macOS with Homebrew:

```bash
brew install swi-prolog
```

For local macOS use, `SWI-Prolog.app` can be placed in the repository root or a
documented runtime directory. The app bundle is ignored by git because it is
large and platform-specific. Submitted code should remain portable and should
not depend on committing `SWI-Prolog.app`.

## Running Experiments

Each environment package will provide its own README with:

- environment description
- required monitor specifications
- training commands
- evaluation commands
- expected output locations

Experiment outputs should be written under each environment's
`results_and_evaluation/` directory. These directories are ignored by git.
Reproduction commands should be kept deterministic where possible and should
expose seed arguments for repeated runs.

## Results

This repository does not track raw experiment outputs. Results should be
regenerated from the submitted code. Final figures or compact summary tables may
be added only when they are clearly documented and small enough for normal
version control.

## Development Checks

After installing the environment, basic checks can be run with:

```bash
python -m pip check
python -m pytest
```

Additional environment-specific smoke tests will be documented as the
environment packages are added.
