# RML Reward Machines

This repository is the clean, submission-ready version of the RML reward
machine experiments. It is being rebuilt from the original research workspace
in small reviewed stages so that the final code is independent, reproducible,
and easy to back up.

The original workspace is used only as source material. It should not be edited
as part of this migration.

## Current Status

This repository currently contains only the top-level project metadata:

- `README.md`
- `pyproject.toml`
- `requirements.txt`
- `.gitignore`

Code, environment packages, experiment scripts, tests, and curated results will
be migrated in later reviewed stages.

## Planned Repository Layout

```text
rml_reward_machines/
  README.md
  pyproject.toml
  requirements.txt
  .gitignore

  rml_rm/
    monitors/
    encodings/
    wrappers/
    agents/
    utils/

  envs/
    letterenv/
    officeworld/
    multitask_letterenv/
    lunarlander/
    cstr/
    safety_point_button/
    safety_point_goal/

  scripts/
  tests/
  results/
  legacy/
```

## Environment Setup

Create and activate a fresh Python environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The Python dependencies are pinned in `requirements.txt` to match the research
workspace as closely as possible.

## SWI-Prolog

Some RML monitor workflows require SWI-Prolog. The preferred setup for a clean
submission is to install SWI-Prolog system-wide so that `swipl` is available on
the command line.

On macOS, one option is Homebrew:

```bash
brew install swi-prolog
```

You can verify the installation with:

```bash
swipl --version
```

For local offline work on macOS, a copy of `SWI-Prolog.app` may also be placed
inside this repository, but it is intentionally ignored by git because it is a
large platform-specific application bundle. The final submitted repository
should document how to install SWI-Prolog rather than commit the app bundle.

## Results Policy

Generated experiment outputs, checkpoints, logs, and large result directories
are ignored by git. Final figures, summary tables, and small manifest files may
be committed later when they are curated and clearly tied to reproducible
commands.

## Migration Policy

Migration is done in small checkpoints:

1. Create submission metadata and dependency files.
2. Make the repository independent from the original workspace.
3. Add local SWI-Prolog support without tracking the app bundle.
4. Connect the repository to GitHub for regular backup.
5. Migrate shared core code.
6. Migrate environments and experiments one at a time.
7. Add tests, reproduction commands, and final documentation.
