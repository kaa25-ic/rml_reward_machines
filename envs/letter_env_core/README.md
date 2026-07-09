# LetterEnv Core

`letter_env_core` contains shared grid mechanics for the LetterEnv family of
environments. It is not an experiment environment by itself.

The package provides movement, boundary handling, proposition placement,
proposition replacement, rendering, and raw observation construction. Task logic
such as `A B C D^n`, multitask task sampling, randomized layouts, monitor
specifications, encodings, trainers, results, and analysis scripts stay in the
environment-specific packages.

Current users of this package:

- `envs.letter_env`
- `envs.multitask_letter_env`
- `envs.randomized_letter_env`
