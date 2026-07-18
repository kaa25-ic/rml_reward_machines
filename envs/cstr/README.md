# CSTR RML Experiment

This package contains an independent CSTR process-control experiment for
testing Runtime Monitoring Language reward-machine specifications in a
continuous safety-critical control task.

The native environment is responsible only for process dynamics, scalar
rewards, and instantaneous physical predicates. Temporal task logic such as
startup, recovery deadlines, warning limits, and stabilized production is
handled by generated RML specifications.

## First Version

The native environment models Jeff Kantor's exothermic CSTR with
coolant-temperature control:

- observation: centered/scaled concentration error, centered/scaled
  temperature error, centered/scaled concentration setpoint, previous
  normalized action;
- action: one continuous coolant-temperature control in `[-1, 1]`;
- termination: critical temperature violation or time limit;
- default operating point: `Ca = 0.5`, `T = 350 K`, `Tc = 300 K`;
- optional disturbances and randomized setpoints are present but disabled by
  default for the first experiment.

The generated RML monitor owns the temporal task:

- approach the unstable target tube around the concentration setpoint;
- once regulated, remain in the target tube for the rest of the episode;
- tolerate a small consecutive out-of-band budget after regulation starts;
- succeed only by reaching the episode limit while regulated;
- fail after critical temperature, unsafe temperature, terminal instability, or
  exhausting the post-regulation violation budget.

Python emits only instantaneous physical predicates such as
`event_temp_safe`, `event_stable_step`, and `event_temp_critical`. The
RML wrapper uses the external monitor response to build a fixed canonical
semantic progress observation: initial, approach, regulate, violation-count states,
success, and failure. The older hand-coded CSTR semantic encoder has been
removed, and train/eval environments share the same state ID semantics.

PPO uses a smaller initial Gaussian action standard deviation
(`log_std_init=-1.5`) so early rollouts do not immediately apply huge coolant
temperature swings. During training, RML failure is logged and penalized but
does not terminate the physical episode by default; evaluation remains strict.

## Tests

The CSTR test suite covers monitor phase decoding, soak counting, semantic
progress encodings, generated RML specs, native threshold predicates,
deterministic resets, and the pure-Python reference automaton/manual-RM
contracts.

```bash
python3 -m pytest tests/cstr
```

These tests are intentionally fast and do not require SWI-Prolog or PPO
training.

## PPO

Full comparison:

```bash
TOTAL_TIMESTEPS=500000 SEED=1 bash envs/cstr/reproduction/run_cstr_ppo.sh
```

Generalization soak-length evaluation:

```bash
SEED=0 bash envs/cstr/reproduction/run_generalization.sh
```

This loads the seed-0 PPO checkpoints trained with `soak_steps=10` and
evaluates the RML variants with `soak_steps=15`.

Figure generation:

```bash
bash envs/cstr/reproduction/run_figures.sh
```

This regenerates the submission figures from completed PPO and generalization
runs:

- `all_variants_best_metrics_multiseed`
- `baseline_vs_rml_graph_trajectory_seed0`
- `rml_graph_phase_trajectory_seed0`
- `all_variants_trajectory_comparison_seed0_rml_hidden_seed2_manual_rm_seed1`
- `generalization_success_by_soak_steps_seed0`

Figures are written to `envs/cstr/results_and_evaluation/figures`. Source CSV
tables are written under `envs/cstr/results_and_evaluation/figures/csv`, and
per-seed trajectory figures are written under
`envs/cstr/results_and_evaluation/figures/cstr_trajectories`.

The script defaults to seeds `0 1 2 3 4`, RML hidden trajectory seed `2`, and
manual-RM trajectory seed `1`. These can be overridden, for example:

```bash
SEEDS="0 1 2 3 4" SEED0=0 bash envs/cstr/reproduction/run_figures.sh
```

## Encoder Pretraining

The graph encoder corpus and the clean four-epoch graph encoder can be
regenerated with:

```bash
bash envs/cstr/reproduction/run_encoder_pretraining.sh
```

This writes the corpus to
`envs/cstr/results_and_evaluation/encoder_pretraining/gnn_corpus_seed0` and
the trained encoder to
`envs/cstr/results_and_evaluation/encoder_pretraining/gnn_dynamics_phase_count_epoch4_seed0`.
