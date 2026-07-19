# CSTR Startup Protocol

This environment evaluates an RML-monitored startup procedure for an exothermic
continuous stirred-tank reactor (CSTR). The task is based on the classic Jeff
Kantor CSTR process-control example: a nonlinear chemical reactor with an
exothermic reaction, coolant-temperature actuation, an unstable operating
region, and the risk of thermal runaway when the temperature is not controlled.

The control objective is to bring the reactor to the target concentration
`Ca = 0.5` while keeping the reactor temperature near `350 K`. This is
difficult because reaction heat generation increases rapidly with temperature:
the controller must heat the reactor enough to reach the productive operating
region, but avoid overshoot, unsafe temperatures, and concentration collapse.

The Python environment implements only the physical dynamics, scalar base
reward, and instantaneous predicates. The ordered startup procedure, counting
requirement, success condition, and failure condition are specified by the RML
monitor in `specs/cstr_startup_procedure.pl`.

## Environment Setup

The environment package contains:

- `env.py`: native CSTR dynamics and physical proposition extraction.
- `builder.py`: RML-monitored environment construction and CSTR-specific reward
  shaping.
- `encodings.py`: semantic-progress and frozen graph monitor-state encoders.
- `manual_rm.py`: manual reward-machine baseline for the startup procedure.
- `reference_automaton.py`: pure-Python reference automaton used to validate the
  generated RML monitor.
- `configs/cstr_startup_procedure.yaml`: monitor runtime configuration.
- `specs/cstr_startup_procedure.pl`: generated RML startup specification.
- `experiments/`: PPO, evaluation, generalization, corpus, and encoder
  pretraining entry points.
- `analysis/`: figure-generation and policy-visualization scripts.
- `reproduction/`: command scripts for the selected runs.

The native observation contains four normalized values: 

- concentration error `(Ca - Ca_setpoint) / 0.15`;
- temperature error `(T - 350) / 25`;
- setpoint offset `(Ca_setpoint - 0.5) / 0.25`;
- previous coolant-temperature offset `(Tc - 300) / 50`.

The action is one continuous coolant-temperature command in `[-1, 1]`, scaled
to a physical coolant range of `250 K` to `350 K`. The default episode length is
`300` control steps. Each control step integrates the reactor dynamics for `50`
internal Euler steps of size `0.001`.

Default initial and target values:

| Quantity | Value |
| --- | --- |
| initial concentration `Ca` | `0.80` |
| initial reactor temperature `T` | `331 K` |
| target concentration `Ca_setpoint` | `0.50` |
| target production temperature | `350 K` |
| initial coolant temperature `Tc` | `302.5 K` |
| nominal coolant temperature | `300 K` |

Optional initial-state randomization, setpoint randomization, and feed
disturbances are implemented in `CSTRConfig` but are disabled for the selected
startup-procedure experiments.

## Startup Procedure

The RML monitor specifies a temporally extended startup protocol rather than a
single steady-state objective. The required phase order is:

1. **Preheat**: remain temperature-safe while driving the reactor toward the
   startup soak band.
2. **Soak**: enter and remain in the soak band for `10` consecutive monitor
   steps.
3. **Approach**: after the soak count is complete, move toward the production
   target without violating safety or overshooting concentration.
4. **Regulate**: enter the stable production region within 100 steps (deadline), and remain regulated until
   the episode terminates.
5. **Success**: the episode terminates while the reactor is regulated. 

The selected reproduction runs use the following instantaneous physical
predicates. These values are set by `run_cstr_ppo.sh` and are also recorded in
the saved PPO `config.json` files:

| Event | Selected condition |
| --- | --- |
| `temp_safe` | `315 K <= T <= 375 K` |
| `critical` | `T >= 405 K` or `T <= 280 K` |
| `in_soak_band` | `343 K <= T <= 347 K` and `0.58 <= Ca <= 0.74` |
| `stable` | `abs(Ca - 0.5) <= 0.08`, `346 K <= T <= 354 K`, and `temp_safe` |
| `overshoot` | `Ca < 0.44` |
| `past_deadline` | exactly step `100` |
| `heating_rate_exceeded` | one-step temperature rise `> 1 K`|

Failure occurs if the reactor becomes critical or unsafe, reaches terminal time
before regulation, misses the startup deadline before regulation, overshoots
concentration, or tries to stabilize before completing the soak count. If the
reactor leaves the soak band before completing the count, the monitor returns to
`Preheat` and the soak count resets. Once in `Regulate`, the default selected
setup is strict: stable operation and temperature-band dwell are allowed, but
falling back to merely safe non-soak operation is a protocol failure.

The RML monitor state is decoded into canonical phases:

```text
<initial>, preheat, soak_1, ..., soak_10, approach, regulate, success, failure
```

These phases are used by the semantic-progress encoder, reward-shaping
components, evaluation metrics, and trajectory figures.

## Encodings and Baselines

The PPO experiments compare five variants:

- `baseline`: native CSTR reward without an observed RML monitor state.
- `manual_rm_semantic_progress`: hand-designed manual reward-machine baseline using 
  semantic-progress encoder.
- `rml_semantic_progress`: RML monitor with semantic-progress encoder.
- `rml_hidden`: RML rewards with hidden monitor progress.
- `rml_graph_encoder`: RML monitor with a frozen GNN graph encoder of the raw RML
  monitor state.

The graph encoder is trained from a CSTR-specific RML monitor corpus using the
shared graph-dynamics pretraining machinery in `rml_rm.encodings`.

## Metrics

Evaluation tracks both standard RL quantities and procedure-specific reactor
metrics:

- RML success rate.
- mean return.
- mean tracking error `abs(Ca - Ca_setpoint)`;
- first stable step within the concentration band.
- soak completion and regulation behavior.
- critical or unsafe temperature failures.

## Tests

The CSTR test suite can be run with:

```bash
python3 -m pytest tests/cstr
```

Run the shared core tests together with the CSTR tests before changing
monitor wrappers, monitor-state normalization, or graph encoders:

```bash
./.venv/bin/python3 -m pytest tests/core tests/cstr
```

These tests are intentionally fast and do not require SWI-Prolog or PPO
training.

## Reproduction

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

This regenerates figures from completed PPO and generalization
runs:

- `all_variants_best_metrics_multiseed`
- `baseline_vs_rml_graph_trajectory_seed0`
- `rml_graph_phase_trajectory_seed0`
- `all_variants_trajectory_comparison_seed0_rml_hidden_seed2_manual_rm_seed1`
- `generalization_success_by_soak_steps_seed0`

## Graph Encoder Pretraining

The graph encoder corpus and graph encoder training can be
regenerated with:

```bash
bash envs/cstr/reproduction/run_encoder_pretraining.sh
```

