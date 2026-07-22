# LunarLander Protocol

This environment evaluates an RML-monitored LunarLander landing protocol. The
agent observes the standard LunarLander state together with an encoded RML
monitor state. The monitor tracks whether the lander enters the descent
corridor, hovers for the required duration, performs controlled descent, and
lands safely inside the target zone.

The Python environment only derives low-level propositions from the simulator
state. Protocol progress, success, and failure are handled by the RML monitor in
`specs/lunar_lander_protocol.pl`.

## Landing Protocol

The protocol is a temporal specification over propositions extracted from the
Gymnasium LunarLander state. The required order is:

1. Enter the descent corridor.
2. Hold a controlled hover for the required count.
3. Confirm hover completion.
4. Enter controlled descent.
5. Land safely in the target zone.

The monitor starts in a waiting-for-corridor phase. A corridor match occurs when
the lander is horizontally close to the pad and within the configured vertical
corridor. After this match, the monitor enters the hover phase and counts hover
events. A hover event requires the lander to remain inside the hover height
band, keep vertical speed below the hover threshold, and keep the body angle
within the hover-angle threshold.

After three counted hover events, the monitor switches to controlled descent. Controlled descent requires the lander
to be below the hover band while keeping descent speed and angle within their
configured limits.

The protocol succeeds only when the simulator terminates with a successful
landing while all landing predicates are true: both legs are in contact, the
lander is inside the horizontal target zone, the landing angle is safe, and the
base LunarLander environment reports a successful landing. If the episode ends
before the required protocol sequence is complete, the RML monitor rejects the
trace.

The default proposition thresholds are defined by
`LunarProtocolThresholds` in `env.py`. They use the standard LunarLander state
components `x`, `y`, vertical velocity `vy`, body `angle`, and the two leg
contact indicators:

| Predicate | Default condition |
| --- | --- |
| `corridor` | horizontal corridor `abs(x) <= 0.7` and vertical corridor `0.7 <= y <= 1.4` |
| `hover` | hover band `0.6 <= y <= 1.0`, vertical-speed limit `abs(vy) <= 0.25`, angle limit `abs(angle) <= 0.35` |
| `controlled_descent` | below hover band `y < 0.6`, descent-speed limit `abs(vy) <= 0.6`, descent-angle limit `abs(angle) <= 0.45` |
| `target_zone` | landing-zone limit `abs(x) <= 0.25` |
| `safe_landing_angle` | terminal landing-angle limit `abs(angle) <= 0.30` |
| `both_contact` | left and right leg-contact indicators are both greater than `0.5` |
| `env_successful_landing` | the simulator terminates and the base LunarLander reward is positive |

The RML monitor therefore distinguishes an ordinary successful simulator
landing from a protocol-compliant landing: the final step must satisfy the
terminal simulator success condition and the protocol-specific target-zone,
angle, and contact predicates.

## Testing

From the repository root, run the shared core tests and LunarLander protocol
tests with:

```bash
./.venv/bin/python3 -m pytest tests/core tests/lunar_lander
```

## Two-Stage PPO

The selected experiment uses a two-stage PPO procedure. Stage 1 is a discovery
stage with a higher learning rate. Stage 2
stabilizes the best stage-1 checkpoint with a lower learning rate. This keeps
the final policy stable while preserving the strictly RML-based protocol
monitoring.

`experiments/train_ppo_two_stage.py` is a separate orchestration script for the
discovery-then-stabilization PPO experiment. Stage 1 trains with the discovery
learning rate and saves its normal artifacts under `stage1_discovery`. Stage 2
then automatically loads `stage1_discovery/best_model.zip` and fine-tunes it
with the lower stabilization learning rate under `stage2_stabilization`.

By default, all two-stage runs are saved under:

```text
envs/lunar_lander/results_and_evaluation/ppo/two_stage_training/
```

## Rendering Policies

`experiments/render_policy.py` renders trained PPO checkpoints through the same
RML monitor stack used during training. It can render one run directory or batch
render all runs under a two-stage result root. Each render writes an episode
summary, a step-by-step trajectory CSV, and a root-level render index. Use
`--record-gif` for visual artifacts in the current project
environment, and `--record-video` is also supported when MP4 video dependencies are
installed.

Render the final stage-2 policy for all two-stage seeds:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig PYTHONPYCACHEPREFIX=/private/tmp/rml_pycache ./.venv/bin/python3 envs/lunar_lander/experiments/render_policy.py \
  --runs-root envs/lunar_lander/results_and_evaluation/ppo/two_stage_training \
  --stage stage2_stabilization \
  --model model_final \
  --record-gif \
  --seed 10000
```

## Figures

Figures are generated from the saved two-stage evaluation CSVs and the
rendered stage-2 trajectories:

- landing vs RML protocol learning curves over successful seeds;
- the landing/protocol success gap;
- seed-0 landing vs protocol learning curve;
- a phase-colored successful landing trajectory.

Generate figures with:

```bash
./.venv/bin/python3 -m envs.lunar_lander.analysis.generate_figures --formats pdf png
```

## Reproducing Results

Reproduction scripts are provided in `reproduction/`. 

Run the selected two-stage PPO experiments for seeds `0..4`:

```bash
bash envs/lunar_lander/reproduction/run_two_stage_ppo.sh
```

Render the final stage-2 policies:

```bash
bash envs/lunar_lander/reproduction/run_render_stage2.sh
```

Generate report figures:

```bash
bash envs/lunar_lander/reproduction/run_figures.sh
```

The selected LunarLander pipeline can be run with:

```bash
bash envs/lunar_lander/reproduction/run_all_selected.sh
```

The full pipeline reruns the five-seed two-stage PPO experiments, renders the
stage-2 final policies, and regenerates the figure artifacts. It does not delete
existing result folders before running. Use `SEEDS="0 1"` with
`run_two_stage_ppo.sh` for a smaller partial rerun.
