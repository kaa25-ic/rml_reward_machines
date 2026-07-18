"""Generate RML monitor files for CSTR safety-control experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
GENERATED_ROOT = PACKAGE_ROOT / "generated"
SPECS_ROOT = PACKAGE_ROOT / "specs"
CONFIGS_ROOT = PACKAGE_ROOT / "configs"


@dataclass(frozen=True)
class GeneratedCSTRRML:
    """Generated RML spec/config paths for one CSTR task."""

    task_key: str
    spec_path: Path
    config_path: Path


def generate_cstr_rml(
    *,
    regulation_violation_steps: int = 10,
    soak_steps: int | None = None,
    recover_from_regulation_failure: bool = False,
    host: str = "127.0.0.1",
    port: int = 18_401,
    max_episode_steps: int = 300,
    generated_root: Path | None = None,
) -> GeneratedCSTRRML:
    """Generate a CSTR RML spec/config pair."""

    if soak_steps is None:
        soak_steps = regulation_violation_steps
    if soak_steps < 1:
        raise ValueError("soak_steps must be at least 1.")

    root = (Path(generated_root) if generated_root is not None else GENERATED_ROOT).resolve()
    specs_root = root / "specs"
    configs_root = root / "configs"
    specs_root.mkdir(parents=True, exist_ok=True)
    configs_root.mkdir(parents=True, exist_ok=True)

    task_key = "cstr_startup_procedure"
    spec_path = specs_root / f"{task_key}.pl"
    config_path = configs_root / f"{task_key}.yaml"
    spec_path.write_text(
        render_cstr_spec(
            soak_steps=soak_steps,
            recover_from_regulation_failure=recover_from_regulation_failure,
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        render_cstr_config(
            env_name=f"cstr-rml-{task_key}",
            host=host,
            port=port,
            max_episode_steps=max_episode_steps,
        ),
        encoding="utf-8",
    )
    return GeneratedCSTRRML(task_key=task_key, spec_path=spec_path, config_path=config_path)


def render_cstr_spec(*, soak_steps: int = 10, recover_from_regulation_failure: bool = False) -> str:
    """Render the ordered CSTR startup procedure as an RML trace expression."""

    soak_states = ",\n".join(
        _render_soak_state(index=index, soak_steps=soak_steps)
        for index in range(1, soak_steps + 1)
    )
    regulate_safe_expr = (
        "(safe:eps) * app(Approach, [])"
        if recover_from_regulation_failure
        else "(safe:0)"
    )
    approach_unsafe_expr = (
        "(unsafe:eps) * app(Approach, [])"
        if recover_from_regulation_failure
        else "(unsafe:0)"
    )
    approach_overshoot_expr = (
        "(overshoot:eps) * app(Approach, [])"
        if recover_from_regulation_failure
        else "(overshoot:0)"
    )
    regulate_unsafe_expr = (
        "(unsafe:eps) * app(Approach, [])"
        if recover_from_regulation_failure
        else "(unsafe:0)"
    )
    regulate_overshoot_expr = (
        "(overshoot:eps) * app(Approach, [])"
        if recover_from_regulation_failure
        else "(overshoot:0)"
    )
    return f""":- module('spec', [trace_expression/2, match/2]).
:- use_module(monitor('deep_subdict')).

match(_event, critical) :-
    deep_subdict(_event, _{{'critical':T}}), T=1.0.
match(_event, unsafe) :-
    deep_subdict(_event, _{{'temp_safe':T}}), T=0.0,
    not(match(_event, critical)).
match(_event, done_regulated) :-
    deep_subdict(_event, _{{'terminate':T}}), T=true,
    deep_subdict(_event, _{{'stable':S}}), S=1.0,
    not(match(_event, critical)),
    not(match(_event, unsafe)).
match(_event, done_unregulated) :-
    deep_subdict(_event, _{{'terminate':T}}), T=true,
    not(match(_event, critical)),
    not(match(_event, unsafe)),
    not(match(_event, done_regulated)).
match(_event, deadline) :-
    deep_subdict(_event, _{{'past_deadline':T}}), T=1.0,
    not(match(_event, critical)),
    not(match(_event, unsafe)),
    not(match(_event, done_regulated)),
    not(match(_event, done_unregulated)).
match(_event, overshoot) :-
    deep_subdict(_event, _{{'overshoot':T}}), T=1.0,
    not(match(_event, critical)),
    not(match(_event, unsafe)),
    not(match(_event, done_regulated)),
    not(match(_event, done_unregulated)),
    not(match(_event, deadline)).
match(_event, stable) :-
    deep_subdict(_event, _{{'stable':T}}), T=1.0,
    not(match(_event, critical)),
    not(match(_event, unsafe)),
    not(match(_event, done_regulated)),
    not(match(_event, done_unregulated)),
    not(match(_event, deadline)),
    not(match(_event, overshoot)).
match(_event, in_soak) :-
    deep_subdict(_event, _{{'in_soak_band':T}}), T=1.0,
    not(match(_event, critical)),
    not(match(_event, unsafe)),
    not(match(_event, done_regulated)),
    not(match(_event, done_unregulated)),
    not(match(_event, deadline)),
    not(match(_event, overshoot)),
    not(match(_event, stable)).
match(_event, safe) :-
    deep_subdict(_event, _{{'temp_safe':T}}), T=1.0,
    not(match(_event, critical)),
    not(match(_event, unsafe)),
    not(match(_event, done_regulated)),
    not(match(_event, done_unregulated)),
    not(match(_event, deadline)),
    not(match(_event, overshoot)),
    not(match(_event, stable)),
    not(match(_event, in_soak)).
match(_event, other) :-
    not(match(_event, critical)),
    not(match(_event, unsafe)),
    not(match(_event, done_regulated)),
    not(match(_event, done_unregulated)),
    not(match(_event, deadline)),
    not(match(_event, overshoot)),
    not(match(_event, stable)),
    not(match(_event, in_soak)),
    not(match(_event, safe)).
match(_, any).

trace_expression('Main', Main) :-
    Main = app(Preheat, []),
    Preheat = gen([], (
        (critical:0)
        \\/ (unsafe:0)
        \\/ (done_regulated:0)
        \\/ (done_unregulated:0)
        \\/ (deadline:0)
        \\/ (overshoot:0)
        \\/ (stable:0)
        \\/ (in_soak:eps) * app(Soak_1, [])
        \\/ (safe:eps) * app(Preheat, [])
        \\/ (other:0)
    )),
{soak_states},
    Approach = gen([], (
        (critical:0)
        \\/ {approach_unsafe_expr}
        \\/ (done_regulated:0)
        \\/ (done_unregulated:0)
        \\/ (deadline:0)
        \\/ {approach_overshoot_expr}
        \\/ (stable:eps) * app(Regulate, [])
        \\/ (in_soak:eps) * app(Approach, [])
        \\/ (safe:eps) * app(Approach, [])
        \\/ (other:0)
    )),
    Regulate = gen([], (
        (critical:0)
        \\/ {regulate_unsafe_expr}
        \\/ (done_regulated:1)
        \\/ (done_unregulated:0)
        \\/ (deadline:eps) * app(Regulate, [])
        \\/ {regulate_overshoot_expr}
        \\/ (stable:eps) * app(Regulate, [])
        \\/ (in_soak:eps) * app(Regulate, [])
        \\/ {regulate_safe_expr}
        \\/ (other:0)
    )).
"""


def _render_soak_state(*, index: int, soak_steps: int) -> str:
    in_soak_target = "Approach" if index >= soak_steps else f"Soak_{index + 1}"
    stable_expr = "(stable:eps) * app(Regulate, [])" if index >= soak_steps else "(stable:0)"
    return f"""    Soak_{index} = gen([], (
        (critical:0)
        \\/ (unsafe:0)
        \\/ (done_regulated:0)
        \\/ (done_unregulated:0)
        \\/ (deadline:0)
        \\/ (overshoot:0)
        \\/ {stable_expr}
        \\/ (in_soak:eps) * app({in_soak_target}, [])
        \\/ (safe:eps) * app(Preheat, [])
        \\/ (other:0)
    ))"""


def render_cstr_config(*, env_name: str, host: str, port: int, max_episode_steps: int) -> str:
    """Render YAML runtime config for the CSTR RML monitor."""

    return f"""env_name: {env_name}
host: {host}
port: {port}
max_episode_steps : {max_episode_steps}
variables:
    - name: critical
      type: float
      location: info
      identifier: event_temp_critical
    - name: stable
      type: float
      location: info
      identifier: event_stable_step
    - name: temp_safe
      type: float
      location: info
      identifier: event_temp_safe
    - name: in_soak_band
      type: float
      location: info
      identifier: event_in_soak_band
    - name: overshoot
      type: float
      location: info
      identifier: event_overshoot
    - name: past_deadline
      type: float
      location: info
      identifier: event_past_deadline
    - name: heating_rate_exceeded
      type: float
      location: info
      identifier: event_heating_rate_exceeded
reward:
      name: task
      true: 0
      currently_true: 0
      currently_false: 0
      false: 0
"""
