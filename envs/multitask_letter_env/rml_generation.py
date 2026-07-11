"""Generate RML monitor specs and config templates for multi-task LetterEnv."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from envs.multitask_letter_env.tasks import CountToken, LetterTaskSpec, TaskToken, get_task_suite


PACKAGE_ROOT = Path(__file__).resolve().parent
SPECS_ROOT = PACKAGE_ROOT / "specs"
CONFIGS_ROOT = PACKAGE_ROOT / "configs"

LETTER_MATCH_NAMES = {
    "A": "a_match(var(n))",
    "B": "b_match",
    "C": "c_match",
    "D": "d_match",
}


@dataclass(frozen=True)
class GeneratedRMLTask:
    """Generated RML files for one task."""

    task_id: int
    task_key: str
    spec_path: Path
    config_path: Path


def generate_task_suite_rml(
    suite_name: str = "small_v1",
    *,
    host: str = "127.0.0.1",
    base_port: int = 18_201,
    max_episode_steps: int = 200,
) -> list[GeneratedRMLTask]:
    """Write specs and config templates for a frozen task suite."""
    SPECS_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIGS_ROOT.mkdir(parents=True, exist_ok=True)
    generated: list[GeneratedRMLTask] = []
    for task in get_task_suite(suite_name):
        spec_path = SPECS_ROOT / f"{task.key}.pl"
        config_path = CONFIGS_ROOT / f"{task.key}.yaml"
        spec_path.write_text(render_task_spec(task), encoding="utf-8")
        config_path.write_text(
            render_task_config(
                task,
                host=host,
                port=base_port + task.task_id,
                max_episode_steps=max_episode_steps,
            ),
            encoding="utf-8",
        )
        generated.append(
            GeneratedRMLTask(
                task_id=task.task_id,
                task_key=task.key,
                spec_path=spec_path,
                config_path=config_path,
            )
        )
    return generated


def render_task_spec(task: LetterTaskSpec) -> str:
    """Render one task as a Prolog RML monitor specification."""
    if not task.tokens or task.tokens[0] != "A":
        raise ValueError(f"Task {task.key!r} must start with A.")

    body_lines = [
        ":- module('spec', [trace_expression/2, match/2]).",
        ":- use_module(monitor('deep_subdict')).",
        "",
        "match(_event, a_match(N)) :- deep_subdict(_event, _{'a':N}), >(N, 0).",
        "match(_event, b_match) :- deep_subdict(_event, _{'b':T}), T=1.0.",
        "match(_event, c_match) :- deep_subdict(_event, _{'c':T}), T=1.0.",
        "match(_event, d_match) :- deep_subdict(_event, _{'d':T}), T=1.0.",
        "match(_event, not_abcd) :-",
        "    not(match(_event, a_match(_))),",
        "    not(match(_event, b_match)),",
        "    not(match(_event, c_match)),",
        "    not(match(_event, d_match)).",
        "match(_, any).",
        "",
        "trace_expression('Main', Main) :-",
    ]

    remaining_tokens = task.tokens[1:]
    assignments: list[str] = []
    if remaining_tokens:
        first_stage = _stage_name(0, remaining_tokens[0])
        assignments.append(
            "    Main = (star((not_abcd:eps)) * "
            f"var(n, ((a_match(var(n)):eps) * app({first_stage}, [var('n')]))))"
        )
        for index, token in enumerate(remaining_tokens):
            assignments.append(_render_stage_assignment(index, token, remaining_tokens))
    else:
        assignments.append("    Main = (star((not_abcd:eps)) * var(n, (a_match(var(n)):eps)))")

    for index, assignment in enumerate(assignments):
        suffix = "." if index == len(assignments) - 1 else ","
        body_lines.append(f"{assignment}{suffix}")
    return "\n".join(body_lines) + "\n"


def render_task_config(
    task: LetterTaskSpec,
    *,
    host: str,
    port: int,
    max_episode_steps: int,
) -> str:
    """Render the YAML config consumed by the RML monitor wrapper."""
    return f"""env_name: multitask-letter-env-{task.key}
host: {host}
port: {port}
max_episode_steps: {max_episode_steps}
variables:
  - name: x
    type: float
    location: obs
    identifier: 0
  - name: yy
    type: float
    location: obs
    identifier: 1
  - name: a
    type: float
    location: obs
    identifier: 2
  - name: b
    type: float
    location: obs
    identifier: 3
  - name: c
    type: float
    location: obs
    identifier: 4
  - name: d
    type: float
    location: obs
    identifier: 5
reward:
  name: task
  true: 100
  currently_true: 100
  currently_false: 0
  false: -40
"""


def _render_stage_assignment(index: int, token: TaskToken, tokens: tuple[TaskToken, ...]) -> str:
    stage_name = _stage_name(index, token)
    next_token = tokens[index + 1] if index + 1 < len(tokens) else None
    next_stage = None if next_token is None else _stage_name(index + 1, next_token)

    if isinstance(token, CountToken):
        if token.letter != "D":
            raise ValueError("Only D-counting tasks are supported.")
        continuation = "1" if next_stage is None else f"app({next_stage}, [var('n')])"
        return (
            f"    {stage_name} = gen(['n'], guarded((var('n') > 0), "
            f"(star((not_abcd:eps)) * ((d_match:eps) * app({stage_name}, "
            f"[(var('n') - 1)]))), {continuation}))"
        )

    match_name = _match_name(token)
    continuation = "1" if next_stage is None else f"app({next_stage}, [var('n')])"
    return (
        f"    {stage_name} = gen(['n'], "
        f"(star((not_abcd:eps)) * (({match_name}:eps) * {continuation})))"
    )


def _stage_name(index: int, token: TaskToken) -> str:
    letter = token.letter if isinstance(token, CountToken) else token
    return f"S{index}_{letter}"


def _match_name(letter: str) -> str:
    try:
        return LETTER_MATCH_NAMES[letter]
    except KeyError as exc:
        raise ValueError(f"Unsupported LetterEnv token: {letter!r}") from exc


if __name__ == "__main__":
    generate_task_suite_rml()
