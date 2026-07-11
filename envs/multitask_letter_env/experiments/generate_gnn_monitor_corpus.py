"""Generate RML monitor-transition data for graph encoder pretraining."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from envs.multitask_letter_env.rml_generation import CONFIGS_ROOT, SPECS_ROOT
from envs.multitask_letter_env.tasks import LETTER_ALPHABET, LetterTaskSpec, get_task_suite
from rml_rm.experiments.runtime import managed_monitor_group, write_json, write_jsonl
from rml_rm.monitors.transaction import (
    MonitorClient,
    WebSocketMonitorClient,
    load_monitor_config,
    monitor_payload_from_observation,
    normalize_monitor_state,
    normalize_verdict,
    reset_monitor,
)


NO_EVENT = "_"
REPO_ROOT = Path(__file__).resolve().parents[3]
MULTITASK_ROOT = REPO_ROOT / "envs" / "multitask_letter_env"
DEFAULT_OUTPUT_PATH = (
    MULTITASK_ROOT
    / "results_and_evaluation"
    / "encoder_pretraining"
    / "gnn_corpus_small_v1_seed0"
    / "monitor_states.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-suite", default="small_v1")
    parser.add_argument("--max-n", type=int, default=5)
    parser.add_argument("--random-traces-per-task-n", type=int, default=200)
    parser.add_argument("--max-random-trace-length", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def generate_gnn_monitor_corpus(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    tasks = get_task_suite(args.task_suite)
    monitor_specs = {task.key: (SPECS_ROOT / f"{task.key}.pl").resolve() for task in tasks}
    monitor_configs = {task.key: (CONFIGS_ROOT / f"{task.key}.yaml").resolve() for task in tasks}
    output_path = args.output_path.resolve()
    runtime_root = output_path.parent / "monitor_runtime"

    rows: list[dict[str, Any]] = []
    trace_index = 0
    with managed_monitor_group(
        output_dir=runtime_root,
        monitor_specs=monitor_specs,
        monitor_config_templates=monitor_configs,
        config_dir_name="monitor_configs",
        log_dir_name="monitor_logs",
        max_episode_steps=args.max_episode_steps,
    ) as runtime:
        config_by_key = {
            key: load_monitor_config(runtime.config_paths[key])
            for key in runtime.config_paths
        }
        client_by_key = {
            key: WebSocketMonitorClient(host="127.0.0.1", port=runtime.ports[key])
            for key in runtime.ports
        }

        for task in tasks:
            client = client_by_key[task.key]
            variables = list(config_by_key[task.key]["variables"])
            for n_value in range(1, int(args.max_n) + 1):
                target_events = task.successful_events(n=n_value)
                traces = _deterministic_traces(task, n_value, target_events)
                for _ in range(int(args.random_traces_per_task_n)):
                    traces.append(
                        {
                            "trace_type": "random",
                            "events": _random_events(rng, max_length=int(args.max_random_trace_length)),
                            "expected_outcome": "unknown",
                        }
                    )
                for trace in traces:
                    rows.extend(
                        _replay_trace(
                            client,
                            variables,
                            task,
                            n_value,
                            target_events,
                            events=list(trace["events"]),
                            trace_index=trace_index,
                            trace_type=str(trace["trace_type"]),
                            expected_outcome=str(trace["expected_outcome"]),
                            task_suite=args.task_suite,
                        )
                    )
                    trace_index += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, rows, sort_keys=True)
    unique_states = {str(row["normalized_monitor_state"]) for row in rows}
    summary = {
        "task_suite": args.task_suite,
        "max_n": int(args.max_n),
        "rows": len(rows),
        "unique_normalized_monitor_states": len(unique_states),
        "output_path": str(output_path),
        "random_traces_per_task_n": int(args.random_traces_per_task_n),
        "max_random_trace_length": int(args.max_random_trace_length),
        "seed": int(args.seed),
        "artifacts": {
            "corpus": str(output_path),
            "summary": str(output_path.parent / "summary.json"),
        },
    }
    write_json(output_path.parent / "summary.json", summary)
    return summary


def _deterministic_traces(
    task: LetterTaskSpec,
    n_value: int,
    target_events: tuple[str, ...],
) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = [
        {
            "trace_type": "success",
            "events": list(target_events),
            "expected_outcome": "success",
        }
    ]
    for prefix_length in range(len(target_events) + 1):
        traces.append(
            {
                "trace_type": "prefix",
                "events": list(target_events[:prefix_length]),
                "expected_outcome": "success" if prefix_length == len(target_events) else "partial",
            }
        )
    for prefix_length, expected in enumerate(target_events):
        for wrong_event in LETTER_ALPHABET:
            if wrong_event == expected:
                continue
            traces.append(
                {
                    "trace_type": "wrong_event",
                    "events": [*target_events[:prefix_length], wrong_event],
                    "expected_outcome": "failure",
                }
            )
    return traces


def _random_events(rng: random.Random, *, max_length: int) -> list[str]:
    return [rng.choice((*LETTER_ALPHABET, NO_EVENT)) for _ in range(rng.randint(1, max_length))]


def _replay_trace(
    client: MonitorClient,
    variables: list[Mapping[str, Any]],
    task: LetterTaskSpec,
    n_value: int,
    target_events: tuple[str, ...],
    *,
    events: list[str],
    trace_index: int,
    trace_type: str,
    expected_outcome: str,
    task_suite: str,
) -> list[dict[str, Any]]:
    reset_monitor(client, variables)
    rows: list[dict[str, Any]] = []
    progress_index = 0
    failed = False

    rows.append(
        _observe(
            client,
            variables,
            task,
            n_value,
            target_events,
            event=NO_EVENT,
            progress_index=progress_index,
            step_index=0,
            trace_index=trace_index,
            trace_type=trace_type,
            expected_outcome=expected_outcome,
            task_suite=task_suite,
        )
    )
    for step_index, event in enumerate(events, start=1):
        if not failed and event != NO_EVENT:
            expected = target_events[progress_index] if progress_index < len(target_events) else None
            if event == expected:
                progress_index += 1
            elif event in LETTER_ALPHABET:
                failed = True
        rows.append(
            _observe(
                client,
                variables,
                task,
                n_value,
                target_events,
                event=event,
                progress_index=progress_index,
                step_index=step_index,
                trace_index=trace_index,
                trace_type=trace_type,
                expected_outcome=expected_outcome,
                task_suite=task_suite,
            )
        )
        if rows[-1]["is_terminal"]:
            break
    return rows


def _observe(
    client: MonitorClient,
    variables: list[Mapping[str, Any]],
    task: LetterTaskSpec,
    n_value: int,
    target_events: tuple[str, ...],
    *,
    event: str,
    progress_index: int,
    step_index: int,
    trace_index: int,
    trace_type: str,
    expected_outcome: str,
    task_suite: str,
) -> dict[str, Any]:
    response = client.send(_payload_for_event(variables, event, n_value=n_value))
    verdict = normalize_verdict(str(response["verdict"]))
    monitor_state = str(response["monitor_state"])
    normalized_state = normalize_monitor_state(monitor_state)
    terminal_type = _terminal_type(normalized_state, verdict)
    phase_index = _phase_index(terminal_type, progress_index, target_events)
    return {
        "family": "multitask_letter_env",
        "task_suite": task_suite,
        "task_id": task.task_id,
        "task_key": task.key,
        "task_expression": task.expression,
        "n": int(n_value),
        "trace_index": int(trace_index),
        "trace_type": trace_type,
        "expected_outcome": expected_outcome,
        "step_index": int(step_index),
        "event": event,
        "monitor_state": monitor_state,
        "normalized_monitor_state": normalized_state,
        "verdict": verdict,
        "progress_index": int(phase_index),
        "phase_label": f"{task.key}:{phase_index}",
        "target_length": len(target_events),
        "terminal_type": terminal_type,
        "success": terminal_type == "success",
        "failure": terminal_type == "failure",
        "is_terminal": terminal_type in {"success", "failure"},
    }


def _phase_index(terminal_type: str, progress_index: int, target_events: tuple[str, ...]) -> int:
    if terminal_type == "failure":
        return len(target_events) + 1
    if terminal_type == "success":
        return len(target_events)
    return min(progress_index, len(target_events))


def _terminal_type(normalized_state: str, verdict: str) -> str:
    if normalized_state == "false_verdict" or verdict == "false":
        return "failure"
    if normalized_state == "1" or verdict in {"true", "currently_true"}:
        return "success"
    return "active"


def _payload_for_event(
    variables: list[Mapping[str, Any]],
    event: str,
    *,
    n_value: int,
) -> dict[str, Any]:
    vector = np.zeros(6, dtype=np.float32)
    if event == "A":
        vector[2] = float(n_value)
    elif event == "B":
        vector[3] = 1.0
    elif event == "C":
        vector[4] = 1.0
    elif event == "D":
        vector[5] = 1.0
    payload = monitor_payload_from_observation(
        variables=variables,
        observation={"position": vector},
    )
    payload["terminate"] = False
    return payload


def main() -> None:
    print(json.dumps(generate_gnn_monitor_corpus(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
