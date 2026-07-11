"""Build RML monitor-state progress values for multi-task LetterEnv."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from envs.multitask_letter_env.rml_generation import generate_task_suite_rml
from envs.multitask_letter_env.tasks import get_task_suite
from rml_rm.experiments.runtime import write_json
from rml_rm.monitors import RMLMonitorProcess, find_free_port
from rml_rm.wrappers.rml_monitor import WebSocketMonitorClient, normalize_monitor_state


DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "monitor_progress_catalogue.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-suite", default="small_v1")
    parser.add_argument("--max-n", type=int, default=5)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    return parser.parse_args()


def build_monitor_progress_catalogue(args: argparse.Namespace) -> dict[str, Any]:
    generated = generate_task_suite_rml(args.task_suite)
    progress_by_task_and_n: dict[str, dict[str, dict[str, int]]] = {}

    for generated_task, task in zip(generated, get_task_suite(args.task_suite), strict=True):
        port = find_free_port(args.host)
        task_progress: dict[str, dict[str, int]] = {}
        process = RMLMonitorProcess(
            spec_path=generated_task.spec_path,
            port=port,
            host=args.host,
            log_path=args.output_path.parent / f"{task.key}_progress_catalogue_monitor.log",
        )
        with process:
            client = WebSocketMonitorClient(host=args.host, port=port)
            for n_value in range(1, args.max_n + 1):
                state_progress = _state_progress_for_trace(
                    client,
                    task.successful_events(n=n_value),
                    n_value=n_value,
                )
                state_progress["1"] = len(task.successful_events(n=n_value))
                state_progress["false_verdict"] = -1
                task_progress[str(n_value)] = dict(sorted(state_progress.items()))
        progress_by_task_and_n[task.key] = task_progress

    payload = {
        "description": (
            "Progress values for normalized RML monitor states. Values are completed "
            "target-event counts for each task and n."
        ),
        "task_suite": args.task_suite,
        "max_n": args.max_n,
        "progress_by_task_and_n": progress_by_task_and_n,
    }
    write_json(args.output_path, payload)
    return payload


def _state_progress_for_trace(
    client: WebSocketMonitorClient,
    target_events: tuple[str, ...],
    *,
    n_value: int,
) -> dict[str, int]:
    state_progress: dict[str, int] = {}
    _reset_monitor(client)
    progress = 0
    for event in _with_neutral_events(target_events):
        response = client.send(_event_payload(event, n_value=n_value))
        if event != "_" and progress < len(target_events) and event == target_events[progress]:
            progress += 1
        state = normalize_monitor_state(str(response["monitor_state"]))
        state_progress[state] = max(progress, state_progress.get(state, -1))
    return state_progress


def _with_neutral_events(events: tuple[str, ...]) -> tuple[str, ...]:
    expanded: list[str] = ["_"]
    for event in events:
        expanded.append(event)
        expanded.append("_")
    return tuple(expanded)


def _event_payload(event: str, *, n_value: int) -> dict[str, Any]:
    payload = {
        "time": [],
        "action": [],
        "x": 0.0,
        "yy": 0.0,
        "a": 0.0,
        "b": 0.0,
        "c": 0.0,
        "d": 0.0,
        "terminate": False,
    }
    if event == "A":
        payload["a"] = float(n_value)
    elif event == "B":
        payload["b"] = 1.0
    elif event == "C":
        payload["c"] = 1.0
    elif event == "D":
        payload["d"] = 1.0
    return payload


def _reset_monitor(client: WebSocketMonitorClient) -> None:
    client.send(
        {
            "time": [],
            "action": [],
            "x": [],
            "yy": [],
            "a": [],
            "b": [],
            "c": [],
            "d": [],
            "terminate": True,
        }
    )


if __name__ == "__main__":
    build_monitor_progress_catalogue(parse_args())
