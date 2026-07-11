"""Build the RML monitor-state catalogue for multi-task LetterEnv."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from envs.multitask_letter_env.encodings import catalogue_to_jsonable
from envs.multitask_letter_env.rml_generation import generate_task_suite_rml
from envs.multitask_letter_env.tasks import LETTER_ALPHABET, get_task_suite
from rml_rm.experiments.runtime import write_json
from rml_rm.monitors import RMLMonitorProcess, find_free_port
from rml_rm.wrappers.rml_monitor import WebSocketMonitorClient, normalize_monitor_state


DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "monitor_state_catalogue.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-suite", default="small_v1")
    parser.add_argument("--max-n", type=int, default=5)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    return parser.parse_args()


def build_monitor_state_catalogue(args: argparse.Namespace) -> dict[str, Any]:
    generated = generate_task_suite_rml(args.task_suite)
    states_by_task: dict[str, list[str]] = {}
    for generated_task, task in zip(generated, get_task_suite(args.task_suite), strict=True):
        port = find_free_port(args.host)
        states: set[str] = set()
        process = RMLMonitorProcess(
            spec_path=generated_task.spec_path,
            port=port,
            host=args.host,
            log_path=args.output_path.parent / f"{task.key}_catalogue_monitor.log",
        )
        with process:
            client = WebSocketMonitorClient(host=args.host, port=port)
            for n_value, trace in _catalogue_traces(task, max_n=args.max_n):
                _reset_monitor(client)
                for event in trace:
                    response = client.send(_event_payload(event, n_value=n_value))
                    states.add(normalize_monitor_state(str(response["monitor_state"])))
        states.update({"1", "false_verdict"})
        states_by_task[task.key] = sorted(states)

    payload = catalogue_to_jsonable(states_by_task)
    payload["task_suite"] = args.task_suite
    payload["max_n"] = args.max_n
    write_json(args.output_path, payload)
    return payload


def _catalogue_traces(task, *, max_n: int) -> list[tuple[int, tuple[str, ...]]]:
    traces: list[tuple[int, tuple[str, ...]]] = []
    for n in range(1, max_n + 1):
        target = task.successful_events(n=n)
        for prefix_length in range(len(target) + 1):
            prefix = target[:prefix_length]
            traces.append((n, prefix))
            traces.append((n, _with_neutral_events(prefix)))
            for wrong_letter in LETTER_ALPHABET:
                if prefix_length < len(target) and wrong_letter != target[prefix_length]:
                    traces.append((n, prefix + (wrong_letter,)))
                    break
    return traces


def _with_neutral_events(events: tuple[str, ...]) -> tuple[str, ...]:
    neutral = "_"
    expanded: list[str] = []
    for event in events:
        expanded.append(neutral)
        expanded.append(event)
    expanded.append(neutral)
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
    build_monitor_state_catalogue(parse_args())
