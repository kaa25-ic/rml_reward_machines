"""Generate a parameterized LetterEnv monitor-transition corpus for GNN pretraining."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rml_rm.experiments.runtime import json_ready, write_json
from rml_rm.monitors import RMLMonitorProcess, find_free_port
from rml_rm.wrappers.rml_monitor import WebSocketMonitorClient, normalize_monitor_state


REPO_ROOT = Path(__file__).resolve().parents[3]
LETTER_ENV_ROOT = REPO_ROOT / "envs" / "letter_env"
DEFAULT_OUTPUT_DIR = (
    LETTER_ENV_ROOT
    / "results_and_evaluation"
    / "encoder_pretraining"
    / "gnn_parameterized_corpus_n1to5_seed0"
)


@dataclass(frozen=True)
class GNNCorpusConfig:
    """Configuration for the LetterEnv GNN monitor corpus."""

    output_dir: Path = DEFAULT_OUTPUT_DIR
    max_count: int = 5
    neutral_events_between_symbols: int = 1
    include_prefixes: bool = True
    host: str = "127.0.0.1"
    websocket_timeout: float = 5.0
    startup_timeout_seconds: float = 10.0


@dataclass(frozen=True)
class ParameterizedTemplate:
    task_id: str
    family: str
    expression: str
    prolog_spec: str


def build_gnn_monitor_corpus(config: GNNCorpusConfig) -> dict[str, Any]:
    """Generate monitor transition rows from parameterized LetterEnv RML templates."""
    output_dir = config.output_dir.expanduser().resolve()
    specs_dir = output_dir / "generated_specs"
    logs_dir = output_dir / "monitor_logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    specs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", {"experiment": "letter_env_gnn_monitor_corpus", "config": asdict(config)})

    corpus_path = output_dir / "monitor_states.jsonl"
    unique_states: set[str] = set()
    unique_normalized_states: set[str] = set()
    verdict_counts: dict[str, int] = {}
    rows_written = 0
    failed_traces = 0

    with corpus_path.open("w", encoding="utf-8") as corpus_file:
        for template_index, template in enumerate(parameterized_templates()):
            spec_path = specs_dir / f"{template_index:04d}_{template.task_id}.pl"
            spec_path.write_text(template.prolog_spec, encoding="utf-8")
            traces = _template_traces(
                template,
                max_count=config.max_count,
                include_prefixes=config.include_prefixes,
            )
            for trace_index, trace in enumerate(traces):
                events = _with_neutral_events(trace["events"], config.neutral_events_between_symbols)
                process = RMLMonitorProcess(
                    spec_path=spec_path,
                    port=find_free_port(config.host),
                    host=config.host,
                    startup_timeout_seconds=config.startup_timeout_seconds,
                    log_path=logs_dir / f"{template_index:04d}_{template.task_id}_{trace_index:04d}.log",
                )
                with process:
                    client = WebSocketMonitorClient(
                        host=config.host,
                        port=process.port,
                        timeout=config.websocket_timeout,
                    )
                    rows, failed = _replay_trace(
                        client=client,
                        template=template,
                        trace={**trace, "events": events},
                        template_index=template_index,
                        trace_index=trace_index,
                    )
                failed_traces += int(failed)
                for row in rows:
                    state = str(row["monitor_state"])
                    normalized_state = normalize_monitor_state(state)
                    unique_states.add(state)
                    unique_normalized_states.add(normalized_state)
                    verdict = str(row["verdict"])
                    verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
                    corpus_file.write(json.dumps(row, sort_keys=True) + "\n")
                    rows_written += 1

    summary = {
        "corpus_path": str(corpus_path),
        "max_count": int(config.max_count),
        "num_templates": len(parameterized_templates()),
        "num_rows": rows_written,
        "num_unique_monitor_states": len(unique_states),
        "num_unique_normalized_monitor_states": len(unique_normalized_states),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "failed_traces": failed_traces,
        "artifacts": {
            "config": str(output_dir / "config.json"),
            "corpus": str(corpus_path),
            "summary": str(output_dir / "summary.json"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parameterized_templates() -> list[ParameterizedTemplate]:
    return [
        ParameterizedTemplate(
            task_id="param_A_of_N_B_C_D_power_N",
            family="parameterized_original",
            expression="A(N) B C D^N",
            prolog_spec=_common_header(a_numeric=True)
            + """
trace_expression('Main', Main) :-
    Main = (
        star((not_abcd:eps)) *
        var(n, ((a_match(var(n)):eps) * app(B, [var('n')])))
    ),
    B = gen(['n'], (star((not_abcd:eps)) * ((b_match:eps) * app(C, [var('n')])))),
    C = gen(['n'], (star((not_abcd:eps)) * ((c_match:eps) * app(D, [var('n')])))),
    D = gen(
        ['n'],
        guarded(
            (var('n') > 0),
            (star((not_abcd:eps)) * ((d_match:eps) * app(D, [(var('n') - 1)]))),
            1
        )
    ).
""",
        ),
        ParameterizedTemplate(
            task_id="param_A_power_N_B_C",
            family="parameterized_counting",
            expression="A(N) A^(N-1) B C",
            prolog_spec=_common_header(a_numeric=True)
            + """
trace_expression('Main', Main) :-
    Main = (
        star((not_abcd:eps)) *
        var(n, ((a_match(var(n)):eps) * app(RepeatA, [(var('n') - 1)])))
    ),
    RepeatA = gen(
        ['n'],
        guarded(
            (var('n') > 0),
            (star((not_abcd:eps)) * ((a_unit_match:eps) * app(RepeatA, [(var('n') - 1)]))),
            (star((not_abcd:eps)) * ((b_match:eps) * (star((not_abcd:eps)) * ((c_match:eps) * star((not_abcd:eps))))))
        )
    ).
""",
        ),
        ParameterizedTemplate(
            task_id="param_A_B_C_of_N_D_power_N",
            family="parameterized_counting",
            expression="A B C(N) D^N",
            prolog_spec=_common_header(c_numeric=True)
            + """
trace_expression('Main', Main) :-
    Main = (
        star((not_abcd:eps)) *
        ((a_match:eps) *
            (star((not_abcd:eps)) *
                ((b_match:eps) *
                    (star((not_abcd:eps)) *
                        var(n, ((c_match(var(n)):eps) * app(RepeatD, [var('n')])))))))
    ),
    RepeatD = gen(
        ['n'],
        guarded(
            (var('n') > 0),
            (star((not_abcd:eps)) * ((d_match:eps) * app(RepeatD, [(var('n') - 1)]))),
            1
        )
    ).
""",
        ),
        ParameterizedTemplate(
            task_id="param_A_of_N_repeat_BC_N_D",
            family="parameterized_repeated_block",
            expression="A(N) (B C)^N D",
            prolog_spec=_common_header(a_numeric=True)
            + """
trace_expression('Main', Main) :-
    Main = (
        star((not_abcd:eps)) *
        var(n, ((a_match(var(n)):eps) * app(RepeatBC, [var('n')])))
    ),
    RepeatBC = gen(
        ['n'],
        guarded(
            (var('n') > 0),
            (star((not_abcd:eps)) *
                ((b_match:eps) *
                    (star((not_abcd:eps)) *
                        ((c_match:eps) * app(RepeatBC, [(var('n') - 1)]))))),
            (star((not_abcd:eps)) * ((d_match:eps) * star((not_abcd:eps))))
        )
    ).
""",
        ),
    ]


def _common_header(*, a_numeric: bool = False, c_numeric: bool = False) -> str:
    lines = [
        ":- module('spec', [trace_expression/2, match/2]).",
        ":- use_module(monitor('deep_subdict')).",
        "",
        "match(_event, a_match(N)) :- deep_subdict(_event, _{'a':N}), >(N, 0)." if a_numeric else "match(_event, a_match) :- deep_subdict(_event, _{'a':T}), T=1.0.",
    ]
    if a_numeric:
        lines.append("match(_event, a_unit_match) :- deep_subdict(_event, _{'a':T}), T=1.0.")
    lines.append("match(_event, b_match) :- deep_subdict(_event, _{'b':T}), T=1.0.")
    lines.append("match(_event, c_match(N)) :- deep_subdict(_event, _{'c':N}), >(N, 0)." if c_numeric else "match(_event, c_match) :- deep_subdict(_event, _{'c':T}), T=1.0.")
    if c_numeric:
        lines.append("match(_event, c_unit_match) :- deep_subdict(_event, _{'c':T}), T=1.0.")
    lines.extend(
        [
            "match(_event, d_match) :- deep_subdict(_event, _{'d':T}), T=1.0.",
            "match(_event, not_abcd) :-",
            "    not(match(_event, a_match)),",
            "    not(match(_event, a_match(_))),",
            "    not(match(_event, a_unit_match)),",
            "    not(match(_event, b_match)),",
            "    not(match(_event, c_match)),",
            "    not(match(_event, c_match(_))),",
            "    not(match(_event, c_unit_match)),",
            "    not(match(_event, d_match)).",
            "match(_, any).",
            "",
        ]
    )
    return "\n".join(lines)


def _template_traces(template: ParameterizedTemplate, *, max_count: int, include_prefixes: bool) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for n in range(1, max_count + 1):
        success_events = _success_events(template.task_id, n)
        traces.append({"label": f"success_n{n}", "events": success_events, "raw_events": success_events, "n": n, "expected_outcome": "success"})
        wrong = _wrong_first_event(success_events)
        traces.append({"label": f"wrong_first_n{n}", "events": wrong, "raw_events": wrong, "n": n, "expected_outcome": "failure"})
        if include_prefixes:
            for prefix_len in range(0, len(success_events) + 1):
                prefix = success_events[:prefix_len]
                traces.append(
                    {
                        "label": f"prefix_n{n}_{prefix_len}",
                        "events": prefix,
                        "raw_events": prefix,
                        "n": n,
                        "expected_outcome": "success" if prefix_len == len(success_events) else "partial",
                    }
                )
    return _dedupe_traces(traces)


def _success_events(task_id: str, n: int) -> list[str]:
    if task_id == "param_A_of_N_B_C_D_power_N":
        return [f"A:{n}", "B", "C", *(["D"] * n)]
    if task_id == "param_A_power_N_B_C":
        return [f"A:{n}", *(["A"] * max(0, n - 1)), "B", "C"]
    if task_id == "param_A_B_C_of_N_D_power_N":
        return ["A", "B", f"C:{n}", *(["D"] * n)]
    if task_id == "param_A_of_N_repeat_BC_N_D":
        return [f"A:{n}", *([event for _ in range(n) for event in ("B", "C")]), "D"]
    raise ValueError(f"Unknown task template: {task_id}")


def _wrong_first_event(success_events: list[str]) -> list[str]:
    first = success_events[0].split(":", maxsplit=1)[0]
    for candidate in ("A", "B", "C", "D"):
        if candidate != first:
            return [candidate]
    return ["B"]


def _dedupe_traces(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    deduped: list[dict[str, Any]] = []
    for trace in traces:
        key = (str(trace["label"]), tuple(trace["events"]))
        if key not in seen:
            seen.add(key)
            deduped.append(trace)
    return deduped


def _with_neutral_events(events: list[str], neutral_count: int) -> list[str]:
    if neutral_count <= 0 or not events:
        return events
    expanded: list[str] = []
    for event in events:
        expanded.extend(["_"] * neutral_count)
        expanded.append(event)
    expanded.extend(["_"] * neutral_count)
    return expanded


def _replay_trace(
    *,
    client: WebSocketMonitorClient,
    template: ParameterizedTemplate,
    trace: dict[str, Any],
    template_index: int,
    trace_index: int,
) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    failed = False
    for step_index, event in enumerate(trace["events"]):
        response = client.send(_event_payload(event))
        verdict = str(response.get("verdict", ""))
        monitor_state = str(response.get("monitor_state", ""))
        failed = failed or monitor_state == "false_verdict" or verdict == "false"
        rows.append(
            {
                "template_index": template_index,
                "task_id": template.task_id,
                "family": template.family,
                "expression": template.expression,
                "trace_index": trace_index,
                "trace_label": trace["label"],
                "expected_outcome": trace["expected_outcome"],
                "n": trace.get("n"),
                "step_index": step_index,
                "event": event,
                "raw_trace_events": trace.get("raw_events", []),
                "verdict": verdict,
                "monitor_state": monitor_state,
                "normalized_monitor_state": normalize_monitor_state(monitor_state),
                "is_terminal": monitor_state in {"1", "false_verdict"},
            }
        )
        if failed:
            break
    return rows, failed


def _event_payload(event: str) -> dict[str, float | bool | list[Any]]:
    payload: dict[str, float | bool | list[Any]] = {
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
    if event == "_":
        return payload
    symbol, value = _parse_event(event)
    payload[symbol.lower()] = value
    return payload


def _parse_event(event: str) -> tuple[str, float]:
    if ":" in event:
        symbol, value = event.split(":", maxsplit=1)
        return symbol.upper(), float(value)
    return event.upper(), 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-count", type=int, default=5)
    parser.add_argument("--neutral-events-between-symbols", type=int, default=1)
    parser.add_argument("--no-prefixes", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--websocket-timeout", type=float, default=5.0)
    parser.add_argument("--startup-timeout-seconds", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_gnn_monitor_corpus(
        GNNCorpusConfig(
            output_dir=args.output_dir,
            max_count=args.max_count,
            neutral_events_between_symbols=args.neutral_events_between_symbols,
            include_prefixes=not args.no_prefixes,
            host=args.host,
            websocket_timeout=args.websocket_timeout,
            startup_timeout_seconds=args.startup_timeout_seconds,
        )
    )
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
