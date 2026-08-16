"""Audit LetterEnv monitor-state encodings for sufficiency on the reachable catalogue.

By default the audit reports injectivity only (a pure collision check over the
stored catalogue). With ``--with-monitor`` it drives the live RML monitor to
enumerate the reachable states and check the reward and transition-closure
assumptions, producing a full sufficiency verdict and concrete witnesses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from envs.letter_env.encodings import (
    build_letter_env_monitor_encoding,
    load_letter_env_monitor_state_catalogue,
)
from rml_rm.verification.encoding_sufficiency import EncodingAudit, audit_encoding


AUDITED_ENCODINGS = (
    "one_hot",
    "numerical",
    "semantic_progress",
    "learned_gru",
    "learned_graph",
)

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "results_and_evaluation"
    / "verification"
    / "encoding_sufficiency.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encodings", nargs="+", default=list(AUDITED_ENCODINGS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--decimals", type=int, default=9)
    parser.add_argument(
        "--with-monitor",
        action="store_true",
        help="drive the live RML monitor to check reward and transition closure",
    )
    parser.add_argument("--max-n", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.with_monitor:
        from envs.letter_env.experiments.monitor_oracle import build_letter_env_monitor_oracle

        oracle = build_letter_env_monitor_oracle(max_n=args.max_n)
        mode = "monitor-backed"
        results = {
            encoding: _audit_with_monitor(encoding, oracle, args.decimals)
            for encoding in args.encodings
        }
    else:
        catalogue = load_letter_env_monitor_state_catalogue()
        states = list(catalogue.values())
        mode = "collision-only"
        results = {
            encoding: _audit_collisions(encoding, states, args.decimals)
            for encoding in args.encodings
        }

    _print_table(mode, results)
    report = {"mode": mode, "encodings": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")


def _audit_collisions(encoding: str, states: list[str], decimals: int) -> dict[str, object]:
    encoder, _, _ = build_letter_env_monitor_encoding(encoding)
    audit = audit_encoding(states, encoder, decimals=decimals)
    return _summarize(audit)


def _audit_with_monitor(encoding: str, oracle, decimals: int) -> dict[str, object]:
    base_encoder, _, _ = build_letter_env_monitor_encoding(encoding)
    audit = audit_encoding(
        list(oracle.states),
        lambda state: base_encoder(oracle.encoder_input(state)),
        events=oracle.events,
        transition=oracle.transition,
        transition_reward=oracle.transition_reward,
        decimals=decimals,
    )
    summary = _summarize(audit)
    summary["witnesses"] = [
        {"kind": witness.kind, "event": witness.event, "states": list(witness.states)}
        for witness in audit.witnesses
    ]
    return summary


def _summarize(audit: EncodingAudit) -> dict[str, object]:
    return {
        "states": sum(len(group) for group in audit.groups),
        "groups": len(audit.groups),
        "injective": audit.injective,
        "sufficient": audit.sufficient,
        "merged_groups": [list(group) for group in audit.groups if len(group) > 1],
    }


def _print_table(mode: str, results: dict[str, dict[str, object]]) -> None:
    labels = {True: "yes", False: "no", None: "inconclusive"}
    print(f"mode: {mode}")
    if mode == "collision-only":
        print("(injectivity only; run with --with-monitor for a sufficiency verdict)")
    header = f"{'encoding':18} {'states':>6} {'groups':>6} {'injective':>10} {'sufficient':>13} {'witnesses':>10}"
    print(header)
    print("-" * len(header))
    for encoding, row in results.items():
        witnesses = row.get("witnesses")
        witness_count = len(witnesses) if isinstance(witnesses, list) else "-"
        print(
            f"{encoding:18} {row['states']:>6} {row['groups']:>6} "
            f"{str(row['injective']):>10} {labels[row['sufficient']]:>13} {str(witness_count):>10}"
        )


if __name__ == "__main__":
    main()
