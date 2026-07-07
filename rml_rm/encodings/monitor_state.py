"""Utilities for encoding RML monitor-state strings."""

from __future__ import annotations

import ast
import operator
import re
from collections import OrderedDict
from collections.abc import Iterable

import numpy as np


ADDITIONAL_NUMERIC_VALUE_SUFFIX = "::ADDITIONAL::"


def normalize_monitor_state(state: str) -> str:
    """Remove runtime-generated variable suffixes from a monitor-state string."""
    return re.sub(r"_[0-9]+", "", str(state))


def split_top_level_factors(state: str) -> list[str]:
    """Split a monitor-state expression at top-level product operators."""
    normalized = normalize_monitor_state(state).replace("@", "")
    if normalized.startswith("(eps*"):
        normalized = normalized[len("(eps*") :]

    factors: list[str] = []
    current: list[str] = []
    paren_depth = 0
    bracket_depth = 0

    for character in normalized:
        if character == "(":
            paren_depth += 1
        elif character == ")":
            paren_depth = max(paren_depth - 1, 0)
        elif character == "[":
            bracket_depth += 1
        elif character == "]":
            bracket_depth = max(bracket_depth - 1, 0)

        if character == "*" and paren_depth == 0 and bracket_depth == 0:
            factor = "".join(current).strip()
            if factor:
                factors.append(factor)
            current = []
            continue

        current.append(character)

    trailing = "".join(current).strip()
    if trailing:
        factors.append(trailing)
    return factors


def replace_numerical_parts(event: str) -> str:
    """Replace concrete bracketed numeric literals with placeholders."""
    return re.sub(
        r"\[(\d+(\.\d+)?(?:\+\d+(\.\d+)?|\-\d+(\.\d+)?)*"
        r"(?:,\d+(\.\d+)?(?:\+\d+(\.\d+)?|\-\d+(\.\d+)?)*?)*)\]",
        lambda match: "[" + ",".join("{num}" for _ in match.group(1).split(",")) + "]",
        event,
    )


def extract_numerical_values(event: str) -> list[float] | None:
    """Extract numeric values from bracketed expressions in a monitor-state factor."""
    matches = re.findall(
        r"\[(\d+(\.\d+)?(?:\+\d+(\.\d+)?|\-\d+(\.\d+)?)*"
        r"(?:,\d+(\.\d+)?(?:\+\d+(\.\d+)?|\-\d+(\.\d+)?)*?)*)\]",
        event,
    )
    values: list[float] = []
    for match in matches:
        for expression in match[0].split(","):
            evaluated = _evaluate_numeric_expression(expression)
            values.append(0.01 if evaluated == 0 else float(evaluated))
    return values or None


def extract_events(state: str) -> list[str]:
    """Return normalized top-level factors from a monitor-state string."""
    return [replace_numerical_parts(part) for part in split_top_level_factors(state)]


def build_one_hot_event_index(states: Iterable[str]) -> dict[str, int]:
    """Construct a stable one-hot event index from monitor states."""
    ordered_events: OrderedDict[str, int] = OrderedDict()
    for state in states:
        for event in extract_events(state):
            if event not in ordered_events:
                ordered_events[event] = len(ordered_events)
    return dict(ordered_events)


def build_numerical_event_index(states: Iterable[str]) -> dict[str, int]:
    """Construct a stable numerical event index from monitor states."""
    ordered_events: OrderedDict[str, int] = OrderedDict()
    next_index = 0
    for state in states:
        for event in extract_events(state):
            if event in ordered_events:
                continue
            ordered_events[event] = next_index
            next_index += 1
            placeholder_count = event.count("{num}")
            for extra_index in range(1, placeholder_count):
                ordered_events[event + ADDITIONAL_NUMERIC_VALUE_SUFFIX * extra_index] = next_index
                next_index += 1
    return dict(ordered_events)


def encode_one_hot_monitor_state(state: str, event_index: dict[str, int]) -> np.ndarray:
    """Encode a monitor state as binary event-presence features."""
    vector = np.zeros(len(event_index), dtype=np.float32)
    for event in extract_events(state):
        if event in event_index:
            vector[event_index[event]] = 1.0
    return vector


def encode_numerical_monitor_state(state: str, event_index: dict[str, int]) -> np.ndarray:
    """Encode a monitor state as numerical event features."""
    vector = np.zeros(len(event_index), dtype=np.float32)
    for event in split_top_level_factors(state):
        normalized_event = replace_numerical_parts(event)
        values = extract_numerical_values(event)
        if normalized_event not in event_index:
            continue
        if values is None:
            vector[event_index[normalized_event]] = 1.0
            continue
        for value_index, value in enumerate(values):
            key = normalized_event + ADDITIONAL_NUMERIC_VALUE_SUFFIX * value_index
            if key in event_index:
                vector[event_index[key]] = value
    return vector


def _evaluate_numeric_expression(expression: str) -> float:
    node = ast.parse(expression, mode="eval")
    return float(_evaluate_ast(node.body))


def _evaluate_ast(node: ast.AST) -> float:
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
    }
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in operators:
        return operators[type(node.op)](_evaluate_ast(node.left), _evaluate_ast(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_evaluate_ast(node.operand)
    raise ValueError(f"Unsupported numeric monitor expression: {ast.dump(node)}")
