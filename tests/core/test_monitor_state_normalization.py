"""Regression tests for shared RML monitor-state normalization."""

from __future__ import annotations

import pytest

from rml_rm.encodings.monitor_state import normalize_monitor_state as normalize_encoding_state
from rml_rm.encodings.monitor_state import split_top_level_factors
from rml_rm.monitors.transaction import normalize_monitor_state as normalize_transaction_state


@pytest.mark.parametrize(
    ("raw_state", "expected"),
    [
        ("foo_12", "foo"),
        ("app_123(foo_99)", "app(foo)"),
        ("star(waiting_for_hover:eps)", "star(waiting_for_hover:eps)"),
        ("star(waiting\\*for:eps)", "star(waiting*for:eps)"),
        (123, "123"),
    ],
)
def test_monitor_state_normalization_is_canonical(raw_state: object, expected: str) -> None:
    assert normalize_encoding_state(raw_state) == expected
    assert normalize_transaction_state(raw_state) == expected


def test_transaction_normalizer_reuses_encoding_normalizer() -> None:
    assert normalize_transaction_state is normalize_encoding_state


def test_escaped_product_delimiters_are_handled_once_before_splitting() -> None:
    assert split_top_level_factors("left\\*middle*right") == ["left", "middle", "right"]
