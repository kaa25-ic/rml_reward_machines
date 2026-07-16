"""Pure task and RML generation tests for multitask LetterEnv."""

from __future__ import annotations

import pytest

from envs.multitask_letter_env.rml_generation import render_task_config, render_task_spec
from envs.multitask_letter_env.tasks import (
    CountToken,
    LetterTaskSpec,
    expand_task_tokens,
    get_task_suite,
    validate_task_suite,
)


def test_expand_task_tokens_and_successful_events() -> None:
    task = next(task for task in get_task_suite() if task.key == "a_b_d_n_c")

    assert expand_task_tokens(task.tokens, n=3) == ["A", "B", "D", "D", "D", "C"]
    assert task.successful_events(n=2) == ("A", "B", "D", "D", "C")


def test_validate_task_suite_rejects_duplicate_keys(monkeypatch) -> None:
    duplicate_tasks = (
        LetterTaskSpec(0, "duplicate", "one", ("A", "B", "C"), ""),
        LetterTaskSpec(1, "duplicate", "two", ("A", "B", CountToken("D")), ""),
    )
    monkeypatch.setitem(
        __import__("envs.multitask_letter_env.tasks", fromlist=["TASK_SUITES"]).TASK_SUITES,
        "bad_duplicate",
        duplicate_tasks,
    )

    with pytest.raises(ValueError, match="unique"):
        validate_task_suite("bad_duplicate")


def test_validate_task_suite_rejects_non_contiguous_ids(monkeypatch) -> None:
    bad_tasks = (
        LetterTaskSpec(0, "task0", "zero", ("A", "B", "C"), ""),
        LetterTaskSpec(2, "task2", "two", ("A", "B", CountToken("D")), ""),
    )
    monkeypatch.setitem(
        __import__("envs.multitask_letter_env.tasks", fromlist=["TASK_SUITES"]).TASK_SUITES,
        "bad_ids",
        bad_tasks,
    )

    with pytest.raises(ValueError, match="contiguous"):
        validate_task_suite("bad_ids")


def test_render_task_spec_contains_counted_stage_and_continuation() -> None:
    task = next(task for task in get_task_suite() if task.key == "a_b_d_n_c")
    spec = render_task_spec(task)

    assert "S1_D = gen(['n'], guarded((var('n') > 0)" in spec
    assert "d_match:eps" in spec
    assert "app(S1_D, [(var('n') - 1)])" in spec
    assert "app(S2_C, [var('n')])" in spec
    assert "S2_C = gen(['n']" in spec
    assert "c_match:eps" in spec


def test_render_task_config_records_task_identity_and_runtime_settings() -> None:
    task = get_task_suite()[0]
    config = render_task_config(task, host="127.0.0.1", port=19000, max_episode_steps=123)

    assert f"env_name: multitask-letter-env-{task.key}" in config
    assert "host: 127.0.0.1" in config
    assert "port: 19000" in config
    assert "max_episode_steps: 123" in config
