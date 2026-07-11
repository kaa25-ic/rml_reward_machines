"""Task definitions for the multi-task LetterEnv."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union


LETTER_ALPHABET = ("A", "B", "C", "D")
CountSymbol = Literal["n"]
TaskToken = Union[str, "CountToken"]


@dataclass(frozen=True)
class CountToken:
    """A repeated letter token whose count is sampled at episode reset."""

    letter: str
    count_symbol: CountSymbol = "n"

    def __post_init__(self) -> None:
        if self.letter not in LETTER_ALPHABET:
            raise ValueError(f"Unsupported LetterEnv letter: {self.letter!r}")

    def describe(self) -> str:
        return f"{self.letter}^{self.count_symbol}"


@dataclass(frozen=True)
class LetterTaskSpec:
    """Symbolic task definition used by the multi-task environment."""

    task_id: int
    key: str
    name: str
    tokens: tuple[TaskToken, ...]
    description: str

    @property
    def expression(self) -> str:
        return " ".join(_describe_token(token) for token in self.tokens)

    def successful_events(self, *, n: int) -> tuple[str, ...]:
        return tuple(expand_task_tokens(self.tokens, n=n))

    def to_jsonable(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "key": self.key,
            "name": self.name,
            "expression": self.expression,
            "description": self.description,
        }


SMALL_V1_TASKS: tuple[LetterTaskSpec, ...] = (
    LetterTaskSpec(
        task_id=0,
        key="a_b_c_d_n",
        name="A B C D^n",
        tokens=("A", "B", "C", CountToken("D")),
        description="Visit A, then B, then C, then visit D n times.",
    ),
    LetterTaskSpec(
        task_id=1,
        key="a_b_d_n_c",
        name="A B D^n C",
        tokens=("A", "B", CountToken("D"), "C"),
        description="Visit A, then B, then visit D n times, then C.",
    ),
    LetterTaskSpec(
        task_id=2,
        key="a_b_c_d_n_c",
        name="A B C D^n C",
        tokens=("A", "B", "C", CountToken("D"), "C"),
        description="Visit A, then B, then C, then visit D n times, then C.",
    ),
    LetterTaskSpec(
        task_id=3,
        key="a_b_d_c_d_n",
        name="A B D C D^n",
        tokens=("A", "B", "D", "C", CountToken("D")),
        description="Visit A, then B, then D, then C, then visit D n times.",
    ),
    LetterTaskSpec(
        task_id=4,
        key="a_b_c_d_c_d_n",
        name="A B C D C D^n",
        tokens=("A", "B", "C", "D", "C", CountToken("D")),
        description="Visit A, then B, then C, then D, then C, then visit D n times.",
    ),
)

TASK_SUITES: dict[str, tuple[LetterTaskSpec, ...]] = {
    "small_v1": SMALL_V1_TASKS,
}


def get_task_suite(name: str = "small_v1") -> tuple[LetterTaskSpec, ...]:
    """Return a frozen task suite."""
    try:
        return TASK_SUITES[name]
    except KeyError as exc:
        known = ", ".join(sorted(TASK_SUITES))
        raise ValueError(f"Unknown task suite {name!r}. Known suites: {known}") from exc


def expand_task_tokens(tokens: tuple[TaskToken, ...], *, n: int) -> list[str]:
    """Expand a symbolic task into the target event sequence for one episode."""
    if n < 1:
        raise ValueError("n must be at least 1.")
    events: list[str] = []
    for token in tokens:
        if isinstance(token, CountToken):
            events.extend([token.letter] * n)
        else:
            _validate_letter(token)
            events.append(token)
    return events


def validate_task_suite(name: str = "small_v1", *, max_n: int = 5) -> None:
    """Validate task identifiers and representative expansions."""
    tasks = get_task_suite(name)
    task_ids = [task.task_id for task in tasks]
    keys = [task.key for task in tasks]
    if sorted(task_ids) != list(range(len(tasks))):
        raise ValueError("Task IDs must be contiguous from zero.")
    if len(set(keys)) != len(keys):
        raise ValueError("Task keys must be unique.")
    for task in tasks:
        if len(task.tokens) < 2 or task.tokens[0] != "A" or task.tokens[1] != "B":
            raise ValueError(f"Task {task.key!r} must start with A B.")
        for n in range(1, max_n + 1):
            for letter in task.successful_events(n=n):
                _validate_letter(letter)


def _describe_token(token: TaskToken) -> str:
    if isinstance(token, CountToken):
        return token.describe()
    _validate_letter(token)
    return token


def _validate_letter(letter: str) -> None:
    if letter not in LETTER_ALPHABET:
        raise ValueError(f"Unsupported LetterEnv letter: {letter!r}")
