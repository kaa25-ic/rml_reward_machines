"""RML-based multi-task LetterEnv."""

from envs.multitask_letter_env.tasks import CountToken, LetterTaskSpec, get_task_suite

__all__ = [
    "CountToken",
    "LetterTaskSpec",
    "MultiTaskLetterEnv",
    "MultiTaskLetterEnvConfig",
    "get_task_suite",
]


def __getattr__(name: str):
    if name in {"MultiTaskLetterEnv", "MultiTaskLetterEnvConfig"}:
        from envs.multitask_letter_env.env import MultiTaskLetterEnv, MultiTaskLetterEnvConfig

        return {
            "MultiTaskLetterEnv": MultiTaskLetterEnv,
            "MultiTaskLetterEnvConfig": MultiTaskLetterEnvConfig,
        }[name]
    raise AttributeError(name)
