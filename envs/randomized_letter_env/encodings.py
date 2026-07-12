"""Monitor-state encodings for randomized LetterEnv."""

from __future__ import annotations

from pathlib import Path

from envs.letter_env.encodings import build_letter_env_monitor_encoding


def build_randomized_letter_env_monitor_encoding(
    encoding: str,
    *,
    learned_gru_checkpoint: str | Path | None = None,
    learned_graph_checkpoint: str | Path | None = None,
):
    """Build the RML monitor-state encoding used by randomized LetterEnv."""
    return build_letter_env_monitor_encoding(
        encoding,
        learned_gru_checkpoint=learned_gru_checkpoint,
        learned_graph_checkpoint=learned_graph_checkpoint,
    )
