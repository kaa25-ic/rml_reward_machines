"""Sequence encoders for raw RML monitor-state strings."""

from __future__ import annotations

import re
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


PAD = "<PAD>"
UNK = "<UNK>"
NUM = "<NUM>"

TOKEN_PATTERN = re.compile(
    r"""
    [A-Za-z_][A-Za-z_0-9]*      |
    \d+\.\d+|\d+                |
    <=|>=|==|!=|->              |
    [()<>+\-*/=|;:,{}[\]]
    """,
    re.VERBOSE,
)


def tokenize_monitor_state(monitor_state: str) -> list[str]:
    """Tokenize an RML monitor-state string."""
    return TOKEN_PATTERN.findall(str(monitor_state))


def normalize_token(token: str) -> str:
    """Replace numeric literals with a shared token."""
    if re.fullmatch(r"\d+\.\d+|\d+", token):
        return NUM
    return token


class MonitorVocab:
    """Token vocabulary for RML monitor-state strings."""

    def __init__(self, *, normalize_numbers: bool = True) -> None:
        self.normalize_numbers = bool(normalize_numbers)
        self.token_to_id = {PAD: 0, UNK: 1, NUM: 2}
        self.id_to_token = [PAD, UNK, NUM]
        self._encode_cache: dict[tuple[str, int], tuple[list[int], int]] = {}

    def __len__(self) -> int:
        return len(self.id_to_token)

    def build(self, monitor_strings: list[str]) -> None:
        """Build the vocabulary from monitor-state strings."""
        self._encode_cache.clear()
        for monitor_string in monitor_strings:
            for token in tokenize_monitor_state(monitor_string):
                prepared = self._prepare_token(token)
                if prepared not in self.token_to_id:
                    self.token_to_id[prepared] = len(self.id_to_token)
                    self.id_to_token.append(prepared)

    def encode(self, monitor_state: str, *, max_len: int) -> tuple[list[int], int]:
        cache_key = (str(monitor_state), int(max_len))
        cached = self._encode_cache.get(cache_key)
        if cached is not None:
            token_ids, length = cached
            return token_ids.copy(), length

        tokens = [self._prepare_token(token) for token in tokenize_monitor_state(monitor_state)]
        if not tokens:
            tokens = [UNK]
        tokens = tokens[:max_len]
        length = max(1, len(tokens))
        token_ids = [self.token_to_id.get(token, self.token_to_id[UNK]) for token in tokens]
        token_ids.extend([self.token_to_id[PAD]] * (max_len - len(token_ids)))
        self._encode_cache[cache_key] = (token_ids.copy(), length)
        return token_ids, length

    def load_mapping(self, *, token_to_id: dict[str, int], id_to_token: list[str]) -> None:
        """Replace the vocabulary with a saved mapping."""
        self.token_to_id = {str(key): int(value) for key, value in token_to_id.items()}
        self.id_to_token = [str(token) for token in id_to_token]
        self._encode_cache.clear()

    def _prepare_token(self, token: str) -> str:
        if self.normalize_numbers:
            return normalize_token(token)
        return token


class RMLSequenceEncoder(nn.Module):
    """Encode tokenized RML monitor strings with a bidirectional GRU."""

    def __init__(
        self,
        *,
        vocab_size: int,
        token_dim: int = 32,
        hidden_dim: int = 64,
        output_dim: int = 64,
        pad_idx: int = 0,
        projection_activation: str | None = "relu",
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, token_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            input_size=token_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        projection_layers: list[nn.Module] = [nn.Linear(hidden_dim * 2, output_dim)]
        activation = "none" if projection_activation is None else str(projection_activation).lower()
        if activation == "relu":
            projection_layers.append(nn.ReLU())
        elif activation == "tanh":
            projection_layers.append(nn.Tanh())
        elif activation in {"identity", "none"}:
            pass
        else:
            raise ValueError(f"Unsupported projection activation: {projection_activation}")
        self.proj = nn.Sequential(*projection_layers)

    def forward(self, token_ids: torch.LongTensor, lengths: torch.LongTensor) -> torch.Tensor:
        lengths = lengths.clamp(min=1, max=token_ids.shape[1]).cpu()
        embedded = self.embedding(token_ids)
        packed = pack_padded_sequence(
            embedded,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        return self.proj(torch.cat([forward_hidden, backward_hidden], dim=-1))


def load_gru_checkpoint(path: str | Path, *, device: torch.device) -> tuple[RMLSequenceEncoder, MonitorVocab, int]:
    """Load a frozen GRU monitor encoder from a checkpoint."""
    checkpoint = torch.load(Path(path), map_location=device)
    model_config = checkpoint["config"]
    vocab = MonitorVocab(normalize_numbers=bool(checkpoint.get("vocab_normalize_numbers", False)))
    vocab.load_mapping(
        token_to_id=checkpoint["vocab_token_to_id"],
        id_to_token=checkpoint["vocab_id_to_token"],
    )
    encoder = RMLSequenceEncoder(
        vocab_size=len(vocab),
        token_dim=int(model_config.get("token_dim", 32)),
        hidden_dim=int(model_config.get("hidden_dim", 64)),
        output_dim=int(model_config.get("monitor_embedding_dim", 16)),
        projection_activation=model_config.get("projection_activation", "relu"),
    ).to(device)
    encoder.load_state_dict(checkpoint["monitor_encoder_state_dict"])
    encoder.eval()
    max_len = int(model_config.get("max_len", 192))
    return encoder, vocab, max_len
