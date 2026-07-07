"""Frozen pretrained monitor-state encoders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rml_rm.encodings.graph_models import GraphEncoderConfig, RMLGraphBatch, RMLGraphDynamicsPredictor
from rml_rm.encodings.rml_sequence import load_gru_checkpoint
from rml_rm.encodings.rml_graph import normalize_generated_variables, rml_to_graph


class FrozenGRUMonitorStateEncoder:
    """Encode monitor states with a pretrained GRU monitor encoder."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"GRU monitor checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device("cpu")
        self.encoder, self.vocab, self.max_len = load_gru_checkpoint(
            self.checkpoint_path,
            device=self.device,
        )
        self._cache: dict[str, np.ndarray] = {}

    def __call__(self, monitor_state: str) -> np.ndarray:
        monitor_state = str(monitor_state)
        cached = self._cache.get(monitor_state)
        if cached is not None:
            return cached.copy()

        token_ids, length = self.vocab.encode(monitor_state, max_len=self.max_len)
        with torch.no_grad():
            ids_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            length_tensor = torch.tensor([length], dtype=torch.long, device=self.device)
            encoded = (
                self.encoder(ids_tensor, length_tensor)
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        self._cache[monitor_state] = encoded.copy()
        return encoded


class FrozenGraphMonitorStateEncoder:
    """Encode monitor states with a pretrained basic RML graph encoder."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Graph monitor checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device("cpu")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        graph_config = GraphEncoderConfig(**checkpoint["graph_config"])
        if int(graph_config.node_value_embedding_dim) != 0:
            raise ValueError("Only the basic graph encoder without node-value embeddings is supported.")

        self.node_kind_vocab = {str(key): int(value) for key, value in checkpoint["node_kind_vocab"].items()}
        self.node_value_vocab = {str(key): int(value) for key, value in checkpoint["node_value_vocab"].items()}
        self.edge_type_vocab = {str(key): int(value) for key, value in checkpoint["edge_type_vocab"].items()}
        self.event_vocab = {str(key): int(value) for key, value in checkpoint["event_vocab"].items()}
        self.phase_vocab = {str(key): int(value) for key, value in checkpoint["phase_vocab"].items()}
        self.model = RMLGraphDynamicsPredictor(
            num_node_kinds=len(self.node_kind_vocab),
            num_node_values=len(self.node_value_vocab),
            num_edge_types=len(self.edge_type_vocab),
            num_events=len(self.event_vocab),
            graph_config=graph_config,
            event_embedding_dim=int(checkpoint["config"].get("event_embedding_dim", 16)),
            num_phase_labels=len(self.phase_vocab),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self._cache: dict[str, np.ndarray] = {}

    def __call__(self, monitor_state: str) -> np.ndarray:
        monitor_state = normalize_generated_variables(str(monitor_state))
        cached = self._cache.get(monitor_state)
        if cached is not None:
            return cached.copy()

        graph = rml_to_graph(monitor_state)
        batch = RMLGraphBatch.from_graphs(
            [graph],
            node_kind_vocab=self.node_kind_vocab,
            node_value_vocab=self.node_value_vocab,
            edge_type_vocab=self.edge_type_vocab,
            device=self.device,
        )
        with torch.no_grad():
            encoded = (
                self.model(batch, torch.zeros(1, dtype=torch.long, device=self.device))["embedding"]
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        norm = float(np.linalg.norm(encoded))
        if norm > 1e-8:
            encoded = encoded / norm
        self._cache[monitor_state] = encoded.copy()
        return encoded
