"""Relation-aware neural encoders for generic RML graphs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from rml_rm.encodings.rml_graph import (
    RMLGraphData,
    build_edge_type_vocab,
    build_node_kind_vocab,
    build_node_value_vocab,
)


TRANSITION_LABELS = {"neutral": 0, "progress": 1, "failure": 2, "success": 3}


@dataclass(frozen=True)
class GraphEncoderConfig:
    """Configuration for a generic RML graph neural encoder."""

    node_embedding_dim: int = 32
    node_value_embedding_dim: int = 0
    node_value_dropout: float = 0.0
    hidden_dim: int = 64
    output_dim: int = 32
    num_layers: int = 2
    dropout: float = 0.0
    output_layer_norm: bool = False
    output_l2_normalize: bool = False


@dataclass(frozen=True)
class RMLGraphBatch:
    """Batched tensor representation of multiple RML graphs."""

    node_kind_ids: torch.LongTensor
    node_value_ids: torch.LongTensor
    edge_index: torch.LongTensor
    edge_type_ids: torch.LongTensor
    graph_ids: torch.LongTensor
    num_graphs: int
    node_kind_vocab: dict[str, int]
    node_value_vocab: dict[str, int]
    edge_type_vocab: dict[str, int]

    @classmethod
    def from_graphs(
        cls,
        graphs: list[RMLGraphData],
        *,
        node_kind_vocab: dict[str, int] | None = None,
        node_value_vocab: dict[str, int] | None = None,
        edge_type_vocab: dict[str, int] | None = None,
        device: str | torch.device = "cpu",
    ) -> "RMLGraphBatch":
        if not graphs:
            raise ValueError("RMLGraphBatch.from_graphs requires at least one graph.")
        node_kind_vocab = node_kind_vocab or build_node_kind_vocab(graphs)
        node_value_vocab = node_value_vocab or build_node_value_vocab(graphs)
        edge_type_vocab = edge_type_vocab or build_edge_type_vocab(graphs)

        node_kind_ids: list[int] = []
        node_value_ids: list[int] = []
        graph_ids: list[int] = []
        edge_sources: list[int] = []
        edge_targets: list[int] = []
        edge_type_ids: list[int] = []
        node_offset = 0
        unk_node = node_kind_vocab.get("<UNK>", 0)
        unk_value = node_value_vocab.get("<UNK>", 0)
        unk_edge = edge_type_vocab.get("<UNK>", 0)
        for graph_index, graph in enumerate(graphs):
            node_kind_ids.extend(node_kind_vocab.get(kind, unk_node) for kind in graph.node_kinds)
            node_value_ids.extend(node_value_vocab.get(value, unk_value) for value in graph.node_values)
            graph_ids.extend([graph_index] * graph.num_nodes)
            for edge_index, edge_type in enumerate(graph.edge_types):
                edge_sources.append(int(graph.edge_index[0, edge_index]) + node_offset)
                edge_targets.append(int(graph.edge_index[1, edge_index]) + node_offset)
                edge_type_ids.append(edge_type_vocab.get(edge_type, unk_edge))
            node_offset += graph.num_nodes

        if edge_sources:
            edge_index_tensor = torch.tensor([edge_sources, edge_targets], dtype=torch.long, device=device)
            edge_type_tensor = torch.tensor(edge_type_ids, dtype=torch.long, device=device)
        else:
            edge_index_tensor = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_type_tensor = torch.zeros((0,), dtype=torch.long, device=device)

        return cls(
            node_kind_ids=torch.tensor(node_kind_ids, dtype=torch.long, device=device),
            node_value_ids=torch.tensor(node_value_ids, dtype=torch.long, device=device),
            edge_index=edge_index_tensor,
            edge_type_ids=edge_type_tensor,
            graph_ids=torch.tensor(graph_ids, dtype=torch.long, device=device),
            num_graphs=len(graphs),
            node_kind_vocab=node_kind_vocab,
            node_value_vocab=node_value_vocab,
            edge_type_vocab=edge_type_vocab,
        )


class RMLGraphNeuralEncoder(nn.Module):
    """Small relation-aware GNN for task-agnostic RML graph embeddings."""

    def __init__(
        self,
        *,
        num_node_kinds: int,
        num_edge_types: int,
        num_node_values: int = 1,
        config: GraphEncoderConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or GraphEncoderConfig()
        self.node_embedding = nn.Embedding(num_node_kinds, self.config.node_embedding_dim)
        self.node_value_embedding: nn.Embedding | None = None
        if self.config.node_value_embedding_dim > 0:
            self.node_value_embedding = nn.Embedding(num_node_values, self.config.node_value_embedding_dim)

        input_dim = self.config.node_embedding_dim + self.config.node_value_embedding_dim
        self.layers = nn.ModuleList(
            [
                RelationalGraphConv(
                    input_dim=input_dim if layer_index == 0 else self.config.hidden_dim,
                    output_dim=self.config.hidden_dim,
                    num_edge_types=num_edge_types,
                    dropout=self.config.dropout,
                )
                for layer_index in range(self.config.num_layers)
            ]
        )
        self.projection = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.output_dim),
        )
        self.output_norm = nn.LayerNorm(self.config.output_dim) if self.config.output_layer_norm else nn.Identity()

    def forward(self, batch: RMLGraphBatch) -> torch.Tensor:
        node_features = self.node_embedding(batch.node_kind_ids)
        if self.node_value_embedding is not None:
            node_value_ids = batch.node_value_ids
            if self.training and self.config.node_value_dropout > 0.0:
                keep = torch.rand_like(node_value_ids, dtype=torch.float32) >= self.config.node_value_dropout
                node_value_ids = torch.where(keep, node_value_ids, torch.zeros_like(node_value_ids))
            node_features = torch.cat([node_features, self.node_value_embedding(node_value_ids)], dim=1)
        for layer in self.layers:
            node_features = layer(node_features, batch.edge_index, batch.edge_type_ids)
        embedding = self.output_norm(self.projection(_mean_max_pool(node_features, batch.graph_ids, batch.num_graphs)))
        if self.config.output_l2_normalize:
            embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


class RMLGraphDynamicsPredictor(nn.Module):
    """GNN encoder plus event embedding for monitor-transition prediction."""

    def __init__(
        self,
        *,
        num_node_kinds: int,
        num_node_values: int,
        num_edge_types: int,
        num_events: int,
        graph_config: GraphEncoderConfig,
        event_embedding_dim: int,
        num_phase_labels: int,
        structural_feature_dim: int = 0,
    ) -> None:
        super().__init__()
        self.structural_feature_dim = int(structural_feature_dim)
        self.encoder = RMLGraphNeuralEncoder(
            num_node_kinds=num_node_kinds,
            num_node_values=num_node_values,
            num_edge_types=num_edge_types,
            config=graph_config,
        )
        self.event_embedding = nn.Embedding(num_events, event_embedding_dim)
        graph_embedding_dim = graph_config.output_dim + self.structural_feature_dim
        self.transition_head = nn.Sequential(
            nn.Linear(graph_embedding_dim + event_embedding_dim, graph_config.hidden_dim),
            nn.ReLU(),
            nn.Linear(graph_config.hidden_dim, len(TRANSITION_LABELS)),
        )
        self.phase_head = nn.Sequential(
            nn.Linear(graph_embedding_dim, graph_config.hidden_dim),
            nn.ReLU(),
            nn.Linear(graph_config.hidden_dim, num_phase_labels),
        )

    def forward(
        self,
        graph_batch: RMLGraphBatch,
        event_ids: torch.LongTensor,
        structural_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        graph_embedding = self.encoder(graph_batch)
        if self.structural_feature_dim > 0:
            if structural_features is None:
                structural_features = torch.zeros(
                    (graph_embedding.shape[0], self.structural_feature_dim),
                    dtype=graph_embedding.dtype,
                    device=graph_embedding.device,
                )
            graph_embedding = torch.cat([graph_embedding, structural_features.to(graph_embedding.device)], dim=1)
        event_embedding = self.event_embedding(event_ids)
        return {
            "embedding": graph_embedding,
            "transition_logits": self.transition_head(torch.cat([graph_embedding, event_embedding], dim=1)),
            "phase_logits": self.phase_head(graph_embedding),
        }


class RelationalGraphConv(nn.Module):
    """Minimal dependency-free relation-aware message passing layer."""

    def __init__(self, *, input_dim: int, output_dim: int, num_edge_types: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.relation_linears = nn.ModuleList(nn.Linear(input_dim, output_dim, bias=False) for _ in range(num_edge_types))
        self.self_linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_type_ids: torch.LongTensor,
    ) -> torch.Tensor:
        aggregate = torch.zeros(
            node_features.shape[0],
            self.self_linear.out_features,
            dtype=node_features.dtype,
            device=node_features.device,
        )
        if edge_index.numel() > 0:
            source_indices = edge_index[0]
            target_indices = edge_index[1]
            for relation_id, relation_linear in enumerate(self.relation_linears):
                mask = edge_type_ids == relation_id
                if bool(mask.any()):
                    messages = relation_linear(node_features[source_indices[mask]])
                    aggregate.index_add_(0, target_indices[mask], messages)
        updated = self.self_linear(node_features) + aggregate
        return self.dropout(self.activation(self.norm(updated)))


def _mean_max_pool(node_features: torch.Tensor, graph_ids: torch.LongTensor, num_graphs: int) -> torch.Tensor:
    hidden_dim = node_features.shape[1]
    mean_pool = torch.zeros(num_graphs, hidden_dim, dtype=node_features.dtype, device=node_features.device)
    counts = torch.zeros(num_graphs, 1, dtype=node_features.dtype, device=node_features.device)
    mean_pool.index_add_(0, graph_ids, node_features)
    counts.index_add_(0, graph_ids, torch.ones_like(node_features[:, :1]))
    mean_pool = mean_pool / counts.clamp(min=1.0)

    max_rows: list[torch.Tensor] = []
    for graph_index in range(num_graphs):
        graph_node_features = node_features[graph_ids == graph_index]
        if graph_node_features.numel() == 0:
            max_rows.append(torch.zeros(hidden_dim, dtype=node_features.dtype, device=node_features.device))
        else:
            max_rows.append(graph_node_features.max(dim=0).values)
    return torch.cat([mean_pool, torch.stack(max_rows, dim=0)], dim=1)
