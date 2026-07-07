"""Convert generic RML ASTs into typed graph data."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from rml_rm.encodings.rml_parser import RMLNode, parse_rml


PARENT_TO_CHILD = "parent_to_child"
CHILD_TO_PARENT = "child_to_parent"
NEXT_SIBLING = "next_sibling"
PREV_SIBLING = "prev_sibling"


@dataclass(frozen=True)
class RMLGraphData:
    """Typed graph representation of one RML monitor state."""

    node_kinds: tuple[str, ...]
    node_values: tuple[str, ...]
    edge_index: np.ndarray
    edge_types: tuple[str, ...]

    @property
    def num_nodes(self) -> int:
        return len(self.node_kinds)

    @property
    def num_edges(self) -> int:
        return len(self.edge_types)


def rml_to_graph(monitor_state: str) -> RMLGraphData:
    """Parse a monitor state and convert it to graph data."""
    return ast_to_graph(parse_rml(monitor_state))


def normalize_generated_variables(monitor_state: str) -> str:
    """Alpha-normalize generated Prolog variable names within one monitor string."""
    variable_ids: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token not in variable_ids:
            variable_ids[token] = f"_v{len(variable_ids)}"
        return variable_ids[token]

    return re.sub(r"_[0-9]+", replace, str(monitor_state))


def ast_to_graph(root: RMLNode) -> RMLGraphData:
    """Convert an RML AST into a bidirectional typed graph."""
    builder = _GraphBuilder()
    builder.add_subtree(root)
    return builder.to_graph()


def build_node_kind_vocab(graphs: list[RMLGraphData]) -> dict[str, int]:
    kinds = sorted({kind for graph in graphs for kind in graph.node_kinds})
    return {"<UNK>": 0, **{kind: index + 1 for index, kind in enumerate(kinds)}}


def build_node_value_vocab(graphs: list[RMLGraphData]) -> dict[str, int]:
    values = sorted({value for graph in graphs for value in graph.node_values})
    return {"<UNK>": 0, **{value: index + 1 for index, value in enumerate(values)}}


def build_edge_type_vocab(graphs: list[RMLGraphData]) -> dict[str, int]:
    kinds = sorted({kind for graph in graphs for kind in graph.edge_types})
    return {"<UNK>": 0, **{kind: index + 1 for index, kind in enumerate(kinds)}}


class _GraphBuilder:
    def __init__(self) -> None:
        self.node_kinds: list[str] = []
        self.node_values: list[str] = []
        self.edges: list[tuple[int, int]] = []
        self.edge_types: list[str] = []

    def add_subtree(self, node: RMLNode, parent_index: int | None = None) -> int:
        node_index = self._add_node(node)
        if parent_index is not None:
            self._add_edge(parent_index, node_index, PARENT_TO_CHILD)
            self._add_edge(node_index, parent_index, CHILD_TO_PARENT)
        previous_child_index: int | None = None
        for child in node.children:
            child_index = self.add_subtree(child, parent_index=node_index)
            if previous_child_index is not None:
                self._add_edge(previous_child_index, child_index, NEXT_SIBLING)
                self._add_edge(child_index, previous_child_index, PREV_SIBLING)
            previous_child_index = child_index
        return node_index

    def to_graph(self) -> RMLGraphData:
        edge_index = np.asarray(self.edges, dtype=np.int64).T if self.edges else np.zeros((2, 0), dtype=np.int64)
        return RMLGraphData(
            node_kinds=tuple(self.node_kinds),
            node_values=tuple(self.node_values),
            edge_index=edge_index,
            edge_types=tuple(self.edge_types),
        )

    def _add_node(self, node: RMLNode) -> int:
        self.node_kinds.append(str(node.kind))
        self.node_values.append("" if node.value is None else str(node.value))
        return len(self.node_kinds) - 1

    def _add_edge(self, source: int, target: int, edge_type: str) -> None:
        self.edges.append((source, target))
        self.edge_types.append(edge_type)
