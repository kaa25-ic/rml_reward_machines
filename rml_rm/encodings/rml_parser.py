"""Generic tokenizer and lightweight parser for RML monitor-state strings."""

from __future__ import annotations

from dataclasses import dataclass, field
import re


TOKEN_PATTERN = re.compile(
    r"""
    \s+                             |
    \d+\.\d+|\d+                   |
    [A-Za-z_][A-Za-z_0-9]*         |
    <=|>=|==|!=|->|\\/             |
    [()[\]{},:;*+\-/=<>@]
    """,
    re.VERBOSE,
)

OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}"}
CALL_GROUPS = {"(": "CALL", "[": "LIST", "{": "SET"}
INFIX_OPERATORS = {"*", "\\/", "+", "-", "/", "=", "<", ">", "<=", ">=", "==", "!=", "->", ":", "@"}


@dataclass(frozen=True)
class RMLToken:
    kind: str
    value: str
    start: int
    end: int


@dataclass
class RMLNode:
    kind: str
    value: str | None = None
    children: list["RMLNode"] = field(default_factory=list)


def tokenize_rml(monitor_state: str) -> list[RMLToken]:
    """Tokenize an RML monitor-state string."""
    text = str(monitor_state)
    tokens: list[RMLToken] = []
    position = 0
    while position < len(text):
        match = TOKEN_PATTERN.match(text, position)
        if match is None:
            snippet = text[position : position + 32]
            raise ValueError(f"Unable to tokenize RML near position {position}: {snippet!r}")
        value = match.group(0)
        start, end = match.span()
        position = end
        if not value.isspace():
            tokens.append(RMLToken(_token_kind(value), value, start, end))
    return tokens


def parse_rml(monitor_state: str) -> RMLNode:
    """Parse an RML residual monitor-state string into a generic AST."""
    normalized = str(monitor_state).strip()
    if normalized == "1":
        return RMLNode("TERMINAL_SUCCESS", "1")
    if normalized == "false_verdict":
        return RMLNode("TERMINAL_FAILURE", "false_verdict")
    if normalized == "":
        return RMLNode("EMPTY")
    parser = _RMLParser(tokenize_rml(normalized))
    return parser.parse()


def _token_kind(value: str) -> str:
    if re.fullmatch(r"\d+\.\d+|\d+", value):
        return "NUMBER"
    if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", value):
        return "IDENTIFIER"
    if value in OPEN_TO_CLOSE or value in OPEN_TO_CLOSE.values():
        return "BRACKET"
    if value in {",", ";"}:
        return "SEPARATOR"
    return "OPERATOR"


class _RMLParser:
    def __init__(self, tokens: list[RMLToken]) -> None:
        self.tokens = tokens
        self.position = 0

    def parse(self) -> RMLNode:
        children = self._parse_sequence(stop_values=set())
        if len(children) == 1:
            return children[0]
        return RMLNode("ROOT", children=children)

    def _parse_sequence(self, stop_values: set[str]) -> list[RMLNode]:
        nodes: list[RMLNode] = []
        pending_operator: str | None = None
        while self.position < len(self.tokens):
            token = self._peek()
            if token.value in stop_values:
                break
            if token.value in {",", ";"}:
                self._advance()
                continue
            if token.value in INFIX_OPERATORS:
                pending_operator = self._advance().value
                continue
            node = self._parse_atom()
            if pending_operator is not None and nodes:
                left = nodes.pop()
                node = RMLNode("INFIX", pending_operator, [left, node])
                pending_operator = None
            elif pending_operator is not None:
                node = RMLNode("PREFIX_OPERATOR", pending_operator, [node])
                pending_operator = None
            nodes.append(node)
        if pending_operator is not None:
            nodes.append(RMLNode("OPERATOR", pending_operator))
        return nodes

    def _parse_atom(self) -> RMLNode:
        token = self._advance()
        if token.value in OPEN_TO_CLOSE:
            return self._parse_group(token.value)
        if token.kind == "IDENTIFIER":
            identifier = RMLNode("IDENTIFIER", token.value)
            if self._peek_value() in CALL_GROUPS:
                opener = self._advance().value
                return self._parse_group(opener, callee=identifier)
            return identifier
        if token.kind == "NUMBER":
            return RMLNode("NUMBER", token.value)
        return RMLNode(token.kind, token.value)

    def _parse_group(self, opener: str, callee: RMLNode | None = None) -> RMLNode:
        closer = OPEN_TO_CLOSE[opener]
        items: list[RMLNode] = []
        current: list[RMLNode] = []
        while self.position < len(self.tokens):
            if self._peek_value() == closer:
                self._advance()
                break
            if self._peek_value() in {",", ";"}:
                self._advance()
                items.append(_collapse_nodes(current))
                current = []
                continue
            current.extend(self._parse_sequence(stop_values={closer, ",", ";"}))
        if current or not items:
            items.append(_collapse_nodes(current))
        if callee is not None:
            return RMLNode(CALL_GROUPS[opener], children=[callee, *items])
        return RMLNode("GROUP" if opener == "(" else CALL_GROUPS[opener], opener + closer, items)

    def _peek(self) -> RMLToken:
        return self.tokens[self.position]

    def _peek_value(self) -> str | None:
        if self.position >= len(self.tokens):
            return None
        return self.tokens[self.position].value

    def _advance(self) -> RMLToken:
        token = self.tokens[self.position]
        self.position += 1
        return token


def _collapse_nodes(nodes: list[RMLNode]) -> RMLNode:
    if not nodes:
        return RMLNode("EMPTY")
    if len(nodes) == 1:
        return nodes[0]
    return RMLNode("SEQUENCE", children=nodes)
