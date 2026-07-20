"""Parser logic for deterministic Europe PMC query validation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence


class TokenType(str, Enum):
    WORD = "WORD"
    QUOTED = "QUOTED"
    SINGLE_QUOTED = "SINGLE_QUOTED"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    LBRACKET = "LBRACKET"
    RBRACKET = "RBRACKET"
    COLON = "COLON"
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    EOF = "EOF"


@dataclass(frozen=True)
class Token:
    type: TokenType
    text: str
    start: int
    end: int


@dataclass
class FieldValue:
    raw: str
    kind: str
    tokens: List[Token] = field(default_factory=list)
    inner_ast: Optional["QueryNode"] = None


@dataclass
class QueryNode:
    start: int
    end: int


@dataclass
class TermNode(QueryNode):
    raw: str
    kind: str


@dataclass
class FieldClauseNode(QueryNode):
    field: str
    raw_field: str
    value: FieldValue


@dataclass
class NotNode(QueryNode):
    operand: QueryNode


@dataclass
class BooleanNode(QueryNode):
    operator: str
    operands: List[QueryNode]


@dataclass
class GroupNode(QueryNode):
    expression: QueryNode


@dataclass
class ParseResult:
    root: Optional[QueryNode]
    tokens: List[Token]
    errors: List[str]
    field_clauses: List[FieldClauseNode]


def tokenize_query(query: str) -> List[Token]:
    tokenizer = QueryTokenizer(query)
    return tokenizer.tokenize()


def parse_query(query: str) -> ParseResult:
    tokens = tokenize_query(query)
    parser = QueryParser(query, tokens)
    return parser.parse()


class QueryTokenizer:
    def __init__(self, query: str):
        self.query = query
        self.length = len(query)
        self.pos = 0

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []

        while self.pos < self.length:
            char = self.query[self.pos]

            if char.isspace():
                self.pos += 1
                continue

            if char == "(":
                tokens.append(Token(TokenType.LPAREN, char, self.pos, self.pos + 1))
                self.pos += 1
                continue

            if char == ")":
                tokens.append(Token(TokenType.RPAREN, char, self.pos, self.pos + 1))
                self.pos += 1
                continue

            if char == "[":
                tokens.append(Token(TokenType.LBRACKET, char, self.pos, self.pos + 1))
                self.pos += 1
                continue

            if char == "]":
                tokens.append(Token(TokenType.RBRACKET, char, self.pos, self.pos + 1))
                self.pos += 1
                continue

            if char == ":":
                tokens.append(Token(TokenType.COLON, char, self.pos, self.pos + 1))
                self.pos += 1
                continue

            if char == '"':
                tokens.append(self._read_quoted(TokenType.QUOTED, char))
                continue

            if char == "'":
                tokens.append(self._read_quoted(TokenType.SINGLE_QUOTED, char))
                continue

            tokens.append(self._read_word())

        tokens.append(Token(TokenType.EOF, "", self.length, self.length))
        return tokens

    def _read_quoted(self, token_type: TokenType, delimiter: str) -> Token:
        start = self.pos
        self.pos += 1

        while self.pos < self.length:
            char = self.query[self.pos]
            if char == delimiter:
                if (
                    delimiter == '"'
                    and self.pos + 1 < self.length
                    and self.query[self.pos + 1] == '"'
                ):
                    self.pos += 2
                    continue
                self.pos += 1
                break
            self.pos += 1

        return Token(token_type, self.query[start : self.pos], start, self.pos)

    def _read_word(self) -> Token:
        start = self.pos
        while (
            self.pos < self.length
            and self.query[self.pos] not in "()[]:\"'"
            and not self.query[self.pos].isspace()
        ):
            self.pos += 1

        text = self.query[start : self.pos]
        token_type = self._word_type(text)
        return Token(token_type, text, start, self.pos)

    @staticmethod
    def _word_type(text: str) -> TokenType:
        if text == "AND":
            return TokenType.AND
        if text == "OR":
            return TokenType.OR
        if text == "NOT":
            return TokenType.NOT
        return TokenType.WORD


class QueryParser:
    def __init__(self, query: str, tokens: Sequence[Token]):
        self.query = query
        self.tokens = list(tokens)
        self.index = 0
        self.errors: List[str] = []
        self.field_clauses: List[FieldClauseNode] = []

    def parse(self) -> ParseResult:
        root = None

        if self._current().type != TokenType.EOF:
            root = self._parse_expression(stop_tokens={TokenType.EOF})

        if self._current().type != TokenType.EOF:
            self.errors.append(f"Unexpected token '{self._current().text}'")

        return ParseResult(
            root=root,
            tokens=self.tokens,
            errors=self.errors,
            field_clauses=self.field_clauses,
        )

    def _parse_expression(self, stop_tokens: set[TokenType]) -> Optional[QueryNode]:
        left = self._parse_conjunction(stop_tokens)

        while self._current().type == TokenType.OR:
            operator = self._advance()
            if self._current().type in stop_tokens or self._current().type in {
                TokenType.OR,
                TokenType.AND,
                TokenType.RPAREN,
            }:
                self.errors.append(
                    f"Operator '{operator.text}' is missing a right-hand operand"
                )
                return left

            right = self._parse_conjunction(stop_tokens)
            if left is None or right is None:
                return left or right
            left = BooleanNode(
                start=left.start,
                end=right.end,
                operator="OR",
                operands=[left, right],
            )

        return left

    def _parse_conjunction(self, stop_tokens: set[TokenType]) -> Optional[QueryNode]:
        first = self._parse_unary(stop_tokens)
        if first is None:
            return None

        operands = [first]

        while True:
            current = self._current().type

            if current == TokenType.AND:
                operator = self._advance()
                if self._current().type in stop_tokens or self._current().type in {
                    TokenType.AND,
                    TokenType.OR,
                    TokenType.RPAREN,
                }:
                    self.errors.append(
                        f"Operator '{operator.text}' is missing a right-hand operand"
                    )
                    break

                operand = self._parse_unary(stop_tokens)
                if operand is None:
                    break
                operands.append(operand)
                continue

            if current == TokenType.NOT or self._starts_operand(
                self._current(), stop_tokens
            ):
                operand = self._parse_unary(stop_tokens)
                if operand is None:
                    break
                operands.append(operand)
                continue

            break

        if len(operands) == 1:
            return operands[0]

        return BooleanNode(
            start=operands[0].start,
            end=operands[-1].end,
            operator="AND",
            operands=operands,
        )

    def _parse_unary(self, stop_tokens: set[TokenType]) -> Optional[QueryNode]:
        current = self._current()

        if current.type == TokenType.NOT:
            operator = self._advance()
            if self._current().type in stop_tokens or self._current().type in {
                TokenType.AND,
                TokenType.OR,
                TokenType.RPAREN,
                TokenType.NOT,
            }:
                self.errors.append("Illegal NOT placement")
                return None

            operand = self._parse_unary(stop_tokens)
            if operand is None:
                self.errors.append(f"Operator '{operator.text}' is missing an operand")
                return None

            return NotNode(start=operator.start, end=operand.end, operand=operand)

        return self._parse_primary(stop_tokens)

    def _parse_primary(self, stop_tokens: set[TokenType]) -> Optional[QueryNode]:
        current = self._current()

        if current.type == TokenType.LPAREN:
            opening = self._advance()
            if self._current().type == TokenType.RPAREN:
                self.errors.append("Empty group '()' is not allowed")
                closing = self._advance()
                return GroupNode(
                    start=opening.start,
                    end=closing.end,
                    expression=TermNode(
                        raw="", kind="empty", start=opening.start, end=closing.end
                    ),
                )

            expression = self._parse_expression(stop_tokens | {TokenType.RPAREN})
            if self._current().type != TokenType.RPAREN:
                self.errors.append("Unmatched parentheses in query")
                end = expression.end if expression else opening.end
                return GroupNode(
                    start=opening.start,
                    end=end,
                    expression=expression
                    or TermNode(raw="", kind="invalid", start=opening.start, end=end),
                )

            closing = self._advance()
            if expression is None:
                self.errors.append("Grouped expression is empty")
                expression = TermNode(
                    raw="", kind="invalid", start=opening.start, end=closing.end
                )

            return GroupNode(
                start=opening.start, end=closing.end, expression=expression
            )

        if current.type == TokenType.WORD and self._peek().type == TokenType.COLON:
            return self._parse_field_clause()

        if current.type in {TokenType.WORD, TokenType.QUOTED, TokenType.SINGLE_QUOTED}:
            token = self._advance()
            kind = {
                TokenType.WORD: "bare",
                TokenType.QUOTED: "quoted",
                TokenType.SINGLE_QUOTED: "single_quoted",
            }[token.type]
            return TermNode(raw=token.text, kind=kind, start=token.start, end=token.end)

        if current.type == TokenType.RPAREN and TokenType.RPAREN not in stop_tokens:
            self.errors.append("Unexpected closing parenthesis")
            self._advance()
            return None

        if current.type == TokenType.RBRACKET:
            self.errors.append("Unexpected closing bracket")
            self._advance()
            return None

        if current.type != TokenType.EOF:
            self.errors.append(f"Unexpected token '{current.text}'")
            self._advance()

        return None

    def _parse_field_clause(self) -> FieldClauseNode:
        field_token = self._advance()
        colon_token = self._advance()

        value = self._parse_field_value()
        node = FieldClauseNode(
            field=field_token.text.upper(),
            raw_field=field_token.text,
            value=value,
            start=field_token.start,
            end=max(
                colon_token.end,
                colon_token.end if not value.tokens else value.tokens[-1].end,
            ),
        )
        self.field_clauses.append(node)
        return node

    def _parse_field_value(self) -> FieldValue:
        current = self._current()

        if current.type in {
            TokenType.EOF,
            TokenType.AND,
            TokenType.OR,
            TokenType.RPAREN,
        }:
            self.errors.append("Field tag is missing a value")
            return FieldValue(raw="", kind="missing")

        if current.type == TokenType.QUOTED:
            token = self._advance()
            return FieldValue(raw=token.text, kind="quoted", tokens=[token])

        if current.type == TokenType.SINGLE_QUOTED:
            token = self._advance()
            return FieldValue(raw=token.text, kind="single_quoted", tokens=[token])

        if current.type == TokenType.LBRACKET:
            return self._parse_bracket_value()

        if current.type == TokenType.LPAREN:
            return self._parse_parenthesized_value()

        return self._parse_bare_value()

    def _parse_bracket_value(self) -> FieldValue:
        start = self._advance()
        tokens = [start]
        depth = 1

        while self._current().type != TokenType.EOF:
            token = self._advance()
            tokens.append(token)

            if token.type == TokenType.LBRACKET:
                depth += 1
            elif token.type == TokenType.RBRACKET:
                depth -= 1
                if depth == 0:
                    raw = self.query[start.start : token.end]
                    return FieldValue(raw=raw, kind="range", tokens=tokens)

        self.errors.append("Unmatched brackets in query")
        raw = self.query[start.start : tokens[-1].end] if tokens else ""
        return FieldValue(raw=raw, kind="range", tokens=tokens)

    def _parse_parenthesized_value(self) -> FieldValue:
        start = self._advance()
        tokens = [start]
        depth = 1

        while self._current().type != TokenType.EOF:
            token = self._advance()
            tokens.append(token)

            if token.type == TokenType.LPAREN:
                depth += 1
            elif token.type == TokenType.RPAREN:
                depth -= 1
                if depth == 0:
                    raw = self.query[start.start : token.end]
                    inner_tokens = tokens[1:-1] + [
                        Token(TokenType.EOF, "", token.start, token.start)
                    ]
                    inner_ast = None
                    if any(
                        t.type
                        in {
                            TokenType.AND,
                            TokenType.OR,
                            TokenType.NOT,
                            TokenType.LPAREN,
                        }
                        for t in inner_tokens[:-1]
                    ):
                        child_parser = QueryParser(self.query, inner_tokens)
                        child_result = child_parser.parse()
                        self.errors.extend(child_result.errors)
                        inner_ast = child_result.root
                        self.field_clauses.extend(child_result.field_clauses)
                    return FieldValue(
                        raw=raw,
                        kind="parenthesized",
                        tokens=tokens,
                        inner_ast=inner_ast,
                    )

        self.errors.append("Unmatched parentheses in query")
        raw = self.query[start.start : tokens[-1].end] if tokens else ""
        return FieldValue(raw=raw, kind="parenthesized", tokens=tokens)

    def _parse_bare_value(self) -> FieldValue:
        tokens: List[Token] = []
        start = self._current().start

        while True:
            current = self._current()
            if current.type in {TokenType.EOF, TokenType.RPAREN, TokenType.RBRACKET}:
                break
            if current.type in {TokenType.AND, TokenType.OR}:
                break
            if current.type == TokenType.NOT and tokens:
                break
            if current.type == TokenType.LPAREN:
                break
            tokens.append(self._advance())

        end = tokens[-1].end if tokens else start
        raw = self.query[start:end].strip()
        return FieldValue(raw=raw, kind="bare", tokens=tokens)

    def _starts_operand(self, token: Token, stop_tokens: set[TokenType]) -> bool:
        if token.type in stop_tokens:
            return False
        if token.type in {
            TokenType.WORD,
            TokenType.QUOTED,
            TokenType.SINGLE_QUOTED,
            TokenType.LPAREN,
        }:
            return True
        return False

    def _current(self) -> Token:
        return self.tokens[self.index]

    def _peek(self) -> Token:
        next_index = min(self.index + 1, len(self.tokens) - 1)
        return self.tokens[next_index]

    def _advance(self) -> Token:
        token = self.tokens[self.index]
        if self.index < len(self.tokens) - 1:
            self.index += 1
        return token
