"""不使用 eval 的布尔表达式与真值表工具。"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

from pydantic import Field

from ie_agent.contracts import StrictModel
from ie_agent.tools.common import ToolComputation

_TOKEN = re.compile(r"\s*(<->|->|[()!&|^]|[A-Za-z][A-Za-z0-9_]*|[01])")
_KEYWORDS = {"not", "and", "or", "xor"}


class TruthTableInput(StrictModel):
    expression: str = Field(min_length=1, max_length=200)


@dataclass(frozen=True)
class Node:
    operator: str
    left: Node | str | bool
    right: Node | str | bool | None = None


class BooleanParser:
    """递归下降解析器，限定可接受的逻辑符号。"""

    def __init__(self, expression: str) -> None:
        self.tokens = self._tokenize(expression)
        self.position = 0
        self.variables: list[str] = []

    def parse(self) -> Node | str | bool:
        node = self._equivalence()
        if self.position != len(self.tokens):
            raise ValueError(f"表达式末尾存在无法解析的符号：{self._peek()}")
        if len(self.variables) > 6:
            raise ValueError("真值表最多支持 6 个变量")
        return node

    @staticmethod
    def _tokenize(expression: str) -> list[str]:
        tokens: list[str] = []
        position = 0
        while position < len(expression):
            match = _TOKEN.match(expression, position)
            if match is None:
                if expression[position:].strip() == "":
                    break
                raise ValueError(f"不支持的逻辑符号：{expression[position]}")
            token = match.group(1)
            tokens.append(token.lower() if token.lower() in _KEYWORDS else token)
            position = match.end()
        if not tokens:
            raise ValueError("逻辑表达式不能为空")
        return tokens

    def _equivalence(self) -> Node | str | bool:
        node = self._implication()
        while self._accept("<->"):
            node = Node("equivalence", node, self._implication())
        return node

    def _implication(self) -> Node | str | bool:
        node = self._or()
        if self._accept("->"):
            node = Node("implication", node, self._implication())
        return node

    def _or(self) -> Node | str | bool:
        node = self._xor()
        while self._accept("or", "|"):
            node = Node("or", node, self._xor())
        return node

    def _xor(self) -> Node | str | bool:
        node = self._and()
        while self._accept("xor", "^"):
            node = Node("xor", node, self._and())
        return node

    def _and(self) -> Node | str | bool:
        node = self._not()
        while self._accept("and", "&"):
            node = Node("and", node, self._not())
        return node

    def _not(self) -> Node | str | bool:
        if self._accept("not", "!"):
            return Node("not", self._not())
        return self._primary()

    def _primary(self) -> Node | str | bool:
        if self._accept("("):
            node = self._equivalence()
            if not self._accept(")"):
                raise ValueError("缺少右括号")
            return node
        token = self._peek()
        if token is None:
            raise ValueError("表达式不完整")
        self.position += 1
        if token in {"0", "1"}:
            return token == "1"
        if token in _KEYWORDS or token in {"<->", "->", ")", "&", "|", "^", "!"}:
            raise ValueError(f"运算符位置错误：{token}")
        if token not in self.variables:
            self.variables.append(token)
        return token

    def _accept(self, *values: str) -> bool:
        if self._peek() in values:
            self.position += 1
            return True
        return False

    def _peek(self) -> str | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None


def truth_table(data: TruthTableInput) -> ToolComputation:
    """枚举变量取值并计算表达式。"""

    parser = BooleanParser(data.expression)
    tree = parser.parse()
    rows: list[dict[str, object]] = []
    for values in itertools.product((False, True), repeat=len(parser.variables)):
        environment = dict(zip(parser.variables, values, strict=True))
        rows.append(
            {
                "inputs": {name: int(value) for name, value in environment.items()},
                "result": int(_evaluate(tree, environment)),
            }
        )
    return ToolComputation(
        result={
            "expression": " ".join(parser.tokens),
            "variables": parser.variables,
            "rows": rows,
            "row_count": len(rows),
        },
        formula="布尔表达式真值枚举",
        steps=["先用白名单 tokenizer 和语法树解析表达式。", "再枚举变量的 0、1 组合。"],
    )


def _evaluate(node: Node | str | bool, environment: dict[str, bool]) -> bool:
    if isinstance(node, bool):
        return node
    if isinstance(node, str):
        return environment[node]
    left = _evaluate(node.left, environment)
    if node.operator == "not":
        return not left
    assert node.right is not None
    right = _evaluate(node.right, environment)
    operations = {
        "and": left and right,
        "or": left or right,
        "xor": left != right,
        "implication": (not left) or right,
        "equivalence": left == right,
    }
    return operations[node.operator]
