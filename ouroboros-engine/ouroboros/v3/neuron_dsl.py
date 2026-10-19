"""Validated scalar-expression trees evaluated elementwise by PyTorch.

All leaves have the shape of the preactivation. There are no imports, attributes,
loops, allocation operators, eval or exec in this language.
"""
from dataclasses import dataclass
import json
import math

import torch

from . import fingerprint

ARITY = {"add": 2, "sub": 2, "mul": 2, "mix": 3, "tanh": 1,
         "sigmoid": 1, "sin": 1, "softsign": 1, "neg": 1}
VARIABLES = {"x", "h", "m", "a", "p"}


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple = ()
    value: float | None = None

    def to_dict(self):
        result = {"op": self.op}
        if self.args:
            result["args"] = [item.to_dict() for item in self.args]
        if self.value is not None:
            result["value"] = self.value
        return result

    @property
    def identity(self):
        return fingerprint(self.to_dict())

    @property
    def size(self):
        return 1 + sum(item.size for item in self.args)


def parse(data, depth=0, counter=None):
    if isinstance(data, str):
        if len(data) > 32768:
            raise ValueError("Expression too large")
        data = json.loads(data)
    counter = [0] if counter is None else counter
    counter[0] += 1
    if depth > 8 or counter[0] > 127 or not isinstance(data, dict):
        raise ValueError("Expression budget exceeded")
    if set(data) - {"op", "args", "value"}:
        raise ValueError("Unknown expression fields")
    op, arguments = data.get("op"), data.get("args", [])
    if not isinstance(op, str):
        raise ValueError("Operator must be a name")
    if not isinstance(arguments, list):
        raise ValueError("Arguments must be a list")
    if op in VARIABLES or op == "const":
        if arguments:
            raise ValueError("Leaf cannot have arguments")
        value = data.get("value")
        if op == "const":
            if type(value) not in (float, int) or not math.isfinite(value) or abs(value) > 4:
                raise ValueError("Constant outside bounded range")
            return Expr(op, value=float(value))
        if value is not None:
            raise ValueError("Variable cannot have a constant")
        return Expr(op)
    if op not in ARITY or len(arguments) != ARITY[op] or "value" in data:
        raise ValueError("Unknown operator or invalid arity")
    return Expr(op, tuple(parse(item, depth + 1, counter) for item in arguments))


def run(expr, signals):
    if expr.op in VARIABLES:
        return signals[expr.op]
    if expr.op == "const":
        return torch.full_like(signals["x"], expr.value)
    values = [run(item, signals) for item in expr.args]
    if expr.op == "add": result = values[0] + values[1]
    elif expr.op == "sub": result = values[0] - values[1]
    elif expr.op == "mul": result = values[0] * values[1]
    elif expr.op == "mix":
        gate = values[2].sigmoid()
        result = gate * values[0] + (1 - gate) * values[1]
    elif expr.op == "tanh": result = values[0].tanh()
    elif expr.op == "sigmoid": result = values[0].sigmoid()
    elif expr.op == "sin": result = values[0].sin()
    elif expr.op == "softsign": result = values[0] / (1 + values[0].abs())
    elif expr.op == "neg": result = -values[0]
    else: raise ValueError("Unvalidated expression")
    return result.clamp(-16, 16)


def compile_expression(data):
    expression = parse(data.to_dict() if isinstance(data, Expr) else data)

    def compiled(**signals):
        if set(signals) != VARIABLES:
            raise ValueError("Missing or unknown signal")
        shape = signals["x"].shape
        if any(value.shape != shape or not torch.isfinite(value).all() for value in signals.values()):
            raise ValueError("Invalid signal shape or value")
        return run(expression, signals)

    return compiled
