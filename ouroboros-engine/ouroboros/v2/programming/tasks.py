"""Tarefas com referências confiáveis e conjuntos separados de exemplos."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random

import torch

from ..common import stable_hash
from ..vm import compile_source


@dataclass(frozen=True)
class Case:
    inputs: tuple[float, ...]
    expected: float
    array: tuple[float, ...] = ()

    def to_dict(self):
        return dict(inputs=list(self.inputs), expected=self.expected, array=list(self.array))


@dataclass
class Task:
    name: str
    description: str
    source: str
    train: list[Case]
    validation: list[Case]
    test: list[Case]

    def reference_program(self):
        return compile_source(self.source, self.name)

    def context(self, device="cpu", cases=8):
        # Somente exemplos públicos de treino entram no modelo; nenhum ID de tarefa.
        values = []
        for i in range(cases):
            if i >= len(self.train):
                values.extend([0.] * 9)
                continue
            case = self.train[i]
            inputs = list(case.inputs[:3]) + [0.] * (3 - len(case.inputs[:3]))
            array = list(case.array[:4]) + [0.] * (4 - len(case.array[:4]))
            row = inputs + [float(len(case.array))] + array + [case.expected]
            values.extend(math.copysign(math.log1p(abs(x)), x) / 5 for x in row)
        return torch.tensor(values, dtype=torch.float32, device=device)


SOURCES = {
    "sum": "def solve(x, y):\n    return x + y\n",
    "product": "def solve(x, y):\n    return x * y\n",
    "difference": "def solve(x, y):\n    return x - y\n",
    "absolute_difference": "def solve(x, y):\n    return abs(x - y)\n",
    "sum_range": "def solve(n):\n    total = 0\n    for i in range(n + 1):\n        total += i\n    return total\n",
    "factorial": "def solve(n):\n    total = 1\n    for i in range(1, n + 1):\n        total *= i\n    return total\n",
    "array_sum": "def solve(arr):\n    total = 0\n    for i in range(len(arr)):\n        total += arr[i]\n    return total\n",
    "count_positive": "def solve(arr):\n    total = 0\n    for i in range(len(arr)):\n        if arr[i] > 0:\n            total += 1\n    return total\n",
    "array_max": "def solve(arr):\n    best = arr[0]\n    for i in range(1, len(arr)):\n        best = max(best, arr[i])\n    return best\n",
    "gcd": "def solve(x, y):\n    a = x\n    b = y\n    while b > 0:\n        remainder = a % b\n        a = b\n        b = remainder\n    return a\n",
    "branch": "def solve(x, y):\n    if x > y:\n        return x * x\n    else:\n        return y + 2\n",
}


def reference(name, inputs, array):
    if name == "sum": return inputs[0] + inputs[1]
    if name == "product": return inputs[0] * inputs[1]
    if name == "difference": return inputs[0] - inputs[1]
    if name == "absolute_difference": return abs(inputs[0] - inputs[1])
    if name == "sum_range": return inputs[0] * (inputs[0] + 1) // 2
    if name == "factorial": return math.factorial(inputs[0])
    if name == "array_sum": return sum(array)
    if name == "count_positive": return sum(x > 0 for x in array)
    if name == "array_max": return max(array)
    if name == "gcd": return math.gcd(*inputs)
    if name == "branch": return inputs[0] ** 2 if inputs[0] > inputs[1] else inputs[1] + 2
    raise KeyError(name)


def make_task(name, seed=42, counts=(8, 8, 12)):
    if name not in SOURCES:
        raise ValueError(f"Tarefa desconhecida: {name}")
    splits = []
    # Domínios maiores nos conjuntos externos, sem reciclar os mesmos casos.
    used = set()
    for split, count in enumerate(counts):
        if name in {"sum_range", "factorial"}:
            count = min(count, (6, 3, 2)[split])
        rng = random.Random(seed + 1009 * split + int(stable_hash(name)[:8], 16))
        cases = []
        while len(cases) < count:
            if name.startswith("array_") or name == "count_positive":
                length = rng.randint(1, 5) if split == 0 else rng.randint(6, 12)
                array = tuple(rng.randint(-7, 9) for _ in range(length))
                inputs = ()
            elif name in {"sum_range", "factorial"}:
                low, high = ((0, 5), (6, 8), (9, 10))[split]
                n = rng.randint(low, high)
                inputs, array = (n,), ()
            else:
                low, high = (-5, 5) if split == 0 else (6, 20)
                if name == "gcd": low, high = (1, 6) if split == 0 else (7, 30)
                inputs, array = (rng.randint(low, high), rng.randint(low, high)), ()
            key = (inputs, array)
            if key in used:
                continue
            used.add(key)
            cases.append(Case(inputs, float(reference(name, inputs, array)), array))
        splits.append(cases)
    return Task(name, name.replace("_", " "), SOURCES[name], *splits)


def affine_task(a, b, c, seed=42):
    if any(type(x) is not int for x in (a, b, c)):
        raise ValueError("Coeficientes inteiros necessários")
    rng = random.Random(seed)
    train, validation, test = [], [], []
    seen = set()
    for split, count in zip((train, validation, test), (8, 8, 12)):
        while len(split) < count:
            x, y = rng.randint(-12, 12), rng.randint(-12, 12)
            if (x, y) in seen: continue
            seen.add((x, y))
            split.append(Case((x, y), float(a * x + b * y + c)))
    source = f"def solve(x, y):\n    return {a} * x + {b} * y + {c}\n"
    return Task(f"affine_{a}_{b}_{c}", "Composição afim parametrizada", source, train, validation, test)


def curriculum(seed=42):
    return [make_task(name, seed) for name in SOURCES]


def task_from_examples(path):
    """Carrega uma especificação por exemplos, sem código de solução."""
    path = Path(path)
    if path.stat().st_size > 262144: raise ValueError("Arquivo de exemplos excede 256 KB")
    data = json.loads(path.read_text(encoding="utf-8"))
    splits, seen = [], set()
    for split in ("train", "validation", "test"):
        rows = data.get(split)
        if not isinstance(rows, list) or not 1 <= len(rows) <= 64:
            raise ValueError("Cada split precisa conter de 1 a 64 exemplos")
        cases = []
        for row in rows:
            inputs, array, expected = tuple(row["inputs"]), tuple(row.get("array", [])), row["expected"]
            if len(inputs) > 8 or len(array) > 128: raise ValueError("Exemplo excede os limites da VM")
            values = inputs + array + (expected,)
            if any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 1e12 for x in values):
                raise ValueError("Exemplo precisa usar números finitos e limitados")
            key = inputs, array
            if key in seen: raise ValueError("Exemplos repetidos entre os splits não são permitidos")
            seen.add(key)
            cases.append(Case(inputs, float(expected), array))
        splits.append(cases)
    return Task(str(data.get("name", "user_task")), str(data.get("description", "Especificação por exemplos")), "", *splits)
