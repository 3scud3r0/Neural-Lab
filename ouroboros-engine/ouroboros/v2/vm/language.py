"""Bytecode tipado, serializável e com endereços de instrução verificáveis."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math


# (pop mínimo, push líquido) auxilia a síntese; a VM verifica o estado concreto.
STACK_EFFECT = {
    "PUSH": (0, 1), "INPUT": (0, 1), "LOAD": (0, 1), "STORE": (1, -1),
    "ADD": (2, -1), "SUB": (2, -1), "MUL": (2, -1), "DIV": (2, -1),
    "MOD": (2, -1), "MIN": (2, -1), "MAX": (2, -1), "LT": (2, -1),
    "GT": (2, -1), "LE": (2, -1), "GE": (2, -1), "EQ": (2, -1),
    "AND": (2, -1), "OR": (2, -1), "NEG": (1, 0), "ABS": (1, 0),
    "NOT": (1, 0), "DUP": (1, 1), "DROP": (1, -1), "SWAP": (2, 0),
    "OVER": (2, 1), "JMP": (0, 0), "JZ": (1, -1), "JNZ": (1, -1),
    "LEN": (0, 1), "AGET": (1, 0), "ALLOC": (0, 1), "READ": (1, 0),
    "WRITE": (2, -2), "OUTPUT": (1, -1), "HALT": (0, 0),
    "CALL": (0, 0), "RET": (0, 0), "NOP": (0, 0),
}
ARGUMENT_OPS = {"PUSH", "INPUT", "LOAD", "STORE", "JMP", "JZ", "JNZ", "ALLOC", "CALL"}
JUMP_OPS = {"JMP", "JZ", "JNZ"}


@dataclass(frozen=True)
class Limits:
    fuel: int = 4000
    stack: int = 128
    heap: int = 128
    registers: int = 16
    instructions: int = 256
    array: int = 128
    call_depth: int = 8
    magnitude: float = 1e12

    def __post_init__(self):
        for key in ("fuel", "stack", "heap", "registers", "instructions", "array", "call_depth"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"Limite inválido: {key}")
        if self.registers > self.heap or not math.isfinite(self.magnitude) or self.magnitude <= 0:
            raise ValueError("Limites de memória/números inválidos")


@dataclass(frozen=True)
class Instruction:
    op: str
    arg: int | float | None = None

    def __post_init__(self):
        if self.op not in STACK_EFFECT:
            raise ValueError(f"Opcode desconhecido: {self.op}")
        if (self.op in ARGUMENT_OPS) != (self.arg is not None):
            raise ValueError(f"Aridade inválida: {self.op}")
        if self.arg is not None:
            if type(self.arg) not in (int, float) or not math.isfinite(self.arg):
                raise ValueError("Operando não numérico ou não finito")
            if self.op != "PUSH" and type(self.arg) is not int:
                raise ValueError("Endereços e índices precisam ser inteiros")


@dataclass(frozen=True)
class Program:
    code: tuple[Instruction, ...]
    name: str = "candidate"

    def validate(self, limits: Limits = Limits()):
        if not 1 <= len(self.code) <= limits.instructions:
            raise ValueError("Comprimento do programa fora do limite")
        for instruction in self.code:
            op, arg = instruction.op, instruction.arg
            if op in JUMP_OPS and not 0 <= arg < len(self.code):
                raise ValueError(f"Salto fora do programa: {arg}")
            if op in {"LOAD", "STORE"} and not 0 <= arg < limits.heap:
                raise ValueError("Endereço de heap inválido")
            if op == "INPUT" and not 0 <= arg < 8:
                raise ValueError("Índice de entrada inválido")
            if op == "ALLOC" and not 1 <= arg <= limits.heap:
                raise ValueError("Alocação fora do limite")
            if op == "CALL" and not 0 <= arg < 256:
                raise ValueError("Identificador de ferramenta inválido")
            if op == "PUSH" and abs(arg) > limits.magnitude:
                raise ValueError("Constante excede a magnitude permitida")
        return self

    def to_dict(self):
        return {"name": self.name, "code": [[i.op, i.arg] for i in self.code]}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or not isinstance(data.get("code"), list):
            raise ValueError("Programa serializado inválido")
        if len(data["code"]) > 4096:
            raise ValueError("Programa serializado grande demais")
        return cls(tuple(Instruction(*item) for item in data["code"]), str(data.get("name", "candidate")))

    @property
    def fingerprint(self):
        # Nomes não mudam a identidade operacional.
        data = json.dumps([[i.op, i.arg] for i in self.code], separators=(",", ":"))
        return hashlib.sha256(data.encode()).hexdigest()


def assemble(source: str, name="candidate") -> Program:
    """Labels são resolvidos em duas passagens; nenhuma linha vira Python."""
    if not isinstance(source, str) or len(source) > 100_000:
        raise ValueError("Assembly ausente ou grande demais")
    labels, pending = {}, []
    for number, raw in enumerate(source.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.endswith(":"):
            label = line[:-1].strip()
            if not label.isidentifier() or label in labels:
                raise ValueError(f"Label inválido na linha {number}")
            labels[label] = len(pending)
            continue
        fields = line.split()
        if len(fields) > 2:
            raise ValueError(f"Operandos demais na linha {number}")
        pending.append((fields[0].upper(), fields[1] if len(fields) == 2 else None))
    code = []
    for op, raw in pending:
        if raw is None:
            arg = None
        elif op in JUMP_OPS and raw in labels:
            arg = labels[raw]
        else:
            try:
                arg = float(raw) if op == "PUSH" and any(c in raw.lower() for c in ".e") else int(raw)
            except ValueError as exc:
                raise ValueError(f"Operando ou label inválido: {raw}") from exc
        code.append(Instruction(op, arg))
    return Program(tuple(code), name).validate()


def disassemble(program: Program) -> str:
    targets = {i.arg for i in program.code if i.op in JUMP_OPS}
    lines = []
    for address, instruction in enumerate(program.code):
        if address in targets:
            lines.append(f"label_{address}:")
        operand = f"label_{instruction.arg}" if instruction.op in JUMP_OPS else instruction.arg
        lines.append(instruction.op + (f" {operand}" if operand is not None else ""))
    return "\n".join(lines) + "\n"
