"""Máquina virtual de pilha, heap e chamadas. Todo trabalho consome combustível."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

from .language import Limits, Program, STACK_EFFECT


class VMFault(Exception):
    def __init__(self, kind, message):
        self.kind = kind
        super().__init__(message)


@dataclass
class Result:
    ok: bool
    outputs: list[float] = field(default_factory=list)
    fault: str | None = None
    message: str = ""
    steps: int = 0
    max_stack: int = 0
    allocated: int = 0
    elapsed_ms: float = 0.0
    trace: list[dict] = field(default_factory=list)


class Machine:
    def __init__(self, limits: Limits = Limits(), tools: dict[int, Program] | None = None):
        self.limits = limits
        self.tools = dict(tools or {})

    def run(self, program: Program, inputs=(), array=(), trace=False) -> Result:
        started = time.perf_counter()
        result = Result(False)
        limits = self.limits
        stack, outputs, frames = [], [], []
        heap = [0.0] * limits.heap
        allocated = limits.registers

        def numeric(value):
            if type(value) not in (float, int) or not math.isfinite(value) or abs(value) > limits.magnitude:
                raise VMFault("number", "Número inválido ou magnitude excedida")
            return float(value)

        def pop():
            if not stack:
                raise VMFault("stack_underflow", "Pilha vazia")
            return stack.pop()

        def index(value, size):
            if value != int(value) or not 0 <= int(value) < size:
                raise VMFault("address", "Índice/endereço fora da memória permitida")
            return int(value)

        try:
            program.validate(limits)
            if len(inputs) > 8 or len(array) > limits.array:
                raise VMFault("input", "Entrada maior que o limite")
            inputs, array = tuple(map(numeric, inputs)), tuple(map(numeric, array))
            current, pc = program, 0
            while True:
                if pc == len(current.code):
                    if frames:
                        current, pc = frames.pop()
                        continue
                    break
                if result.steps >= limits.fuel:
                    raise VMFault("fuel", "Combustível esgotado (inclui loops e chamadas)")
                instruction = current.code[pc]
                op, arg = instruction.op, instruction.arg
                if len(stack) < STACK_EFFECT[op][0]:
                    raise VMFault("stack_underflow", f"Pilha insuficiente para {op}")
                if trace and len(result.trace) < 128:
                    result.trace.append(dict(pc=pc, op=op, arg=arg, stack=stack[-8:].copy(), depth=len(frames)))
                pc += 1
                result.steps += 1
                if op == "PUSH":
                    stack.append(numeric(arg))
                elif op == "INPUT":
                    stack.append(inputs[index(arg, len(inputs))])
                elif op == "LOAD":
                    stack.append(heap[index(arg, allocated)])
                elif op == "STORE":
                    heap[index(arg, allocated)] = pop()
                elif op == "ALLOC":
                    if allocated + arg > limits.heap:
                        raise VMFault("heap", "Heap esgotado")
                    stack.append(float(allocated))
                    allocated += arg
                elif op == "READ":
                    stack.append(heap[index(pop(), allocated)])
                elif op == "WRITE":
                    value, address = pop(), pop()
                    heap[index(address, allocated)] = value
                elif op == "LEN":
                    stack.append(float(len(array)))
                elif op == "AGET":
                    stack.append(array[index(pop(), len(array))])
                elif op in {"ADD", "SUB", "MUL", "DIV", "MOD", "MIN", "MAX", "LT", "GT", "LE", "GE", "EQ", "AND", "OR"}:
                    b, a = pop(), pop()
                    if op in {"DIV", "MOD"} and b == 0:
                        raise VMFault("division", "Divisão por zero")
                    if op == "ADD": value = a + b
                    elif op == "SUB": value = a - b
                    elif op == "MUL": value = a * b
                    elif op == "DIV": value = a / b
                    elif op == "MOD": value = a % b
                    elif op == "MIN": value = min(a, b)
                    elif op == "MAX": value = max(a, b)
                    elif op == "LT": value = int(a < b)
                    elif op == "GT": value = int(a > b)
                    elif op == "LE": value = int(a <= b)
                    elif op == "GE": value = int(a >= b)
                    elif op == "EQ": value = int(a == b)
                    elif op == "AND": value = int(bool(a) and bool(b))
                    else: value = int(bool(a) or bool(b))
                    stack.append(numeric(value))
                elif op in {"NEG", "ABS", "NOT"}:
                    a = pop()
                    stack.append(numeric(-a if op == "NEG" else abs(a) if op == "ABS" else int(not a)))
                elif op == "DUP": stack.append(stack[-1])
                elif op == "OVER": stack.append(stack[-2])
                elif op == "DROP": pop()
                elif op == "SWAP": stack[-1], stack[-2] = stack[-2], stack[-1]
                elif op == "JMP": pc = arg
                elif op in {"JZ", "JNZ"}:
                    condition = bool(pop())
                    if condition == (op == "JNZ"): pc = arg
                elif op == "CALL":
                    if arg not in self.tools:
                        raise VMFault("tool", "Ferramenta não registrada")
                    if len(frames) >= limits.call_depth:
                        raise VMFault("call_depth", "Profundidade de chamada excedida")
                    callee = self.tools[arg].validate(limits)
                    frames.append((current, pc))
                    current, pc = callee, 0
                elif op == "RET":
                    if not frames: break
                    current, pc = frames.pop()
                elif op == "OUTPUT":
                    if len(outputs) >= 16:
                        raise VMFault("output", "Saídas demais")
                    outputs.append(pop())
                elif op == "HALT": break
                elif op == "NOP": pass
                if len(stack) > limits.stack:
                    raise VMFault("stack_overflow", "Pilha excede o limite")
                result.max_stack = max(result.max_stack, len(stack))
            if not outputs:
                raise VMFault("output", "Programa terminou sem OUTPUT")
            result.ok = True
            result.outputs = outputs
        except VMFault as exc:
            result.fault, result.message = exc.kind, str(exc)
        except (ValueError, TypeError, OverflowError) as exc:
            result.fault, result.message = "validation", str(exc)[:240]
        result.allocated = allocated
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        return result
