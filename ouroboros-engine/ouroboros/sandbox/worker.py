"""Worker independente: interpreta somente uma AST aritmética restrita.

Não usa eval/exec, imports do usuário, acesso a atributos ou chamadas livres.
Limites de processo são defesa adicional; isto não executa Python arbitrário.
"""
import ast
import json
import math
import operator
import resource
import sys
import time


OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv}


def bounded(value):
    if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e12:
        raise ValueError("Número fora do limite permitido")
    return float(value)


def interpret(node, variables, depth=0):
    if depth > 12:
        raise ValueError("Expressão profunda demais")
    if isinstance(node, ast.Constant):
        return bounded(node.value)
    if isinstance(node, ast.Name) and node.id in ("x", "y"):
        return bounded(variables[node.id])
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return bounded(OPS[type(node.op)](interpret(node.left, variables, depth + 1),
                                         interpret(node.right, variables, depth + 1)))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return bounded((1 if isinstance(node.op, ast.UAdd) else -1) * interpret(node.operand, variables, depth + 1))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        functions = {"abs": (abs, 1), "min": (min, 2), "max": (max, 2)}
        if node.func.id in functions:
            function, count = functions[node.func.id]
            if len(node.args) == count:
                return bounded(function(*(interpret(arg, variables, depth + 1) for arg in node.args)))
    raise ValueError("Construção não permitida na linguagem aritmética")


def main():
    request = json.loads(sys.stdin.read(65536))
    limit = int(request["memory_mb"]) * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
    started = time.perf_counter()
    try:
        source, cases = request["expression"], request["cases"]
        if not isinstance(source, str) or len(source) > 256:
            raise ValueError("Expressão deve ter até 256 caracteres")
        if not isinstance(cases, list) or not 1 <= len(cases) <= 64:
            raise ValueError("Informe de 1 a 64 casos")
        tree = ast.parse(source, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 64:
            raise ValueError("AST excede 64 nós")
        values, errors = [], []
        for x, y, expected in cases:
            value = interpret(tree.body, {"x": bounded(x), "y": bounded(y)})
            expected = bounded(expected)
            values.append(value)
            errors.append(abs(value - expected) / max(1, abs(expected)))
        result = {"ok": True, "values": values, "error": sum(errors) / len(errors), "message": ""}
    except (ValueError, TypeError, SyntaxError, ZeroDivisionError, OverflowError, RecursionError) as exc:
        result = {"ok": False, "values": [], "error": 1.0, "message": str(exc)[:200]}
    elapsed_ms = (time.perf_counter() - started) * 1000
    # ru_maxrss pode carregar o pico do pai PyTorch através de fork/exec.
    # VmHWM mede o espaço do worker após exec, sem atribuir a memória do pai.
    with open("/proc/self/status", encoding="ascii") as status:
        memory_kb = next(int(line.split()[1]) for line in status if line.startswith("VmHWM:"))
    result.update(elapsed_ms=elapsed_ms, memory_kb=memory_kb)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
