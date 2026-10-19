from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import sys
import time


@dataclass
class ExecutionResult:
    ok: bool
    error: float = 1.0
    values: list = field(default_factory=list)
    message: str = ""
    elapsed_ms: float = 0.0
    wall_ms: float = 0.0
    memory_kb: int = 0
    returncode: int = 0


class SafeExecutor:
    """Executor Linux de DSL aritmética; não é sandbox de Python/C/ASM geral."""

    def __init__(self, timeout_seconds=2.0, memory_mb=256):
        if timeout_seconds <= 0 or memory_mb < 64:
            raise ValueError("Timeout positivo e pelo menos 64 MB são necessários")
        self.timeout, self.memory_mb = timeout_seconds, memory_mb

    def execute(self, expression, cases):
        payload = json.dumps({"expression": expression, "cases": cases, "memory_mb": self.memory_mb}, allow_nan=False)
        if len(payload.encode()) > 60000:
            return ExecutionResult(False, message="Entrada excede o limite de 60 KB")
        started = time.perf_counter()
        try:
            process = subprocess.run([sys.executable, "-I", str(Path(__file__).with_name("worker.py"))],
                                     input=payload, text=True, capture_output=True, timeout=self.timeout,
                                     env={"LANG": "C.UTF-8"}, cwd="/tmp")
            wall = (time.perf_counter() - started) * 1000
            if process.returncode:
                return ExecutionResult(False, message="Worker encerrado ou limite de recursos atingido",
                                       returncode=process.returncode, wall_ms=wall)
            return ExecutionResult(**json.loads(process.stdout), wall_ms=wall)
        except subprocess.TimeoutExpired:
            return ExecutionResult(False, message="Tempo limite excedido", returncode=-1,
                                   wall_ms=(time.perf_counter() - started) * 1000)
        except (OSError, ValueError) as exc:
            return ExecutionResult(False, message=str(exc)[:200], returncode=-1)
