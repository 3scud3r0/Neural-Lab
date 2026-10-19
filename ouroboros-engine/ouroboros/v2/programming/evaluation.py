from __future__ import annotations

from dataclasses import dataclass
import math

from ..vm import Limits, Machine, Program


@dataclass
class Evaluation:
    passed: int
    total: int
    error: float
    faults: dict[str, int]
    mean_steps: float

    @property
    def solved(self):
        return self.total > 0 and self.passed == self.total

    @property
    def reward(self):
        if not self.total: return -1.0
        # Um erro não pode ser compensado por baixo tempo de execução.
        return 1.0 if self.solved else -min(1.0, self.error) - 0.25 * sum(self.faults.values()) / self.total


def evaluate(program: Program, cases, limits=Limits(), tools=None):
    machine = Machine(limits, tools)
    passed, errors, steps, faults = 0, [], [], {}
    for case in cases:
        result = machine.run(program, case.inputs, case.array)
        steps.append(result.steps)
        if not result.ok or len(result.outputs) != 1:
            kind = result.fault or "output_count"
            faults[kind] = faults.get(kind, 0) + 1
            errors.append(10.0)
            continue
        error = abs(result.outputs[0] - case.expected) / max(1, abs(case.expected))
        correct = math.isclose(result.outputs[0], case.expected, rel_tol=1e-7, abs_tol=1e-7)
        passed += int(correct)
        errors.append(min(10.0, error))
    total = len(errors)
    return Evaluation(passed, total, sum(errors) / max(1, total), faults, sum(steps) / max(1, total))
