"""Typed module plans and bounded synthesis using the existing VM and compiler."""
from dataclasses import dataclass

from ..v2.programming.evaluation import evaluate
from ..v2.programming.genetics import GeneticProgrammer
from ..v2.programming.library import ToolLibrary
from ..v2.vm import Instruction, Limits, Machine, Program, compile_source, disassemble


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    arguments: tuple[str, ...]
    returns: str = "number"
    dependencies: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.name.isidentifier() or self.returns != "number" or any(t not in {"number", "array"} for t in self.arguments):
            raise ValueError("Unsupported function signature")
        if len(self.arguments) > 4:
            raise ValueError("Too many function arguments")


class ModulePlan:
    def __init__(self, specs):
        self.specs = {spec.name: spec for spec in specs}
        if len(self.specs) != len(specs):
            raise ValueError("Duplicate module names")

    def order(self):
        visiting, visited, result = set(), set(), []

        def visit(name):
            if name in visiting:
                raise ValueError("Cyclic tool plan")
            if name in visited:
                return
            if name not in self.specs:
                raise ValueError("Missing tool dependency")
            visiting.add(name)
            for dependency in self.specs[name].dependencies:
                visit(dependency)
            visiting.remove(name)
            visited.add(name)
            result.append(name)

        for name in self.specs:
            visit(name)
        return result


class Programmer:
    def __init__(self, directory, fuel=4000):
        self.library = ToolLibrary(directory)
        self.machine = Machine(Limits(fuel=fuel))

    def compile_and_test(self, source, task):
        program = compile_source(source)
        result = evaluate(program, task.validation)
        return {"program": program, "accepted": result.solved,
                "validation": result, "assembly": disassemble(program)}

    def debug(self, program, cases):
        failures = []
        for case in cases:
            result = self.machine.run(program, case.inputs, case.array, trace=True)
            if not result.ok or not result.outputs or abs(result.outputs[-1] - case.expected) > 1e-6:
                failures.append({"inputs": list(case.inputs), "array": list(case.array),
                                 "expected": case.expected, "outputs": result.outputs,
                                 "fault": result.fault, "trace": result.trace})
        return failures

    def synthesize(self, task, seed=42, generations=80):
        return GeneticProgrammer(seed=seed, generations=generations).search(task)

    def promote(self, tool_id, name, program, task):
        return self.library.promote(tool_id, name, program, task, require_improvement=True)

    def compose(self, tool_ids):
        if not tool_ids or len(tool_ids) > 16:
            raise ValueError("Tool composition size exceeded")
        available = self.library.programs()
        if any(tool not in available for tool in tool_ids):
            raise ValueError("Unknown tool")
        return Program(tuple([Instruction("CALL", tool) for tool in tool_ids] + [Instruction("HALT")]), "composition")

    def generate_with_checkpoint(self, checkpoint, task, directory, samples=16, seed=42):
        from ..v2.programming.generation import generate_candidates, load_programmer
        from .experiments import isolated_seed
        with isolated_seed(seed):
            return generate_candidates(load_programmer(checkpoint), task, directory, samples=samples)
