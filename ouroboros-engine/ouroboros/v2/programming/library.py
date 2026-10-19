"""Autoalteração delimitada: substituições versionadas de ferramentas da VM.

O agente muda seu repertório executável. A VM e os critérios de aprovação
permanecem no host; uma ferramenta não recebe acesso ao filesystem ou ao Python.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from ..common import atomic_json, stable_hash
from ..vm import Instruction, Limits, Machine, Program
from .evaluation import evaluate


class ToolLibrary:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "registry.json"
        self.registry = json.loads(self.path.read_text()) if self.path.is_file() else {"schema": 1, "tools": {}, "history": []}

    def programs(self):
        return {int(key): Program.from_dict(item["program"]) for key, item in self.registry["tools"].items()}

    def callable_programs(self):
        return {key: Program(tuple(Instruction("RET") if i.op == "HALT" else i for i in program.code), program.name)
                for key, program in self.programs().items()}

    def _check_calls(self, tool_id, program):
        programs = self.programs()
        programs[tool_id] = program
        graph = {key: {i.arg for i in p.code if i.op == "CALL"} for key, p in programs.items()}
        visiting, visited = set(), set()

        def walk(key):
            if key in visiting: raise ValueError("Chamadas recursivas não são promovidas")
            if key in visited: return
            if key not in graph: raise ValueError("Chamada para ferramenta ausente")
            visiting.add(key)
            for child in graph[key]: walk(child)
            visiting.remove(key)
            visited.add(key)

        walk(tool_id)

    def promote(self, tool_id, name, candidate, task, require_improvement=False):
        if type(tool_id) is not int or not 0 <= tool_id < 256:
            raise ValueError("Identificador de ferramenta inválido")
        candidate.validate()
        self._check_calls(tool_id, candidate)
        validation = evaluate(candidate, task.validation, tools=self.callable_programs())
        if not validation.solved:
            return {"accepted": False, "reason": "validation_failed", "evaluation": asdict(validation)}
        # O conjunto final de teste é reservado para relatório posterior, não promoção.
        previous = self.registry["tools"].get(str(tool_id))
        if previous and require_improvement:
            old = evaluate(Program.from_dict(previous["program"]), task.validation, tools=self.callable_programs())
            if old.solved and validation.mean_steps >= old.mean_steps:
                return {"accepted": False, "reason": "no_fuel_improvement"}
        revision = 1 if previous is None else previous["revision"] + 1
        record = {"id": tool_id, "name": name, "revision": revision, "program": candidate.to_dict(),
                  "fingerprint": candidate.fingerprint, "validation": asdict(validation),
                  "parent": previous["fingerprint"] if previous else None}
        self.registry["history"].append({"tool": tool_id, "previous": previous, "next": record})
        self.registry["tools"][str(tool_id)] = record
        atomic_json(self.path, self.registry)
        return {"accepted": True, "revision": revision, "fingerprint": candidate.fingerprint}

    def rollback(self, tool_id):
        for event in reversed(self.registry["history"]):
            if event["tool"] == tool_id and not event.get("reverted"):
                if event["previous"] is None:
                    self.registry["tools"].pop(str(tool_id), None)
                else:
                    self.registry["tools"][str(tool_id)] = event["previous"]
                event["reverted"] = True
                atomic_json(self.path, self.registry)
                return True
        return False

    def call(self, tool_id, inputs=(), array=()):
        # OUTPUT permanece no corpo da ferramenta; RET devolve ao chamador.
        programs = self.programs()
        if tool_id not in programs: raise KeyError(tool_id)
        tools = self.callable_programs()
        caller = Program((Instruction("CALL", tool_id), Instruction("HALT")), "tool_caller")
        return Machine(tools=tools).run(caller, inputs, array)
