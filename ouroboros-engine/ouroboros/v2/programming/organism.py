"""Ciclo de percepção, geração, execução, metabolismo e revisão de ferramentas."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random

import torch

from ..common import Journal, atomic_json
from ..vm import disassemble
from .evaluation import evaluate
from .genetics import mutate_bytecode
from .library import ToolLibrary


@dataclass
class ProgrammerMetabolism:
    energy: float = 1.0
    stress: float = 0.0
    failures: int = 0
    successes: int = 0
    resets: int = 0

    def update(self, evaluation, program_length):
        reward = evaluation.reward if evaluation else -1.25
        solved = bool(evaluation and evaluation.solved)
        fuel = evaluation.mean_steps if evaluation else 0
        self.energy = max(0., min(1., self.energy - 0.002 - program_length * 0.00001 - fuel * 0.000005 + (0.025 if solved else 0)))
        self.stress = max(0., min(2., 0.9 * self.stress + 0.1 * max(0., -reward)))
        self.failures += int(not solved)
        self.successes += int(solved)
        if self.energy <= 0:
            self.resets += 1
            self.energy, self.stress = 0.7, 0.0
        return reward


class ProgramOrganism:
    def __init__(self, model, directory, learning_rate=0.0002):
        self.model = model
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.library = ToolLibrary(self.directory / "tools")
        self.journal = Journal(self.directory)
        self.metabolism = ProgrammerMetabolism()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.baseline = 0.0
        self.attempts = 0

    def attempt(self, task, learn=True, greedy=False):
        device = next(self.model.parameters()).device
        temperature = max(0.3, min(1.2, 0.5 + 0.3 * self.metabolism.stress))
        context = task.context(device)
        program, log_probability, entropy, tokens = self.model.generate(context, temperature=temperature, greedy=greedy)
        evaluation = evaluate(program, task.train, tools=self.library.callable_programs()) if program else None
        reward = self.metabolism.update(evaluation, len(program.code) if program else 0)
        if learn:
            loss = -(reward - self.baseline) * log_probability - 0.001 * entropy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
        self.baseline = 0.9 * self.baseline + 0.1 * reward
        self.attempts += 1
        record = {"attempt": self.attempts, "task": task.name, "reward": reward,
                  "valid": program is not None, "metabolism": asdict(self.metabolism),
                  "training_evaluation": asdict(evaluation) if evaluation else None,
                  "token_count": len(tokens)}
        self.journal.record("programmer", **record)
        if program:
            generated = self.directory / "programs"
            generated.mkdir(exist_ok=True)
            stem = f"{self.attempts:05d}"
            (generated / (stem + ".ovm")).write_text(disassemble(program), encoding="utf-8")
            atomic_json(generated / (stem + ".json"), {"program": program.to_dict(), **record})
        return program, record

    def improve_and_promote(self, tool_id, task, candidates=16, seed=42):
        rng = random.Random(seed)
        with torch.no_grad():
            initial, *_ = self.model.generate(task.context(next(self.model.parameters()).device), greedy=True)
        if initial is None:
            return {"accepted": False, "reason": "neural_candidate_invalid"}
        best, best_error = initial, evaluate(initial, task.train).error
        for _ in range(candidates):
            candidate = mutate_bytecode(best, rng)
            result = evaluate(candidate, task.train)
            if result.error < best_error:
                best, best_error = candidate, result.error
        promotion = self.library.promote(tool_id, task.name, best, task)
        self.journal.record("tool_promotion", task=task.name, tool_id=tool_id, **promotion)
        return promotion
