"""A bounded hypothesis scheduler with UCB exploration and durable evidence."""
from dataclasses import dataclass, field
import math

from .experiments import Budget, atomic_json


@dataclass
class Hypothesis:
    name: str
    question: str
    trials: int = 0
    mean_utility: float = 0.0
    evidence: list = field(default_factory=list)

    def observe(self, utility, evidence):
        if not math.isfinite(utility):
            raise ValueError("Nonfinite research utility")
        self.trials += 1
        self.mean_utility += (utility - self.mean_utility) / self.trials
        self.evidence.append(evidence)


class ResearchController:
    def __init__(self, hypotheses, seconds=60, trials=10):
        if not hypotheses or len({h.name for h in hypotheses}) != len(hypotheses):
            raise ValueError("Distinct hypotheses required")
        self.hypotheses = hypotheses
        self.budget = Budget(seconds, trials)

    def choose(self):
        untried = next((h for h in self.hypotheses if not h.trials), None)
        if untried is not None:
            return untried
        total = sum(h.trials for h in self.hypotheses)
        return max(self.hypotheses, key=lambda h: h.mean_utility + math.sqrt(2 * math.log(total) / h.trials))

    def run(self, execute, output=None):
        while True:
            try:
                self.budget.claim()
            except TimeoutError:
                break
            hypothesis = self.choose()
            utility, evidence = execute(hypothesis, hypothesis.trials)
            hypothesis.observe(utility, evidence)
            if output:
                atomic_json(output, self.report())
        return self.report()

    def report(self):
        return {"evaluations": self.budget.used,
                "hypotheses": [{"name": h.name, "question": h.question, "trials": h.trials,
                                "mean_utility": h.mean_utility, "evidence": h.evidence} for h in self.hypotheses]}
