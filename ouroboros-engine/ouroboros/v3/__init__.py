"""Ouroboros v3: sixteen-module, budgeted neuron research laboratory.

Candidates are equations in a bounded tensor DSL, never executable host code.
The package is an experimental implementation, not a claim of general intelligence.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

VERSION = "3.0.0"
MODULES = ("__init__", "experiments", "neuron_dsl", "neurons", "plasticity",
           "benchmarks", "search", "evaluation", "architecture", "memory", "world",
           "society", "programming", "selfmod", "research", "application")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class SearchConfig:
    candidates: int = 1024
    hidden: int = 12
    screening_steps: int = 3
    promotion_steps: int = 24
    finalists: int = 8
    batch: int = 24
    length: int = 12
    seed: int = 42
    max_seconds: float = 1800

    def __post_init__(self):
        for name in ("candidates", "hidden", "screening_steps", "promotion_steps",
                     "finalists", "batch", "length"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.finalists > self.candidates or self.hidden > 128 or not 3 <= self.length <= 128 or self.batch > 1024:
            raise ValueError("Invalid search dimensions")
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("Invalid time budget")

    def to_dict(self):
        return asdict(self)


@dataclass
class Score:
    candidate: str
    validation_loss: float
    parameters: int
    seconds: float
    stage: str
    error: str | None = None

    @property
    def valid(self):
        return self.error is None and math.isfinite(self.validation_loss)

    def to_dict(self):
        result = asdict(self)
        if not math.isfinite(result["validation_loss"]):
            result["validation_loss"] = None
        return result
