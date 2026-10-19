"""Geração de código para novas especificações, com orçamento e teste externo."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from ..common import atomic_json
from ..vm import disassemble
from .evaluation import evaluate
from .token_model import TokenProgrammer


def load_programmer(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = TokenProgrammer(hidden=checkpoint["hidden"], context_size=checkpoint["context_size"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


@torch.no_grad()
def generate_candidates(model, task, directory, samples=16):
    if not 1 <= samples <= 256: raise ValueError("Use de 1 a 256 amostras")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    candidates, results = [], []
    for index in range(samples):
        program, *_ = model.generate(task.context(device), greedy=index == 0, temperature=0.8)
        result = evaluate(program, task.validation) if program else None
        results.append({"candidate": index, "valid": program is not None,
                        "validation": asdict(result) if result else None})
        if program:
            candidates.append((program, result, index))
            (directory / f"candidate_{index:04d}.ovm").write_text(disassemble(program), encoding="utf-8")
    if candidates:
        # O teste final não participa da escolha do candidato.
        best, validation, index = max(candidates, key=lambda row: (row[1].passed / row[1].total, -row[1].error, -len(row[0].code)))
        test = evaluate(best, task.test)
        report = {"task": task.name, "selected": index, "validation": asdict(validation),
                  "test": asdict(test), "solved": test.solved, "candidates": results}
        atomic_json(directory / "selected.json", best.to_dict())
    else:
        report = {"task": task.name, "selected": None, "solved": False, "candidates": results}
    atomic_json(directory / "report.json", report)
    return report
