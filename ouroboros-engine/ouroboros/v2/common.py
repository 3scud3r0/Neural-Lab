"""Persistência atômica e reprodutibilidade compartilhadas pelos experimentos."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def seed_everything(seed: int, threads: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)


def device_for(name: str) -> torch.device:
    if name not in ("cpu", "cuda", "auto"):
        raise ValueError("device deve ser cpu, cuda ou auto")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponível nesta máquina")
    return torch.device(name)


def json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False,
                                    indent=2, default=json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def stable_hash(data) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False, default=json_default).encode()).hexdigest()


def new_run(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"Use uma pasta vazia para preservar execuções: {path}")
    return path


class Journal:
    """Um registro por evento; o snapshot é substituído atomicamente."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()

    def record(self, event: str, **fields):
        record = dict(event=event, elapsed=time.monotonic() - self.started, **fields)
        encoded = json.dumps(record, ensure_ascii=False, allow_nan=False, default=json_default)
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
        atomic_json(self.directory / "latest.json", record)
        return record


def wilson_interval(successes: int, trials: int, z: float = 1.96):
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("Contagens binomiais inválidas")
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    spread = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0.0, center - spread), min(1.0, center + spread)]
