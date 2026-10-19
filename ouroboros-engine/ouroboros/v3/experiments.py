"""Atomic artifacts, durable SQLite observations and deterministic RNG recovery."""
from contextlib import contextmanager
import json
from pathlib import Path
import random
import sqlite3
import time

import torch

from . import fingerprint


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


@contextmanager
def isolated_seed(seed):
    python_state, torch_state = random.getstate(), torch.get_rng_state()
    try:
        random.seed(seed)
        torch.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        torch.set_rng_state(torch_state)


class Budget:
    def __init__(self, seconds, evaluations):
        self.started = time.monotonic()
        self.seconds, self.evaluations, self.used = seconds, evaluations, 0

    def claim(self):
        if self.used >= self.evaluations or time.monotonic() - self.started >= self.seconds:
            raise TimeoutError("Experiment budget exhausted")
        self.used += 1


class ExperimentStore:
    def __init__(self, directory, config):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        config_path = self.directory / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError("Run directory belongs to another configuration")
        atomic_json(config_path, config)
        self.db = sqlite3.connect(self.directory / "experiments.sqlite")
        self.db.execute("CREATE TABLE IF NOT EXISTS observations "
                        "(id TEXT, stage TEXT, payload TEXT, PRIMARY KEY(id, stage))")
        self.db.commit()

    def record(self, candidate, stage, payload):
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO observations VALUES (?, ?, ?)",
                            (candidate, stage, encoded))

    def get(self, candidate, stage):
        row = self.db.execute("SELECT payload FROM observations WHERE id=? AND stage=?",
                              (candidate, stage)).fetchone()
        return json.loads(row[0]) if row else None

    def rows(self, stage):
        return [json.loads(row[0]) for row in self.db.execute(
            "SELECT payload FROM observations WHERE stage=? ORDER BY id", (stage,))]

    def close(self):
        self.db.close()


def save_checkpoint(path, model, optimizer=None, metadata=None):
    payload = {"model": model.state_dict(), "torch_rng": torch.get_rng_state(),
               "python_rng": random.getstate(), "metadata": metadata or {}}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return fingerprint(payload["metadata"])


def load_checkpoint(path, model, optimizer=None, restore_rng=True):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        torch.set_rng_state(payload["torch_rng"])
        random.setstate(payload["python_rng"])
    return payload["metadata"]
