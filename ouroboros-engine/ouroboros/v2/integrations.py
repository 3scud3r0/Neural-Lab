"""Inventário, integridade e ativação das peças upstream incluídas na entrega."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

from .common import ROOT


def verify_upstream(directory=None):
    root = Path(directory or ROOT / "third_party")
    lock = json.loads((root / "UPSTREAM_LOCK.json").read_text())
    reports = []
    for project in lock["projects"]:
        missing, mismatched = [], []
        for relative, digest in project["files"].items():
            path = root / project["name"] / relative
            if not path.is_file(): missing.append(relative)
            elif hashlib.sha256(path.read_bytes()).hexdigest() != digest: mismatched.append(relative)
        reports.append({"name": project["name"], "version": project["version"],
                        "commit": project["commit"], "repository": project["repository"],
                        "license": project["license"], "files": len(project["files"]),
                        "missing": missing, "mismatched": mismatched,
                        "ok": not missing and not mismatched})
    return reports


def doctor():
    import torch
    packages = {}
    for name in ("torch", "numpy", "PyYAML", "ncps", "neat-python", "pettingzoo", "gymnasium"):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    from ncps.torch import CfCCell
    import neat
    from pettingzoo import ParallelEnv
    return {"packages": packages, "cuda_available": torch.cuda.is_available(),
            "upstream": verify_upstream(),
            "integrations": {"CfCCell": CfCCell.__module__, "NEAT": neat.__file__, "ParallelEnv": ParallelEnv.__module__}}
