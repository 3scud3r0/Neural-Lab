import math
from pathlib import Path
import torch
import yaml


ROOT = Path(__file__).resolve().parent.parent


def read_yaml(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Configuração inválida: {path}")
    return value


def validate(config, primitives):
    for name in ("population", "hidden_size", "signal_size", "steps"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} precisa ser inteiro positivo")
    if config["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("device deve ser auto, cpu ou cuda")
    for value in (config["dt"], config["learning_rate"], config["world"]["size"],
                  config["world"]["communication_radius"], config["world"]["harvest_radius"]):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Tempos, taxas e dimensões precisam ser positivos e finitos")
    if not 0 <= config["world"]["friction"] <= 1:
        raise ValueError("friction precisa estar entre 0 e 1")
    if type(config["world"]["resource_count"]) is not int or config["world"]["resource_count"] < 1:
        raise ValueError("resource_count precisa ser inteiro positivo")
    if type(config["programmer"]["interval"]) is not int or config["programmer"]["interval"] < 1:
        raise ValueError("interval precisa ser inteiro positivo")
    if not 0 < config["physiology"]["initial_energy"] <= 1:
        raise ValueError("initial_energy precisa estar entre 0 (exclusivo) e 1")
    if not 0 < primitives["liquid"]["tau_min"] <= primitives["liquid"]["tau_max"]:
        raise ValueError("Constantes de tempo inválidas")
    for name in ("plastic", "ttt"):
        if not 0 <= primitives[name]["decay"] <= 1 or primitives[name]["learning_rate"] < 0:
            raise ValueError("Taxas de plasticidade inválidas")


def resolve_device(request):
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA solicitada, mas indisponível. Use --device cpu ou auto.")
    return torch.device("cuda" if request == "auto" and torch.cuda.is_available() else
                        "cpu" if request == "auto" else request)
