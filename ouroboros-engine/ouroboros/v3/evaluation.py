"""Independent retraining, equal-update baselines and conservative promotion."""
import math
import statistics
import time

from .benchmarks import Baseline, evaluate, train_steps
from .experiments import isolated_seed
from .neurons import SequenceModel, seed_expressions


def interval(values):
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError("Finite observations required")
    mean = statistics.mean(values)
    # Conservative normal approximation; small samples must not establish a discovery.
    error = 1.96 * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else float("inf")
    return {"mean": mean, "low": mean - error if math.isfinite(error) else None,
            "high": mean + error if math.isfinite(error) else None, "n": len(values)}


def compare_models(expression, hidden, steps, seeds, batch=24, length=12):
    records = []
    factories = {"candidate": lambda: SequenceModel(3, hidden, 1, expression),
                 "rate_ablation": lambda: SequenceModel(3, hidden, 1, seed_expressions()["rate"])}
    factories.update({kind: (lambda kind=kind: Baseline(kind, hidden)) for kind in ("mlp", "rnn", "gru", "lstm")})
    for name, factory in factories.items():
        observations = []
        for seed in seeds:
            with isolated_seed(seed):
                model = factory()
                started = time.perf_counter()
                train_steps(model, steps, seed, batch, length)
                test = evaluate(model, "test", seed, batch, length)
                transfer = evaluate(model, "transfer", seed, batch, length + 4)
                observations.append({"seed": seed, "test": test, "transfer": transfer,
                                     "seconds": time.perf_counter() - started,
                                     "parameters": sum(p.numel() for p in model.parameters())})
        records.append({"name": name, "runs": observations,
                        "test_interval": interval([r["test"]["mean"] for r in observations])})
    return records


def promotion_decision(records, min_seeds=5, margin=0.02):
    by_name = {record["name"]: record for record in records}
    candidate = by_name["candidate"]
    if candidate["test_interval"]["n"] < min_seeds:
        return {"accepted": False, "reason": "insufficient_independent_seeds"}
    # Paired differences; Bonferroni-style conservative z=3.0 across five controls.
    differences = {}
    for name, baseline in by_name.items():
        if name == "candidate":
            continue
        others = {run["seed"]: run for run in baseline["runs"]}
        delta = [others[r["seed"]]["test"]["mean"] - r["test"]["mean"] for r in candidate["runs"]]
        mean = statistics.mean(delta)
        error = 3.0 * statistics.stdev(delta) / math.sqrt(len(delta))
        threshold = margin * baseline["test_interval"]["mean"]
        differences[name] = {"mean_improvement": mean, "conservative_low": mean - error,
                             "required_improvement": threshold}
    accepted = bool(differences) and all(d["conservative_low"] > d["required_improvement"] for d in differences.values())
    return {"accepted": accepted, "reason": "replicated_improvement" if accepted else "improvement_not_established",
            "comparisons": differences,
            "scope": "synthetic sequence tasks only; equal optimizer updates, measured parameters and runtime"}
