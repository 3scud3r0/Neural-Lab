"""Equation evolution, successive promotion and a diverse Pareto archive."""
from dataclasses import asdict
import random
import time

import torch

from . import Score, fingerprint
from .benchmarks import evaluate, train_steps
from .experiments import Budget, ExperimentStore, isolated_seed, load_checkpoint, save_checkpoint
from .neuron_dsl import ARITY, Expr, parse, run
from .neurons import SequenceModel, seed_expressions


def random_expression(rng, depth=3):
    if depth <= 0 or rng.random() < 0.3:
        if rng.random() < 0.2:
            return Expr("const", value=round(rng.uniform(-2, 2), 4))
        return Expr(rng.choice(("x", "h", "m", "a", "p")))
    op = rng.choice(tuple(ARITY))
    return Expr(op, tuple(random_expression(rng, depth - 1) for _ in range(ARITY[op])))


def mutate(expression, rng):
    if not expression.args or rng.random() < 0.35:
        return random_expression(rng, rng.randint(1, 3))
    children = list(expression.args)
    index = rng.randrange(len(children))
    children[index] = mutate(children[index], rng)
    proposal = Expr(expression.op, tuple(children), expression.value)
    try:
        return parse(proposal.to_dict())
    except ValueError:
        return expression


def crossover(left, right, rng):
    if not left.args:
        return right
    children = list(left.args)
    children[rng.randrange(len(children))] = right
    try:
        return parse(Expr(left.op, tuple(children)).to_dict())
    except ValueError:
        return left


def pareto(scores):
    valid = [score for score in scores if score.valid]
    return [score for score in valid if not any(
        other.validation_loss <= score.validation_loss and other.parameters <= score.parameters
        and (other.validation_loss < score.validation_loss or other.parameters < score.parameters)
        for other in valid)]


def behavior_signature(expression):
    """A finite probe descriptor, not a proof of functional equivalence."""
    x = torch.linspace(-2, 2, 17)
    signals = {"x": x, "h": x.flip(0) * 0.7, "m": x.sin(),
               "a": x.abs() * 0.2, "p": x * 1.7}
    with torch.no_grad():
        values = run(expression, signals)
    return fingerprint([round(value, 3) for value in values.tolist()])


def diverse_archive(expressions, scores):
    archive = {}
    for score in scores:
        if not score.valid:
            continue
        expr = expressions[score.candidate]
        key = (expr.op, min(expr.size // 5, 6))
        if key not in archive or score.validation_loss < archive[key].validation_loss:
            archive[key] = score
    return list(archive.values())


class NeuronSearch:
    def __init__(self, directory, config):
        self.config = config
        self.store = ExperimentStore(directory, config.to_dict())
        self.directory = self.store.directory

    def _fit(self, expr, steps, stage, parent=None):
        cfg, identity = self.config, expr.identity
        cached = self.store.get(identity, stage)
        if cached:
            return Score(**cached)
        started = time.perf_counter()
        parameters = 0
        try:
            with isolated_seed(cfg.seed):
                model = SequenceModel(3, cfg.hidden, 1, expr)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
                if parent:
                    load_checkpoint(parent, model, optimizer, restore_rng=False)
                start = cfg.screening_steps if parent else 0
                train_steps(model, steps, cfg.seed, cfg.batch, cfg.length, optimizer, start)
                loss = evaluate(model, "validation", cfg.seed, cfg.batch, cfg.length)["mean"]
                parameters = sum(p.numel() for p in model.parameters())
                score = Score(identity, loss, parameters, time.perf_counter() - started, stage)
                save_checkpoint(self.directory / "weights" / f"{identity}-{stage}.pt", model, optimizer,
                                {"expression": expr.to_dict(), "steps": start + steps, "parent": str(parent) if parent else None})
        except (ValueError, RuntimeError, FloatingPointError) as error:
            score = Score(identity, float("inf"), parameters, time.perf_counter() - started,
                          stage, f"{type(error).__name__}: {error}")
        self.store.record(identity, "expression", expr.to_dict())
        # Store null for invalid numerical scores; restore infinity only in memory.
        payload = asdict(score)
        if not score.valid:
            payload["validation_loss"] = 1e30
        self.store.record(identity, stage, payload)
        return score

    def run(self):
        cfg, rng = self.config, random.Random(self.config.seed)
        budget = Budget(cfg.max_seconds, cfg.candidates + cfg.finalists)
        expressions, scores = {}, []
        initial = list(seed_expressions().values())
        attempts, exhausted = 0, False
        while len(scores) < cfg.candidates:
            attempts += 1
            if attempts > cfg.candidates * 200:
                raise RuntimeError("Could not generate enough unique equations")
            if len(scores) < min(len(initial), cfg.candidates):
                expr = initial[len(scores)]
            elif rng.random() < 0.65 and scores:
                parents = sorted((score for score in scores if score.valid), key=lambda s: s.validation_loss)
                pool = parents[:max(1, len(parents) // 4)]
                expr = mutate(expressions[rng.choice(pool).candidate], rng) if pool else random_expression(rng)
                if pool and rng.random() < 0.2:
                    expr = crossover(expr, expressions[rng.choice(pool).candidate], rng)
            else:
                expr = random_expression(rng, rng.randint(2, 4))
            if expr.identity in expressions:
                continue
            try:
                budget.claim()
            except TimeoutError:
                exhausted = True
                break
            expressions[expr.identity] = expr
            scores.append(self._fit(expr, cfg.screening_steps, "screen"))
            if len(scores) % 128 == 0:
                print(f"screened {len(scores)}/{cfg.candidates}", flush=True)
        finalists = sorted((s for s in scores if s.valid), key=lambda s: s.validation_loss)[:cfg.finalists]
        promoted = []
        for score in finalists:
            try:
                budget.claim()
            except TimeoutError:
                exhausted = True
                break
            parent = self.directory / "weights" / f"{score.candidate}-screen.pt"
            promoted.append(self._fit(expressions[score.candidate], cfg.promotion_steps, "promoted", parent))
        winners = sorted((s for s in promoted if s.valid), key=lambda s: s.validation_loss)
        if not winners:
            raise RuntimeError("No valid promoted candidates within budget")
        return {"winner": winners[0].to_dict(), "screened": len(scores), "promoted": len(promoted),
                "unique_equations": len(expressions), "budget_exhausted": exhausted,
                "distinct_probe_behaviors": len({behavior_signature(expr) for expr in expressions.values()}),
                "pareto": [s.to_dict() for s in pareto(promoted)],
                "diverse_archive": [s.to_dict() for s in diverse_archive(expressions, scores)],
                "expression": expressions[winners[0].candidate].to_dict()}
