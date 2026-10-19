"""Orquestra experimentos reais e publica seus resultados, incluindo falhas."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from .common import Journal, atomic_json, device_for, new_run, seed_everything
from .communication import CommunicationGame, GameConfig, train_communication
from .evolution import MorphogenicBrain, MorphologyConfig
from .evolution.neat_control import evolve_controller
from .integrations import doctor
from .programming.evaluation import evaluate
from .programming.genetics import GeneticProgrammer
from .programming.library import ToolLibrary
from .programming.organism import ProgramOrganism
from .programming.tasks import affine_task, curriculum, make_task
from .programming.token_model import TokenProgrammer, supervised_bootstrap
from .vm import assemble, disassemble
from .world import ArenaConfig
from .world.training import train_world


PROFILES = {
    "quick": {"program_epochs": 100, "communication_steps": 150, "world_episodes": 4,
              "genetic_generations": 60, "neat_generations": 5},
    "full": {"program_epochs": 400, "communication_steps": 1500, "world_episodes": 24,
             "genetic_generations": 120, "neat_generations": 25},
}


def bootstrap_programmer(directory, epochs=400, seed=42, device="cpu"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    tasks = curriculum(seed)
    model = TokenProgrammer().to(device)
    journal = Journal(directory)
    history = supervised_bootstrap(model, tasks, epochs=epochs, seed=seed,
                                   callback=lambda record: journal.record("program_bootstrap", **record))
    reports, generated = [], directory / "generated"
    generated.mkdir(exist_ok=True)
    model.eval()
    with torch.no_grad():
        for task in tasks:
            program, _, _, tokens = model.generate(task.context(device), greedy=True)
            result = evaluate(program, task.test) if program else None
            record = {"task": task.name, "valid": program is not None, "test": asdict(result) if result else None,
                      "solved": bool(result and result.solved), "tokens": len(tokens),
                      "test_unique_inputs": len({(case.inputs, case.array) for case in task.test}),
                      "instructions": len(program.code) if program else 0}
            reports.append(record)
            if program:
                (generated / (task.name + ".ovm")).write_text(disassemble(program), encoding="utf-8")
                atomic_json(generated / (task.name + ".json"), program.to_dict())
    # Avalia também novos conjuntos de exemplos públicos; não seleciona pesos com eles.
    context_checks = []
    with torch.no_grad():
        for task in tasks:
            for repetition in range(5):
                context_seed = seed + 50_000 + repetition
                public = make_task(task.name, context_seed, counts=(8, 0, 0))
                candidate, *_ = model.generate(public.context(device), greedy=True)
                result = evaluate(candidate, task.test) if candidate else None
                context_checks.append({"task": task.name, "context_seed": context_seed,
                                       "solved": bool(result and result.solved)})
    report = {"history": history, "tasks": reports, "solved": sum(r["solved"] for r in reports),
              "total": len(reports), "heldout_scope": "Novas entradas e comprimentos para famílias de algoritmos demonstradas"}
    report["public_context_generalization"] = {"solved": sum(r["solved"] for r in context_checks),
                                                "total": len(context_checks), "checks": context_checks}
    atomic_json(directory / "report.json", report)
    # Salva apenas pesos e tensores, carregáveis com weights_only=True.
    torch.save({"schema": 2, "model": model.state_dict(), "hidden": model.hidden_size,
                "context_size": model.context_size, "seed": seed}, directory / "token_programmer.pt")
    atomic_json(directory / "public_examples.json", [dict(name=t.name, cases=[c.to_dict() for c in t.train]) for t in tasks])
    return model, report


def run_synthesis(directory, generations=120, seed=42, coefficients=(3, 2, -1), task_override=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    journal = Journal(directory)
    # O solver acessa task.train; o programa de referência nunca é fornecido à busca.
    task = task_override or affine_task(*coefficients, seed=seed)
    if any(len(case.inputs) != 2 or case.array for case in task.train):
        raise ValueError("A busca genética de árvores usa dois escalares; use generate para bytecode com arrays e loops")
    solver = GeneticProgrammer(generations=generations, seed=seed)
    candidate, history = solver.search(task, callback=lambda r: journal.record("genetics", **r))
    validation, test = evaluate(candidate, task.validation), evaluate(candidate, task.test)
    (directory / "discovered.ovm").write_text(disassemble(candidate), encoding="utf-8")
    atomic_json(directory / "discovered.json", candidate.to_dict())
    atomic_json(directory / "examples.json", {"public_training": [c.to_dict() for c in task.train],
                                               "validation": [c.to_dict() for c in task.validation],
                                               "test": [c.to_dict() for c in task.test]})
    report = {"coefficients": None if task_override else coefficients, "task": task.name,
              "history": history, "validation": asdict(validation),
              "test": asdict(test), "solved_test": test.solved, "candidate": candidate.to_dict(),
              "scope": "Síntese de composição afim sem seed de solução; não prova novidade científica"}
    atomic_json(directory / "report.json", report)
    return report


def structural_experiment(directory, seed=42, device="cpu"):
    seed_everything(seed)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    brain = MorphogenicBrain(MorphologyConfig(initial_nodes=12)).to(device)
    before = brain.topology()
    initial_parameters = sum(p.numel() for p in brain.parameters())
    for _ in range(2):
        brain.adapt(stress=1.0, energy=0.9)
    brain.adapt(stress=0.1, energy=0.1)
    rewiring = brain.rewire(3)
    # Um passo de treino após resize verifica que os novos Parameters participam.
    optimizer = torch.optim.Adam(brain.parameters(), lr=0.001)
    output = brain(torch.randn(8, 12, device=device))
    loss = output["thrust"].square().mean() + output["value"].square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    report = {"before": before, "after": brain.topology(), "events": brain.events,
              "parameter_count_before": initial_parameters,
              "parameter_count_after": sum(p.numel() for p in brain.parameters()),
              "loss_after_resize": float(loss.detach()), "rewiring": rewiring,
              "scope": "Regra estrutural explícita guiada por energia e estresse; não organogênese espontânea"}
    atomic_json(directory / "report.json", report)
    torch.save(brain.export(), directory / "morphology.pt")
    return report


def self_modification_experiment(model, directory, seed=42):
    directory = Path(directory)
    organism = ProgramOrganism(model, directory)
    task = make_task("sum", seed)
    baseline = assemble("INPUT 0\nINPUT 1\nADD\nNOP\nNOP\nOUTPUT\nHALT", "redundant_sum")
    first = organism.library.promote(0, "sum", baseline, task)
    with torch.no_grad():
        replacement, *_ = model.generate(task.context(next(model.parameters()).device), greedy=True)
    second = (organism.library.promote(0, "sum", replacement, task, require_improvement=True)
              if replacement else {"accepted": False, "reason": "invalid_neural_program"})
    called = organism.library.call(0, (17, -9))
    attempts = []
    for name in ("sum", "array_sum", "gcd", "factorial"):
        _, record = organism.attempt(make_task(name, seed), learn=True, greedy=False)
        attempts.append(record)
    report = {"initial_promotion": first, "neural_revision": second,
              "runtime_call": asdict(called), "runtime_expected": 8,
              "metabolism": asdict(organism.metabolism), "online_attempts": attempts,
              "scope": "O agente revisa ferramentas em bytecode; não altera a VM, testes ou Python do host"}
    torch.save({"model": model.state_dict(), "optimizer": organism.optimizer.state_dict(),
                "metabolism": asdict(organism.metabolism)}, directory / "online_programmer.pt")
    atomic_json(directory / "report.json", report)
    return report


def run_experiment(directory, profile="full", seed=42, device="auto", overrides=None):
    if profile not in PROFILES: raise ValueError("Perfil desconhecido")
    directory = new_run(Path(directory))
    configuration = dict(PROFILES[profile], **(overrides or {}))
    actual_device = device_for(device)
    journal, started = Journal(directory), time.monotonic()
    environment = doctor()
    if not all(record["ok"] for record in environment["upstream"]):
        raise RuntimeError("As fontes upstream diferem dos hashes registrados")
    atomic_json(directory / "environment.json", environment)
    atomic_json(directory / "config.json", dict(profile=profile, seed=seed, device=str(actual_device), **configuration))

    def stage(name):
        journal.record("stage", stage=name, completed=False)
        print(f"\nOuroboros 2 — {name}", flush=True)

    stage("programming")
    programmer, programming = bootstrap_programmer(directory / "programming", configuration["program_epochs"], seed, actual_device)
    stage("genetic_synthesis")
    synthesis = run_synthesis(directory / "synthesis", configuration["genetic_generations"], seed)
    stage("tool_self_modification")
    self_modification = self_modification_experiment(programmer, directory / "self_modification", seed)
    communication, languages = {}, {}
    for protocol in ("free", "factorized"):
        stage("communication_" + protocol)
        seed_everything(seed)
        language = CommunicationGame(GameConfig(protocol=protocol)).to(actual_device)
        communication[protocol] = train_communication(language, directory / ("communication_" + protocol),
                                                     steps=configuration["communication_steps"], seed=seed)
        languages[protocol] = language
    stage("structural_morphology")
    morphology = structural_experiment(directory / "morphology", seed, actual_device)
    stage("neat_evolution")
    _, neat_report = evolve_controller(directory / "neat", configuration["neat_generations"], seed)
    stage("cooperative_world")
    seed_everything(seed)
    _, world = train_world(languages["factorized"], directory / "world", configuration["world_episodes"],
                           ArenaConfig(batch=32, horizon=80), seed=seed)
    causal = communication["factorized"]["train_domain"]
    checks = {"all_demonstrated_program_families": programming["solved"] == programming["total"],
              "unseen_affine_composition": synthesis["solved_test"],
              "tool_revision_accepted": bool(self_modification["neural_revision"]["accepted"]),
              "communication_causal_effect": causal["intact"]["accuracy"] >= 0.85 and causal["causal_drop_muted"] >= 0.4,
              "actual_growth_and_pruning": any(e["added"] for e in morphology["events"]) and any(e["removed"] for e in morphology["events"]),
              "neat_structural_evolution": neat_report["structural_variation_observed"],
              "world_cooperation": world["evaluation"]["intact"]["success_rate"] > 0.35 and world["evaluation"]["solo"]["successes"] == 0}
    report = {"version": "2.0.0", "profile": profile, "seed": seed, "device": str(actual_device),
              "elapsed_seconds": time.monotonic() - started, "checks": checks,
              "programming": programming, "synthesis": synthesis, "self_modification": self_modification,
              "communication": communication, "morphology": morphology, "neat": neat_report,
              "world": world, "all_checks_passed": all(checks.values())}
    atomic_json(directory / "report.json", report)
    journal.record("completed", stage="completed", completed=True, checks=checks,
                   all_checks_passed=report["all_checks_passed"])
    return report
