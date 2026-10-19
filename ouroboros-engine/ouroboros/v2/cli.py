"""Interface unificada dos componentes executáveis da versão 2.0."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from .common import atomic_json, device_for, json_default, new_run, seed_everything


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ouroboros Engine 2.0 — laboratório executável")
    parser.add_argument("--version", action="version", version="2.0.0")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor_parser = commands.add_parser("doctor", help="Verifica pacotes e hashes das fontes upstream")
    experiment = commands.add_parser("experiment", help="Executa todos os experimentos e avaliações")
    experiment.add_argument("--output", type=Path, required=True)
    experiment.add_argument("--profile", choices=["quick", "full"], default="full")
    experiment.add_argument("--require-success", action="store_true")
    experiment.add_argument("--world-episodes", type=int)
    bootstrap = commands.add_parser("bootstrap", help="Treina o decoder de bytecode")
    bootstrap.add_argument("--output", type=Path, required=True)
    bootstrap.add_argument("--epochs", type=int, default=400)
    synthesis = commands.add_parser("synthesize", help="Busca uma composição a partir de exemplos numéricos")
    synthesis.add_argument("--output", type=Path, required=True)
    synthesis.add_argument("--generations", type=int, default=120)
    synthesis.add_argument("--coefficients", type=int, nargs=3, default=[3, 2, -1])
    synthesis.add_argument("--examples", type=Path, help="JSON de treino, validação e teste, sem código da solução")
    generate = commands.add_parser("generate", help="Gera candidatos de bytecode para uma nova especificação")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--examples", type=Path, required=True)
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--samples", type=int, default=16)
    language = commands.add_parser("communication", help="Treina o jogo de informação assimétrica")
    language.add_argument("--output", type=Path, required=True)
    language.add_argument("--steps", type=int, default=1500)
    language.add_argument("--protocol", choices=["free", "factorized"], default="factorized")
    language.add_argument("--resume", type=Path)
    morphology = commands.add_parser("morphology", help="Executa crescimento, poda e reconexão")
    morphology.add_argument("--output", type=Path, required=True)
    neat = commands.add_parser("neat", help="Evolui um controlador com NEAT-Python upstream")
    neat.add_argument("--output", type=Path, required=True)
    neat.add_argument("--generations", type=int, default=25)
    world = commands.add_parser("world", help="Treina o cérebro CfC/morfogênico no mundo cooperativo")
    world.add_argument("--output", type=Path, required=True)
    world.add_argument("--language", type=Path, required=True)
    world.add_argument("--episodes", type=int, default=24)
    world.add_argument("--batch", type=int, default=32)
    world.add_argument("--horizon", type=int, default=80)
    vm = commands.add_parser("vm", help="Executa assembly ou compila a DSL imperativa")
    vm.add_argument("source", type=Path)
    vm.add_argument("--compile", action="store_true")
    vm.add_argument("--inputs", type=float, nargs="*", default=[])
    vm.add_argument("--array", type=float, nargs="*", default=[])
    vm.add_argument("--fuel", type=int, default=4000)
    vm.add_argument("--trace", action="store_true")
    dashboard = commands.add_parser("dashboard", help="Painel da versão 2.0 e console da VM")
    dashboard.add_argument("--output", type=Path, required=True)
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8080)
    for sub in (experiment, bootstrap, synthesis, generate, language, morphology, neat, world):
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    args = parser.parse_args(argv)
    if hasattr(args, "seed"): seed_everything(args.seed)
    if args.command == "doctor":
        from .integrations import doctor
        report = doctor()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if all(p["ok"] for p in report["upstream"]) else 1
    if args.command == "dashboard":
        from .dashboard import serve
        serve(args.output, args.host, args.port)
        return 0
    if args.command == "vm":
        from .vm import Limits, Machine, assemble, compile_source
        source = args.source.read_text(encoding="utf-8")
        program = compile_source(source) if args.compile else assemble(source)
        result = Machine(Limits(fuel=args.fuel)).run(program, args.inputs, args.array, args.trace)
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return 0 if result.ok else 1
    device = device_for(args.device)
    if args.command == "experiment":
        from .experiment import run_experiment
        override = {"world_episodes": args.world_episodes} if args.world_episodes is not None else None
        report = run_experiment(args.output, args.profile, args.seed, args.device, override)
        print(json.dumps({"output": str(args.output), "checks": report["checks"]}, indent=2))
        return 1 if args.require_success and not report["all_checks_passed"] else 0
    directory = new_run(args.output)
    if args.command == "bootstrap":
        from .experiment import bootstrap_programmer
        _, report = bootstrap_programmer(directory, args.epochs, args.seed, device)
    elif args.command == "synthesize":
        from .experiment import run_synthesis
        from .programming.tasks import task_from_examples
        task = task_from_examples(args.examples) if args.examples else None
        report = run_synthesis(directory, args.generations, args.seed, tuple(args.coefficients), task)
    elif args.command == "generate":
        from .programming.generation import generate_candidates, load_programmer
        from .programming.tasks import task_from_examples
        model = load_programmer(args.checkpoint, device)
        report = generate_candidates(model, task_from_examples(args.examples), directory, args.samples)
    elif args.command == "communication":
        from .communication import CommunicationGame, GameConfig, train_communication
        model = CommunicationGame(GameConfig(protocol=args.protocol)).to(device)
        report = train_communication(model, directory, args.steps, seed=args.seed, resume=args.resume)
    elif args.command == "morphology":
        from .experiment import structural_experiment
        report = structural_experiment(directory, args.seed, device)
    elif args.command == "neat":
        from .evolution.neat_control import evolve_controller
        _, report = evolve_controller(directory, args.generations, args.seed)
    elif args.command == "world":
        from .communication.training import load_game
        from .world import ArenaConfig
        from .world.training import train_world
        model, _ = load_game(args.language, device)
        _, report = train_world(model, directory, args.episodes, ArenaConfig(batch=args.batch, horizon=args.horizon), seed=args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
