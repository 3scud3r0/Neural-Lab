"""CLI, integrated experiments, read-only dashboard and source-only delivery."""
import argparse
import hashlib
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch

from . import MODULES, SearchConfig, VERSION
from .architecture import HeterogeneousBrain, Region, resize_region, search_architectures
from .benchmarks import evaluate, train_steps
from .evaluation import compare_models, promotion_decision
from .experiments import atomic_json, isolated_seed, seed_all
from .neuron_dsl import parse
from .neurons import SequenceModel, seed_expressions
from .programming import Programmer
from .research import Hypothesis, ResearchController
from .search import NeuronSearch
from .selfmod import RevisionWorkspace
from .society import Population
from .world import ConstructionWorld, WorldConfig

ROOT = Path(__file__).resolve().parents[2]


def integrated_checks(directory, expression, seed=42):
    from ..v2.programming.tasks import make_task
    from ..v2.vm import assemble
    directory = Path(directory)
    with isolated_seed(seed):
        brain = HeterogeneousBrain(3, 1, [Region("first", 6, expression),
                                         Region("second", 4, seed_expressions()["rate"])])
        optimizer = torch.optim.Adam(brain.parameters(), lr=0.01)
        values = torch.randn(4, 3)
        prediction, states = brain(values)
        prediction.square().mean().backward()
        optimizer.step()
        optimizer, states = resize_region(brain, "first", 9, optimizer, states)
        optimizer, states = resize_region(brain, "first", 7, optimizer, states)
        optimizer.zero_grad()
        output, _ = brain(values, states)
        output.square().mean().backward()
        optimizer.step()
        morphology = {"sizes": [6, 9, 7], "finite": bool(torch.isfinite(output).all()),
                      "optimizer_states": len(optimizer.state)}
        architecture = search_architectures([expression, *seed_expressions().values()], candidates=6, steps=4, seed=seed)
        population = Population(2, expression, hidden=8)
        environment = ConstructionWorld(WorldConfig(horizon=16))
        training = [population.episode(environment, seed + i, train=True) for i in range(4)]
        interventions = {mode: [population.episode(environment, seed + 100 + i, intervention=mode)
                                for i in range(4)] for mode in ("intact", "muted", "shuffled")}
        distillation = population.distill(0, 1, torch.randn(8, 16), steps=3)
        child = population.reproduce(0)
        society = {"training": training, "interventions": interventions,
                   "distillation_losses": distillation, "child_generation": child.generation,
                   "claim": "execution and learning-path validation; competence not established"}
        programmer = Programmer(directory / "tools")
        task = make_task("sum", seed)
        compiled = programmer.compile_and_test("def solve(x, y):\n    return x + y\n", task)
        promotion = programmer.promote(0, "sum", compiled["program"], task)
        called = programmer.library.call(0, (17, -9))
        wrong = programmer.debug(assemble("PUSH 0\nOUTPUT\nHALT"), task.validation)
        programming = {"accepted": promotion["accepted"], "call_outputs": called.outputs,
                       "diagnosed_failures": len(wrong)}
        workspace = RevisionWorkspace(directory / "revisions", {"benchmark": "sequence-v1", "min_seeds": 5})
        proposal = workspace.propose(expression, parent=workspace.state["active"])
        rejection = workspace.evaluate_and_promote(proposal, lambda expr, contract:
                            {"accepted": False, "reason": "independent_evidence_required"})
        research = ResearchController([Hypothesis("long_memory", "Does the equation retain delayed input?"),
                                       Hypothesis("noise", "Does the equation filter noisy input?")], trials=4)

        def execute(hypothesis, trial):
            model = SequenceModel(3, 6, 1, expression)
            train_steps(model, 4, seed + trial)
            result = evaluate(model, "validation", seed + trial)
            task_name = "delay" if hypothesis.name == "long_memory" else "denoise"
            loss = result["tasks"][task_name]
            return -loss, {"validation_loss": loss, "task": task_name, "trial": trial}

        research_report = research.run(execute, directory / "research.json")
        return {"morphology": morphology, "architecture": architecture, "society": society, "programming": programming,
                "selfmod": rejection, "research": research_report}


def run_experiment(directory, config, seeds=5):
    if seeds < 1:
        raise ValueError("At least one replication seed required")
    directory = Path(directory)
    seed_all(config.seed)
    started = time.perf_counter()
    search = NeuronSearch(directory / "search", config)
    try:
        discovery = search.run()
    finally:
        search.store.close()
    expression = parse(discovery["expression"])
    records = compare_models(expression, config.hidden, config.screening_steps + config.promotion_steps,
                             [config.seed + 1000 + i for i in range(seeds)], config.batch, config.length)
    decision = promotion_decision(records)
    # Candidate was selected only on validation. Test evidence is an audit, not a new search signal.
    workspace = RevisionWorkspace(directory / "accepted-neurons", {"benchmark": "sequence-v1", "min_seeds": 5})
    proposal = workspace.propose(expression, parent=workspace.state["active"])
    workspace.evaluate_and_promote(proposal, lambda expr, contract: decision)
    integrations = integrated_checks(directory / "integration", expression, config.seed)
    report = {"version": VERSION, "modules": list(MODULES), "config": config.to_dict(),
              "search": discovery, "independent_comparison": records, "promotion": decision,
              "integration": integrations, "seconds": time.perf_counter() - started,
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "device": "cpu", "platform": platform.platform()},
              "limitations": ["Synthetic tasks and a small world; no general-intelligence conclusion.",
                              "Equal optimizer updates; model parameter counts and runtime differ.",
                              "Neuron search is resumed from cached candidates; interrupted candidate restarts.",
                              "Society learning is short and does not establish spontaneous language or competent construction.",
                              "Self-modification changes DSL neuron definitions and validated VM tools, not host source.",
                              "The hypothesis scheduler is UCB over declared hypotheses, not an autonomous scientist."]}
    atomic_json(directory / "report.json", report)
    return report


def source_paths(root=ROOT):
    extensions = {".py", ".sh", ".js", ".ts", ".html", ".css", ".c", ".h", ".s", ".ovm"}
    paths = []
    for name in ("ouroboros", "scripts", "tests", "data", "third_party"):
        for path in (root / name).rglob("*"):
            if (path.is_file() and not path.is_symlink() and path.suffix in extensions
                    and not any(part in {"__pycache__", "build", "dist", ".venv"} or part.endswith(".egg-info") for part in path.parts)):
                paths.append(path)
    return sorted(paths)


def export_code(destination, root=ROOT):
    destination, root = Path(destination).resolve(), Path(root).resolve()
    paths = source_paths(root)
    if destination in [path.resolve() for path in paths]:
        raise ValueError("Source export cannot overwrite source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    chunks, inventory = [], []
    for path in paths:
        contents = path.read_text(encoding="utf-8")
        # Only source and source comments; no report, prose wrapper or test output.
        chunks.append(f"\n# SOURCE: {path.relative_to(root).as_posix()}\n" + contents
                      + ("" if contents.endswith("\n") else "\n"))
        inventory.append({"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    destination.write_text("".join(chunks), encoding="utf-8")
    return {"files": len(paths), "bytes": destination.stat().st_size,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "inventory": inventory}


def run_tests(destination, report_path=None):
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    text = f"COMMAND: {' '.join(command)}\nEXIT_CODE: {process.returncode}\n\n"
    text += process.stdout + process.stderr
    if report_path:
        report = json.loads(Path(report_path).read_text())
        text += "\nINTEGRATION AND RESEARCH RESULTS\n" + json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return process.returncode


def dashboard_handler(directory):
    directory = Path(directory)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ("/", "/api/report"):
                self.send_error(404)
                return
            path = directory / "report.json"
            if not path.is_file():
                self.send_error(404, "No experiment report")
                return
            payload = path.read_text()
            if self.path == "/":
                payload = ('<!doctype html><html lang="pt-BR"><meta charset="utf-8">'
                           '<meta name="viewport" content="width=device-width,initial-scale=1">'
                           '<title>Ouroboros 3 — pesquisa</title><style>body{font:16px system-ui;'
                           'max-width:1100px;margin:24px auto;padding:16px;background:#101827;color:#e5edf9}'
                           'pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
                           '<h1>Ouroboros 3 — resultados medidos</h1><pre>' + html.escape(payload) + '</pre></html>')
            body = payload.encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json" if self.path == "/api/report" else "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--candidates", type=int, default=1024)
    run.add_argument("--screening-steps", type=int, default=3)
    run.add_argument("--promotion-steps", type=int, default=24)
    run.add_argument("--finalists", type=int, default=8)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--seeds", type=int, default=5)
    run.add_argument("--max-seconds", type=float, default=1800)
    export = commands.add_parser("export")
    export.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("test")
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--report", type=Path)
    serve = commands.add_parser("dashboard")
    serve.add_argument("--output", type=Path, required=True)
    serve.add_argument("--port", type=int, default=8083)
    args = parser.parse_args(argv)
    if args.command == "run":
        config = SearchConfig(candidates=args.candidates, screening_steps=args.screening_steps,
                              promotion_steps=args.promotion_steps, finalists=args.finalists,
                              seed=args.seed, max_seconds=args.max_seconds)
        report = run_experiment(args.output, config, args.seeds)
        print(json.dumps({"screened": report["search"]["screened"], "promotion": report["promotion"],
                          "seconds": report["seconds"]}, indent=2))
    elif args.command == "export":
        result = export_code(args.output)
        print(json.dumps({k: v for k, v in result.items() if k != "inventory"}))
    elif args.command == "test":
        return run_tests(args.output, args.report)
    elif args.command == "dashboard":
        server = ThreadingHTTPServer(("127.0.0.1", args.port), dashboard_handler(args.output))
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
