import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from ouroboros.config import ROOT, read_yaml, resolve_device


def run():
    from ouroboros.simulation import simulate
    parser = argparse.ArgumentParser(description="Executa o ecossistema experimental Ouroboros")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ecosystem_config.yaml")
    parser.add_argument("--primitives", type=Path, default=ROOT / "configs/neuron_primitives.yaml")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/bootstrap_code/tasks.json")
    parser.add_argument("--pace", type=float, default=0, help="Pausa por passo, em segundos, para observação ao vivo")
    args = parser.parse_args()
    config = read_yaml(args.config)
    for key in ("steps", "device"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.output:
        config["output_dir"] = str(args.output)
    if args.pace < 0:
        parser.error("--pace não pode ser negativo")
    simulate(config, read_yaml(args.primitives), args.tasks, args.checkpoint, args.pace)


def bootstrap():
    from ouroboros.agents.programmer_agent import CodePolicy, EXPRESSIONS, load_tasks
    from ouroboros.sandbox import SafeExecutor
    from ouroboros.simulation import atomic_json
    parser = argparse.ArgumentParser(description="Treina o seletor de expressões aritméticas")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/programmer_bootstrap.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    config = read_yaml(args.config)
    epochs = args.epochs if args.epochs is not None else config["epochs"]
    if epochs < 1 or config["context_size"] != 6:
        parser.error("É necessário epochs >= 1 e context_size = 6")
    device = resolve_device(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    tasks = load_tasks(ROOT / config["task_file"])
    executor = SafeExecutor()
    for task in tasks:
        reference = executor.execute(task["expression"], task["cases"] + task["validation_cases"])
        if not reference.ok or reference.error > 1e-6:
            raise ValueError(f"Referência inválida no dataset: {task['name']}")
    model = CodePolicy(config["context_size"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    features = torch.tensor([t["features"] for t in tasks], dtype=torch.float32, device=device)
    labels = torch.tensor([EXPRESSIONS.index(t["expression"]) for t in tasks], device=device)
    for _ in range(epochs):
        loss = F.cross_entropy(model(features), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        choices = model(features).argmax(-1).tolist()
    checks = []
    for task, choice in zip(tasks, choices):
        expression = EXPRESSIONS[choice]
        result = executor.execute(expression, task["validation_cases"])
        checks.append({"task": task["name"], "expression": expression,
                       "passed": result.ok and result.error < 1e-6, "error": result.error})
    output = args.output or ROOT / config["checkpoint"]
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": model.state_dict(), "expressions": EXPRESSIONS, "config": config}, output)
    report = {"epochs": epochs, "loss": loss.item(), "passed": sum(c["passed"] for c in checks),
              "total": len(checks), "checks": checks,
              "scope": "Mesmas seis tarefas, novos valores numéricos; não mede generalização para novas tarefas."}
    atomic_json(output.with_suffix(".json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["passed"] != report["total"]:
        raise SystemExit(1)


def dashboard():
    from ouroboros.dashboard import serve
    parser = argparse.ArgumentParser(description="Painel de métricas sem dependências web adicionais")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    serve(args.output, args.host, args.port)
