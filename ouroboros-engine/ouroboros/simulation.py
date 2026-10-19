import json
import os
from pathlib import Path
import random
import time
import torch
from ouroboros.agents.social_agent import SocialAgent
from ouroboros.agents.programmer_agent import ProgrammerAgent, load_tasks
from ouroboros.config import resolve_device, validate
from ouroboros.environment import SharedWorld
from ouroboros.physiology.free_energy import predictive_loss


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def simulate(config, primitives, task_file, checkpoint=None, pace=0.0, quiet=False):
    validate(config, primitives)
    device = resolve_device(config["device"])
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    # Matrizes pequenas não se beneficiam de dezenas de threads na CPU.
    torch.set_num_threads(1)
    tasks = load_tasks(task_file)
    world = SharedWorld(config["population"], device, config["world"])
    agents = SocialAgent(config, primitives, device)
    programmer = ProgrammerAgent(config, primitives, device, checkpoint)
    optimizer = torch.optim.Adam(agents.brain.parameters(), lr=config["learning_rate"])
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    # Evita misturar medições e checkpoints de execuções diferentes.
    if any(output.iterdir()):
        raise ValueError(f"Diretório de saída deve estar vazio: {output}")
    atomic_json(output / "config.json", {"ecosystem": config, "primitives": primitives,
                                        "task_file": str(task_file), "checkpoint": str(checkpoint),
                                        "torch": torch.__version__, "device": str(device)})
    baseline = torch.zeros(config["population"], device=device)
    latest_program = None
    started = time.perf_counter()
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for step in range(1, config["steps"] + 1):
            observation = agents.perceive(world)
            state = agents.think(observation, config["dt"])
            harvest, reward = agents.act(world, config["dt"])
            agents.broadcast()
            shock = torch.zeros_like(reward)
            if step % config["programmer"]["interval"] == 0:
                task = tasks[(step // config["programmer"]["interval"] - 1) % len(tasks)]
                latest_program = programmer.attempt(task)
                programs = output / "programs"
                programs.mkdir(exist_ok=True)
                stem = f"attempt_{programmer.count:04d}"
                atomic_json(programs / (stem + ".json"), latest_program)
                (programs / (stem + ".py")).write_text(
                    f"# Tarefa: {task['name']} | DSL aritmética; gerado pelo seletor de ASTs.\n"
                    f"def solve(x, y):\n    return {latest_program['expression']}\n", encoding="utf-8")
            dead = agents.metabolize(harvest, reward, shock)
            target = agents.perceive(world).detach()
            prediction_error = predictive_loss(state["prediction"], target, state["activity"])
            advantage = (reward - baseline).detach()
            loss = prediction_error - 0.1 * (state["log_prob"] * advantage).mean()
            signal_cost = config["physiology"]["signal_cost"] * state["signal"].detach().square().mean(-1)
            communication_advantage = (advantage.mean() - signal_cost).detach()
            loss = loss - 0.02 * (state["signal_log_prob"] * communication_advantage).mean()
            loss = loss + 0.01 * agents.physiology.metabolic_loss(state["activity"])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Perda não finita; interrompendo a simulação")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agents.brain.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                baseline.lerp_(reward, 0.05)
                if bool(dead.any()):
                    world.respawn(dead)
                    agents.physiology.respawn(dead)
                    agents.brain.reset_agents(dead)
                    agents.signals[dead] = 0
                    baseline[dead] = 0
            metrics = {"step": step, "device": str(device), "population": config["population"],
                       "energy": agents.physiology.energy.mean().item(),
                       "stress": agents.physiology.stress.mean().item(),
                       "entropy": agents.physiology.entropy.mean().item(),
                       "spike_fraction": state["spikes"].detach().mean().item(),
                       "reward": reward.mean().item(), "loss": loss.item(),
                       "prediction_error": prediction_error.item(),
                       "deaths": agents.physiology.deaths, "programmer": latest_program,
                       "elapsed_seconds": time.perf_counter() - started}
            log.write(json.dumps(metrics, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
            snapshot = dict(metrics, positions=world.positions.tolist(), resources=world.sources.tolist(),
                            obstacles=world.obstacles.tolist(), obstacle_radius=config["world"].get("obstacle_radius", 0.35),
                            stock=world.stock.tolist(), energies=agents.physiology.energy.tolist(),
                            signals=agents.signals.tolist(), world_size=config["world"]["size"],
                            activations=state["activity"][0].detach().tolist(),
                            completed=(step == config["steps"]))
            atomic_json(output / "latest.json", snapshot)
            if not quiet and (step == 1 or step % 25 == 0 or step == config["steps"]):
                print(f"step={step} energy={metrics['energy']:.3f} loss={metrics['loss']:.4f} device={device}", flush=True)
            if pace > 0:
                time.sleep(pace)
    torch.save({"brain": agents.brain.state_dict(), "programmer": programmer.policy.state_dict(),
                "config": config, "step": config["steps"]}, output / "ecosystem.pt")
    atomic_json(output / "summary.json", metrics)
    return metrics
