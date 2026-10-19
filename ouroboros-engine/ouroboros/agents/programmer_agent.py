import json
from pathlib import Path
import torch
from torch import nn
from torch.distributions import Categorical
from .base_organism import BaseOrganism
from ouroboros.sandbox import SafeExecutor
from ouroboros.sandbox.feedback_bridge import feedback_from_result


# Gramática finita e pública: o agente escolhe ASTs válidas, não inventa Python.
EXPRESSIONS = ["x", "y", "0", "x + y", "x - y", "x * y", "x * x",
               "(x + y) / 2", "abs(x - y)", "min(x, y)", "max(x, y)"]


class CodePolicy(nn.Module):
    def __init__(self, context_size=6):
        super().__init__()
        self.decoder = nn.Sequential(nn.Linear(context_size, 32), nn.Tanh(), nn.Linear(32, len(EXPRESSIONS)))

    def forward(self, features, activity=None):
        if activity is not None:
            features = features + 0.05 * activity.detach().mean()
        return self.decoder(features)


def load_tasks(path):
    tasks = json.loads(Path(path).read_text(encoding="utf-8"))
    if not tasks:
        raise ValueError("Dataset vazio")
    for task in tasks:
        if task["expression"] not in EXPRESSIONS or len(task["features"]) != 6:
            raise ValueError("Tarefa incompatível com a gramática/contexto do protótipo")
        if not task.get("cases") or not task.get("validation_cases"):
            raise ValueError("Cada tarefa precisa de treino e validação separados")
    return tasks


class ProgrammerAgent(BaseOrganism):
    """Organismo com decoder de ASTs e reforço online a partir de testes.

    O ciclo ecológico usa SocialAgent em lote; esta subclasse fornece um
    organismo individual de programação e converte feedback em fisiologia.
    """

    def __init__(self, config, primitives, device, checkpoint=None, syntax_penalty=1.0):
        individual = dict(config, population=1)
        super().__init__(individual, primitives, device)
        self.device = device
        self.policy = CodePolicy().to(device)
        self.syntax_penalty = syntax_penalty
        if checkpoint and not Path(checkpoint).is_file():
            raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint}")
        if checkpoint:
            saved = torch.load(checkpoint, map_location=device, weights_only=True)
            if saved["expressions"] != EXPRESSIONS:
                raise ValueError("Checkpoint usa outra gramática")
            self.policy.load_state_dict(saved["policy"])
            self.syntax_penalty = saved.get("config", {}).get("syntax_penalty", syntax_penalty)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.003)
        self.executor = SafeExecutor(config["programmer"]["timeout_seconds"], config["programmer"]["memory_mb"])
        self.count, self.solved = 0, 0

    def perceive(self, world=None):
        return torch.zeros(1, self.brain.predictor.out_features, device=self.device)

    def attempt(self, task):
        observation = self.perceive()
        observation[0, 5] = self.physiology.energy[0]
        observation[0, 6] = self.physiology.stress[0]
        self.think(observation, 0.1)
        features = torch.tensor(task["features"], device=self.device, dtype=torch.float32)
        distribution = Categorical(logits=self.policy(features, self.last["activity"]))
        choice = distribution.sample()
        expression = EXPRESSIONS[int(choice.item())]
        result = self.executor.execute(expression, task["validation_cases"])
        feedback = feedback_from_result(result, self.syntax_penalty)
        loss = -distribution.log_prob(choice) * feedback.reward - 0.01 * distribution.entropy()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        reward = torch.tensor([feedback.reward], device=self.device)
        shock = torch.tensor([feedback.shock], device=self.device)
        harvest = torch.tensor([max(0, feedback.reward) * 0.2], device=self.device)
        dead = self.metabolize(harvest, reward, shock)
        if bool(dead.any()):
            self.physiology.respawn(dead)
            self.brain.reset_agents(dead)
        self.count += 1
        self.solved += int(feedback.solved)
        return {"task": task["name"], "expression": expression, "reward": feedback.reward,
                "solved": feedback.solved, "error": result.error, "message": result.message,
                "elapsed_ms": result.elapsed_ms, "wall_ms": result.wall_ms, "memory_kb": result.memory_kb,
                "returncode": result.returncode, "energy": float(self.physiology.energy[0]),
                "stress": float(self.physiology.stress[0]), "attempts": self.count, "solved_total": self.solved}
