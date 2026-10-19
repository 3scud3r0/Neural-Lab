"""Córtex recorrente com alteração REAL das dimensões e do grafo de conexões."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from ncps.torch import CfCCell
import torch
from torch import nn
from torch.nn import functional as F

from ...primitives.spiking_gate import SpikingGate


@dataclass
class MorphologyConfig:
    inputs: int = 12
    initial_nodes: int = 24
    min_nodes: int = 8
    max_nodes: int = 64
    sensor_nodes: int = 16
    density: float = 0.25
    growth: int = 3
    stress_threshold: float = 0.35
    prune_energy: float = 0.35

    def __post_init__(self):
        if not 1 <= self.min_nodes <= self.initial_nodes <= self.max_nodes:
            raise ValueError("Limites de neurônios inválidos")
        if not 0 < self.density <= 1 or self.sensor_nodes < 1 or self.growth < 1 or self.inputs < 1:
            raise ValueError("Configuração da morfogênese inválida")


class MorphogenicBrain(nn.Module):
    def __init__(self, config=MorphologyConfig()):
        super().__init__()
        self.config = config
        n, sensor = config.initial_nodes, config.sensor_nodes
        # Implementação upstream de CfC, não uma aproximação renomeada.
        self.sensor = CfCCell(config.inputs, sensor, backbone_units=32, backbone_layers=1)
        self.input_weight = nn.Parameter(torch.randn(n, sensor) / math.sqrt(sensor))
        self.recurrent = nn.Parameter(torch.randn(n, n) * 0.1)
        self.bias = nn.Parameter(torch.zeros(n))
        self.motor = nn.Linear(n, 3)
        self.value = nn.Linear(n, 1)
        self.gate = SpikingGate(threshold=0.15)
        adjacency = torch.rand(n, n) < config.density
        adjacency.fill_diagonal_(False)
        self.register_buffer("adjacency", adjacency)
        self.register_buffer("utility", torch.zeros(n))
        self.node_ids = list(range(n))
        self.next_node_id = n
        self.stress_average = 0.0
        self.events = []

    @property
    def nodes(self): return self.input_weight.shape[0]

    def initial_state(self, batch):
        device = self.input_weight.device
        return {"sensor": torch.zeros(batch, self.config.sensor_nodes, device=device),
                "cortex": torch.zeros(batch, self.nodes, device=device),
                "fast": torch.zeros(batch, self.nodes, self.nodes, device=device)}

    def forward(self, observation, state=None, dt=0.15):
        if dt <= 0 or not math.isfinite(dt): raise ValueError("dt inválido")
        if state is None: state = self.initial_state(len(observation))
        if state["cortex"].shape != (len(observation), self.nodes):
            raise ValueError("Estado antigo após morfogênese; reinicie o estado no limite do episódio")
        sensor, _ = self.sensor(observation, state["sensor"], dt)
        previous, fast = state["cortex"], state["fast"]
        drive = F.linear(sensor, self.input_weight, self.bias)
        drive = drive + F.linear(previous, self.recurrent * self.adjacency)
        drive = drive + 0.1 * torch.bmm(fast, previous.unsqueeze(-1)).squeeze(-1)
        cortex = torch.tanh(drive)
        gated, spikes = self.gate(cortex)
        activation = 0.5 * (cortex + gated)
        motor = self.motor(activation)
        with torch.no_grad():
            next_fast = (0.97 * fast + 0.01 * cortex.detach().unsqueeze(-1) * previous.detach().unsqueeze(-2)).clamp(-1, 1)
            if self.training:
                self.utility.lerp_(activation.detach().abs().mean(0), 0.03)
        next_state = {"sensor": sensor.detach(), "cortex": cortex.detach(), "fast": next_fast.detach()}
        return {"thrust": motor[:, :2].tanh(), "grip_logit": motor[:, 2],
                "value": self.value(activation).squeeze(-1), "spikes": spikes,
                "activation": activation, "state": next_state}

    @torch.no_grad()
    def resize(self, keep, additions=0, reason="manual"):
        """Preserva submatrizes sobreviventes, aloca novos neurônios e atualiza IDs.

        A identidade dos Parameters muda. O chamador precisa recriar o otimizador
        e os estados recorrentes, sempre na fronteira entre episódios.
        """
        keep = list(keep)
        if len(set(keep)) != len(keep) or any(i < 0 or i >= self.nodes for i in keep):
            raise ValueError("Seleção de neurônios inválida")
        old_count, new_count = self.nodes, len(keep) + additions
        if not self.config.min_nodes <= new_count <= self.config.max_nodes:
            raise ValueError("Resize excede capacidade configurada")
        if additions < 0: raise ValueError("Número de adições inválido")
        device, dtype = self.input_weight.device, self.input_weight.dtype
        k = torch.tensor(keep, device=device, dtype=torch.long)
        preserved = len(keep)
        input_weight = torch.randn(new_count, self.config.sensor_nodes, device=device, dtype=dtype) / math.sqrt(self.config.sensor_nodes)
        recurrent = torch.randn(new_count, new_count, device=device, dtype=dtype) * 0.1
        bias = torch.zeros(new_count, device=device, dtype=dtype)
        input_weight[:preserved] = self.input_weight[k]
        recurrent[:preserved, :preserved] = self.recurrent[k][:, k]
        bias[:preserved] = self.bias[k]
        adjacency = torch.rand(new_count, new_count, device=device) < self.config.density
        adjacency[:preserved, :preserved] = self.adjacency[k][:, k]
        adjacency.fill_diagonal_(False)
        utility = torch.zeros(new_count, device=device)
        utility[:preserved] = self.utility[k]
        motor = nn.Linear(new_count, 3, device=device, dtype=dtype)
        value = nn.Linear(new_count, 1, device=device, dtype=dtype)
        motor.weight[:, :preserved] = self.motor.weight[:, k]
        value.weight[:, :preserved] = self.value.weight[:, k]
        # Novos neurônios começam com influência pequena para preservar comportamento.
        motor.weight[:, preserved:] *= 0.1
        value.weight[:, preserved:] *= 0.1
        motor.bias.copy_(self.motor.bias)
        value.bias.copy_(self.value.bias)
        self.input_weight, self.recurrent, self.bias = nn.Parameter(input_weight), nn.Parameter(recurrent), nn.Parameter(bias)
        self.motor, self.value = motor, value
        self.adjacency, self.utility = adjacency, utility
        removed = [node for i, node in enumerate(self.node_ids) if i not in keep]
        added = list(range(self.next_node_id, self.next_node_id + additions))
        self.node_ids = [self.node_ids[i] for i in keep] + added
        self.next_node_id += additions
        event = {"reason": reason, "before": old_count, "after": new_count,
                 "added": added, "removed": removed, "optimizer_reset_required": True}
        self.events.append(event)
        return event

    @torch.no_grad()
    def rewire(self, count=2):
        if count < 1: return {"added": 0, "pruned": 0}
        active = self.adjacency.nonzero()
        absent = (~self.adjacency & ~torch.eye(self.nodes, device=self.adjacency.device, dtype=torch.bool)).nonzero()
        removed, added = 0, 0
        if len(active):
            strengths = self.recurrent[active[:, 0], active[:, 1]].abs()
            selected = active[strengths.argsort()[:count]]
            self.adjacency[selected[:, 0], selected[:, 1]] = False
            removed = len(selected)
        if len(absent):
            selected = absent[torch.randperm(len(absent), device=absent.device)[:count]]
            self.adjacency[selected[:, 0], selected[:, 1]] = True
            added = len(selected)
        return {"added": added, "pruned": removed}

    def adapt(self, stress, energy):
        if not (0 <= stress <= 2 and 0 <= energy <= 1): raise ValueError("Estado metabólico inválido")
        self.stress_average = 0.6 * self.stress_average + 0.4 * stress
        if energy < self.config.prune_energy and self.nodes > self.config.min_nodes:
            count = min(self.config.growth, self.nodes - self.config.min_nodes)
            discarded = set(self.utility.argsort()[:count].tolist())
            return self.resize([i for i in range(self.nodes) if i not in discarded], reason="energy_pruning")
        if self.stress_average > self.config.stress_threshold and self.nodes < self.config.max_nodes:
            count = min(self.config.growth, self.config.max_nodes - self.nodes)
            return self.resize(range(self.nodes), count, "sustained_stress_growth")
        return None

    def topology(self):
        edges = self.adjacency.nonzero().tolist()
        return {"nodes": self.nodes, "sensor_nodes": self.config.sensor_nodes, "node_ids": self.node_ids,
                "edges": [[self.node_ids[a], self.node_ids[b], float(self.recurrent[a, b].detach())] for a, b in edges],
                "edge_count": len(edges), "utility": self.utility.tolist(), "events": self.events,
                "config": asdict(self.config)}

    def export(self):
        return {"config": dict(asdict(self.config), initial_nodes=self.nodes), "state_dict": self.state_dict(),
                "node_ids": self.node_ids, "next_node_id": self.next_node_id,
                "stress_average": self.stress_average, "events": self.events}

    @classmethod
    def restore(cls, payload, device="cpu"):
        brain = cls(MorphologyConfig(**payload["config"])).to(device)
        brain.load_state_dict(payload["state_dict"])
        brain.node_ids, brain.next_node_id = payload["node_ids"], payload["next_node_id"]
        brain.stress_average, brain.events = payload["stress_average"], payload["events"]
        return brain
