"""Differentiable stateful neurons: rate, leaky, spikes, dendrites and hybrids."""
from dataclasses import dataclass

import torch
from torch import nn

from .neuron_dsl import Expr, parse, run


@dataclass
class CellState:
    h: torch.Tensor
    memory: torch.Tensor
    adaptation: torch.Tensor
    phase: torch.Tensor

    def detach(self):
        return CellState(*(value.detach() for value in
                           (self.h, self.memory, self.adaptation, self.phase)))


def seed_expressions():
    x, h, m, a, p = (Expr(name) for name in ("x", "h", "m", "a", "p"))
    return {
        "rate": Expr("tanh", (x,)),
        "leaky": Expr("mix", (h, Expr("tanh", (x,)), Expr("const", value=1.0))),
        "spiking": Expr("sigmoid", (Expr("sub", (x, a)),)),
        "dendritic": Expr("mul", (Expr("tanh", (x,)), Expr("sigmoid", (h,)))),
        "oscillatory": Expr("tanh", (Expr("add", (x, Expr("sin", (p,)))),)),
        "associative": Expr("tanh", (Expr("add", (x, m)),)),
        "adaptive": Expr("tanh", (Expr("sub", (x, a)),)),
        "modulated": Expr("mix", (m, Expr("tanh", (x,)), h)),
    }


class InventedCell(nn.Module):
    def __init__(self, inputs, hidden, expression, spiking=False):
        super().__init__()
        if not 1 <= inputs <= 256 or not 1 <= hidden <= 128:
            raise ValueError("Cell dimensions exceed budget")
        self.expression = parse(expression.to_dict())
        self.inputs, self.hidden, self.spiking = inputs, hidden, spiking
        self.input = nn.Linear(inputs, hidden)
        self.recurrent = nn.Linear(hidden, hidden, bias=False)
        self.decay = nn.Parameter(torch.zeros(hidden))
        self.threshold = nn.Parameter(torch.ones(hidden) * 0.4)
        self.frequency = nn.Parameter(torch.linspace(0.01, 0.2, hidden))

    def initial(self, batch, device=None):
        zero = torch.zeros(batch, self.hidden, device=device or self.decay.device)
        return CellState(*(zero.clone() for _ in range(4)))

    def forward(self, value, state=None):
        if value.ndim != 2 or value.shape[-1] != self.inputs:
            raise ValueError("Cell expects [batch, inputs]")
        state = self.initial(value.shape[0], value.device) if state is None else state
        if state.h.shape != (value.shape[0], self.hidden):
            raise ValueError("Incompatible cell state")
        drive = self.input(value) + self.recurrent(state.h)
        phase = state.phase + self.frequency
        proposed = run(self.expression, {"x": drive, "h": state.h, "m": state.memory,
                                         "a": state.adaptation, "p": phase})
        if self.spiking:
            soft = torch.sigmoid(8 * (proposed - self.threshold - state.adaptation))
            hard = (soft > 0.5).to(soft.dtype)
            proposed = hard + soft - soft.detach()
        decay = self.decay.sigmoid()
        h = (decay * state.h + (1 - decay) * proposed).clamp(-8, 8)
        memory = (0.95 * state.memory + 0.05 * h * drive.tanh()).clamp(-4, 4)
        adaptation = 0.9 * state.adaptation + 0.1 * h.abs()
        return h, CellState(h, memory, adaptation, phase.remainder(2 * torch.pi))


class SequenceModel(nn.Module):
    def __init__(self, inputs, hidden, outputs, expression):
        super().__init__()
        self.cell = InventedCell(inputs, hidden, expression)
        self.readout = nn.Linear(hidden, outputs)

    def forward(self, sequence):
        state, outputs = None, []
        for value in sequence.unbind(1):
            hidden, state = self.cell(value, state)
            outputs.append(self.readout(hidden))
        return torch.stack(outputs, dim=1)
