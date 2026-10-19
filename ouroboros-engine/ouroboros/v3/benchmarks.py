"""Synthetic sequence benchmarks with separate deterministic data partitions.

These tasks measure narrow capabilities. They are not an intelligence benchmark.
"""
from dataclasses import dataclass

import torch
from torch import nn

TASKS = ("delay", "integrate", "switch", "denoise")
SPLITS = {"train": 0, "validation": 100000, "test": 200000, "transfer": 300000}


@dataclass
class Batch:
    inputs: torch.Tensor
    targets: torch.Tensor
    task: str
    split: str


def make_batch(task, split, seed, batch=24, length=12):
    if task not in TASKS or split not in SPLITS or batch < 1 or length < 3:
        raise ValueError("Invalid benchmark request")
    generator = torch.Generator().manual_seed(seed + SPLITS[split] + TASKS.index(task) * 1000000)
    latent = torch.randn(batch, length, generator=generator).tanh()
    inputs = torch.zeros(batch, length, 3)
    inputs[:, :, 0] = latent
    if task == "delay":
        delay = 3 if split == "transfer" else 2
        target = torch.zeros_like(latent)
        target[:, delay:] = latent[:, :-delay]
    elif task == "integrate":
        target = latent.cumsum(1) / torch.arange(1, length + 1).sqrt()
        target = target.tanh()
    elif task == "switch":
        switches = torch.rand(batch, length, generator=generator) < 0.25
        inputs[:, :, 1] = switches.float()
        target = torch.zeros_like(latent)
        remembered = torch.zeros(batch)
        for time in range(length):
            remembered = torch.where(switches[:, time], latent[:, time], remembered)
            target[:, time] = remembered
    else:
        target = torch.zeros_like(latent)
        target[:, 0] = latent[:, 0]
        for time in range(1, length):
            target[:, time] = 0.8 * target[:, time - 1] + 0.2 * latent[:, time]
        noise = 0.5 if split == "transfer" else 0.15
        inputs[:, :, 0] += noise * torch.randn(batch, length, generator=generator)
    inputs[:, :, 2] = TASKS.index(task) / (len(TASKS) - 1)
    return Batch(inputs, target.unsqueeze(-1), task, split)


class Baseline(nn.Module):
    def __init__(self, kind, hidden=12):
        super().__init__()
        self.kind = kind
        if kind == "gru": self.core = nn.GRU(3, hidden, batch_first=True)
        elif kind == "lstm": self.core = nn.LSTM(3, hidden, batch_first=True)
        elif kind == "rnn": self.core = nn.RNN(3, hidden, batch_first=True)
        elif kind == "mlp": self.core = nn.Sequential(nn.Linear(3, hidden), nn.Tanh())
        else: raise ValueError("Unknown baseline")
        self.readout = nn.Linear(hidden, 1)

    def forward(self, inputs):
        hidden = self.core(inputs)
        return self.readout(hidden if self.kind == "mlp" else hidden[0])


def train_steps(model, steps, seed, batch=24, length=12, optimizer=None, start=0):
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=0.015)
    model.train()
    for step in range(start, start + steps):
        task = TASKS[step % len(TASKS)]
        data = make_batch(task, "train", seed + step, batch, length)
        optimizer.zero_grad(set_to_none=True)
        loss = (model(data.inputs) - data.targets).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    return optimizer


@torch.no_grad()
def evaluate(model, split, seed, batch=24, length=12):
    model.eval()
    losses = {}
    for task in TASKS:
        data = make_batch(task, split, seed, batch, length)
        prediction = model(data.inputs)
        loss = (prediction - data.targets).square().mean().item()
        if not torch.isfinite(torch.tensor(loss)):
            raise FloatingPointError("Nonfinite evaluation")
        losses[task] = loss
    return {"mean": sum(losses.values()) / len(losses), "tasks": losses}
