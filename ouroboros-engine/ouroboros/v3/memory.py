"""Bounded episodic retrieval, priority replay and continual-learning penalties."""
from dataclasses import dataclass
import math
import random

import torch


@dataclass
class Episode:
    key: torch.Tensor
    value: object
    priority: float
    timestamp: int


class EpisodicMemory:
    def __init__(self, capacity=256):
        if capacity < 1:
            raise ValueError("Positive memory capacity required")
        self.capacity, self.clock, self.items = capacity, 0, []

    def add(self, key, value, priority=1.0):
        if key.ndim != 1 or not torch.isfinite(key).all() or not math.isfinite(priority) or priority <= 0:
            raise ValueError("Invalid episode")
        if self.items and key.shape != self.items[0].key.shape:
            raise ValueError("Memory key dimension changed")
        self.clock += 1
        self.items.append(Episode(key.detach().cpu().clone(), value, float(priority), self.clock))
        if len(self.items) > self.capacity:
            victim = min(range(len(self.items)), key=lambda i: (self.items[i].priority, self.items[i].timestamp))
            self.items.pop(victim)

    def retrieve(self, query, count=3):
        if not self.items:
            return []
        if count < 1 or query.shape != self.items[0].key.shape or not torch.isfinite(query).all():
            raise ValueError("Invalid retrieval query")
        keys = torch.stack([item.key for item in self.items])
        similarities = torch.nn.functional.cosine_similarity(keys, query.detach().cpu().unsqueeze(0), dim=1)
        indices = similarities.topk(min(count, len(self.items))).indices.tolist()
        return [(self.items[i].value, similarities[i].item()) for i in indices]

    def replay(self, count, rng=None):
        if not self.items:
            return []
        rng = rng or random
        return rng.choices(self.items, weights=[i.priority for i in self.items], k=count)


class Consolidation:
    def __init__(self):
        self.reference, self.importance = {}, {}

    def capture(self, model, losses):
        parameters = dict(model.named_parameters())
        importance = {name: torch.zeros_like(value) for name, value in parameters.items()}
        count = 0
        for loss in losses:
            gradients = torch.autograd.grad(loss, tuple(parameters.values()), retain_graph=True, allow_unused=True)
            for (name, _), gradient in zip(parameters.items(), gradients):
                if gradient is not None:
                    importance[name].add_(gradient.detach().square())
            count += 1
        if not count:
            raise ValueError("Consolidation needs observed losses")
        self.reference = {name: p.detach().clone() for name, p in parameters.items()}
        self.importance = {name: value / count for name, value in importance.items()}

    def penalty(self, model):
        penalties = []
        for name, parameter in model.named_parameters():
            if name in self.reference:
                if parameter.shape != self.reference[name].shape:
                    raise ValueError("Reconsolidate after structural changes")
                penalties.append((self.importance[name] * (parameter - self.reference[name]).square()).sum())
        return sum(penalties) if penalties else next(model.parameters()).sum() * 0
