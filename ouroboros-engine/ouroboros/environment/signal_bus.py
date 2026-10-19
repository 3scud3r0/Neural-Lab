import torch


class SignalBus:
    def __init__(self, radius):
        self.radius = radius

    def receive(self, positions, signals):
        distance = torch.cdist(positions, positions)
        weights = (1 - distance / self.radius).clamp_min(0)
        weights.fill_diagonal_(0)
        return weights @ signals / weights.sum(-1, keepdim=True).clamp_min(1)
