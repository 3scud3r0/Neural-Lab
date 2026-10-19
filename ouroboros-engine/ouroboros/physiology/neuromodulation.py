import torch


class Neuromodulation:
    def __init__(self, population, device):
        self.baseline = torch.zeros(population, device=device)
        self.dopamine = torch.zeros_like(self.baseline)
        self.attention = torch.ones_like(self.baseline)

    @torch.no_grad()
    def update(self, reward, stress):
        self.dopamine = (reward - self.baseline).tanh()
        self.baseline.lerp_(reward, 0.05)
        self.attention = 1 + stress.clamp(0, 1)

    def plasticity(self):
        return ((1 + self.dopamine) * self.attention).clamp(0.1, 3)
