import torch
from .free_energy import activation_entropy


class Homeostasis:
    def __init__(self, population, device, config):
        self.config = config
        self.energy = torch.full((population,), config["initial_energy"], device=device)
        self.stress = torch.zeros_like(self.energy)
        self.entropy = torch.zeros_like(self.energy)
        self.deaths = 0

    @torch.no_grad()
    def update(self, actions, signals, activations, harvest, shock):
        c = self.config
        cost = c["basal_cost"] + c["movement_cost"] * actions.square().mean(-1)
        cost = cost + c["signal_cost"] * signals.square().mean(-1)
        self.energy.add_(harvest * c["harvest_gain"] - cost - 0.01 * shock).clamp_(0, 1)
        overload = (activations.abs().mean(-1) - 0.6).clamp_min(0)
        self.stress = (0.8 * self.stress + 0.2 * (1 - self.energy + overload + shock)).clamp(0, 2)
        self.entropy = activation_entropy(activations)
        dead = self.energy <= 0
        self.deaths += int(dead.sum().item())
        return dead

    def metabolic_loss(self, activations):
        # Gradiente sobre a atividade, com intensidade modulada pelo estresse.
        return (self.stress.detach() * activations.square().mean(-1)).mean()

    @torch.no_grad()
    def respawn(self, mask):
        self.energy[mask] = self.config["initial_energy"]
        self.stress[mask] = 0
