import torch
from .base_organism import BaseOrganism
from ouroboros.environment.signal_bus import SignalBus


class SocialAgent(BaseOrganism):
    def __init__(self, config, primitives, device):
        super().__init__(config, primitives, device)
        self.bus = SignalBus(config["world"]["communication_radius"])
        self.signals = torch.zeros(config["population"], config["signal_size"], device=device)

    def perceive(self, world):
        incoming = self.bus.receive(world.positions, self.signals)
        return world.observe(self.physiology.energy, self.physiology.stress, incoming)

    def broadcast(self):
        self.signals = self.last["signal"].detach()
