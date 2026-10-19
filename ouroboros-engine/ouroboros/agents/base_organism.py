from abc import ABC, abstractmethod
import torch
from ouroboros.core.dynamic_brain import DynamicBrain
from ouroboros.physiology.homeostasis import Homeostasis
from ouroboros.physiology.neuromodulation import Neuromodulation


class BaseOrganism(ABC):
    """Uma instância contém a população vetorizada para evitar loops na GPU."""

    def __init__(self, config, primitives, device):
        count = config["population"]
        self.brain = DynamicBrain(7 + config["signal_size"], config["hidden_size"],
                                  config["signal_size"], count, primitives).to(device)
        self.physiology = Homeostasis(count, device, config["physiology"])
        self.modulation = Neuromodulation(count, device)
        self.last = None

    @abstractmethod
    def perceive(self, world):
        raise NotImplementedError

    def think(self, observation, dt):
        self.last = self.brain(observation, dt, self.modulation.plasticity())
        return self.last

    def act(self, world, dt):
        return world.step(self.last["action"].detach(), dt)

    def metabolize(self, harvest, reward, shock):
        dead = self.physiology.update(self.last["action"].detach(), self.last["signal"].detach(),
                                     self.last["activity"].detach(), harvest, shock)
        self.modulation.update(reward, self.physiology.stress)
        return dead
