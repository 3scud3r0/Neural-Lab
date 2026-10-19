"""Física vetorizada e coleta que exige contato e ação conjunta de dois papéis."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..communication.game import GameConfig, feature_vectors


@dataclass
class ArenaConfig:
    batch: int = 32
    size: float = 6.0
    horizon: int = 100
    dt: float = 0.15
    acceleration: float = 4.0
    drag: float = 1.2
    speed_limit: float = 1.5
    radius: float = 0.65
    cooperation_steps: int = 3
    decision_deadline: int = 3
    basal_cost: float = 0.002
    thrust_cost: float = 0.001
    signal_cost: float = 0.0001
    obstacles: int = 2
    obstacle_radius: float = 0.22
    predator_speed: float = 0.08
    radio_range: float = 8.0
    signal_loss: float = 0.0
    night_visibility: float = 1.2

    def __post_init__(self):
        for name in ("batch", "horizon", "cooperation_steps", "decision_deadline"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"Configuração inválida: {name}")
        if type(self.obstacles) is not int or self.obstacles < 0: raise ValueError("obstacles inválido")
        for name, value in vars(self).items():
            if isinstance(value, float) and (not math.isfinite(value) or value < 0):
                raise ValueError(f"Configuração inválida: {name}")
        if self.dt <= 0 or self.size < 4 or self.signal_loss > 1 or self.drag <= 0:
            raise ValueError("Geometria, timestep ou probabilidade inválidos")


class CooperativeArena:
    """A cor/forma do alvo é privada do scout; o mapa dos objetos é público.

    Os dois agentes precisam segurar o MESMO objeto correto por vários passos.
    Um agente sozinho nunca recebe a recompensa de colheita.
    """

    def __init__(self, config=ArenaConfig(), game=GameConfig(), device="cpu"):
        self.config, self.game, self.device = config, game, torch.device(device)
        self.generator = torch.Generator(device=self.device)
        self.reset(42)

    def reset(self, seed=None):
        if seed is not None: self.generator.manual_seed(seed)
        c, b = self.config, self.config.batch
        random = lambda *shape: torch.rand(*shape, device=self.device, generator=self.generator)
        self.positions = c.size / 2 + (random(b, 2, 2) - 0.5) * 0.8
        self.velocity = torch.zeros_like(self.positions)
        angles = torch.arange(self.game.candidates, device=self.device) * (2 * math.pi / self.game.candidates)
        angles = angles[None, :] + random(b, 1) * 2 * math.pi
        self.objects = c.size / 2 + torch.stack((angles.cos(), angles.sin()), -1) * (c.size * 0.32)
        self.concepts = random(b, self.game.concepts).argsort(-1)[:, :self.game.candidates]
        self.target = torch.randint(self.game.candidates, (b,), device=self.device, generator=self.generator)
        self.energy = torch.ones(b, 2, device=self.device)
        self.held = torch.zeros(b, self.game.candidates, dtype=torch.long, device=self.device)
        self.messages = torch.zeros(b, 2, self.game.message_length, dtype=torch.long, device=self.device)
        # Obstáculos não cobrem a posição inicial ou as fontes de recurso.
        self.obstacles = c.size / 2 + torch.stack((torch.cos(angles[:, :1] + 0.8),
                                                  torch.sin(angles[:, :1] + 0.8)), -1) * 0.8
        if c.obstacles == 0:
            self.obstacles = self.obstacles[:, :0]
        elif c.obstacles > 1:
            self.obstacles = self.obstacles.repeat(1, c.obstacles, 1)
            offsets = torch.linspace(-0.3, 0.3, c.obstacles, device=self.device)
            self.obstacles[:, :, 0] += offsets
        self.predator = torch.zeros(b, 2, device=self.device)
        self.done = torch.zeros(b, dtype=torch.bool, device=self.device)
        self.success = torch.zeros_like(self.done)
        self.claims = torch.full((b, 2), -1, dtype=torch.long, device=self.device)
        self.steps = 0
        self.daylight = 1.0
        return self.observation()

    def target_features(self):
        ids = self.concepts.gather(1, self.target[:, None]).squeeze(1)
        return feature_vectors(ids, self.game)

    def candidate_features(self):
        return feature_vectors(self.concepts, self.game)

    def receive(self):
        near = torch.linalg.vector_norm(self.positions[:, 0] - self.positions[:, 1], dim=-1) <= self.config.radio_range
        delivered = self.messages.flip(1).clone()
        if self.config.signal_loss:
            near &= torch.rand(len(near), device=self.device, generator=self.generator) >= self.config.signal_loss
        # O wrapper informa separadamente se há recepção: símbolo 0 é um símbolo válido.
        return delivered, near

    def observation(self):
        b, c = self.config.batch, self.config
        features = self.candidate_features()
        relative = self.objects[:, None, :, :] - self.positions[:, :, None, :]
        distances = torch.linalg.vector_norm(relative, dim=-1)
        visibility = distances < (c.size if self.daylight > 0.35 else c.night_visibility)
        # Ocultação por discos entre câmera e objeto.
        if c.obstacles:
            ray = relative.unsqueeze(-2)
            offset = self.obstacles[:, None, None, :, :] - self.positions[:, :, None, None, :]
            projection = (offset * ray).sum(-1) / ray.square().sum(-1).clamp_min(1e-8)
            closest = offset - projection.unsqueeze(-1) * ray
            occluded = ((projection > 0) & (projection < 1) &
                        (closest.square().sum(-1) < c.obstacle_radius ** 2)).any(-1)
            visibility &= ~occluded
        vision = torch.cat((relative / c.size, features[:, None].expand(-1, 2, -1, -1), visibility[..., None]), -1)
        # Um mapa inicial fornece posições e atributos estáticos. A flag visibility
        # representa visão local; o mapa não revela qual objeto é o referente privado.
        role = torch.tensor([0., 1.], device=self.device)[None, :, None].expand(b, -1, -1)
        partner = self.positions.flip(1) - self.positions
        body = torch.cat((2 * self.positions / c.size - 1, self.velocity / c.speed_limit,
                          self.energy[..., None], role, partner / c.size,
                          torch.full((b, 2, 1), self.daylight, device=self.device)), -1)
        clue = torch.zeros(b, 2, self.game.features, device=self.device)
        clue[:, 0] = self.target_features()
        vector = torch.cat((body, vision.flatten(-2), clue), -1)
        messages, received = self.receive()
        return {"features": vector, "messages": messages, "received": received[:, None].expand(-1, 2)}

    @torch.no_grad()
    def step(self, thrust, grip, messages=None, choices=None):
        c = self.config
        if thrust.shape != (c.batch, 2, 2) or grip.shape != (c.batch, 2):
            raise ValueError("Formato de ação incorreto")
        if not torch.isfinite(thrust).all(): raise ValueError("Ação não finita")
        thrust = thrust.clamp(-1, 1)
        if choices is not None:
            if choices.shape != (c.batch, 2) or choices.dtype != torch.long:
                raise ValueError("Formato das escolhas incorreto")
            if bool(((choices < -1) | (choices >= self.game.candidates)).any()):
                raise ValueError("Escolha de objeto inválida")
            if self.steps < c.decision_deadline:
                commit = (self.claims < 0) & (choices >= 0)
                self.claims[commit] = choices[commit]
        alive = ~self.done
        target_position = self.objects.gather(1, self.target[:, None, None].expand(-1, 1, 2)).squeeze(1)
        before = torch.linalg.vector_norm(self.positions - target_position[:, None], dim=-1).mean(-1)
        previous = self.positions.clone()
        # Integração semi-implícita com amortecimento estável e limite de velocidade.
        drag = c.drag * (1 + 0.25 * math.sin(self.steps * c.dt / 10))
        self.velocity = self.velocity * math.exp(-drag * c.dt) + c.acceleration * thrust * c.dt
        norm = torch.linalg.vector_norm(self.velocity, dim=-1, keepdim=True)
        self.velocity *= torch.minimum(torch.ones_like(norm), c.speed_limit / norm.clamp_min(1e-8))
        self.positions = (self.positions + self.velocity * c.dt * alive[:, None, None]).clamp(0, c.size)
        collision = torch.zeros(c.batch, 2, dtype=torch.bool, device=self.device)
        if c.obstacles:
            collision = (torch.cdist(self.positions, self.obstacles) < c.obstacle_radius).any(-1)
            self.positions[collision] = previous[collision]
            self.velocity[collision] = -0.1 * self.velocity[collision]
        near = torch.linalg.vector_norm(self.positions[:, :, None] - self.objects[:, None], dim=-1) < c.radius
        together = (near & grip.bool()[:, :, None]).all(1)
        self.held = torch.where(together & alive[:, None], self.held + 1, torch.zeros_like(self.held))
        ready = self.held >= c.cooperation_steps
        keys_correct = (self.claims == self.target[:, None]).all(-1)
        correct = ready.gather(1, self.target[:, None]).squeeze(1) & alive & keys_correct
        wrong = ready.any(-1) & ~correct & alive
        # Predador lento: alcance físico retira energia; não causa teletransporte.
        center = self.positions.mean(1)
        direction = center - self.predator
        self.predator += c.predator_speed * c.dt * direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1e-8)
        danger = torch.linalg.vector_norm(self.positions - self.predator[:, None], dim=-1) < 0.45
        cost = c.basal_cost + c.thrust_cost * thrust.square().sum(-1) + danger * 0.03 + collision * 0.003
        if messages is not None:
            if messages.shape != self.messages.shape or messages.dtype != torch.long:
                raise ValueError("Formato de mensagens incorreto")
            if bool(((messages < 0) | (messages >= self.game.vocabulary)).any()):
                raise ValueError("Símbolo fora do vocabulário")
            self.messages = messages.clone()
            cost += c.signal_cost
        self.energy = (self.energy - cost * alive[:, None] + correct[:, None] * 0.2 - wrong[:, None] * 0.03).clamp(0, 1)
        after = torch.linalg.vector_norm(self.positions - target_position[:, None], dim=-1).mean(-1)
        reward = 0.2 * (before - after) + 3.0 * correct - 0.15 * wrong - 0.01 * collision.float().mean(-1)
        reward *= alive
        self.steps += 1
        self.daylight = 0.5 + 0.5 * math.cos(self.steps * c.dt / 4)
        self.success |= correct
        self.done |= correct | (self.energy.min(-1).values <= 0) | (self.steps >= c.horizon)
        return self.observation(), reward, self.done.clone(), {"harvest": correct, "wrong": wrong, "collision": collision}

    def snapshot(self, index=0):
        return {"positions": self.positions[index].tolist(), "velocity": self.velocity[index].tolist(),
                "objects": self.objects[index].tolist(), "concepts": self.concepts[index].tolist(),
                "target": int(self.target[index]), "energy": self.energy[index].tolist(),
                "obstacles": self.obstacles[index].tolist(), "predator": self.predator[index].tolist(),
                "daylight": self.daylight, "size": self.config.size, "step": self.steps,
                "success": bool(self.success[index]), "messages": self.messages[index].tolist()}
