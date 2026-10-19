"""Jogo referencial de Lewis: somente o emissor conhece o referente correto."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GameConfig:
    colors: int = 4
    shapes: int = 4
    candidates: int = 4
    vocabulary: int = 8
    message_length: int = 3
    hidden: int = 64
    protocol: str = "free"

    def __post_init__(self):
        for key, value in vars(self).items():
            if key == "protocol": continue
            if type(value) is not int or value < 1: raise ValueError(f"Configuração inválida: {key}")
        if self.protocol not in {"free", "factorized"}: raise ValueError("Protocolo inválido")
        if self.protocol == "factorized" and self.message_length < 2: raise ValueError("Protocolo fatorado exige dois símbolos")
        if not 2 <= self.candidates <= self.colors * self.shapes:
            raise ValueError("Quantidade de candidatos incompatível com os conceitos")
        if self.colors < 2 or self.shapes < 2 or self.vocabulary < 2:
            raise ValueError("O jogo exige pelo menos dois valores por fator e dois símbolos")

    @property
    def concepts(self): return self.colors * self.shapes

    @property
    def features(self): return self.colors + self.shapes


@dataclass
class GameBatch:
    sender_observation: torch.Tensor
    receiver_candidates: torch.Tensor
    target_index: torch.Tensor
    target_concept: torch.Tensor


def feature_vectors(ids, config):
    return torch.cat((F.one_hot(ids // config.shapes, config.colors),
                      F.one_hot(ids % config.shapes, config.shapes)), -1).float()


def concept_partition(config, heldout=False):
    # Diagonal é reservada para avaliar combinações de atributos não treinadas.
    values = [i for i in range(config.concepts)
              if ((i // config.shapes) == (i % config.shapes)) == heldout]
    if not values: raise ValueError("Partição vazia")
    return values


def make_batch(config, size, device="cpu", generator=None, partition="train"):
    if size < 1 or partition not in {"train", "heldout", "all"}: raise ValueError("Batch inválido")
    concepts = (list(range(config.concepts)) if partition == "all" else concept_partition(config, partition == "heldout"))
    pool = torch.tensor(concepts, device=device)
    target = pool[torch.randint(len(pool), (size,), device=device, generator=generator)]
    # Distratores são distintos. A posição correta é embaralhada por exemplo.
    random_scores = torch.rand(size, config.concepts, device=device, generator=generator)
    random_scores.scatter_(1, target[:, None], -1)
    distractors = random_scores.topk(config.candidates - 1, dim=1).indices
    choices = torch.cat((target[:, None], distractors), dim=1)
    permutation = torch.rand(size, config.candidates, device=device, generator=generator).argsort(-1)
    choices = choices.gather(1, permutation)
    labels = (choices == target[:, None]).long().argmax(-1)
    return GameBatch(feature_vectors(target, config), feature_vectors(choices, config), labels, target)


class Sender(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.protocol == "factorized":
            self.factor_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(size, config.hidden), nn.Tanh(), nn.Linear(config.hidden, config.vocabulary))
                for size in (config.colors, config.shapes)])
            return
        self.encoder = nn.Sequential(nn.Linear(config.features, config.hidden), nn.Tanh())
        self.embedding = nn.Parameter(torch.randn(config.vocabulary, config.hidden) * 0.1)
        self.cell = nn.GRUCell(config.hidden, config.hidden)
        self.head = nn.Linear(config.hidden, config.vocabulary)

    def forward(self, observation, temperature=1.0, discrete=False):
        if self.config.protocol == "factorized":
            parts = observation.split([self.config.colors, self.config.shapes], -1)
            messages, logits_list = [], []
            for slot in range(self.config.message_length):
                logits = self.factor_heads[slot % 2](parts[slot % 2])
                message = (F.one_hot(logits.argmax(-1), self.config.vocabulary).float() if discrete else
                           F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1))
                messages.append(message)
                logits_list.append(logits)
            return torch.stack(messages, 1), torch.stack(logits_list, 1)
        hidden = self.encoder(observation)
        previous = torch.zeros_like(hidden)
        messages, logits_list = [], []
        for _ in range(self.config.message_length):
            hidden = self.cell(previous, hidden)
            logits = self.head(hidden)
            if discrete:
                message = F.one_hot(logits.argmax(-1), self.config.vocabulary).float()
            else:
                message = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
            messages.append(message)
            logits_list.append(logits)
            previous = message @ self.embedding
        return torch.stack(messages, 1), torch.stack(logits_list, 1)


class Receiver(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.protocol == "factorized":
            self.factor_readouts = nn.ModuleList([nn.Linear(config.vocabulary, size, bias=False)
                                                 for size in (config.colors, config.shapes)])
            return
        self.embedding = nn.Parameter(torch.randn(config.vocabulary, config.hidden) * 0.1)
        self.decoder = nn.GRU(config.hidden, config.hidden, batch_first=True)
        self.candidates = nn.Sequential(nn.Linear(config.features, config.hidden), nn.Tanh())

    def forward(self, messages, candidates):
        # Não recebe target_index, clue ou a observação privada do emissor.
        if self.config.protocol == "factorized":
            parts = candidates.split([self.config.colors, self.config.shapes], -1)
            scores = torch.zeros(candidates.shape[:2], device=candidates.device)
            for slot in range(self.config.message_length):
                affinity = self.factor_readouts[slot % 2](messages[:, slot])
                scores = scores + (parts[slot % 2] * affinity[:, None]).sum(-1)
            return scores
        _, state = self.decoder(messages @ self.embedding)
        options = self.candidates(candidates)
        return (options * state[0, :, None, :]).sum(-1) / math.sqrt(options.shape[-1])


class CommunicationGame(nn.Module):
    def __init__(self, config=GameConfig()):
        super().__init__()
        self.config = config
        self.sender, self.receiver = Sender(config), Receiver(config)

    def forward(self, batch, temperature=1.0, discrete=False):
        message, logits = self.sender(batch.sender_observation, temperature, discrete)
        return self.receiver(message, batch.receiver_candidates), message, logits
